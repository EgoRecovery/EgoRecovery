"""Training-time output staging on tmpfs, with background rsync.

Motivation: network filesystems mounted with blocking semantics can turn a
transient server hiccup into an indefinite `read`/`write` stall on rank 0.
While rank 0 is stuck, the other ranks reach the next DDP collective (every
training step calls `_broadcast_sigterm_tensor`) and the NCCL watchdog can
tear the job down after a timeout.

The repo already stages large artefacts off shared storage --
`AsyncCopyModelCheckpoint`, `IntentDiagnosticCallback`, `EvalVideo` -- but
everything driven by `${paths.output_dir}` still writes to the final output
tree. This module extends the same pattern to the whole output tree: training
writes to `/dev/shm`, a daemon thread rsyncs to the final mirror dir
periodically. Rank 0's hot path never touches shared storage again.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

_STAGING_ROOT = "/dev/shm/run_staging"
_NAS_MIRROR_ROOT_MARKER = ".nas_mirror_root"


def heal_logs_symlink(local_logs: str) -> Optional[str]:
    """Replace a `local_logs -> NAS` symlink with a real tmpfs directory.

    A common setup is to `ln -s /path/to/shared/logs /dev/shm/.../logs` so
    Hydra's relative `run.dir: ./logs/...` lands on shared storage. That makes
    `trainHydra.log`, `train_ddp_process_{N}.log`, and `.hydra/` write there
    from rank 0, and any filesystem hiccup can D-state the whole training
    process.

    This function breaks that by:
      1. Reading the symlink target (the NAS mirror root), if present.
      2. Removing the symlink and creating a real tmpfs dir in its place.
      3. Writing the NAS target to `<local_logs>/.nas_mirror_root` so
         subsequent runs (which see a real dir, no symlink) still know where
         to rsync back to.
      4. On subsequent runs, reading the marker file to recover the target.

    Returns the NAS mirror root, or None if no symlink and no marker is
    present (e.g. a fresh /dev/shm where the user has not yet set up the
    symlink — in that case staging stays local-only and the user must run
    the final sync manually).

    Env var NAS_LOGS_ROOT overrides the returned mirror target -- handy for CI
    or setups where the symlink convention isn't used. The symlink itself is
    still healed when present, because any time the path is a symlink to shared
    storage, Hydra writes will hit that storage from the hot path regardless of
    config.
    """
    env_override = os.environ.get("NAS_LOGS_ROOT", "").strip() or None

    if os.path.islink(local_logs):
        # os.path.realpath resolves relative symlink targets against the
        # symlink's parent directory, which is what we want.
        symlink_target = os.path.realpath(local_logs)
        os.remove(local_logs)
        os.makedirs(local_logs, exist_ok=True)
        persisted = env_override or symlink_target
        marker_path = os.path.join(local_logs, _NAS_MIRROR_ROOT_MARKER)
        with open(marker_path, "w") as f:
            f.write(persisted + "\n")
        log.info(
            "Healed %s: was symlink -> %s, now real tmpfs dir; "
            "NAS target %s persisted in %s",
            local_logs,
            symlink_target,
            persisted,
            marker_path,
        )
        return persisted

    if env_override:
        return env_override

    if os.path.isdir(local_logs):
        marker_path = os.path.join(local_logs, _NAS_MIRROR_ROOT_MARKER)
        if os.path.isfile(marker_path):
            with open(marker_path) as f:
                target = f.read().strip()
            return target or None

    return None


def staging_dir_for(nas_output_dir: str) -> str:
    """Return a deterministic tmpfs staging dir for this run.

    Name derives from the NAS target's last two path segments (so `ls` on
    `/dev/shm/run_staging/` is legible) plus a short sha1 suffix (so distinct
    runs never collide, even if the timestamp is identical).
    """
    parent = os.path.basename(os.path.dirname(nas_output_dir.rstrip("/"))) or "run"
    leaf = os.path.basename(nas_output_dir.rstrip("/")) or "out"
    tag = f"{parent}_{leaf}"[:80]
    tag = "".join(c if c.isalnum() or c in "_-." else "_" for c in tag)
    digest = hashlib.sha1(nas_output_dir.encode("utf-8")).hexdigest()[:12]
    path = os.path.join(_STAGING_ROOT, f"{tag}_{digest}")
    os.makedirs(path, exist_ok=True)
    return path


class NasMirrorDaemon:
    """Daemon thread that periodically rsyncs a staging dir to NAS.

    Contract: the main thread calls `start()` once and `stop(final_sync=True)`
    once. Between those, rsync runs in a background thread every
    `interval_s` seconds, each attempt bounded by `periodic_timeout_s`.
    If a prior rsync is still running when the next tick fires, the tick is
    skipped (same non-blocking policy as `AsyncCopyModelCheckpoint._mirror_async`).
    If an rsync exceeds the periodic timeout, subprocess.run kills it so the
    lock frees and the next tick can start fresh — NAS hangs therefore cannot
    leak rsync subprocesses indefinitely.

    `stop(final_sync=True)` issues one last rsync with a longer timeout
    (`final_timeout_s`, default 30min to cover big last.ckpt mirrors), so
    recent vis/video/csv artefacts land on NAS before the process exits.
    Best-effort: if NAS is genuinely hung, the final sync times out and logs
    a warning + manual recovery command rather than blocking the teardown.
    """

    def __init__(
        self,
        local: str,
        nas: str,
        interval_s: float = 30.0,
        periodic_timeout_s: float = 120.0,
        final_timeout_s: float = 1800.0,
    ):
        self.local = local
        self.nas = nas
        self.interval_s = float(interval_s)
        self.periodic_timeout_s = float(periodic_timeout_s)
        self.final_timeout_s = float(final_timeout_s)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rsync_lock = threading.Lock()
        self._atexit_registered = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        os.makedirs(self.local, exist_ok=True)
        os.makedirs(self.nas, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="nas-mirror"
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self._atexit_flush)
            self._atexit_registered = True
        log.info(
            "NAS mirror: %s -> %s (interval=%.0fs)",
            self.local,
            self.nas,
            self.interval_s,
        )

    def stop(self, final_sync: bool = True) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if final_sync:
            self._final_sync()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._maybe_rsync_once()
            # Wait in small slices so stop() is responsive.
            end = time.monotonic() + self.interval_s
            while time.monotonic() < end and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

    def _maybe_rsync_once(self) -> None:
        # Skip if previous rsync still running — the next tick catches up.
        if not self._rsync_lock.acquire(blocking=False):
            return
        try:
            self._rsync(timeout_s=self.periodic_timeout_s)
        except subprocess.TimeoutExpired:
            # NAS stalled past our bound: kill this rsync so the next tick can
            # start fresh. subprocess.run already reaped the child on timeout.
            log.warning(
                "NAS mirror rsync %s -> %s exceeded %.0fs; killed and will retry next tick.",
                self.local,
                self.nas,
                self.periodic_timeout_s,
            )
        except Exception as e:
            log.warning("NAS mirror rsync %s -> %s failed: %s", self.local, self.nas, e)
        finally:
            self._rsync_lock.release()

    def _rsync(self, timeout_s: Optional[float]) -> None:
        # -a archive, --inplace --partial to keep large files (wandb .wandb,
        # ckpts) appendable instead of full-copy per tick. No --delete: a run's
        # nas_output_dir is unique, there's nothing on NAS to reconcile.
        cmd = [
            "rsync",
            "-a",
            "--inplace",
            "--partial",
            self.local.rstrip("/") + "/",
            self.nas.rstrip("/") + "/",
        ]
        # check=False: rsync returns 24 (some files vanished before xfer) when
        # a tmp file got rotated mid-sync; not an error for our purposes.
        subprocess.run(cmd, check=False, timeout=timeout_s, capture_output=True)

    def _final_sync(self) -> None:
        # If a prior tick is still stuck, don't block teardown on it.
        if not self._rsync_lock.acquire(blocking=False):
            log.warning(
                "NAS mirror final sync skipped: prior rsync still running "
                "(NAS likely hung). Staged artefacts remain in %s.",
                self.local,
            )
            return
        t0 = time.monotonic()
        log.info(
            "NAS mirror final sync starting: %s -> %s (timeout=%.0fs)",
            self.local,
            self.nas,
            self.final_timeout_s,
        )
        try:
            self._rsync(timeout_s=self.final_timeout_s)
            log.info(
                "NAS mirror final sync done in %.1fs: %s -> %s",
                time.monotonic() - t0,
                self.local,
                self.nas,
            )
        except subprocess.TimeoutExpired:
            log.warning(
                "NAS mirror final sync timed out after %.0fs; "
                "some artefacts may not be on NAS. Staged at %s (recover manually with: "
                "rsync -a --inplace --partial %s/ %s/).",
                self.final_timeout_s,
                self.local,
                self.local.rstrip("/"),
                self.nas.rstrip("/"),
            )
        except Exception as e:
            log.warning("NAS mirror final sync failed: %s", e)
        finally:
            self._rsync_lock.release()

    def _atexit_flush(self) -> None:
        # Only runs if the user never called stop() (e.g. abnormal exit path).
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._final_sync()

"""Checkpoint callback that stages saves locally and mirrors to the final dir asynchronously.

Motivation: on slow/shared storage (NAS, NFS), writing a large checkpoint from rank 0
can exceed the NCCL collective timeout because the other ranks are stuck at the
post-save barrier. Writing to a fast local path (e.g. /dev/shm or a local SSD) and
copying to the NAS in a background thread keeps the barrier short.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading

from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)

# Per-file rsync bound. NAS hang -> rsync gets SIGKILL'd and the mirror worker
# returns; the serializing lock releases so the next save can try again.
# Previous implementation used shutil.copy2 which has no timeout: an NFS hiccup
# during copy would D-state the worker thread indefinitely, holding file handles
# that could interact with the rank-0 main thread via kernel-level file-table
# contention. See CLAUDE.md "DDP Rank-0 I/O Rule" for the underlying constraint.
_MIRROR_RSYNC_TIMEOUT_S = 300


class AsyncCopyModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that writes to ``local_stage_dir`` and async-copies to ``dirpath``.

    If ``local_stage_dir`` is ``None`` or empty, falls back to the default
    ``ModelCheckpoint`` behavior (writes directly to ``dirpath``).

    Otherwise ``dirpath`` is treated as the final NAS destination. Writes go to
    ``<local_stage_dir>/<hash-of-dirpath>``, and after each save/remove rank 0
    fires a background thread that copies the new/changed files to NAS.
    ``teardown`` blocks on pending copies so the final checkpoint is flushed.
    """

    def __init__(self, local_stage_dir: str | None = None, **kwargs):
        self._mirror_dir: str | None = None
        self._mirror_lock = threading.Lock()
        self._mirror_worker_lock = threading.Lock()
        self._mirror_threads: list[threading.Thread] = []

        if local_stage_dir:
            nas_dirpath = kwargs.pop("dirpath", None)
            if nas_dirpath is None:
                raise ValueError(
                    "`dirpath` is required when `local_stage_dir` is set; it is used as the final NAS destination."
                )
            self._mirror_dir = nas_dirpath
            uniq = hashlib.sha1(nas_dirpath.encode("utf-8")).hexdigest()[:12]
            local_dirpath = os.path.join(local_stage_dir, uniq)
            os.makedirs(local_dirpath, exist_ok=True)
            kwargs["dirpath"] = local_dirpath

        super().__init__(**kwargs)

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        self._mirror_async(trainer)

    def _remove_checkpoint(self, trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        self._mirror_async(trainer)

    def _mirror_async(self, trainer) -> None:
        if not self._mirror_dir or not trainer.is_global_zero:
            return
        src = self.dirpath
        dst = self._mirror_dir
        t = threading.Thread(
            target=self._mirror_worker,
            args=(src, dst, self._mirror_worker_lock),
            daemon=True,
            name="ckpt-mirror",
        )
        t.start()
        with self._mirror_lock:
            self._mirror_threads.append(t)
            self._mirror_threads[:] = [x for x in self._mirror_threads if x.is_alive()]

    @staticmethod
    def _mirror_worker(src: str, dst: str, worker_lock: threading.Lock) -> None:
        # Serialize concurrent mirrors on the same instance: if one is already
        # running, skip — its scan will pick up whatever we'd have copied.
        if not worker_lock.acquire(blocking=False):
            return
        try:
            os.makedirs(dst, exist_ok=True)
            # Delegate the whole mirror (copy new/changed, delete removed) to a
            # single rsync invocation with a bounded timeout. --delete keeps
            # dst in sync when save_top_k rotates checkpoints locally.
            # --inplace + --partial so big ckpt files stream incrementally;
            # rsync's own --timeout handles per-IO stalls on the network, and
            # subprocess timeout is the outer guard if rsync itself wedges.
            cmd = [
                "rsync",
                "-a",
                "--inplace",
                "--partial",
                "--delete",
                "--timeout=120",
                src.rstrip("/") + "/",
                dst.rstrip("/") + "/",
            ]
            subprocess.run(
                cmd,
                check=False,
                timeout=_MIRROR_RSYNC_TIMEOUT_S,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            log.warning(
                "Async checkpoint mirror %s -> %s exceeded %ds; killed rsync, "
                "next save will retry. NAS likely stalled.",
                src,
                dst,
                _MIRROR_RSYNC_TIMEOUT_S,
            )
        except Exception as e:
            log.warning("Async checkpoint mirror %s -> %s failed: %s", src, dst, e)
        finally:
            worker_lock.release()

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if self._mirror_dir and trainer.is_global_zero:
            # Drain pending mirrors, then do one final synchronous pass so
            # last.ckpt is guaranteed on NAS even if the last async mirror
            # was skipped due to the serializing lock.
            with self._mirror_lock:
                threads = list(self._mirror_threads)
            for t in threads:
                t.join()
            self._mirror_worker(self.dirpath, self._mirror_dir, self._mirror_worker_lock)
            with self._mirror_lock:
                self._mirror_threads.clear()
        super().teardown(trainer, pl_module, stage)

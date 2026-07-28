#!/usr/bin/env python3
"""
EgoVerse ACT Policy Server for ALOHA

Implements the same WebSocket protocol as openpi, so the existing
client_delta.py can connect without modification.

Usage:
    python serve_policy_aloha.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --port 8000

The server introspects the loaded checkpoint to find camera_keys,
proprio_keys, embodiment_id, and chunk_size automatically.

Camera key mapping (client → model zarr key):
    cam_high        → front_img_1
    cam_left_wrist  → left_wrist_img
    cam_right_wrist → right_wrist_img
    state / qpos    → joint_positions  (first proprio_key)
"""

import asyncio
import dataclasses
import http
import logging
import socket
import time
import traceback
from typing import Optional

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# msgpack helpers — prefer openpi_client if installed, else fall back to
# standalone msgpack + msgpack_numpy
# ---------------------------------------------------------------------------
try:
    from openpi_client import msgpack_numpy as _msgpack_numpy  # type: ignore
    import msgpack as _msgpack_lib  # type: ignore

    # Prefer explicit object_hook so numpy arrays are decoded correctly
    # regardless of openpi_client version.
    _opc_decode = getattr(_msgpack_numpy, "decode", None)
    print(f"[DEBUG-INIT] openpi_client branch: _opc_decode={_opc_decode}, "
          f"attrs={[a for a in dir(_msgpack_numpy) if not a.startswith('_')]}", flush=True)

    def _unpackb(data: bytes) -> dict:
        if _opc_decode is not None:
            result = _msgpack_lib.unpackb(data, object_hook=_opc_decode, raw=False)
        else:
            result = _msgpack_numpy.unpackb(data)
        return result

    class _Packer:
        def __init__(self):
            self._p = _msgpack_numpy.Packer()

        def pack(self, obj) -> bytes:
            return self._p.pack(obj)

except ImportError:
    import msgpack  # type: ignore
    import msgpack_numpy as _mnp  # type: ignore

    _mnp.patch()

    def _decode_numpy(obj: dict):
        """Handle both openpi format (b'__ndarray__') and standard msgpack_numpy format."""
        if b'__ndarray__' in obj:
            # openpi_client encoding: {b'__ndarray__': True, b'data': bytes,
            #                          b'dtype': str, b'shape': list}
            return np.frombuffer(obj[b'data'], dtype=np.dtype(obj[b'dtype'])).reshape(
                tuple(obj[b'shape'])
            )
        # Fall back to standard msgpack_numpy decoder
        return _mnp.decode(obj)

    def _unpackb(data: bytes) -> dict:
        return msgpack.unpackb(data, raw=False, object_hook=_decode_numpy)

    def _to_openpi(obj):
        """Recursively convert numpy arrays to openpi msgpack format before packing."""
        if isinstance(obj, np.ndarray):
            return {
                b'__ndarray__': True,
                b'data': obj.tobytes(),
                b'dtype': str(obj.dtype),
                b'shape': list(obj.shape),
            }
        if isinstance(obj, dict):
            return {k: _to_openpi(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_openpi(v) for v in obj]
        return obj

    class _Packer:
        def __init__(self):
            # Use a fresh Packer without patch interference
            self._p = msgpack.Packer(use_bin_type=True)

        def pack(self, obj) -> bytes:
            return self._p.pack(_to_openpi(obj))


# ---------------------------------------------------------------------------
# Camera key mapping: openpi/ALOHA client key → EgoVerse zarr key
# ---------------------------------------------------------------------------
CLIENT_TO_ZARR_CAM: dict[str, str] = {
    "cam_high": "front_img_1",
    "cam_left_wrist": "left_wrist_img",
    "cam_right_wrist": "right_wrist_img",
}


# ---------------------------------------------------------------------------
# EgoVerse policy wrapper
# ---------------------------------------------------------------------------
class EgoVerseAlohaPolicy:
    """Wraps a loaded EgoVerse ModelWrapper for ALOHA inference."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from egomimic.pl_utils.pl_model import ModelWrapper

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, weights_only=False, map_location="cpu"
        )
        self.wrapper = self.wrapper.to(self.device)
        self.wrapper.eval()

        self.model = self.wrapper.model
        self.model.device = self.device

        # Introspect model
        self.camera_keys: list[str] = self.model.camera_keys
        self.proprio_keys: list[str] = self.model.proprio_keys
        self.embodiment_id: int = self.model.embodiment_id
        self.chunk_size: int = self.model.chunk_size
        self.ac_key: str = self.model.ac_key
        self.data_schematic = self.model.data_schematic

        logger.info(f"  camera_keys:    {self.camera_keys}")
        logger.info(f"  proprio_keys:   {self.proprio_keys}")
        logger.info(f"  embodiment_id:  {self.embodiment_id}")
        logger.info(f"  chunk_size:     {self.chunk_size}")
        logger.info(f"  ac_key:         {self.ac_key}")
        logger.info(f"  device:         {self.device}")

        # Build reverse mapping: zarr_key → model camera_key
        # (they should be the same, but just in case)
        self._cam_zarr_to_model: dict[str, str] = {k: k for k in self.camera_keys}

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """
        Args:
            obs: dict from ALOHA client with keys:
                images.cam_high / cam_left_wrist / cam_right_wrist  (CHW uint8)
                state or qpos  (14-dim float)
        Returns:
            dict with key 'actions': np.ndarray (chunk_size, action_dim)
        """
        batch = self._prepare_batch(obs)
        preds = self.model.forward_eval(batch)
        actions = preds[self.ac_key]  # (1, chunk_size, action_dim)
        actions = actions.detach().cpu().numpy().squeeze(0)  # (chunk_size, action_dim)
        return {"actions": actions}

    def _prepare_batch(self, obs: dict) -> dict:
        """Convert client observation dict to model batch format."""
        batch: dict[str, torch.Tensor] = {}

        # ---- images --------------------------------------------------------
        images_dict: dict = obs.get("images", {})
        for client_key, zarr_key in CLIENT_TO_ZARR_CAM.items():
            if zarr_key not in self.camera_keys:
                continue
            img = images_dict.get(client_key)
            if img is None:
                raise ValueError(
                    f"Missing image key '{client_key}' in observation. "
                    f"Available: {list(images_dict.keys())}"
                )
            # img: CHW uint8 numpy → float [0,1] tensor (1, C, H, W)
            img_t = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)  # (1, C, H, W)
            batch[zarr_key] = img_t.to(self.device)

        # ---- proprio -------------------------------------------------------
        state = obs.get("state") if obs.get("state") is not None else obs.get("qpos")
        if state is None:
            raise ValueError("Observation must contain 'state' or 'qpos'.")
        state_np = np.asarray(state, dtype=np.float32)
        # shape: (state_dim,) → (1, state_dim)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        proprio_key = self.proprio_keys[0]
        batch[proprio_key] = state_t  # (1, state_dim)

        print(f"[PROPRIO-RAW] joints={state_np[:6].round(3)}  "
              f"grip_l={state_np[6]:.4f}  "
              f"joints_r={state_np[7:13].round(3)}  "
              f"grip_r={state_np[13]:.4f}", flush=True)

        # ---- normalize proprio (cameras are handled by eval_image_augs) ----
        batch = self.data_schematic.normalize_data(batch, self.embodiment_id)

        normed = batch[proprio_key][0].cpu().numpy()
        print(f"[PROPRIO-NORM] joints={normed[:6].round(3)}  "
              f"grip_l={normed[6]:.4f}  "
              f"joints_r={normed[7:13].round(3)}  "
              f"grip_r={normed[13]:.4f}", flush=True)

        # ---- add time dimension to proprio ---------------------------------
        # process_batch_for_training does: batch[key] = batch[key][:, None, :]
        batch[proprio_key] = batch[proprio_key].unsqueeze(1)  # (1, 1, state_dim)

        # ---- pad_mask ------------------------------------------------------
        batch["pad_mask"] = torch.ones(
            1, self.chunk_size, 1, device=self.device, dtype=torch.float32
        )

        return batch

    @property
    def metadata(self) -> dict:
        return {
            "model": "egoverse_act_aloha",
            "camera_keys": self.camera_keys,
            "proprio_keys": self.proprio_keys,
            "chunk_size": self.chunk_size,
        }


# ---------------------------------------------------------------------------
# WebSocket server (same protocol as openpi WebsocketPolicyServer)
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    def __init__(
        self,
        policy: EgoVerseAlohaPolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata if metadata is not None else policy.metadata

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as _ws_server

        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"Server listening on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets

        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = _Packer()

        # Send metadata first (same as openpi protocol)
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw = await websocket.recv()
                obs = _unpackb(raw)

                infer_start = time.monotonic()
                result = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                result["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                packed = packer.pack(result)
                await websocket.send(packed)
                prev_total_time = time.monotonic() - start_time

                logger.debug(f"Inference: {infer_time * 1000:.1f}ms")

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                tb = traceback.format_exc()
                logger.error(f"Error during inference:\n{tb}")
                try:
                    await websocket.send(tb)
                    import websockets.frames
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                except Exception:
                    pass
                raise

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    # Path to EgoVerse ModelWrapper checkpoint (.ckpt)
    checkpoint: str
    # Port to serve on
    port: int = 8000
    # Host to bind
    host: str = "0.0.0.0"
    # CUDA device, e.g. "cuda:0" or "cpu"
    device: str = "cuda"


def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(f"Host: {hostname} ({local_ip})")

    policy = EgoVerseAlohaPolicy(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
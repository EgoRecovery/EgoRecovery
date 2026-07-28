#!/usr/bin/env python3
"""
EgoVerse HPT Policy Server for ALOHA

Serves an HPT (hpt_cotrain_aloha_hand) checkpoint over WebSocket using the
same protocol as openpi, so client_aloha.py can connect without modification.

Usage:
    python aloha/server_policy_hpt.py \
        --checkpoint logs/<run>/checkpoints/last.ckpt \
        --port 8000

Camera key mapping (client → model key):
    cam_high        → front_img_1
    cam_left_wrist  → left_wrist_img
    cam_right_wrist → right_wrist_img
    state / qpos    → joint_positions  (14D, raw joint angles)

The server always uses the aloha_bimanual embodiment head for inference.
"""

import asyncio
import dataclasses
import http
import logging
import socket
import time
import traceback

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# msgpack helpers
# ---------------------------------------------------------------------------
try:
    from openpi_client import msgpack_numpy as _msgpack_numpy  # type: ignore
    import msgpack as _msgpack_lib  # type: ignore

    _opc_decode = getattr(_msgpack_numpy, "decode", None)

    def _unpackb(data: bytes) -> dict:
        if _opc_decode is not None:
            return _msgpack_lib.unpackb(data, object_hook=_opc_decode, raw=False)
        return _msgpack_numpy.unpackb(data)

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
        if b'__ndarray__' in obj:
            return np.frombuffer(obj[b'data'], dtype=np.dtype(obj[b'dtype'])).reshape(
                tuple(obj[b'shape'])
            )
        return _mnp.decode(obj)

    def _unpackb(data: bytes) -> dict:
        return msgpack.unpackb(data, raw=False, object_hook=_decode_numpy)

    def _to_openpi(obj):
        if isinstance(obj, np.ndarray):
            return {b'__ndarray__': True, b'data': obj.tobytes(),
                    b'dtype': str(obj.dtype), b'shape': list(obj.shape)}
        if isinstance(obj, dict):
            return {k: _to_openpi(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_openpi(v) for v in obj]
        return obj

    class _Packer:
        def __init__(self):
            self._p = msgpack.Packer(use_bin_type=True)

        def pack(self, obj) -> bytes:
            return self._p.pack(_to_openpi(obj))


# ---------------------------------------------------------------------------
# Camera key mapping: client key → model key
# ---------------------------------------------------------------------------
CLIENT_TO_MODEL_CAM: dict[str, str] = {
    "cam_high":        "front_img_1",
    "cam_left_wrist":  "left_wrist_img",
    "cam_right_wrist": "right_wrist_img",
}

EMBODIMENT_NAME = "aloha_bimanual"
DEFAULT_AC_KEY = "actions_joints"
DEFAULT_PROPRIO_KEY = "joint_positions"
ACTION_HORIZON = 100  # matches head_specs.aloha_bimanual.action_horizon


# ---------------------------------------------------------------------------
# HPT policy wrapper
# ---------------------------------------------------------------------------
class HPTAlohaPolicy:
    """Wraps a loaded HPT ModelWrapper for ALOHA inference."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from egomimic.pl_utils.pl_model import ModelWrapper
        from egomimic.rldb.embodiment.embodiment import get_embodiment_id

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, weights_only=False, map_location="cpu"
        )
        wrapper = wrapper.to(self.device)
        wrapper.eval()

        self.model = wrapper.model
        self.model.device = self.device
        self.data_schematic = self.model.data_schematic

        self.embodiment_id: int = get_embodiment_id(EMBODIMENT_NAME)
        self.camera_keys: list[str] = self.model.camera_keys[self.embodiment_id]
        self.proprio_keys: list[str] = self.model.proprio_keys[self.embodiment_id]
        self.proprio_key: str = (
            DEFAULT_PROPRIO_KEY
            if DEFAULT_PROPRIO_KEY in self.proprio_keys
            else self.proprio_keys[0]
        )
        self.ac_key: str = self.model.ac_keys.get(
            self.embodiment_id,
            self.model.ac_keys.get(EMBODIMENT_NAME, DEFAULT_AC_KEY),
        )

        head = self._resolve_head()
        self.chunk_size: int = getattr(head, "action_horizon", None) or ACTION_HORIZON
        self.action_dim: int = self._infer_action_dim(head)

        # output key from forward_eval: "{embodiment_name}_{ac_key}"
        self._pred_key = f"{EMBODIMENT_NAME}_{self.ac_key}"

        logger.info(f"  embodiment_id:  {self.embodiment_id}")
        logger.info(f"  camera_keys:    {self.camera_keys}")
        logger.info(f"  proprio_keys:   {self.proprio_keys}")
        logger.info(f"  proprio_key:    {self.proprio_key}")
        logger.info(f"  ac_key:         {self.ac_key}")
        logger.info(f"  pred_key:       {self._pred_key}")
        logger.info(f"  chunk_size:     {self.chunk_size}")
        logger.info(f"  action_dim:     {self.action_dim}")
        logger.info(f"  device:         {self.device}")

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        batch = self._prepare_batch(obs)
        preds = self.model.forward_eval(batch)
        actions = preds[self._pred_key]          # (1, chunk_size, 14)
        actions = actions.detach().cpu().numpy().squeeze(0)  # (chunk_size, 14)
        return {"actions": actions}

    def _prepare_batch(self, obs: dict) -> dict:
        inner: dict[str, torch.Tensor] = {}

        # images
        images_dict: dict = obs.get("images", {})
        for client_key, model_key in CLIENT_TO_MODEL_CAM.items():
            if model_key not in self.camera_keys:
                continue
            img = images_dict.get(client_key)
            if img is None:
                raise ValueError(f"Missing image '{client_key}'. Available: {list(images_dict)}")
            img_t = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)       # (1, C, H, W)
            inner[model_key] = img_t.to(self.device)

        # proprio
        state = obs.get("state") if obs.get("state") is not None else obs.get("qpos")
        if state is None:
            raise ValueError("Observation must contain 'state' or 'qpos'.")
        state_np = np.asarray(state, dtype=np.float32)
        inner[self.proprio_key] = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        # HPT.forward_eval expects the pre-conversion robomimic batch layout:
        # images as (B, C, H, W) and proprio as (B, D). It will add its own
        # extra sequence/modal dimensions in _robomimic_to_hpt_data.
        inner = self.data_schematic.normalize_data(inner, self.embodiment_id)

        # pad_mask and embodiment token
        inner["pad_mask"] = torch.ones(
            1, self.chunk_size, 1, device=self.device, dtype=torch.float32
        )
        inner["embodiment"] = torch.tensor(
            [self.embodiment_id], device=self.device, dtype=torch.int64
        )

        # forward_eval uses _batch[ac_key] only to get shape (B, T, D) for output crop.
        # Provide a dummy zero tensor so it doesn't KeyError during inference.
        inner[self.ac_key] = torch.zeros(
            1, self.chunk_size, self.action_dim, device=self.device, dtype=torch.float32
        )

        # HPT forward_eval expects {embodiment_id: inner_batch}
        return {self.embodiment_id: inner}

    def _resolve_head(self):
        heads = getattr(self.model, "heads", None)
        if heads is None:
            return None

        if hasattr(heads, "get"):
            head = heads.get(EMBODIMENT_NAME)
            if head is not None:
                return head

        try:
            return heads[EMBODIMENT_NAME]
        except Exception:
            return None

    def _infer_action_dim(self, head) -> int:
        output_dim = getattr(head, "output_dim", None)
        if output_dim is not None:
            return int(output_dim)

        try:
            shape = self.data_schematic.key_shape(self.ac_key, self.embodiment_id)
        except Exception:
            shape = None

        if shape:
            return int(shape[-1])

        return int(self.data_schematic.key_shape(self.proprio_key, self.embodiment_id)[-1])

    @property
    def metadata(self) -> dict:
        return {
            "model": "egoverse_hpt_aloha",
            "camera_keys": self.camera_keys,
            "proprio_keys": self.proprio_keys,
            "chunk_size": self.chunk_size,
        }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    def __init__(self, policy: HPTAlohaPolicy, host: str = "0.0.0.0", port: int = 8000):
        self._policy = policy
        self._host = host
        self._port = port

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as _ws_server

        async with _ws_server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"Server listening on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets

        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = _Packer()
        await websocket.send(packer.pack(self._policy.metadata))

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

                await websocket.send(packer.pack(result))
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
    checkpoint: str
    """Path to HPT ModelWrapper checkpoint (.ckpt)"""
    port: int = 8000
    host: str = "0.0.0.0"
    device: str = "cuda"


def main(args: Args) -> None:
    hostname = socket.gethostname()
    logger.info(f"Host: {hostname} ({socket.gethostbyname(hostname)})")

    policy = HPTAlohaPolicy(checkpoint_path=args.checkpoint, device=args.device)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

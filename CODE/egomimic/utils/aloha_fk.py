"""Lightweight ALOHA forward-kinematics wrapper for visualization.

Exposes ``AlohaFK.fk_pos(jnts_Nx6) -> xyz_Nx3`` in the robot base frame, matching
the interface that ``egomimic.utils.egomimicUtils.draw_actions(type="joints", ...)``
expects for its ``kinematics_solver`` argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch

_ALOHA_URDF = (
    Path(__file__).resolve().parents[1] / "resources" / "model_aloha.urdf"
)
_ALOHA_EE_LINK = "vx300s/ee_gripper_link"


class AlohaFK:
    """Per-arm 6-DoF FK via pytorch_kinematics.

    The VX300s left and right arms share the same kinematic chain (only the base
    transform differs, which is handled downstream by ``EXTRINSICS[...]["left"|"right"]``).
    One chain object is enough.
    """

    _cache: "AlohaFK | None" = None

    def __init__(
        self,
        urdf_path: Path = _ALOHA_URDF,
        ee_link: str = _ALOHA_EE_LINK,
    ):
        with open(urdf_path) as f:
            urdf_str = f.read()
        self.chain = pk.build_serial_chain_from_urdf(urdf_str, ee_link)
        assert self.chain.n_joints == 6, (
            f"Expected 6 joints for VX300s arm, got {self.chain.n_joints}"
        )

    @classmethod
    def get(cls) -> "AlohaFK":
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache

    def fk_pos(self, jnts: np.ndarray) -> np.ndarray:
        """Joint angles (N, 6) in rad → EE xyz (N, 3) in base frame (m)."""
        if jnts.ndim == 1:
            jnts = jnts[None]
        q = torch.as_tensor(jnts, dtype=torch.float32)
        with torch.no_grad():
            mat = self.chain.forward_kinematics(q, end_only=True).get_matrix()
        return mat[:, :3, 3].cpu().numpy().astype(np.float32)

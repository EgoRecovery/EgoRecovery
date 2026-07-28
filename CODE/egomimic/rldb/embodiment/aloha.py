from __future__ import annotations

from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    InterpolateLinear,
    NumpyToTensor,
    Transform,
)


class CastFloat32(Transform):
    """Cast specified keys to float32 (numpy or torch)."""

    def __init__(self, keys: list[str]):
        self.keys = keys

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            v = batch[key]
            if isinstance(v, np.ndarray):
                batch[key] = v.astype(np.float32)
        return batch


class Aloha(Embodiment):
    """ALOHA VX300s bimanual robot embodiment. Joint-space only."""

    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @classmethod
    def get_kinematics_solver(cls):
        from egomimic.utils.aloha_fk import AlohaFK

        return AlohaFK.get()

    @staticmethod
    def get_transform_list(
        mode: Literal["joint"] = "joint",
        chunk_length: int = 100,
        stride: int = 1,
        include_intent_targets: bool = False,
        include_rw_targets: bool = False,
    ) -> list[Transform]:
        if mode == "joint":
            return _build_aloha_bimanual_joint_transform_list(
                chunk_length=chunk_length,
                stride=stride,
                include_intent_targets=include_intent_targets,
                include_rw_targets=include_rw_targets,
            )
        raise ValueError(f"Unsupported mode: {mode}")

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: str,
        include_intent_targets: bool = False,
        include_rw_targets: bool = False,
    ):
        key_map = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.cam_high",
            },
            "observations.images.left_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.cam_left_wrist",
            },
            "observations.images.right_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.cam_right_wrist",
            },
            "observations.state.joint_positions": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.joint_positions",
            },
            "actions_joints": {
                "key_type": "action_keys",
                "zarr_key": "actions_joints",
                "horizon": 100,
            },
        }
        if include_intent_targets:
            # Intent-Head Co-Train targets (see docs/intent_head_cotrain.md).
            # Only enabled when the dataset has been preprocessed by
            # egomimic/scripts/precompute_gt_intent.py AND the top-level config
            # asks for intent keys. DataSchematic treats "intent_keys" like
            # proprio for zscore norm but HPT does NOT route it through the stem.
            key_map["gt_intent"] = {
                "key_type": "intent_keys",
                "zarr_key": "gt_intent_raw",
            }
            # Bool mask, True = frame's future phase-window passed active filter.
            # key_type != proprio/action so DataSchematic does not normalize.
            # horizon=1 forces the reader to return a (1,) ndarray instead of a
            # numpy.bool scalar (which NumpyToTensor rejects).
            key_map["gt_intent_valid"] = {
                "key_type": "metadata_keys",
                "zarr_key": "gt_intent_valid",
                "horizon": 1,
            }
        if include_rw_targets:
            # Recovery-Window-Conditioned Intent Bottleneck masks
            # (doc §A / §C / §E.3). Both are per-anchor-frame bools written
            # by scripts/data_analysis/apply_recovery_labels.py. They flow
            # through the same metadata path as gt_intent_valid (no z-score).
            key_map["recovery_intent_valid"] = {
                "key_type": "metadata_keys",
                "zarr_key": "recovery_intent_valid",
                "horizon": 1,
            }
            key_map["sr_label"] = {
                "key_type": "metadata_keys",
                "zarr_key": "sr_label",
                "horizon": 1,
            }
        return key_map


def _build_aloha_bimanual_joint_transform_list(
    *,
    obs_key: str = "observations.state.joint_positions",
    action_key: str = "actions_joints",
    chunk_length: int = 100,
    stride: int = 1,
    include_intent_targets: bool = False,
    include_rw_targets: bool = False,
) -> list[Transform]:
    """Joint-space transform pipeline for ALOHA bimanual.

    The raw data is already 14D joint positions per frame.
    We only need to interpolate the action chunk to the target chunk_length
    and convert to tensors. Optionally, when the keymap exposes intent
    targets, cast/convert those too.
    """
    cast_keys = [action_key, obs_key]
    tensor_keys = [action_key, obs_key]
    if include_intent_targets:
        cast_keys.append("gt_intent")
        tensor_keys.extend(["gt_intent", "gt_intent_valid"])
    if include_rw_targets:
        # bool masks; no float cast needed, only NumpyToTensor.
        tensor_keys.extend(["recovery_intent_valid", "sr_label"])

    return [
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=action_key,
            output_action_key=action_key,
            stride=stride,
        ),
        CastFloat32(keys=cast_keys),
        NumpyToTensor(keys=tensor_keys),
    ]

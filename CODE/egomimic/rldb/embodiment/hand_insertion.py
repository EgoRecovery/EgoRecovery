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


class HandInsertion(Embodiment):
    """Human hand bimanual embodiment for insertion tasks.

    End-effector space: 14D (7D per arm: x, y, z, rx, ry, rz, gripper).
    No coordinate frame transforms applied — raw ee_pose used directly.
    """

    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list(
        mode: Literal["ee"] = "ee",
        chunk_length: int = 100,
        stride: int = 1,
        include_intent_targets: bool = False,
        include_rw_targets: bool = False,
    ) -> list[Transform]:
        if mode == "ee":
            return _build_hand_ee_transform_list(
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
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "actions_ee": {
                "key_type": "action_keys",
                "zarr_key": "actions_ee",
                "horizon": 100,
            },
        }
        if include_intent_targets:
            # Intent-Head Co-Train targets (see docs/intent_head_cotrain.md).
            # Only enabled when the dataset has been preprocessed by
            # egomimic/scripts/precompute_gt_intent.py AND the top-level config
            # asks for intent keys.
            key_map["gt_intent"] = {
                "key_type": "intent_keys",
                "zarr_key": "gt_intent_raw",
            }
            # horizon=1 forces reader to return a (1,) ndarray instead of a
            # numpy.bool scalar (which NumpyToTensor rejects).
            key_map["gt_intent_valid"] = {
                "key_type": "metadata_keys",
                "zarr_key": "gt_intent_valid",
                "horizon": 1,
            }
        if include_rw_targets:
            # Recovery-Window-Conditioned Intent Bottleneck masks
            # (doc §A / §C / §E.3). See aloha.py for full notes.
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


def _build_hand_ee_transform_list(
    *,
    obs_key: str = "observations.state.ee_pose",
    action_key: str = "actions_ee",
    chunk_length: int = 100,
    stride: int = 1,
    include_intent_targets: bool = False,
    include_rw_targets: bool = False,
) -> list[Transform]:
    """End-effector transform pipeline for human hand bimanual.

    The raw data is 14D ee_pose per frame (xyz+rpy+gripper per arm).
    We only need to interpolate the action chunk to the target chunk_length
    and convert to tensors. Optionally cast/convert intent targets when the
    keymap exposes them.
    """
    cast_keys = [action_key, obs_key]
    tensor_keys = [action_key, obs_key]
    if include_intent_targets:
        cast_keys.append("gt_intent")
        tensor_keys.extend(["gt_intent", "gt_intent_valid"])
    if include_rw_targets:
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

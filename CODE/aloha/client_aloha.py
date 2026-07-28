#!/usr/bin/env python3
"""
ALOHA Client for EgoVerse ACT Policy Server

Connects to serve_policy_aloha.py via WebSocket (openpi-compatible protocol).
Reads observations from ALOHA robot via ROS and executes predicted actions.

Usage:

# 1. Optional SSH tunnel (on local machine, forward server port to localhost)
ssh -i /path/to/key -p <port> -L 8000:localhost:8000 <user>@<server>

# 2. Start policy server (on server node)
source .venv/bin/activate

python aloha/server_policy_aloha.py \
    --checkpoint /path/to/logs/act_aloha_30ep/insertion_30_new_2026-04-17_06-31-18/checkpoints/last.ckpt \
    --port 8000

    
python aloha/server_policy_hpt.py \
 --checkpoint /path/to/logs/E3_hpt_cotrain_human_success/robot_success45_human_success_68_2026-04-19_11-06-58/checkpoints/last.ckpt \
 --port 8000
    
    # 3. Run client (on ALOHA robot)
python client_aloha.py \
    --host 127.0.0.1 \
    --port 8000 \
    --prompt "Pick up the cup brush on the left, grasp the measuring cylinder on the right, insert the brush into the cylinder, and place it on the cup rack." \
    --ros-master-uri http://agilex:11311 \
    --rate 40.0 \
    --use-actions-interpolation \
    --num-interpolation-steps 10 \
    --action-exec-horizon 40 \
    --image-height 480 \
    --image-width 640

# Key tuning args:
#   --rate                    action execution frequency in Hz (default 10.0, lower = slower)
#   --num-interpolation-steps interpolated frames inserted between each predicted step (default 5)
#   --action-exec-horizon     predicted steps executed per chunk before re-querying (default 50)
#   --image-height            resize height to match training (default 240)
#   --image-width             resize width to match training (default 320)
#   --no-use-actions-interpolation  disable interpolation
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import collections
import cv2
import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

# ---------------------------------------------------------------------------
# WebSocket client (openpi-compatible, no openpi_client dependency)
# ---------------------------------------------------------------------------
try:
    from openpi_client import websocket_client_policy as _wcp  # type: ignore

    class _PolicyClient:
        """Thin wrapper around openpi_client if available."""

        def __init__(self, host: str, port: int, api_key: Optional[str] = None):
            self._client = _wcp.WebsocketClientPolicy(host=host, port=port, api_key=api_key)

        def get_server_metadata(self):
            return self._client.get_server_metadata()

        def infer(self, obs: dict) -> dict:
            return self._client.infer(obs)

except ImportError:
    import msgpack  # type: ignore
    import msgpack_numpy as _mnp  # type: ignore
    import websocket  # type: ignore  (websocket-client)

    _mnp.patch()

    class _PolicyClient:  # type: ignore
        """Standalone WebSocket client using websocket-client + msgpack-numpy."""

        def __init__(self, host: str, port: int, api_key: Optional[str] = None):
            url = f"ws://{host}:{port}"
            self._ws = websocket.WebSocket()
            self._ws.connect(url)
            # Receive metadata frame
            raw = self._ws.recv_bytes()
            self._metadata = msgpack.unpackb(raw, raw=False)
            self._packer = msgpack.Packer(default=_mnp.encode, use_bin_type=True)

        def get_server_metadata(self):
            return self._metadata

        def infer(self, obs: dict) -> dict:
            self._ws.send_binary(self._packer.pack(obs))
            raw = self._ws.recv_bytes()
            return msgpack.unpackb(raw, raw=False)


# ---------------------------------------------------------------------------
# Logging via rospy
# ---------------------------------------------------------------------------
class RosLogger:
    def debug(self, msg, *args, **kwargs):
        rospy.logdebug(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        rospy.loginfo(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        rospy.logwarn(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        rospy.logerr(msg % args if args else msg)


logger = RosLogger()


# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------
class RollingStats:
    def __init__(self, window: int = 100):
        self.values: list[float] = []
        self.window = window

    def add(self, v: float):
        self.values.append(v)
        if len(self.values) > self.window:
            self.values.pop(0)

    def avg(self) -> float:
        return float(sum(self.values) / len(self.values)) if self.values else 0.0

    def count(self) -> int:
        return len(self.values)


# ---------------------------------------------------------------------------
# Action interpolation
# ---------------------------------------------------------------------------
def actions_interpolation(
    pre_action: np.ndarray, actions: np.ndarray, num_inserts: int = 5
) -> np.ndarray:
    assert pre_action.ndim == 1 and pre_action.shape[0] == 14
    assert actions.ndim == 2 and actions.shape[1] == 14
    keyframes = np.vstack([pre_action[None, :], actions])
    if keyframes.shape[0] == 1:
        return keyframes.copy()
    pieces = [keyframes[0][None, :]]
    for i in range(keyframes.shape[0] - 1):
        seg = np.linspace(keyframes[i], keyframes[i + 1], num=num_inserts + 2, endpoint=True)[1:]
        pieces.append(seg)
    return np.vstack(pieces)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: Optional[str] = None
    rate: float = 10.0
    steps: int = 0

    img_front_topic: str = "/camera_f/color/image_raw"
    img_left_topic: str = "/camera_l/color/image_raw"
    img_right_topic: str = "/camera_r/color/image_raw"
    img_front_depth_topic: str = "/camera_f/depth/image_raw"
    img_left_depth_topic: str = "/camera_l/depth/image_raw"
    img_right_depth_topic: str = "/camera_r/depth/image_raw"

    puppet_arm_left_cmd_topic: str = "/master/joint_left"
    puppet_arm_right_cmd_topic: str = "/master/joint_right"
    puppet_arm_left_topic: str = "/puppet/joint_left"
    puppet_arm_right_topic: str = "/puppet/joint_right"
    robot_base_topic: str = "/odom_raw"
    robot_base_cmd_topic: str = "/cmd_vel"

    use_robot_base: bool = False
    use_depth_image: bool = False
    publish_rate: int = 30
    arm_steps_length: List[float] = dataclasses.field(
        default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
    )

    prompt: str = "pick up the cube"
    ros_master_uri: Optional[str] = None
    ros_hostname: Optional[str] = None
    ros_ip: Optional[str] = None

    use_actions_interpolation: bool = True
    num_interpolation_steps: int = 5       # frames inserted at chunk boundary (between steps)
    num_intra_interpolation_steps: int = 0 # frames inserted between predicted steps within chunk

    # Image resolution — must match training data resolution
    image_height: int = 240
    image_width: int = 320

    # Gripper normalization params (must match training config)
    binarize_gripper: bool = False
    gripper_threshold: float = 0.5
    left_gripper_raw_min: float = 0.0
    left_gripper_raw_max: float = 4.5
    left_gripper_flip: bool = True
    right_gripper_raw_min: float = -0.0505457
    right_gripper_raw_max: float = 4.5
    right_gripper_flip: bool = True

    reset_joints_open: List[float] = dataclasses.field(
        default_factory=lambda: [
            -0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
            -0.032616615295410156, -0.00286102294921875, 0.00095367431640625,
            3.557830810546875,
            -0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
            -0.053597450256347656, -0.00476837158203125, -0.00209808349609375,
            3.557830810546875,
        ]
    )
    reset_joints_closed: List[float] = dataclasses.field(
        default_factory=lambda: [
            -0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
            -0.032616615295410156, -0.00286102294921875, 0.00095367431640625,
            -0.3393220901489258,
            -0.00133514404296875, 0.00247955322265625, 0.01583099365234375,
            -0.032616615295410156, -0.00286102294921875, 0.00095367431640625,
            -0.3397035598754883,
        ]
    )

    action_exec_horizon: int = 50
    max_action_chunk_size: int = 50
    action_completion_timeout: float = 4.0
    action_completion_pos_tol: float = 0.02
    apply_joint_deltas_to_current: bool = False
    log_commanded_targets: bool = True


# ---------------------------------------------------------------------------
# ROS operator (identical to client_delta.py)
# ---------------------------------------------------------------------------
class RosOperator:
    def __init__(self, args: Args):
        self.args = args
        self.bridge = CvBridge()
        self.img_left_deque: deque = deque()
        self.img_right_deque: deque = deque()
        self.img_front_deque: deque = deque()
        self.img_left_depth_deque: deque = deque()
        self.img_right_depth_deque: deque = deque()
        self.img_front_depth_deque: deque = deque()
        self.puppet_arm_left_deque: deque = deque()
        self.puppet_arm_right_deque: deque = deque()
        self.robot_base_deque: deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.last_log_time = 0.0
        self._init_ros()

    def _init_ros(self):
        rospy.init_node("aloha_egoverse_client", anonymous=True, disable_signals=True)
        a = self.args
        rospy.Subscriber(a.img_left_topic, Image, self._cb_img_left, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(a.img_right_topic, Image, self._cb_img_right, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(a.img_front_topic, Image, self._cb_img_front, queue_size=1000, tcp_nodelay=True)
        if a.use_depth_image:
            rospy.Subscriber(a.img_left_depth_topic, Image, self._cb_img_left_depth, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(a.img_right_depth_topic, Image, self._cb_img_right_depth, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(a.img_front_depth_topic, Image, self._cb_img_front_depth, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(a.puppet_arm_left_topic, JointState, self._cb_arm_left, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(a.puppet_arm_right_topic, JointState, self._cb_arm_right, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(a.robot_base_topic, Odometry, self._cb_base, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(a.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(a.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(a.robot_base_cmd_topic, Twist, queue_size=10)

    # ---- callbacks ----
    def _append(self, q: deque, msg):
        if len(q) >= 2000:
            q.popleft()
        q.append(msg)

    def _cb_img_left(self, msg): self._append(self.img_left_deque, msg)
    def _cb_img_right(self, msg): self._append(self.img_right_deque, msg)
    def _cb_img_front(self, msg): self._append(self.img_front_deque, msg)
    def _cb_img_left_depth(self, msg): self._append(self.img_left_depth_deque, msg)
    def _cb_img_right_depth(self, msg): self._append(self.img_right_depth_deque, msg)
    def _cb_img_front_depth(self, msg): self._append(self.img_front_depth_deque, msg)
    def _cb_arm_left(self, msg): self._append(self.puppet_arm_left_deque, msg)
    def _cb_arm_right(self, msg): self._append(self.puppet_arm_right_deque, msg)
    def _cb_base(self, msg): self._append(self.robot_base_deque, msg)

    # ---- joint names ----
    def _joint_names(self, side: str) -> list[str]:
        q = self.puppet_arm_left_deque if side == "left" else self.puppet_arm_right_deque
        try:
            names = list(q[-1].name)
            return names if len(names) == 7 else None
        except Exception:
            return None

    # ---- publish ----
    def puppet_arm_publish(self, left, right):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = self._joint_names("left") or [f"joint{i}" for i in range(7)]
        msg.position = left
        self.puppet_arm_left_publisher.publish(msg)
        msg.name = self._joint_names("right") or msg.name
        msg.position = right
        self.puppet_arm_right_publisher.publish(msg)

    def puppet_arm_publish_continuous_smooth(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = right_arm = None
        while not rospy.is_shutdown():
            if self.puppet_arm_left_deque:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if self.puppet_arm_right_deque:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is not None and right_arm is not None:
                break
            rate.sleep()
        left_sym = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_sym = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        while flag and not rospy.is_shutdown():
            flag = False
            for i in range(len(left)):
                d = abs(left[i] - left_arm[i])
                if d < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_sym[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                d = abs(right[i] - right_arm[i])
                if d < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_sym[i] * self.args.arm_steps_length[i]
                    flag = True
            self.puppet_arm_publish(left_arm, right_arm)
            rate.sleep()

    # ---- get_frame ----
    def get_frame(self):
        queues = [self.img_left_deque, self.img_right_deque, self.img_front_deque,
                  self.puppet_arm_left_deque, self.puppet_arm_right_deque]
        if any(len(q) == 0 for q in queues):
            return False

        frame_time = min(q[-1].header.stamp.to_sec() for q in queues)

        def _pop_at(q, t):
            while len(q) > 1 and q[0].header.stamp.to_sec() < t:
                q.popleft()
            return q.popleft()

        def _to_rgb(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                img = np.zeros((self.args.image_height, self.args.image_width, 3), dtype=np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to training resolution
            img = cv2.resize(img, (self.args.image_width, self.args.image_height), interpolation=cv2.INTER_AREA)
            return np.transpose(img, (2, 0, 1)).astype(np.uint8)  # CHW

        img_front = _to_rgb(_pop_at(self.img_front_deque, frame_time))
        img_left = _to_rgb(_pop_at(self.img_left_deque, frame_time))
        img_right = _to_rgb(_pop_at(self.img_right_deque, frame_time))
        arm_left = _pop_at(self.puppet_arm_left_deque, frame_time)
        arm_right = _pop_at(self.puppet_arm_right_deque, frame_time)
        return img_front, img_left, img_right, arm_left, arm_right


# ---------------------------------------------------------------------------
# AlohaController
# ---------------------------------------------------------------------------
class AlohaController:
    def __init__(self, args: Args):
        self.args = args
        self.ros = RosOperator(args)
        rospy.loginfo("Waiting for ROS data (2s)...")
        time.sleep(2.0)
        self.last_action: Optional[np.ndarray] = None

    def reset_to_default_pose(self):
        rospy.loginfo("Resetting to open gripper pose...")
        self.ros.puppet_arm_publish_continuous_smooth(
            self.args.reset_joints_open[:7], self.args.reset_joints_open[7:14]
        )
        time.sleep(0.5)
        rospy.loginfo("Resetting to closed gripper pose...")
        self.ros.puppet_arm_publish_continuous_smooth(
            self.args.reset_joints_closed[:7], self.args.reset_joints_closed[7:14]
        )
        self.last_action = np.array(
            self.args.reset_joints_closed[:7] + self.args.reset_joints_closed[7:14]
        )
        rospy.loginfo("Reset complete.")

    def read_observation(self) -> Optional[Dict[str, Any]]:
        ret = self.ros.get_frame()
        if not ret:
            return None
        img_front, img_left, img_right, arm_left, arm_right = ret

        qpos_left = np.array(arm_left.position)
        qpos_right = np.array(arm_right.position)
        left_arm = qpos_left[:6]
        right_arm = qpos_right[:6]

        left_grip = float(qpos_left[6]) if qpos_left.size > 6 else 0.0
        right_grip = float(qpos_right[6]) if qpos_right.size > 6 else 0.0
        qpos = np.concatenate([left_arm, [left_grip], right_arm, [right_grip]])

        return {
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "state": qpos,
            "qpos": qpos,
            "prompt": self.args.prompt,
        }

    def apply_action(self, action: np.ndarray):
        if action.ndim == 1:
            action = action[None, :]
        action = action.copy()

        # Clip to execution horizon first (in terms of original predicted steps)
        action = action[: self.args.action_exec_horizon]

        if self.args.use_actions_interpolation:
            n_boundary = self.args.num_interpolation_steps
            n_intra = self.args.num_intra_interpolation_steps

            # Use actual robot joint positions as the chunk boundary start point
            # to avoid jumps when the arm hasn't finished executing the previous chunk.
            actual_start = None
            try:
                if self.ros.puppet_arm_left_deque and self.ros.puppet_arm_right_deque:
                    cur_l = np.array(self.ros.puppet_arm_left_deque[-1].position)[:7]
                    cur_r = np.array(self.ros.puppet_arm_right_deque[-1].position)[:7]
                    actual_start = np.concatenate([cur_l, cur_r])
            except Exception:
                pass

            expanded = []
            for i, act in enumerate(action):
                if i == 0:
                    # Chunk boundary: interpolate from actual robot position
                    n = n_boundary
                    prev = actual_start if actual_start is not None else (
                        self.last_action if self.last_action is not None else act
                    )
                else:
                    # Within chunk: interpolate between consecutive predicted steps
                    n = n_intra
                    prev = action[i - 1]

                if n > 0:
                    interp = np.linspace(prev, act, num=n + 2)[1:]  # n interp frames + act
                    expanded.extend(interp)
                else:
                    expanded.append(act)
            action = np.array(expanded)
        rate = rospy.Rate(float(self.args.rate))
        for i, act in enumerate(action):
            left_action = act[:7]
            right_action = act[7:14]
            if self.args.apply_joint_deltas_to_current:
                try:
                    if self.ros.puppet_arm_left_deque:
                        cur_l = np.array(self.ros.puppet_arm_left_deque[-1].position)[:7]
                        left_action[:6] += cur_l[:6]
                    if self.ros.puppet_arm_right_deque:
                        cur_r = np.array(self.ros.puppet_arm_right_deque[-1].position)[:7]
                        right_action[:6] += cur_r[:6]
                except Exception as e:
                    logger.warning(f"apply_joint_deltas failed: {e}")
            self.ros.puppet_arm_publish(left_action, right_action)
            rate.sleep()
        self.last_action = action[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_client(args: Args):
    if args.ros_master_uri:
        os.environ["ROS_MASTER_URI"] = args.ros_master_uri
    if args.ros_hostname:
        os.environ["ROS_HOSTNAME"] = args.ros_hostname
    if args.ros_ip:
        os.environ["ROS_IP"] = args.ros_ip

    import subprocess
    try:
        r = subprocess.run(["rostopic", "list"], capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            print("Cannot connect to ROS master. Is roscore running?")
            return
    except Exception as e:
        print(f"ROS master check failed: {e}")
        return

    controller = AlohaController(args)

    # Optional reset
    try:
        import select
        print("Reset arms to default pose? (y / Enter to skip, 5s timeout)")
        i, _, _ = select.select([sys.stdin], [], [], 5.0)
        if i and sys.stdin.readline().strip().lower() == "y":
            controller.reset_to_default_pose()
    except Exception:
        pass

    # Connect to policy server
    rospy.loginfo(f"Connecting to EgoVerse server at {args.host}:{args.port} ...")
    policy = _PolicyClient(host=args.host, port=args.port, api_key=args.api_key)
    rospy.loginfo(f"Server metadata: {policy.get_server_metadata()}")

    # Warmup
    rospy.loginfo("Warmup (2 iterations)...")
    for _ in range(2):
        obs = controller.read_observation()
        if obs:
            policy.infer(obs)
    rospy.loginfo("Warmup done.")

    step = 0
    infer_stats = RollingStats(200)
    step_stats = RollingStats(200)

    try:
        while not rospy.is_shutdown():
            if args.steps > 0 and step >= args.steps:
                break
            t0 = time.time()
            obs = controller.read_observation()
            if obs is None:
                time.sleep(0.05)
                continue

            t1 = time.time()
            result = policy.infer(obs)
            actions = result["actions"]
            infer_t = time.time() - t1
            infer_stats.add(infer_t)

            rospy.loginfo(
                f"Step {step}: infer={infer_t * 1000:.1f}ms  "
                f"actions shape={actions.shape}  "
                f"range=[{actions.min():.3f}, {actions.max():.3f}]"
            )

            controller.apply_action(actions)
            step_stats.add(time.time() - t0)

            if step % 10 == 0 and step_stats.count() > 0:
                rospy.loginfo(
                    f"Perf: step_avg={step_stats.avg() * 1000:.1f}ms  "
                    f"infer_avg={infer_stats.avg() * 1000:.1f}ms"
                )
            step += 1

    except KeyboardInterrupt:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}", exc_info=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = tyro.cli(Args)
    run_client(args)


if __name__ == "__main__":
    main()

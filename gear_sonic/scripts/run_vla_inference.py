"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> send initial pose and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
import time

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.dataset_initial_poses import (
    DatasetInitialPoses,
    load_dataset_initial_poses,
)
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.tactile_subscriber import TactileSubscriber
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    policy_timeout_ms: int = 60_000
    """Timeout for one policy request, including a slow first inference."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Camera modality
    stereo_ego_view: bool = True
    """Feed the ego view to the policy as a stereo pair (``ego_view_left`` +
    ``ego_view_right``) instead of a single monocular ``ego_view``. Must match
    the modality the policy was trained with. Requires the camera server to be
    started with ``--ego-view-camera usb_stereo`` (which publishes both
    ``ego_view_left`` and ``ego_view_right`` image keys)."""

    use_tactile: bool = True
    """Require and forward current vest/left-arm/right-arm tactile frames."""

    tactile_zmq_host: str = "localhost"
    """Host for the JuQiao tactile publisher."""

    tactile_zmq_port: int = 5558
    """Port for the JuQiao tactile publisher."""

    tactile_max_age_sec: float = 0.1
    """Maximum accepted age for each tactile device frame."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    dataset_path: str = ""
    """Optional LeRobot dataset path. When set, the initial motion token sent
    on 'i' is the per-prompt mean of first-frame ``action.motion_token`` values
    from that dataset (falling back to a global mean over all episodes, and to
    the hardcoded LATENT_INITIAL_MOTION_TOKEN if the dataset can't be loaded)."""

    initial_pose_ramp_seconds: float = 1.0
    """Duration over which to cosine-ease the motion token (and hand joints)
    from the last sent value to the initial-pose target when 'i' is pressed.
    Set to 0 to disable smoothing and snap directly to the target (legacy
    behavior). Ignored on the first 'i' press of a session (no last-sent
    value to ramp from)."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
    stereo_ego_view: bool = True,
    tactile_subscriber=None,
    use_tactile: bool = True,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    tactile = None
    if use_tactile:
        if tactile_subscriber is None:
            raise ValueError("use_tactile=True requires a tactile_subscriber")
        tactile = tactile_subscriber.read()
        if tactile is None:
            if log_errors:
                print(
                    f"[DEBUG] prepare_observation: {tactile_subscriber.status()}",
                    flush=True,
                )
            return None

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    images = camera_msg["images"]
    video = {}
    if stereo_ego_view:
        for eye_key in ("ego_view_left", "ego_view_right"):
            if eye_key not in images:
                if log_errors:
                    print(
                        f"[DEBUG] prepare_observation: stereo_ego_view set but "
                        f"'{eye_key}' missing from camera images "
                        f"(available: {list(images.keys())}). Is the camera server "
                        f"running with --ego-view-camera usb_stereo?",
                        flush=True,
                    )
                return None
            video[eye_key] = images[eye_key][np.newaxis, np.newaxis]
        ego_timestamp = camera_msg["timestamps"]["ego_view_left"]
    else:
        video["ego_view"] = images["ego_view"][np.newaxis, np.newaxis]
        ego_timestamp = camera_msg["timestamps"]["ego_view"]

    if "left_wrist" in images:
        video["left_wrist"] = images["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in images:
        video["wrist_view"] = images["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": ego_timestamp,
    }
    if tactile is not None:
        observation["tactile"] = {
            key: np.asarray(value, dtype=np.uint8)[np.newaxis, np.newaxis]
            for key, value in tactile.items()
        }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.policy_timeout_ms,
    )

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    deployment_metadata = n1_policy.get_deployment_metadata()
    if deployment_metadata:
        expected_video_keys = ["ego_view_left", "ego_view_right"] if config.stereo_ego_view else ["ego_view"]
        checks = {
            "video_keys": expected_video_keys,
            "requires_tactile": config.use_tactile,
            "action_horizon": config.action_horizon,
        }
        mismatches = [
            f"{key}={deployment_metadata.get(key)!r} (client expects {expected!r})"
            for key, expected in checks.items()
            if deployment_metadata.get(key) != expected
        ]
        if mismatches:
            raise ValueError("Policy deployment contract mismatch: " + "; ".join(mismatches))
        print_green(f"Policy deployment contract verified: {deployment_metadata}")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    tactile_subscriber = None
    if config.use_tactile:
        tactile_subscriber = TactileSubscriber(
            host=config.tactile_zmq_host,
            port=config.tactile_zmq_port,
            max_age_sec=config.tactile_max_age_sec,
        )
        print_green(
            f"Tactile subscriber connected to tcp://{config.tactile_zmq_host}:"
            f"{config.tactile_zmq_port} (max age {config.tactile_max_age_sec:.3f}s)"
        )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    # Last action we put on the wire — used by publish_initial_pose to ramp
    # from the robot's current commanded pose to the initial-pose target on
    # 'i', avoiding a one-frame snap. Remains None until the main loop has
    # sent at least one frame.
    last_sent_motion_token: np.ndarray | None = None
    last_sent_left_hand_joints: np.ndarray | None = None
    last_sent_right_hand_joints: np.ndarray | None = None

    # Optional dataset-derived initial poses (per-prompt average of first-frame
    # motion tokens). Falls back to LATENT_INITIAL_MOTION_TOKEN if not provided
    # or if loading fails.
    dataset_poses: DatasetInitialPoses | None = None
    if config.dataset_path:
        try:
            dataset_poses = load_dataset_initial_poses(config.dataset_path)
            print_green(
                f"Loaded dataset initial poses from {config.dataset_path} "
                f"({dataset_poses.n_episodes_loaded} episodes, "
                f"{len(dataset_poses.by_prompt)} tasks)"
            )
            print(dataset_poses.summary())
        except Exception as e:
            print(
                f"WARNING: failed to load dataset initial poses from "
                f"{config.dataset_path}: {e}. Falling back to hardcoded "
                f"LATENT_INITIAL_MOTION_TOKEN."
            )
            dataset_poses = None

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position.

        If a previous action frame has been sent and ``initial_pose_ramp_seconds
        > 0``, cosine-ease the motion token (and hand joints) from the last
        sent value to the target over that many seconds, one ZMQ frame per
        ``loop_period``. Otherwise snap directly to the target in a single
        frame (legacy behavior, used on the very first 'i' press).
        """
        nonlocal last_sent_motion_token
        nonlocal last_sent_left_hand_joints
        nonlocal last_sent_right_hand_joints

        print("Moving to initial pose")
        target_left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        target_right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )

        target_motion_token = np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)
        token_source = "hardcoded LATENT_INITIAL_MOTION_TOKEN"
        if dataset_poses is not None:
            current_prompt = language_prompt_ref[0]
            dataset_token = dataset_poses.lookup(current_prompt)
            if dataset_token is not None:
                target_motion_token = np.asarray(dataset_token, dtype=np.float32)
                if current_prompt in dataset_poses.by_prompt:
                    token_source = f"dataset mean for prompt {current_prompt!r}"
                else:
                    token_source = f"dataset global mean (prompt {current_prompt!r} not in dataset)"

        ramp_seconds = float(config.initial_pose_ramp_seconds)
        can_ramp = ramp_seconds > 0.0 and last_sent_motion_token is not None
        if can_ramp:
            n_steps = max(1, int(round(ramp_seconds * config.action_publish_rate)))
            start_token = np.asarray(last_sent_motion_token, dtype=np.float32)
            start_left = np.asarray(
                (
                    last_sent_left_hand_joints
                    if last_sent_left_hand_joints is not None
                    else target_left_hand
                ),
                dtype=np.float32,
            )
            start_right = np.asarray(
                (
                    last_sent_right_hand_joints
                    if last_sent_right_hand_joints is not None
                    else target_right_hand
                ),
                dtype=np.float32,
            )
            for i in range(1, n_steps + 1):
                # Cosine ease (smoothstep): zero velocity at both endpoints,
                # so the robot doesn't jerk into or out of the ramp.
                alpha = 0.5 - 0.5 * np.cos(np.pi * (i / n_steps))
                blended_token = ((1.0 - alpha) * start_token + alpha * target_motion_token).astype(
                    np.float32
                )
                blended_left = ((1.0 - alpha) * start_left + alpha * target_left_hand).astype(
                    np.float32
                )
                blended_right = ((1.0 - alpha) * start_right + alpha * target_right_hand).astype(
                    np.float32
                )
                ramp_msg = pack_latent_action_message(
                    motion_token=blended_token,
                    frame_index=np.array([0], dtype=np.int64),
                    left_hand_joints=blended_left,
                    right_hand_joints=blended_right,
                )
                zmq_socket.send(ramp_msg)
                time.sleep(loop_period)
            print_green(
                f"Sent ramped latent initial pose via ZMQ "
                f"({n_steps} steps over {ramp_seconds:.2f}s, target: {token_source})"
            )
        else:
            zmq_message = pack_latent_action_message(
                motion_token=target_motion_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=target_left_hand,
                right_hand_joints=target_right_hand,
            )
            zmq_socket.send(zmq_message)
            print_green(f"Sent latent initial pose via ZMQ ({token_source})")

        last_sent_motion_token = target_motion_token
        last_sent_left_hand_joints = target_left_hand
        last_sent_right_hand_joints = target_right_hand

        time.sleep(1.0)
        print("Initial pose done.")

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            print("Moving to initial pose")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk")
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
                stereo_ego_view=config.stereo_ego_view,
                tactile_subscriber=tactile_subscriber,
                use_tactile=config.use_tactile,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s)"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    last_sent_motion_token = motion_token
                    last_sent_left_hand_joints = left_hand_joints
                    last_sent_right_hand_joints = right_hand_joints
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        if tactile_subscriber is not None:
            tactile_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)

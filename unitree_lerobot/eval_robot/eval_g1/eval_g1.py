''''
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/common/robot_devices/control_utils.py
'''

import time
import torch
import logging
import threading
import numpy as np
from copy import copy
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from multiprocessing import shared_memory, Array, Lock
from collections import deque

from lerobot.policies.factory import make_policy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig


# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert tensors to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name, value in list(observation.items()):
            if isinstance(value, torch.Tensor):
                if "images" in name:
                    value = value.type(torch.float32) / 255
                    value = value.permute(2, 0, 1).contiguous()
                observation[name] = value.unsqueeze(0).to(device)
            else:
                # keep non-tensor entries (e.g., 'task') as is
                observation[name] = value

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
    cfg: EvalRealConfig,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    # image from cfg overrides with sensible defaults
    img_config = {
        'fps': cfg.head_camera_fps if cfg.head_camera_fps is not None else 30,
        'head_camera_type': cfg.head_camera_type or 'opencv',
        'head_camera_image_shape': [
            cfg.head_camera_image_shape_h if cfg.head_camera_image_shape_h is not None else 480,
            cfg.head_camera_image_shape_w if cfg.head_camera_image_shape_w is not None else 1280,
        ],
        'head_camera_id_numbers': [int(x) for x in (cfg.head_camera_ids_csv.split(',') if cfg.head_camera_ids_csv else ['0'])],
        'wrist_camera_type': (cfg.wrist_camera_type if cfg.wrist_camera_type else 'opencv') if (cfg.wrist_enabled is None or int(cfg.wrist_enabled) == 1) else None,
        'wrist_camera_image_shape': [
            cfg.wrist_camera_image_shape_h if cfg.wrist_camera_image_shape_h is not None else 480,
            cfg.wrist_camera_image_shape_w if cfg.wrist_camera_image_shape_w is not None else 640,
        ],
        'wrist_camera_id_numbers': [int(x) for x in (cfg.wrist_camera_ids_csv.split(',') if cfg.wrist_camera_ids_csv else ['2','4'])],
    }
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config and img_config['wrist_camera_type']:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name,
                                 image_show = bool(getattr(cfg, 'image_show', False)))
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name,
                                 image_show = bool(getattr(cfg, 'image_show', False)))

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    robot_config = {
        'arm_type': getattr(cfg, 'arm_type', 'g1'),
        'hand_type': getattr(cfg, 'hand_type', 'dex3'),
    }

    # init pose from configured episode index (default 0)
    _ep_idx = int(getattr(cfg, 'init_from_episode_index', 0))
    _ep_idx = max(0, min(_ep_idx, len(dataset.episode_data_index["from"]) - 1))
    from_idx = dataset.episode_data_index["from"][_ep_idx].item()
    step = dataset[from_idx]
    to_idx = dataset.episode_data_index["to"][_ep_idx].item()

    # arm
    arm_ctrl = G1_29_ArmController()
    init_left_arm_pose = step['observation.state'][:14].cpu().numpy()

    # hand
    if robot_config['hand_type'] == "dex3":
        left_hand_array = Array('d', 7, lock = True)          # [input]
        right_hand_array = Array('d', 7, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
        init_left_hand_pose = step['observation.state'][14:21].cpu().numpy()
        init_right_hand_pose = step['observation.state'][21:].cpu().numpy()

    elif robot_config['hand_type'] == "gripper":
        left_hand_array = Array('d', 1, lock=True)             # [input]
        right_hand_array = Array('d', 1, lock=True)            # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
        init_left_hand_pose = step['observation.state'][14].cpu().numpy()
        init_right_hand_pose = step['observation.state'][15].cpu().numpy()
    else:
        pass

    #===============init robot=====================
    if getattr(cfg, 'default_start', True):
        user_input_ok = True
    else:
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
        user_input_ok = (user_input.lower() == 's')
    if user_input_ok:

        if getattr(cfg, 'startup_delay_s', 0.0) and cfg.startup_delay_s > 0:
            time.sleep(cfg.startup_delay_s)

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        print("init robot pose")
        arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
        left_hand_array[:] = init_left_hand_pose
        right_hand_array[:] = init_right_hand_pose

        print("wait robot to pose")
        time.sleep(1)

        frequency = float(getattr(cfg, 'frequency', 50.0))

        # EMA smoothing params (configurable). Fall back order: per-part -> global -> default 0.15
        def _resolve_alpha(part_value, global_value, default_value=0.15):
            return (
                float(part_value)
                if part_value is not None
                else (float(global_value) if global_value is not None else float(default_value))
            )

        alpha_global = getattr(cfg, 'smoothing_alpha_global', None)
        alpha_arm = _resolve_alpha(getattr(cfg, 'smoothing_alpha_arm', None), alpha_global)
        alpha_hand = _resolve_alpha(getattr(cfg, 'smoothing_alpha_hand', None), alpha_global)
        alpha_leg = _resolve_alpha(getattr(cfg, 'smoothing_alpha_leg', None), alpha_global)

        # Last sent action snapshot
        last_action = None  # will store the previously sent action for EMA

        while True:

            # Get images
            current_tv_image = tv_img_array.copy()
            current_wrist_image = wrist_img_array.copy() if WRIST else None

            # Assign image data
            left_top_camera = current_tv_image[:, :tv_img_shape[1] // 2] if BINOCULAR else current_tv_image
            right_top_camera = current_tv_image[:, tv_img_shape[1] // 2:] if BINOCULAR else None
            left_wrist_camera, right_wrist_camera = (
                (current_wrist_image[:, :wrist_img_shape[1] // 2], current_wrist_image[:, wrist_img_shape[1] // 2:])
                if WRIST else (None, None)
            )

            observation = {
                "observation.images.cam_left_high": torch.from_numpy(left_top_camera),
                "observation.images.cam_right_high": torch.from_numpy(right_top_camera) if BINOCULAR else None,
                "observation.images.cam_left_wrist": torch.from_numpy(left_wrist_camera) if WRIST else None,
                "observation.images.cam_right_wrist": torch.from_numpy(right_wrist_camera) if WRIST else None,
            }

            # get current state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            # dex hand or gripper
            if robot_config['hand_type'] == "dex3":
                with dual_hand_data_lock:
                    left_hand_state = dual_hand_state_array[:7]
                    right_hand_state = dual_hand_state_array[-7:]
            elif robot_config['hand_type'] == "gripper":
                with dual_gripper_data_lock:
                    left_hand_state = [dual_gripper_state_array[1]]
                    right_hand_state = [dual_gripper_state_array[0]]
            
            observation["observation.state"] = torch.from_numpy(np.concatenate((current_lr_arm_q, left_hand_state, right_hand_state), axis=0)).float()

            observation = {k: v for k, v in observation.items() if v is not None}
            # Move only tensor observations to target device
            tensor_obs = {k: v for k, v in observation.items() if isinstance(v, torch.Tensor)}
            observation = {
                key: tensor_obs[key].to(device, non_blocking=device.type == "cuda") for key in tensor_obs
            }
            # Inject natural language task as list[str] for batch size 1
            task_text = getattr(cfg, 'task', "") if getattr(cfg, 'task', None) is not None else ""
            observation['task'] = [task_text]

            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )
            action = action.cpu().numpy()

            # Exponential Moving Average smoothing
            if last_action is None or last_action.shape != action.shape:
                smoothed_action = action
            else:
                smoothed_action = action.copy()
                # arm (0:14)
                smoothed_action[:14] = last_action[:14] * (1.0 - alpha_arm) + action[:14] * alpha_arm
                # hand (dex3: 14:28 or gripper: 14:16)。统一对 14: 末尾做手的 alpha_hand
                smoothed_action[14:] = last_action[14:] * (1.0 - alpha_hand) + action[14:] * alpha_hand
                # leg 保留接口，若未来有腿部维度，可在此单独平滑对应切片
            last_action = smoothed_action
            
            # exec action
            arm_ctrl.ctrl_dual_arm(smoothed_action[:14], np.zeros(14))
            if robot_config['hand_type'] == "dex3":
                left_hand_array[:] = smoothed_action[14:21]
                right_hand_array[:] = smoothed_action[21:]
            elif robot_config['hand_type'] == "gripper":
                left_hand_array[:] = smoothed_action[14]
                right_hand_array[:] = smoothed_action[15]
        
            #time.sleep(1/frequency)
            from lerobot.utils.robot_utils import busy_wait
            busy_wait(1/frequency)
            # exec action


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    # prefer local dataset path when provided, else fall back to repo_id
    if getattr(cfg, 'dataset_path', None):
        dataset = LeRobotDataset(cfg.repo_id or "local_dataset", root = cfg.dataset_path)
    else:
        dataset = LeRobotDataset(repo_id = cfg.repo_id)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(policy, dataset, cfg)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()

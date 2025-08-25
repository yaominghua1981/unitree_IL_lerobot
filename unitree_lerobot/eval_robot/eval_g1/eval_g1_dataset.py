''''
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/common/robot_devices/control_utils.py
'''

import torch
import tqdm
import logging
import time
import numpy as np
# Enable matplotlib with NumPy 1.24.3 compatibility
import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("matplotlib successfully imported")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"matplotlib not available: {e}")
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"matplotlib import failed: {e}")
from copy import copy
from pprint import pformat
from torch import nn
from contextlib import nullcontext
from multiprocessing import Array, Lock

import os
import sys

# Add the project root to Python path to ensure lerobot can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the correct lerobot source directory
lerobot_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lerobot', 'src'))
if os.path.exists(lerobot_src_path) and lerobot_src_path not in sys.path:
    sys.path.insert(0, lerobot_src_path)
    
print(f"Added lerobot path: {lerobot_src_path}")

import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

try:
    from lerobot.policies.factory import make_policy
    from lerobot.utils.utils import (
        get_safe_torch_device,
        init_logging,
    )
    from lerobot.configs import parser
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    logging.error("Failed to import lerobot module. Please ensure it's in your PYTHONPATH.")
    logging.error(f"Current PYTHONPATH: {sys.path}")
    raise

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
@dataclass
class CameraConfig:
    """Configuration for camera display during evaluation.
    
    Attributes:
        show: Whether to display video during evaluation
    """
    show: bool = True

@dataclass
class RobotConfig:
    """Configuration for the robot.
    
    Attributes:
        arm_type: Type of the robot arm (e.g., 'g1')
        hand_type: Type of the robot hand (e.g., 'dex3')
        send_real_robot: Whether to send commands to the real robot
    """
    arm_type: str = "g1"
    hand_type: str = "dex3"
    send_real_robot: bool = False

@dataclass
class PolicyConfig:
    """Configuration for the policy model.
    
    Attributes:
        type: Type of the policy model (e.g., 'smolvla')
        path: Path to the policy model
        use_amp: Whether to use automatic mixed precision
        device: Device to run the policy on (e.g., 'cuda' or 'cpu')
    """
    type: str = "smolvla"
    path: str = ""
    use_amp: bool = True
    device: str = "cuda"

@dataclass
class EvalRealConfig:
    """Configuration for evaluation.
    
    Attributes:
        dataset_path: Path to the dataset
        repo_id: Repository ID for the dataset (if using Hugging Face datasets)
        policy: Policy configuration
        robot: Robot configuration
        episodes: Number of episodes to evaluate
        init_from_episode_index: Starting episode index
        camera: Camera configuration
        task: Task description for the policy
        image_show: Whether to show video during evaluation (alias for camera.show)
    """
    dataset_path: str = ""
    repo_id: str = ""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    episodes: int = 1
    init_from_episode_index: int = 0
    camera: CameraConfig = field(default_factory=CameraConfig)
    task: str = "Pick up the object and place it in the target location."
    image_show: bool = field(init=False)
    
    def __post_init__(self):
        # Make image_show an alias for camera.show for backward compatibility
        object.__setattr__(self, 'image_show', self.camera.show)
        
        # Log robot configuration
        if self.robot.send_real_robot:
            logging.info(f"Robot configuration: arm_type={self.robot.arm_type}, hand_type={self.robot.hand_type}, send_real_robot={self.robot.send_real_robot}")

# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(observation, policy, device, use_amp, **kwargs):
    """
    Get action from policy given observation.
    
    Args:
        observation: Dictionary containing observation data
        policy: The policy model
        device: Device to run the policy on
        use_amp: Whether to use automatic mixed precision
        **kwargs: Additional arguments (not used, for compatibility)
        
    Returns:
        Action tensor
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if isinstance(observation[name], torch.Tensor):
                if observation[name].dtype == torch.uint8:
                    observation[name] = observation[name].float() / 255.0
                # Add batch dimension if needed
                if len(observation[name].shape) == 1:  # 1D tensor (state)
                    observation[name] = observation[name].unsqueeze(0)  # Add batch dim
                elif len(observation[name].shape) == 3:  # 3D tensor (image)
                    observation[name] = observation[name].unsqueeze(0)  # Add batch dim
                observation[name] = observation[name].to(device)

        # Get action from policy - handle different policy interfaces
        if hasattr(policy, 'select_action'):
            action = policy.select_action(observation)
        elif hasattr(policy, 'forward'):
            action = policy(observation)
            if isinstance(action, dict):
                action = action.get('action', action.get('actions'))
        else:
            raise ValueError("Policy must have either select_action or forward method")

        # Ensure action is a tensor and remove batch dimension if present
        if isinstance(action, (list, np.ndarray)):
            action = torch.from_numpy(np.array(action))
        if len(action.shape) > 1:  # Remove batch dimension if present
            action = action.squeeze(0)

        return action.to('cpu')


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
    cfg: EvalRealConfig,
    arm_controller=None,
    hand_controller=None
):
    """
    Evaluate the policy on the dataset.
    
    Args:
        policy: The policy model to evaluate
        dataset: The dataset to evaluate on
        cfg: Evaluation configuration containing camera and robot settings
        arm_controller: Optional arm controller for real robot control
        hand_controller: Optional hand controller for real robot control
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    
    # Reset the policy and environments.
    policy.reset()

    # Use robot configuration from the passed config
    show_camera = cfg.camera.show
    robot_config = {
        'arm_type': cfg.robot.arm_type,
        'hand_type': cfg.robot.hand_type,
        'send_real_robot': cfg.robot.send_real_robot,
    }

    ground_truth_actions = []
    predicted_actions = []
    
    # Get available camera names from the dataset
    sample = dataset[0]
    if 'observation' in sample:
        camera_names = [name for name in sample['observation'].keys() 
                       if name.startswith('camera_') or name.startswith('cam_')]
    else:
        # Check for camera keys directly in the sample
        camera_names = [name for name in sample.keys() 
                       if (name.startswith('observation.images.') or 
                           name.startswith('camera_') or name.startswith('cam_'))]

    # Initialize robot control if needed
    if robot_config['send_real_robot']:
        # Get initial pose from first episode
        init_from_idx = dataset.episode_data_index["from"][cfg.init_from_episode_index].item()
        init_step = dataset[init_from_idx]
        
        # arm
        arm_ctrl = G1_29_ArmController()
        init_left_arm_pose = init_step['observation.state'][:14].cpu().numpy()

        # hand
        if robot_config['hand_type'] == "dex3":
            left_hand_array = Array('d', 7, lock = True)          # [input]
            right_hand_array = Array('d', 7, lock = True)         # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
            init_left_hand_pose = init_step['observation.state'][14:21].cpu().numpy()
            init_right_hand_pose = init_step['observation.state'][21:].cpu().numpy()

        elif robot_config['hand_type'] == "gripper":
            left_hand_array = Array('d', 1, lock=True)             # [input]
            right_hand_array = Array('d', 1, lock=True)            # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
            init_left_hand_pose = init_step['observation.state'][14].cpu().numpy()
            init_right_hand_pose = init_step['observation.state'][15].cpu().numpy()

        #===============init robot=====================
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
        if user_input.lower() == 's':
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            print("init robot pose")
            arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
            left_hand_array[:] = init_left_hand_pose
            right_hand_array[:] = init_right_hand_pose

            print("wait robot to pose")
            time.sleep(1)
    else:
        # For non-robot mode, still ask for start signal
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
        if user_input.lower() != 's':
            return

    frequency = 50.0

    # Process multiple episodes as configured
    total_episodes = min(cfg.episodes, len(dataset.episode_data_index["from"]) - cfg.init_from_episode_index)
    logging.info(f"Processing {total_episodes} episodes starting from episode {cfg.init_from_episode_index}")
    
    for episode_idx in range(cfg.init_from_episode_index, cfg.init_from_episode_index + total_episodes):
        episode_from_idx = dataset.episode_data_index["from"][episode_idx].item()
        episode_to_idx = dataset.episode_data_index["to"][episode_idx].item()
        
        logging.info(f"Processing episode {episode_idx}: steps {episode_from_idx} to {episode_to_idx} ({episode_to_idx - episode_from_idx} steps)")
        
        episode_ground_truth_actions = []
        episode_predicted_actions = []
        
        for step_idx in tqdm.tqdm(range(episode_from_idx, episode_to_idx), desc=f"Episode {episode_idx}"):
            step = dataset[step_idx]
            # Handle different dataset structures
            if 'observation' in step:
                observation = {k: torch.from_numpy(v) for k, v in step['observation'].items()}
            else:
                # Extract observation keys from the step directly, keeping the full key names
                observation = {}
                for k, v in step.items():
                    if k.startswith('observation.'):
                        # Keep the full key name as the policy expects it
                        if isinstance(v, np.ndarray):
                            observation[k] = torch.from_numpy(v)
                        else:
                            observation[k] = v
            
            # Add task field if it exists in the step, or use the one from config
            if 'task' in step:
                observation['task'] = step['task']
            else:
                # Use task from configuration
                observation['task'] = cfg.task
            
            # Display video frames if enabled
            if show_camera and camera_names:
                try:
                    # Display the first available camera feed
                    frame = observation.get(camera_names[0], None)
                    if frame is not None:
                        # Convert from CHW to HWC for OpenCV
                        if len(frame.shape) == 3 and frame.shape[0] in [1, 3]:  # CHW format
                            frame = frame.permute(1, 2, 0).numpy()
                            if frame.shape[2] == 1:  # Grayscale
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            elif frame.shape[2] == 3:  # RGB to BGR
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Resize if needed for display
                        h, w = frame.shape[:2]
                        if h > 720 or w > 1280:  # Limit display size
                            scale = min(720/h, 1280/w)
                            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                        
                        cv2.imshow('Video Feed', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nVideo display stopped by user")
                            show_camera = False
                            cv2.destroyAllWindows()
                except Exception as e:
                    logging.warning(f"Error displaying video: {e}")
                    show_camera = False
                    if 'cv2' in locals():
                        cv2.destroyAllWindows()

            # Get action from policy
            action = predict_action(
                observation=observation,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                use_amp=policy.config.use_amp
            )
            
            # Send command to robot if controllers are provided
            if arm_controller is not None and hand_controller is not None:
                try:
                    send_robot_command(arm_controller, hand_controller, action)
                except Exception as e:
                    logging.error(f"Error sending command to robot: {str(e)}")
                    raise

            action = action.cpu().numpy()

            episode_ground_truth_actions.append(step["action"].numpy())
            episode_predicted_actions.append(action)

            if robot_config['send_real_robot']:
                # Execute action on real robot
                arm_ctrl.ctrl_dual_arm(action[:14], np.zeros(14))
                if robot_config['hand_type'] == "dex3":
                    left_hand_array[:] = action[14:21]
                    right_hand_array[:] = action[21:]
                elif robot_config['hand_type'] == "gripper":
                    left_hand_array[:] = action[14]
                    right_hand_array[:] = action[15]
            
                time.sleep(1/frequency)
        
        episode_ground_truth_actions = np.array(episode_ground_truth_actions)
        episode_predicted_actions = np.array(episode_predicted_actions)

        ground_truth_actions.append(episode_ground_truth_actions)
        predicted_actions.append(episode_predicted_actions)

        # Get the number of timesteps and action dimensions
        n_timesteps, n_dims = episode_ground_truth_actions.shape

        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Skipping action visualization: matplotlib not available")
            continue

        try:
            # Create a figure with subplots for each action dimension
            fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims), sharex=True)
            fig.suptitle('Ground Truth vs Predicted Actions')

            # Plot each dimension
            for dim in range(n_dims):
                axes[dim].plot(episode_ground_truth_actions[:, dim], 'b-', label='Ground Truth')
                axes[dim].plot(episode_predicted_actions[:, dim], 'r--', label='Predicted')
                axes[dim].set_ylabel(f'Action Dim {dim}')
                axes[dim].grid(True)
                if dim == 0:
                    axes[dim].legend()

            # Set common x-label
            axes[-1].set_xlabel('Timestep')

            plt.tight_layout()
            
            # Save the figure
            output_file = f'episode_{episode_idx}_action_comparison.png'
            plt.savefig(output_file)
            logging.info(f"Action comparison plot saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Failed to generate action comparison plot: {str(e)}")


def parse_args():
    """Parse command line arguments and configuration file.
    
    Returns:
        EvalRealConfig: Configuration object with all settings
    """
    # Create a temporary parser to get the config file path
    temp_parser = argparse.ArgumentParser()
    temp_parser.add_argument("--config", type=str, default="",
                           help="Path to configuration file (JSON format)")
    temp_args, _ = temp_parser.parse_known_args()
    
    # If config file is specified, load it
    if temp_args.config and os.path.exists(temp_args.config):
        import json
        logging.info(f"Loading configuration from {temp_args.config}")
        with open(temp_args.config, 'r') as f:
            config_data = json.load(f)
        
        # Extract camera config
        camera_data = config_data.get('camera', {})
        camera_config = CameraConfig(
            show=camera_data.get('show', True)
        )
        
        # Extract policy config
        policy_data = config_data.get('policy', {})
        policy_config = PolicyConfig(
            type=policy_data.get('type', 'smolvla'),
            path=policy_data.get('path', ''),
            use_amp=policy_data.get('use_amp', True),
            device=policy_data.get('device', 'cuda')
        )
        
        # Extract robot config
        robot_data = config_data.get('robot', {})
        robot_config = RobotConfig(
            arm_type=robot_data.get('arm_type', 'g1'),
            hand_type=robot_data.get('hand_type', 'dex3'),
            send_real_robot=robot_data.get('send_real_robot', False)
        )
        
        # Create config object
        return EvalRealConfig(
            dataset_path=config_data.get('dataset', {}).get('path', ''),
            repo_id=config_data.get('dataset', {}).get('repo_id', ''),
            policy=policy_config,
            robot=robot_config,
            episodes=config_data.get('evaluation', {}).get('episodes', 1),
            init_from_episode_index=config_data.get('evaluation', {}).get('init_from_episode_index', 0),
            camera=camera_config,
            task=config_data.get('evaluation', {}).get('task', 'Pick up the object and place it in the target location.')
        )
    
    # If no config file, use command line arguments
    parser = argparse.ArgumentParser(description='Evaluate policy on dataset')
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument('--dataset_path', type=str, default="",
                             help='Path to the dataset directory')
    dataset_group.add_argument('--repo_id', type=str, default="",
                             help='Hugging Face repository ID for the dataset')
    
    # Policy arguments
    policy_group = parser.add_argument_group('Policy')
    policy_group.add_argument('--policy.type', type=str, dest='policy_type', default='smolvla',
                            help='Type of the policy model (e.g., smolvla)')
    policy_group.add_argument('--policy.path', type=str, dest='policy_path', default="",
                            help='Path to the policy model')
    policy_group.add_argument('--policy.use_amp', type=lambda x: x.lower() == 'true', 
                            dest='policy_use_amp', default=True,
                            help='Whether to use automatic mixed precision')
    policy_group.add_argument('--policy.device', type=str, dest='device', default='cuda',
                            help='Device to run the policy on (e.g., cuda, cpu)')
    
    # Robot arguments
    robot_group = parser.add_argument_group('Robot')
    robot_group.add_argument('--robot_config.arm_type', type=str, dest='robot_arm_type', default="g1", help="Type of the robot arm (e.g., 'g1')")
    robot_group.add_argument('--robot_config.hand_type', type=str, dest='robot_hand_type', default="dex3", help="Type of the robot hand (e.g., 'dex3')")
    robot_group.add_argument('--robot_config.send_real_robot', type=lambda x: x.lower() == 'true', dest='robot_send_real_robot', default=False, help="Whether to send commands to the real robot")
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--episodes', type=int, default=1,
                          help='Number of episodes to evaluate')
    eval_group.add_argument('--init_from_episode_index', type=int, default=0,
                          help='Starting episode index')
    eval_group.add_argument('--task', type=str, default="Pick up the object and place it in the target location.",
                          help='Task description for the policy')
    
    # Display arguments
    display_group = parser.add_argument_group('Display')
    display_group.add_argument('--image_show', action='store_true',
                             help='Show video during evaluation')
    
    args = parser.parse_args()
    
    # Create config objects
    camera_config = CameraConfig(show=args.image_show)
    
    policy_config = PolicyConfig(
        type=args.policy_type,
        path=args.policy_path,
        use_amp=args.policy_use_amp,
        device=args.device
    )
    
    robot_config = RobotConfig(
        arm_type=args.robot_arm_type,
        hand_type=args.robot_hand_type,
        send_real_robot=args.robot_send_real_robot
    )
    
    return EvalRealConfig(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        policy=policy_config,
        robot=robot_config,
        episodes=args.episodes,
        init_from_episode_index=args.init_from_episode_index,
        camera=camera_config,
        task=args.task
    )

def init_robot_control(cfg: EvalRealConfig):
    """Initialize robot control based on configuration.
    
    Args:
        cfg: Evaluation configuration
        
    Returns:
        Tuple containing arm_controller and hand_controller if send_real_robot is True, else (None, None)
    """
    # Align with eval_g1.py: initialize controllers inside eval loop function
    return None, None

def send_robot_command(arm_controller, hand_controller, action):
    """Send command to the robot.
    
    Args:
        arm_controller: The arm controller instance
        hand_controller: The hand controller instance
        action: The action to execute (should include arm and hand commands)
    """
    if arm_controller is None or hand_controller is None:
        return
        
    try:
        # Extract arm and hand commands from action
        # This assumes action is a dictionary with 'arm' and 'hand' keys
        arm_cmd = action.get('arm', None)
        hand_cmd = action.get('hand', None)
        
        if arm_cmd is not None:
            arm_controller.send_joint_command(arm_cmd)
            
        if hand_cmd is not None:
            hand_controller.send_hand_command(hand_cmd)
            
    except Exception as e:
        logging.error(f"Error sending robot command: {str(e)}")

def eval_main(cfg: EvalRealConfig):
    """
    Main evaluation function that loads the dataset, creates the policy, and runs evaluation.
    
    Args:
        cfg: Evaluation configuration
    """
    logging.info("Starting evaluation with configuration:")
    logging.info(pformat(asdict(cfg)))

    # Initialize device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Update device in config in case it was changed by get_safe_torch_device
    cfg.policy.device = device.type
    logging.info(f"Using device: {device}")

    # Initialize robot control if needed
    arm_controller, hand_controller = init_robot_control(cfg)

    try:
        # Initialize dataset
        logging.info(f"Loading dataset from path: {cfg.dataset_path}")
        try:
            # prefer local dataset path when provided, else fall back to repo_id
            if cfg.dataset_path:
                dataset = LeRobotDataset("local_dataset", root=cfg.dataset_path)
            else:
                dataset = LeRobotDataset(repo_id=cfg.repo_id)
        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            raise
        
        logging.info(f"Dataset loaded with {len(dataset)} samples")

        # Create policy config object
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.normalize import NormalizationMode
        
        # Load config from the local model directory
        config_path = os.path.join(cfg.policy.path, 'config.json')
        if os.path.exists(config_path):
            # Load the config file manually and filter out invalid fields
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Remove invalid fields for SmolVLAConfig
            invalid_fields = ['type', 'proj_width', 'attention_implementation']
            for field in invalid_fields:
                if field in config_data:
                    del config_data[field]
            
            # Convert normalization mapping strings to enum values
            if 'normalization_mapping' in config_data:
                norm_map = config_data['normalization_mapping']
                for key, value in norm_map.items():
                    if isinstance(value, str):
                        norm_map[key] = NormalizationMode(value)
            
            # Create SmolVLAConfig directly from the filtered data
            policy_cfg = SmolVLAConfig(**config_data)

        else:
            # Fallback to default config if local config not found
            policy_cfg = SmolVLAConfig()
        
        # Load smolvla config from dataset config file to override parameters
        dataset_config_path = '/home/lin/youkechuiguo/youkechuiguo_robot/config/eval/config_eval_smolvla/config_dataset.json'
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                dataset_config = json.load(f)
            if 'smolvla' in dataset_config:
                smolvla_config = dataset_config['smolvla']
                # Override all SmolVLA parameters from config_dataset.json
                for key, value in smolvla_config.items():
                    if hasattr(policy_cfg, key):
                        setattr(policy_cfg, key, value)
                        print(f"Override {key}: {value}")
                    else:
                        print(f"Warning: {key} is not a valid SmolVLAConfig parameter")
        
        # Update with our settings
        policy_cfg.use_amp = cfg.policy.use_amp
        policy_cfg.device = cfg.policy.device
        
        # Set the model to use local files only
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        logging.info(f"Creating policy with config: {policy_cfg}")
        
        # Create policy
        policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
        policy.eval()
        policy.to(device)
        logging.info(f"Policy loaded on device: {device}")

        # Run evaluation
        logging.info(f"Starting evaluation for {cfg.episodes} episodes...")
        try:
            with torch.no_grad():
                # Use autocast if AMP is enabled
                autocast_ctx = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
                with autocast_ctx:
                    # Pass robot controllers to eval_policy if needed
                    eval_policy(
                        policy=policy,
                        dataset=dataset,
                        cfg=cfg,
                        arm_controller=arm_controller,
                        hand_controller=hand_controller
                    )
            
            logging.info("Evaluation completed successfully!")
            
        except KeyboardInterrupt:
            logging.info("Evaluation interrupted by user.")
            
        finally:
            # Make sure to stop the robot when done or on error
            if cfg.robot.send_real_robot and arm_controller is not None:
                logging.info("Stopping robot...")
                try:
                    arm_controller.stop()
                    if hand_controller is not None:
                        hand_controller.stop()
                except Exception as e:
                    logging.error(f"Error while stopping robot: {str(e)}")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    cfg = parse_args()
    eval_main(cfg)

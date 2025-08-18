''''
Refer to:   lerobot/configs/eval.py
'''

import logging
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class EvalRealConfig:
    repo_id: str = ""
    policy: PreTrainedConfig | None = None
    # runtime overrides
    arm_type: str = "g1"
    hand_type: str = "dex3"
    frequency: float = 50.0
    # local dataset path (optional)
    dataset_path: str | None = None
    # evaluation controls
    default_start: bool = True
    startup_delay_s: float = 0.0
    # natural language task instruction
    task: str = ""
    # dataset episode selection
    episodes: str | None = None  # csv list of episode indices
    init_from_episode_index: int = 0
    # camera flat config from centralized JSON
    camera_fps: int | None = None
    head_enabled: int | None = None  # 0/1
    head_camera_type: str | None = None
    head_camera_image_shape_h: int | None = None
    head_camera_image_shape_w: int | None = None
    head_camera_ids_csv: str | None = None
    wrist_enabled: int | None = None  # 0/1
    head_camera_fps: int | None = None
    wrist_camera_fps: int | None = None
    wrist_camera_type: str | None = None
    wrist_camera_image_shape_h: int | None = None
    wrist_camera_image_shape_w: int | None = None
    wrist_camera_ids_csv: str | None = None

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )


    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a model."""

import json
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple
import re
import random

import hydra
import torch
from torch.utils.data import DataLoader
import numpy as np

from mtrl.env.types import EnvType
from mtrl.logger import Logger
from mtrl.utils import checkpointable
from mtrl.utils import config as config_utils
from mtrl.utils import utils, video
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType

import importlib
import copy

from pathlib import Path


class Experiment(checkpointable.Checkpointable):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        self.id = experiment_id
        self.config = config
        self.robot_type = self.detect_robot_type()

        # Initialize robot configuration for automatic XML loading
        try:
            import sys
            import os
            metaworld_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Metaworld')
            if os.path.exists(metaworld_path) and metaworld_path not in sys.path:
                sys.path.insert(0, metaworld_path)
            
            from metaworld.envs.robot_utils import set_robot_type, patch_all_metaworld_envs
            if self.robot_type and self.robot_type != 'unknown':
                set_robot_type(self.robot_type)
                patch_all_metaworld_envs()
                print(f"[Experiment] Initialized robot configuration: {self.robot_type}")
        except Exception as e:
            print(f"[Experiment] Could not initialize robot configuration: {e}")
            import traceback
            traceback.print_exc()

        self.device = torch.device(self.config.setup.device)

        self.get_env_metadata = get_env_metadata
        self.envs, self.env_metadata = self.build_envs() 
        self.env_counter = 0

        with open(config.setup.metadata_path) as f:
            self.metadata = json.load(f)

        key = "ordered_task_list"
        if key in self.env_metadata and self.env_metadata[key]:
            ordered_task_dict = {
                task: index for index, task in enumerate(self.env_metadata[key])
            }
        else:
            ordered_task_dict = {}
            raise NotImplemented

        key = "envs_to_exclude_during_training"
        if key in self.config.experiment and self.config.experiment[key]:
            self.envs_to_exclude_during_training = {
                ordered_task_dict[task] for task in self.config.experiment[key]
            }
            print(
                f"Excluding the following environments: {self.envs_to_exclude_during_training}"
            )
            self.env_indices = [id for task, id in ordered_task_dict.items() if task not in self.config.experiment[key]]
        else:
            self.envs_to_exclude_during_training = set()
            self.env_indices = [id for task, id in ordered_task_dict.items()]
        
        self.env_indices_i = list(range(len(self.env_indices)))

        self.action_space = self.env_metadata["action_space"]
        assert self.action_space.low.min() >= -1
        assert self.action_space.high.max() <= 1

        self.env_obs_space = self.env_metadata["env_obs_space"]

        env_obs_shape = self.env_obs_space.shape
        action_shape = self.action_space.shape
        self.env_obs_shape = self.env_obs_space.shape[0]
        self.action_shape = self.action_space.shape[0]

        self.task_name_to_idx = ordered_task_dict

        self.config = prepare_config(config=self.config, env_metadata=self.env_metadata)
        should_resume_experiment = self.config.experiment.should_resume

        self.seq_len = self.config.transformer_collective_network.transformer_encoder.representation_transformer.sequence_len
        self.cls_token = self.config.transformer_collective_network.transformer_encoder.transformer.with_cls
        
        metadata_keys = list(self.metadata.keys())
        self.task_names = [x for x in ordered_task_dict.keys() if x not in self.config.experiment.envs_to_exclude_during_training]
        self.policy_import_names = self.task_names
        self.num_envs = len(self.env_indices)
        if "MT5" in config.env.benchmark._target_:
            self.task_embedding = [self.metadata[name[:-2]] for name in self.task_names]
            self.task_num = np.array([metadata_keys.index(name[:-2]) for name in self.task_names])
            self.policy_import_names = [name[:-2] for name in self.task_names]
        else:
            self.task_embedding = [self.metadata[name] for name in self.task_names]
            self.task_num = np.array([metadata_keys.index(name) for name in self.task_names])

        self.scripted_policies = [self.import_policy_class(name[:-3]) for name in self.policy_import_names]

        if self.config.experiment.mode == 'train_worker':
            # Worker
            self.agent = [hydra.utils.instantiate( 
                self.config.worker.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            ) for _ in range(self.num_envs)]

            self.model_dir = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'model'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir_distill_tmp = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer_distill_tmp'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir_distill = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer_distill'
                )
            ) for task_name in self.task_names]

            self.replay_buffer = [hydra.utils.instantiate(
                self.config.replay_buffer.replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            self.replay_buffer_distill_tmp = [hydra.utils.instantiate(
                self.config.replay_buffer.col_tmp_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            assert self.config.experiment.col_sampling_freq==0 or self.config.replay_buffer.col_tmp_replay_buffer.capacity >= self.config.experiment.num_train_steps / self.config.experiment.col_sampling_freq * self.config.experiment.col_training_samples
            assert self.config.replay_buffer.col_replay_buffer.capacity >= self.config.replay_buffer.col_tmp_replay_buffer.capacity

            self.start_step = 0

            if should_resume_experiment:
                for i in range(self.num_envs):
                    self.start_step = self.agent[i].load_latest_step(model_dir=self.model_dir[i])
                    self.replay_buffer[i].load(save_dir=self.buffer_dir[i])
                    self.replay_buffer_distill_tmp[i].load(save_dir=self.buffer_dir_distill_tmp[i])

        elif self.config.experiment.mode == 'record':
            self.buffer_dir_distill = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'recording_buffer_distill'
                )
            ) for task_name in self.task_names]
            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            self.start_step = 0
        elif self.config.experiment.mode == 'distill_collective_transformer':
            self.col_buffer_loc = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "buffer/collective_buffer/train")
            )
            self.distill_buffer_names = os.listdir(self.col_buffer_loc)
            self.distill_names_with_seed = [name.replace("buffer_distill_", "") for name in self.distill_buffer_names]
            pattern = r'(-\d+)?_seed_\d+$'
            self.distill_names = [re.sub(pattern, '', name) for name in self.distill_names_with_seed]
            self.num_datasets = len(self.distill_names)

            # collective network
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                experiment=self.config.experiment.experiment,
                seed=self.config.setup.seed,
                device=self.device,
            )
                
            self.col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir", self.config.experiment.experiment, f"model_col_seed_{self.config.setup.seed}")
            )
            self.buffer_dir_distill = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/train/{i}")
            ) for i in self.distill_buffer_names]
            self.buffer_dir_val = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/validation/{i}")
            ) for i in self.distill_buffer_names]

            replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )

            replay_buffer.load_multiple_buffer(self.buffer_dir_distill)
            self.replay_buffer_distill = replay_buffer

            self.replay_buffer_val = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )
            
            self.replay_buffer_val.load_multiple_buffer(self.buffer_dir_val)
            
            
            self.col_start_step = 0

            if should_resume_experiment:
                self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)

        elif self.config.experiment.mode == 'evaluate_predictive_adapter':
            # --- START: Evaluate pa mode initialization logic ---
            
            # 1) Instantiate TransformerAgent
            #    Note: env_obs_shape here is just a placeholder for the model structure,
            #    since we will load a pretrained model whose structure is already fixed.
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                experiment=self.config.experiment.experiment,
                seed=self.config.setup.seed,
                device=self.device,
            )
            
            # 2) Set model directory and load model
            #    Here we reuse the model directory configuration from 'distill_collective_transformer'
            self.col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir", self.config.experiment.experiment, f"model_col_seed_{self.config.setup.seed}")
            )
            print(f"Loading collective agent (with predictive adapter) from: {self.col_model_dir}")
            self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)
            if self.col_start_step == 0:
                 print("[WARN] No model found to evaluate in `model_dir/model_col`. Make sure a trained agent exists.")


            # 3) Load validation dataset (Validation Set)
            #    We use the buffer under the 'validation' folder for evaluation
            self.val_buffer_loc = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "buffer/collective_buffer/validation")
            )
            self.distill_buffer_names_val = os.listdir(self.val_buffer_loc)
            if not self.distill_buffer_names_val:
                raise FileNotFoundError(f"Validation data not found in {self.val_buffer_loc}")
            
            buffer_dirs_val = [
                utils.make_dir(os.path.join(self.val_buffer_loc, i))
                for i in self.distill_buffer_names_val
            ]

            self.replay_buffer_val = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )
            self.replay_buffer_val.load_multiple_buffer(buffer_dirs_val)
            print(f"Loaded validation buffer with {len(self.replay_buffer_val)} samples.")

            # --- END: Evaluate pa mode initialization logic ---
        elif self.config.experiment.mode == 'evaluate_collective_transformer':
            # collective network
            if self.config.experiment.evaluate_transformer == "collective_network":
                self.col_agent = hydra.utils.instantiate(
                    self.config.transformer_collective_network.builder,
                    env_obs_shape=env_obs_shape,
                    action_shape=action_shape,
                    action_range=[
                        float(self.action_space.low.min()),
                        float(self.action_space.high.max()),
                    ],
                    experiment=self.config.experiment.experiment,
                    seed=self.config.setup.seed,
                    device=self.device,
                )

                self.col_model_dir = utils.make_dir(
                    os.path.join(self.config.setup.save_dir, "model_dir", self.config.experiment.experiment, f"model_col_seed_{self.config.setup.seed}")
                )

                self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)
            else:
                self.col_agent = hydra.utils.instantiate( 
                    self.config.worker.builder,
                    env_obs_shape=env_obs_shape,
                    action_shape=action_shape,
                    action_range=[
                        float(self.action_space.low.min()),
                        float(self.action_space.high.max()),
                    ],
                    device=self.device,
                )

                self.col_model_dir = utils.make_dir(
                    os.path.join(self.config.setup.save_dir, "evaluation_models")
                )

                self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)

                self.agent = [self.col_agent]

            self.embedding_save_path = utils.make_dir(self.config.transformer_collective_network.embedding_save_path)
            self.embedding_save_path = self.config.transformer_collective_network.embedding_save_path + 'emb.pth'

            self.col_buffer_loc = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "buffer/collective_buffer/validation")
            )
            self.distill_buffer_names = os.listdir(self.col_buffer_loc)
            self.distill_names = [name.replace("buffer_distill_", "") for name in self.distill_buffer_names]
            pattern = r'(-\d+)?_seed_\d+$'
            self.distill_names = [re.sub(pattern, '', name) for name in self.distill_names]
            self.num_datasets = len(self.distill_names)

            self.buffer_dir_distill = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/validation/{i}")
            ) for i in self.distill_buffer_names]

            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len,
            ) for _ in range(self.num_datasets)]
            
            for i in range(self.num_datasets):
                self.replay_buffer_distill[i].load(save_dir=self.buffer_dir_distill[i], seq_len=self.config.transformer_collective_network.transformer_encoder.representation_transformer.sequence_len)

        elif self.config.experiment.mode == 'online_distill_collective_transformer':
            # collective network
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )

            self.expert_model_dir = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'model'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir = utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    self.task_names[0], 
                    self.config.setup.seed, 
                    'online_buffer'
                )
            )

            # Expert
            self.expert = [hydra.utils.instantiate( 
                self.config.worker.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            ) for _ in self.env_indices_i]
            
            self.replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                normalize_rewards=False,
                #batch_size=self.config.replay_buffer['batch_size'],
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )

            self.start_step = 0
            self.replay_buffer.load(save_dir=self.buffer_dir)
            for i in self.env_indices_i:
                self.expert[i].load_latest_step(model_dir=self.expert_model_dir[i])
        elif self.config.experiment.mode == 'train_predictive_adapter':
            # === Only train the TransformerAgent's predictive adapter ===
            env_obs_shape_pa = [21]
            # 1) Instantiate TransformerAgent (uses the same builder as distill_collective_transformer)
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape_pa,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                experiment=self.config.experiment.experiment,
                seed=self.config.setup.seed,
                device=self.device,
            )

            # 2) Model save directory (dedicated to predictive adapter)
            self.col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir", self.config.experiment.experiment, f"model_predictive_adapter_seed_{self.config.setup.seed}")
            )

            # 3) Training data: aggregate train split collective buffers (same approach as distill_collective_transformer)
            self.col_buffer_loc = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "buffer/collective_buffer/train")
            )
            self.distill_buffer_names = os.listdir(self.col_buffer_loc)
            print(self.distill_buffer_names)
            if len(self.distill_buffer_names) == 0:
                raise RuntimeError(f"Training data directory not found: {self.col_buffer_loc}")

            buffer_dirs = [
                utils.make_dir(os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/train/{i}"))
                for i in self.distill_buffer_names
            ]

            # 4) Aggregate into a single TransformerReplayBuffer (the one used for col in this project)
            self.replay_buffer_distill = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )
            self.replay_buffer_distill.load_multiple_buffer(buffer_dirs)
            self.pa_task_pair_specs = self._build_pa_task_pair_specs(
                self.replay_buffer_distill, buffer_dirs
            )

            # 5) (Optional) Prepare a validation buffer for intermediate evaluation
            self.buffer_dir_val = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/validation/{i}")
            ) for i in self.distill_buffer_names]

            self.replay_buffer_val = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )
            self.replay_buffer_val.load_multiple_buffer(self.buffer_dir_val)
            self.pa_val_task_pair_specs = self._build_pa_task_pair_specs(
                self.replay_buffer_val, self.buffer_dir_val
            )

            # 6) Starting step (optional resume)
            self.pa_start_step = 0
            if should_resume_experiment:
                # Directly reuse the agent's load_latest_step
                self.pa_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)

        elif self.config.experiment.mode == 'generate_distill_data':
            # 1. Create save path (Buffer Distill)
            self.buffer_dir_distill = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer_distill'
                )
            ) for task_name in self.task_names]

            # 2. Instantiate Distill Replay Buffer
            # Note: here we use col_replay_buffer (typically a simple array buffer), not the transformer buffer
            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            # 3. Set starting step (from 0)
            self.start_step = 0

        else:
            raise NotImplementedError(
                f"experiment-mode {self.config.experiment.mode} is not supported"
            )

        self.logger = Logger(
            self.config.setup.save_dir,
            config=self.config,
            retain_logs=should_resume_experiment,
        )
        self.max_episode_steps = self.env_metadata[
            "max_episode_steps"
        ]  # maximum steps that the agent can take in one environment.

        self.video_dir = utils.make_dir(
            os.path.join(self.config.setup.save_dir, f"video")
        )
        self.video = video.VideoRecorder(
            self.video_dir if self.config.experiment.save_video else None
        ) 

        self.startup_logs()

    def import_policy_class(self, policy_name):
            if policy_name == "peg-insert-side":
                module_name, class_name = "peg_insertion_side", "PegInsertionSide"
            else:
                module_name, class_name = policy_name.replace("-", "_"), "".join([name.capitalize() for name in policy_name.split("-")])
            module_name = f"metaworld.policies.sawyer_{module_name}_v2_policy"
            policy_class = getattr(importlib.import_module(module_name), f"Sawyer{class_name}V2Policy") 
            return policy_class
    
    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Subclasses should implement this method to build the environments.

        Raises:
            NotImplementedError: this method should be implemented by the subclasses.

        Returns:
            Tuple[EnvsDictType, EnvMetaDataType]: Tuple of environment dictionary
            and environment metadata.
        """
        raise NotImplementedError(
            "`build_envs` is not defined for experiment.Experiment"
        )

    def startup_logs(self) -> None:
        """Write some logs at the start of the experiment."""
        config_file = f"{self.config.setup.save_dir}/config.json"
        with open(config_file, "w") as f:
            f.write(json.dumps(config_utils.to_dict(self.config)))

    def _parse_collective_buffer_name(self, buffer_name: str) -> Optional[dict]:
        prefix = "online_buffer_"
        if not buffer_name.startswith(prefix):
            return None
        core = buffer_name[len(prefix):]
        seed_marker = core.rfind("_seed_")
        if seed_marker == -1:
            return None
        core = core[:seed_marker]
        for task_name in sorted(self.task_names, key=len, reverse=True):
            marker = f"_{task_name}"
            if core.endswith(marker):
                robot_name = core[:-len(marker)]
                return {"robot": robot_name, "task": task_name}
        return None

    def _build_pa_task_pair_specs(self, replay_buffer, buffer_dirs: List[str]) -> dict:
        task_groups = {}
        source_ranges = getattr(replay_buffer, "source_ranges", {})
        for buffer_dir in buffer_dirs:
            meta = self._parse_collective_buffer_name(os.path.basename(buffer_dir))
            range_info = source_ranges.get(str(buffer_dir))
            if meta is None or range_info is None:
                continue
            spec = {
                "robot": meta["robot"],
                "task": meta["task"],
                "start": int(range_info["start"]),
                "end": int(range_info["end"]),
                "size": int(range_info["size"]),
            }
            if spec["end"] <= spec["start"]:
                continue
            task_groups.setdefault(meta["task"], []).append(spec)
        return {
            task: specs for task, specs in task_groups.items() if len(specs) >= 2
        }

    def sample_pa_delta_pair_batches(self, replay_buffer=None, pair_specs=None):
        if replay_buffer is None:
            replay_buffer = getattr(self, "replay_buffer_distill", None)
        if pair_specs is None:
            pair_specs = getattr(self, "pa_task_pair_specs", {})
        if replay_buffer is None or not pair_specs:
            if replay_buffer is not None and not pair_specs:
                if not getattr(self, "_warned_empty_pa_pair_specs", False):
                    print(
                        "[PA-DELTA/WARN] No task pair specs were built for delta-state loss. "
                        "Delta loss will be skipped for this run."
                    )
                    self._warned_empty_pa_pair_specs = True
            return []
        pa_cfg = self.config.transformer_collective_network.predictive_adapter
        loss_weight = float(pa_cfg.get("delta_state_loss_weight", 0.0))
        if loss_weight <= 0.0:
            return []

        num_task_pairs = int(pa_cfg.get("delta_state_num_task_pairs", 4))
        pair_batch_size = int(
            pa_cfg.get(
                "delta_state_pair_batch_size",
                self.config.replay_buffer.transformer_col_replay_buffer.batch_size // 2,
            )
        )
        eligible_tasks = list(pair_specs.keys())
        if not eligible_tasks:
            return []

        chosen_tasks = random.sample(
            eligible_tasks, k=min(num_task_pairs, len(eligible_tasks))
        )
        pair_batches = []
        for task_name in chosen_tasks:
            spec_a, spec_b = random.sample(pair_specs[task_name], 2)
            idxs_a = np.random.randint(spec_a["start"], spec_a["end"], size=pair_batch_size)
            idxs_b = np.random.randint(spec_b["start"], spec_b["end"], size=pair_batch_size)

            state_a, action_a, _, next_state_a, _, _, valid_mask_a, _ = \
                self.col_agent._prepare_predictive_adapter_batch_from_buffer(
                    replay_buffer, idxs=idxs_a
                )
            state_b, action_b, _, next_state_b, _, _, valid_mask_b, _ = \
                self.col_agent._prepare_predictive_adapter_batch_from_buffer(
                    replay_buffer, idxs=idxs_b
                )
            pair_batches.append(
                {
                    "task": task_name,
                    "robot_a": spec_a["robot"],
                    "robot_b": spec_b["robot"],
                    "state_a": state_a,
                    "action_a": action_a,
                    "next_state_a": next_state_a,
                    "valid_mask_a": valid_mask_a,
                    "state_b": state_b,
                    "action_b": action_b,
                    "next_state_b": next_state_b,
                    "valid_mask_b": valid_mask_b,
                }
            )
        return pair_batches

    def periodic_save(self, epoch: int) -> None:
        """Perioridically save the experiment.

        This is a utility method, built on top of the `save` method.
        It performs an extra check of wether the experiment is configured to
        be saved during the current epoch.
        Args:
            epoch (int): current epoch.
        """
        persist_frequency = self.config.experiment.persist_frequency
        if persist_frequency > 0 and epoch % persist_frequency == 0:
            self.save(epoch)

    def save(self, epoch: int) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def load(self, epoch: Optional[int]) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def run(self) -> None:
        """Run the experiment.

        Raises:
            NotImplementedError: This method should be implemented by the subclasses.
        """
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def close_envs(self):
        """Close all the environments."""
        for env in self.envs.values():
            env.close()
    
    def reset_goal_locations(self, env="train"):
        #new_tasks = [random.choice(self.env_name_task_dict[env]) for env in self.env_metadata["ordered_task_list"]]
        new_tasks = {key: random.choice(value) for key, value in self.env_name_task_dict.items()}
        self.envs[env].call("set_task_with_dict", new_tasks)

    def reset_goal_locations_in_order(self, env="train"):
        #new_tasks = [random.choice(self.env_name_task_dict[env]) for env in self.env_metadata["ordered_task_list"]]
        new_tasks = {key: value[self.env_counter] for key, value in self.env_name_task_dict.items()}
        self.env_counter = (self.env_counter+1) % 50
        self.envs[env].call("set_task_with_dict", new_tasks)

    def reset_goal_locations_for_listenv(self, index, env_name):
        new_tasks = {env_name: random.choice(self.env_name_task_dict[env_name])}
        self.list_envs[index].env.set_task_with_dict(new_tasks)

    def detect_robot_type(self) -> str:
        """
        Detect robot type from environment configuration or asset files.
        
        Returns:
            Robot type string ('sawyer', 'ur5e', 'ur10e', etc.)
        """
        # Method 1: Check environment assets
        if hasattr(self.config, 'experiment'):
            # Look for robot-specific indicators in experiment config
            robot_type = getattr(self.config.experiment, 'robot_type', '')

            # Check if specific robot is mentioned in config
            robot_indicators = {
                'ur5e': ['ur5e', 'ur_5e'],
                'ur10e': ['ur10e', 'ur_10e'], 
                'kuka': ['kuka'],
                'sawyer': ['sawyer', 'default'],  # sawyer as default
                'panda': ['panda'],
                'unitree_z1': ['unitree', 'z1'],
                'viperx': ['viperx'],
                'gen3': ['gen3'],
                'xarm7': ['xarm7']
            }
            
            for robot, indicators in robot_indicators.items():
                if any(indicator in robot_type.lower() for indicator in indicators):
                    return robot

        # Default to unknown if cannot detect
        return 'unknown'
    def get_env_metadata(
        env: EnvType,
        max_episode_steps: Optional[int] = None,
        ordered_task_list: Optional[List[str]] = None,
    ) -> EnvMetaDataType:
        """Method to get the metadata from an environment"""
        dummy_env = env.env_fns[0]().env
        metadata: EnvMetaDataType = {
            "env_obs_space": dummy_env.observation_space,
            "action_space": dummy_env.action_space,
            "ordered_task_list": ordered_task_list,
        }
        if max_episode_steps is None:
            metadata["max_episode_steps"] = dummy_env._max_episode_steps
        else:
            metadata["max_episode_steps"] = max_episode_steps
        return metadata


def prepare_config(config: ConfigType, env_metadata: EnvMetaDataType) -> ConfigType:
    """Infer some config attributes during runtime.

    Args:
        config (ConfigType): config to update.
        env_metadata (EnvMetaDataType): metadata of the environment.

    Returns:
        ConfigType: updated config.
    """
    config = config_utils.make_config_mutable(config_utils.unset_struct(config))
    key = "type_to_select"
    if key in config.worker.encoder:
        encoder_type_to_select = config.worker.encoder[key]
    else:
        encoder_type_to_select = config.worker.encoder.type
    if encoder_type_to_select in ["identity"]:
        # if the encoder is an identity encoder infer the shape of the input dim.
        config.worker.encoder_feature_dim = env_metadata["env_obs_space"].shape[0]

    key = "ordered_task_list"
    if key in env_metadata and env_metadata[key]:
        config.env.ordered_task_list = deepcopy(env_metadata[key])
    config = config_utils.make_config_immutable(config)

    return config


def get_env_metadata(
    env: EnvType,
    max_episode_steps: Optional[int] = None,
    ordered_task_list: Optional[List[str]] = None,
) -> EnvMetaDataType:
    """Method to get the metadata from an environment"""
    dummy_env = env.env_fns[0]().env
    metadata: EnvMetaDataType = {
        "env_obs_space": dummy_env.observation_space,
        "action_space": dummy_env.action_space,
        "ordered_task_list": ordered_task_list,
    }
    if max_episode_steps is None:
        metadata["max_episode_steps"] = dummy_env._max_episode_steps
    else:
        metadata["max_episode_steps"] = max_episode_steps
    return metadata
 
def create_robot_aware_path(base_path: str, robot_type: str, task_name: str, seed: int, path_type: str) -> str:
    """
    Create robot-aware paths for all different path types.
    
    Args:
        base_path: Base directory path
        robot_type: Detected robot type ('sawyer', 'ur5e', 'ur10e')
        task_name: Task name
        seed: Random seed
        path_type: Type of path ('model', 'buffer', 'buffer_distill', 'buffer_distill_tmp', etc.)
    
    Returns:
        Robot-aware path string
    """
    if robot_type == 'unknown':
        # Fallback to original naming if robot detection fails
        if path_type == 'model':
            return os.path.join(base_path, f"model_dir/model_{task_name}_seed_{seed}")
        elif path_type == 'buffer':
            return os.path.join(base_path, f"buffer/buffer/buffer_{task_name}_seed_{seed}")
        elif path_type == 'buffer_distill':
            return os.path.join(base_path, f"buffer/buffer_distill/buffer_distill_{task_name}_seed_{seed}")
        elif path_type == 'buffer_distill_tmp':
            return os.path.join(base_path, f"buffer/buffer_distill_tmp/buffer_distill_tmp_{task_name}_seed_{seed}")
        # Add other path types as needed
    
    # Robot-aware naming
    if path_type == 'model':
        return os.path.join(base_path, f"model_dir/model_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'buffer':
        return os.path.join(base_path, f"buffer/buffer/buffer_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'buffer_distill':
        return os.path.join(base_path, f"buffer/buffer_distill/buffer_distill_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'buffer_distill_tmp':
        return os.path.join(base_path, f"buffer/buffer_distill_tmp/buffer_distill_tmp_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'recording_buffer_distill':
        return os.path.join(base_path, f"buffer/buffer_distill/recording_buffer_distill_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'online_buffer':
        return os.path.join(base_path, f"buffer/online_buffer_{robot_type}_{task_name}")
    else:
        raise ValueError(f"Unknown path_type: {path_type}")

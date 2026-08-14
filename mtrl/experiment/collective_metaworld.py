# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import Dict, Tuple

import hydra
import numpy as np
import torch
import random

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import collective_learning
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType

from collections import defaultdict

class Experiment(collective_learning.Experiment):
    """Experiment Class"""

    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        super().__init__(config, experiment_id)
        self.should_reset_env_manually = True

    def create_eval_modes_to_env_ids(self):
        eval_modes_to_env_ids = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if self.config.env.benchmark._target_ in [
                "metaworld.ML1",
                "metaworld.MT1",
                "metaworld.MT10",
                "metaworld.MT50",
                "metaworld.COL_MT5"
            ]:
                eval_modes_to_env_ids[mode] = list(range(self.config.env.num_envs))
            else:
                raise ValueError(
                    f"`{self.config.env.benchmark._target_}` env is not supported by metaworld experiment."
                )
        return eval_modes_to_env_ids

    def build_envs(self):
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        
        self.env_name_task_dict = defaultdict(list)
        train_tasks = benchmark._train_tasks

        for task in train_tasks:
            self.env_name_task_dict[task.env_name].append(task)

        if "COL" in self.config.env.benchmark._target_:
            build_vec_func = env_builder.build_metaworld_vec_env_col
            build_vec_func_list = env_builder.build_metaworld_env_list_col
        else:
            build_vec_func = env_builder.build_metaworld_vec_env
            build_vec_func_list = env_builder.build_metaworld_env_list

        envs = {}
        mode = "train"
        envs[mode], env_id_to_task_map = build_vec_func(
            config=self.config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )
        self.env_id_to_task_map = env_id_to_task_map 
        mode = "eval"
        envs[mode], env_id_to_task_map = build_vec_func(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=env_id_to_task_map,
        )
        # In MT10 and MT50, the tasks are always sampled in the train mode.
        # For more details, refer https://github.com/rlworkgroup/metaworld

        max_episode_steps = 400
        # hardcoding the steps as different environments return different
        # values for max_path_length. MetaWorld uses 150 as the max length.
        metadata = self.get_env_metadata(
            env=envs["train"],
            max_episode_steps=max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
        )

        ### andi
        self.list_envs, self.env_id_to_task_map_recording = build_vec_func_list(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=self.env_id_to_task_map ,
        )
        ### andi

        return envs, metadata
    
    def create_env_id_to_index_map(self) -> Dict[str, int]:
        env_id_to_index_map: Dict[str, int] = {}
        current_id = 0
        for env in self.envs.values():
            assert isinstance(env, VecEnv)
            for env_name in env.ids:
                if env_name not in env_id_to_index_map:
                    env_id_to_index_map[env_name] = current_id
                    current_id += 1
        return env_id_to_index_map

    def _get_eval_step_limit(self) -> int:
        limit = int(getattr(self.config.experiment, "eval_episode_step_limit", 0) or 0)
        if limit <= 0:
            return self.max_episode_steps
        return min(limit, self.max_episode_steps)

    def _cap_eval_history(
        self,
        state_buffer: torch.Tensor,
        action_buffer: torch.Tensor,
        reward_buffer: torch.Tensor,
        episode_step: int,
    ):
        history_cap = int(getattr(self.config.experiment, "eval_history_steps", 0) or 0)
        if history_cap <= 0:
            return state_buffer, action_buffer, reward_buffer

        state_cap = max(min(history_cap, state_buffer.shape[1]), 1)
        ar_cap = max(min(history_cap - 1, action_buffer.shape[1]), 0)

        capped_states = state_buffer.clone()
        capped_actions = action_buffer.clone()
        capped_rewards = reward_buffer.clone()

        if state_cap < state_buffer.shape[1]:
            pad_state = state_buffer[:, -1:, :].expand(-1, state_buffer.shape[1] - state_cap, -1)
            capped_states[:, : state_buffer.shape[1] - state_cap, :] = pad_state

        if ar_cap < action_buffer.shape[1]:
            if ar_cap == 0:
                capped_actions.zero_()
                capped_rewards.zero_()
            else:
                capped_actions[:, : action_buffer.shape[1] - ar_cap, :] = 0.0
                capped_rewards[:, : reward_buffer.shape[1] - ar_cap, :] = 0.0

        return capped_states, capped_actions, capped_rewards

    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int, agent: str = "worker"):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        import time  # ensure time is imported

        episode_step = 0
        self.logger.log(f"eval/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        step_limit = self._get_eval_step_limit()
        while episode_step < step_limit:
            # --- Timing checkpoint 1: Policy/Inference phase (CPU Prep + GPU Inference) ---
            t0 = time.time()

            action = np.full(shape=(vec_env.num_envs, np.prod(self.action_space.shape)), fill_value=0.)
            if agent=="worker":
                for i in self.env_indices_i:
                    env_obs_i = {
                        key: value[self.env_indices[i]] for key, value in multitask_obs.items()
                    }
                    with agent_utils.eval_mode(self.agent[i]):
                        action[self.env_indices[i]] = self.agent[i].select_action(
                            multitask_obs=env_obs_i, modes=["eval"]
                        )
            elif agent=="scripted":
                for i in self.env_indices_i:
                    env_obs_i = {
                        key: value[self.env_indices[i]] for key, value in multitask_obs.items()
                    }
                    action[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)

            multitask_obs, reward, done, info = vec_env.step(action)

            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        
        success = (success > 0).astype("float")

        success = success[self.env_indices]
        # if success==0:
            # self.record_videos(eval_agent="scripted", tag=f"sample_{i}")
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(
            f"eval/episode_reward",
            episode_reward.mean(),
            step,
        )
        self.logger.log(
            f"eval/success",
            success.mean(),
            step,
        )

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(
                f"eval/episode_reward_env_index_{_current_env_id}",
                episode_reward[_current_env_index].mean(),
                step,
            )
            self.logger.log(
                f"eval/success_env_index_{_current_env_id}",
                success[_current_env_index].mean(),
                step,
            )

        self.logger.dump(step)
        return success.sum()

    def evaluate_transformer(self, agent, vec_env: VecEnv, step: int, episode: int, seq_len: int, sample_actions: bool, reward_unscaled: bool):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.
        """
        
        # Get device (assuming agent is already on the correct device)
        device = agent.device if hasattr(agent, "device") else torch.device("cpu")

        episode_step = 0
        self.logger.log(f"col_eval/episode", episode, step)

        # Initialize statistics variables (keeping original Numpy format)
        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ] 
        
        # === 1. Initialization & Buffer pre-allocation (replaces original left_pad_batch logic) ===
        multitask_obs = vec_env.reset()
        
        # Process initial state (compatible with both Tensor and Numpy inputs)
        obs_data = multitask_obs['env_obs']
        if isinstance(obs_data, torch.Tensor):
            obs_tensor = obs_data
        else:
            obs_tensor = torch.from_numpy(obs_data)
        
        # Extract p1, p2 and concatenate (corresponds to original: multitask_obs['env_obs'][:,:18] + [:,36:])
        first_state = torch.cat((obs_tensor[:, :18], obs_tensor[:, 36:]), dim=1).float().to(device)
        
        # Automatically get dimensions
        B = vec_env.num_envs
        obs_dim = first_state.shape[-1]
        action_dim = np.prod(self.action_space.shape)

        # [State Buffer]: [Num_Envs, T, Obs_Dim]
        # Logic: fill the entire sequence with the first frame, equivalent to left_pad_batch at t=0
        state_buffer = first_state.unsqueeze(1).repeat(1, seq_len, 1)

        # [Action/Reward Buffer]: fill with 0
        action_buffer = torch.zeros((B, seq_len, action_dim), device=device)
        reward_buffer = torch.zeros((B, seq_len, 1), device=device)

        # Cache Task IDs
        task_ids_tensor = torch.tensor(self.task_num[self.env_indices_i], device=device)

        step_limit = self._get_eval_step_limit()
        while episode_step < step_limit:
            # Initialize current step's Action Numpy array (keeping original logic)
            action_np = np.full(shape=(B, action_dim), fill_value=0.)
            
            # === 2. Inference (using Buffer slices) ===
            with agent_utils.eval_mode(agent):
                capped_states, capped_actions, capped_rewards = self._cap_eval_history(
                    state_buffer,
                    action_buffer,
                    reward_buffer,
                    episode_step=episode_step,
                )
                # Slicing: State takes full length, Action/Reward takes last T-1 entries (corresponds to original T-1)
                active_states = capped_states[self.env_indices]
                active_actions = capped_actions[self.env_indices, 1:, :]
                active_rewards = capped_rewards[self.env_indices, 1:, :]

                if sample_actions:
                    action_out = agent.sample_action(
                        states=active_states,
                        actions=active_actions, 
                        rewards=active_rewards,
                        task_ids=task_ids_tensor
                    )
                else:
                    action_out = agent.select_action(
                        states=active_states,
                        actions=active_actions,
                        rewards=active_rewards,
                        task_ids=task_ids_tensor
                    )
            
            # === 3. Data type conversion (Tensor -> Numpy) ===
            # Ensure env.step receives numpy, and handle partially active envs
            if isinstance(action_out, torch.Tensor):
                action_subset_np = action_out.detach().cpu().numpy()
            else:
                action_subset_np = action_out
            
            action_np[self.env_indices] = action_subset_np

            # === 4. Environment interaction ===
            multitask_obs, reward, done, info = vec_env.step(action_np)
            if reward_unscaled:
                reward = info['unscaled_reward']

            # === 5. Rolling Buffer update (replaces original torch.cat) ===
            # A. Shift left overall (discard the oldest frame)
            state_buffer = torch.roll(state_buffer, shifts=-1, dims=1)
            action_buffer = torch.roll(action_buffer, shifts=-1, dims=1)
            reward_buffer = torch.roll(reward_buffer, shifts=-1, dims=1)

            # B. Prepare new data
            # Process new state
            obs_data_next = multitask_obs['env_obs']
            if isinstance(obs_data_next, torch.Tensor):
                obs_next_tensor = obs_data_next
            else:
                obs_next_tensor = torch.from_numpy(obs_data_next)
            
            new_state = torch.cat((obs_next_tensor[:, :18], obs_next_tensor[:, 36:]), dim=1).float().to(device)
            
            # Process new reward
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward
            else:
                reward_tensor = torch.from_numpy(reward)
            new_reward = reward_tensor.float().to(device).unsqueeze(-1) # [B, 1]

            # Process action (needs to be stored in Buffer, so convert back to Tensor)
            new_action = torch.from_numpy(action_np).float().to(device) # [B, A]

            # C. Fill into Buffer tail
            state_buffer[:, -1, :] = new_state
            action_buffer[:, -1, :] = new_action
            reward_buffer[:, -1, :] = new_reward

            # === 6. Statistics (keeping original logic) ===
            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        
        success = (success > 0).astype("float")
        success = success[self.env_indices]
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(f"col_eval/episode_reward", episode_reward.mean(), step)
        self.logger.log(f"col_eval/success", success.mean(), step)

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(f"col_eval/episode_reward_env_index_{_current_env_id}", episode_reward[_current_env_index].mean(), step)
            self.logger.log(f"col_eval/success_env_index_{_current_env_id}", success[_current_env_index].mean(), step)

        self.logger.dump(step)
        # Keep return value consistent with the original source code
        return success

    def evaluate_decision_tf(self, vec_env: VecEnv, step: int, episode: int, seq_len: int, reward_unscaled: bool):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        episode_step = 0
        self.logger.log(f"col_eval/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        state_src = multitask_obs['env_obs'][:,None].float()
        action_src = torch.zeros((vec_env.num_envs, 1, self.action_shape))
        rewards_src = torch.zeros((vec_env.num_envs, 1, 1))

        #TODO: improve evaluate_unused
        action = np.full(shape=(vec_env.num_envs, np.prod(self.action_space.shape)), fill_value=0.)
        step_limit = self._get_eval_step_limit()
        while episode_step < step_limit:

            with agent_utils.eval_mode(self.col_agent):
                action[self.env_indices] = self.col_agent.select_action(
                    state_src, action_src, rewards_src
                )
            
            multitask_obs, reward, done, info = vec_env.step(action)
            if reward_unscaled:
                reward = info['unscaled_reward']

            state_src = torch.cat((state_src[:, -seq_len+1:], multitask_obs['env_obs'][:,None].float()), dim=1)
            action_src = torch.cat((action_src[:, -seq_len+1:], torch.from_numpy(action)[:,None].float()), dim=1)
            rewards_src = torch.cat((rewards_src[:, -seq_len+1:], torch.tensor(reward)[:,None,None].float()), dim=1)

            success += info['success']
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        success = (success > 0).astype("float")

        success = success[self.env_indices]
        episode_reward = episode_reward[self.env_indices]
        
        self.logger.log(
            f"col_eval/episode_reward",
            episode_reward.mean(),
            step,
        )
        self.logger.log(
            f"col_eval/success",
            success.mean(),
            step,
        )

        for _current_env_index, _current_env_id in enumerate(self.env_indices):
            self.logger.log(
                f"col_eval/episode_reward_env_index_{_current_env_id}",
                episode_reward[_current_env_index].mean(),
                step,
            )
            self.logger.log(
                f"col_eval/success_env_index_{_current_env_id}",
                success[_current_env_index].mean(),
                step,
            )

        self.logger.dump(step)

    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        env_indices = multitask_obs["task_obs"]
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        for _ in range(num_steps):
            with agent_utils.eval_mode(self.agent):
                action = self.agent.sample_action(
                    multitask_obs=multitask_obs, mode="train"
                )  # (num_envs, action_dim)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()
            episode_reward += reward

            # allow infinite bootstrap
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                self.replay_buffer.add(
                    multitask_obs["env_obs"][index],
                    action[index],
                    reward[index],
                    next_multitask_obs["env_obs"][index],
                    done_bool,
                    env_index=env_index,
                )

            multitask_obs = next_multitask_obs
            episode_step += 1

    def build_video_envs(self):
        
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        #benchmark = hydra.utils.instantiate(self.config.env.kuka_benchmark)

        list_envs, env_id_to_task_map_recording = env_builder.build_metaworld_env_list(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=self.env_id_to_task_map ,
        )
        self.list_envs = list_envs
        self.env_id_to_task_map_recording = env_id_to_task_map_recording
        
        return list_envs
    

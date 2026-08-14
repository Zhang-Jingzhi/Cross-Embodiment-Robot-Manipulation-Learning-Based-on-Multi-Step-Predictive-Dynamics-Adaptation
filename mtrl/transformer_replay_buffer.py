# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mtrl.utils.types import TensorType


@dataclass
class TransformerReplayBufferSample:
    __slots__ = [
        "env_obs",
        "next_env_obs",
        "action",
        "reward",
        "not_done",
        "task_obs",
        "buffer_index",
        "task_encoding",
        "policy_mu",
        "policy_log_std",
        "q_target",
    ]
    env_obs: TensorType
    next_env_obs: TensorType
    action: TensorType
    reward: TensorType
    not_done: TensorType
    task_obs: TensorType
    buffer_index: TensorType
    task_encoding: TensorType
    policy_mu: TensorType
    policy_log_std: TensorType
    q_target: TensorType


class TransformerReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(
        self, env_obs_shape, task_obs_shape, action_shape, capacity, batch_size, device, normalize_rewards, seq_len, task_encoding_shape, compressed_state
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.task_obs_shape = task_obs_shape
        self.normalize_rewards = normalize_rewards
        self.task_encoding_shape = task_encoding_shape
        self.compressed_state = compressed_state

        # the proprioceptive env_obs is stored as float32, pixels env_obs as uint8
        task_obs_dtype = np.int64
        
        assert self.capacity % 400 == 0

        if self.compressed_state:
            self.env_obses = np.empty((capacity//400, 400, 21), dtype=np.float32)
        else:
            self.env_obses = np.empty((capacity//400, 400, 39), dtype=np.float32)
        self.next_env_obses = np.empty((capacity//400, 400, 39), dtype=np.float32)
        self.actions = np.empty((capacity//400, 400, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity//400, 400, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity//400, 400, 1), dtype=np.float32)
        self.task_obs = np.empty((capacity//400, 400, *task_obs_shape), dtype=task_obs_dtype)
        self.task_encodings = np.empty((capacity//400, 400, task_encoding_shape), dtype=np.float32)
        self.policy_mu = np.empty((capacity//400, 400, *action_shape), dtype=np.float32)
        self.policy_log_std = np.empty((capacity//400, 400, *action_shape), dtype=np.float32)
        self.q_target = np.empty((capacity//400, 400, 1), dtype=np.float32)

        self.idx = 0
        self.idx_sample = 0
        self.last_save = 0
        self.full = False
        self.seq_len = seq_len

        self.max_reward = 0
        self.min_reward = 0

    def is_empty(self):
        return self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, env_obs, next_env_obs, action, reward, done, task_obs, encoding, q_value, mu, log_std):
        if env_obs.shape[0]==39 and self.compressed_state:
            env_obs = np.concatenate((env_obs[:18], env_obs[36:]))
        np.copyto(self.env_obses[self.idx//400, self.idx%400], env_obs)
        np.copyto(self.next_env_obses[self.idx//400, self.idx%400], next_env_obs)
        np.copyto(self.actions[self.idx//400, self.idx%400], action)
        np.copyto(self.rewards[self.idx//400, self.idx%400], reward)
        np.copyto(self.task_obs[self.idx//400, self.idx%400], task_obs)
        np.copyto(self.not_dones[self.idx//400, self.idx%400], not done)
        np.copyto(self.task_encodings[self.idx//400, self.idx%400], encoding)
        np.copyto(self.policy_mu[self.idx//400, self.idx%400], mu)
        np.copyto(self.policy_log_std[self.idx//400, self.idx%400], log_std)
        np.copyto(self.q_target[self.idx//400, self.idx%400], q_value)

        self.idx = (self.idx + 1) % self.capacity
        self.idx_sample = self.idx // 400 * (400 - self.seq_len + 1)
        self.full = self.full or self.idx == 0

    def add_array(self, env_obs, action, reward, next_env_obs, done, task_obs, q_value, mu, log_std, size):
        raise NotImplementedError
    
    def sample_indices(self):
        idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
        return idxs

    def sample(self, index=None) -> TransformerReplayBufferSample: 
        if index is None:
            idxs = self.sample_indices()
        else:
            idxs = index

        env_obs = torch.as_tensor(self.env_obses[idxs//400, idxs%400], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs//400, idxs%400], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs//400, idxs%400], device=self.device).float()
        next_env_obs = torch.as_tensor(self.next_env_obses[idxs//400, idxs%400], device=self.device).float()
        env_indices = torch.as_tensor(self.task_obs[idxs//400, idxs%400], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs//400, idxs%400], device=self.device)
        task_encoding = torch.as_tensor(self.task_encodings[idxs//400, idxs%400], device=self.device).float()
        mus = torch.as_tensor(self.policy_mu[idxs//400, idxs%400], device=self.device).float()
        log_stds = torch.as_tensor(self.policy_log_std[idxs//400, idxs%400], device=self.device).float()
        q_targets = torch.as_tensor(self.q_target[idxs//400, idxs%400], device=self.device).float()

        return TransformerReplayBufferSample(env_obs, next_env_obs, actions, rewards, not_dones, env_indices, idxs, task_encoding, mus, log_stds, q_targets)
    
    def sample_new(self, index=None): 
        if index is None:
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
        else:
            idxs = index

        env_obs = torch.as_tensor(self.env_obses[idxs//400, idxs%400], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs//400, idxs%400], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs//400, idxs%400], device=self.device).float()
        next_env_obs = torch.as_tensor(self.next_env_obses[idxs//400, idxs%400], device=self.device).float()
        env_indices = torch.as_tensor(self.task_obs[idxs//400, idxs%400], device=self.device)
        task_encoding = torch.as_tensor(self.task_encodings[idxs//400, idxs%400], device=self.device).float()
        mus = torch.as_tensor(self.policy_mu[idxs//400, idxs%400], device=self.device).float()
        log_stds = torch.as_tensor(self.policy_log_std[idxs//400, idxs%400], device=self.device).float()
        q_targets = torch.as_tensor(self.q_target[idxs//400, idxs%400], device=self.device).float()

        return env_obs, actions, rewards, next_env_obs, mus, log_stds, q_targets, env_indices, task_encoding

    def sample_multi_step(self, horizon, index=None, device=None):
        """Sample a current state together with the next `horizon` transitions."""
        device = self.device if device is None else device
        if index is None:
            idxs = self.sample_indices()
        else:
            idxs = np.asarray(index)

        horizon = max(int(horizon), 1)
        ep_len = 400

        episode_ids = idxs // ep_len
        timesteps = idxs % ep_len
        future_offsets = np.arange(horizon)[None, :]
        future_timesteps = timesteps[:, None] + future_offsets
        within_episode = future_timesteps < ep_len
        future_timesteps = np.clip(future_timesteps, 0, ep_len - 1)

        current_states = torch.as_tensor(
            self.env_obses[episode_ids, timesteps], device=device
        ).float()
        action_seq = torch.as_tensor(
            self.actions[episode_ids[:, None], future_timesteps], device=device
        ).float()
        reward_seq = torch.as_tensor(
            self.rewards[episode_ids[:, None], future_timesteps], device=device
        ).float()
        next_state_seq = torch.as_tensor(
            self.next_env_obses[episode_ids[:, None], future_timesteps], device=device
        ).float()
        env_indices = torch.as_tensor(
            self.task_obs[episode_ids, timesteps], device=device
        )
        task_encoding = torch.as_tensor(
            self.task_encodings[episode_ids, timesteps], device=device
        ).float()

        not_done_seq = torch.as_tensor(
            self.not_dones[episode_ids[:, None], future_timesteps], device=device
        ).float()
        within_episode = torch.as_tensor(
            within_episode, device=device, dtype=torch.float32
        ).unsqueeze(-1)
        carry_mask = torch.cat(
            [
                torch.ones((len(idxs), 1, 1), device=device, dtype=torch.float32),
                torch.cumprod(not_done_seq[:, :-1], dim=1),
            ],
            dim=1,
        )
        valid_mask = within_episode * carry_mask

        return current_states, action_seq, reward_seq, next_state_seq, valid_mask, env_indices, task_encoding
    
    def build_sequences_for_indices(self, idxs, seq_len, device=None):

        device = self.device if device is None else device
        ep_len = 400  # Originally fixed at 400; infer from data instead
        B      = len(idxs)
        T      = seq_len

        # 1) Build all indices and slices on CPU (avoid moving the entire buffer to GPU)
        #    Note: from_numpy does not copy data, so overhead is small
        env_obs = torch.from_numpy(self.env_obses)   # [E, L, S]
        acts    = torch.from_numpy(self.actions)     # [E, L, A]
        rews    = torch.from_numpy(self.rewards)     # [E, L, 1]
        tasks   = torch.from_numpy(self.task_obs)    # [E, L]

        # 2) Batch-compute (episode, timestep)
        idxs_np = np.asarray(idxs)
        ep = torch.from_numpy(idxs_np // ep_len).long()   # [B]
        t  = torch.from_numpy(idxs_np %  ep_len).long()   # [B]

        # 3) Build time-window indices
        # states window length = T, actions/rewards window length = T-1
        base_T  = torch.arange(T,   dtype=torch.long).view(1, T)       # [1, T]
        base_T1 = torch.arange(T-1, dtype=torch.long).view(1, T-1)     # [1, T-1]

        # Left-padding start offset: t-(T-1); negative values indicate left padding is needed
        start = (t - (T - 1)).view(B, 1)  # [B, 1]

        # Key: clamp(min=0) achieves left-padding by repeating the first frame
        # For states: final index range is [max(0, t-(T-1)) .. t]
        time_idx_T  = (start + base_T).clamp(min=0)     # [B, T]    last element per row is t
        # For actions/rewards: last timestep is t-1
        time_idx_T1 = (start + base_T1).clamp(min=0)    # [B, T-1]  last element per row is t-1

        # Expand episode dimension for element-wise advanced indexing
        ep_T  = ep.view(B, 1).expand(-1, T)     # [B, T]
        ep_T1 = ep.view(B, 1).expand(-1, T-1)   # [B, T-1]

        # 4) Fetch windows via advanced indexing in one shot (still on CPU), then move to target device
        state_seqs  = env_obs[ep_T,  time_idx_T ].to(device=device, dtype=torch.float32)  # [B, T,   S]
        action_seqs = acts[   ep_T1, time_idx_T1].to(device=device, dtype=torch.float32)  # [B, T-1, A]
        reward_seqs = rews[   ep_T1, time_idx_T1].to(device=device, dtype=torch.float32)  # [B, T-1, 1]
        curr_states = env_obs[ep, t].to(device=device, dtype=torch.float32)               # [B, S]
        env_indices = tasks[ep, t].to(device=device).long().unsqueeze(-1)                 # [B, 1]

        return state_seqs, action_seqs, reward_seqs, curr_states, env_indices

    
    def sample_trajectories(self, seq_len, index=None): 
        raise NotImplementedError  
    
    def sample_trajectories2(self):
        trajectory_idxs = np.random.randint(
            0, self.capacity//400 if self.full else self.idx//400, size=self.batch_size
        )
        end_idxs = np.random.randint(0, 400, size=self.batch_size)

        start_idxs = end_idxs[:,None]+np.arange(-self.seq_len+1, 1)
        mask = start_idxs < 0
        start_idxs[mask] = 0 # will be masked

        env_obs = torch.as_tensor(self.env_obses[trajectory_idxs[:, None], start_idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[trajectory_idxs[:, None], start_idxs[:,1:]], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[trajectory_idxs[:, None], start_idxs[:,1:]], device=self.device).float()

        env_obs[mask] = torch.zeros(21, device=self.device)
        actions[mask[:,1:]] = torch.zeros(4, device=self.device)
        rewards[mask[:,1:]] = 0
        
        next_env_obs = torch.as_tensor(self.next_env_obses[trajectory_idxs, end_idxs], device=self.device).float()
        env_indices = torch.as_tensor(self.task_obs[trajectory_idxs, end_idxs], device=self.device)
        task_encoding = torch.as_tensor(self.task_encodings[trajectory_idxs, end_idxs], device=self.device).float()
        mus = torch.as_tensor(self.policy_mu[trajectory_idxs, end_idxs], device=self.device).float()
        log_stds = torch.as_tensor(self.policy_log_std[trajectory_idxs, end_idxs], device=self.device).float()
        q_targets = torch.as_tensor(self.q_target[trajectory_idxs, end_idxs], device=self.device).float()

        mask =  torch.as_tensor(mask).unsqueeze(-1).repeat(1,1,3).view(self.batch_size,-1)[:,:-1].to(self.device)

        return env_obs, actions, rewards, next_env_obs, mus, log_stds, q_targets, env_indices, task_encoding, mask

    def delete_from_filesystem(self, dir_to_delete_from: str):
        for filename in os.listdir(dir_to_delete_from):
            file_path = os.path.join(dir_to_delete_from, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"Deleted files from: {dir_to_delete_from}")

    def save(self, save_dir, size_per_chunk: int, num_samples_to_save: int):
        if self.idx == self.last_save:
            return
        if num_samples_to_save == -1:
            # Save the entire replay buffer
            self._save_all(
                save_dir=save_dir,
                size_per_chunk=size_per_chunk,
            )
        else:        
            if num_samples_to_save > self.idx:
                num_samples_to_save = self.idx
                replay_buffer_to_save = self
            else:
                replay_buffer_to_save = self._sample_a_replay_buffer(
                    num_samples=num_samples_to_save
                )
                replay_buffer_to_save.idx = num_samples_to_save
                replay_buffer_to_save.last_save = 0
            backup_dir_path = Path(f"{save_dir}_bk")
            if not backup_dir_path.exists():
                backup_dir_path.mkdir()
            replay_buffer_to_save._save_all(
                save_dir=str(backup_dir_path),
                size_per_chunk=size_per_chunk,
            )
            replay_buffer_to_save.delete_from_filesystem(dir_to_delete_from=save_dir)
            backup_dir_path.rename(save_dir)
        self.last_save = self.idx

    def _save_all(self, save_dir, size_per_chunk: int):
        if self.idx == self.last_save:
            return
        if self.last_save == self.capacity:
            self.last_save = 0
        if self.idx > self.last_save:
            self._save_payload(
                save_dir=save_dir,
                start_idx=self.last_save,
                end_idx=self.idx,
                size_per_chunk=size_per_chunk,
            )
        else:
            self._save_payload(
                save_dir=save_dir,
                start_idx=self.last_save,
                end_idx=self.capacity,
                size_per_chunk=size_per_chunk,
            )
            self._save_payload(
                save_dir=save_dir,
                start_idx=0,
                end_idx=self.idx,
                size_per_chunk=size_per_chunk,
            )
        self.last_save = self.idx

    def _save_payload(
        self, save_dir: str, start_idx: int, end_idx: int, size_per_chunk: int
    ):
        while True:
            if size_per_chunk > 0:
                current_end_idx = min(start_idx + size_per_chunk, end_idx)
            else:
                current_end_idx = end_idx
            self._save_payload_chunk(
                save_dir=save_dir, start_idx=start_idx, end_idx=current_end_idx
            )
            if current_end_idx == end_idx:
                break
            start_idx = current_end_idx

    def _save_payload_chunk(self, save_dir: str, start_idx: int, end_idx: int):
        path = os.path.join(save_dir, f"{start_idx}_{end_idx-1}.pt")
        payload = [
            self.env_obses.reshape(-1,21)[start_idx:end_idx] if self.compressed_state else self.env_obses.reshape(-1,39)[start_idx:end_idx],
            self.next_env_obses.reshape(-1,39)[start_idx:end_idx],
            self.actions.reshape(-1,4)[start_idx:end_idx],
            self.rewards.reshape(-1,1)[start_idx:end_idx],
            self.not_dones.reshape(-1,1)[start_idx:end_idx],
            self.task_encodings.reshape(-1,self.task_encoding_shape)[start_idx:end_idx],
            self.task_obs.reshape(-1,1)[start_idx:end_idx],
            self.policy_mu.reshape(-1,4)[start_idx:end_idx],
            self.policy_log_std.reshape(-1,4)[start_idx:end_idx],
            self.q_target.reshape(-1,1)[start_idx:end_idx],
        ]
        print(f"Saving transformer replay buffer at {path}")
        torch.save(payload, path)

    def min_max_normalize(self, tensor, min_value, max_value):
        return (tensor - min_value) / (max_value - min_value)

    def load(self, save_dir, seq_len=None):
        if seq_len==None:
            seq_len=self.seq_len
        chunks = os.listdir(save_dir)
        chunks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        start = 0

        if self.compressed_state:
            env_obses = np.empty((self.capacity, 21), dtype=np.float32)
        else:
            env_obses = np.empty((self.capacity, 39), dtype=np.float32)
        next_env_obses = np.empty((self.capacity, 39), dtype=np.float32)
        actions = np.empty((self.capacity, 4), dtype=np.float32)
        rewards = np.empty((self.capacity, 1), dtype=np.float32)
        task_obs = np.empty((self.capacity, *self.task_obs_shape), dtype=np.int64)
        not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        task_encodings = np.empty((self.capacity, self.task_encoding_shape), dtype=np.float32)
        policy_mu = np.empty((self.capacity, 4), dtype=np.float32)
        policy_log_std = np.empty((self.capacity, 4), dtype=np.float32)
        q_target = np.empty((self.capacity, 1), dtype=np.float32)

        for chunk in chunks:
            path = os.path.join(save_dir, chunk)
            try:
                payload = torch.load(path, weights_only=False)
                end = start + payload[0].shape[0]
                if end > self.capacity:
                    # this condition is added for resuming some very old experiments.
                    # This condition should not be needed with the new experiments
                    # and should be removed going forward.
                    select_till_index = payload[0].shape[0] - (end - self.capacity)
                    end = start + select_till_index
                else:
                    select_till_index = payload[0].shape[0]
                if payload[0].shape[1] == 39 and self.compressed_state:
                    env_obses[start:end] = np.concatenate((payload[0][:select_till_index, :18], payload[0][:select_till_index, 36:]), axis=1)
                else:
                    env_obses[start:end] = payload[0][:select_till_index]
                next_env_obses[start:end] = payload[1][:select_till_index]
                actions[start:end] = payload[2][:select_till_index]
                rewards[start:end] = payload[3][:select_till_index]
                not_dones[start:end] = payload[4][:select_till_index]

                # Support two on-disk layouts:
                # 1) TransformerReplayBuffer (len=10):
                #    [env_obs, next_env_obs, action, reward, not_done, task_encoding, task_obs, mu, log_std, q_target]
                # 2) DistilledReplayBuffer (len=9):
                #    [env_obs, next_env_obs, action, reward, not_done, task_obs, mu, log_std, q_target]
                if len(payload) >= 10:
                    task_encodings[start:end] = payload[5][:select_till_index]
                    task_obs[start:end] = payload[6][:select_till_index]
                    policy_mu[start:end] = payload[7][:select_till_index]
                    policy_log_std[start:end] = payload[8][:select_till_index]
                    q_target[start:end] = payload[9][:select_till_index]
                else:
                    task_obs[start:end] = payload[5][:select_till_index]
                    policy_mu[start:end] = payload[6][:select_till_index]
                    policy_log_std[start:end] = payload[7][:select_till_index]
                    q_target[start:end] = payload[8][:select_till_index]
                    # DistilledReplayBuffer has no task-encoding field.
                    task_encodings[start:end] = 0.0
                self.idx = end # removed - 1
                start = end
                print(f"Loaded transformer replay buffer from path: {path})")
            except EOFError as e:
                print(
                    f"Skipping loading transformer replay buffer from path: {path} due to error: {e}"
                )
        if self.normalize_rewards:
            self.max_reward = np.max(rewards[:self.idx])
            self.min_reward = np.min(rewards[:self.idx])
            rewards[:self.idx] = self.min_max_normalize(rewards[:self.idx], self.min_reward, self.max_reward)
            
        if self.compressed_state:
            self.env_obses = env_obses.reshape(self.capacity//400,400,21)
        else:
            self.env_obses = env_obses.reshape(self.capacity//400,400,39)            
        self.next_env_obses = next_env_obses.reshape(self.capacity//400,400,39)
        self.actions = actions.reshape(self.capacity//400,400,4)
        self.rewards = rewards.reshape(self.capacity//400,400,1)
        self.not_dones = not_dones.reshape(self.capacity//400,400,1)
        self.task_obs = task_obs.reshape(self.capacity//400,400,*self.task_obs_shape)
        self.task_encodings = task_encodings.reshape(self.capacity//400,400,self.task_encoding_shape)
        self.policy_mu = policy_mu.reshape(self.capacity//400,400,4)
        self.policy_log_std = policy_log_std.reshape(self.capacity//400,400,4)
        self.q_target = q_target.reshape(self.capacity//400,400,1)

        if self.idx >= self.capacity:
            self.idx = 0
            self.idx_sample = 0
            self.last_save = 0
            self.full = True
        else:
            self.idx_sample = self.idx // 400 * (400 - seq_len + 1)
            self.last_save = self.idx
        # self.delete_from_filesystem(dir_to_delete_from=save_dir)

    def load_multiple_buffer(self, save_dirs):
        start = 0
        self.source_ranges = {}

        if self.compressed_state:
            env_obses = np.empty((self.capacity, 21), dtype=np.float32)
        else:
            env_obses = np.empty((self.capacity, 39), dtype=np.float32)
        next_env_obses = np.empty((self.capacity, 39), dtype=np.float32)
        actions = np.empty((self.capacity, 4), dtype=np.float32)
        rewards = np.empty((self.capacity, 1), dtype=np.float32)
        task_obs = np.empty((self.capacity, *self.task_obs_shape), dtype=np.int64)
        not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        task_encodings = np.empty((self.capacity, self.task_encoding_shape), dtype=np.float32)
        policy_mu = np.empty((self.capacity, 4), dtype=np.float32)
        policy_log_std = np.empty((self.capacity, 4), dtype=np.float32)
        q_target = np.empty((self.capacity, 1), dtype=np.float32)

        for save_dir in save_dirs:
            dir_start = start
            chunks = os.listdir(save_dir)
            chunks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
            for chunk in chunks:
                path = os.path.join(save_dir, chunk)
                try:
                    payload = torch.load(path, weights_only=False)
                    end = start + payload[0].shape[0]
                    if end > self.capacity:
                        # this condition is added for resuming some very old experiments.
                        # This condition should not be needed with the new experiments
                        # and should be removed going forward.
                        select_till_index = payload[0].shape[0] - (end - self.capacity)
                        end = start + select_till_index
                    else:
                        select_till_index = payload[0].shape[0]
                    if payload[0].shape[1] == 39:
                        env_obs = payload[0][:select_till_index]
                        env_obses[start:end] = np.concatenate((env_obs[:,:18], env_obs[:,-3:]), axis=1)
                    else:
                        env_obses[start:end] = payload[0][:select_till_index]
                    next_env_obses[start:end] = payload[1][:select_till_index]
                    actions[start:end] = payload[2][:select_till_index]
                    rewards[start:end] = payload[3][:select_till_index]
                    not_dones[start:end] = payload[4][:select_till_index]

                    if len(payload) >= 10:
                        task_encodings[start:end] = payload[5][:select_till_index]
                        task_obs[start:end] = payload[6][:select_till_index]
                        policy_mu[start:end] = payload[7][:select_till_index]
                        policy_log_std[start:end] = payload[8][:select_till_index]
                        q_target[start:end] = payload[9][:select_till_index]
                    else:
                        task_obs[start:end] = payload[5][:select_till_index]
                        policy_mu[start:end] = payload[6][:select_till_index]
                        policy_log_std[start:end] = payload[7][:select_till_index]
                        q_target[start:end] = payload[8][:select_till_index]
                        task_encodings[start:end] = 0.0
                    self.idx = end # removed - 1
                    start = end
                    print(f"Loaded transformer replay buffer from path: {path})")
                except EOFError as e:
                    print(
                        f"Skipping loading transformer replay buffer from path: {path} due to error: {e}"
                    )
            if start > dir_start:
                self.source_ranges[str(save_dir)] = {
                    "start": dir_start,
                    "end": start,
                    "size": start - dir_start,
                }
        if self.normalize_rewards:
            self.max_reward = np.max(rewards[:self.idx])
            self.min_reward = np.min(rewards[:self.idx])
            rewards[:self.idx] = self.min_max_normalize(rewards[:self.idx], self.min_reward, self.max_reward)
            
        if self.compressed_state:
            self.env_obses = env_obses.reshape(self.capacity//400,400,21)
        else:
            self.env_obses = env_obses.reshape(self.capacity//400,400,39)
        self.next_env_obses = next_env_obses.reshape(self.capacity//400,400,39)
        self.actions = actions.reshape(self.capacity//400,400,4)
        self.rewards = rewards.reshape(self.capacity//400,400,1)
        self.not_dones = not_dones.reshape(self.capacity//400,400,1)
        self.task_obs = task_obs.reshape(self.capacity//400,400,*self.task_obs_shape)
        self.task_encodings = task_encodings.reshape(self.capacity//400,400,self.task_encoding_shape)
        self.policy_mu = policy_mu.reshape(self.capacity//400,400,4)
        self.policy_log_std = policy_log_std.reshape(self.capacity//400,400,4)
        self.q_target = q_target.reshape(self.capacity//400,400,1)

        self.idx_sample = self.idx // 400 * (400 - self.seq_len + 1)
        self.last_save = self.idx

    def reset(self):
        self.idx = 0
        self.idx_sample = 0

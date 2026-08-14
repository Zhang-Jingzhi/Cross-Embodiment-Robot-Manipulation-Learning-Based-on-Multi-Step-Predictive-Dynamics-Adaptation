from typing import Any, Dict, List, Optional, Tuple, Union, overload
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from mtrl.logger import Logger
from mtrl.col_replay_buffer import DistilledReplayBuffer, DistilledReplayBufferSample
from mtrl.agent import utils as agent_utils
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.utils.types import ConfigType, ModelType, TensorType, ModelType, OptimizerType, ComponentType, ParameterType
from mtrl.utils.utils import is_integer, make_dir
from mtrl.agent.components.predictive_adapter import PredictiveAdapter
from mtrl.agent.components.encoder import FeedForwardEncoder
ComponentOrOptimizerType = Union[ComponentType, OptimizerType]

from pathlib import Path


class TransformerAgent:
    """SAC algorithm."""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        transformer_encoder_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        transformer_encoder_optimizer_cfg: ConfigType,
        init_temperature,
        critic_tau: float,
        critic_target_update_freq,
        actor_update_freq: int,
        use_tra_preprocessing,
        use_state_embedding,
        use_cls_prediction_head,
        discount: float,
        tra_preprocessing_dim,
        cluster_latent_space,
        additional_input_state,
        mse_loss_actor,
        use_task_id,
        use_zeros,
        dropout,
        cls_alpha,
        loss_reduction: str = "mean",
        # experiment identification
        experiment: str = "none",
        seed: int = 1,
        # predictive adapter parameters
        use_predictive_adapter: bool=False,
        predictive_adapter_cfg: Optional[ConfigType] = None,
        predictive_adapter_optimizer_cfg: Optional[ConfigType] = None
    ):
        self.env_obs_shape = env_obs_shape
        self.action_shape = action_shape
        self.action_range = action_range
        self.device = device
        self.critic_tau = critic_tau
        self.discount = discount
        self.critic_target_update_freq = critic_target_update_freq
        self.actor_update_freq = actor_update_freq
        self.use_tra_preprocessing = use_tra_preprocessing
        self.use_state_embedding = use_state_embedding
        self.use_cls_prediction_head = use_cls_prediction_head
        self.tra_preprocessing_dim = tra_preprocessing_dim
        self.additional_input_state = additional_input_state
        self.cluster_latent_space = cluster_latent_space
        self.use_task_id = use_task_id
        self.use_zeros = use_zeros
        self.mse_loss_actor = mse_loss_actor
        self.cls_dim = transformer_encoder_cfg.prediction_head_cls.latent_dim
        self._optimizer_suffix = "_optimizer"
        self._components: Dict[str, ModelType] = {}
        self._optimizers: Dict[str, OptimizerType] = {}
        self.seq_len = 20

        self.use_predictive_adapter = use_predictive_adapter

        # components
        if self.use_predictive_adapter and predictive_adapter_cfg is not None:
            # 1. Copy the config dict so we can safely modify it
            pa_cfg_for_init = dict(predictive_adapter_cfg)

            # 2. Pop all load-related parameters from the dict
            load_on_init = pa_cfg_for_init.pop("load_on_init", True) # Read new flag, default True
            pretrained_dir = pa_cfg_for_init.pop("pretrained_dir", None)
            pretrained_dir = os.path.join(pretrained_dir, experiment, f"predictive_adapter_seed_{seed}")
            pretrained_step = pa_cfg_for_init.pop("pretrained_step", None)
            freeze_after_load = pa_cfg_for_init.pop("freeze_after_load", True)
            pa_cfg_for_init.pop("delta_state_num_task_pairs", None)
            pa_cfg_for_init.pop("delta_state_pair_batch_size", None)
            
            # 3. Always create the PredictiveAdapter instance first (using the clean config)
            self.predictive_adapter = PredictiveAdapter(
                state_dim=21,
                action_dim=action_shape[0],
                task_encoding_dim=self.cls_dim,
                **pa_cfg_for_init
            ).to(self.device)

            # 4. Only execute loading logic when load_on_init is True
            if load_on_init:
                print("[INFO] `load_on_init` is True. Attempting to load pretrained predictive adapter.")
                if not pretrained_dir or not os.path.exists(pretrained_dir):
                    raise FileNotFoundError(f"Pretrained predictive adapter directory not found: {pretrained_dir}")
                
                try:
                    if hasattr(self, "load_predictive_adapter"):
                        step_loaded = self.load_predictive_adapter(pretrained_dir, step=pretrained_step, load_optimizer=False)
                        print(f"[PA] Successfully loaded predictive_adapter from: {pretrained_dir} (step={step_loaded})")
                    else: # Fallback
                        self.load(pretrained_dir, step=pretrained_step if pretrained_step is not None else self.load_latest_step(pretrained_dir))
                except Exception as e:
                    print(f"[PA][WARN] failed to load pretrained predictive_adapter from {pretrained_dir}: {e}")
                    raise e # Re-raise the exception: if loading was configured but failed, we should stop

                if freeze_after_load:
                    self.predictive_adapter.eval()
                    for p in self.predictive_adapter.parameters():
                        p.requires_grad_(False)
                    print("[PA] frozen after loading (eval mode).")
                self._exclude_pa_from_ckpt = True
            else:
                print("[INFO] `load_on_init` is False. Starting with a fresh, untrained Predictive Adapter.")

            # When using predictive adapter, actor uses latent states
            actor_input_dim = predictive_adapter_cfg.get('latent_dim', 512)
            if self.additional_input_state:
                actor_input_dim += self.cls_dim  # Add task encoding dimension
        else:
            self.predictive_adapter = None
            actor_input_dim = 21
            if self.additional_input_state:
                actor_input_dim += self.cls_dim

        self.actor = hydra.utils.instantiate(
            actor_cfg, 
            env_obs_shape=[actor_input_dim],
            action_shape=action_shape,
            use_tra_preprocessing=use_tra_preprocessing,
            use_cls_prediction_head=use_cls_prediction_head,
            tra_preprocessing_dim=tra_preprocessing_dim,
            additional_input_state=additional_input_state,
            use_task_id=use_task_id or use_zeros,
            dropout=dropout,
        ).to(self.device)

        self.critic = hydra.utils.instantiate(
            critic_cfg, 
            env_obs_shape=[actor_input_dim],
            action_shape=action_shape,
            use_tra_preprocessing=use_tra_preprocessing,
            use_cls_prediction_head=use_cls_prediction_head,
            tra_preprocessing_dim=tra_preprocessing_dim,
            additional_input_state=additional_input_state,
            use_task_id=use_task_id or use_zeros,
            dropout=dropout,
        ).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            critic_cfg, 
            env_obs_shape=[actor_input_dim],  
            action_shape=action_shape,
            use_tra_preprocessing=use_tra_preprocessing,
            use_cls_prediction_head=use_cls_prediction_head,
            tra_preprocessing_dim=tra_preprocessing_dim,
            additional_input_state=additional_input_state,
            use_task_id=use_task_id or use_zeros,
            dropout=0.,
        ).to(self.device)

        self.log_alpha = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(init_temperature, dtype=np.float32)
                    for _ in range(1)
                ]
            ).to(self.device)
        )

        self.target_entropy = -np.prod(action_shape)

        repr_trans_cfg = transformer_encoder_cfg.representation_transformer
        
        # 2. Check whether we should use the new MLP mode
        is_mlp_encoder_mode = repr_trans_cfg._target_ == "mtrl.agent.components.encoder.FeedForwardEncoder"

        if is_mlp_encoder_mode:
            print("✅ [INFO] MLP Encoder mode detected. Preparing flattened input dimensions.")
            # Compute total flattened input dimension
            state_dim = 21 # Compressed state dimension
            action_dim = self.action_shape[0]
            repr_trans_cfg_dict = dict(repr_trans_cfg)
            sequence_len = repr_trans_cfg_dict.pop('sequence_len', self.seq_len) # Get from config or default
            
            flattened_input_dim = (sequence_len * state_dim) + \
                                ((sequence_len - 1) * action_dim) + \
                                ((sequence_len - 1) * 1)
            
            # Use hydra.utils.instantiate but override env_obs_shape
            self.task_encoder = hydra.utils.instantiate(
                repr_trans_cfg_dict,
                env_obs_shape=[flattened_input_dim] # Critical override!
            ).to(self.device)
            self.prediction_head_cls = None
            self.bnpy_model = None
            print(f"✅ [INFO] FeedForwardEncoder created as task encoder with input dim {flattened_input_dim}.")

        else:
            # If not MLP mode, execute the original Transformer creation logic
            print("✅ [INFO] Transformer Encoder mode detected.")
            self.task_encoder = hydra.utils.instantiate(
                repr_trans_cfg
            ).to(self.device)

            self.prediction_head_cls = hydra.utils.instantiate(
                transformer_encoder_cfg.prediction_head_cls, encoding_dim=transformer_encoder_cfg.representation_transformer.d_model
            ).to(self.device)

            self.bnpy_model = hydra.utils.instantiate(
                transformer_encoder_cfg.bnpy_model, device=self.device
            )

        if transformer_encoder_cfg.transformer.compression_dim != 0:
            self.tra_preprocessing = torch.nn.Linear(transformer_encoder_cfg.transformer.compression_dim, self.tra_preprocessing_dim).to(self.device)
        else:
            self.tra_preprocessing = torch.nn.Linear(transformer_encoder_cfg.transformer.d_model, self.tra_preprocessing_dim).to(self.device)
        
        self._components = {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "transformer_encoder": self.task_encoder,
            "tra_preprocessing": self.tra_preprocessing,
        }

        if self.use_predictive_adapter:
            self._components["predictive_adapter"] = self.predictive_adapter

        # optimizers
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, params=self.get_parameters(name="actor")
        )
        self.critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="critic")
        )
        self.transformer_encoder_optimizer = hydra.utils.instantiate(
            transformer_encoder_optimizer_cfg, params=self.get_parameters(name="transformer_encoder")
        )
        self.log_alpha_optimizer = hydra.utils.instantiate(
            alpha_optimizer_cfg, params=self.get_parameters(name="log_alpha")
        )
        
        self._optimizers = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
            "transformer_encoder": self.transformer_encoder_optimizer,
            "log_alpha": self.log_alpha_optimizer,
        }

        if self.use_predictive_adapter and predictive_adapter_optimizer_cfg is not None:
            self.predictive_adapter_optimizer = hydra.utils.instantiate(   
                predictive_adapter_optimizer_cfg, params=self.get_parameters(name="predictive_adapter")
            )
            self._optimizers["predictive_adapter"] = self.predictive_adapter_optimizer
        
        if loss_reduction not in ["mean", "none"]:
            raise ValueError(
                f"{loss_reduction} is not a supported value for `loss_reduction`."
            )
        self.loss_reduction = loss_reduction

        self.CLS_TOKEN = torch.zeros(transformer_encoder_cfg.transformer.ntoken).to(device)

        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for comp_name, component in self._components.items():
            if comp_name == 'transformer_encoder' and self.use_cls_prediction_head:
                component.train(False)
            else:
                component.train(training)

    def get_last_state_encoding(
        self, 
        states: TensorType, 
        actions: TensorType, 
        rewards: TensorType,
        disable_grad: bool
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                encoding = self.task_encoder(states, actions, rewards)
                return encoding[-2]
        encoding = self.task_encoder(states, actions, rewards)
        return encoding[-2]
    
    def get_cls_encoding(
        self, 
        states: TensorType, 
        actions: TensorType, 
        rewards: TensorType,
        disable_grad: bool,
        mask = None
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if isinstance(self.task_encoder, FeedForwardEncoder):
            # If MLP encoder, execute flattening logic
            if disable_grad:
                with torch.no_grad():
                    # 1. Flatten the sequence
                    flat_states = states.view(states.shape[0], -1)
                    flat_actions = actions.view(actions.shape[0], -1)
                    flat_rewards = rewards.view(rewards.shape[0], -1)
                    mlp_input = torch.cat([flat_states, flat_actions, flat_rewards], dim=1)
                    
                    # 2. FeedForwardEncoder.forward requires an MTObs object; create a temporary one
                    dummy_mtobs = MTObs(env_obs=mlp_input, task_obs=None, task_info=None)
                    
                    # 3. Call the encoder, which directly outputs the cls_token
                    cls = self.task_encoder(dummy_mtobs)
                    cls = F.normalize(cls, p=2, dim=-1, eps=1e-8)
            else: # If gradients are needed
                flat_states = states.view(states.shape[0], -1)
                flat_actions = actions.view(actions.shape[0], -1)
                flat_rewards = rewards.view(rewards.shape[0], -1)
                mlp_input = torch.cat([flat_states, flat_actions, flat_rewards], dim=1)
                dummy_mtobs = MTObs(env_obs=mlp_input, task_obs=None, task_info=None)
                cls = self.task_encoder(dummy_mtobs)
                cls = F.normalize(cls, p=2, dim=-1, eps=1e-8)
            return cls
        
        # If not MLP, execute the original Transformer logic
        if disable_grad:
            with torch.no_grad():
                encoding = self.task_encoder(states, actions, rewards, mask)
                cls = self.prediction_head_cls(encoding[-1])
                cls = F.normalize(cls, p=2, dim=-1, eps=1e-8)
                return cls
        encoding = self.task_encoder(states, actions, rewards, mask)
        cls = self.prediction_head_cls(encoding[-1])
        cls = F.normalize(cls, p=2, dim=-1, eps=1e-8)
        return cls
    
    def get_task_encoding(
        self, 
        states: TensorType, 
        actions: TensorType, 
        rewards: TensorType,
        disable_grad: bool,
        mask = None
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                encoding = self.task_encoder(states, actions, rewards)
                return encoding[0]
                #return torch.zeros(encoding[0].shape).to(self.device)
        encoding = self.task_encoder(states, actions, rewards)
        return encoding[0]


    def act(
        self,
        states: TensorType, 
        actions: TensorType, 
        rewards: TensorType,
        task_ids,
        sample: bool,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            states: State observations [batch_size, seq_len, state_dim]
            actions: Action history [batch_size, seq_len-1, action_dim]
            rewards: Reward history [batch_size, seq_len-1, 1]
            task_ids: Task identifiers
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        task_ids = task_ids.to(self.device)
        with torch.no_grad():
            if self.use_cls_prediction_head:
                task_encoding = self.get_cls_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True,
                )
            elif self.use_state_embedding:
                task_encoding = self.get_last_state_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True
                )
            elif self.use_task_id:
                task_encoding = task_ids.repeat(1,self.cls_dim)
            elif self.use_zeros:
                task_encoding = torch.zeros((states.shape[0],self.cls_dim)).to(self.device)
            else:
                task_encoding = self.get_task_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True
                )

            if self.use_tra_preprocessing:
                task_encoding = self.tra_preprocessing(task_encoding)

            if self.cluster_latent_space:
                task_encoding, _ = self.bnpy_model.cluster(task_encoding)
                #print(f"{torch.unique(task_encoding, dim=0)}")
            
            if self.additional_input_state:
                current_state = states[:, -1]
                if self.use_predictive_adapter:
                    latent_state = self.predictive_adapter.encode(current_state, task_encoding)
                    actor_input = torch.cat((task_encoding, latent_state), dim=1)
                else:
                    actor_input = torch.cat((task_encoding, current_state), dim=1)
            else:
                if self.use_predictive_adapter:
                    current_state = states[:, -1]
                    actor_input = self.predictive_adapter.encode(current_state, task_encoding)
                else:
                    actor_input = task_encoding
                
            mu, pi, _, _ = self.actor(actor_input)
            if sample:
                action = pi
            else:
                action = mu
            action = action.clamp(*self.action_range)
            # assert action.ndim == 2 and action.shape[0] == 1
            return action.detach().cpu().numpy()

    def select_action(self, states: TensorType, actions: TensorType, rewards: TensorType, task_ids) -> np.ndarray:
        return self.act(states, actions, rewards, task_ids, sample=False)

    def sample_action(self, states: TensorType, actions: TensorType, rewards: TensorType, task_ids) -> np.ndarray:
        return self.act(states, actions, rewards, task_ids, sample=True)

    def get_parameters(self, name: str) -> List[torch.nn.parameter.Parameter]: 
        """Get parameters corresponding to a given component.

        Args:
            name (str): name of the component.

        Returns:
            List[torch.nn.parameter.Parameter]: list of parameters.
        """
        if name == "actor":
            return list(self.actor.model.parameters())
        elif name in ["log_alpha", "alpha"]:
            return [self.log_alpha]
        return list(self._components[name].parameters())

    def _prepare_predictive_adapter_batch_from_buffer(
        self,
        replay_buffer,
        idxs=None,
    ):
        """Prepare either single-step or multi-step targets for predictive-adapter training."""
        if idxs is None:
            idxs = replay_buffer.sample_indices()

        rollout_horizon = getattr(self.predictive_adapter, "rollout_horizon", 1) if self.predictive_adapter is not None else 1
        valid_mask = None

        if rollout_horizon > 1 and hasattr(replay_buffer, "sample_multi_step"):
            states, actions, rewards, next_states, valid_mask, task_ids, task_encoding = replay_buffer.sample_multi_step(
                rollout_horizon, index=idxs, device=self.device
            )
            next_states = _compress_meta_state(next_states)
        else:
            states, actions, rewards, next_states, _, _, _, task_ids, task_encoding = replay_buffer.sample_new(idxs)
            next_states = _compress_meta_state(next_states)

        if self.use_zeros:
            task_encoding = torch.zeros((states.shape[0], self.cls_dim), device=self.device)
        elif self.use_task_id:
            task_encoding = task_ids.repeat(1, self.cls_dim)
        elif self.use_cls_prediction_head:
            states_seq, actions_seq, rewards_seq, _, _ = replay_buffer.build_sequences_for_indices(
                idxs, self.seq_len, device=self.device
            )
            task_encoding = self.get_cls_encoding(
                states=states_seq,
                actions=actions_seq,
                rewards=rewards_seq,
                disable_grad=True,
                mask=None,
            )

        return states, actions, rewards, next_states, task_ids, task_encoding, valid_mask, idxs

    def _compute_delta_pair_loss(self, delta_pair_batches):
        if (
            self.predictive_adapter is None
            or getattr(self.predictive_adapter, "delta_state_loss_weight", 0.0) <= 0.0
            or len(delta_pair_batches) == 0
        ):
            return None

        pair_losses = []
        for pair_batch in delta_pair_batches:
            pair_losses.append(
                self.predictive_adapter.compute_delta_state_similarity_loss(
                    state_a=pair_batch["state_a"],
                    action_a=pair_batch["action_a"],
                    next_state_a=pair_batch["next_state_a"],
                    state_b=pair_batch["state_b"],
                    action_b=pair_batch["action_b"],
                    next_state_b=pair_batch["next_state_b"],
                    valid_mask_a=pair_batch.get("valid_mask_a"),
                    valid_mask_b=pair_batch.get("valid_mask_b"),
                )
            )

        if not pair_losses:
            return None
        return torch.stack(pair_losses).mean()

    def distill_actor(
        self,
        replay_buffer_list: DistilledReplayBuffer,
        logger: Logger,
        step: int,
        **kwargs
    ) -> None:
        """Update the actor and alpha component.

        
        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        
        if isinstance(replay_buffer_list, list):
            state_list, action_list, rewards_list, mu_list, log_std_list, task_encoding_list = [], [], [], [], [], []
            for buffer in replay_buffer_list:
                #states, actions, rewards, _, mu_sample, log_stds_sample, _, task_id_sample = buffer.sample_trajectories(seq_len)
                states, actions, rewards, _, mu_sample, log_stds_sample, _, _, encoding_sample = buffer.sample_new()
                state_list.append(states)
                action_list.append(actions)
                rewards_list.append(rewards)
                mu_list.append(mu_sample)
                log_std_list.append(log_stds_sample)
                task_encoding_list.append(encoding_sample)
            states = torch.cat(state_list)
            actions = torch.cat(action_list)
            rewards = torch.cat(rewards_list)
            batch_mu = torch.cat(mu_list)
            batch_log_std = torch.cat(log_std_list)
            task_encoding = torch.cat(task_encoding_list)
        else:
            idxs = replay_buffer_list.sample_indices()
            #states, actions, rewards, _, batch_mu, batch_log_std, _, task_ids = replay_buffer_list.sample_trajectories(seq_len)
            states, actions, rewards, _, batch_mu, batch_log_std, _, task_ids, task_encoding = replay_buffer_list.sample_new(idxs)

        if self.use_zeros:
            task_encoding = torch.zeros((states.shape[0],self.cls_dim)).to(self.device)
        elif self.use_task_id:
            task_encoding = task_ids.repeat(1, self.cls_dim)
        elif self.use_cls_prediction_head:
            states_seq, actions_seq, rewards_seq, current_state, _ = replay_buffer_list.build_sequences_for_indices(idxs, self.seq_len, device=self.device)
            task_encoding = self.get_cls_encoding(
                states=states_seq, actions=actions_seq, rewards=rewards_seq,
                disable_grad=True, mask=None
            )

        if self.additional_input_state:
            current_state = states
            if self.use_predictive_adapter:
                latent_state = self.predictive_adapter.encode(current_state, task_encoding)
                actor_input = torch.cat((task_encoding, latent_state), dim=1)
            else:
                actor_input = torch.cat((task_encoding, current_state), dim=1)
        else:
            if self.use_predictive_adapter:
                current_state = states
                actor_input = self.predictive_adapter.encode(current_state, task_encoding)
            else:
                actor_input = task_encoding

        mu, _, _, log_std = self.actor(actor_input)

        mu_diff = batch_mu - mu
        
        if self.mse_loss_actor:
            # MSE loss hard to balance the weight of sdt vs mu
            mse_loss_mu = (mu - batch_mu) ** 2
            mse_loss_log_std = (log_std - batch_log_std) ** 2 * (2 / torch.maximum(torch.tensor([2.]).to(self.device), -batch_log_std)) # weight for reducing div when std anyways low
            loss = mse_loss_mu + 0.2*mse_loss_log_std
        else:
            # KL problem: for lowlog_std the kl can be very high 
            # (e.g. istribution with mu [ 1.0000,  1.0000,  0.3982, -0.9998]  and std_log [-15.3441,  -2.5173, -15.6552, -17.8864] 
            #  and the normal distribution with mu [ 1.0000,  0.9841,  1.0000, -0.9997] and std_log [-14.4620,  -1.6691, -15.2972, -17.0109])
            
            distribution_dim = mu.size(-1)
            loss = 0.5 * (
                torch.exp(log_std - batch_log_std).sum(-1) +
                (mu_diff ** 2 / torch.exp(batch_log_std)).sum(-1) -
                distribution_dim +
                (batch_log_std.sum(-1) - log_std.sum(-1))
            )
        
        if self.loss_reduction == "mean":
            actor_loss = loss.mean()
            logger.log("train/actor_loss", actor_loss, step, tb_log=kwargs['tb_log'])
            logger.log("train/mu_div", torch.abs(mu_diff).mean(), step, tb_log=kwargs['tb_log'])
            logger.log("train/log_std_div", torch.abs(log_std - batch_log_std).mean(), step, tb_log=kwargs['tb_log'])

        elif self.loss_reduction == "none":
            actor_loss = loss.sum()
            logger.log("train/actor_loss", actor_loss, step, tb_log=kwargs['tb_log'])
            logger.log("train/mu_div", torch.abs(mu_diff).sum(), step, tb_log=kwargs['tb_log'])
            logger.log("train/log_std_div", torch.abs(log_std - batch_log_std).sum(), step, tb_log=kwargs['tb_log'])

        self._optimizers["actor"].zero_grad()
        self._optimizers["transformer_encoder"].zero_grad()
        # actor loss backward
        actor_loss.backward()
        self.actor_optimizer.step()
        # self.transformer_encoder_optimizer.step()

    def distill_critic(
        self,
        replay_buffer_list: DistilledReplayBuffer,
        logger: Logger,
        step: int,
        **kwargs
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        if isinstance(replay_buffer_list, list):
            state_list, rewards_list, q_target_list, task_encoding_list = [], [], [], []
            for buffer in replay_buffer_list:
                states, _, rewards, _, _, _, q_target, _, encoding_sample = buffer.sample_new()
                state_list.append(states)
                rewards_list.append(rewards)
                q_target_list.append(q_target)
                task_encoding_list.append(encoding_sample)
            states = torch.cat(state_list)
            rewards = torch.cat(rewards_list)
            q_targets = torch.cat(q_target_list)
            task_encoding = torch.cat(task_encoding_list)
        else:
            idxs = replay_buffer_list.sample_indices()
            states, actions, rewards, _, _, _, q_targets, _, task_encoding = replay_buffer_list.sample_new(idxs)

        if self.use_cls_prediction_head:
            states_seq, actions_seq, rewards_seq, current_state, _ = replay_buffer_list.build_sequences_for_indices(idxs, self.seq_len, device=self.device)
            task_encoding = self.get_cls_encoding(
                states=states_seq, actions=actions_seq, rewards=rewards_seq,
                disable_grad=True, mask=None
            )

        if self.additional_input_state:
            if self.use_predictive_adapter:
                latent_state = self.predictive_adapter.encode(states, task_encoding)
                critic_input = torch.cat((task_encoding, latent_state, actions), dim=1)
            else:
                critic_input = torch.cat((task_encoding, states, actions), dim=1)
        else:
            raise NotImplemented

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(critic_input)
        # mean loss
        critic_loss = F.mse_loss(
            current_Q1, q_targets, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, q_targets, reduction=self.loss_reduction)
        
        loss_to_log = critic_loss
        if self.loss_reduction == "mean":
            logger.log("train/critic_loss", loss_to_log.mean(), step, tb_log=kwargs['tb_log'])
        elif self.loss_reduction == "none":
            logger.log("train/critic_loss", loss_to_log.mean(), step, tb_log=kwargs['tb_log'])

        # critic loss backward
        self._optimizers["critic"].zero_grad()
        critic_loss.backward()
        # Optimize the critic
        self._optimizers["critic"].step()

        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )

    def train_predictive_adapter(
        self,
        replay_buffer_list: DistilledReplayBuffer,
        logger: Logger,
        step: int,
        **kwargs
    ) -> None:
        """Train the predictive adapter components.
        Args:
            replay_buffer_list: Replay buffer(s) containing training data
            logger: Logger for metrics
            step: Training step
            **kwargs: Additional arguments
        """
        if not self.use_predictive_adapter:
            return

        if isinstance(replay_buffer_list, list):
            state_list, action_list, reward_list, next_state_list = [], [], [], []
            task_id_list, task_encoding_list, valid_mask_list = [], [], []
            for buffer in replay_buffer_list:
                states, actions, rewards, next_states, task_ids, encoding_sample, valid_mask, _ = \
                    self._prepare_predictive_adapter_batch_from_buffer(buffer)
                state_list.append(states)
                action_list.append(actions)
                reward_list.append(rewards)
                next_state_list.append(next_states)
                task_id_list.append(task_ids)
                task_encoding_list.append(encoding_sample)
                if valid_mask is not None:
                    valid_mask_list.append(valid_mask)
            states = torch.cat(state_list)
            actions = torch.cat(action_list)
            rewards = torch.cat(reward_list)
            next_states = torch.cat(next_state_list)
            task_ids = torch.cat(task_id_list)
            task_encoding = torch.cat(task_encoding_list)
            valid_mask = torch.cat(valid_mask_list) if valid_mask_list else None
        else:
            states, actions, rewards, next_states, task_ids, task_encoding, valid_mask, idxs = \
                self._prepare_predictive_adapter_batch_from_buffer(replay_buffer_list)

        if step == 0:  # Print only once
            print(f"[TA->pa/DEBUG] call compute_loss with:")
            print(f"  states{tuple(states.shape)} actions{tuple(actions.shape)} rewards{tuple(rewards.shape)}")
            print(f"  next_states{tuple(next_states.shape)} task_encoding{tuple(task_encoding.shape)}")
            if valid_mask is not None:
                print(f"  valid_mask{tuple(valid_mask.shape)}")

        # Compute predictive adapter losses
        dynamics_loss, reward_loss, total_loss = self.predictive_adapter.compute_loss(
            state=states,
            action=actions,
            next_state=next_states,
            reward=rewards,
            task_encoding=task_encoding,
            valid_mask=valid_mask,
        )
        delta_state_loss = torch.zeros((), device=states.device, dtype=states.dtype)
        delta_pair_batches = kwargs.get("delta_pair_batches") or []
        maybe_delta_loss = self._compute_delta_pair_loss(delta_pair_batches)
        if maybe_delta_loss is not None:
            delta_state_loss = maybe_delta_loss
            total_loss = total_loss + self.predictive_adapter.delta_state_loss_weight * delta_state_loss
        # Optimize predictive adapter
        self._optimizers["predictive_adapter"].zero_grad()
        total_loss.backward()
        self._optimizers["predictive_adapter"].step()
        # Log metrics
        if kwargs.get('tb_log', False):
            logger.log("train/predictive_adapter_dynamics_loss", dynamics_loss.item(), step, tb_log=True)
            logger.log("train/predictive_adapter_reward_loss", reward_loss.item(), step, tb_log=True)
            logger.log("train/predictive_adapter_delta_state_loss", delta_state_loss.item(), step, tb_log=True)
            logger.log("train/predictive_adapter_total_loss", total_loss.item(), step, tb_log=True)


    def _get_target_V(
        self, current_state, policy_action, task_encoding,
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        critic_input = torch.cat((task_encoding, current_state, policy_action), dim=1)
        target_Q1, target_Q2 = self.critic_target(critic_input)
        return torch.min(target_Q1, target_Q2)
    
    def evaluate_predictive_adapter_total(
        self,
        validation_buffer: DistilledReplayBuffer,
        batch_size: int = 512,
        delta_pair_batches=None,
    ):
        """
        Evaluates the predictive adapter on a validation buffer.
        Iterates through the entire buffer, computes losses without backpropagation,
        and returns the average losses.
        """
        if not self.use_predictive_adapter or self.predictive_adapter is None:
            print("[Agent-WARN] Predictive adapter is not configured for this agent.")
            return {}

        self.predictive_adapter.eval()  # Switch to eval mode (e.g., disable dropout)
        
        total_dynamics_loss = 0.0
        total_reward_loss = 0.0
        total_pa_loss = 0.0
        
        buffer_size = len(validation_buffer)
        num_batches = (buffer_size + batch_size - 1) // batch_size
        
        print(f"Starting evaluation on {buffer_size} samples in {num_batches} batches...")

        with torch.no_grad():  # Critical! Disable gradient computation to save memory and time
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, buffer_size)
                idxs = np.arange(start_idx, end_idx)

                states, actions, rewards, next_states, _, task_encoding, valid_mask, _ = \
                    self._prepare_predictive_adapter_batch_from_buffer(validation_buffer, idxs=idxs)
                dynamics_loss, reward_loss, pa_total_loss = self.predictive_adapter.compute_loss(
                    state=states,
                    action=actions,
                    next_state=next_states,
                    reward=rewards,
                    task_encoding=task_encoding,
                    valid_mask=valid_mask,
                )
                
                total_dynamics_loss += dynamics_loss.item()
                total_reward_loss += reward_loss.item()
                total_pa_loss += pa_total_loss.item()

        # Compute average losses
        avg_dynamics_loss = total_dynamics_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_total_loss = total_pa_loss / num_batches
        avg_delta_state_loss = 0.0
        maybe_delta_loss = self._compute_delta_pair_loss(delta_pair_batches or [])
        if maybe_delta_loss is not None:
            avg_delta_state_loss = maybe_delta_loss.item()
            avg_total_loss += (
                self.predictive_adapter.delta_state_loss_weight * avg_delta_state_loss
            )

        self.predictive_adapter.train()
        
        return {
            "avg_dynamics_loss": avg_dynamics_loss,
            "avg_reward_loss": avg_reward_loss,
            "avg_delta_state_loss": avg_delta_state_loss,
            "avg_total_loss": avg_total_loss,
        }
    
    def evaluate_predictive_adapter(
        self,
        validation_buffer: DistilledReplayBuffer,
        batch_size: int = 512,
        max_batches: int = 100,
        delta_pair_batches=None,
    ):
        """
        Evaluate the Predictive Adapter on a validation set.
        For speed, randomly samples up to max_batches batches by default instead of iterating over the entire set.
        
        Args:
            max_batches (int): Maximum number of evaluation batches. Set to None for full evaluation.
        """
        if not self.use_predictive_adapter or self.predictive_adapter is None:
            print("[Agent-WARN] Predictive adapter is not configured for this agent.")
            return {}

        self.predictive_adapter.eval()
        
        total_dynamics_loss = 0.0
        total_reward_loss = 0.0
        total_pa_loss = 0.0
        
        buffer_size = len(validation_buffer)
        if buffer_size == 0:
            self.predictive_adapter.train()
            return {}

        # 1. Determine how many batches to run
        full_batches = (buffer_size + batch_size - 1) // batch_size
        if max_batches is not None:
            num_batches = min(full_batches, max_batches)
        else:
            num_batches = full_batches

        print(f"Starting evaluation: sampling {num_batches} batches (approx {num_batches*batch_size} steps)...")

        # 2. [Key] Generate random indices to ensure coverage of different tasks/robots
        # Randomly shuffle all indices, then take only the portion we need
        all_indices = np.random.permutation(buffer_size)
        # Take the first num_batches * batch_size data points
        target_indices = all_indices[:num_batches * batch_size]

        with torch.no_grad():
            for i in range(num_batches):
                # Get the random indices for the current batch
                batch_idxs = target_indices[i * batch_size : (i + 1) * batch_size]
                
                # Boundary guard (the slicing above usually won't go out of bounds, but for robustness)
                if len(batch_idxs) == 0:
                    continue

                states, actions, rewards, next_states, _, task_encoding, valid_mask, _ = \
                    self._prepare_predictive_adapter_batch_from_buffer(validation_buffer, idxs=batch_idxs)
                dynamics_loss, reward_loss, pa_total_loss = self.predictive_adapter.compute_loss(
                    state=states,
                    action=actions,
                    next_state=next_states,
                    reward=rewards,
                    task_encoding=task_encoding,
                    valid_mask=valid_mask,
                )
                
                total_dynamics_loss += dynamics_loss.item()
                total_reward_loss += reward_loss.item()
                total_pa_loss += pa_total_loss.item()

        # Compute averages
        # Note: if batch sizes are inconsistent, a sample-weighted average would be better,
        # but since we randomly sample with a fixed batch_size, dividing by num_batches is precise enough.
        if num_batches > 0:
            avg_dynamics_loss = total_dynamics_loss / num_batches
            avg_reward_loss = total_reward_loss / num_batches
            avg_total_loss = total_pa_loss / num_batches
        else:
            avg_dynamics_loss = avg_reward_loss = avg_total_loss = 0.0
        avg_delta_state_loss = 0.0
        maybe_delta_loss = self._compute_delta_pair_loss(delta_pair_batches or [])
        if maybe_delta_loss is not None:
            avg_delta_state_loss = maybe_delta_loss.item()
            avg_total_loss += (
                self.predictive_adapter.delta_state_loss_weight * avg_delta_state_loss
            )

        self.predictive_adapter.train()
        
        return {
            "avg_dynamics_loss": avg_dynamics_loss,
            "avg_reward_loss": avg_reward_loss,
            "avg_delta_state_loss": avg_delta_state_loss,
            "avg_total_loss": avg_total_loss,
        }
    
    def evaluate_critic(
        self,
        replay_buffer_list: DistilledReplayBuffer,
        logger: Logger,
        step: int,
        seq_len: int,
        **kwargs
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        if isinstance(replay_buffer_list, list):
            state_list, actions_list, q_target_list, rewards_list, task_id_list, task_encoding_list = [], [], [], [], [], []
            for buffer in replay_buffer_list:
                idxs = buffer.sample_indices()
                states, actions, rewards, _, _, _, q_target, task_id_sample, task_encoding = buffer.sample_new(idxs)
                if self.use_cls_prediction_head:
                    states_seq, actions_seq, rewards_seq, current_state, _ = buffer.build_sequences_for_indices(idxs, self.seq_len, device=self.device)
                    task_encoding = self.get_cls_encoding(
                        states=states_seq, actions=actions_seq, rewards=rewards_seq,
                        disable_grad=True, mask=None
                    )
                state_list.append(states)
                actions_list.append(actions)
                q_target_list.append(q_target)
                rewards_list.append(rewards)
                task_id_list.append(task_id_sample)
                task_encoding_list.append(task_encoding)
            states = torch.cat(state_list)
            actions = torch.cat(actions_list)
            q_target = torch.cat(q_target_list)
            rewards = torch.cat(rewards_list)
            task_ids = torch.cat(task_id_list)
            task_encoding = torch.cat(task_encoding_list)
        else:
            idxs = replay_buffer_list.sample_indices()
            states, actions, rewards, _, _, _, q_target, task_ids, task_encoding = replay_buffer_list.sample_new(idxs)

            if self.use_cls_prediction_head:
                states_seq, actions_seq, rewards_seq, current_state, _ = replay_buffer_list.build_sequences_for_indices(idxs, self.seq_len, device=self.device)
                task_encoding = self.get_cls_encoding(
                    states=states_seq, actions=actions_seq, rewards=rewards_seq,
                    disable_grad=True, mask=None
                )

        with torch.no_grad():
            if self.additional_input_state:
                if self.use_predictive_adapter:
                    # Use predictive adapter to encode the state
                    latent_state = self.predictive_adapter.encode(states, task_encoding)
                    critic_input = torch.cat((task_encoding, latent_state, actions), dim=1)
                else:
                    # Keep original logic
                    critic_input = torch.cat((task_encoding, states, actions), dim=1)
            else:
                raise NotImplemented

            # get current Q estimates
            current_Q1, current_Q2 = self.critic(critic_input)

            # mean loss
            critic_loss = F.mse_loss(
                current_Q1, q_target, reduction=self.loss_reduction
            ) + F.mse_loss(current_Q2, q_target, reduction=self.loss_reduction)
            
            if self.loss_reduction == "mean":
                logger.log("col_eval/critic_loss", critic_loss.mean(), step, tb_log=kwargs['tb_log'])
            elif self.loss_reduction == "none":
                logger.log("col_eval/critic_loss", critic_loss.mean(), step, tb_log=kwargs['tb_log'])

    def calculate_q_targets_expert_action(
        self,
        states: TensorType, 
        actions: TensorType, 
        rewards: TensorType,
        task_ids,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        task_ids = task_ids.to(self.device)
        with torch.no_grad():
            if self.use_cls_prediction_head:
                task_encoding = self.get_cls_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True,
                )
            elif self.use_state_embedding:
                task_encoding = self.get_last_state_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True
                )
            elif self.use_task_id:
                task_encoding = task_ids.repeat(1,self.cls_dim)
            else:
                task_encoding = self.get_task_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True
                )

            if self.use_tra_preprocessing:
                task_encoding = self.tra_preprocessing(task_encoding)

            if self.cluster_latent_space:
                task_encoding, _ = self.bnpy_model.cluster(task_encoding)
                #print(f"{torch.unique(task_encoding, dim=0)}")
            
            if self.additional_input_state:
                current_state = states[:, -1]
                actor_input = torch.cat((task_encoding, current_state), dim=1)
            else:
                actor_input = task_encoding
                
            mu, _, _, log_std = self.actor(actor_input)
            mu = mu.clamp(*self.action_range)

            critic_input = torch.cat((actor_input, mu), dim=1)
            current_Q1, current_Q2 = self.critic(critic_input)
            
            return mu.detach().cpu().numpy(), log_std.detach().cpu().numpy(), torch.min(current_Q1, current_Q2).detach().cpu().numpy()
        
    def calculate_task_encoding(
        self,
        states: TensorType, 
        actions: TensorType, 
        rewards: TensorType,
        task_ids: TensorType,
        **kwargs
    ) -> None:
        """Update the actor and alpha component.

        
        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        task_ids = task_ids.to(self.device)

        #if cls_token:
        #    cls_token_expanded = self.CLS_TOKEN.unsqueeze(0).expand(batch_input.shape[0], -1).unsqueeze(1)
        #    batch_input = torch.cat((cls_token_expanded, batch_input), dim=1)

        if self.use_cls_prediction_head:
                task_encoding = self.get_cls_encoding(
                    states,
                    actions,
                    rewards,
                    disable_grad=True,
                    mask=None,
                )
        elif self.use_state_embedding:
            task_encoding = self.get_last_state_encoding(
                states,
                actions,
                rewards,
                disable_grad=False
            )
        elif self.use_task_id:
            task_encoding = task_ids.repeat(1,self.cls_dim) 
        else:
            task_encoding = self.get_task_encoding(
                states,
                actions,
                rewards,
                disable_grad=False
            )

        if self.use_tra_preprocessing:
            task_encoding = self.tra_preprocessing(task_encoding)

        if self.cluster_latent_space:
            task_encoding, _ = self.bnpy_model.cluster(task_encoding)

        return task_encoding
    
    def update(
        self,
        replay_buffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}
        
        batch = replay_buffer.sample()
        logger.log("train/batch_reward", batch.reward.mean(), step)
        
        if self.additional_input_state:
            next_env_obs = torch.cat((batch.next_env_obs[:, :18], batch.next_env_obs[:, 36:]), dim=1)
            actor_input = torch.cat((batch.task_encoding, next_env_obs), dim=1)
        else:
            raise NotImplemented
            actor_input = batch.task_encoding

        _, policy_action, log_pi, _ = self.actor(actor_input)
        
        with torch.no_grad():
            target_V = self._get_target_V(next_env_obs, policy_action, batch.task_encoding) - self.get_alpha(env_index=batch.task_obs).detach() * log_pi
            # task_info = kwargs['experts'][2.0].get_task_info(
            #         task_encoding=None,
            #         component_name="critic",
            #         env_index=batch.task_obs
            #     )
            # mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
            # target_Q1, target_Q2 = kwargs['experts'][2.0].critic_target(mtobs=mtobs, action=policy_action)
            # target_V = torch.min(target_Q1, target_Q2) - self.get_alpha(env_index=batch.task_obs).detach() * log_pi
            #target_V = kwargs['experts'][2.0]._get_target_V(batch, task_info)
            target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        if self.additional_input_state:
            critic_input = torch.cat((batch.task_encoding, batch.env_obs, batch.action), dim=1)
        else:
            raise NotImplemented
            critic_input = batch.task_encoding
        current_Q1, current_Q2 = self.critic(critic_input)
        critic_loss = F.mse_loss(
            current_Q1, target_Q, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, target_Q, reduction=self.loss_reduction)

        loss_to_log = critic_loss
        if self.loss_reduction == "none":
            loss_to_log = loss_to_log.mean()
        logger.log("train/critic_loss", loss_to_log, step)

        if loss_to_log > 1e8:
            raise RuntimeError(
                f"critic_loss = {loss_to_log} is too high. Stopping training."
            )

        # critic loss backward
        self._optimizers["critic"].zero_grad()
        critic_loss.backward()

        # Optimize the critic
        self.critic_optimizer.step()

        if step % self.actor_update_freq == 0:
            # detach encoder, so we don't update it with the actor loss
            if self.additional_input_state:
                actor_input = torch.cat((batch.task_encoding, batch.env_obs), dim=1)
            else:
                raise NotImplemented
                actor_input = batch.task_encoding
            _, pi, log_pi, log_std = self.actor(actor_input)
            critic_input = critic_input = torch.cat((actor_input, pi), dim=1)
            actor_Q1, actor_Q2 = self.critic(critic_input)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            if self.loss_reduction == "mean":
                actor_loss = (
                    self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
                ).mean()
                logger.log("train/actor_loss", actor_loss, step)

            elif self.loss_reduction == "none":
                actor_loss = self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
                logger.log("train/actor_loss", actor_loss.mean(), step)

            logger.log("train/actor_target_entropy", self.target_entropy, step)

            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )

            logger.log("train/actor_entropy", entropy.mean(), step)

            # optimize the actor
            self._optimizers["actor"].zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            if self.loss_reduction == "mean":
                alpha_loss = (
                    self.get_alpha(batch.task_obs)
                    * (-log_pi - self.target_entropy).detach()
                ).mean()
                logger.log("train/alpha_loss", alpha_loss, step)
            elif self.loss_reduction == "none":
                alpha_loss = (
                    self.get_alpha(batch.task_obs)
                    * (-log_pi - self.target_entropy).detach()
                )
                logger.log("train/alpha_loss", alpha_loss.mean(), step)
            # breakpoint()
            # logger.log("train/alpha_value", self.get_alpha(batch.task_obs), step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )

    def get_alpha(self, env_index: TensorType) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        #if self.multitask_cfg.should_use_disentangled_alpha:
        #    return self.log_alpha[env_index].exp()
        #else:
        return self.log_alpha[0].exp()

    def get_component_name_list_for_checkpointing(self) -> List[Tuple[ModelType, str]]:
        """Get the list of tuples of (model, name) from the agent to checkpoint.

        Returns:
            List[Tuple[ModelType, str]]: list of tuples of (model, name).
        """
        items = [(value, key) for key, value in self._components.items()]
        if getattr(self, "_exclude_pa_from_ckpt", False):
            items = [(m, n) for (m, n) in items if n != "predictive_adapter"]
        return items


    def get_optimizer_name_list_for_checkpointing(
        self,
    ) -> List[Tuple[OptimizerType, str]]:
        """Get the list of tuples of (optimizer, name) from the agent to checkpoint.

        Returns:
            List[Tuple[OptimizerType, str]]: list of tuples of (optimizer, name).
        """
        items = [(value, key) for key, value in self._optimizers.items()]
        if getattr(self, "_exclude_pa_from_ckpt", False):
            items = [(opt, n) for (opt, n) in items if n != "predictive_adapter"]
        return items

    def save(
        self,
        model_dir: str,
        step: int,
        retain_last_n: int,
        should_save_metadata: bool = True,
    ) -> None:
        """Save the agent.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.
            should_save_metadata (bool, optional): should training metadata be
                saved. Defaults to True.
        """
        if retain_last_n == 0:
            print("Not saving the models as retain_last_n = 0")
            return
        make_dir(model_dir)
        # write a test case for save/load

        self.save_components(model_dir, step, retain_last_n)

        self.save_optimizers(model_dir, step, retain_last_n)

        if should_save_metadata:
            self.save_metadata(model_dir, step)

    def save_components(self, model_dir: str, step: int, retain_last_n: int) -> None:
        """Save the different components of the agent.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.

        """
        return self.save_components_or_optimizers(
            component_or_optimizer_list=self.get_component_name_list_for_checkpointing(),
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            suffix="",
        )

    def save_optimizers(self, model_dir: str, step: int, retain_last_n: int) -> None:
        """Save the different optimizers of the agent.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.

        """

        return self.save_components_or_optimizers(
            component_or_optimizer_list=self.get_optimizer_name_list_for_checkpointing(),
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            suffix=self._optimizer_suffix,
        )

    def save_components_or_optimizers(
        self,
        component_or_optimizer_list: Union[
            List[Tuple[ComponentType, str]], List[Tuple[OptimizerType, str]]
        ],
        model_dir: str,
        step: int,
        retain_last_n: int,
        suffix: str = "",
    ) -> None:
        """Save the components and optimizers from the given list.

        Args:
            component_or_optimizer_list
                (Union[ List[Tuple[ComponentType, str]], List[Tuple[OptimizerType, str]] ]):
                list of components and optimizers to save.
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.
            suffix (str, optional): suffix to add at the name of the model before
                checkpointing. Defaults to "".
        """
        model_dir_path = Path(model_dir)

        for component_or_optimizer, name in component_or_optimizer_list:
            if component_or_optimizer is not None:
                name = name + suffix
                path_to_save_at = f"{model_dir}/{name}_{step}.pt"
                torch.save(component_or_optimizer.state_dict(), path_to_save_at)
                print(f"Saved {path_to_save_at}")
                if retain_last_n == -1:
                    continue
                reverse_sorted_existing_versions = (
                    _get_reverse_sorted_existing_versions(model_dir_path, name)
                )
                if len(reverse_sorted_existing_versions) > retain_last_n:
                    # assert len(reverse_sorted_existing_versions) == retain_last_n + 1
                    for path_to_del in reverse_sorted_existing_versions[retain_last_n:]:
                        if os.path.lexists(path_to_del):
                            os.remove(path_to_del)
                            print(f"Deleted {path_to_del}")

    def save_metadata(self, model_dir: str, step: int) -> None:
        """Save the metadata.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.

        """
        metadata = {"step": step}
        path_to_save_at = f"{model_dir}/metadata.pt"
        torch.save(metadata, path_to_save_at)
        print(f"Saved {path_to_save_at}")

    def load(self, model_dir: Optional[str], step: Optional[int]) -> None:
        """Load the agent.

        Args:
            model_dir (Optional[str]): directory to load the model from.
            step (Optional[int]): step for tracking the training of the agent.
        """

        if model_dir is None or step is None:
            return
        for component, name in self.get_component_name_list_for_checkpointing():
            component = _load_component_or_optimizer(
                component,
                model_dir=model_dir,
                name=name,
                step=step,
            )
            if isinstance(component, ComponentType):
                component = component.to(self.device)
        for optimizer, name in self.get_optimizer_name_list_for_checkpointing():
            optimizer = _load_component_or_optimizer(
                component_or_optimizer=optimizer,
                model_dir=model_dir,
                name=name + self._optimizer_suffix,
                step=step,
            )

    def save_only_predictive_adapter(self, model_dir: str, step: int, retain_last_n: int):
        items = []
        if hasattr(self, "predictive_adapter") and self.predictive_adapter is not None:
            items.append((self.predictive_adapter, "predictive_adapter"))
        if not items:
            print("No predictive_adapter to save.")
            return
        return self.save_components_or_optimizers(
            component_or_optimizer_list=items,
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            suffix="",  # No optimizer suffix for weights
        )

    def save_predictive_adapter_optimizer(self, model_dir: str, step: int, retain_last_n: int):
        """Save only the predictive_adapter optimizer (optional)."""
        opt = self._optimizers.get("predictive_adapter", None) if hasattr(self, "_optimizers") else None
        items = []
        if opt is not None:
            items.append((opt, "predictive_adapter"))
        if not items:
            print("No predictive_adapter optimizer to save.")
            return
        return self.save_components_or_optimizers(
            component_or_optimizer_list=items,
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            suffix=self._optimizer_suffix,  # Uniform suffix for optimizers
        )

    def load_latest_step(self, model_dir: str) -> int:
        """Load the agent using the latest training step.

        Args:
            model_dir (Optional[str]): directory to load the model from.

        Returns:
            int: step for tracking the training of the agent.
        """
        latest_step = -1
        if model_dir is None:
            print("model_dir is None.")
            return latest_step
        metadata = self.load_metadata(model_dir=model_dir)
        if metadata is None:
            return latest_step + 1
        latest_step = metadata["step"]
        self.load(model_dir, step=latest_step)
        return latest_step + 1

    def load_metadata(self, model_dir: str) -> Optional[Dict[Any, Any]]:
        """Load the metadata of the agent.

        Args:
            model_dir (str): directory to load the model from.

        Returns:
            Optional[Dict[Any, Any]]: metadata.
        """
        metadata_path = f"{model_dir}/metadata.pt"
        if not os.path.exists(metadata_path):
            print(f"{metadata_path} does not exist.")
            metadata = None
        else:
            metadata = torch.load(metadata_path)
        return metadata
    
    def load_predictive_adapter(self, model_dir: str, step: Optional[int] = None, load_optimizer: bool = False) -> int:
        """Load only the predictive_adapter (and optionally the optimizer). Returns the loaded step."""
        from pathlib import Path

        def _latest_step(name: str) -> Optional[int]:
            p = Path(model_dir)
            cands = sorted(p.glob(f"{name}_*.pt"))
            if not cands:
                return None
            steps = [int(x.stem.split("_")[-1]) for x in cands]
            return max(steps)

        # If step is not given, prefer getting it from metadata; otherwise infer from filename
        meta = self.load_metadata(model_dir)
        if step is None:
            step = (meta or {}).get("step", None)
            if step is None:
                step = _latest_step("predictive_adapter")
        if step is None:
            raise FileNotFoundError(f"No predictive_adapter_*.pt found under {model_dir}")

        # Load only the predictive_adapter
        pa_path = os.path.join(model_dir, f"predictive_adapter_{step}.pt")
        state = torch.load(pa_path, map_location=self.device)
        self.predictive_adapter.load_state_dict(state)
        print(f"[TransformerAgent] predictive_adapter loaded from {pa_path}")

        # Optional: load optimizer
        if load_optimizer and hasattr(self, "_optimizers") and "predictive_adapter" in self._optimizers:
            opt_path = os.path.join(model_dir, f"predictive_adapter{self._optimizer_suffix}_{step}.pt")
            if os.path.exists(opt_path):
                self._optimizers["predictive_adapter"].load_state_dict(torch.load(opt_path, map_location=self.device))
                print(f"[TransformerAgent] predictive_adapter optimizer loaded from {opt_path}")
            else:
                print(f"[TransformerAgent] No optimizer file: {opt_path}")

        return step


def _get_reverse_sorted_existing_versions(model_dir_path: Path, name: str) -> List[str]:
    """List of model components in reverse sorted order.

    Args:
        model_dir_path (Path): directory to find components in.
        name (str): name of the component.

    Returns:
        List[str]: list of model components in reverse sorted order.
    """
    existing_versions: List[str] = [str(x) for x in model_dir_path.glob(f"{name}_*.pt")]
    existing_versions = [
        x
        for x in existing_versions
        if is_integer(x.rsplit("/", 1)[-1].replace(f"{name}_", "").replace(".pt", ""))
    ]
    existing_versions.sort(reverse=True, key=_get_step_from_model_path)
    return existing_versions


def _get_step_from_model_path(_path: str) -> int:
    """Parse the model path to obtain the

    Args:
        _path (str): path to the model.

    Returns:
        int: step for tracking the training of the agent.
    """
    return int(_path.rsplit("/", 1)[-1].replace(".pt", "").rsplit("_", 1)[-1])

def _compress_meta_state(x: torch.Tensor) -> torch.Tensor:
    # x: [..., 39] or [..., 21]; idempotent processing on the last dimension
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    if x.size(-1) == 39:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, 39)
        x_flat = torch.cat([x_flat[:, :18], x_flat[:, 36:39]], dim=-1)
        x = x_flat.view(*original_shape, 21)
    if squeeze:
        x = x.squeeze(0)
    return x


@overload
def _load_component_or_optimizer(
    component_or_optimizer: ComponentType,
    model_dir: str,
    name: str,
    step: int,
) -> ComponentType:
    ...


@overload
def _load_component_or_optimizer(
    component_or_optimizer: OptimizerType,
    model_dir: str,
    name: str,
    step: int,
) -> OptimizerType:
    ...


def _load_component_or_optimizer(
    component_or_optimizer: ComponentOrOptimizerType,
    model_dir: str,
    name: str,
    step: int,
) -> ComponentOrOptimizerType:
    """Load a component/optimizer for the agent.

    Args:
        component_or_optimizer (ComponentOrOptimizerType): component or
            optimizer to load.
        model_dir (str): directory to load from.
        name (str): name of the component.
        step (int): step for tracking the training of the agent.

    Returns:
        ComponentOrOptimizerType: loaded component or
            optimizer.
    """

    assert component_or_optimizer is not None
    # if component_or_optimizer is not None:
    path_to_load_from = f"{model_dir}/{name}_{step}.pt"
    print(f"path_to_load_from: {path_to_load_from}")
    if os.path.exists(path_to_load_from):
        component_or_optimizer.load_state_dict(torch.load(path_to_load_from, weights_only=False))
    else:
        print(f"No component to load from {path_to_load_from}")
    return component_or_optimizer        

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import math
import argparse
from dataset_tf import TFDataset
import os
from InfoNceLoss import InfoNCE
from torch.utils.tensorboard import SummaryWriter

import bnpy
from bnpy.data.XData import XData
from itertools import cycle
from matplotlib import pylab
import shutil


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TrainablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10):
        """
        Initialize the Trainable Positional Encoding layer.

        Args:
        - max_seq_len (int): The maximum sequence length.
        - d_model (int): The dimension of the model (the size of the embeddings).
        """
        super(TrainablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a learnable parameter for positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Trainable Positional Encoding layer.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        - torch.Tensor: Output tensor with added positional encoding of shape (batch_size, seq_len, d_model).
        """
        # Add the positional encoding up to the sequence length
        x = x + self.positional_encoding[:x.size(0), :]

        return self.dropout(x)


class EncoderOnlyTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 d_model,
                 nhead,
                 dim_feedforward,
                 nlayers,
                 sequence_len,
                 device,
                 dropout=0.25):
        super(EncoderOnlyTransformerModel, self).__init__()
        self.pos_encoder = TrainablePositionalEncoding(d_model, dropout, sequence_len)

        self.state_emb = nn.Linear(state_dim, d_model)
        self.action_emb = nn.Linear(action_dim, d_model)
        self.reward_emb = nn.Linear(1, d_model)

        self.state_norm = nn.LayerNorm(d_model)
        self.action_norm = nn.LayerNorm(d_model)
        self.reward_norm = nn.LayerNorm(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.device = device
        self.d_model = d_model

        #self.special_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.special_token = nn.Parameter(torch.full((1, 1, d_model), -2.))

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def forward(
        self,
        states_src,
        actions_src,
        rewards_src,
        src_mask=None,
    ):
        batch_size, seq_len = actions_src.shape[0], actions_src.shape[1]

        state_token_src = self.state_norm(
            self.state_emb(states_src))  # * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        action_token_src = self.action_norm(
            self.action_emb(actions_src))  # * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        reward_token_src = self.reward_norm(
            self.reward_emb(rewards_src))  # * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))

        state_token_src = self.pos_encoder(state_token_src.permute(1, 0, 2))
        reward_token_src = self.pos_encoder(reward_token_src.permute(1, 0, 2))
        action_token_src = self.pos_encoder(action_token_src.permute(1, 0, 2))

        special_token = self.special_token.expand(-1, batch_size, -1)

        src = torch.stack((action_token_src, reward_token_src, state_token_src[1:]),
                          dim=0).transpose(0, 1).reshape(3 * seq_len, -1,
                                                         self.d_model)  # shape: 30, batchsize, hidden size
        src = torch.cat((state_token_src[0][None], src, special_token), dim=0)

        output = self.transformer_encoder(src, mask=src_mask)
        return output


class PredictionHead(nn.Module):

    def __init__(self, encoding_dim, output_dim):
        super(PredictionHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, output_dim)

    def forward(self, encoding):
        return self.linear(encoding)


class ClassificationHead(nn.Module):

    def __init__(self, encoding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, num_classes)

    def forward(self, encoding):
        logits = self.linear(encoding)
        prob_vec = F.softmax(logits, dim=1)
        return prob_vec


class RobotHead(nn.Module):
    """Robot head for distinguishing different robot embodiments.
    Similar to task head but specifically for robot identification.
    """

    def __init__(self, encoding_dim, output_dim):
        super(RobotHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, output_dim)

    def forward(self, encoding):
        return self.linear(encoding)


class RobotClassificationHead(nn.Module):
    """Robot classification head for CE loss."""

    def __init__(self, encoding_dim, num_robots):
        super(RobotClassificationHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, num_robots)

    def forward(self, encoding):
        logits = self.linear(encoding)
        return logits  # Return logits for CE loss


class DynamicPredictionHead(nn.Module):
    """Dynamic prediction head for predicting next state st+1 = f(st, at, z).
    This implements implicit system identification by conditioning on the cls_token (z),
    which should encode physical parameters like mass, friction, etc.
    The cls_token gradient flows back to the encoder, forcing it to learn system dynamics.
    """

    def __init__(self, state_dim, action_dim, z_dim, hidden_dim=64):
        super(DynamicPredictionHead, self).__init__()
        # Combine state, action, and z (cls_token) to predict next state
        # Input: concat(state, action, z) where z is expanded to match sequence length
        self.fc1 = nn.Linear(state_dim + action_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)
        self.relu = nn.ReLU()

    def forward(self, state, action, z):
        """
        Args:
            state: [batch, seq_len, state_dim] - current states (unmasked, real physics)
            action: [batch, seq_len, action_dim] - actions taken
            z: [batch, z_dim] - cls_token encoding (from masked input for robustness)
        Returns:
            next_state_pred: [batch, seq_len, state_dim] - predicted next states
        """
        batch_size, seq_len, _ = state.shape
        
        # Expand z to match sequence length: [batch, z_dim] -> [batch, seq_len, z_dim]
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Concatenate state, action, and expanded z
        x = torch.cat([state, action, z_expanded], dim=-1)  # [batch, seq_len, state_dim + action_dim + z_dim]
        
        # Predict next state through MLP
        x = self.relu(self.fc1(x))  # [batch, seq_len, hidden_dim]
        next_state_pred = self.fc2(x)  # [batch, seq_len, state_dim]
        
        return next_state_pred


def universal_train_loop(encoder,
                        prediction_head_state,
                        prediction_head_action,
                        prediction_head_reward,
                        prediction_head_cls,
                        opt,
                        loss_fn,
                        contrastive_loss_fn,
                        data_loader,
                        robot_head=None,
                        robot_contrastive_loss_fn=None,
                        robot_loss_type='infonce',
                        robot_loss_weight=0.0,
                        use_decoupling=False,
                        decoupling_weight=0.0,
                        unique_robots=None,
                        dynamic_head=None,
                        use_dynamic_loss=False,
                        dynamic_loss_weight=0.0):
    """Universal training loop supporting all optional features.
    
    Core: Always runs encoder, prediction heads, masking, reconstruction loss, and task contrastive loss.
    Robot Head: Only calculates robot_loss and decouple_loss if robot_head is not None.
    Dynamic Loss: Only calculates dynamic_loss if dynamic_head is not None and use_dynamic_loss is True.
    """
    encoder.train()
    prediction_head_state.train()
    prediction_head_action.train()
    prediction_head_reward.train()
    prediction_head_cls.train()
    if robot_head is not None:
        robot_head.train()
    if dynamic_head is not None:
        dynamic_head.train()

    recon_loss = 0
    contra_loss = 0
    robot_loss_total = 0
    decouple_loss_total = 0
    dynamic_loss_total = 0
    counter = 0
    len_dataloader = len(data_loader)

    state_indices = torch.arange(0, sequence_len * 3, 3)
    action_indices = torch.arange(1, (sequence_len - 1) * 3, 3)
    reward_indices = torch.arange(2, (sequence_len - 1) * 3, 3)

    # Create efficient robot ID to index mapping for CE loss
    robot_id_to_idx = None
    robots_registry = None
    if robot_head is not None and robot_loss_type == 'ce' and unique_robots is not None:
        robot_id_to_idx = {int(robot_id): idx for idx, robot_id in enumerate(unique_robots)}
        robots_registry = torch.tensor(unique_robots, device=device)

    for batch in data_loader:
        # select feature for the input and output
        states, actions, _, rewards, task_ind, task_arm = batch
        states = states.to(device)
        actions = actions[:, :-1].to(device)
        rewards = rewards[:, :-1].to(device)
        task_ind = task_ind.to(device)
        task_arm = task_arm.to(device)

        state_inp, state_inp_mask, action_inp, action_inp_mask, reward_inp, reward_inp_mask = mask_tokens(
            states, actions, rewards)

        # forward pass trough the network
        encoding = encoder(state_inp, action_inp, reward_inp)
        encoding_state, encoding_action, encoding_reward, encoding_cls_token = encoding[state_indices], encoding[
            action_indices], encoding[reward_indices], encoding[-1]
        pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
        pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
        pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)
        cls_token = prediction_head_cls(encoding_cls_token)

        pred_state, state_tar = pred_state[state_inp_mask], states[state_inp_mask]
        pred_action, action_tar = pred_action[action_inp_mask], actions[action_inp_mask]
        pred_reward, reward_tar = pred_reward[reward_inp_mask], rewards[reward_inp_mask]

        # positive samples
        state_inp_pos, state_inp_mask_pos, action_inp_pos, action_inp_mask_pos, reward_inp_pos, reward_inp_mask_pos = mask_tokens(
            states, actions, rewards)

        # forward pass trough the network
        encoding_pos = encoder(state_inp_pos, action_inp_pos, reward_inp_pos)
        encoding_state_pos, encoding_action_pos, encoding_reward_pos, encoding_cls_token_pos = encoding_pos[
            state_indices], encoding_pos[action_indices], encoding_pos[reward_indices], encoding_pos[-1]
        pred_state_pos = prediction_head_state(encoding_state_pos).permute(1, 0, 2)
        pred_action_pos = prediction_head_action(encoding_action_pos).permute(1, 0, 2)
        pred_reward_pos = prediction_head_reward(encoding_reward_pos).permute(1, 0, 2)
        cls_token_pos = prediction_head_cls(encoding_cls_token_pos)

        pred_state_pos, state_tar_pos = pred_state_pos[state_inp_mask_pos], states[state_inp_mask_pos]
        pred_action_pos, action_tar_pos = pred_action_pos[action_inp_mask_pos], actions[action_inp_mask_pos]
        pred_reward_pos, reward_tar_pos = pred_reward_pos[reward_inp_mask_pos], rewards[reward_inp_mask_pos]

        # calculate the state-/action-/reward-loss independently
        rec_loss = loss_fn(pred_state, state_tar) + loss_fn(pred_action, action_tar) + loss_fn(pred_reward, reward_tar) + \
                     loss_fn(pred_state_pos, state_tar_pos) + loss_fn(pred_action_pos, action_tar_pos) + loss_fn(pred_reward_pos, reward_tar_pos)

        # Task contrastive loss
        if not use_own_contrastive_loss:
            contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos)
        elif use_task_arm:
            contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos, task_arm)
        else:
            contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos, task_ind)

        # Robot loss (only if robot_head is not None)
        robot_loss = torch.tensor(0.0).to(device)
        if robot_head is not None:
            robot_token = robot_head(encoding_cls_token)
            robot_token_pos = robot_head(encoding_cls_token_pos)
            
            if robot_loss_type == 'infonce':
                robot_loss = robot_loss_weight * robot_contrastive_loss_fn(robot_token, robot_token_pos, task_arm)
            elif robot_loss_type == 'ce':
                matches = (task_arm.unsqueeze(1) == robots_registry.unsqueeze(0))
                robot_indices = matches.long().argmax(dim=1)
                robot_loss = robot_loss_weight * (F.cross_entropy(robot_token, robot_indices) +
                                                  F.cross_entropy(robot_token_pos, robot_indices))

        # Decoupling loss (only if robot_head is not None and use_decoupling is True)
        decouple_loss = torch.tensor(0.0).to(device)
        if robot_head is not None and use_decoupling:
            robot_token = robot_head(encoding_cls_token)
            cls_norm = F.normalize(cls_token, p=2, dim=-1)
            robot_norm = F.normalize(robot_token, p=2, dim=-1)
            cosine_sim = torch.abs(torch.sum(cls_norm * robot_norm, dim=-1))
            decouple_loss = decoupling_weight * cosine_sim.mean()

        # Dynamic loss (only if dynamic_head is not None and use_dynamic_loss is True)
        dynamic_loss = torch.tensor(0.0).to(device)
        if dynamic_head is not None and use_dynamic_loss:
            current_states = states[:, :-1]
            current_actions = actions
            next_states_target = states[:, 1:]
            next_states_pred = dynamic_head(current_states, current_actions, cls_token)
            dynamic_loss = dynamic_loss_weight * loss_fn(next_states_pred, next_states_target)

        loss = rec_loss + contrastive_loss + robot_loss + decouple_loss + dynamic_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        recon_loss += rec_loss.detach().item()
        contra_loss += contrastive_loss.detach().item()
        robot_loss_total += robot_loss.detach().item()
        decouple_loss_total += decouple_loss.detach().item()
        dynamic_loss_total += dynamic_loss.detach().item()

        if counter % 500 == 0:
            print(f"[{counter}/{len_dataloader}]")
        counter += 1

    return (recon_loss / len(data_loader), contra_loss / len(data_loader), robot_loss_total / len(data_loader),
            decouple_loss_total / len(data_loader), dynamic_loss_total / len(data_loader))





def universal_validation_loop(encoder,
                             prediction_head_state,
                             prediction_head_action,
                             prediction_head_reward,
                             prediction_head_cls,
                             loss_fn,
                             contrastive_loss_fn,
                             data_loader,
                             robot_head=None,
                             robot_contrastive_loss_fn=None,
                             robot_loss_type='infonce',
                             robot_loss_weight=0.0,
                             use_decoupling=False,
                             decoupling_weight=0.0,
                             unique_robots=None,
                             dynamic_head=None,
                             use_dynamic_loss=False,
                             dynamic_loss_weight=0.0):
    """Universal validation loop supporting all optional features.
    
    Core: Always runs encoder, prediction heads, masking, reconstruction loss, and task contrastive loss.
    Robot Head: Only calculates robot_loss and decouple_loss if robot_head is not None.
    Dynamic Loss: Only calculates dynamic_loss if dynamic_head is not None and use_dynamic_loss is True.
    """
    encoder.eval()
    prediction_head_state.eval()
    prediction_head_action.eval()
    prediction_head_reward.eval()
    prediction_head_cls.eval()
    if robot_head is not None:
        robot_head.eval()
    if dynamic_head is not None:
        dynamic_head.eval()

    recon_loss = 0
    contra_loss = 0
    robot_loss_total = 0
    decouple_loss_total = 0
    dynamic_loss_total = 0

    state_indices = torch.arange(0, sequence_len * 3, 3)
    action_indices = torch.arange(1, (sequence_len - 1) * 3, 3)
    reward_indices = torch.arange(2, (sequence_len - 1) * 3, 3)

    # Create efficient robot ID to index mapping for CE loss
    robot_id_to_idx = None
    robots_registry = None
    if robot_head is not None and robot_loss_type == 'ce' and unique_robots is not None:
        robot_id_to_idx = {int(robot_id): idx for idx, robot_id in enumerate(unique_robots)}
        robots_registry = torch.tensor(unique_robots, device=device)

    with torch.no_grad():
        for batch in data_loader:
            # select feature for the input and output
            states, actions, _, rewards, task_ind, task_arm = batch
            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)
            task_ind = task_ind.to(device)
            task_arm = task_arm.to(device)

            state_inp, state_inp_mask, action_inp, action_inp_mask, reward_inp, reward_inp_mask = mask_tokens(
                states, actions, rewards)

            # forward pass trough the network
            encoding = encoder(state_inp, action_inp, reward_inp)
            encoding_state, encoding_action, encoding_reward, encoding_cls_token = encoding[state_indices], encoding[
                action_indices], encoding[reward_indices], encoding[-1]
            pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
            pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
            pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)
            cls_token = prediction_head_cls(encoding_cls_token)

            pred_state, state_tar = pred_state[state_inp_mask], states[state_inp_mask]
            pred_action, action_tar = pred_action[action_inp_mask], actions[action_inp_mask]
            pred_reward, reward_tar = pred_reward[reward_inp_mask], rewards[reward_inp_mask]

            # positive samples
            state_inp_pos, state_inp_mask_pos, action_inp_pos, action_inp_mask_pos, reward_inp_pos, reward_inp_mask_pos = mask_tokens(
                states, actions, rewards)

            # forward pass trough the network
            encoding_pos = encoder(state_inp_pos, action_inp_pos, reward_inp_pos)
            encoding_state_pos, encoding_action_pos, encoding_reward_pos, encoding_cls_token_pos = encoding_pos[
                state_indices], encoding_pos[action_indices], encoding_pos[reward_indices], encoding_pos[-1]
            pred_state_pos = prediction_head_state(encoding_state_pos).permute(1, 0, 2)
            pred_action_pos = prediction_head_action(encoding_action_pos).permute(1, 0, 2)
            pred_reward_pos = prediction_head_reward(encoding_reward_pos).permute(1, 0, 2)
            cls_token_pos = prediction_head_cls(encoding_cls_token_pos)

            pred_state_pos, state_tar_pos = pred_state_pos[state_inp_mask_pos], states[state_inp_mask_pos]
            pred_action_pos, action_tar_pos = pred_action_pos[action_inp_mask_pos], actions[action_inp_mask_pos]
            pred_reward_pos, reward_tar_pos = pred_reward_pos[reward_inp_mask_pos], rewards[reward_inp_mask_pos]

            # calculate the state-/action-/reward-loss independently
            rec_loss = loss_fn(pred_state, state_tar) + loss_fn(pred_action, action_tar) + loss_fn(pred_reward, reward_tar) + \
                     loss_fn(pred_state_pos, state_tar_pos) + loss_fn(pred_action_pos, action_tar_pos) + loss_fn(pred_reward_pos, reward_tar_pos)

            # Task contrastive loss
            if not use_own_contrastive_loss:
                contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos)
            elif use_task_arm:
                contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos, task_arm)
            else:
                contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(
                    query=cls_token, positive_key=cls_token_pos, task_ind=task_ind)

            # Robot loss (only if robot_head is not None)
            robot_loss = torch.tensor(0.0).to(device)
            if robot_head is not None:
                robot_token = robot_head(encoding_cls_token)
                robot_token_pos = robot_head(encoding_cls_token_pos)
                
                if robot_loss_type == 'infonce':
                    robot_loss = robot_loss_weight * robot_contrastive_loss_fn(robot_token, robot_token_pos, task_arm)
                elif robot_loss_type == 'ce':
                    robot_indices = torch.tensor([robot_id_to_idx[int(robot.item())] for robot in task_arm],
                                                 dtype=torch.long,
                                                 device=device)
                    robot_loss = robot_loss_weight * (F.cross_entropy(robot_token, robot_indices) +
                                                      F.cross_entropy(robot_token_pos, robot_indices))

            # Decoupling loss (only if robot_head is not None and use_decoupling is True)
            decouple_loss = torch.tensor(0.0).to(device)
            if robot_head is not None and use_decoupling:
                robot_token = robot_head(encoding_cls_token)
                cls_norm = F.normalize(cls_token, p=2, dim=-1)
                robot_norm = F.normalize(robot_token, p=2, dim=-1)
                cosine_sim = torch.abs(torch.sum(cls_norm * robot_norm, dim=-1))
                decouple_loss = decoupling_weight * cosine_sim.mean()

            # Dynamic loss (only if dynamic_head is not None and use_dynamic_loss is True)
            dynamic_loss = torch.tensor(0.0).to(device)
            if dynamic_head is not None and use_dynamic_loss:
                current_states = states[:, :-1]
                current_actions = actions
                next_states_target = states[:, 1:]
                next_states_pred = dynamic_head(current_states, current_actions, cls_token)
                dynamic_loss = dynamic_loss_weight * loss_fn(next_states_pred, next_states_target)

            recon_loss += rec_loss.detach().item()
            contra_loss += contrastive_loss.detach().item()
            robot_loss_total += robot_loss.detach().item()
            decouple_loss_total += decouple_loss.detach().item()
            dynamic_loss_total += dynamic_loss.detach().item()

    return (recon_loss / len(data_loader), contra_loss / len(data_loader), robot_loss_total / len(data_loader),
            decouple_loss_total / len(data_loader), dynamic_loss_total / len(data_loader))


def predict(encoder,
            prediction_head_state,
            prediction_head_action,
            prediction_head_reward,
            data_loader,
            sample_count=5):
    encoder.eval()
    prediction_head_state.eval()
    prediction_head_action.eval()
    prediction_head_reward.eval()
    total_loss = 0
    data_loader = iter(data_loader)

    state_indices = torch.arange(0, sequence_len * 3, 3)
    action_indices = torch.arange(1, (sequence_len - 1) * 3, 3)
    reward_indices = torch.arange(2, (sequence_len - 1) * 3, 3)

    with torch.no_grad():
        for i in range(sample_count):
            sample = next(data_loader)
            states, actions, _, rewards, _, _ = sample
            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            encoding = encoder(states, actions, rewards)
            encoding_state, encoding_action, encoding_reward = encoding[state_indices], encoding[
                action_indices], encoding[reward_indices]
            pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
            pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
            pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)

            state_loss = loss_fn(pred_state, states)
            action_loss = loss_fn(pred_action, actions)
            reward_loss = loss_fn(pred_reward, rewards)
            loss = state_loss + action_loss + reward_loss

            # NO contrastive loss since we only look at single sample

            total_loss += loss.detach().item()

            print(f"States predicition: \n{pred_state}")
            print(f"States target: \n{states}")
            print(f"State loss: {state_loss}")
            print(f"Action predicition: \n{pred_action}")
            print(f"Action target: \n{actions}")
            print(f"Action loss: {action_loss}")
            print(f"Reward predicition: \n{pred_reward}")
            print(f"Reward target: \n{rewards}")
            print(f"Reward loss: {reward_loss}")
            print("-" * 25 + f" Sample {i} " + "-" * 25)

    return total_loss / sample_count


def predict_last_state(encoder, prediction_head_state, data_loader):
    encoder.eval()
    prediction_head_state.eval()
    total_loss = 0
    data_loader = iter(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            states, actions, _, rewards, _, _ = batch
            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            state_inp, _ = mask_last_state(states)

            encoding = encoder(state_inp, actions, rewards)
            encoding_state = encoding[-1]
            pred_state = prediction_head_state(encoding_state)

            state_loss = loss_fn(pred_state, states[:, -1])
            loss = state_loss

            # NO contrastive loss since we only look at single sample

            total_loss += loss.detach().item()

    return total_loss / len(data_loader)


def embedding_valdation(encoder, prediction_head_cls, samples_num, data_loader, save_path):
    encoder.eval()
    prediction_head_cls.eval()

    counter = 0
    z_arr, t_arr, task_arr, states_arr, action_arr, reward_arr, task_arm_arr, summed_reward_arr = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            states, actions, _, rewards, task_obs, task_arm = batch

            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            states_arr.append(states)
            action_arr.append(actions)
            reward_arr.append(rewards)

            counter += states.shape[0]

            task_arr.append(task_obs)
            task_arm_arr.append(task_arm)

            summed_reward_arr.append(torch.sum(rewards, dim=1))

            encoding = encoder(states, actions, rewards)
            encoding = encoding.permute(1, 0, 2)

            state_embedding = encoding[:, -2]

            if cluster_state:
                trajectory_embedding = encoding[:, -2]
            else:
                trajectory_embedding = prediction_head_cls(encoding[:, -1])
            #trajectory_embedding = encoding[:, -1]

            t_arr.append(trajectory_embedding)
            z_arr.append(state_embedding)

            if counter > samples_num:
                break
    task = torch.cat(task_arr).cpu().detach().numpy()
    task_arm = torch.cat(task_arm_arr).cpu().detach().numpy()
    rewards = torch.cat(summed_reward_arr).cpu().detach().numpy()
    z = torch.cat(z_arr).cpu().detach().numpy()
    t = torch.cat(t_arr).cpu().detach().numpy()
    obs_state = torch.cat(states_arr).cpu().detach().numpy()
    obs_actions = torch.cat(action_arr).cpu().detach().numpy()
    obs_rewards = torch.cat(reward_arr).cpu().detach().numpy()
    print(f"Saving embeddings {save_path}...")
    torch.save(
        {
            "task": task,
            "tra_emb": t,
            "state_emb": z,
            "rewards": rewards,
            "task_arm": task_arm,
            "states": obs_state,
            "actions": obs_actions,
            "rew": obs_rewards
        }, save_path)

    print("saving cluster assignment")
    manage_latent_representation(torch.cat(t_arr), torch.cat(task_arr), 'end')


def trajectory_embedding_valdation(encoder, prediction_head_cls, samples_num, dataset, save_path):
    encoder.eval()
    prediction_head_cls.eval()

    avaible_length = loaded_dataset.avaible_length
    counter = 0
    z_arr, t_arr, task_arr, states_arr, action_arr, reward_arr, summed_reward_arr, task_arm_arr, time_step_arr = [], [], [], [], [], [], [], [], []
    unique_env_idx = np.unique(dataset.task_obs)
    with torch.no_grad():
        while counter < samples_num:
            unique_tasks = np.unique(dataset.task_obs)
            unique_arms = np.unique(dataset.task_arm)
            sample_per_task = []
            for t in unique_tasks:
                for a in unique_arms:
                    mask = (dataset.task_obs == t) & (dataset.task_arm == a)
                    candidates = np.arange(len(dataset.task_obs))[mask]
                    if candidates.size == 0:
                        continue
                    pick = np.random.choice(candidates, size=1, replace=False)[0]
                    windows = np.arange(avaible_length) + pick * avaible_length
                    sample_per_task.append(windows)

            sample_per_task = np.concatenate(sample_per_task)
            batch = [dataset[idx] for idx in sample_per_task]

            # Now you can convert your batch into tensors, or further process it as needed
            states = torch.tensor(np.stack([item[0] for item in batch]))
            actions = torch.tensor(np.stack([item[1] for item in batch]))
            rewards = torch.tensor(np.stack([item[3] for item in batch]))
            task_obs = torch.tensor(np.stack([item[4] for item in batch]))
            task_arm = torch.tensor(np.stack([item[5] for item in batch]))

            counter += avaible_length * len(unique_env_idx)

            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            states_arr.append(states)
            action_arr.append(actions)
            reward_arr.append(rewards)
            time_step_arr.append(np.tile(np.arange(avaible_length), len(unique_env_idx)))

            task_arr.append(task_obs)
            task_arm_arr.append(task_arm)

            summed_reward_arr.append(torch.sum(rewards, dim=1))

            encoding = encoder(states, actions, rewards)
            encoding = encoding.permute(1, 0, 2)

            state_embedding = encoding[:, -2]

            if cluster_state:
                trajectory_embedding = encoding[:, -2]
            else:
                trajectory_embedding = prediction_head_cls(encoding[:, -1])

            t_arr.append(trajectory_embedding)
            z_arr.append(state_embedding)

    task = torch.cat(task_arr).cpu().detach().numpy()
    task_arm = torch.cat(task_arm_arr).cpu().detach().numpy()
    rewards = torch.cat(summed_reward_arr).cpu().detach().numpy()
    z = torch.cat(z_arr).cpu().detach().numpy()
    t = torch.cat(t_arr).cpu().detach().numpy()
    obs_state = torch.cat(states_arr).cpu().detach().numpy()
    obs_actions = torch.cat(action_arr).cpu().detach().numpy()
    obs_rewards = torch.cat(reward_arr).cpu().detach().numpy()
    time_steps = np.concatenate(time_step_arr)
    print("Saving embeddings...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "task": task,
            "tra_emb": t,
            "state_emb": z,
            "rewards": rewards,
            "task_arm": task_arm,
            "states": obs_state,
            "actions": obs_actions,
            "rew": obs_rewards,
            "timesteps": time_steps
        }, save_path)

    print("saving cluster assignment")
    manage_latent_representation(torch.cat(t_arr), torch.cat(task_arr), 'end')


# ------ bnpy ----------
def bnpy_cluster(encoder, prediction_head_cls, data_loader):
    encoder.eval()
    prediction_head_cls.eval()

    if not os.path.exists(os.path.join(bnpy_save_dir, 'birth_debug')):
        os.makedirs(os.path.join(bnpy_save_dir, 'birth_debug'))
    if not os.path.exists(os.path.join(bnpy_save_dir, 'data')):
        os.makedirs(os.path.join(bnpy_save_dir, 'data'))

    for i in range(bnpy_epochs):
        counter = 0
        z_arr, task_arr = [], []
        with torch.no_grad():
            for batch in data_loader:
                states, actions, _, rewards, task_obs, _ = batch

                states = states.to(device)
                actions = actions[:, :-1].to(device)
                rewards = rewards[:, :-1].to(device)

                counter += states.shape[0]

                task_arr.append(task_obs)

                encoding = encoder(states, actions, rewards)
                encoding = encoding.permute(1, 0, 2)

                if cluster_state:
                    trajectory_embedding = encoding[:, -2]
                else:
                    trajectory_embedding = prediction_head_cls(encoding[:, -1])

                z_arr.append(trajectory_embedding)

                if counter > sample_per_iter:
                    break
        task = torch.cat(task_arr)
        z = torch.cat(z_arr)

        print(f"--- Fitting bnpy at step [{i}] ---")
        out_path = os.path.join(bnpy_save_dir, str(next(iterator)))
        fit(z, out_path)

    manage_latent_representation(z, task, 'end')
    print("Saving model ...")
    save_bnpy_model(bnpy_save_dir, out_path + '/1')


def fit(z, out_path):
    '''
        fit the model, input z should be torch.tensor format
        '''
    global bnpy_model
    global info_dict

    z = XData(z.detach().cpu().numpy())
    if not bnpy_model:
        print("=== Initialing DPMM model ===")
        bnpy_model, info_dict = bnpy.run(
            z,
            'DPMixtureModel',
            'DiagGauss',
            'memoVB',
            output_path=out_path,
            #output_path=bnpy_save_dir,
            initname='randexamples',
            K=1,
            gamma0=gamma0,
            sF=sF,
            ECovMat='eye',
            moves='birth,merge',
            nBatch=4,
            nLap=num_lap,
            **dict(sum(map(list, [birth_kwargs.items(), merge_kwargs.items()]), [])))
        print("=== DPMM model initialized ===")
    else:
        bnpy_model, info_dict = bnpy.run(
            z,
            'DPMixtureModel',
            'DiagGauss',
            'memoVB',
            output_path=out_path,
            #output_path=bnpy_save_dir,
            initname=info_dict['task_output_path'],
            K=info_dict['K_history'][-1],
            gamma0=gamma0,
            sF=sF,
            ECovMat='eye',
            moves='birth,merge',
            nBatch=4,
            nLap=num_lap,
            **dict(sum(map(list, [birth_kwargs.items(), merge_kwargs.items()]), [])))
    calc_cluster_component_params()


def calc_cluster_component_params():
    global comp_mu
    global comp_var
    comp_mu = [torch.Tensor(bnpy_model.obsModel.get_mean_for_comp(i)) for i in np.arange(0, bnpy_model.obsModel.K)]
    comp_var = [
        torch.Tensor(np.sum(bnpy_model.obsModel.get_covar_mat_for_comp(i), axis=0))
        for i in np.arange(0, bnpy_model.obsModel.K)
    ]


def cluster_assignments(z):
    z = XData(z.detach().cpu().numpy())
    LP = bnpy_model.calc_local_params(z)
    # Here, resp is a 2D array of size N x K.
    # Each entry resp[n, k] gives the probability
    # that data atom n is assigned to cluster k under
    # the posterior.
    resp = LP['resp']
    # To convert to hard assignments
    # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
    Z = resp.argmax(axis=1)
    return resp, Z


def cluster_assignments_with_comp_para(z):
    z = XData(z.detach().cpu().numpy())
    LP = bnpy_model.calc_local_params(z)
    Z = LP['resp'].argmax(axis=1)

    comp_mu = [torch.Tensor(bnpy_model.obsModel.get_mean_for_comp(i)) for i in np.arange(0, bnpy_model.obsModel.K)]

    m = torch.stack([comp_mu[x] for x in Z])
    return m, Z


def manage_latent_representation(z, env_idx, prefix: str = ''):
    '''
    calculate the latent z and their corresponding clusters, save the values for later evaluation
    saved value:
        z: latent encoding from VAE Encoder
        env_idx: corresponding env idx of each z (true label)
        env_name: corresponding env name of each z
        cluster_label: corresponding cluster No. that z should be assigned to
        cluster_param: corresponding cluster parameters (mu & var of diagGauss) of each z
    '''

    comps_mu_list = []
    comps_var_list = []
    # cluster label
    _, cluster_label = cluster_assignments(z)
    # get clusters param
    comp_mu = [bnpy_model.obsModel.get_mean_for_comp(i) for i in np.arange(0, bnpy_model.obsModel.K)]
    comp_var = [
        np.sum(bnpy_model.obsModel.get_covar_mat_for_comp(i), axis=0) for i in np.arange(0, bnpy_model.obsModel.K)
    ]
    for i in cluster_label:
        comps_mu_list.append(comp_mu[i])
        comps_var_list.append(comp_var[i])
    # summarize data into dict
    data = dict(
        z=z.detach().cpu().numpy(),
        env_idx=env_idx.detach().cpu().numpy(),
        cluster_label=cluster_label,
        cluster_mu=comps_mu_list,
        cluster_var=comps_var_list,
    )
    # save the file
    np.savez(bnpy_save_dir + '/data' + '/latent_samples_{}.npz'.format(prefix), **data)


def load_bnpy_model(bnpy_load_path):
    global bnpy_model
    global info_dict
    if os.path.exists(bnpy_load_path) and os.path.isdir(bnpy_load_path):
        bnpy_model = bnpy.load_model_at_lap(bnpy_load_path, 100)[0]
        info_dict = np.load(bnpy_load_path + '/info_dict.npy', allow_pickle=True).item()
    else:
        print(f"bnpy model at {bnpy_load_path} not found!!!")


def save_info_dict(save_path, new_output_path=None):
    if new_output_path:
        info = dict(
            task_output_path=new_output_path,
            K_history=[info_dict['K_history'][-1]],
        )
    else:
        info = dict(
            task_output_path=info_dict['task_output_path'],
            K_history=[info_dict['K_history'][-1]],
        )
    np.save(save_path + '/info_dict.npy', info)


def save_bnpy_model(save_path, current_dict):
    if os.path.exists(current_dict) and os.path.isdir(current_dict):
        if os.path.exists(save_path + '/save'):
            shutil.rmtree(save_path + '/save')
        print(f"Folder at {current_dict} already exists")
    shutil.copytree(current_dict, save_path + '/save')
    save_info_dict(save_path + '/save', save_path + '/save')
    save_comps_parameters(save_path + '/data')


def save_comps_parameters(save_path):
    '''
    save the model param in form of *npy
    '''
    comp_mu = [bnpy_model.obsModel.get_mean_for_comp(i) for i in np.arange(0, bnpy_model.obsModel.K)]
    comp_var = [
        np.sum(bnpy_model.obsModel.get_covar_mat_for_comp(i), axis=0)  # save diag value in a 1-dim array
        for i in np.arange(0, bnpy_model.obsModel.K)
    ]
    data = dict(comp_mu=comp_mu, comp_var=comp_var)
    np.save(save_path + '/comp_params.npy', data)


# ------ bnpy ----------


def mask_tokens(input_state, input_action, input_reward, mask_prob=0.15, unchanged_prob=0.10, random_prob=0.10):
    state_feature = input_state.shape[-1]
    action_feature = input_action.shape[-1]

    masked_state_tensor = input_state.clone().to(device)
    masked_action_tensor = input_action.clone().to(device)
    masked_reward_tensor = input_reward.clone().to(device)

    mask_state = torch.rand((masked_state_tensor.shape[:2]), device=device) < mask_prob
    mask_action = torch.rand((masked_action_tensor.shape[:2]), device=device) < mask_prob
    mask_reward = torch.rand((masked_reward_tensor.shape[:2]), device=device) < mask_prob

    ensure_masking = torch.zeros(mask_state.shape[1], dtype=torch.bool).to(device)
    ensure_masking[-1] = True
    mask_state[(mask_state == False).all(dim=1)] = ensure_masking
    ensure_masking = torch.zeros(mask_action.shape[1], dtype=torch.bool).to(device)
    ensure_masking[-1] = True
    mask_action[(mask_action == False).all(dim=1)] = ensure_masking
    mask_reward[(mask_reward == False).all(dim=1)] = ensure_masking

    probability_sample_state = torch.rand(masked_state_tensor.shape[:2], device=device)
    probability_sample_action = torch.rand(masked_action_tensor.shape[:2], device=device)
    probability_sample_reward = torch.rand(masked_reward_tensor.shape[:2], device=device)
    random_mask_state = probability_sample_state < random_prob
    unchanged_mask_state = probability_sample_state > (random_prob + unchanged_prob)
    random_mask_action = probability_sample_action < random_prob
    unchanged_mask_action = probability_sample_action > (random_prob + unchanged_prob)
    random_mask_reward = probability_sample_reward < random_prob
    unchanged_mask_reward = probability_sample_reward > (random_prob + unchanged_prob)

    masked_state_tensor[unchanged_mask_state & mask_state] = torch.zeros(state_feature, device=device)
    masked_state_tensor[random_mask_state & mask_state] = torch.rand(state_feature, device=device)
    masked_action_tensor[unchanged_mask_action & mask_action] = torch.zeros(action_feature, device=device)
    masked_action_tensor[random_mask_action & mask_action] = torch.rand(action_feature, device=device)
    masked_reward_tensor[unchanged_mask_reward & mask_reward] = torch.zeros(1, device=device)
    masked_reward_tensor[random_mask_reward & mask_reward] = torch.rand(1, device=device)

    return masked_state_tensor, mask_state, masked_action_tensor, mask_action, masked_reward_tensor, mask_reward


def mask_last_state(input_state):
    state_feature = input_state.shape[-1]

    masked_state_tensor = input_state.clone().to(device)

    mask_state = torch.zeros(masked_state_tensor.shape[:2], dtype=torch.bool)
    mask_state[:, -1] = True

    masked_state_tensor[mask_state] = torch.zeros(state_feature, device=device)

    return masked_state_tensor, mask_state


def euclidean_distance(vec1, vec2):
    # Calculate the Euclidean distance
    distance = torch.sqrt(torch.sum((vec1 - vec2)**2))

    return distance.item()


def save_model(model,
               prediciton_head_state,
               prediciton_head_action,
               prediciton_head_reward,
               prediction_head_cls,
               model_path,
               robot_head=None,
               dynamic_head=None):
    """Saves the model and decoder state dictionaries to the specified path."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'prediciton_head_state': prediciton_head_state.state_dict(),
        'prediciton_head_action': prediciton_head_action.state_dict(),
        'prediciton_head_reward': prediciton_head_reward.state_dict(),
        'prediction_head_cls': prediction_head_cls.state_dict(),
    }
    if robot_head is not None:
        checkpoint['robot_head'] = robot_head.state_dict()
    if dynamic_head is not None:
        checkpoint['dynamic_head'] = dynamic_head.state_dict()
    torch.save(checkpoint, model_path)
    print(f"Model and decoder saved to {model_path}")


def load_model(model,
               prediciton_head_state,
               prediciton_head_action,
               prediciton_head_reward,
               prediction_head_cls,
               model_path,
               robot_head=None,
               dynamic_head=None):
    """Loads the model and decoder state dictionaries from the specified path."""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        prediciton_head_state.load_state_dict(checkpoint['prediciton_head_state'])
        prediciton_head_action.load_state_dict(checkpoint['prediciton_head_action'])
        prediciton_head_reward.load_state_dict(checkpoint['prediciton_head_reward'])
        prediction_head_cls.load_state_dict(checkpoint['prediction_head_cls'])
        if robot_head is not None and 'robot_head' in checkpoint:
            robot_head.load_state_dict(checkpoint['robot_head'])
            print(f"Model, decoder, and robot head loaded from {model_path}")
        else:
            print(f"Model and decoder loaded from {model_path}")
        if dynamic_head is not None and 'dynamic_head' in checkpoint:
            dynamic_head.load_state_dict(checkpoint['dynamic_head'])
            print(f"Dynamic head loaded from {model_path}")
    else:
        print(f"No checkpoint found at {model_path}. Initializing model from scratch.")


def set_seed(seed):
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


########################### Hyperparameter ###########################

state_dim = 21
action_dim = 4
cls_dim = 6

d_model = 32  #20 32 64 | Model dimension
nhead = 8  # 4 8 | Number of attention heads
num_layers = 3  # Number of transformer encoder layers
dim_feedforward = 128  #128 #256
dropout = 0.1
mask_prob = 0.2
mask_unchanged = 0.15
random_mask_prob = 0.1

project_root = os.environ.get("PROJECT_ROOT")
bnpy_save_dir = project_root + '/Transformer_RNN/bnpy_save'

gamma0 = 5.0
num_lap = 20
sF = 1.  #0.1
bnpy_model = None
info_dict = None
iterator = cycle(range(2))
comp_mu = None
comp_var = None
sample_per_iter = 10_000
birth_kwargs = {
    'b_startLap': 1,
    'b_stopLap': 2,
    'b_Kfresh': 10,
    'b_minNumAtomsForNewComp': 10.0,
    'b_minNumAtomsForTargetComp': 10.0,
    'b_minNumAtomsForRetainComp': 10.0,
    'b_minPercChangeInNumAtomsToReactivate': 0.03,
    'b_debugWriteHTML': 0
}
merge_kwargs = {
    'm_startLap': 2,
    'm_maxNumPairsContainingComp': 50,
    'm_nLapToReactivate': 1,
    'm_pair_ranking_procedure': 'obsmodel_elbo',
    'm_pair_ranking_direction': 'descending'
}

batch_size = 256
learning_rate = 0.0002
epochs = 0
sequence_len = 20
predict_samples_num = 0
bnpy_epochs = 1

contrastive_temp = 0.15  #0.25
contrastive_loss_weight = 0.25
use_own_contrastive_loss = True
use_task_arm = False
cluster_state = False

# Robot head configuration
use_robot_head = False  # Enable robot head for multi-robot learning
robot_loss_type = 'infonce'  # 'infonce' or 'ce' (cross-entropy)
robot_loss_weight = 0.25  # Weight for robot head loss
use_decoupling_loss = False  # Enable task-robot subspace decoupling
decoupling_loss_weight = 0.1  # Weight for decoupling loss

# Dynamic loss configuration for future state prediction
use_dynamic_loss = True  # Enable dynamic loss to predict next state (predictive adapter style)
dynamic_loss_weight = 200  # Weight for dynamic loss
dynamic_hidden_dim = 128  # Hidden dimension for dynamic prediction head

should_continue = True
should_safe = True
evaluate_embedding = True
should_continue_bnpy = False
train_bnpy = True
should_predict_last_state = True

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 5

embedding_num = 5000
model_path = 'Transformer_RNN/checkpoints_task_dyn_true_seed_3/representation_cls_transformer_checkpoint.pth'
base_dir = 'Transformer_RNN/decision_tf_dataset/train/'
dataset_path = os.path.join(base_dir, "data")
#dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_envs/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_faucet/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/eval_embodiments/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/kuka_saywer/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/expert_10/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/distill+kuka+saywer/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/distill+saywer/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/expert_30/'

val_base_dir = 'Transformer_RNN/decision_tf_dataset/validation/'
val_dataset_path = os.path.join(val_base_dir, "data")
#val_dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_faucet/'
#val_dataset_path = 'Transformer_RNN/decision_tf_dataset/eval_embodiments/'

embeddings_path = 'Transformer_RNN/embedding_log/emb.pth'
log_path = 'Transformer_RNN/tensorboard_log'

######################################################################

import torch, numpy as np, builtins


def log_blue(msg):
    print(f"\033[34m{msg}\033[0m")


log_blue("[SAVE] Replay buffer to ./save/path")

_orig_save = torch.save


def save_with_log(obj, f, *args, **kwargs):
    log_blue(f"[TORCH SAVE] → {f}")
    return _orig_save(obj, f, *args, **kwargs)


torch.save = save_with_log

_orig_load = torch.load


def load_with_log(f, *args, **kwargs):
    log_blue(f"[TORCH LOAD] ← {f}")
    return _orig_load(f, *args, **kwargs)


torch.load = load_with_log

# Similarly for np.save / np.load
np_save = np.save
np_load = np.load


def logged_np_save(file, arr, *args, **kwargs):
    log_blue(f"[NP SAVE] → {file}")
    return np_save(file, arr, *args, **kwargs)


np.save = logged_np_save


def logged_np_load(file, *args, **kwargs):
    log_blue(f"[NP LOAD] ← {file}")
    return np_load(file, *args, **kwargs)


np.load = logged_np_load

######################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the representation transformer used by PACE.")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--sequence-len", type=int, default=sequence_len)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--model-path", default=model_path)
    parser.add_argument("--train-dataset-path", default=dataset_path)
    parser.add_argument("--val-dataset-path", default=val_dataset_path)
    parser.add_argument("--embeddings-path", default=embeddings_path)
    parser.add_argument("--log-path", default=log_path)
    parser.add_argument("--predict-samples-num", type=int, default=predict_samples_num)
    parser.add_argument("--embedding-num", type=int, default=embedding_num)
    parser.add_argument("--bnpy-epochs", type=int, default=bnpy_epochs)
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore any existing checkpoint and train from scratch.",
    )
    parser.add_argument(
        "--skip-bnpy",
        action="store_true",
        help="Skip bnpy clustering after training.",
    )
    parser.add_argument(
        "--skip-embedding-eval",
        action="store_true",
        help="Skip embedding export/evaluation after training.",
    )
    parser.add_argument(
        "--skip-last-state-eval",
        action="store_true",
        help="Skip final last-state prediction evaluation.",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    sequence_len = args.sequence_len
    predict_samples_num = args.predict_samples_num
    bnpy_epochs = args.bnpy_epochs
    seed = args.seed
    embedding_num = args.embedding_num
    model_path = args.model_path
    dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    embeddings_path = args.embeddings_path
    log_path = args.log_path
    should_continue = should_continue and (not args.fresh_start)
    train_bnpy = train_bnpy and (not args.skip_bnpy)
    evaluate_embedding = evaluate_embedding and (not args.skip_embedding_eval)
    should_predict_last_state = should_predict_last_state and (not args.skip_last_state_eval)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    # Set initial seed for reproducibility
    set_seed(seed)

    # Load the dataset
    loaded_dataset = TFDataset.load(dataset_path, sequence_length=sequence_len)  # here one more
    loaded_val_dataset = TFDataset.load(val_dataset_path, sequence_length=sequence_len)
    total_size = len(loaded_val_dataset)
    val_size = int(0.3 * total_size)
    not_used_size = 0  #int(0.5 * total_size)
    test_size = total_size - val_size - not_used_size
    unique_labels = torch.tensor(loaded_dataset.unique_task_obs)

    # create Dataloader
    val_dataset, test_dataset, _ = random_split(loaded_val_dataset, [val_size, test_size, not_used_size])
    optimal_workers = 8

    val_dataset, test_dataset, _ = random_split(loaded_val_dataset, [val_size, test_size, not_used_size])

    train_loader = DataLoader(
        loaded_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    single_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create the Encoder and simple MLP
    encoder = EncoderOnlyTransformerModel(state_dim=state_dim,
                                          action_dim=action_dim,
                                          d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          nlayers=num_layers,
                                          sequence_len=sequence_len,
                                          device=device,
                                          dropout=dropout).to(device)
    prediction_head_state = PredictionHead(d_model, state_dim).to(device)
    prediction_head_action = PredictionHead(d_model, action_dim).to(device)
    prediction_head_reward = PredictionHead(d_model, 1).to(device)
    prediction_head_cls = PredictionHead(d_model, cls_dim).to(device)
    classification_head = ClassificationHead(d_model, len(unique_labels)).to(device)

    # Create robot head if enabled
    robot_head = None
    unique_robots = None
    if use_robot_head:
        # Get number of unique robots from dataset
        unique_robots = np.unique(loaded_dataset.task_arm)
        num_robots = len(unique_robots)
        print(f"Detected {num_robots} unique robots: {unique_robots}")

        if robot_loss_type == 'infonce':
            # For InfoNCE, output dimension same as cls_dim
            robot_head = RobotHead(d_model, cls_dim).to(device)
        else:  # 'ce'
            # For CE loss, output dimension is number of robot classes
            robot_head = RobotClassificationHead(d_model, num_robots).to(device)

    # Create dynamic head if enabled
    dynamic_head = None
    if use_dynamic_loss:
        # Initialize with correct dimensions: state_dim, action_dim, z_dim (cls_dim)
        dynamic_head = DynamicPredictionHead(state_dim, action_dim, cls_dim, hidden_dim=dynamic_hidden_dim).to(device)
        print(f"Dynamic prediction head enabled with weight: {dynamic_loss_weight}")
        print(f"  Input: state_dim={state_dim}, action_dim={action_dim}, z_dim={cls_dim}")
        print(f"  Hidden dim: {dynamic_hidden_dim}")

    if should_continue:
        load_model(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls,
                   model_path, robot_head, dynamic_head)
    if should_continue_bnpy:
        load_bnpy_model(bnpy_save_dir + '/save')

    # Creating optimizer and loss function
    params_to_optimize = list(encoder.parameters()) + list(prediction_head_state.parameters()) + \
                         list(prediction_head_action.parameters()) + list(prediction_head_reward.parameters()) + \
                         list(prediction_head_cls.parameters()) + list(classification_head.parameters())
    if robot_head is not None:
        params_to_optimize += list(robot_head.parameters())
    if dynamic_head is not None:
        params_to_optimize += list(dynamic_head.parameters())

    opt = torch.optim.Adam(params_to_optimize, lr=learning_rate)
    loss_fn = nn.MSELoss(reduction='mean')
    contrastive_loss = InfoNCE(temperature=contrastive_temp)
    robot_contrastive_loss = InfoNCE(temperature=contrastive_temp) if robot_loss_type == 'infonce' else None

    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir=log_path)

    # Training loop
    print("Training and validating model")
    print(f"Robot head enabled: {use_robot_head}")
    if use_robot_head:
        print(f"Robot loss type: {robot_loss_type}, weight: {robot_loss_weight}")
        print(f"Decoupling enabled: {use_decoupling_loss}, weight: {decoupling_loss_weight}")
    print(f"Dynamic loss enabled: {use_dynamic_loss}, weight: {dynamic_loss_weight}")

    start_time = time.time()
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        # Use universal training loop with conditional parameters
        rec_loss, con_loss, robot_loss, decouple_loss, dynamic_loss = universal_train_loop(
            encoder,
            prediction_head_state,
            prediction_head_action,
            prediction_head_reward,
            prediction_head_cls,
            opt,
            loss_fn,
            contrastive_loss,
            train_loader,
            robot_head=robot_head if use_robot_head else None,
            robot_contrastive_loss_fn=robot_contrastive_loss,
            robot_loss_type=robot_loss_type,
            robot_loss_weight=robot_loss_weight if use_robot_head else 0.0,
            use_decoupling=use_decoupling_loss,
            decoupling_weight=decoupling_loss_weight if use_decoupling_loss else 0.0,
            unique_robots=unique_robots,
            dynamic_head=dynamic_head if use_dynamic_loss else None,
            use_dynamic_loss=use_dynamic_loss,
            dynamic_loss_weight=dynamic_loss_weight if use_dynamic_loss else 0.0)
        
        writer.add_scalar('Loss/train_rec', rec_loss, epoch)
        writer.add_scalar('Loss/train_cont', con_loss, epoch)
        writer.add_scalar('Loss/train_robot', robot_loss, epoch)
        writer.add_scalar('Loss/train_decouple', decouple_loss, epoch)
        writer.add_scalar('Loss/train_dynamic', dynamic_loss, epoch)

        val_rec_loss, val_con_loss, val_robot_loss, val_decouple_loss, val_dynamic_loss = universal_validation_loop(
            encoder,
            prediction_head_state,
            prediction_head_action,
            prediction_head_reward,
            prediction_head_cls,
            loss_fn,
            contrastive_loss,
            val_loader,
            robot_head=robot_head if use_robot_head else None,
            robot_contrastive_loss_fn=robot_contrastive_loss,
            robot_loss_type=robot_loss_type,
            robot_loss_weight=robot_loss_weight if use_robot_head else 0.0,
            use_decoupling=use_decoupling_loss,
            decoupling_weight=decoupling_loss_weight if use_decoupling_loss else 0.0,
            unique_robots=unique_robots,
            dynamic_head=dynamic_head if use_dynamic_loss else None,
            use_dynamic_loss=use_dynamic_loss,
            dynamic_loss_weight=dynamic_loss_weight if use_dynamic_loss else 0.0)
        
        writer.add_scalar('Loss/validation_rec', val_rec_loss, epoch)
        writer.add_scalar('Loss/validation_cont', val_con_loss, epoch)
        writer.add_scalar('Loss/validation_robot', val_robot_loss, epoch)
        writer.add_scalar('Loss/validation_decouple', val_decouple_loss, epoch)
        writer.add_scalar('Loss/validation_dynamic', val_dynamic_loss, epoch)

        print(
            f"Training - rec: {rec_loss:.4f} | task: {con_loss:.4f} | robot: {robot_loss:.4f} | decouple: {decouple_loss:.4f} | dynamic: {dynamic_loss:.4f} | Time: {time.time() - start_time:.2f}s"
        )
        print(
            f"Validation - rec: {val_rec_loss:.4f} | task: {val_con_loss:.4f} | robot: {val_robot_loss:.4f} | decouple: {val_decouple_loss:.4f} | dynamic: {val_dynamic_loss:.4f}\n"
        )

        if should_safe:
            save_model(encoder, prediction_head_state, prediction_head_action, prediction_head_reward,
                       prediction_head_cls, model_path, robot_head, dynamic_head)

        start_time = time.time()

    if predict_samples_num > 0:
        predict(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls,
                single_test_loader, predict_samples_num)

    if should_predict_last_state:
        last_state_loss = predict_last_state(encoder, prediction_head_state, test_loader)
        writer.add_scalar('Loss/last_state', last_state_loss, 0)
        print(f"Loss on prediciting the last state on the testset: {last_state_loss}")

    if train_bnpy:
        bnpy_cluster(encoder, prediction_head_cls, train_loader)

    if evaluate_embedding:
        trajectory_embedding_valdation(encoder, prediction_head_cls, embedding_num, loaded_dataset, embeddings_path)
        #embedding_valdation(encoder, prediction_head_cls, embedding_num, train_loader, embeddings_path)
        #embedding_valdation(encoder, prediction_head_cls, embedding_num, val_loader, embeddings_path)

    # Close the SummaryWriter when done
    writer.close()

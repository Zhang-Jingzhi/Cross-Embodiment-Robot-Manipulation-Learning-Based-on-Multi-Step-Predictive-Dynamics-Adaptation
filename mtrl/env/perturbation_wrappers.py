import collections
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np


def _clip_to_space(value: np.ndarray, space: gym.Space) -> np.ndarray:
    if not isinstance(space, gym.spaces.Box):
        return value
    low = space.low
    high = space.high
    if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
        return np.clip(value, low, high)
    return value


class ObservationNoiseWrapper(gym.Wrapper):
    """Inject Gaussian noise into observations during reset/step."""

    def __init__(self, env: gym.Env, obs_noise_std: float):
        super().__init__(env)
        self.obs_noise_std = float(obs_noise_std)

    def _perturb(self, obs):
        if self.obs_noise_std <= 0.0:
            return obs
        obs_arr = np.asarray(obs, dtype=np.float32)
        noisy = obs_arr + np.random.normal(
            loc=0.0,
            scale=self.obs_noise_std,
            size=obs_arr.shape,
        ).astype(np.float32)
        return _clip_to_space(noisy, self.observation_space)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._perturb(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._perturb(obs), reward, done, truncated, info


class ActionNoiseWrapper(gym.Wrapper):
    """Inject Gaussian noise into actions before stepping the env."""

    def __init__(self, env: gym.Env, action_noise_std: float):
        super().__init__(env)
        self.action_noise_std = float(action_noise_std)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        if self.action_noise_std > 0.0:
            action_arr = action_arr + np.random.normal(
                loc=0.0,
                scale=self.action_noise_std,
                size=action_arr.shape,
            ).astype(np.float32)
            action_arr = _clip_to_space(action_arr, self.action_space)
        return self.env.step(action_arr)


class ActionDelayWrapper(gym.Wrapper):
    """Delay actions by a fixed number of steps using a FIFO queue."""

    def __init__(self, env: gym.Env, action_delay_steps: int):
        super().__init__(env)
        self.action_delay_steps = max(int(action_delay_steps), 0)
        self._queue: Deque[np.ndarray] = collections.deque()
        self._zero_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._reset_queue()

    def _reset_queue(self) -> None:
        self._queue.clear()
        for _ in range(self.action_delay_steps):
            self._queue.append(self._zero_action.copy())

    def reset(self, **kwargs):
        self._reset_queue()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.action_delay_steps <= 0:
            delayed_action = np.asarray(action, dtype=np.float32)
        else:
            delayed_action = self._queue.popleft()
            self._queue.append(np.asarray(action, dtype=np.float32).copy())
        delayed_action = _clip_to_space(delayed_action, self.action_space)
        return self.env.step(delayed_action)


def apply_metaworld_perturbation_wrappers(
    env: gym.Env,
    obs_noise_std: float = 0.0,
    action_delay_steps: int = 0,
    action_noise_std: float = 0.0,
) -> gym.Env:
    """Apply lightweight perturbation wrappers used by robustness sweeps."""

    if action_noise_std > 0.0:
        env = ActionNoiseWrapper(env, action_noise_std=action_noise_std)
    if action_delay_steps > 0:
        env = ActionDelayWrapper(env, action_delay_steps=action_delay_steps)
    if obs_noise_std > 0.0:
        env = ObservationNoiseWrapper(env, obs_noise_std=obs_noise_std)
    return env

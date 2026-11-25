import numpy as np
import gymnasium as gym


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        return self.update_stats_and_infos(*super().step(action))

    def update_stats_and_infos(self, *args):
        observations, rewards, terminated, truncated, infos = args
        dones = np.logical_or(terminated, truncated)
        self.episode_returns += infos.get("reward", rewards)
        self.episode_lengths += 1
        self.returned_episode_returns = np.where(
            dones, self.episode_returns, self.returned_episode_returns
        )
        self.returned_episode_lengths = np.where(
            dones, self.episode_lengths, self.returned_episode_lengths
        )
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths

        return (
            observations,
            rewards,
            dones,
            infos,
        )

    def async_reset(self):
        self.env.async_reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    
    def recv(self):
        return self.update_stats_and_infos(*self.env.recv())

    def send(self, action):
        return self.env.send(action)


class CompatEnv(gym.Wrapper):

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        return observations, infos

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = np.logical_or(terminated, truncated)
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class ChainEffectObservation(gym.Wrapper):
    """Expose the engine's public effect slots as a model observation."""

    def __init__(self, env):
        super().__init__(env)
        self.num_envs = env.num_envs
        spaces = dict(env.observation_space.spaces)
        spaces["chain_effect_"] = gym.spaces.Box(
            0, 14, shape=spaces["chain_"].shape[:-1], dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {**obs, "chain_effect_": info["chain_effect_slots"]}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = {**obs, "chain_effect_": info["chain_effect_slots"]}
        return obs, reward, terminated, truncated, info


class EnvPreprocess(gym.Wrapper):

    def __init__(self, env, skip_mask):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.skip_mask = skip_mask

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        if self.skip_mask:
            observations['mask_'] = None
        return observations, infos

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        if self.skip_mask:
            observations['mask_'] = None
        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )
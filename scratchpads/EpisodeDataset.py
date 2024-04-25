
import gym

from procgen import ProcgenGym3Env
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm


class EpisodeDataset(Dataset):
    def __init__(self, env_fn, num_envs=1, max_steps=1000, episodes_per_epoch=2, skip_first=0, repeat_action=1, device='cpu'):
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.episodes_per_epoch = episodes_per_epoch
        self.envs = gym.vector.AsyncVectorEnv([self.env_fn for _ in range(num_envs)])
        self.n_classes = self.envs.single_action_space.n
        self.device = device
        self.skip_first = skip_first
        self.repeat_action = repeat_action
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        if skip_first > max_steps:
            raise Exception('EpisodeDataset Error: skip_first must be less than max_steps.')

        # Initially populate the dataset
        self._generate_episodes()

    def _generate_episodes(self):
        # Reset dataset
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        for _ in tqdm(range(self.episodes_per_epoch)):
            obs, _ = self.envs.reset()

            for step in range(self.max_steps):
                if step % self.repeat_action == 0:
                    action = self.envs.action_space.sample()

                next_obs, rs, ds, truncated, infos = self.envs.step(action)

                if step >= self.skip_first:
                    self.states.extend(obs.copy())
                    self.actions.extend(action.copy())
                    self.rewards.extend(rs.copy())
                    self.next_states.extend(next_obs.copy())
                    self.dones.extend(ds.copy())

                obs = next_obs

                if any(ds) or any(truncated):
                    # If episode ends, start a new episode until we reach the desired number of steps
                    obs, _ = self.envs.reset()

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        state = torch.tensor(self.states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        action = torch.tensor(self.actions[idx], dtype=torch.long, device=self.device)
        action = F.one_hot(action, self.n_classes).to(torch.float32).requires_grad_()
        reward = torch.tensor(self.rewards[idx], dtype=torch.float, device=self.device, requires_grad=True)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        done = torch.tensor(self.dones[idx], dtype=torch.float, device=self.device, requires_grad=True)

        return state, action, reward, done, next_state

    def __len__(self):
        # The total length is just the size of the accumulated states
        return len(self.states)
    

class ProcgenEpisodeDataset(EpisodeDataset):
    def __init__(self, env, wrapper = None, num_envs=1, max_steps=1000, episodes_per_epoch=2, skip_first=0, repeat_action=1, device='cpu'):
        self.env_name = env
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.episodes_per_epoch = episodes_per_epoch    
        self.envs = ProcgenGym3Env(num=num_envs, env_name=env)
        self.n_classes = self.envs.single_action_space.n
        self.device = device
        self.skip_first = skip_first
        self.repeat_action = repeat_action
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        if wrapper:
            self.envs = wrapper(self.envs)

        if skip_first > max_steps:
            raise Exception('EpisodeDataset Error: skip_first must be less than max_steps.')

        # Initially populate the dataset
        self._generate_episodes()

    def _generate_episodes(self):
        # Reset dataset
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        for _ in tqdm(range(self.episodes_per_epoch)):
            obs, _ = self.envs.reset()

            for step in range(self.max_steps):
                if step % self.repeat_action == 0:
                    action = self.envs.action_space.sample()

                next_obs, rs, ds, truncated, infos = self.envs.step(action)

                if step >= self.skip_first:
                    self.states.extend(obs.copy())
                    self.actions.extend(action.copy())
                    self.rewards.extend(rs.copy())
                    self.next_states.extend(next_obs.copy())
                    self.dones.extend(ds.copy())

                obs = next_obs

                if any(ds) or any(truncated):
                    # If episode ends, start a new episode until we reach the desired number of steps
                    obs, _ = self.envs.reset()

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        state = torch.tensor(self.states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        action = torch.tensor(self.actions[idx], dtype=torch.long, device=self.device)
        action = F.one_hot(action, self.n_classes).to(torch.float32).requires_grad_()
        reward = torch.tensor(self.rewards[idx], dtype=torch.float, device=self.device, requires_grad=True)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        done = torch.tensor(self.dones[idx], dtype=torch.float, device=self.device, requires_grad=True)

        return state, action, reward, done, next_state

    def __len__(self):
        # The total length is just the size of the accumulated states
        return len(self.states)
    



class MultiEpisodeDataset(Dataset):
    def __init__(self, env_fn, max_steps=1000, episodes_per_epoch=2, skip_first=0, repeat_action=1, device='cpu'):
        self.env_fn = env_fn
        self.max_steps = max_steps
        self.episodes_per_epoch = episodes_per_epoch
        self.env = env_fn()
        self.n_classes = self.env.single_action_space.n
        self.device = device
        self.skip_first = skip_first
        self.repeat_action = repeat_action
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        if skip_first > max_steps:
            raise Exception('EpisodeDataset Error: skip_first must be less than max_steps.')

        # Initially populate the dataset
        self._generate_episodes()

    def _generate_episodes(self):
        # Reset dataset
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        for _ in tqdm(range(self.episodes_per_epoch)):
            obs, _ = self.env.reset()

            for step in range(self.max_steps):
                if step % self.repeat_action == 0:
                    action = self.env.action_space.sample()

                next_obs, rs, ds, truncated, infos = self.env.step(action)

                if step >= self.skip_first:
                    self.states.extend(obs.copy())
                    self.actions.extend(action.copy())
                    self.rewards.extend(rs.copy())
                    self.next_states.extend(next_obs.copy())
                    self.dones.extend(ds.copy())

                obs = next_obs

                if any(ds) or any(truncated):
                    # If episode ends, start a new episode until we reach the desired number of steps
                    obs, _ = self.env.reset()

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        state = torch.tensor(self.states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        action = torch.tensor(self.actions[idx], dtype=torch.long, device=self.device)
        action = F.one_hot(action, self.n_classes).to(torch.float32).requires_grad_()
        reward = torch.tensor(self.rewards[idx], dtype=torch.float, device=self.device, requires_grad=True)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        done = torch.tensor(self.dones[idx], dtype=torch.float, device=self.device, requires_grad=True)

        return state, action, reward, done, next_state

    def __len__(self):
        # The total length is just the size of the accumulated states
        return len(self.states)
    
class GridworldEpisodeDataset(Dataset):
    def __init__(self, env_fn, max_steps=1000, episodes_per_epoch=2, skip_first=0, repeat_action=1, device='cpu'):
        self.env_fn = env_fn
        self.max_steps = max_steps
        self.episodes_per_epoch = episodes_per_epoch
        self.env = env_fn()
        self.n_classes = self.env.single_action_space.n
        self.device = device
        self.skip_first = skip_first
        self.repeat_action = repeat_action
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.player_positions = []
        self.dones = []

        if skip_first > max_steps:
            raise Exception('EpisodeDataset Error: skip_first must be less than max_steps.')

        # Initially populate the dataset
        self._generate_episodes()

    def _generate_episodes(self):
        # Reset dataset
        self.states.clear()
        self.actions.clear()
        self.player_positions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        for _ in tqdm(range(self.episodes_per_epoch)):
            obs, _ = self.env.reset()

            for step in range(self.max_steps):
                if step % self.repeat_action == 0:
                    action = self.env.action_space.sample()

                next_obs, rs, ds, truncated, infos = self.env.step(action)
                player_pos = self.env.get_player_position()
                if step >= self.skip_first:
                    self.states.extend(obs.copy())
                    self.actions.extend(action.copy())
                    self.rewards.extend(rs.copy())
                    self.next_states.extend(next_obs.copy())
                    self.player_positions.extend(player_pos.copy())
                    self.dones.extend(ds.copy())

                obs = next_obs

                if any(ds) or any(truncated):
                    # If episode ends, start a new episode until we reach the desired number of steps
                    obs, _ = self.env.reset()

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        state = torch.tensor(self.states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        action = torch.tensor(self.actions[idx], dtype=torch.long, device=self.device)
        action = F.one_hot(action, self.n_classes).to(torch.float32).requires_grad_()
        reward = torch.tensor(self.rewards[idx], dtype=torch.float, device=self.device, requires_grad=True)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float, device=self.device, requires_grad=True)
        done = torch.tensor(self.dones[idx], dtype=torch.float, device=self.device, requires_grad=True)
        player_pos = torch.tensor(self.player_positions[idx], dtype=torch.int32, device=self.device, requires_grad=False)

        return state, action, reward, done, next_state, player_pos

    def __len__(self):
        # The total length is just the size of the accumulated states
        return len(self.states)
    
import collections
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchhd as hd
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings
import os
import sys
from types import ModuleType
from typing import Sequence, Tuple
import gymnasium as gym
# import gym

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn

from EpisodeDataset import EpisodeDataset, ProcgenEpisodeDataset

import torch
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  import matplotlib.pyplot as plt
import torch
import numpy as np

from typing import List

import visual_gridworld
from visual_gridworld.gridworld.minigrid_procgen import GridworldResizeObservation # absolute import




VSA='MAP'

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size * 3, out_features, vsa=VSA)
        self.value = embeddings.Level(levels, out_features, vsa=VSA)


    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)
    
    
class HDCNoveltyDetector3D:
    def __init__(self, dim, threshold=0.5, memory_size=100):
        self.dim = dim  # Dimension of the HD vectors
        self.threshold = threshold  # Threshold for determining novelty
        self.visited_states_memory = []  # List to store HD vectors of visited states
        self.memory_size = memory_size
        self.sim_memory = []
        # Pre-generate HD vectors for each possible position within a channel
        # self.encoder = embeddings.Projection(7*7*3, dim)
        self.encoder = Encoder(dim, 7, 1000)

    def encode_state(self, tensor):
        """Encode a (3,7,7) tensor into a high-dimensional vector."""
        tensor = tensor.flatten(1)
        return self.encoder(tensor)

    def novelty(self, state_vector):
        """Check for novelty of a state vector."""
        if len(self.visited_states_memory) == 0:
            return 1.
        return self.__novelty(state_vector, torch.stack(list(self.visited_states_memory), dim=0))

    def __novelty(self, state_vector, memory):
        return (1 - hd.cosine_similarity(state_vector, memory).mean())

    def recalc_novelty(self):
        novelties = []
        for i in range(len(self.visited_states_memory)):
            state_vector = self.visited_states_memory[i]
            
            memory = None
            if i == 0:
                memory = self.visited_states_memory[1:]
            elif i == (len(self.visited_states_memory)-1):
                memory = self.visited_states_memory[:-1]
            else:
                memory = self.visited_states_memory[:i] + self.visited_states_memory[i+1:]
                

            novelties.append(self.__novelty(state_vector, torch.stack(memory, dim=0)))
        self.sim_memory = novelties

    def add_to_memory(self, state_vector, novelty):
        """Adds a state vector to the memory of visited states."""

        if len(self.visited_states_memory) < self.memory_size:
            self.visited_states_memory.append(state_vector)
            self.sim_memory.append(novelty)
            if len(self.visited_states_memory) > 3:
                self.recalc_novelty()
            return True
        print(min(self.sim_memory) + self.threshold)
        if novelty < (min(self.sim_memory) + self.threshold):
            return False
        
        min_index = np.argmin(self.sim_memory)
        self.visited_states_memory.pop(min_index)
        self.sim_memory.pop(min_index)
        self.visited_states_memory.append(state_vector)
        self.sim_memory.append(novelty)
        return True

def collect_samples(env_name, num_episodes, max_steps_per_episode):
    """
    Collects samples from the specified MiniGrid environment.

    Args:
    - env_name: The name of the MiniGrid environment to sample from.
    - num_episodes: The number of episodes to run and collect samples.
    - max_steps_per_episode: The maximum number of steps to take per episode.

    Returns:
    - samples: A list of samples, where each sample is a tuple (observation, action, reward, next_observation, done).
    """
    env = gym.make(env_name)
    samples = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        max_steps = max_steps_per_episode if episode < 100 else max_steps_per_episode * 2
        for step in range(max_steps):
            action = env.action_space.sample()  # Taking random actions
            next_observation, reward, done, trunc, info = env.step(action)

            done = done or trunc or step == (max_steps-1)
            # Store the sample (s, a, r, s', done)
            samples.append((observation, action, reward, next_observation, done))

            observation = next_observation
            if done:
                break

    env.close()
    return samples

# Example usage
env_name = 'Visual/DoorKey16x16-Gridworld-v0'  # Specify the environment
num_episodes = 10
max_steps_per_episode = 50
samples = collect_samples(env_name, num_episodes, max_steps_per_episode)

print(f"Collected {len(samples)} samples.")




# Example usage
novelty_detector = HDCNoveltyDetector3D(dim=1024, threshold=0.0001, memory_size=50)
import matplotlib
matplotlib.use('TkAgg')
        
novelty_scores = []
new_sample_added = []
samples_added = 0
# Initialize plot
plt.ion()
fig, ax = plt.subplots()
ax.set_title('Real-time Novelty Tracking')
ax.set_xlabel('Step')
ax.set_ylabel('Novelty Score')
line, = ax.plot(novelty_scores, label='Novelty Score')
line2, = ax.plot(new_sample_added, label='Samples Added')
ax.legend()
ax.grid(True)

# Assuming `input_tensor` is your (3,7,7) tensor
# Here's an example tensor initialized with random values for demonstration

i = 0
stacking = 6
state_vector = hd.structures.BundleSequence(1024, vsa=VSA)
obs = []
for sample in samples:
    i += 1
    observation, action, reward, next_observation, done = sample
    if done:
        i = 0
        state_vector = hd.structures.BundleSequence(1024, vsa=VSA)
        obs = []
    

    obs_enc = novelty_detector.encode_state(torch.Tensor(observation)[None, :])
    obs.insert(0, obs_enc)
    state_vector.append(obs_enc)

    if i < stacking:
        continue                                 

    v = state_vector.value
    novelty = novelty_detector.novelty(v)    
    added = novelty_detector.add_to_memory(v, novelty)
    if novelty != 1:
        novelty_scores.append(novelty)
        new_sample_added.append(samples_added)
    if added:
        samples_added += 1
    state_vector.popleft(obs.pop())
    
    # Encode the state
    # Check for novelty and add to memory if novel

line.set_ydata(novelty_scores)
line.set_xdata(list(range(len(novelty_scores))))

new_sample_added = [ v / samples_added for v in new_sample_added]
line2.set_ydata(new_sample_added)
line2.set_xdata(list(range(len(new_sample_added))))

ax.relim()
ax.autoscale_view(True,True,True)
fig.canvas.draw()
fig.canvas.flush_events()


while True:
    time.sleep(0.1)
    pass
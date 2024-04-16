import os
import sys

import wandb

from scratchpads.template import ActionPredictor, preprocess
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from tqdm import tqdm
    
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


from torchmetrics.functional.classification import multiclass_accuracy

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



## Get Data

from scratchpads.EpisodeDataset import MultiEpisodeDataset



class LimitActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env, actions: List[int]):
        assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.single_action_space, gym.spaces.Discrete)

        super().__init__(env)
        if hasattr(env, 'single_action_space'):
            self.single_action_space = gym.spaces.Discrete(len(actions))
            if len(env.action_space.shape) > 1:
                self.action_space = gym.spaces.MultiDiscrete((env.action_space.shape[0], ) * len(actions))
            else:
                self.action_space = gym.spaces.MultiDiscrete((len(actions), ))
        else:
            self.action_space = gym.spaces.Discrete(len(actions))

        self.action_indices = np.array(actions)

    def action(self, action):
        return self.action_indices[np.array(action)]
    
env_fn = lambda: LimitActions(GridworldResizeObservation(gym.make("Visual/DoorKey8x8-Gridworld-v0"), (84, 84)), [0, 1, 2, 3])
# train_data = ProcgenEpisodeDataset('procgen-coinrun-v0', wrapper=lambda x:  gym.wrappers.ResizeObservation(gym.wrappers.GrayScaleObservation(x), shape=(84,84)),
#             num_envs=1, max_steps=100, episodes_per_epoch=100, skip_first=10, repeat_action=1, device='cpu') 
train_data = MultiEpisodeDataset(env_fn, max_steps=300, episodes_per_epoch=200, skip_first=0, repeat_action=1, device='cpu')

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)



def construct_model(config):
    obs_space = env_fn().single_observation_space
    action_n = env_fn().single_action_space.n
    """

                    input_dim: Sequence[int],
                    action_dim: int,
                    stages: Sequence[int],
                    backbone_stages: Sequence[int],
                    dense_units: Sequence[int],
                    template_var: Union[int, Iterable[int]], 
                    templates: int, 
                    activation: ModuleType = nn.SiLU,
                    layer_norm: bool = False,
                    device=None):
    """
    print(obs_space)


    net = ActionPredictor(input_dim=(3, 84, 84),
                        input_channels=3,
                        action_dim=action_n, 
                        cutoff=config.cutoff,
                        template_var=[70, config.template_size],
                        )
    net.template.set_mixin_factor(0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)

    return net, criterion, optimizer


def train():
    with wandb.init() as run:
        _train(run.config)

def _train(config):
    mask = False
    mask_at_epoch = config.mask_at_epoch
    mask_stop_at = mask_at_epoch + config.mask_stop_At

    net, criterion, optimizer = construct_model(config)
    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            state, action, reward, done, next_state = data

            state = preprocess(state)
            next_state = preprocess(next_state)
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if mask:
                outputs, _, _, local_loss = net(state, next_state, mask)
                loss = local_loss.sum()
            else:
                outputs, _, _ = net(state, next_state, mask)
                loss = 0

            loss += criterion(outputs, action)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_accuracy += multiclass_accuracy(outputs.argmax(dim=-1), action.argmax(dim=-1), num_classes=int(action_n))
            

            
        if epoch >= mask_at_epoch:
            mask = True
        net.template.set_mixin_factor((epoch - mask_at_epoch) * (1/(mask_stop_at - mask_at_epoch)))
        _max_acc = running_accuracy / len(train_loader)

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(train_loader):.3f} acc: {_max_acc:.3f}')

        if epoch >= mask_stop_at:
            wandb.log({'loss': running_loss / len(train_loader), 'accuracy': _max_acc})


        running_accuracy = 0.0
        running_loss = 0.0
        
sweep_config = {
    'method': 'grid',  # can also be grid, random, bayesian
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'mask_at_epoch': {
            'values': [5, 7, 9, 11]
        },
        'mask_stop_at': {
            'values': [5, 10, 15]
        },
        'cutoff': {
            'values': [0.1, 0.2, 0.5, 0.7]
        },
        'template_size': {
            'values': [20, 15, 10, 5]
        }
    }
}
# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="DoorKey8x8_action_predictor_sweep")

print(sweep_id)

# %%
import os
import sys
from typing import Tuple
import gymnasium as gym
# import gym

import torch
import torch.nn as nn


import torch
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  import matplotlib.pyplot as plt
import torch
import numpy as np

from typing import List

from visual_gridworld.gridworld.minigrid_procgen import DoorKey16x16Gridworld, DoorKey5x5Gridworld, DoorKey6x6Gridworld, DoorKey8x8Gridworld, GridworldResizeObservation, MultiRoomS10N6GridWorld, MultiRoomS4N2GridWorld, MultiRoomS5N4GridWorld, NoisyDoorKey16x16Gridworld, NoisyDoorKey5x5Gridworld, NoisyDoorKey6x6Gridworld, NoisyDoorKey8x8Gridworld, NoisyMultiRoomS10N6GridWorld, NoisyMultiRoomS4N2GridWorld, NoisyMultiRoomS5N4GridWorld # absolute import


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
from torch import Tensor
from typing import List, Optional, Tuple, Union

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from EpisodeDataset import GridworldEpisodeDataset
# %%
from tqdm import tqdm
from torchmetrics.functional.classification import multiclass_accuracy

from template import ActionPredictor, evaluate, preprocess

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


@script
def gaussian_fn(M: int, std: int) -> Tensor:
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

@script
def gkern(kernlen: int = 256, std: int = 128, vmin: float = 0, vmax: float = 1) -> Tensor:
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d * (vmax - vmin) + vmin


class Template(nn.Module):
    def __init__(self, M: int, out_size: int, var: torch.IntTensor, stride: int = 1,
                 cutoff: float = 0.2, initial_mixin_factor: float = 0.0,
                 device: torch.device = None):
        super().__init__()
        # Initialize basic parameters
        self.out_size = out_size  # Template size
        self.channels = M  # Number of channels

        # Mixin factor controls the interpolation between identity and templates
        self._mixin_factor = initial_mixin_factor
        self.cutoff = cutoff  # Cutoff threshold for template values
        self.device = device  # Device to run the model on (CPU/GPU)
        self.var = var  # Variance, can be a range or a fixed value
        self.stride = stride # Distance between mean of filters
        # Initially create templates based on the current variance
        self.create_new_templates(self._curr_var())

    @torch.jit.export
    def _curr_var(self, mixin: Optional[float] = None) -> int:
        # Determine the current variance based on the mixin factor
        # If `var` is a fixed int, just return it. If it's a tuple, interpolate.
        if mixin is None:
            return int(self.var[0] + self._mixin_factor * (self.var[1] - self.var[0]))
        return int(self.var[0] + mixin * (self.var[1] - self.var[0]))
    
    @torch.jit.export
    def set_mixin_factor(self, mixin_factor: float) -> None:
        # self.cutoff = max(0., min(1., mixin_factor))
        # # Update the mixin factor, ensuring it remains between 0 and 1
        _mixin_factor = max(0., min(1., mixin_factor))
        
        # If `var` is not a fixed value, recreate templates with the new variance
        if self._mixin_factor != _mixin_factor:
            self.create_new_templates(self._curr_var(_mixin_factor))
        self._mixin_factor = float(_mixin_factor)
        
    def create_new_templates(self, var: int) -> None:
        # Method to generate new templates based on given variance `var`
        
        n_square = (self.out_size * self.out_size)  # Total number of pixels
        tau = 0.5 / n_square  # Scaling factor for templates
        self.tau = tau
        alpha = n_square / (1 + n_square)  # Weight for positive template contribution
        beta = 4  # Scaling factor to amplify template values
        
        # Generate a base mask with Gaussian blur, cutoff, and scaling
        base = gkern(self.out_size*2-1, var, vmin=-self.cutoff, vmax=1.)[:,:,None]
        base = F.relu(base) * 2 - 1
        base = tau * torch.clamp(beta * base / self.out_size, min=-1)

        # Extract patches from the base mask to create templates
        templates = base.unfold(0, self.out_size, self.stride) \
                        .unfold(1, self.out_size, self.stride) \
                        .reshape(-1, self.out_size, self.out_size)
        templates = torch.flip(templates, dims=(0,))  # Correct orientation
        
        # Transfer templates to the specified device
        self.templates_f = templates.requires_grad_(False).to(self.device)
        neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
        templates = torch.cat([templates, neg_template], 0)
        self.templates_b = templates.requires_grad_(False).to(self.device)

        if not hasattr(self, 'p_T'):
        # Probability distribution over templates
            p_T = [alpha / n_square for _ in range(self.templates_f.shape[0])]
            p_T.append(1 - alpha)  # Add probability for the negative template
            self.p_T = torch.FloatTensor(p_T).requires_grad_(False).to(self.device)
        
    def get_mask_from_indices(self, indices: Tensor) -> Tensor:
        # Select templates for each index found by max pooling
        selected_templates = torch.stack([self.templates_f[(i/(self.stride**2)).long()] for i in indices], dim=0)
        # Apply the selected templates to the input and return the masked input and the templates
        mask = (selected_templates / selected_templates.max())
        mask = F.relu(mask - self.cutoff) / self.cutoff
        return mask        

    def get_masked_output(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # For each element in the batch, find the max pool index to select the corresponding template
        _, indices = F.max_pool2d(x, self.out_size, return_indices=True)
        indices = indices.view(x.shape[:2]).long()
        
        # Interpolate between the identity mask and the filtered templates based on mixin_factor
        mask = self.get_mask_from_indices(indices)
        # templates = torch.lerp(self.identity_mask, self.templates_f, self._mixin_factor)
        x_masked = x * mask
        return x_masked, mask
    
    def compute_local_loss(self, x: Tensor) -> Tensor:  
        # Calculate the tensor dot product between input x and templates, then softmax to get probabilities
        tr_x_T = torch.einsum('bcwh,twh->cbt', x * self.tau, self.templates_b)
        p_x_T = F.softmax(tr_x_T, dim=1)
        # Calculate the adjusted probability distribution of x given T
        p_x = torch.einsum('t,cbt->cb', self.p_T, p_x_T)
        p_x = torch.log(p_x_T/p_x[:, :, None])
        # Calculate the log probability loss
        p_x_T_log = torch.einsum('cbt,cbt->ct', p_x_T, p_x)
        # Negative log likelihood loss
        loss = -torch.einsum('t,ct->c', self.p_T, p_x_T_log)
        return loss

    def forward(self, x: Tensor, train: bool = True) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        # Main forward pass
        x, obs_mask = self.get_masked_output(x)  # Get masked output based on the current mixin_factor
        if train:
            # If training, also compute the local loss
            loss_1 = self.compute_local_loss(x)
            return x, obs_mask, loss_1  # Return the masked input and the computed loss
        return x, obs_mask # For inference, just return the masked input

class SiameseAttentionNetwork(nn.Module):
    def __init__(self, base_network, attention_hidden_size=128, num_classes=10):
        super(SiameseAttentionNetwork, self).__init__()
        self.base_network = base_network
        self.mask = False
        # Attention mechanism
        self.attention_fc = nn.Sequential(
            nn.LazyLinear(attention_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_hidden_size, 1)
        )


        self.template_counts = attention_hidden_size
        self.template = torch.jit.script(Template(M=1, cutoff=0.2, out_size=21, var=[6,5], stride=4, device=device))
        
        
        # Spatial attention module
        self.spatial_attention_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
        # Classifier
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x1, x2):
        # Forward pass through base network
        out1 = self.base_network(x1)
        out2 = self.base_network(x2)
        
        # Compute attention weights
        att1 = self.attention_fc(out1.reshape(out1.size(0), -1))
        att2 = self.attention_fc(out2.reshape(out2.size(0), -1))
        att1 = F.softmax(att1, dim=1).unsqueeze(2).unsqueeze(3)
        att2 = F.softmax(att2, dim=1).unsqueeze(2).unsqueeze(3)

        if self.mask:
            out1, obs_mask, local_loss = self.template(out1.sum(dim=1)[:,None], train=True)
            out2, obs_mask, local_loss = self.template(out2.sum(dim=1)[:,None], train=True)

        
        # Apply spatial attention
        out1 = out1 * att1
        out2 = out2 * att2
        
        # Merge features
        merged_features = torch.cat((out1, out2), dim=1)
        
        # Classifier
        out = F.relu(self.fc1(merged_features.reshape(merged_features.size(0), -1)))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out, att1, att2




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
    
env_fn = lambda: LimitActions(GridworldResizeObservation(gym.make("Visual/MultiRoomS10N6-Gridworld-v0"), (84, 84)), [0, 1, 2, 3])
# train_data = ProcgenEpisodeDataset('procgen-coinrun-v0', wrapper=lambda x:  gym.wrappers.ResizeObservation(gym.wrappers.GrayScaleObservation(x), shape=(84,84)),
#             num_envs=1, max_steps=100, episodes_per_epoch=100, skip_first=10, repeat_action=1, device='cpu') 
train_data = GridworldEpisodeDataset(env_fn, max_steps=200, episodes_per_epoch=50, skip_first=0, repeat_action=1, device=device)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)


# %%
for i in train_loader:
    print(i[0].shape)
    break


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
"""
self, M: int, out_size: int, var: torch.IntTensor, stride: int = 1,
                 cutoff: float = 0.2, initial_mixin_factor: float = 0.0,
                 device: torch.device = None
"""
# PatchDepthwiseNetwork
# def __init__(self, input_channels, num_classes, patch_size=3):
# net = ActionPredictor(input_dim=(3, 84, 84),
#                       input_channels=3,
#                       action_dim=action_n, 
#                       cutoff=0.2,
#                       template_var=[70, 26],
#                       )
def train(train_loader, name):
    # Create the base network
    base_network = BaseNetwork()

    # Create the Siamese network
    net = SiameseAttentionNetwork(base_network, num_classes=4)
    # net = ImageActionPredictor()
    # net.template.set_mixin_factor(0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    mask = False
    epochs = 50
    mask_at_epoch = 15
    mask_stop_at = 1
    max_acc = 0
    count = 0

    # evaluate(net, train_loader)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_pos_dev = 0.0
        running_loss = 0.0
        running_accuracy = 0.0
        running_loss_scale = 0.0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            state, action, reward, done, next_state, player_pos = data

            state = preprocess(state)
            next_state = preprocess(next_state)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = net.train_interpretable_step(state, next_state)
            outputs, att1, att2 = net(state, next_state)
            loss = 0

            loss += criterion(outputs, action) #+ 0.01/(F.sigmoid(loss_scale/16)+1e-6)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_accuracy += multiclass_accuracy(outputs.argmax(dim=-1), action.argmax(dim=-1), num_classes=int(action_n))
            # running_pos_dev += np.linalg.norm(player_pos/8. - pos2.detach().numpy(), axis=1).sum() / len(pos2)
            # running_loss_scale += loss_scale
        if epoch >= mask_at_epoch:
            mask = True
        # net.template.set_mixin_factor((epoch - mask_at_epoch) * (1/(mask_stop_at - mask_at_epoch)))
        _max_acc = running_accuracy / len(train_loader)
        _max_dist = running_pos_dev / len(train_loader)
        _loss_scale = running_loss_scale / len(train_loader)
        if epoch > mask_at_epoch:
            net.mask = True
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(train_loader):.3f} acc: {_max_acc:.3f} _max_dist: {_max_dist} loss_scale: {_loss_scale:3f}')

        running_pos_dev = 0.0
        running_accuracy = 0.0
        running_loss = 0.0
        running_loss_scale = 0.0
    
    torch.save(net.state_dict(), f'models/template_{name}.pt')


# Define the folder path relative to the current working directory
folder_path = "models"

# Check if the folder exists
if not os.path.exists(folder_path):
    # If the folder doesn't exist, create it
    os.makedirs(folder_path)
    print("Folder 'models' created successfully.")
else:
    print("Folder 'models' already exists.")

envs = [
    # MultiRoom
    MultiRoomS4N2GridWorld,
    MultiRoomS5N4GridWorld,
    MultiRoomS10N6GridWorld,
    # DoorKey
    DoorKey8x8Gridworld, 
    DoorKey16x16Gridworld, 
    # Noise MultiRoom
    NoisyMultiRoomS4N2GridWorld,
    NoisyMultiRoomS5N4GridWorld,
    NoisyMultiRoomS10N6GridWorld,
    # Noise DoorKey
    NoisyDoorKey8x8Gridworld,
    NoisyDoorKey16x16Gridworld,]
for env in envs:
    env_fn = lambda: LimitActions(GridworldResizeObservation(gym.make(f"Visual/{env.env_name}"), (84, 84)), [0, 1, 2, 3])
    train_data = GridworldEpisodeDataset(env_fn, max_steps=200, episodes_per_epoch=100, skip_first=0, repeat_action=1, device=device)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    train(train_loader, env.env_name)

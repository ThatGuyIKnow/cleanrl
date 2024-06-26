from itertools import pairwise
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multiclass_accuracy
import torchvision

import visual_gridworld
from visual_gridworld.gridworld.minigrid_procgen import BlockyBackgroundGridworldWrapper, GridworldResizeObservation, NoisyGridworldWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
from torch import Tensor
from typing import Optional, Tuple, Union


class MaskedRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = np.ones(shape, 'float64') * epsilon

    def update(self, x, mask):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = mask.sum(axis=0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


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

    def get_masked_output(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # For each element in the batch, find the max pool index to select the corresponding template
        _, indices = F.max_pool2d(x, self.out_size, return_indices=True)
        indices = indices.view(x.shape[:2]).long()
        
        # Interpolate between the identity mask and the filtered templates based on mixin_factor
        mask = self.get_mask_from_indices(indices)
        mask /= mask.max()
        # templates = torch.lerp(self.identity_mask, self.templates_f, self._mixin_factor)
        x_masked = x * mask
        x_i = indices % self.out_size
        y_i = indices // self.out_size
        v = self.var[0]
        crop_x_masked = F.pad(x_masked, (v-1, v, v-1, v)).unfold(-2, 2*v, 1).unfold(-2, 2*v, 1)
        crop_x_masked = crop_x_masked.view(-1, 21, 21, 2*v, 2*v)
        crop_x_masked = crop_x_masked[torch.arange(x.size(0)*x.size(1)), y_i.view(-1), x_i.view(-1)].view(x.size(0), x.size(1), 2*v, 2*v)
        return x_masked, mask, crop_x_masked
    
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

    def forward(self, x: Tensor, train: bool = True) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # Main forward pass
        x, obs_mask, crop_obs_mask = self.get_masked_output(x)  # Get masked output based on the current mixin_factor
        if train:
            # If training, also compute the local loss
            # loss_1 = self.compute_local_loss(x)
            loss_1 = torch.zeros([x.size(0)]).to(self.device)
            return x, obs_mask, loss_1, crop_obs_mask  # Return the masked input and the computed loss
        return x, obs_mask, crop_obs_mask # For inference, just return the masked input

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
        self.template = torch.jit.script(Template(M=1, cutoff=0.2, out_size=21, var=[5,5], stride=1, device=device))
        # self.template = Template(M=1, cutoff=0.2, out_size=21, var=[5,5], stride=1, device=device)
        
        
        # Classifier
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def prong(self, x, train=False):
        # Forward pass through base network
        out1 = self.base_network(x)
        out1, obs_mask, local_loss1, crop_out1 = self.template(out1, train=True)
    
        # Compute attention weights
        att1 = self.attention_fc(crop_out1.view(*crop_out1.shape[:2], -1))
        if train:
            att1 = F.dropout(att1, 0.25)
        att1 = F.softmax(att1, dim=1)
        out1 = F.adaptive_max_pool2d(out1, 1).view(out1.size(0), -1)
        return out1, att1, obs_mask

    def forward(self, x1, x2):
        # Forward pass through base network
        out1, att1, _ = self.prong(x1, train=True)
        out2, att2, _ = self.prong(x2, train=True)

        local_loss1 = local_loss2 = torch.zeros([out1.size(0)]).to(device)

        out1 = out1 * att1.squeeze(2)
        out2 = out2 * att2.squeeze(2)

        # Merge features
        merged_features = out1 - out2
        
        # Classifier
        out = F.relu(self.fc1(merged_features.reshape(merged_features.size(0), -1)))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out, 0.5*(local_loss1 + local_loss2)

    def get_mask(self, x, full_output=False):
        # Forward pass through base network
        with torch.no_grad():
            out1, att1, obs_mask = self.prong(x)
        if full_output:
            return out1, att1, obs_mask
        
        return obs_mask * att1[...,None]
        

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    fixed: bool = True
    """Fix the gridworld to a single world"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device: str = None
    """Device to train on if cuda is used"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_tags: tuple[str] = tuple('wandb', )
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Visual/MultiRoomS10N6-Gridworld-v0"
    """the id of the environment"""
    env_mode: Optional[str] = None
    """Environemt mode (random or hard)"""
    camera_mode: Literal['full', 'agent_centric', 'room_centric'] = "full"
    """camera mode. What does the camera follow"""
    agent_view: Tuple[int, int] = (2, 2)
    """Number of tiles the agent can see on either side in agent_centric view"""
    background_noise: Optional[Literal['noisy', 'blocky']] = None
    """Type of background noise (if any)"""
    show_key_icon: bool = True
    """whether to show the key icon"""
    cell_size: int = 10
    '''cell pixel size'''
    total_timesteps: int = int(13e6)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RND arguments
    update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    int_coef: float = 1.0
    """coefficient of extrinsic reward"""
    ext_coef: float = 2.0
    """coefficient of intrinsic reward"""
    int_gamma: float = 0.99
    """Intrinsic reward discount rate"""
    num_iterations_obs_norm_init: int = 50
    """number of iterations to initialize the observations normalization parameters"""
    use_mean: bool = True
    """whether to use the mean variance normalize"""

    # Early stopping arguments
    early_stopping_threshold: float = None
    """margin for early stopping"""
    early_stopping_patience: int = int(1e5)
    """patience for early stopping"""

    # masking
    use_template: bool = True
    """use_template"""
    template_size: int = 4
    """masking template cell size"""
    alpha: float = 0.0
    """transparancy"""
    train_mask_at: int = 0
    """start masking at step"""
    template_batch: int = 64
    """train batches"""
    template_train_every: int = 8
    """train every"""
    template_lr: float = 1e-4
    '''learning rate'''
    template_epochs: int = 0
    """template epochs"""
    template_training_schedule: Tuple[List[int], List[int]] = tuple([[],[]])
    """epoch training schedule. Useful for faster training"""
    masking_pretraining_epochs: int = 15
    """pretraining epochs for masking"""
    masking_pretraining_steps: int = 20
    """pretraining epochs for masking"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    max_reward: float = None
    """max reward for early termination"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = 1
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)
        self.num_envs = observations.shape[0]
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        return observations, info

    def step(self, action):
        observations, rewards, dones, truncated, infos = super().step(action)
        self.episode_returns = infos["episodic_return"]
        self.returned_episode_returns[:] = self.episode_returns
        self.episode_returns *= 1 - (dones | truncated)
        infos["r"] = self.returned_episode_returns
        infos["l"] = infos["step_count"]
        return (
            observations,
            rewards,
            dones,
            truncated,
            infos,
        )


class FirstChannelPositionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (obs_shape[0], obs_shape[-1], *obs_shape[1:3]))
        self.single_observation_space = gym.spaces.Box(0, 255, (obs_shape[-1], *obs_shape[1:3]))

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)
        observations = np.moveaxis(observations, -1, -3)
        return observations, info

    def step(self, action):
        observations, rewards, dones, truncated, infos = super().step(action)
        observations = np.moveaxis(observations, -1, -3)
        return (
            observations,
            rewards,
            dones,
            truncated,
            infos,
        )


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

class TemplateMasking(nn.Module):
    def __init__(self, obs_shape, action_n: int):
            super().__init__()
            self.base_network = BaseNetwork()
            # Create the Siamese network
            self.net = SiameseAttentionNetwork(self.base_network, num_classes=action_n)
            
            self.shape = obs_shape
            
    def to(self, device):
        super().to(device=device)
        self.base_network.to(device)
        self.net.to(device)


    def forward(self, x1, x2):
        return self.net(x1  / 255.0, x2 / 255.0)

    def get_mask(self, x):
        m = self.net.get_mask(x / 255.0)
        m = m.sum(1, keepdim=True)
        norm = m.view(m.size(0), -1).max(dim=1)[0]
        m /= norm.view(-1, 1, 1, 1)
        return F.interpolate(m, self.shape[-2:]) #/ m.max()
    
class RNDModel(nn.Module):
    def __init__(self):
        super().__init__()


        feature_output = 7 * 7 * 64

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False
        b, w, h = envs.get_grid().shape
        self.shape = torch.LongTensor((w, h)).to(device)
        self.obs_shape = torch.LongTensor(envs.observation_space.shape[-2:]).to(device)


    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
    

class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
    
# Assuming you have a function to gather samples from the environment
def gather_samples(envs, batch_size):
    obs_batch, next_obs_batch, action_batch = [], [], []
    obs, reward, done, _, _ = envs.step(envs.action_space.sample())
    for _ in range(batch_size):
        action = envs.action_space.sample() 
        next_obs, reward, done, _, _ = envs.step(action)
        obs_batch.append(obs)
        next_obs_batch.append(next_obs)
        action_batch.append(action)
        obs = next_obs
    return (
        torch.tensor(np.array(obs_batch)).to(device),
        torch.tensor(np.array(next_obs_batch)).to(device),
        torch.tensor(np.array(action_batch)).to(device)
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args.wandb_tags[0].split(','))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    early_stopping_counter = args.total_timesteps
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert (s := args.template_training_schedule) is None or len(s[0]) == len(s[1])
    args.template_training_schedule = tuple([[100,], [0,]]) 
    args.template_training_schedule[0].insert(0, -1)
    args.template_training_schedule[1].insert(0, args.template_epochs)
    args.template_training_schedule[0].append(np.inf)
    args.template_training_schedule[1].append(0)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            tags=args.wandb_tags[0].split(','),
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.cuda else "cpu")
        
    # env setup
    env_kwargs = dict(
        num_envs=args.num_envs,
        cell_size=args.cell_size,
        fixed=args.fixed,
        seed=args.seed,
        camera_mode = args.camera_mode,
        camera_kwargs = {'agent_view': np.array((2, 2) if args.agent_view is None else args.agent_view)},
        show_key_icon=args.show_key_icon)
    if args.env_mode is not None:
        env_kwargs['mode'] = args.env_mode

    envs = gym.make(
        args.env_id,
        **env_kwargs
    )

    if args.early_stopping_threshold and hasattr(envs, 'max_reward'):
        args.max_reward = envs.max_reward * (1-args.early_stopping_threshold)
    
    if args.background_noise == "noisy":
        envs = NoisyGridworldWrapper(envs, alpha = 1.)
    elif args.background_noise == "blocky":
        envs = BlockyBackgroundGridworldWrapper(envs)
        
    envs = GridworldResizeObservation(envs, (84, 84))
    envs.num_envs = args.num_envs
    envs = RecordEpisodeStatistics(envs)
    envs = FirstChannelPositionWrapper(envs)

    obs_shape = envs.observation_space.shape
    action_n = envs.single_action_space.n
    agent = Agent(envs).to(device)
    template = TemplateMasking(obs_shape, action_n)
    template.to(device)
    template.net.mask = False
    rnd_model = RNDModel().to(device)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )
    mask_optimizer = optim.AdamW(
        template.parameters(),
        lr=args.template_lr,
    )
    mask_criterion = nn.CrossEntropyLoss()
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, *obs_shape[1:]))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    player_masks = torch.zeros_like(obs).to(device)
    # player_pos = torch.zeros((args.num_steps, args.num_envs, 2), dtype=torch.int32).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    # next_player_pos = torch.from_numpy(envs.get_player_position()).to(device)
    next_player_masks = template.get_mask(next_obs.to(device))
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print("Start pretraining masking/template")

    for i in range(args.masking_pretraining_epochs):
        if not args.use_template:
            break
        b_obs, b_next_obs, b_actions = gather_samples(envs, args.template_batch * args.masking_pretraining_steps)

        b_obs = b_obs.swapdims(0, 1).reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = b_next_obs.swapdims(0, 1).reshape((-1,) + envs.single_observation_space.shape)
        b_actions = b_actions.swapdims(0, 1).reshape(-1)
        for start, end in pairwise(range(0, b_obs.shape[0] + 1, args.template_batch)):
            b_act_pred, _ = template(b_obs[start:end], b_next_obs[start:end])
            b_act = F.one_hot(b_actions[start:end].long(), action_n).float()
            action_loss = mask_criterion(b_act_pred, b_act)
            total_loss = action_loss
            print(total_loss)

            mask_optimizer.zero_grad()
            total_loss.backward()
            mask_optimizer.step()

        writer.add_scalar("losses/pretrain_loss", total_loss.item(), i)
        
    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    masks = []
    for step in tqdm(range(args.num_steps * args.num_iterations_obs_norm_init), smoothing=0.05):
        if not args.use_mean:
            break
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, r, d, t, _ = envs.step(acs)
        next_ob += list(s)
        with torch.no_grad():
            m = template.get_mask(torch.from_numpy(s).to(device)).cpu().numpy()
        masks += list(m)

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            mask = np.stack(masks)
            if args.use_template:
                masked_b_obs = next_ob * mask + (1-mask) * obs_rms.mean
                obs_rms.update(masked_b_obs)
            else:
                obs_rms.update(next_ob)
            next_ob = []
            masks = []


    print("End to initialize...")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            player_masks[step] = next_player_masks
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            done = done | truncated
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            mask = template.get_mask(next_obs.to(device))

            if args.use_template:
                masked_next_obs = next_obs * mask
            else:
                masked_next_obs = next_obs
            if args.use_mean:
                rnd_next_obs = (
                    (
                        (masked_next_obs - torch.from_numpy(obs_rms.mean).to(device))
                        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                    ).clip(-5, 5)
                ).float()
            else:
                rnd_next_obs = masked_next_obs
            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data

            visited_rooms = []
            for idx, d in enumerate(done):
                if d.astype(int) == 1:
                    avg_returns.append(info["r"][idx])
                    epi_ret = np.average(avg_returns)
                    print(
                        f"global_step={global_step}, episodic_return={info['r'][idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                    )
                    writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar(
                        "charts/episode_curiosity_reward",
                        curiosity_rewards[step][idx],
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                    visited_rooms.append(info["rooms_visited"][idx])
                    if args.max_reward is not None and \
                            args.max_reward > epi_ret and \
                            early_stopping_counter == args.total_timesteps:
                        early_stopping_counter = global_step + args.early_stopping_patience
                    else:
                        early_stopping_counter = args.total_timesteps
                        
            if len(visited_rooms) > 0:
                visited_rooms = np.array(visited_rooms)
                writer.add_scalar("charts/rooms_visited", visited_rooms.mean(), global_step)
                writer.add_scalar("charts/max_rooms_visited", visited_rooms.max(), global_step)

        if global_step > early_stopping_counter:
            break

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_dones = dones.reshape((-1,))
        b_player_masks = player_masks.reshape((-1,)+envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # mask = rnd_model.make_template(b_player_pos)
        masked_b_obs = b_obs.clone().detach()
        mean_img = torch.from_numpy(obs_rms.mean).to(device)
        if args.use_template:
            masked_b_obs = masked_b_obs * b_player_masks + (1-b_player_masks) * mean_img
        obs_rms.update(masked_b_obs.cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        if args.use_mean:
            rnd_next_obs = (
                (
                    (masked_b_obs - mean_img)
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
            ).float()
        else:
            rnd_next_obs = masked_b_obs
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()
                
                if global_step > args.train_mask_at:
                    template.net.mask = True
                    
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        if update % args.template_train_every == 0 and args.use_template:
            b_obs = obs.swapdims(0, 1).reshape((-1,) + envs.single_observation_space.shape)
            b_actions = actions.swapdims(0, 1).reshape(-1)
            b_dones = dones.swapdims(0, 1).reshape(-1).cpu().bool().numpy()
            running_accuracy = []
            running_action_loss = []
            running_total_loss = []
            red_found = []
            
            epochs = next(args.template_training_schedule[1][i-1] for i, s in enumerate(args.template_training_schedule[0]) if update < s)
            writer.add_scalar("losses/action_training_epochs", epochs, global_step)
            for _ in range(epochs):
                for start, end in pairwise(range(0, len(b_inds), args.template_batch)):
                    mb_dones = b_dones[start:end]
                    mb_mask_inds = b_inds[start:end]
                    valid_inds = ((~mb_dones) & (mb_mask_inds != (len(b_obs)-1)))
                    mb_mask_inds = mb_mask_inds[valid_inds]

                    
                    
                    if len(mb_mask_inds) == 0:
                        continue
                    b_act_pred, _ = template(b_obs[mb_mask_inds],
                                                        b_obs[mb_mask_inds + 1])
                    b_act = F.one_hot(b_actions[mb_mask_inds].long(), action_n).float()
                    action_loss = mask_criterion(b_act_pred, b_act)
                    total_loss = action_loss
                    
                    running_accuracy += [multiclass_accuracy(b_act_pred.argmax(dim=-1), b_act.argmax(dim=-1), num_classes=int(action_n)).cpu(),]
                    running_action_loss += [action_loss.item(), ]
                    running_total_loss += [total_loss.item(), ]

                    mask_optimizer.zero_grad()
                    total_loss.backward()
                    mask_optimizer.step()
            writer.add_scalar("losses/action_loss", np.array(running_action_loss).mean(), global_step)
            writer.add_scalar("losses/total_action_loss", np.array(running_total_loss).mean(), global_step)
            writer.add_scalar("losses/action_accuracy", np.array(running_accuracy).mean(), global_step)

            b_obs_subset = b_obs[b_inds[:16]]
            if args.track and len(b_obs_subset) > 0:
                # Assuming b_obs_subset is a tensor
                _, att, raw_m = template.net.get_mask(b_obs_subset, full_output=True)
                m = (raw_m * att[...,None]).sum(1, keepdim=True)
                m = F.interpolate(m, obs_shape[-2:])
                raw_m = raw_m.mean(dim=1, keepdim=True)
                raw_m = F.interpolate(raw_m, obs_shape[-2:])

                masked = b_obs_subset*m
                mask = m.tile((1, 3, 1, 1))
                raw_m = raw_m.tile((1, 3, 1, 1))
                grid = torchvision.utils.make_grid(torch.cat([b_obs_subset, raw_m, mask, masked]), nrow=16, scale_each=True, normalize=True)
                writer.add_image("images/masked_images", grid, global_step)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

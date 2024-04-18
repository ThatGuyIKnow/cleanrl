import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
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

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

import visual_gridworld
from visual_gridworld.gridworld.minigrid_procgen import GridworldResizeObservation # absolute import

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    fixed: bool = False
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
    env_id: str = "Visual/DoorKey8x8-Gridworld-v0"
    """the id of the environment"""
    env_mode: Optional[str] = None
    """Environemt mode (random or hard)"""
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

    # Early stopping arguments
    early_stopping_threshold: float = None
    """margin for early stopping"""
    early_stopping_patience: int = int(1e5)
    """patience for early stopping"""

    # masking
    use_template: bool = False
    """use templating approach"""
    template_size: int = 3
    """masking template cell size"""

    recon_coef: float = 0.05
    """Reconstruction coef loss"""
    decorr_coef: float = 0.1
    """VAE channelwise decorrelation coef loss"""
    vae_update_freq: int = 1    
    """VAE update frequency"""
    vae_batch_size: int = 64
    """VAE batch size"""
    start_novelty_detector: int = 5000

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


def channelwise_covariance_loss(Z):
    """
    Computes the loss to encourage decorrelation between channels of the latent variable Z.
    
    Parameters:
    - Z: The latent variable with shape (B, N, C) where B is batch size, N is feature dimension, and C is channel count.
    
    Returns:
    - A scalar tensor representing the regularization loss.
    """
    # Calculate the mean of Z across the batch dimension, resulting in shape (N, C)
    Z_mean = torch.mean(Z, dim=0)
    
    # Center Z by subtracting the mean (broadcasting automatically applies)
    Z_centered = Z - Z_mean
    
    # Calculate the covariance matrix
    covariance_matrix = Z_centered.transpose(1, 2) @ Z_centered / (Z_centered.size(0) - 1)
    
    # Penalize the off-diagonal elements. Diagonal elements represent variance, which we don't want to penalize.
    off_diagonal_mask = 1 - torch.eye(Z.size(2), device=Z.device)
    off_diagonal_covariance = covariance_matrix * off_diagonal_mask
    
    # The loss is the sum of squares of off-diagonal elements
    loss = off_diagonal_covariance.pow(2).sum()
    return loss



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=512, latent_channels=8):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        
        self.cnn_out = 7*7*64

        # Encoder
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self.cnn_out, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.fc_mu = layer_init(nn.Linear(448, latent_dim * latent_channels))
        self.fc_log_var = layer_init(nn.Linear(448, latent_dim * latent_channels))
        
        # Decoder
        self.decoder_input = layer_init(nn.Linear(latent_dim * latent_channels, self.cnn_out))
        self.decoder = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 1, 8, stride=4)),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        mu = mu.view(-1, self.latent_dim, self.latent_channels)
        log_var = log_var.view(-1, self.latent_dim, self.latent_channels)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(std.shape[:-1])[...,None]
        return mu + eps * std

    def decode(self, z):
        z = z.view(-1, self.latent_dim * self.latent_channels)
        h = self.decoder_input(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.latent_dim = 2
        self.latent_channels = 32
        self.latent_size = self.latent_channels * self.latent_dim
        self.network = VAE(input_channels=4, latent_dim=self.latent_dim, latent_channels=self.latent_channels)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01)
        )
        
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network.encoder(x / 255.0)
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
        hidden = self.network.encoder(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)



VSA='MAP'

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size, out_features, vsa=VSA)
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
        self.encoder = Encoder(dim, 64, 1000)

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
        return (1 - torchhd.cosine_similarity(state_vector, memory).mean())

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
        if novelty < (min(self.sim_memory) + self.threshold):
            return False
        
        min_index = np.argmin(self.sim_memory)
        self.visited_states_memory.pop(min_index)
        self.sim_memory.pop(min_index)
        self.visited_states_memory.append(state_vector)
        self.sim_memory.append(novelty)
        return True


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64

        # Prediction network
        self._predictor = nn.Sequential(
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
        self._target = nn.Sequential(
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
        for param in self._target.parameters():
            param.requires_grad = False
        b, w, h = envs.get_grid().shape
        self.shape = torch.LongTensor((w, h)).to(device)
        self.obs_shape = torch.LongTensor(envs.observation_space.shape[-2:]).to(device)
        self.templates = self.construct_templates()

        # self.make_template = torch.jit.trace(self.make_template, (torch.rand(1) * self.obs_shape[0], torch.rand(1) * self.obs_shape[1]))

    def construct_templates(self):
        w, h = self.shape.cpu().numpy()
        x, y = self.obs_shape.cpu().numpy()
        lin_w = w - np.absolute(np.linspace(1-w, w-1, 2*w-1))
        lin_h = h - np.absolute(np.linspace(1-h, h-1, 2*h-1))
        t = np.outer(lin_w, lin_h)  
        t = torch.Tensor(t).unfold(1, w, 1) \
                     .unfold(0, h, 1) \
                     .reshape(-1, w, h)
        t = cv2.resize(t.swapdims(0, -1).numpy(), (x, y), interpolation=cv2.INTER_NEAREST_EXACT)
        t = torch.Tensor(t) > (w * (w - args.template_size))
        t = t.swapdims(-1, 0).bool()
        return t.to(device)
    
    def make_template(self, positions):
        pos = self.shape - positions - 1
        indices = pos[:, 1] * self.shape[0] + pos[:,0]
        m = self.templates[indices]
        return m[:, None]

    def predictor(self, x, pos):
        if not args.use_template:
            return self._predictor(x)
        m = self.make_template(pos)
        return self._predictor(x * m)

    def target(self, x, pos):
        if not args.use_template:
            return self._target(x)
        m = self.make_template(pos)
        return self._target(x * m)

    def forward(self, next_obs, pos):
        if args.use_template:
            m = self.make_template(pos)
            next_obs *= m

        target_feature = self._target(next_obs)
        predict_feature = self._predictor(next_obs)

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


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args.wandb_tags[0].split(','))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    early_stopping_counter = args.total_timesteps
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        cell_size=10,
        fixed=args.fixed,
        seed=args.seed,)
    if args.env_mode is not None:
        env_kwargs['mode'] = args.env_mode

    envs = gym.make(
        args.env_id,
        **env_kwargs
    )

    if args.early_stopping_threshold and hasattr(envs, 'max_reward'):
        args.max_reward = envs.max_reward * (1-args.early_stopping_threshold)

    envs = GridworldResizeObservation(envs, (84, 84))
    envs.num_envs = args.num_envs
    envs = RecordEpisodeStatistics(envs)
    envs = FirstChannelPositionWrapper(envs)

    obs_shape = envs.observation_space.shape
    agent = Agent(envs).to(device)
    novelty_detector = HDCNoveltyDetector3D(dim=1024, threshold=0.0001, memory_size=50)
    combined_parameters = list(agent.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, *obs_shape[1:]))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    player_pos = torch.zeros((args.num_steps, args.num_envs, 2), dtype=torch.int32).to(device)
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
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in tqdm(range(args.num_steps * args.num_iterations_obs_norm_init)):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, r, d, t, _ = envs.step(acs)
        next_ob += list(s)

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            obs_rms.update(next_ob)
            next_ob = []
    print("End to initialize...")

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            player_pos[step] = torch.Tensor(envs.get_player_position())
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

            hdc_obs = agent.network.encode(obs[step])
            hdc_next_obs = agent.network.encode(next_obs)
            curiosity_rewards[step] = args.int_coef * (novelty_detector.novelty(hdc_obs)-novelty_detector.novelty(hdc_next_obs))
            for idx, d in enumerate(done):
                if d:
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
                    writer.add_scalar("charts/rooms_visited", info["rooms_visited"][idx], global_step)

                    if args.max_reward is not None and \
                            args.max_reward > epi_ret and \
                            early_stopping_counter == args.total_timesteps:
                        early_stopping_counter = global_step + args.early_stopping_patience
                    else:
                        early_stopping_counter = args.total_timesteps
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
        b_player_pos = player_pos.reshape((-1, 2))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        obs_rms.update(b_obs.cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        rnd_next_obs = (
            (
                (b_obs - torch.from_numpy(obs_rms.mean).to(device))
                / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
            ).clip(-5, 5)
        ).float()

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()
                if global_step % args.vae_update_freq == 0:
                    no_minibatches = len(b_inds)//args.vae_batch_size
                    for batch_inds in np.split(b_inds, no_minibatches):
                        # Reconstruction loss
                        mu, log_var = agent.network.encode(b_obs[batch_inds] / 255.)
                        z = agent.network.reparameterize(mu, log_var)
                        recon = agent.network.decode(z)

                        rc_loss = F.huber_loss(recon, b_obs[batch_inds, -1:] / 255.)
                        kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                        c_decorr_loss = channelwise_covariance_loss(mu)

                        loss = args.recon_coef * (rc_loss + kl_div_loss + args.decorr_coef * c_decorr_loss)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()



                        if global_step > args.start_novelty_detector:
                            for obs in mu:
                                obs_enc = novelty_detector.encode_state(torch.Tensor(mu))
                                nov = novelty_detector.novelty(obs_enc)
                                novelty_detector.add_to_memory(obs_enc, nov)


            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

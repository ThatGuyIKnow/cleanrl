# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union
from gymnasium import spaces

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)

from torch.utils.tensorboard import SummaryWriter

# Network = namedtuple('Network', ['network', 'optimizer', 'target_network'])
class Network(NamedTuple):
    network: nn.Module
    optimizer: optim.Optimizer
    target_network: nn.Module
    
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    aux_rewards: torch.Tensor
    new_partition: torch.Tensor

class EEReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    future_observations: torch.Tensor
    first_aux_rewards: torch.Tensor
    discounted_aux_rewards: torch.Tensor


class PelletReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.aux_rewards = np.zeros((self.buffer_size, self.n_envs, self.action_space.n), dtype=action_space.dtype)
        self.new_partition = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        aux_reward: np.ndarray,
        new_partition: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:        
        super().add(obs, next_obs, action, reward, done, infos)

        self.aux_rewards[self.pos] = np.array(aux_reward).copy()
        self.new_partition[self.pos] = np.array(new_partition).copy()



    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = 0

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.aux_rewards[batch_inds, env_indices],
            self.new_partition[batch_inds, env_indices],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


    def sample_state_pairs(self, batch_size: int) -> Optional[EEReplayBufferSamples]:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
       
        eps_starts = []
        eps_ends = []
        aux_rewards = []
        aux_rewards_index = []
        size = self.size()
        for index in batch_inds:
            if self.dones[index] or self.pos == (index):
                continue

            for offset in range(1, size):
                arr_index = (index + offset) % size
                if self.dones[arr_index] or self.pos == (arr_index) or offset+1 == size:
                    end_offset = np.random.randint(1, offset + 2)
                    end_index = (end_offset + index) % self.buffer_size

                    if offset == 1:
                        aux = torch.ones((self.action_space.n, )) / self.action_space.n
                    else:
                        aux = np.take(self.aux_rewards[:, 0], range(index, index + end_offset), axis=0, mode='wrap')
                        gammas = args.gamma**np.arange(end_offset)
                        # Vector-Matrix multiplication + Sum
                        aux = torch.Tensor(np.einsum('i,ik', gammas, aux))

                    aux_rewards_index.append(index)
                    eps_starts.append((index + 1) % size)
                    aux_rewards.append(aux)
                    eps_ends.append(end_index)
                    break
        
        if len(eps_starts) == 0:
            return None
        
        env_indices = np.zeros((len(eps_starts), ), dtype=np.int64)
        
        data = (
            self.observations[eps_starts, env_indices, -1, :],
            self.observations[eps_ends, env_indices, -1, :],
            self.aux_rewards[aux_rewards_index, env_indices],
            torch.stack(aux_rewards)
        )
        return EEReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RecordVideoWandb(gym.wrappers.RecordVideo):
    """This wrapper records videos of rollouts. 
    It is based on the gymnasium RecordVideo wrapper.
    In addition, it will send the video the wandb if initialized.
    """

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        super().close_video_recorder()
        if self.recording and wandb.run is not None:
            wandb.Video(self.video_recorder.path)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--capture-video-length", type=int, default=500, nargs="?", const=True,
        help="the length of the capture video")
    parser.add_argument("--capture-video-step", type=int, default=1e6, nargs="?", const=True,
        help="how often to capture video")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--checkpoint-ee", type=int, default=None, nargs="?", const=True,
        help="whether to save checkpoints of EE model into the `runs/{run_name}/EE/` folder")
    parser.add_argument("--checkpoint-q", type=int, default=None, nargs="?", const=True,
        help="whether to save checkpoints of Q model into the `runs/{run_name}/Q/` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=5,
        help="partitions to start learning q")
    parser.add_argument("--partition-starts", type=int, default=8*1e6,
        help="steps to start partitioning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--partition-delta", type=int, default=80000,
        help="the frequency of partitioning")
    parser.add_argument("--partition-time-multiplier", type=float, default=1.2,
        help="the increase in partitioning frequency")
    parser.add_argument("--beta", type=float, default=1,
        help="constant bonus factor for pellets")
    parser.add_argument("--eta", type=float, default=0.1,
        help="monte carlo mixing factor for ee")
    parser.add_argument("--mc_clip", type=float, default=0.5,
        help="monte carlo clipping factor for ee")
    parser.add_argument("--terminal-on-loss", type=bool, default=True,
        help="should agent terminate when losing a life")
    args = parser.parse_args()
    # fmt: on

    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideoWandb(env, f"videos/{run_name}", step_trigger=lambda x: (x % args.capture_video_step) == 0, video_length=args.capture_video_length)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# class QNetworkPellet(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         self.ffn = nn.Sequential(
#             nn.Linear(3136 + 1, 512),
#             nn.ReLU(),
#             nn.Linear(512, env.action_space.n),
#         )

#     def forward(self, x, pellet_value):
#         x = self.conv(x / 255.0)
#         return self.ffn(torch.cat([x, pellet_value]))


class EENetwork(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.prong = nn.Sequential(
            # nn.Flatten(1, 2),
            nn.Conv2d(1, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.body = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        x1, x2 = torch.split(x / 255.0, split_size_or_sections=1, dim=1)

        x1 = self.prong(x1)
        x2 = self.prong(x2)

        x = self.body(torch.sub(x1, x2))

        return x

def calculate_aux_reward(q_values: torch.Tensor, action: int):
    aux_reward = q_values.clone()
    mag = aux_reward.abs().sum()
    aux_reward = - (aux_reward / mag) + aux_reward.min()

    aux_reward[action].add(1)

    return aux_reward

def euclidian_distance(s: torch.Tensor):
    return s.pow(2).sum(dim=1).sqrt()


def distance(s1: torch.Tensor, s2: torch.Tensor, repr_states: torch.Tensor, distance_func: Callable[[torch.Tensor], torch.Tensor]):
    N = len(repr_states)
    tile_shape = [N, ] + [1, ] * (len(repr_states.shape) - 1)

    s1_tile = torch.tile(torch.unsqueeze(s1, dim=0), tile_shape)
    s2_tile = torch.tile(torch.unsqueeze(s2, dim=0), tile_shape)

    s_hat = torch.unsqueeze(repr_states, dim=1)
    s1_tile = torch.unsqueeze(s1_tile, dim=1)
    s2_tile = torch.unsqueeze(s2_tile, dim=1)
    
    # s_hat & s1
    t1 = torch.concat([s_hat, s1_tile], dim=1)   
    # s_hat & s2
    t2 = torch.concat([s_hat, s2_tile], dim=1)
    # s1 & s_hat
    t3 = torch.concat([s1_tile, s_hat], dim=1)
    # s2 & s_hat
    t4 = torch.concat([s2_tile, s_hat], dim=1)

    samples = torch.concat([t1, t2, t3, t4], dim=0)
    distances = distance_func(samples)

    t1, t2, t3, t4 = torch.split(distances, split_size_or_sections=N, dim=0)

    return torch.max(euclidian_distance(t1 - t2), euclidian_distance(t3 - t4))

def closest_partition(s: np.array, repr_states: torch.Tensor, distance_func: Callable[[torch.Tensor], torch.Tensor]):
    # Get the distances of all partitions
    with torch.no_grad():
        state = torch.Tensor(s[-1]).to(device)
        distances = list([distance(state, s_hat, repr_states, distance_func).max() for s_hat in repr_states])

    # Find the closest partition
    partition_distances = torch.stack(distances)
    min_index = partition_distances.argmin()

    return repr_states[min_index], min_index, partition_distances[min_index]

def linear_schedule(start_e: float, end_e: float, duration: int, t: int, start: int = 0):
    # Evaluate if the starting step has NOT been reached OR if we decided not to run
    if t < start or start == -1:
        return start_e
    slope = (end_e - start_e) / duration
    return max(slope * (t - start) + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
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

    # Setup run directory
    if args.checkpoint_q is not None:
        os.mkdir(f"runs/{run_name}/Q")
    if args.checkpoint_ee is not None:
        os.mkdir(f"runs/{run_name}/EE")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    env= make_env(args.env_id, args.seed, 0, args.capture_video, run_name)()
    
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(env).to(device)
    q_optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    q_target_network = QNetwork(env).to(device)
    q_target_network.load_state_dict(q_network.state_dict())
    q = Network(q_network, q_optimizer, q_target_network)

    ee_network = EENetwork(env).to(device)
    ee_optimizer = optim.Adam(ee_network.parameters(), lr=args.learning_rate)
    ee_target_network = EENetwork(env).to(device)
    ee_target_network.load_state_dict(ee_network.state_dict())
    ee = Network(ee_network, ee_optimizer, ee_target_network)


    rb = PelletReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    empty_partition = torch.Tensor(env.observation_space.sample()[-1, :, :] * 0)
    partitions = torch.unsqueeze(torch.Tensor(obs[-1]), dim=0).to(device)
    time_to_partition = 0
    visited_partitions = set()
    visitation_counts = torch.zeros((1,)).to(device)
    distance_from_partition = 0
    partition_add_threshold = args.partition_delta
    time_since_reward = 0
    q_learning_started = -1
    last_life = 99999999
    epsiode = 0
    epsilon = args.start_e

    # Variables for plotting purposes
    episodic_return = 0
    episodic_length = 0

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step <= args.partition_starts:
            # ==============================
            # PRETRAINING OF THE EE FUNCTION
            # ==============================
            q_values = torch.ones((env.action_space.n, )) / env.action_space.n
            actions = env.action_space.sample() 

            # Calculate Auxilary Reward of EE
            aux_reward = calculate_aux_reward(q_values, actions).cpu()
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminated, truncated, info = env.step(actions)


            # Determin the current partition
            # Update the set of visited partitions
            partition_distance = 0
            index = -1
            visited_partitions_next = set()

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs[-1].copy()
            
            # Don't care about partitions in this step
            new_partition = -1
        else:
            # ============================================
            # AFTER PRETRAINING OF THE EE FUNCTION IS DONE
            # ============================================
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step, q_learning_started)
            if random.random() < epsilon or time_since_reward > 500:
                q_values = torch.ones((env.action_space.n, )) / env.action_space.n
                actions = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q.network(torch.Tensor(np.array(obs)).to(device).unsqueeze(dim=0))[0]
                    actions = torch.argmax(q_values).cpu().numpy()   

            # Calculate Auxilary Reward of EE
            aux_reward = calculate_aux_reward(q_values, actions).cpu()
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminated, truncations, info = env.step(actions)

            # Determin the current partition
            # Update the set of visited partitions
            curr_partition, index, partition_distance = closest_partition(next_obs, partitions, ee.network)
            visited_partitions_next = visited_partitions.copy().union([index.item(), ])

            if rewards and index in visited_partitions:
                time_since_reward += 1
            else:
                time_since_reward = 0

            # Update the best candidate to the distance measure
            if partition_distance > distance_from_partition:
                next_partition = next_obs[-1].copy()
                distance_from_partition = partition_distance


            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs[-1].copy()
            
            # Add to replay buffer if we've discovered a new partition
            new_partition = visited_partitions_next.difference(visited_partitions)
            if len(new_partition) == 0:
                new_partition = -1
            else:
                new_partition = new_partition.pop()

            
        if args.terminal_on_loss:
            life = info['lives']
                
        rb.add(obs, real_next_obs, actions, rewards, terminated, aux_reward, new_partition, info)
        # Adding new partition logic
        if global_step > args.partition_starts:
            time_to_partition += 1
            if time_to_partition > partition_add_threshold:
                new_partition = np.expand_dims(next_partition, axis=0)
                new_partition = torch.Tensor(new_partition).to(device)
                
                partitions = torch.cat([partitions, new_partition])
                visitation_counts = torch.cat([visitation_counts, torch.zeros((1,), device=device)])
                partition_add_threshold *= args.partition_time_multiplier
                distance_from_partition = 0
                time_to_partition = 0

                writer.add_scalar("partitions/no_of_partitions", partitions.shape[0], global_step)

        # On epsiode termination
        if truncated or terminated or (args.terminal_on_loss and life < last_life):
            # Reset epsisode variables
            distance_from_partition=0
            time_since_reward = 0
            visited_partitions_next = set()
            next_obs, _ = env.reset()
            last_life = 999999

            # Plotting
            intrinsic_reward = torch.gather(visitation_counts, 0, torch.LongTensor(list(visited_partitions)).to(device))
            intrinsic_reward = (args.beta / intrinsic_reward.sqrt()).sum().item()
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            writer.add_scalar("rewards/intrinsic_reward", intrinsic_reward, global_step)
            writer.add_scalar("rewards/episodic_return", episodic_return, global_step)
            writer.add_scalar("rewards/total_reward", episodic_return + intrinsic_reward, global_step)
            writer.add_scalar("partitions/no_visited_partitions", len(visited_partitions), global_step)

            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
            writer.add_scalar("charts/epsiode", epsiode, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

            writeable_visitations = visitation_counts.cpu().numpy()
            for i, count in enumerate(writeable_visitations):
                writer.add_scalar(f"visitation_counts/partition_{i}", count, global_step=global_step)

            # Reset plotting variables
            episodic_return = 0
            episodic_length = 0
            epsiode += 1

            # Update visitation count
            for partition_index in visited_partitions:
                visitation_counts[partition_index] += 1
        
        # UPDATING PLOTTING VARIABLES
        episodic_return += rewards
        episodic_length += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        visited_partitions = visited_partitions_next

        if args.terminal_on_loss:
            last_life = life

        # ALGO LOGIC: training Q.
        if global_step > args.partition_starts:
            if q_learning_started == -1 and partitions.shape[0] >= args.learning_starts:
                q_learning_started = global_step
                
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    target_max, _ = q.target_network(data.next_observations).max(dim=1)
                    rewards = data.rewards.flatten()
                    
                    # Adding intrinsic reward
                    temp_visitation_counts = torch.cat([torch.zeros((1,), device=device), visitation_counts])
                    visitations = torch.gather(temp_visitation_counts, 0, data.new_partition + 1)
                    mask = (visitations != 0)
                    
                    rewards[mask] += args.beta / torch.sqrt(visitations[mask])

                    td_target = rewards + args.gamma * target_max * (1 - data.dones.flatten())
                    
                old_val = q.network(data.observations).gather(1, data.actions).squeeze()
                loss = F.smooth_l1_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("q_losses/td_loss", loss, global_step)
                    writer.add_scalar("q_losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                q.optimizer.zero_grad()
                loss.backward()
                q.optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(q.target_network.parameters(), q.network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
                
            if args.checkpoint_q is not None and global_step % args.checkpoint_q == 0:
                model_path = f"runs/{run_name}/Q/{global_step}_{args.exp_name}.cleanrl_model"
                torch.save(q.network.state_dict(), model_path)
                if args.track:
                    wandb.save(model_path, policy="now")
                print(f"Q model saved to {model_path}")


        # ALGO LOGIC: training EE.
        if global_step % args.train_frequency == 0:
            data: EEReplayBufferSamples = rb.sample_state_pairs(args.batch_size)
            if data is not None:
                input = torch.stack([data.observations, data.future_observations], dim=1)
                
                pred = ee.network(input)
                target_onestep = data.first_aux_rewards + args.gamma * pred
                target_mc = data.discounted_aux_rewards

                delta_mc = torch.clip(target_mc - target_onestep, -args.mc_clip, args.mc_clip)

                target = (1 - args.eta) * target_onestep + args.eta * delta_mc
                loss = F.mse_loss(target, pred)

                if global_step % 100 == 0:
                    print(f"Training EE. Loss: {loss}")
                    writer.add_scalar("ee_losses/ee_loss", loss, global_step)
                    writer.add_scalar("ee_losses/ee_distance_predicted", euclidian_distance(target_onestep).mean().item(), global_step)
                    writer.add_scalar("ee_losses/ee_distance", euclidian_distance(target_mc).mean().item(), global_step)

                # optimize the model
                ee.optimizer.zero_grad()
                loss.backward()
                ee.optimizer.step()

                if args.checkpoint_ee is not None and global_step % args.checkpoint_ee == 0:
                    model_path = f"runs/{run_name}/EE/{global_step}_{args.exp_name}.cleanrl_model"
                    torch.save(ee.network.state_dict(), model_path)
                    if args.track:
                        wandb.save(model_path, policy="now")
                    print(f"EE model saved to {model_path}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q.network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        model_path = f"runs/{run_name}/{args.exp_name}_ee.cleanrl_model"
        torch.save(ee.network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    env.close()
    writer.close()

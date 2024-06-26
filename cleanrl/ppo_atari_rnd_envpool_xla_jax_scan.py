# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_tag: str = None
    """the tag of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
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
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rnd_alpha: float = 1.0
    """The alpha weighting for intrinsic reward"""
    rnd_update_porpotions: float = 0.25
    """Random dropout for rand update"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)

class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class RND(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = x.reshape((x.shape[0], -1))
        return x

class RNDStaticEmbedding(RND):
    @nn.compact
    def __call__(self, x):
        x = super().__call__(x)
        return nn.Dense(512)(x)
    
class RNDPredictionEmbedding(RND):
    @nn.compact
    def __call__(self, x):
        x = super().__call__(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512, 512)(x)
        x = nn.relu(x)
        x = nn.Dense(512, 512)(x)
        return x

@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array
    static_embedding: jnp.array
    predicted_embedding: jnp.array
    intrinsic_value: jnp.array
    intrinsic_reward: jnp.array
    intrinsic_returns: jnp.array
    intrinsic_advantages: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            tags=[args.wandb_tag] if args.wandb_tag is not None else None,
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key, rnd_static_key, rnd_predict_key = jax.random.split(key, 6)

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs)()
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    handle, recv, send, step_env = envs.xla()

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, info) = step_env(handle, action)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_length, episode_stats.returned_episode_lengths
            ),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    rnd_static = RNDStaticEmbedding()
    rnd_predictor = RNDPredictionEmbedding()
    intrinsic_critic = Critic()


    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    params = {
            'ppo': {
                'network': network_params, 
                'actor': actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
                'critic': critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
                'intrinsic_critic': intrinsic_critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
            },
            'rnd':{
                'static': rnd_static.init(rnd_static_key, np.array([envs.single_observation_space.sample()])),
                'predictor': rnd_predictor.init(rnd_predict_key, np.array([envs.single_observation_space.sample()])),
        }
    }
    
    partition_optimizers = {
        'trainable': optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ), 
        'frozen': optax.set_to_zero()}
    param_partitions = flax.traverse_util.path_aware_map(
        lambda path, v: 'frozen' if 'static' in path else 'trainable', params)
    tx = optax.multi_transform(partition_optimizers, param_partitions)
    
    agent_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    rnd_static.apply = jax.jit(rnd_static.apply)
    rnd_predictor.apply = jax.jit(rnd_predictor.apply)
    intrinsic_critic.apply = jax.jit(intrinsic_critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
         # .network_params
        hidden = network.apply(agent_state.params['ppo']['network'], next_obs)
        # actor_params
        logits = actor.apply(agent_state.params['ppo']['actor'], hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # critic_params
        value = critic.apply(agent_state.params['ppo']['critic'], hidden)
        int_value = critic.apply(agent_state.params['ppo']['intrinsic_critic'], hidden)
        return action, logprob, value.squeeze(1), int_value.squeeze(1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        # network_params
        hidden = network.apply(params['ppo']['network'], x)
        # actor_params
        logits = actor.apply(params['ppo']['actor'], hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        # critic_params
        value = critic.apply(params['ppo']['critic'], hidden).squeeze()
        intrinsic_value = critic.apply(params['ppo']['intrinsic_critic'], hidden).squeeze()
        return logprob, entropy, value, intrinsic_value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        # critic_params, network_params
        next_value = critic.apply(
            agent_state.params['ppo']['critic'], network.apply(agent_state.params['ppo']['network'], next_obs)
        ).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        intrinsic_advantages = jnp.zeros((args.num_envs,))
        intrinsic_dones = jnp.full_like(dones, False, dtype=bool)
        intrinsic_values = jnp.concatenate([storage.intrinsic_value, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        _, intrinsic_advantages = jax.lax.scan(
            compute_gae_once, intrinsic_advantages, (intrinsic_dones[1:], intrinsic_values[1:], intrinsic_values[:-1], storage.intrinsic_reward), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            intrinsic_advantages=intrinsic_advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, int_value, pred_emb, static_emb, int_advantage):
        newlogprob, entropy, newvalue, newintvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        mb_advantages += int_advantage

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        i_loss = 0.5 * ((newintvalue - int_value) ** 2).mean()

        entropy_loss = entropy.mean()

        # RND loss
        rnd_loss = optax.l2_loss(pred_emb, static_emb).mean(-1)
        mask = jax.random.bernoulli(rnd_predict_key, p=args.rnd_update_porpotions,
                                    shape=rnd_loss.shape).astype(jnp.float32)
        rnd_loss = (rnd_loss * mask).sum() / jnp.clip(rnd_loss.sum(), a_min=-jnp.inf, a_max=1.)


        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + rnd_loss + i_loss
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), rnd_loss)

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.intrinsic_value,
                    minibatch.predicted_embedding,
                    minibatch.static_embedding,
                    minibatch.intrinsic_advantages
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn):
        agent_state, episode_stats, obs, done, key, handle = carry
        action, logprob, value, int_value, key = get_action_and_value(agent_state, obs, key)

        episode_stats, handle, (next_obs, reward, next_done, _) = env_step_fn(episode_stats, handle, action)

        # rnd_static_params
        static_embedding = rnd_static.apply(agent_state.params['rnd']['static'], next_obs)
        # rnd_predict_params
        predictor_embedding = rnd_predictor.apply(agent_state.params['rnd']['predictor'], next_obs)
        intrinsic_reward = args.rnd_alpha * jax.numpy.linalg.norm((static_embedding - predictor_embedding), ord=2)
        
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            predicted_embedding=predictor_embedding,
            static_embedding=static_embedding,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
            intrinsic_value=int_value,
            intrinsic_reward=intrinsic_reward,
            intrinsic_returns=jnp.zeros_like(reward),
            intrinsic_advantages=jnp.zeros_like(intrinsic_reward),
        )
        return ((agent_state, episode_stats, next_obs, next_done, key, handle), storage)

    def rollout(agent_state, episode_stats, next_obs, next_done, key, handle, step_once_fn, max_steps):
        (agent_state, episode_stats, next_obs, next_done, key, handle), storage = jax.lax.scan(
            step_once_fn, (agent_state, episode_stats, next_obs, next_done, key, handle), (), max_steps
        )
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle

    rollout = partial(rollout, step_once_fn=partial(step_once, env_step_fn=step_env_wrappeed), max_steps=args.num_steps)

    for iteration in range(1, args.num_iterations + 1):
        iteration_time_start = time.time()
        agent_state, episode_stats, next_obs, next_done, storage, key, handle = rollout(
            agent_state, episode_stats, next_obs, next_done, key, handle
        )
        global_step += args.num_steps * args.num_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, rnd_loss, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        avg_episodic_intrisic_reward = np.mean(jax.device_get(episode_stats.returned_episode_returns))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_intrisic_reward", avg_episodic_intrisic_reward, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        writer.add_scalar("charts/learning_rate", agent_state.opt_state.inner_states['trainable'].inner_state[1].hyperparams['learning_rate'].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/rnd_loss", rnd_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)), global_step
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params['ppo']['network'], # .network_params,
                            agent_state.params['ppo']['actor'], # .actor_params,
                            agent_state.params['ppo']['critic'], # .critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-sppoeed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()

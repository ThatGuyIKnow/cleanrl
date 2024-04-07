from multiprocessing import Pipe, Process
import numpy as np
from tqdm import tqdm
import visual_gridworld # absolute import
import pickle
import cloudpickle

import time
import gymnasium as gym
import datetime as dt
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    num_envs=16
    e = gym.make('Visual/DoorKey6x6-Gridworld-v0', cell_size=14, num_envs=num_envs, render_mode='rgb_array')
    e.reset()

    step_count = int(2e5 / num_envs)
    start_time = dt.datetime.now()
    for _ in tqdm(range(step_count)):
        a = [ e.action_space.sample(), ] * num_envs
        observation, reward, done, truncated, info = e.step(a)
    end_time = dt.datetime.now()

    print(f"Time for {num_envs*step_count} steps: {end_time - start_time}. SPS: {num_envs*step_count/(end_time-start_time).total_seconds()}")


    
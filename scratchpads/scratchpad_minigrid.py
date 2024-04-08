from multiprocessing import Pipe, Process
from typing import Callable, Dict, List
import numpy as np
from tqdm import tqdm
import visual_gridworld # absolute import
import pickle
import cloudpickle

import time
import gymnasium as gym
import datetime as dt

from multi_subproc import SubprocVecEnv


class MultiSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: str | None = None):
        super().__init__(env_fns, start_method)
        self.no_proc = len(env_fns)


    def step(self, actions: np.ndarray) -> cloudpickle.Tuple[np.ndarray | Dict[str, np.ndarray] | cloudpickle.Tuple[np.ndarray] | List[Dict]]:
        actions = np.split(actions, self.no_proc)
        return super().step(actions)

if __name__ == "__main__":
    num_envs = 16
    no_proc = 1
    e = gym.make('Visual/MultiRoomS5N4-Gridworld-v0', cell_size=14, num_envs=num_envs, render_mode='human')
    e = visual_gridworld.NoisyGridworldWrapper(e)
    e.reset()

    step_count = int(2e6 / (num_envs * no_proc))
    start_time = dt.datetime.now()
    for _ in tqdm(range(step_count)):
        a = np.concatenate([e.action_space.sample(), ] * no_proc)
        observation, reward, done, truncated, info = e.step(a)
        e.render()
    end_time = dt.datetime.now()

    print(f"Time for {no_proc*num_envs*step_count} steps: {end_time - start_time}. SPS: {no_proc*num_envs*step_count/(end_time-start_time).total_seconds()}")

    
    
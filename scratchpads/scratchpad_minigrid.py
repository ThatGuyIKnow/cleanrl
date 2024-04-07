import visual_gridworld # absolute import

import time
import gymnasium as gym
import datetime as dt
# from ... import envs

e = gym.make('Visual/DoorKey5x5-Gridworld-v0', render_mode='human')
e.reset()
e.render()


step_count = int(2e5)
start_time = dt.datetime.now()
for _ in range(step_count):
    a = e.action_space.sample()
    observation, reward, done, truncated, info = e.step(a)
    if done or truncated:
        e.reset()
    e.render()
    time.sleep(0.05)
end_time = dt.datetime.now()

print(f"Time for {step_count} steps: {end_time - start_time}. SPS: {step_count/(end_time-start_time).total_seconds()}")

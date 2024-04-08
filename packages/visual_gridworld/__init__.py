
import gymnasium
from gymnasium.envs.registration import register
from visual_gridworld.gridworld.minigrid_procgen import (DoorKey5x5Gridworld, 
                                                         DoorKey6x6Gridworld,
                                                         DoorKey8x8Gridworld,
                                                         DoorKey16x16Gridworld,
                                                         NoisyDoorKey6x6Gridworld,
                                                         MultiRoomS4N2GridWorld,
                                                         MultiRoomS5N4GridWorld,
                                                         MultiRoomS10N6GridWorld
                                                         )

envs = [
    # MultiRoom
    MultiRoomS4N2GridWorld,
    MultiRoomS5N4GridWorld,
    MultiRoomS10N6GridWorld,
    # DoorKey
    DoorKey5x5Gridworld, 
    DoorKey6x6Gridworld, 
    DoorKey8x8Gridworld, 
    DoorKey16x16Gridworld, 
    # Noise DoorKey
    NoisyDoorKey6x6Gridworld]

for env_cls in envs:
    register(
        id=f"Visual/{env_cls.env_name}",
        entry_point=f"visual_gridworld.gridworld.minigrid_procgen:{env_cls.__name__}",
        max_episode_steps=env_cls.max_episode_steps,
    )

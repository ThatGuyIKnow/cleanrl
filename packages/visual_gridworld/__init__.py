
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import pygame
from visual_gridworld.gridworld.minigrid_procgen import (DoorKey5x5Gridworld, 
                                                         DoorKey6x6Gridworld,
                                                         DoorKey8x8Gridworld,
                                                         DoorKey16x16Gridworld, 
                                                         NoisyDoorKey16x16Gridworld, 
                                                         NoisyDoorKey5x5Gridworld,
                                                         NoisyDoorKey6x6Gridworld,
                                                         MultiRoomS4N2GridWorld,
                                                         MultiRoomS5N4GridWorld,
                                                         MultiRoomS10N6GridWorld, 
                                                         NoisyDoorKey8x8Gridworld, 
                                                         NoisyMultiRoomS10N6GridWorld, 
                                                         NoisyMultiRoomS4N2GridWorld, 
                                                         NoisyMultiRoomS5N4GridWorld,
                                                         
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
    # Noise MultiRoom
    NoisyMultiRoomS4N2GridWorld,
    NoisyMultiRoomS5N4GridWorld,
    NoisyMultiRoomS10N6GridWorld,
    # Noise DoorKey
    NoisyDoorKey5x5Gridworld,
    NoisyDoorKey6x6Gridworld,
    NoisyDoorKey8x8Gridworld,
    NoisyDoorKey16x16Gridworld,
    # Blocky Background Multiroom
    ]

for env_cls in envs:
    register(
        id=f"Visual/{env_cls.env_name}",
        entry_point=f"visual_gridworld.gridworld.minigrid_procgen:{env_cls.__name__}",
    )


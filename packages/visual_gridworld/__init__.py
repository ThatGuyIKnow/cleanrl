
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import pygame
from visual_gridworld.gridworld.minigrid_procgen import (DoorKey5x5Gridworld, 
                                                         DoorKey6x6Gridworld,
                                                         DoorKey8x8Gridworld,
                                                         DoorKey16x16Gridworld,
                                                         NoisyDoorKey6x6Gridworld,
                                                         MultiRoomS4N2GridWorld,
                                                         MultiRoomS5N4GridWorld,
                                                         MultiRoomS10N6GridWorld,
                                                         Gridworld,
                                                         CellType
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



class NoisyGridworldWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: Gridworld):
        super().__init__(env)
        self.env = env
        
    def observation(self, observation):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        c = self.env.cell_size
        grid = self.env.grids
        noise = np.random.randint(0, 255, self.env.observation_space.shape, dtype=np.uint8)
        floor_index = np.where(grid == CellType.FLOOR.value)
        floor_mask = np.zeros_like(grid, dtype=np.uint8)
        floor_mask[floor_index] = 1
        floor_mask = floor_mask.repeat(c, axis=-2).repeat(c, axis=-1)[...,None]

        x_full, y_full = np.rollaxis(self.env.player_positions * c, axis=1)
        player = self.env.cell_render[CellType.PLAYER][self.env.player_directions]
        player_mask, _ = np.rollaxis(player, axis=1)
        for i, (x, y, mask) in enumerate(zip(x_full, y_full, player_mask)):
            floor_mask[i, x:x+c, y:y+c] *= mask[...,:1]

        observation = observation * (1-floor_mask) + floor_mask * noise
        return observation
    
    def render(self):
        self.env.screen.fill((255, 255, 255))  # Fill the screen with white
        # draw the array onto the surface
        w,h = self.env.screen_size
        obs = self.observation(self.env.rgb_render())
        screen = np.zeros((w * 4, h * 4, 3), dtype=np.uint8)
        for index, o in enumerate(obs):
            i = (index % 4)*w
            j = index // 4*h
            screen[i:i+w,j:j+h] = o
        surf = pygame.surfarray.make_surface(screen)
        self.env.screen.blit(surf, (0, 0))
        pygame.display.update()
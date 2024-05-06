
import abc
import math
import sys
from typing import Dict, List, Literal
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from gymnasium.error import DependencyNotInstalled
from enum import Enum

class Direction(Enum):
    UP=0
    RIGHT=1
    DOWN=2
    LEFT=3

    @classmethod
    def turn_left(cls, dir, amount=1):
        if not isinstance(dir, Enum):
            return cls.turn(dir, -amount)
        return cls((4 + dir.value-amount) % 4)

    @classmethod
    def turn_right(cls, dir, amount=1):
        if not isinstance(dir, Enum):
            return cls.turn(dir, amount)
        return cls((dir.value + amount) % 4)

    @classmethod
    def turn(cls, dir, amount=1):
        return (dir + amount) % 4

    @classmethod
    def to_vectors(cls):
        return [(-1,0),(0,1),(1,0),(0,-1)]

class CellType(Enum):
    NOTHING=-1
    FLOOR=0
    WALL=1
    DOOR=2
    KEY_DOOR=3
    KEY=4
    GOAL=5
    PLAYER=6


class GridWorldGeneration(abc.ABC):
    def __init__(self, seed, fixed) -> None:
        super().__init__()
        self.random = np.random.RandomState(seed)
        self.seed = seed
        self.fixed = fixed

    def set_random(self, random: np.random.RandomState):
        self.random = random

    @abc.abstractmethod
    def generate_grid_world(self):
        pass

class GridWorldTiles(abc.ABC):
    def __init__(self, cell_size, tile_count, seed=None) -> None:
        super().__init__()
        self.cell_size = cell_size
        self.tile_count = tile_count
        self.random = np.random.RandomState(seed)

        self.tiles = self._preload_tiles()

    def set_random(self, random: np.random.RandomState):
        self.random = random

    def construct_tiles(self):
        cell_size = self.cell_size
        tile_index = [self.random.choice(v) for v  in self.tile_count]
        ids = [CellType.FLOOR, CellType.WALL, CellType.GOAL, CellType.KEY, CellType.DOOR, CellType.KEY_DOOR]
        tiles = {id: self.tiles[id][tile_i] for id, tile_i in zip(ids, tile_index)}

        # Select the correct player tile and correct background tile
        tiles[CellType.PLAYER] = self.tiles[CellType.PLAYER][tile_index[0]]

        tiles[CellType.FLOOR] = self.tiles[CellType.FLOOR][tile_index[0]]
        tiles[CellType.WALL] = self.tiles[CellType.WALL][tile_index[1]]
        tiles[CellType.GOAL] = self.tiles[CellType.GOAL][tile_index[2]]
        tiles[CellType.KEY] = self.tiles[CellType.KEY][tile_index[3]]
        tiles[CellType.DOOR] = self.tiles[CellType.DOOR][tile_index[4]]
        tiles[CellType.KEY_DOOR] = self.tiles[CellType.KEY_DOOR][tile_index[5]]

        return tiles

    def _preload_tiles(self):
        cell_size = self.cell_size
        tiles: Dict[CellType, np.typing.NDArray] = {}
        tiles[CellType.FLOOR] = self.construct_floors(cell_size)
        tiles[CellType.WALL] = self.construct_walls(cell_size)
        tiles[CellType.GOAL] = self.construct_goals(cell_size)
        tiles[CellType.KEY] = self.construct_keys(cell_size)
        tiles[CellType.DOOR] = self.construct_key_doors(cell_size)
        tiles[CellType.KEY_DOOR] = self.construct_doors(cell_size)
        tiles[CellType.PLAYER] = self.construct_players(cell_size)
        


        return tiles

    @abc.abstractmethod
    def construct_floors(self):
        pass
    @abc.abstractmethod
    def construct_walls(self):
        pass
    @abc.abstractmethod
    def construct_goals(self):
        pass
    @abc.abstractmethod
    def construct_keys(self):
        pass
    @abc.abstractmethod
    def construct_doors(self):
        pass
    @abc.abstractmethod
    def construct_players(self):
        pass

class RandomLuminationTiles(GridWorldTiles):
    def __init__(self, cell_size, seed) -> None:
        super().__init__(cell_size, [10, ]*7, seed)

    def construct_floors(self, cell_size, count=10):
        base = np.ones((cell_size, cell_size, 3), np.float32)
        start = 0.7
        end = 0.8
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        arr = np.stack(arr, axis=0)
        arr[:, 0, :] -= 0.05
        arr[:, -1, :] -= 0.05
        arr[:, :, 0] -= 0.05
        arr[:, :, -1] -= 0.05
        return (arr * 255.).astype(np.uint8)

    def construct_walls(self, cell_size, count=10):
        base = np.zeros((cell_size, cell_size, 3), np.float32)
        base[:,:,2] = 1.
        start = 0.3
        end = 0.7
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        return (np.stack(arr, axis=0) * 255).astype(np.uint8)


    def construct_goals(self, cell_size, count=10):
        base = np.zeros((cell_size, cell_size, 3), np.float32)
        base[:,:,1] = 1.
        start = 0.3
        end = 0.7
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        return (np.stack(arr, axis=0) * 255).astype(np.uint8)

    def construct_keys(self, cell_size, count=10):
        base = np.zeros((cell_size, cell_size, 3), np.float32)
        base[:,:,0] = 1.
        base[:,:,1] = 1.
        start = 0.3
        end = 0.7
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        return (np.stack(arr, axis=0) * 255).astype(np.uint8)


    def construct_doors(self, cell_size, count=10):
        base = np.zeros((cell_size, cell_size, 3), np.float32)
        base[:,:,0] = 1.
        base[:,:,2] = 1.
        start = 0.3
        end = 0.7
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        return (np.stack(arr, axis=0) * 255).astype(np.uint8)


    def construct_key_doors(self, cell_size, count=10):
        return self.construct_doors(cell_size, count)


    def construct_players(self, cell_size, count=10):
        base = np.zeros((cell_size, cell_size, 3), np.float32)
        triangle = np.tril(m=np.arange(1,self.cell_size*2+1))[:self.cell_size:2,:self.cell_size]
        triangle = np.rot90(np.concatenate([triangle, triangle[::-1]]))
        if triangle.shape[-1] != self.cell_size:
            triangle = np.delete(triangle, self.cell_size//2, axis=-1)
        i = np.where(triangle != 0)
        triangle[i] = 1.
        base[:,:,0] = triangle
        start = 0.7
        end = 1.0
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        arr = np.stack(arr, axis=0)
        all_direction = np.stack([np.rot90(arr, -i, axes=(1,2)) for i in range(4)], axis=1) * 255
        mask = np.ones_like(all_direction) * ((all_direction == 0).sum(axis=-1) == 3)[...,None]
        
        return np.stack([mask, all_direction, ], axis=2).astype(np.uint8)


    def mix_player_and_floor(self, player_rgb, floor_rgb):
        arr = []
        for player in player_rgb:
            i = np.where(player.sum(-1) != 0)
            masked_floor = np.copy(floor_rgb / 255.).astype(np.float32)
            masked_floor[:, i[0], i[1], :] = 0
            arr.append(masked_floor + player)
        return np.stack(arr)

class DoorKeyGridworld(GridWorldGeneration):
    def __init__(self, seed, fixed, width, height, mode: Literal['random', 'hard'] = 'random') -> None:
        super().__init__(seed, fixed)
        self.width = width
        self.height = height

        self.mode = mode
        self.max_room_size = max(self.width, self.height)

    def generate_grid_world(self):
        if self.fixed:
            self.random = np.random.RandomState(self.seed)
        w, h = self.width, self.height

        if self.mode == 'random':
            # Generate the location of the wall, with the 2 left and right most positions being invalid
            wall_poss = self.random.choice(w - 4) + 2
        elif self.mode == 'hard':
            wall_poss = w - 3

        # Generate the location of the wall, with the upper and lower most positions being invalid
        door_poss = self.random.choice(h - 2) + 1

        # Build room with walls
        floors = np.ones((w-2, h-2)) * CellType.FLOOR.value
        grid = np.pad(floors, 1, constant_values=CellType.WALL.value)

        # Add dividing wall and door
        grid[wall_poss,1:-1] = CellType.WALL.value
        grid[wall_poss,door_poss] = CellType.KEY_DOOR.value
        room_map = np.zeros_like(grid) - 1
        room_map[:wall_poss] = 0
        room_map[wall_poss+1:] = 1
        # Add goal
        grid[-2,-2] = CellType.GOAL.value

        # Return grid and player pos
        player_pos = (1,1)
        player_direction = Direction(self.random.choice(4))

        key_x = self.random.choice(wall_poss - 1) + 1
        key_y = self.random.choice(h - 3) + 2
        grid[key_x,key_y] = CellType.KEY.value
        room_conf = np.zeros([2, 3, 2], dtype=np.int32)
        room_conf[0] = np.array([[0, wall_poss+1],[0, h],[w, h]])
        room_conf[1] = np.array([[wall_poss, w],[0, h],[w, h]])

        return grid, player_pos, player_direction, room_map, room_conf


class MultiRoomGridworld(GridWorldGeneration):
    def __init__(self, seed: np.random.RandomState, fixed: bool, width: int, height: int, room_count: int, max_room_size: int) -> None:
        super().__init__(seed, fixed)
        self.width = width
        self.height = height
        self.room_count = room_count
        self.max_room_size = max_room_size


        self.grid = None
        if self.fixed:
            self.grid, self.player_pos, self.player_direction, self.door_list, self.room_confs = self.generate_grid_world()

    def generate_grid_world(self):
        if self.fixed:
            self.random = np.random.RandomState(self.seed)

        if self.grid is not None and self.fixed:
            return (self.grid.copy(),
                    self.player_pos.copy(),
                    Direction(self.player_direction.value),
                    self.door_list.copy(),
                    self.room_confs.copy())
        width = self.width
        height = self.height
        room_count = self.room_count
        max_room_size = self.max_room_size
        grid = -np.ones([width, height])
        room_map = np.zeros_like(grid)-1
        room_list, door_list = self.get_room_map(width, height, room_count, max_room_size, 8)
        room_conf = np.zeros([self.room_count, 3, 2], dtype=np.int32)
        # room_list = [[(10, 0), (5, 4)],]
        for i, ((x, y), (w, h)) in enumerate(room_list):
            right = x + w
            bottom = y + h
            grid[x:right,y:bottom] = CellType.WALL.value
            grid[x+1:right-1,y+1:bottom-1] = CellType.FLOOR.value
            room_map[x+1:right-1,y+1:bottom-1] = i
            room_conf[i] = np.array([[x, right],[y, bottom],[self.max_room_size,self.max_room_size]])
        for x, y in door_list:
            grid[x,y] = CellType.DOOR.value

        (last_x, last_y), _ = room_list[-1]
        grid[last_x + 1, last_y + 1] = CellType.GOAL.value
        player_pos = np.array((room_list[0][0][0] + 1, room_list[0][0][1] + 1))

        player_direction = Direction.turn_left(Direction.UP, self.random.choice(4))

        return grid, player_pos, player_direction, room_map, room_conf


    def get_room_map(self, width, height, room_count, max_room_size, max_tries):
        def aux(room_list, door_pos_list, entry_face: Direction):
            dims = self.random.choice(max_room_size - 4 + 1, (2,)) + 4
            door_pos = door_pos_list[-1]
            if entry_face is None:
                left, top = door_pos
            elif entry_face == Direction.UP:
                left = door_pos[0] - self.random.choice(np.arange(dims[0] - 2)) - 1
                top = door_pos[1] - dims[1] + 1
            elif entry_face == Direction.DOWN:
                left = door_pos[0] - self.random.choice(np.arange(dims[0] - 2)) - 1
                top = door_pos[1]
            elif entry_face == Direction.LEFT:
                left = door_pos[0] - dims[0] + 1
                top = door_pos[1] - self.random.choice(np.arange(dims[1] - 2)) - 1
            elif entry_face == Direction.RIGHT:
                left = door_pos[0]
                top = door_pos[1] - self.random.choice(np.arange(dims[1] - 2)) - 1

            if top < 0 or left < 0:
                return None
            if (top + dims[1]) > height or (left + dims[0]) > width:
                return None

            for (left_other, top_other), (w_other, h_other) in room_list[:-1]:
                if not (top_other > top + dims[1] or
                        top > top_other + h_other or
                        left + dims[0] < left_other or
                        left_other + w_other < left):
                    return None

            room_list.append([(left, top), dims])

            if len(room_list) == room_count:
                return room_list, door_pos_list

            for _ in range(max_tries):
                entry_face = Direction.UP if entry_face is None else entry_face
                exit_face = Direction.turn_right(entry_face, amount=self.random.choice(3)+2)
                if exit_face == Direction.UP:
                    exit_door = (left + self.random.choice(dims[0]-2) + 1, top)
                elif exit_face == Direction.DOWN:
                    exit_door = (left + self.random.choice(dims[0]-2) + 1, top + dims[1] - 1)
                elif exit_face == Direction.LEFT:
                    exit_door = (left, top + self.random.choice(dims[1]-2) + 1)
                elif exit_face == Direction.RIGHT:
                    exit_door = (left + dims[0] - 1, top + self.random.choice(dims[1]-2) + 1)

                success = aux(room_list, door_pos_list + [exit_door,], exit_face)
                if success is not None:
                    room_list, door_pos_list = success
                    return room_list, door_pos_list

            return None

        res = None
        while res is None:
            first_room_y = self.random.choice(height-4)
            first_room_x = self.random.choice(width-4)
            res = aux([], [[first_room_y, first_room_x],], None)
            if res is not None:
                rooms, doors = res
        return rooms, doors[1:]

class Gridworld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, width, height, door_count, cell_size = 30, num_envs=1, render_mode: Literal['human', 'rgb_array'] = 'human',
                 grid_generation: GridWorldGeneration=None, tile_generation: GridWorldTiles = None, seed=None, max_time_steps=np.inf, 
                 camera_mode: Literal['full', 'agent_centric', 'room_centric'] = 'full', camera_kwargs: dict = {}, show_key_icon: bool = True):
        self.width = width
        self.height = height
        self.num_envs = num_envs

        self.divide_rooms = False
        self.agent_centric = False
        self.show_key_icon = show_key_icon

        if camera_mode == 'room_centric':
            self.divide_rooms = True
        elif camera_mode == 'agent_centric':
            self.agent_centric = True
            self.agent_view = np.array(camera_kwargs['agent_view']) if 'agent_view' in camera_kwargs else np.array([2, 2])

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.single_observation_space = spaces.Box(0, 255, (self.width * cell_size, self.height * cell_size, 3), np.uint8)
        self.observation_space = spaces.Box(0, 255, (num_envs, self.width * cell_size, self.height * cell_size, 3), np.uint8)

        # We have 5 actions, corresponding to "right", "up", "left", "down", "use"
        self.single_action_space = spaces.Discrete(5)
        self.action_space = spaces.MultiDiscrete([5, ] * num_envs)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.clock = None

        self.max_time_step = max_time_steps
        self.cell_size = cell_size  # Size of each cell in pixels
        self.screen_size = (width * self.cell_size, height * self.cell_size)
        if grid_generation is None:
            self.generation=MultiRoomGridworld(self.np_random)
        else:
            self.generation = grid_generation
        if tile_generation is None:
            self.tile_generation = RandomLuminationTiles(self.cell_size, self.np_random)
        else:
            self.tile_generation = tile_generation
        self.cell_render = self.tile_generation.construct_tiles()

        # Pygame setup
        self.key_mapping = [pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 1000))
            pygame.display.set_caption('Gridworld Game')
            self.clock = pygame.time.Clock()
            self.render = self.human_render
        elif render_mode == 'rgb_array':
            self.render = self.rgb_render

        self.rgb_obs = np.zeros((num_envs, *self.screen_size, 3), dtype=np.uint8)
        self.floor_mask = np.zeros((num_envs, *self.screen_size), dtype=np.uint8)
        self.grids = np.zeros((num_envs, width, height), dtype=np.int8)
        self.room_maps = np.zeros_like(self.grids)
        self.room_confs = None
        self.player_positions = np.zeros((num_envs, 2), dtype=np.int32)
        self.last_player_positions = np.zeros((num_envs, 2), dtype=np.int32)
        self.player_directions = np.zeros((num_envs,), dtype=np.uint8)

        self.step_counts = np.zeros((num_envs, ), dtype=np.uint16)
        self.visited_rooms = np.zeros((num_envs, ), dtype=np.uint16)
        self.current_room = np.zeros((num_envs, ), dtype=np.uint8)

        self.dones = np.full((num_envs, ), fill_value=False)
        self.next_dones = np.full((num_envs, ), fill_value=False)
        self.has_key = np.full((num_envs, ), fill_value=False)
        self.info = {
            'rooms_visited': np.ones((num_envs, )),
            'step_count': np.zeros((num_envs, )),
            'current_room': np.zeros((num_envs, )),
            'episodic_return': np.zeros((num_envs, )),
        }

        self.max_c = 0


    def handle_single_action(self, action, env_index):
        if action == 0:
            temp_pos = self.player_positions[env_index].copy()
            self.move_player(env_index)
            if self.l1_norm(self.player_positions[env_index], self.last_player_positions[env_index]) >= 2:
                self.last_player_positions[env_index] = temp_pos
        elif action == 1:
            self.player_directions[env_index] = Direction.turn_right(self.player_directions[env_index])
        elif action == 2:
            self.player_directions[env_index] = Direction.turn_left(self.player_directions[env_index])
        elif action == 3:
            self.use(env_index)

    def get_info(self):
        return {k: v.copy() for k, v in self.info.items() }

    def step(self, actions):
        for i, action in enumerate(actions):
            self.handle_single_action(action, i)
        self.max_c = max(self.max_c, self.step_counts.max())
        self.step_counts += 1
        truncated = self.step_counts == self.max_time_step
        dones = self.dones.copy()
        self.next_dones = dones | truncated
        reward = self.get_reward(self.step_counts, self.max_time_step) * dones
        self.info['step_count'] = self.step_counts.copy()
        self.info['episodic_return'] += reward 
        self.info['current_room'] = self.current_room
        self.info['rooms_visited'] = self.visited_rooms.copy()

        info = self.get_info()

        if self.next_dones.any():
            self.step_counts *= ~self.next_dones
            d = np.where(self.next_dones)[0]
            for index in d:
                self.reset_env(index)
            self.next_dones = self.next_dones & False

        observation = self.rgb_render()
        return observation, reward, dones, truncated, info

    def get_reward(self, step_counts, max_step):
        return (1. - 0.9 * (step_counts / max_step))

    def get_player_position(self):
        return self.player_positions.copy()
    
    def get_grid(self):
        return self.grids.copy()
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        # We need the following line to seed self.np_random
        self.generation.set_random(self.np_random)
        self.tile_generation.set_random(self.np_random)

        for i in range(self.num_envs):
            self.reset_env(i)

        return self.rgb_render(), self.info


    def reset_env(self, index: int, seed = None, options = None):
        self.dones[index] = False
        grid, player_position, player_direction, room_map, room_confs = \
            self.generation.generate_grid_world()

        self.grids[index,...] = grid
        self.room_maps[index] = room_map
        if self.room_confs is None:
            self.room_confs = np.zeros([self.num_envs, self.room_maps.max()+1, 3, 2], dtype=np.int32)

        self.room_confs[index] = room_confs

        self.player_positions[index] = player_position
        self.player_directions[index] = player_direction.value
        self.rgb_obs[index], self.floor_mask[index] = self.construct_rgb_obs(index)
        self.last_player_positions[index] = player_position
        self.has_key[index] = False
        self.visited_rooms[index] = 0
        self.current_room[index] = 0

        self.info['rooms_visited'][index] = 1
        self.info['step_count'][index] = 0
        self.info['current_room'][index] = 0
        self.info['episodic_return'][index] = 0

        # observation = self.rgb_render()
        # return observation, self.info

    ################################################################################################
    ### VALIDATION
    ################################################################################################

    def reached_goal(self, env_index, position):
        x, y = position
        return self.grids[env_index, x, y] == CellType.GOAL.value

    def coordinate_valid(self, position):
        return 0 <= position[0] < self.height and 0 <= position[1] < self.width

    def looking_at(self, env_index):
            dir_vec = Direction.to_vectors()[self.player_directions[env_index]]
            return (self.player_positions[env_index, 0] + dir_vec[0], self.player_positions[env_index, 1] + dir_vec[1])

    def can_move(self, env_index, position):
        x, y = position
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        if self.grids[env_index, x, y] in [CellType.DOOR.value, CellType.KEY_DOOR.value, \
                                                   CellType.WALL.value, CellType.KEY.value]:
            return False
        return True


    def get_current_room(self, env_index, position):
        return self.room_maps[env_index, position[0], position[1]]

    def l1_norm(self, pos1, pos2):
        if len(pos1.shape) == 3:
            return abs(pos1[:, 0] - pos2[:, 0]) + abs(pos1[:, 1] - pos2[:, 1])
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    ################################################################################################
    ### INTERACTION
    ################################################################################################

    def place_object(self, env_index, position, symbol):
        x, y = position
        if self.coordinate_valid(position):
            self.grids[env_index, x, y] = symbol
            x *= self.cell_size
            y *= self.cell_size
            self.rgb_obs[env_index, x:x+self.cell_size,y:y+self.cell_size] = self.cell_render[CellType(symbol)]
            self.floor_mask[env_index, x:x+self.cell_size,y:y+self.cell_size] = 0

    def move_player(self, env_index):
        move = Direction.to_vectors()[self.player_directions[env_index]]
        x, y = self.player_positions[env_index]
        new_position = (x + move[0], y + move[1])
        if self.can_move(env_index, new_position):
            cr =  self.get_current_room(env_index, new_position).copy()
            self.current_room[env_index] = cr if cr != -1 else self.current_room[env_index]
            # print(f'Curent room: {self.current_room}')
            self.visited_rooms[env_index] = max(self.visited_rooms[env_index], self.current_room[env_index])
            self.player_positions[env_index] = new_position
        if self.reached_goal(env_index, new_position):
            self.dones[env_index]=True

    def use(self, env_index):
        x, y = self.looking_at(env_index)
        next_tile = self.grids[env_index, x, y]
        if self.coordinate_valid((x,y)) is False:
            return
        # Check if the player can interact with the door
        elif not self.has_key[env_index] and next_tile == CellType.KEY.value:
            self.has_key[env_index] = True
            self.place_object(env_index, (x,y), CellType.FLOOR.value)

        elif self.has_key[env_index] and next_tile == CellType.KEY_DOOR.value:
            # print("You've unlocked the door!")
            # Optionally, update the grid to reflect the door being opened
            self.place_object(env_index, (x, y), CellType.FLOOR.value)
            self.has_key[env_index] = False

        # Check if the player can interact with the door
        elif next_tile == CellType.DOOR.value:
            # print("You've opened the door!")
            self.place_object(env_index, (x, y), CellType.FLOOR.value)

        elif self.has_key[env_index] and next_tile == CellType.FLOOR.value:
            self.place_object(env_index, (x, y), CellType.KEY.value)
            self.has_key[env_index] = False


    def pick_up_key(self, env_index):
        x, y = self.looking_at(env_index)
        if self.grids[env_index, x, y] == CellType.KEY.value:
            self.has_key[env_index] = True
            self.place_object(env_index, (x,y), CellType.FLOOR.value)

    ################################################################################################
    ### RENDERING
    ################################################################################################

    def get_room(self, obs, mask=False):
        mw, mh = (self.generation.max_room_size * self.cell_size, ) * 2
        divided_rgb_obs = np.zeros([self.num_envs, mw, mh, obs.shape[-1]], dtype=np.uint8)
        for i in range(self.num_envs):
            curr_room = self.current_room[i].copy()
            x, y = self.player_positions[i].copy()
            room_conf = self.room_confs[i][curr_room].copy()
            xc, yc = room_conf[0], room_conf[1]
            direction = self.player_directions[i]
            x -= xc[0]
            y -= yc[0]
            x *= self.cell_size
            y *= self.cell_size
            xc *= self.cell_size
            yc *= self.cell_size
            player_mask, player_sprite = self.cell_render[CellType.PLAYER][direction]

            room_rgb = obs[i, xc[0]:xc[1], yc[0]:yc[1]].copy()
            if mask:
                room_rgb[x:x+self.cell_size,y:y+self.cell_size] *= player_mask[...,0]
            else:
                room_rgb[x:x+self.cell_size,y:y+self.cell_size] *= player_mask
                room_rgb[x:x+self.cell_size,y:y+self.cell_size] += player_sprite
                if self.show_key_icon and self.has_key[i]:
                    room_rgb[i, 0:5,0:5] = np.array([255, 255, 0])

            rw, rh = room_rgb.shape[-3:-1]
            offset_x, offset_y = (mw-rw)//2, (mh-rh)//2,

            divided_rgb_obs[i, offset_x:offset_x+rw, offset_y:offset_y+rh] = room_rgb

        return divided_rgb_obs

    def get_centric(self, obs, mask=False):

        # rgb_obs = self.rgb_obs.copy()
        centric_rgb = np.zeros([self.num_envs, *((2*self.agent_view+1)*self.cell_size), obs.shape[-1]], dtype=np.uint8)
        ox, oy = self.agent_view
        for i in range(len(obs)):
            x, y = self.player_positions[i]
            direction = self.player_directions[i]
            bx = np.array([x - ox, x + ox + 1]) * self.cell_size
            by = np.array([y - oy, y + oy + 1]) * self.cell_size
            player_mask, player_sprite = self.cell_render[CellType.PLAYER][direction]

            # Create padding in case of out of bounds
            xpb, xpa = -min(0, bx[0]), max(obs.shape[1], bx[1]) - obs.shape[1]
            ypb, ypa = -min(0, by[0]), max(obs.shape[2], by[1]) - obs.shape[2]
            o = np.pad( obs[i], ((xpb, xpa), (ypb, ypa), (0, 0)), constant_values=0)

            # Move pixel according to padding of OOB
            bx += xpb - xpa
            by += ypb - ypa

            centric_rgb[i] = o[bx[0]:bx[1], by[0]:by[1]].copy()
            if mask:
                centric_rgb[i, ox*self.cell_size:(ox+1)*self.cell_size, oy*self.cell_size:(oy+1)*self.cell_size] *= player_mask[...,0]
            else:
                centric_rgb[i, ox*self.cell_size:(ox+1)*self.cell_size, oy*self.cell_size:(oy+1)*self.cell_size] *= player_mask
                centric_rgb[i, ox*self.cell_size:(ox+1)*self.cell_size, oy*self.cell_size:(oy+1)*self.cell_size] += player_sprite
                if self.show_key_icon and self.has_key[i]:
                    centric_rgb[i, 0:5,0:5] = np.array([255, 255, 0])
        return centric_rgb

    def get_floor_mask(self):
        # return rgb_obs
        if self.divide_rooms:
            return self.get_room(self.floor_mask, mask=True)

        if self.agent_centric:
            return self.get_centric(self.floor_mask, mask=True)

        floor_mask = self.floor_mask.copy()
        for i in range(len(floor_mask)):
            x, y = self.player_positions[i]
            direction = self.player_directions[i]
            x *= self.cell_size
            y *= self.cell_size
            player_mask, _ = self.cell_render[CellType.PLAYER][direction]
            floor_mask[i, x:x+self.cell_size,y:y+self.cell_size] *= player_mask[...,0]
        
        return floor_mask[...,None]
        


    def rgb_render(self):
        # return rgb_obs
        if self.divide_rooms:
            return self.get_room(self.rgb_obs)

        if self.agent_centric:
            return self.get_centric(self.rgb_obs)

        rgb_obs = self.rgb_obs.copy()
        for i in range(len(rgb_obs)):
            x, y = self.player_positions[i]
            direction = self.player_directions[i]
            x *= self.cell_size
            y *= self.cell_size
            player_mask, player_sprite = self.cell_render[CellType.PLAYER][direction]
            rgb_obs[i, x:x+self.cell_size,y:y+self.cell_size] *= player_mask
            rgb_obs[i, x:x+self.cell_size,y:y+self.cell_size] += player_sprite

            if self.show_key_icon and self.has_key[i]:
                rgb_obs[i, 0:5,0:5] = np.array([255, 255, 0])
        return rgb_obs

    def human_render(self):
        self.screen.fill((255, 255, 255))  # Fill the screen with white
        # draw the array onto the surface
        w,h = self.screen_size
        obs = self.rgb_render()
        screen = np.zeros((w * 4, h * 4, 3), dtype=np.uint8)
        for index, o in enumerate(obs):
            i = (index % 4)*w
            j = index // 4*h
            screen[i:i+w,j:j+h] = o
        surf = pygame.surfarray.make_surface(screen)
        self.screen.blit(surf, (0, 0))
        pygame.display.update()

    def construct_rgb_obs(self, env_index):
        floor_arr = []
        arr = []

        for i in range(self.width * self.height):
            x, y = i % self.width, i // self.height
            type = self.grids[env_index, x, y]
            if type == -1:
                arr.append(np.zeros((self.cell_size, self.cell_size, 3), dtype=np.uint8))
            else:
                arr.append(self.cell_render[CellType(type)])

            if type == CellType.FLOOR:
                floor_arr.append(np.ones((self.cell_size, self.cell_size), dtype=np.uint8))
            else:
                floor_arr.append(np.zeros((self.cell_size, self.cell_size), dtype=np.uint8))
                
        arr = np.concatenate(arr, axis=0)
        arr = np.concatenate(np.split(arr, self.height), axis=1)

        floor_arr = np.concatenate(floor_arr, axis=0)
        floor_arr = np.concatenate(np.split(floor_arr, self.height), axis=1)

        return arr, floor_arr


    ################################################################################################
    ### Human input
    ################################################################################################
    def handle_input(self):
        if self.done:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key in self.key_mapping:
                self.step(self.key_mapping.index(event.key))

    def run_game_loop(self):
        running = True
        while running:
            self.handle_input()  # Call the input handling method
            self.render()
            self.clock.tick(60)  # Limit the frame rate to 60 FPS


# ############################################################
# ############################################################
# ################### ENVIONRMENTS #####################
# ############################################################
# ############################################################




# ############################################################
# ################### DoorKey Env. ###########################
# ############################################################


class DoorKey5x5Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey5x5-Gridworld-v0"
    width = height = 5
    _max_episode_steps = 10 * (width**2)

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        grid_gen = DoorKeyGridworld(seed, width=self.width, height=self.height, fixed=fixed, mode=mode)
        random_tile = RandomLuminationTiles(cell_size, seed)
        self.max_reward = self.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        super().__init__(self.width, self.height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)


class DoorKey6x6Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey6x6-Gridworld-v0"
    width = height = 6
    _max_episode_steps = 10 * (width**2)

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        grid_gen = DoorKeyGridworld(seed, width=self.width, height=self.height, fixed=fixed, mode=mode)
        random_tile = RandomLuminationTiles(cell_size, seed)
        self.max_reward = self.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        super().__init__(self.width, self.height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)


class DoorKey8x8Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey8x8-Gridworld-v0"
    width = height = 8
    _max_episode_steps = 10 * (width**2)

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        grid_gen = DoorKeyGridworld(seed, width=self.width, height=self.height, fixed=fixed, mode=mode)
        random_tile = RandomLuminationTiles(cell_size, seed)
        self.max_reward = self.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        super().__init__(self.width, self.height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)



class DoorKey16x16Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey16x16-Gridworld-v0"
    width = height = 16
    _max_episode_steps = 10 * (width**2)

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        grid_gen = DoorKeyGridworld(seed, width=self.width, height=self.height, fixed=fixed, mode=mode)
        random_tile = RandomLuminationTiles(cell_size, seed)
        self.max_reward = self.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        super().__init__(self.width, self.height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)


# ############################################################
# ################### MultiRoom Env. ######################
# ############################################################

class MultiRoomS4N2GridWorld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "MultiRoomS4N2-Gridworld-v0"
    room_count = 2
    max_room_size = 4
    _max_episode_steps= room_count * 20

    def __init__(self, cell_size = 30, num_envs=1, fixed = False, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None, **kwargs):
        width = height = self.room_count * self.max_room_size
        grid_gen = MultiRoomGridworld(seed, fixed, width=width, height=height, room_count=self.room_count, max_room_size=self.max_room_size)
        random_tile = RandomLuminationTiles(cell_size, seed)
        super().__init__(width, height, self.room_count-1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)

class MultiRoomS5N4GridWorld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "MultiRoomS5N4-Gridworld-v0"
    room_count = 4
    max_room_size = 5
    _max_episode_steps= room_count * 20

    def __init__(self, cell_size = 30, num_envs=1, fixed = False, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None, **kwargs):
        width = height = self.room_count * self.max_room_size
        grid_gen = MultiRoomGridworld(seed, fixed, width=width, height=height, room_count=self.room_count, max_room_size=self.max_room_size)
        random_tile = RandomLuminationTiles(cell_size, seed)
        super().__init__(width, height, self.room_count-1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)

class MultiRoomS10N6GridWorld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "MultiRoomS10N6-Gridworld-v0"
    room_count = 6
    max_room_size = 10
    _max_episode_steps= room_count * 20

    def __init__(self, cell_size = 30, num_envs=1, fixed = False, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None, **kwargs):
        width = height = int(self.room_count * self.max_room_size)
        grid_gen = MultiRoomGridworld(seed, fixed, width=width, height=height, room_count=self.room_count, max_room_size=self.max_room_size)
        random_tile = RandomLuminationTiles(cell_size, seed)
        super().__init__(width, height, self.room_count-1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed, self._max_episode_steps, **kwargs)


# ############################################################
# ############################################################
# ################### NOISY ENVIONRMENTS #####################
# ############################################################
# ############################################################


class NoisyGridworldWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: Gridworld, alpha = 1.0):
        super().__init__(env)
        self.env = env
        self.cutoff = 255 - int(255 * alpha)
        self.alpha = alpha


        self.step = self.env.step
        self.reset = self.env.reset
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = self.env.get_player_position
        self.get_grid = self.env.get_grid
        
        if hasattr(self.env, 'screen'):
            self.screen = self.env.screen
        
    def observation(self, observation):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        noise = np.random.randint(0, 255, observation.shape, dtype=np.uint8)
        noise = ((np.clip(noise, self.cutoff, 255) - self.cutoff) / self.alpha).astype(np.uint8)
        floor_index = np.where((observation[...,0] > 10)
                               & (observation[...,0] == observation[...,1]) 
                               & (observation[...,0] == observation[...,2]))
        floor_mask = np.zeros_like(observation, dtype=np.uint8)
        floor_mask[floor_index] = 1
        # floor_mask = self.env.get_floor_mask()
        # floor_mask = floor_mask.repeat(c, axis=-2).repeat(c, axis=-1)[...,None]

        # x_full, y_full = np.rollaxis(self.env.player_positions * c, axis=1)
        # player = self.env.cell_render[CellType.PLAYER][self.env.player_directions]
        # player_mask, _ = np.rollaxis(player, axis=1)
        # for i, (x, y, mask) in enumerate(zip(x_full, y_full, player_mask)):
        #     floor_mask[i, x:x+c, y:y+c] *= mask[...,:1]

        observation = observation * (1-floor_mask) + floor_mask * noise
        return observation
    
    def rgb_render(self):
        return self.observation(self.env.rgb_render())

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

# ############################################################
# ################### DoorKey Env. ###########################
# ############################################################


class NoisyDoorKey5x5Gridworld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyDoorKey5x5-Gridworld-v0"
    width = height = 5
    _max_episode_steps = 10 * width**2

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        super().__init__()
        env = DoorKey5x5Gridworld(cell_size, num_envs, fixed, max_steps, render_mode, mode, seed, **kwargs)
        self.max_reward = env.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        self.env = NoisyGridworldWrapper(env)

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = env.get_player_position
        self.get_grid = env.get_grid

class NoisyDoorKey6x6Gridworld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyDoorKey6x6-Gridworld-v0"
    width = height = 6
    _max_episode_steps = 10 * width**2

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        super().__init__()
        env = DoorKey6x6Gridworld(cell_size, num_envs, fixed, max_steps, render_mode, mode, seed, **kwargs)
        self.max_reward = env.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        self.env = NoisyGridworldWrapper(env)

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = env.get_player_position
        self.rgb_render = self.env.rgb_render
        self.get_grid = env.get_grid

class NoisyDoorKey8x8Gridworld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyDoorKey8x8-Gridworld-v0"
    width = height = 8
    _max_episode_steps = 10 * width**2

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        super().__init__()
        env = DoorKey8x8Gridworld(cell_size, num_envs, fixed, max_steps, render_mode, mode, seed, **kwargs)
        self.max_reward = env.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        self.env = NoisyGridworldWrapper(env)

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = env.get_player_position
        self.rgb_render = self.env.rgb_render
        self.get_grid = self.env.get_grid



class NoisyDoorKey16x16Gridworld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyDoorKey16x16-Gridworld-v0"
    width = height = 16
    _max_episode_steps = 10 * width**2

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', mode='hard', seed=None, **kwargs):
        super().__init__()
        env = DoorKey16x16Gridworld(cell_size, num_envs, fixed, max_steps, render_mode, mode, seed, **kwargs)
        self.max_reward = env.get_reward((self.width-1) + 3*(self.height-2), self._max_episode_steps)
        self.env = NoisyGridworldWrapper(env)

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = self.env.get_player_position
        self.get_grid = self.env.get_grid
        self.rgb_render = self.env.rgb_render




# ############################################################
# ################### MultiRoom Env. ######################
# ############################################################
class NoisyMultiRoomS4N2GridWorld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyMultiRoomS4N2-Gridworld-v0"
    room_count = 2
    max_room_size = 4
    _max_episode_steps= room_count * 20

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None, **kwargs):
        super().__init__()
        self.env = NoisyGridworldWrapper(MultiRoomS4N2GridWorld(cell_size, num_envs, fixed, render_mode, seed, **kwargs))

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = self.env.get_player_position
        self.get_grid = self.env.get_grid
        self.rgb_render = self.env.rgb_render



class NoisyMultiRoomS5N4GridWorld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyMultiRoomS5N4-Gridworld-v0"
    room_count = 4
    max_room_size = 5
    _max_episode_steps= room_count * 20

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None, **kwargs):
        super().__init__()
        self.env = NoisyGridworldWrapper(MultiRoomS5N4GridWorld(cell_size, num_envs, fixed, render_mode, seed,  **kwargs))

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = self.env.get_player_position
        self.get_grid = self.env.get_grid
        self.rgb_render = self.env.rgb_render



class NoisyMultiRoomS10N6GridWorld(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "NoisyMultiRoomS10N6-Gridworld-v0"
    room_count = 6
    max_room_size = 10
    _max_episode_steps= room_count * 20

    def __init__(self, cell_size = 30, num_envs=1, fixed=False, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None,  **kwargs):
        super().__init__()
        self.env = NoisyGridworldWrapper(MultiRoomS10N6GridWorld(cell_size, num_envs, fixed, render_mode, seed,  **kwargs))

        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation = self.env.observation
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = self.env.get_player_position
        self.get_grid = self.env.get_grid
        self.rgb_render = self.env.rgb_render


# ############################################################
# ############################################################
# ################### Non-random Noise ######################
# ############################################################
# ############################################################

import colorsys as c

class BlockyBackgroundGridworldWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: Gridworld, colors=12, speed=2, width = 15):
        super().__init__(env)
        self.env = env
        self.colors = self.get_colors(colors)
        self.background = None
        self.screen_width = None
        self.width = width
        self.speed = speed
        self.scroll = 0


        self.step = self.env.step
        self.reset = self.env.reset
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.action_space =self.env.action_space
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.get_player_position = self.env.get_player_position
        self.get_grid = self.env.get_grid
        
        if hasattr(self.env, 'screen'):
            self.screen = self.env.screen

    def get_colors(self, colors):
        rgb_colors = [c.hsv_to_rgb(i/float(colors), 1., 1.) for i in range(colors)]
        return np.array(rgb_colors, dtype=np.uint8) * 255


    def construct_background(self, colors, width, screen_width):
        # Construct colors
        bg = colors[:, None].repeat(width, axis=0)
        # Repeat in case of too small
        repeats = math.ceil(screen_width/bg.shape[0])
        bg = np.concatenate([bg,]*repeats, axis=0)

        # Add screen width to end to avoid annoying wrapping indexing
        return np.concatenate([bg, bg[:screen_width]], axis=0)

    def observation(self, observation):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        if self.background is None:
            self.screen_width = observation.shape[1]
            self.background = self.construct_background(self.colors, self.width, self.screen_width)

        floor_index = np.where((observation[...,0] > 10)
                                & (observation[...,0] == observation[...,1]) 
                                & (observation[...,0] == observation[...,2]))
        floor_mask = np.zeros_like(observation, dtype=np.uint8)
        floor_mask[floor_index] = 1
        scroll = (self.env.step_counts * self.speed) % (self.background.shape[0] - self.screen_width)
        for i in range(len(observation)):
            bg = self.background[scroll[i]:scroll[i]+self.screen_width]            
            observation[i] = observation[i] * (1-floor_mask[i]) + floor_mask[i] * bg
        return observation
    
    def rgb_render(self):
        return self.observation(self.env.rgb_render())

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



class GridworldResizeObservation(gymnasium.ObservationWrapper, gymnasium.utils.RecordConstructorArgs):
    """Resize the image observation.

    This wrapper works on environments with image observations. More generally,
    the input can either be two-dimensional (AxB, e.g. grayscale images) or
    three-dimensional (AxBxC, e.g. color images). This resizes the observation
    to the shape given by the 2-tuple :attr:`shape`.
    The argument :attr:`shape` may also be an integer, in which case, the
    observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ResizeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gymnasium.Env, shape: tuple[int, int] | int) -> None:
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gymnasium.ObservationWrapper.__init__(self, env)

        if isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2 and all(
            x > 0 for x in shape
        ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"

        self.shape = tuple(shape)
        self.env = env
        assert isinstance(
            env.observation_space, gymnasium.spaces.Box
        ), f"Expected the observation space to be Box, actual type: {type(env.observation_space)}"
        dims = len(env.observation_space.shape)
        assert (
            dims==4
        ), f"Expected the observation space to have 4 dimensions, got: {dims}"

        E, _, _, C = env.observation_space.shape
        W, H = self.shape
        obs_shape = (E, W, H, C)
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(1, ) + obs_shape[1:], dtype=np.uint8)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.get_player_position = env.get_player_position
        self.get_grid = self.env.get_grid

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e

        B, W, H, C = observation.shape
        observation = observation.transpose(1,2,3,0).reshape((W, H, C * B))

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_NEAREST_EXACT
        )
        
        observation = observation.reshape((*self.shape[::-1], C, B)).transpose(3, 0, 1, 2)
        return observation

    def render(self):
        if self.env.render_mode == 'rgb_array':
            return self.env.rgb_render()
        self.env.screen.fill((255, 255, 255))  # Fill the screen with white
        # draw the array onto the surface
        w,h = self.shape
        obs = self.env.rgb_render()
        obs = self.observation(obs)
        screen = np.zeros((w * 4, h * 4, 3), dtype=np.uint8)
        for index, o in enumerate(obs):
            i = (index % 4)*w
            j = index // 4*h
            screen[i:i+w,j:j+h] = o
        surf = pygame.surfarray.make_surface(screen)
        surf = pygame.transform.scale(surf, (1000, 1000))
        self.env.screen.blit(surf, (0, 0))
        pygame.display.update()
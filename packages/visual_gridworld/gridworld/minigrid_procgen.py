
import abc
import sys
from typing import Dict, List, Literal
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
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
    FLOOR=0
    WALL=1
    DOOR=2
    KEY_DOOR=3
    KEY=4
    GOAL=5
    PLAYER=6


class GridWorldGeneration(abc.ABC):
    def __init__(self, random) -> None:
        super().__init__()
        if random is None:
            self.random = np.random.RandomState()
        else:
            self.random = random

    def set_random(self, random: np.random.RandomState):
        self.random = random

    @abc.abstractmethod
    def generate_grid_world(self):
        pass

class GridWorldTiles(abc.ABC):
    def __init__(self, cell_size, tile_count, random=None) -> None:
        super().__init__()
        self.cell_size = cell_size
        self.tile_count = tile_count
        if random is None:
            self.random = np.random.RandomState(None)
        else:
            self.random = random

        self.tiles = self._preload_tiles()

    def set_random(self, random: np.random.RandomState):
        self.random = random

    def construct_tiles(self):
        cell_size = self.cell_size
        tile_index = [self.random.choice(v) for v  in self.tile_count]
        ids = [CellType.FLOOR, CellType.WALL, CellType.GOAL, CellType.KEY, CellType.DOOR, CellType.KEY_DOOR]
        tiles = {id: self.tiles[id][tile_i] for id, tile_i in zip(ids, tile_index)}

        # Select the correct player tile and correct background tile
        tiles[CellType.PLAYER] = self.tiles[CellType.PLAYER][tile_index[0], tile_index[-1]]

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
        tiles[CellType.PLAYER] = self.construct_players(cell_size, tiles[CellType.FLOOR])

        

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
    def __init__(self, cell_size, random) -> None:
        super().__init__(cell_size, [10, ]*7, random)
    
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
        
    
    def construct_players(self, cell_size, floor_tiles, count=10):
        base = np.zeros((cell_size, cell_size, 3), np.float32)
        triangle = np.tril(m=np.arange(1,self.cell_size*2+1))[:self.cell_size:2,:self.cell_size]
        triangle = np.rot90(np.concatenate([triangle, triangle[::-1]]))
        i = np.where(triangle != 0)
        triangle[i] = 1.
        base[:,:,0] = triangle
        start = 0.7
        end = 1.0
        step = (end - start) / count
        arr = [base * lum for lum in np.arange(start,end,step)]
        arr = np.stack(arr, axis=0)
        all_direction = [np.rot90(arr, -i, axes=(1,2)) for i in range(4)]
        final_tiles = [self.mix_player_and_floor(player, floor_tiles) for player in all_direction]
        return (np.stack(final_tiles, axis=2) * 255).astype(np.uint8)


    def mix_player_and_floor(self, player_rgb, floor_rgb):
        arr = []
        for player in player_rgb:
            i = np.where(player.sum(-1) != 0)
            masked_floor = np.copy(floor_rgb / 255.).astype(np.float32)
            masked_floor[:, i[0], i[1], :] = 0
            arr.append(masked_floor + player)
        return np.stack(arr)

class DoorKeyGridworld(GridWorldGeneration):
    def __init__(self, random, width, height) -> None:
        super().__init__(random)
        self.width = width
        self.height = height

    def generate_grid_world(self):
        w, h = self.width, self.height
        # Generate the location of the wall, with the 2 left and right most positions being invalid
        wall_poss = self.random.choice(w - 4) + 2
        # Generate the location of the wall, with the upper and lower most positions being invalid
        door_poss = self.random.choice(h - 2) + 1
        
        # Build room with walls
        floors = np.ones((w-2, h-2)) * CellType.FLOOR.value
        grid = np.pad(floors, 1, constant_values=CellType.WALL.value)

        # Add dividing wall and door
        grid[wall_poss,1:-1] = CellType.WALL.value
        grid[wall_poss,door_poss] = CellType.KEY_DOOR.value
        room_entrances = [(wall_poss, door_poss)]
        # Add goal
        grid[-2,-2] = CellType.GOAL.value

        # Return grid and player pos
        player_pos = (1,1)
        player_direction = Direction(self.random.choice(4))

        key_x = self.random.choice(wall_poss - 1) + 1
        key_y = self.random.choice(h - 3) + 2
        grid[key_x,key_y] = CellType.KEY.value

        return grid, player_pos, player_direction, room_entrances
    

class MultiRoomGridworld(GridWorldGeneration):
    def __init__(self, random: np.random.RandomState, width: int, height: int, room_count: int, max_room_size: int) -> None:
        super().__init__(random)
        self.width = width
        self.height = height
        self.room_count = room_count
        self.max_room_size = max_room_size
    

    def generate_grid_world(self):
        width = self.width
        height = self.height
        room_count = self.room_count
        max_room_size = self.max_room_size
        grid = np.zeros([width, height])
        room_list, door_list = self.get_room_map(width, height, room_count, max_room_size, 8)
        # room_list = [[(10, 0), (5, 4)],]
        for (x, y), (w, h) in room_list:
            right = x + w
            bottom = y + h
            grid[x:right,y:bottom] = CellType.WALL.value            
            grid[x+1:right-1,y+1:bottom-1] = CellType.FLOOR.value            
        for x, y in door_list:
            grid[x,y] = CellType.DOOR.value
        
        (last_x, last_y), _ = room_list[-1]
        grid[last_x + 1, last_y + 1] = CellType.GOAL.value
        player_pos = (room_list[0][0][0] + 1, room_list[0][0][1] + 1)
        
        player_direction = Direction.turn_left(Direction.UP, self.random.choice(4))

        return grid, player_pos, player_direction, door_list
    

    def get_room_map(self, width, height, room_count, max_room_size, max_tries):
        def aux(room_list, door_pos_list, entry_face: Direction):
            dims = self.random.choice(max_room_size - 4, (2,)) + 4
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
                 grid_generation: GridWorldGeneration=None, tile_generation: GridWorldTiles = None, seed=None):
        self.width = width
        self.height = height
        self.num_envs = num_envs
        
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

        self.max_step = 100
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
            self.screen = pygame.display.set_mode((self.screen_size[0] * 4, self.screen_size[1] * 4))
            pygame.display.set_caption('Gridworld Game')
            self.clock = pygame.time.Clock()
            self.render = self.human_render
        elif render_mode == 'rgb_array':
            self.render = self.rgb_render

        self.rgb_obs = np.zeros((num_envs, *self.screen_size, 3), dtype=np.uint8)
        self.grids = np.zeros((num_envs, width, height), dtype=np.uint8)
        self.room_entrances = np.zeros((num_envs, door_count, 2), dtype=np.int32)
        self.player_positions = np.zeros((num_envs, 2), dtype=np.int32)
        self.last_player_positions = np.zeros((num_envs, 2), dtype=np.int32)
        self.player_directions = np.zeros((num_envs,), dtype=np.uint8)

        self.step_counts = np.zeros((num_envs, ), dtype=np.uint8)
        self.visited_rooms = [set(), ] * num_envs
        self.current_rooms = np.zeros((num_envs, ), dtype=np.uint8)

        self.dones = np.full((num_envs, ), fill_value=False)
        self.has_key = np.full((num_envs, ), fill_value=False)
        self.info = {
            'rooms_visited': np.ones((num_envs, )),
            'step_count': np.zeros((num_envs, )),
            'current_room': np.zeros((num_envs, )),
            'episodic_return': np.zeros((num_envs, )),
        }
        

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

    def step(self, actions):
        self.step_counts += 1
        truncated = self.step_counts == self.max_step
        
        if self.dones.any() or truncated.any():
            d = np.where(self.dones | truncated)[0]
            for index in d:
                self.reset_env(index)

        for i, action in enumerate(actions):
            self.handle_single_action(action, i)

        observation = self.rgb_render()
        reward = (1. - 0.9 * (self.step_counts / self.max_step)) * self.dones
        self.info['step_count'] = self.step_counts
        self.info['episodic_return'] += reward
        self.info['current_room'] = self.current_rooms
        self.info['rooms_visited'] = [len(visited_rooms) for visited_rooms in self.visited_rooms]

        return observation, reward, self.dones, truncated, self.info

    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        # We need the following line to seed self.np_random
        self.generation.set_random(self.np_random)
        self.tile_generation.set_random(self.np_random)

        for i in range(self.num_envs):
            self.reset_env(i)

        return self.rgb_obs, self.info


    def reset_env(self, index: int, seed = None, options = None):
        self.dones[index] = False
        grid, player_position, player_direction, room_entrances = \
            self.generation.generate_grid_world()
        
        self.grids[index,...] = grid
        self.room_entrances[index] = room_entrances

        self.player_positions[index] = player_position
        self.player_directions[index] = player_direction.value
        self.rgb_obs[index] = self.construct_rgb_obs(index)
        self.last_player_positions[index] = player_position
        self.has_key[index] = False
        self.step_counts[index] = 0
        self.visited_rooms[index] = set()
        self.current_rooms[index] = 0
        
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
    

    def get_room(self, env_index, position):
        if self.l1_norm(self.last_player_positions[env_index], position) != 2:
            return self.current_rooms[env_index]
        current_tile_door = np.where((self.room_entrances[env_index] == self.player_positions[env_index]).all(axis=1))[0]
        
        if len(current_tile_door) == 0:
            return self.current_rooms[env_index]
        movement = 1 if current_tile_door == self.current_rooms[env_index] else -1
        return self.current_rooms[env_index] + movement

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
    
    def move_player(self, env_index):
        move = Direction.to_vectors()[self.player_directions[env_index]]
        x, y = self.player_positions[env_index]
        new_position = (x + move[0], y + move[1])
        if self.can_move(env_index, new_position):
            self.current_room = self.get_room(env_index, new_position)
            # print(f'Curent room: {self.current_room}')
            self.visited_rooms[env_index].add(self.current_room)
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
    
    def rgb_render(self):
        rgb_obs = self.rgb_obs.copy()
        # return rgb_obs
        for i in range(len(rgb_obs)):
            x, y = self.player_positions[i]
            direction = self.player_directions[i]
            x *= self.cell_size
            y *= self.cell_size
            rgb_obs[i, x:x+self.cell_size,y:y+self.cell_size] = self.cell_render[CellType.PLAYER][direction]
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
        arr = []
        for i in range(self.width * self.height):
            x, y = i % self.width, i // self.height
            type = self.grids[env_index, x, y]
            arr.append(self.cell_render[CellType(type)])
        arr = np.concatenate(arr, axis=0)
        arr = np.concatenate(np.split(arr, self.height), axis=1)
        return arr


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


class DoorKey5x5Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey5x5-Gridworld-v0"
    max_episode_steps=300

    def __init__(self, cell_size = 30, num_envs=1, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None):
        width = height = 5
        if max_steps is None:
            self.max_episode_steps = 10 * width**2
        random = np.random.RandomState(seed)
        grid_gen = DoorKeyGridworld(random, width=5, height=5)
        random_tile = RandomLuminationTiles(cell_size, random)
        super().__init__(width, height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed)

class DoorKey6x6Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey6x6-Gridworld-v0"
    max_episode_steps=300

    def __init__(self, cell_size = 30, num_envs=1, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None):
        width = height = 6
        if max_steps is None:
            self.max_episode_steps = 10 * width**2
        random = np.random.RandomState(seed)
        grid_gen = DoorKeyGridworld(random, width=6, height=6)
        random_tile = RandomLuminationTiles(cell_size, random)
        super().__init__(width, height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed)

class DoorKey8x8Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey8x8-Gridworld-v0"
    max_episode_steps=300

    def __init__(self, cell_size = 30, num_envs=1, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None):
        width = height = 8
        if max_steps is None:
            self.max_episode_steps = 10 * width**2
        random = np.random.RandomState(seed)
        grid_gen = DoorKeyGridworld(random, width=width, height=height)
        random_tile = RandomLuminationTiles(cell_size, random)
        super().__init__(width, height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed)


class DoorKey16x16Gridworld(Gridworld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_name = "DoorKey16x16-Gridworld-v0"
    max_episode_steps=300

    def __init__(self, cell_size = 30, num_envs=1, max_steps = None, render_mode: Literal['human', 'rgb_array'] = 'rgb_array', seed=None):
        width = height = 16
        if max_steps is None:
            self.max_episode_steps = 10 * width**2
        random = np.random.RandomState(seed)
        grid_gen = DoorKeyGridworld(random, width=width, height=height)
        random_tile = RandomLuminationTiles(cell_size, random)
        super().__init__(width, height, 1, cell_size, num_envs, render_mode, grid_gen, random_tile, seed)

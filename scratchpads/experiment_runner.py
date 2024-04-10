from dataclasses import dataclass
import subprocess
import concurrent.futures
import queue
from typing import List 
import tyro

@dataclass
class Args:
    track: bool = False
    """track the experiment"""
    wandb_project_name: str = 'gridworld-rnd'
    """track the experiment"""
    repeats: int = 1
    """number of times to repeat"""
    use_tag: bool = True
    """use programmed tags for logging in wandb"""
    devices: tuple[int] = (0, 1, 2, 3)
    """avaliable device ids"""
    include_fixed: bool = True
    """avaliable device ids"""
    seed: int = 42
    """seed"""
    

args = tyro.cli(Args)

# Simulated list of variables, each could be a GPU device ID for instance
devices = [f'cuda:{d}' for d in args.devices]

# Creating a Queue and populating it with the variables
device_queue = queue.Queue()
for var in devices:
    device_queue.put(var)

base_cmd = f"python cleanrl/ppo_rnd_gridworld.py --seed {args.seed}"
if args.track:
    base_cmd += f' --track --wandb-project-name {args.wandb_project_name}'

env_ids = [
    'Visual/DoorKey5x5-Gridworld-v0',
    # 'Visual/DoorKey6x6-Gridworld-v0',
    # 'Visual/DoorKey8x8-Gridworld-v0',
    # 'Visual/DoorKey16x16-Gridworld-v0',
    # 'Visual/NoisyDoorKey5x5-Gridworld-v0',
    # 'Visual/NoisyDoorKey6x6-Gridworld-v0',
    # 'Visual/NoisyDoorKey8x8-Gridworld-v0',
    # 'Visual/NoisyDoorKey16x16-Gridworld-v0',
    # 'Visual/MultiRoomS4N2-Gridworld-v0',
    # 'Visual/MultiRoomS5N4-Gridworld-v0',
    # 'Visual/MultiRoomS10N6-Gridworld-v0',
    # 'Visual/NoisyMultiRoomS4N2-Gridworld-v0',
    # 'Visual/NoisyMultiRoomS5N4-Gridworld-v0',
    # 'Visual/NoisyMultiRoomS10N6-Gridworld-v0',
]

_tags = [
    "doorkey5x5",
    "doorkey6x6",
    "doorkey8x8",
    "doorkey16x16",
    "doorkey5x5,noisy",
    "doorkey6x6,noisy",
    "doorkey8x8,noisy",
    "doorkey16x16,noisy",
    "multiroomS4N2",
    "multiroomS5N4",
    "multiroomS10N6",
    "multiroomS4N2,noisy",
    "multiroomS5N4,noisy",
    "multiroomS10N6,noisy",
]
tags = [f'\'{t}\'' for t in _tags]

if args.include_fixed:
    tags.extend([f'\'{t},fixed\'' for t in _tags])

# Preparing a list to hold commands that will be formatted with variables later
commands = [f'{base_cmd} --env-id {id}' for id in env_ids]
if args.include_fixed:
    commands.extend([f'{base_cmd} --env-id {id} --fixed' for id in env_ids])

if args.use_tag:
    commands = [f'{cmd} --wandb-tags {tag}' for cmd, tag in zip(commands, tags)]

commands = [f'{cmd} --device ' + '{0}' for cmd in commands]

print('\n'.join(commands))

# Number of concurrent commands you want to run. Adjust as per your needs.
max_workers = 4

def run_command(base_cmd):
    # Get a variable from the queue
    var = device_queue.get()
    """Function to execute a command in the terminal."""
    try:
        # Format the command with the variable
        cmd = base_cmd.format(var)

        # Execute the command
        result = subprocess.run(cmd, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Return the standard output
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Return the standard error if the command failed
        return e.stderr.strip()
    finally:
        # Always return the variable to the queue
        device_queue.put(var)


# Using ThreadPoolExecutor to manage concurrent execution
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Map commands to future tasks
    future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in commands}
    # As each command completes, print its result
    for future in concurrent.futures.as_completed(future_to_cmd):
        cmd = future_to_cmd[future]
        try:
            result = future.result()
            print(f"Result: '{result}' from '{cmd}'")
        except Exception as exc:
            print(f"Command '{cmd}' generated an exception: {exc}")


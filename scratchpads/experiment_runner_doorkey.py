from dataclasses import dataclass
import itertools
import subprocess
import concurrent.futures
import queue
from typing import List, Optional, Union 
import tyro

@dataclass
class Args:
    track: bool = False
    """track the experiment"""
    wandb_project_name: str = 'gridworld-rnd-hp-tuning-no-temp'
    """track the experiment"""
    repeats: int = 1
    """number of times to repeat"""
    use_tag: bool = True
    """use programmed tags for logging in wandb"""
    devices: tuple[int] = (0, 1, 2, 3)
    """avaliable device ids"""
    include_random: bool = False
    """avaliable device ids"""
    include_none_rnd: bool = False
    """avaliable device ids"""
    include_noisy: bool = False
    """avaliable device ids"""
    include_template: bool = False
    """avaiable"""
    mode: str = 'hard'
    """difficulty"""
    seed: int = 1
    """seed"""
    max_workers: int = 4
    """Max gpu workeres"""

args = tyro.cli(Args)

env_ids_and_tags = [
    # ('Visual/DoorKey5x5-Gridworld-v0' + ' --total-timesteps 2000000 --int-coef 0.2', 'doorkey5x5'),
    # ('Visual/DoorKey6x6-Gridworld-v0' + ' --total-timesteps 3000000 --int-coef 0.2', 'doorkey6x6'),
    ('Visual/DoorKey8x8-Gridworld-v0' + ' --total-timesteps 2500000 --int-coef 0.2', 'doorkey8x8'),
    # ('Visual/DoorKey16x16-Gridworld-v0' + ' --total-timesteps 20000000 --int-coef 1.0 --ext-coef 4.0 --update_epochs 8', 'doorkey16x16'),
    ('Visual/MultiRoomS10N6-Gridworld-v0'+ ' --cell_size 3 --total-timesteps 5000000 --int-coef 1.0 --update_epochs 8', 'multiroomS10N6'),
]

noisy_env_ids_and_tags = [
    # ('Visual/NoisyDoorKey5x5-Gridworld-v0', 'doorkey5x5,noisy'),
    # ('Visual/NoisyDoorKey6x6-Gridworld-v0', 'doorkey6x6,noisy'),
    ('Visual/NoisyDoorKey8x8-Gridworld-v0' + ' --total-timesteps 5000000 --int-coef 0.01 --update_epochs 8', 'doorkey8x8,noisy'),
    # ('Visual/NoisyDoorKey16x16-Gridworld-v0' + ' --total-timesteps 5000000 --int-coef 0.01  --update-epochs 8', 'doorkey16x16,noisy'),
    ('Visual/NoisyMultiRoomS10N6-Gridworld-v0'+ ' --cell_size 3 --total-timesteps 5000000 --int-coef 0.01 --update_epochs 8', 'multiroomS10N6,noisy'),
]

@dataclass(frozen=True)
class Option:
    name: str
    tags: Union[str, List[str]]
    args: Optional[List[str]]
    cmd_option: str

    def get_options(self):
        if args is None:
            return self.cmd_option, self.tags
        assert len(self.tags) == len(self.args)
        return [(f'{self.cmd_option} {arg}', tag) for arg, tag in zip(self.args, self.tags)]

def transpose(l):
    return list(map(list, zip(*l)))


def construct_all_commands(base_cmd: str, options: List[Option]):
    cmds = []
    str_options = map(lambda o: o.get_options(), options)
    for options in itertools.product(*str_options):
        opt, tags = transpose(options)
        cmd = f'{base_cmd} {" ".join(opt)}'
        if args.track:
            tags = filter(lambda t: t is not None, tags)
            cmd += f' --wandb-tags \'{",".join(tags)}\''
        cmds.append(cmd)
    return cmds


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

if __name__ == '__main__':
    # Create shared device allocation tracker
    # Simulated list of variables, each could be a GPU device ID for instance
    devices = [f'cuda:{d}' for d in args.devices]

    # Creating a Queue and populating it with the variables
    device_queue = queue.Queue()
    for var in devices:
        device_queue.put(var)
        

    # Construct commands
    base_cmd = f"python cleanrl/ppo_rnd_gridworld.py --device " + "{0}"
    if args.track:
        base_cmd += f' --track --wandb-project-name {args.wandb_project_name}'

    all_options = []
    if args.include_noisy:
        env_ids, env_tags = transpose(env_ids_and_tags + noisy_env_ids_and_tags)
    else:
        env_ids, env_tags = transpose(env_ids_and_tags)
    
    all_options.append(Option('env_id', env_tags, env_ids, '--env-id'))

    if args.include_random:
        all_options.append(Option('include_fixed', ['random','fixed_seed'], ['--no-fixed', '--fixed'], ''))
    else:
        all_options.append(Option('include_fixed', ['fixed_seed'], ['--fixed'], ''))
        

    if args.include_template:
        all_options.append(Option('include_template', ['template',None], ['--use-template', '--no-use-template'], ''))

    if args.include_none_rnd:
        all_options.append(Option('include_rnd', ['base','rnd'], [0, 1], '--int-coef'))


    # if args.mode == 'random':
    #     all_options.append(Option('wall_mode', ['wall_random'], ['random'], '--env-mode'))
    # elif args.mode == 'hard':
    #     all_options.append(Option('wall_mode', ['wall_hard'], ['hard'], '--env-mode'))


    if args.repeats:
        all_options.append(Option('seed', 
                                  [None,]*args.repeats, 
                                  list(args.seed+i for i in range(args.repeats)), 
                                  '--seed'))


    ext_coef = [2.0, 1.0, 0.5]
    ent_coef = [0.001, 0.005, 0.01]
    lr = [1e-4, 5e-5, 1e-5]
    all_options.append(Option('external coef', [None, ] * len(ext_coef), ext_coef, '--ext-coef'))
    all_options.append(Option('entropy coef', [None, ] * len(ent_coef), ent_coef, '--ent-coef'))
    all_options.append(Option('learning rate', [None, ] * len(lr), lr, '--learning-rate'))
    all_options.append(Option('include_template', [None, ], ['--no-use-template', ], ''))
    commands = construct_all_commands(base_cmd, all_options)
    print(f' ===== No. experiments: {len(commands)} ===== ', *commands, f' ===== No. experiments: {len(commands)} ===== ', sep='\n')


    # Using ThreadPoolExecutor to manage concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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


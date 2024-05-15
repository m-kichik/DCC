from argparse import ArgumentParser
from functools import partial
import os
import pickle
import random

import numpy as np
import torch
import torch.multiprocessing as mp
from pogema import pogema_v0, GridConfig

from model import Network
import config
from pogema_tools import PogemaWrapper, run_task

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)


def binary_map_to_str(binary_map: np.ndarray):
    str_map = "\n".join(["".join(line.astype(int).astype(str).tolist()) for line in binary_map])
    str_map = str_map.replace("0", ".")
    str_map = str_map.replace("1", "#")
    return str_map

def prepare_tests(test_set:list, on_target: str = "nothing", collision_system: str = "soft"):
    envs = []
    for test_map, agents_xy, targets_xy in test_set:
        map_size = test_map.shape[0]
        test_map = binary_map_to_str(test_map)
        agents_xy = agents_xy.tolist()
        targets_xy = targets_xy.tolist()

        
        grid_config = GridConfig(
            map=test_map,
            num_agents=len(agents_xy),
            agents_xy=agents_xy,
            targets_xy=targets_xy,
            size=map_size,
            max_episode_steps=config.max_episode_length,
            obs_radius=config.obs_radius,
            on_target=on_target,
            collision_system=collision_system,
            seed=config.test_seed,
        )

        env = PogemaWrapper(pogema_v0(grid_config=grid_config))
        envs.append(env)
        
    return envs


def run_test_set(
    model_file: str,
    test_set_file: str,
    on_target: str = "nothing",
    collision_system: str = "soft",
):
    network = Network()
    network.eval()
    network.to(DEVICE)
    state_dict = torch.load(os.path.join(model_file), map_location=DEVICE)
    network.load_state_dict(state_dict)
    network.eval()
    network.share_memory()

    with open(test_set_file, "rb") as file:
        test_set = pickle.load(file)

    test_envs = prepare_tests(test_set, on_target, collision_system)

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(partial(run_task, network=network), test_envs)

    # for grid_config in tqdm(test_configs):
    #     results.append(run_task(grid_config, network))
    
    success, steps, num_comm = zip(*results)
    print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
    print("average step: {}".format(sum(steps)/len(steps)))
    print("communication times: {}".format(sum(num_comm)/len(num_comm)))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str, help="path to model weights")
    parser.add_argument("--test_set", type=str, help="path to test set")
    parser.add_argument("--on_target", type=str, default="nothing", help="on_target in pogema")
    parser.add_argument("--collision_system", type=str, default="soft", help="collision_system in pogema")

    return parser.parse_args()


def main():
    args = parse_args()
    run_test_set(args.weights, args.test_set, args.on_target, args.collision_system)


if __name__ == "__main__":
    main()

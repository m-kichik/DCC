import os
import random
import warnings

warnings.simplefilter("ignore", UserWarning)
import numpy as np
import torch
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from tasks import crowdy_task

from model import Network
import config

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device("cpu")
torch.set_num_threads(1)


MAP_ACTIONS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}


class PogemaWrapper:
    def __init__(self, pogema_env, animate: bool = False):
        pogema_env.reset()

        self.animate = animate
        if self.animate:
            animation_config = AnimationConfig()
            animation_config.save_every_idx_episode = 0
            self.pogema_env = AnimationMonitor(
                pogema_env, animation_config=animation_config
            )
        else:
            self.pogema_env = pogema_env

        self.lifelong = self.pogema_env.grid_config.on_target == "restart"

        self.num_agents = len(
            np.array(self.pogema_env.get_agents_xy(ignore_borders=True))
        )

        self.map = np.array(self.pogema_env.get_obstacles(ignore_borders=True))
        self.obs_radius = self.pogema_env.grid_config.obs_radius
        self.targets = np.array(self.pogema_env.get_targets_xy(ignore_borders=True))

        self.heuristic_map = None
        self._get_heuristic_map()

        if self.pogema_env.grid_config.map_name is None:
            params = (
                str(self.pogema_env.grid_config.size),
                str(np.round(self.pogema_env.grid_config.density, 3)),
                str(self.pogema_env.grid_config.num_agents),
                str(self.pogema_env.grid_config.seed),
            )
            self.pogema_env.grid_config.map_name = "_".join(params)

    def reset(self, *args, **kwargs):
        obs, info = self.pogema_env.reset(*args, **kwargs)
        obs = self._process_observation(obs)
        return obs

    def step(self, actions):
        # obs = p_agents, p_obstacles, 4 * heuristic_map
        actions = [MAP_ACTIONS[a] for a in actions]

        obs, reward, terminated, truncated, info = self.pogema_env.step(actions)

        # FIXME: this is very bad
        if self.lifelong:
            targets = np.array(self.pogema_env.get_targets_xy(ignore_borders=True))
            if not np.array_equal(targets, self.targets):
                self.targets = targets
                self._get_heuristic_map()

        obs = self._process_observation(obs)

        pos = np.array(self.pogema_env.get_agents_xy(ignore_borders=False))
        done = np.sum(terminated) == self.num_agents

        return obs, pos, done

    def get_agents_xy(self, *args, **kwargs):
        return self.pogema_env.get_agents_xy(*args, **kwargs)

    def save_animation(self, name=None, animation_config=None):
        if not self.animate:
            print('This pogema env is not animated, use flag "animate" in constructor.')
        else:
            if name is None:
                name = f"{self.pogema_env.grid_config.map_name}.svg"
            return self.pogema_env.save_animation(name, animation_config)

    def _process_observation(self, obs):
        agents = np.array([o[1:-1] for o in obs], dtype=bool)
        agents[:, :, self.obs_radius, self.obs_radius] = False

        map_size = (self.map.shape[0], self.map.shape[1])
        obstacles = np.array([o[0:1] for o in obs], dtype=bool)

        # FIXME: the worst part in the project, replacing boarders with False
        for idx, agent_pos in enumerate(
            self.pogema_env.get_agents_xy(ignore_borders=True)
        ):
            x, y = agent_pos
            left_boarder = -(y + 1) + self.obs_radius
            if 0 <= left_boarder <= self.obs_radius * 2:
                obstacles[idx, 0, :, left_boarder] = False
            right_boarder = map_size[1] - y + self.obs_radius
            if 0 <= right_boarder <= self.obs_radius * 2:
                obstacles[idx, 0, :, right_boarder] = False
            top_boarder = -(x + 1) + self.obs_radius
            if 0 <= top_boarder <= self.obs_radius * 2:
                obstacles[idx, 0, top_boarder, :] = False
            bottom_boarder = map_size[0] - x + self.obs_radius
            if 0 <= bottom_boarder <= self.obs_radius * 2:
                obstacles[idx, 0, bottom_boarder, :] = False

        heuristics = self._get_heuristic_observations()

        return np.concatenate((agents, obstacles, heuristics), axis=1)

    def _get_heuristic_map(self):
        map_size = (self.map.shape[0], self.map.shape[1])

        dist_map = (
            np.ones((self.num_agents, *map_size), dtype=np.int32)
            * np.iinfo(np.int32).max
        )

        empty_pos = np.argwhere(self.map == 0).tolist()
        empty_pos = set([tuple(pos) for pos in empty_pos])

        for i in range(self.num_agents):
            open_list = set()
            x, y = tuple(self.targets[i])
            open_list.add((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop()
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up in empty_pos and dist_map[i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    open_list.add(up)

                down = x + 1, y
                if down in empty_pos and dist_map[i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    open_list.add(down)

                left = x, y - 1
                if left in empty_pos and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    open_list.add(left)

                right = x, y + 1
                if right in empty_pos and dist_map[i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    open_list.add(right)

        self.heuristic_map = np.zeros((self.num_agents, 4, *map_size), dtype=bool)

        for x, y in empty_pos:
            for i in range(self.num_agents):

                if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                    self.heuristic_map[i, 0, x, y] = 1

                if x < map_size[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                    self.heuristic_map[i, 1, x, y] = 1

                if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                    self.heuristic_map[i, 2, x, y] = 1

                if y < map_size[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                    self.heuristic_map[i, 3, x, y] = 1

        self.heuristic_map = np.pad(
            self.heuristic_map,
            (
                (0, 0),
                (0, 0),
                (self.obs_radius, self.obs_radius),
                (self.obs_radius, self.obs_radius),
            ),
        )

    def _get_heuristic_observations(self):
        agents_positions = self.pogema_env.get_agents_xy(ignore_borders=True)
        h_obs = np.zeros(
            (self.num_agents, 4, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
            dtype=bool,
        )

        for i, agent_pos in enumerate(agents_positions):
            x, y = agent_pos

            h_obs[i] = self.heuristic_map[
                i, :, x : x + 2 * self.obs_radius + 1, y : y + 2 * self.obs_radius + 1
            ]

        return h_obs


def run_task(env, network):
    observation = env.reset()
    positions = np.array(env.get_agents_xy(ignore_borders=True))
    num_agents = len(positions)
    last_actions = np.zeros((num_agents, 5))

    done = False
    network.reset()

    step = 0
    num_comm = 0
    while not done and step < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(
            torch.as_tensor(observation.astype(np.float32)).to(DEVICE),
            torch.as_tensor(last_actions.astype(np.float32)).to(DEVICE),
            torch.as_tensor(positions.astype(int)),
        )
        last_actions = np.zeros((num_agents, 5))
        last_actions[np.arange(num_agents), np.array(actions)] = 1
        observation, positions, done = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)

    return done, step, num_comm


def main():
    network = Network()
    network.eval()
    network.to(DEVICE)
    state_dict = torch.load(
        os.path.join(config.save_path, f"128000.pth"), map_location=DEVICE
    )
    network.load_state_dict(state_dict)
    network.eval()

    grid_config = GridConfig(
        map=crowdy_task["map"],
        map_name=crowdy_task["map_name"],
        num_agents=len(crowdy_task["agents_xy"]),
        agents_xy=crowdy_task["agents_xy"],
        targets_xy=crowdy_task["targets_xy"],
        # size=16,
        # density=0.25,
        max_episode_steps=config.max_episode_length,
        obs_radius=config.obs_radius,
        # on_target="restart",
        # on_target="nothing",
        on_target="finish",
        # collision_system="priority",
        # collision_system="block_both",
        collision_system="soft",
        seed=1,
    )
    # grid_config.targets_xy=targets_xy

    env = PogemaWrapper(pogema_v0(grid_config=grid_config), animate=True)

    print(run_task(env, network))

    env.save_animation()


if __name__ == "__main__":
    main()

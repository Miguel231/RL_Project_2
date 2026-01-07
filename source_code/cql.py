from collections import defaultdict
import random
from typing import List, Tuple

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class CQL:
    """
    Centralized Q-Learning agent
    Learns a single Q-function over joint observations and joint actions
    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(a) for a in action_spaces]

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Joint Q-table
        # Access: Q[(obs_tuple, action_tuple)]
        self.q_table = defaultdict(lambda: 0)

        # Precompute all joint actions
        self.joint_actions = self._compute_joint_actions()

    def _compute_joint_actions(self) -> List[Tuple[int, ...]]:
        """
        Returns all possible joint actions
        """
        if self.num_agents == 2:
            return [(a1, a2)
                    for a1 in range(self.n_acts[0])
                    for a2 in range(self.n_acts[1])]
        else:
            raise NotImplementedError("CQL implemented only for 2 agents")

    def act(self, obss) -> List[int]:
        """
        Epsilon-greedy selection over joint actions
        """
        obs_tuple = tuple(obss)

        if random.random() < self.epsilon:
            joint_action = random.choice(self.joint_actions)
        else:
            q_values = [
                self.q_table[str((obs_tuple, ja))]
                for ja in self.joint_actions
            ]
            joint_action = self.joint_actions[int(np.argmax(q_values))]

        return list(joint_action)

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Centralized Q-learning update
        """
        obs_tuple = tuple(obss)
        next_obs_tuple = tuple(n_obss)
        action_tuple = tuple(actions)

        current_q = self.q_table[str((obs_tuple, action_tuple))]

        # Centralized reward: sum of individual rewards
        reward = sum(rewards)

        if done:
            target = reward
        else:
            next_qs = [
                self.q_table[str((next_obs_tuple, ja))]
                for ja in self.joint_actions
            ]
            target = reward + self.gamma * max(next_qs)

        self.q_table[str((obs_tuple, action_tuple))] = (
            current_q + self.learning_rate * (target - current_q)
        )

        return self.q_table[str((obs_tuple, action_tuple))]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Linear epsilon decay (same as IQL)
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99

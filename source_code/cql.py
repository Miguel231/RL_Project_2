from collections import defaultdict
import random
from typing import List, DefaultDict, Tuple

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim
import itertools

class CQL:
    """
    Agent using the Central Q-Learning algorithm
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 0.1,
        **kwargs,
    ):
        """
        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.alpha = alpha

        self.q_table = defaultdict(lambda:0.0)
        self.joint_actions = list(itertools.product(*[range(n) for n in self.n_acts]))



    def act(self, obss) -> List[int]:
        """
        Implement the epsilon-greedy action
        """
        ### PUT YOUR CODE HERE ###

        joint_state = tuple(obss)
        if random.random() < self.epsilon:
            action = random.choice(self.joint_actions)
        else:
            q_values = [float(self.q_table[str((joint_state, tuple(a)))]) for a in self.joint_actions]
            idx = int(np.argmax(q_values))
            action = self.joint_actions[idx]

        return list(action)

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        reward: float,
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the Q-tables based on experience

        **IMPLEMENT THIS FUNCTION**

        """
        ### PUT YOUR CODE HERE ###
        joint_state = tuple(float(x) for obs in obss for x in np.ravel(obs))
        joint_action = tuple(actions)
        next_joint_state = tuple(float(x) for obs in n_obss for x in np.ravel(obs))
        current_q = float(self.q_table[str((joint_state, joint_action))])

        #penalty
        q_values = [float(self.q_table[str((joint_state, a))]) for a in self.joint_actions]
        penalty = np.log(np.sum(np.exp(q_values))) - current_q

        if done:
            target = reward
        else:
            q_values_next = [float(self.q_table[str((next_joint_state, tuple(a)))]) for a in self.joint_actions]
            target = reward + self.gamma * max(q_values_next)

        target = target - self.alpha * penalty
        self.q_table[str((joint_state, joint_action))] = current_q + self.learning_rate * (target - current_q)

        return self.q_table[str((joint_state, joint_action))]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99

import enum

import numpy as np
from reinforcement_learning.network.action_value_network import ActionValueNetwork
from .algorithm import Algorithm


class TDAlgorithmName(enum.Enum):
    SARSA = 1,
    Q = 2,
    EXPECTED_SARSA = 3,
    MCQL = 4

    @staticmethod
    def all():
        return [TDAlgorithmName.SARSA, TDAlgorithmName.Q, TDAlgorithmName.EXPECTED_SARSA, TDAlgorithmName.MCQL]


class TDAlgorithm(Algorithm):
    policy = None

    def calculate_target_value(self, a, s_, r, active, network):
        a_ = self.get_potential_action(network, s_, a, network.is_double)
        return r + (self.discount_factor * self.get_scalar(s_, a_, network, False) * active)

    def get_target_error(self, s, a, s_, r, active, network):
        return self.calculate_target_value(a, s_, r, active, network) - self.get_scalar(s, a, network, False)

    def get_potential_action(self, network: ActionValueNetwork, s, a, use_target):
        if self.algorithm_name == TDAlgorithmName.SARSA:
            return self.policy.choose_action(s, network=network, use_target=use_target)
        elif self.algorithm_name == TDAlgorithmName.Q:
            return self.policy.argmax(
                network.get_target_action_values(s) if use_target else network.get_action_values(s))
        elif self.algorithm_name == TDAlgorithmName.MCQL:
            return a
        else:
            return -1

    def get_scalar(self, s, a, network: ActionValueNetwork, use_target):
        q_mat = network.get_target_action_values(s) if use_target else network.get_action_values(s)
        if self.algorithm_name == TDAlgorithmName.EXPECTED_SARSA:
            policy_mat = self.policy.derive(s, network=network, use_target=use_target)
            dot_product = np.dot(policy_mat, q_mat.T)
            try:
                return dot_product[0, 0]
            except IndexError:
                return dot_product[0]

        else:
            try:
                return q_mat[0, a]
            except IndexError:
                return q_mat[a]

import enum

import numpy as np


class AlgorithmName(enum.Enum):
    SARSA = 1,
    Q = 2,
    EXPECTED_SARSA = 3,
    MCQL = 4

    @staticmethod
    def all():
        return [AlgorithmName.SARSA, AlgorithmName.Q, AlgorithmName.EXPECTED_SARSA, AlgorithmName.MCQL]


class Algorithm:
    discount_factor = 0.0
    policy = None
    enable_iterator = False
    iterator = None
    algorithm_name = None

    def __init__(self, args={}):
        for key in args:
            setattr(self, key, args[key])

    def calculate_target_value(self, a, s_, r, active, net1, net2):
        a_ = self.get_potential_action(net2, s_,
                                       a) if net2 is not None else self.get_potential_action(net1,
                                                                                             s_, a)
        return r + (self.discount_factor * self.get_scalar(s_, a_, net1) * active)

    def get_potential_action(self, network, s, a):
        if self.algorithm_name == AlgorithmName.SARSA:
            return self.policy.choose_action(s, network)
        elif self.algorithm_name == AlgorithmName.Q:
            return self.policy.argmax(network.get_action_values(s))
        elif self.algorithm_name == AlgorithmName.MCQL:
            return a
        else:
            return -1

    def get_scalar(self, s, a, network):
        q_mat = network.get_action_values(s)
        if self.algorithm_name == AlgorithmName.EXPECTED_SARSA:
            policy_mat = self.policy.derive(s, network)
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

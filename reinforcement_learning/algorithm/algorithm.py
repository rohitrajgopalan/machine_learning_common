import enum

import numpy as np

from reinforcement_learning.supervised.target_value_predictor import TargetValuePredictor


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

    def iterator_init(self, csv_dir, state_dim, dl_args=None):
        if not self.enable_iterator:
            pass
        self.iterator = TargetValuePredictor(csv_dir, state_dim, self, self.policy, dl_args)

    def calculate_target_value(self, a, s_, r, active, policy_network, target_network=None):
        a_ = self.get_potential_action(target_network, s_,
                                       a) if target_network is not None else self.get_potential_action(policy_network,
                                                                                                       s_, a)
        return r + (self.discount_factor * self.get_scalar(s_, a_, policy_network) * active)

    def get_target_value(self, s, a, s_, r, active, network):
        target_value = self.calculate_target_value(a, s_, r, active, network)
        if self.enable_iterator:
            predicted_value = self.iterator.get_target_value(s, a)
            self.iterator.update_predictor(s, a, target_value)
            return predicted_value
        else:
            return target_value

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
        if self.algorithm_name == AlgorithmName.EXPECTED_SARSA:
            policy_mat = self.policy.derive(s, network)
            q_mat = network.get_action_values(s)
            return np.dot(policy_mat, q_mat.T)[0]
        else:
            q_mat = network.get_action_values(s)
            return q_mat[0, a]

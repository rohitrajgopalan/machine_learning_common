import enum

import numpy as np

from reinforcement_learning.supervised.target_value_predictor import TargetValuePredictor


class AlgorithmName(enum.Enum):
    SARSA = 1,
    Q = 2,
    EXPECTED_SARSA = 3,
    MCQL = 4


class Algorithm:
    discount_factor = 0.0
    policy = None
    lambda_val = 0.0
    enable_e_traces = False
    enable_iterator = False
    iterator = None
    num_actions = 0
    algorithm_name = None

    def __init__(self, args={}):
        for key in args:
            setattr(self, key, args[key])
        self.e_traces = {}

    def add_state(self, s):
        if s is None:
            pass
        if not type(s) == int:
            if type(s) == np.ndarray:
                s = tuple(s.reshape(1, -1)[0])
            elif type(s) == list:
                s = tuple(s)
        if s not in self.e_traces:
            self.e_traces.update({s: [0] * self.num_actions})

    def add_action(self):
        self.num_actions += 1
        for s in self.e_traces:
            self.e_traces[s].append(0)

    def iterator_init(self, csv_dir, state_dim, network_type):
        if not self.enable_iterator:
            pass
        self.iterator = TargetValuePredictor(csv_dir, state_dim, self, self.policy, network_type)

    def calculate_target_error(self, s, a, s_, r, active, network, current_q):
        self.add_state(s)
        coin_side = network.determine_coin_side()

        if not type(s) == int:
            if type(s) == np.ndarray:
                s = tuple(s.reshape(1, -1)[0])
            elif type(s) == list:
                s = tuple(s)
        if self.enable_e_traces and a in range(len(self.e_traces[s])):
            self.e_traces[s][a] += 1

        target_error = self.get_target_value(s, a, s_, r, active, current_q, coin_side) - self.get_scalar(s, a,
                                                                                                          coin_side,
                                                                                                          network)
        if self.enable_e_traces and a in range(len(self.e_traces[s])):
            target_error *= self.e_traces[s][a]

        if self.enable_e_traces:
            max_num_actions = self.num_actions
            for state in self.e_traces:
                if len(self.e_traces[state]) > max_num_actions:
                    max_num_actions = len(self.e_traces[state])
                for a_ in range(len(self.e_traces[state])):
                    self.e_traces[state][a_] *= (self.lambda_val * self.discount_factor)
            self.num_actions = max_num_actions

        return target_error, coin_side

    def get_target_error(self, s, a, s_, r, active, network, current_q):
        return 0, 0

    def calculate_target_value(self, a, s_, r, active, current_q, coin_side):
        a_ = self.get_potential_action(current_q, coin_side, s_, a)
        return r + (self.discount_factor * self.get_scalar(s_, a_, current_q.network_type.value - 1 - coin_side,
                                                           current_q) * active)

    def get_target_value(self, s, a, s_, r, active, current_q, coin_side):
        target_value = self.calculate_target_value(a, s_, r, active, current_q, coin_side)
        if self.enable_iterator:
            predicted_value = self.iterator.predict(s, a)
            self.iterator.add(s, a, target_value)
            return predicted_value
        else:
            return target_value

    def get_potential_action(self, current_q, coin_side, s, a):
        if self.algorithm_name == AlgorithmName.SARSA:
            return self.policy.choose_action(s, current_q, coin_side)
        elif self.algorithm_name == AlgorithmName.Q:
            return self.policy.argmax(current_q.get_action_values(s, coin_side))
        elif self.algorithm_name == AlgorithmName.MCQL:
            return a
        else:
            return -1

    def get_scalar(self, s, a, coin_side, network):
        if self.algorithm_name == AlgorithmName.EXPECTED_SARSA:
            policy_mat = self.policy.derive(s, network, coin_side)
            q_mat = network.get_action_values(s, coin_side)
            return np.dot(policy_mat, q_mat.T)[0]
        else:
            q_mat = network.get_action_values(s, coin_side)
            return q_mat[0, a]


class TDAlgorithm(Algorithm):
    def get_target_error(self, s, a, s_, r, terminal, network, current_q):
        return self.calculate_target_error(s, a, s_, r, terminal, network, current_q)


class TDLambdaAlgorithm(TDAlgorithm):
    def get_target_error(self, s, a, s_, r, terminal, network, current_q):
        target_error_prime, coin_side = self.calculate_target_error(s, a, s_, r, terminal, network, current_q)
        target_value = self.get_target_value(s, a, s_, r, terminal, current_q, coin_side)
        target_error = target_value - self.get_scalar(s, self.get_potential_action(network, coin_side, s, a), coin_side,
                                                      network)
        return target_error_prime + target_error, coin_side

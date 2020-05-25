from reinforcement_learning.algorithm.base_algorithm import Algorithm
import numpy as np


class TDAlgorithm(Algorithm):
    def get_target_error(self, s, a, s_, r, terminal, network, current_q):
        return self.calculate_target_error(s, a, s_, r, terminal, network, current_q)


class SARSA(TDAlgorithm):
    def get_potential_next_action(self, current_q, coin_side, s_, a):
        return self.policy.choose_action(current_q.get_action_values_with_weights(s_, coin_side))


class Q(TDAlgorithm):
    def get_potential_next_action(self, current_q, coin_side, s_, a):
        return self.argmax(current_q.get_action_values_with_weights(s_, coin_side))


class ExpectedSARSA(TDAlgorithm):
    def get_scalar(self, s, a, coin_side, network):
        q_mat = network.get_action_values(s, coin_side)
        policy_mat = self.policy.derive(q_mat)
        return np.dot(policy_mat, q_mat.T)[0]


class MCQL(TDAlgorithm):
    def get_potential_action(self, current_q, coin_side, s, a):
        return a

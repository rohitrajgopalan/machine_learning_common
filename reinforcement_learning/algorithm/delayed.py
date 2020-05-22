from reinforcement_learning.algorithm.base_algorithm import Algorithm
import numpy as np


class DelayedAlgorithm(Algorithm):
    # used for attempted updates
    U = []
    # counters
    counter = {}
    # time of last attempted update
    last_attempted_update = {}
    LEARN = {}
    timesteps = 0
    # time of most recent Q-value change
    t_most_recent_Q_change = 0
    # max counter value for any state-action pair
    max_counter = 0
    # threshold delta between Q and update values
    epsilon = 0.0

    def add_state(self, s, network_type):
        if s is None:
            pass
        super().add_state(s, network_type)
        if len(self.U) == 0:
            self.U = [dict() for _ in range(network_type.value)]
        for i in range(network_type.value):
            if s not in self.U[i]:
                self.U[i].update({s: [0] * self.policy.num_actions})
        if s not in self.counter:
            self.counter.update({s: [0] * self.policy.num_actions})
        if s not in self.last_attempted_update:
            self.last_attempted_update.update({s: [0] * self.policy.num_actions})
        if s not in self.LEARN:
            self.LEARN.update({s: [0] * self.policy.num_actions})

    def get_target_error(self, s, a, s_, r, terminal, network, current_q):
        coin_side = network.determine_coin_side()
        target_error = 0
        self.timesteps += 1
        if self.LEARN[s][a]:
            self.U[coin_side][s][a] += self.get_target_value(a, s_, r, terminal, current_q, coin_side)
            self.counter[s][a] += 1
            if self.counter[s][a] == self.max_counter:
                q_mat = network.get_action_values(s, coin_side)
                if (q_mat[a] - self.U[coin_side][s][a]) / self.max_counter >= (2 * self.epsilon):
                    target_error = self.U[coin_side][s][a] / self.max_counter
                    if coin_side == 0:
                        target_error += self.epsilon
                    else:
                        target_error -= self.epsilon
                    self.t_most_recent_Q_change = self.timesteps
                elif self.last_attempted_update[s][a] >= self.t_most_recent_Q_change:
                    self.LEARN[s][a] = False
                self.last_attempted_update[s][a] = 0
                self.U[coin_side][s][a] = 0
                self.counter[s][a] = 0
        elif self.last_attempted_update[s][a] < self.t_most_recent_Q_change:
            self.LEARN[s][a] = True
        return target_error, coin_side


class DelayedSARSA(DelayedAlgorithm):
    def get_potential_action(self, current_q, coin_side, s_, a):
        return self.policy.choose_action(current_q.get_action_values(s_))


class DelayedQ(DelayedAlgorithm):
    def get_potential_action(self, current_q, coin_side, s_, a):
        return self.argmax(current_q.get_action_values(s_, coin_side))


class DelayedExpectedSARSA(DelayedAlgorithm):
    def get_scalar(self, s, a, coin_side, network):
        q_mat = network.get_action_values(s, coin_side)
        policy_mat = self.policy.derive(q_mat)
        return np.dot(policy_mat, q_mat.T)[0]


class DelayedMCQL(DelayedAlgorithm):
    def get_potential_next_action(self, current_q, coin_side, s_, a):
        return a

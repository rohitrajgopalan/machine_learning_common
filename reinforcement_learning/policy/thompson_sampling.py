from .base_policy import Policy
import numpy as np


class ThompsonSampling(Policy):
    alpha = None
    beta = None

    def __init__(self, args=None):
        super().__init__(args)
        self.alpha = np.ones(self.num_actions)
        self.beta = np.ones(self.num_actions)

    def generate_theta(self):
        return np.random.beta(self.alpha, self.beta)

    def derive(self, action_values):
        policy_probs = np.zeros(self.num_actions)
        actions_with_max_theta = self.actions_with_max(self.generate_theta())
        for action in range(self.num_actions):
            if action in actions_with_max_theta:
                policy_probs[action] = 1 / len(actions_with_max_theta)

        return policy_probs

    def choose_action(self, action_values):
        return self.argmax(self.generate_theta())

    def update(self, action, reward):
        r = 0
        if reward > -2:
            r = 1
        self.alpha[action] += r
        self.beta[action] += 1 - r

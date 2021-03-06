import numpy as np

from .policy import Policy


class ThompsonSampling(Policy):
    alpha = None
    beta = None

    def __init__(self, args):
        super().__init__(args)
        self.alpha = np.ones(self.num_actions)
        self.beta = np.ones(self.num_actions)

    def generate_theta(self):
        return np.random.beta(self.alpha, self.beta)

    def derive_policy_based_from_values(self, action_values):
        policy_probs = np.zeros(self.num_actions)
        actions_with_max_theta = self.actions_with_max_value(self.generate_theta())
        for action in range(self.num_actions):
            if action in actions_with_max_theta:
                policy_probs[action] = 1 / len(actions_with_max_theta)

        return policy_probs

    def choose_action_based_from_values(self, action_values):
        return self.argmax(self.generate_theta()) if self.num_actions > 1 else 0

    def update(self, action, should_action_be_blocked=False, **args):
        r = 0 if should_action_be_blocked else 1
        self.alpha[action] += r
        self.beta[action] += 1 - r


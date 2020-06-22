import numpy as np

from .base_policy import Policy


class EpsilonGreedy(Policy):
    epsilon = 0.0

    def derive_policy_based_from_values(self, action_values):
        actions_with_max = self.actions_with_max_value(action_values)
        policy_probs = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            if action in actions_with_max:
                policy_probs[action] = (1 - self.epsilon) / len(actions_with_max)
            else:
                policy_probs[action] = self.epsilon / (self.num_actions - len(actions_with_max))

        return policy_probs

    def choose_action_based_from_values(self, action_values):
        if self.rand_generator.rand() < self.epsilon:
            return self.rand_generator.choice(self.num_actions)
        else:
            return self.argmax(action_values)

import numpy as np

from .policy import Policy


class Softmax(Policy):
    # tau (float): The temperature parameter scalar.
    tau = 0.0

    def derive_policy_based_from_values(self, action_values):
        if self.num_actions == 1:
            return 1
        else:
            action_probs = np.exp(action_values/self.tau)
            policy = action_probs/np.sum(action_probs)
            if np.isnan(policy.any()):
                return np.full(self.num_actions, 1/self.num_actions)
            else:
                return policy

    def choose_action_based_from_values(self, action_values):
        policy = self.derive_policy_based_from_values(action_values)
        if self.num_actions == 1:
            return 0
        else:
            try:
                return self.rand_generator.choice(self.num_actions, p=policy.flatten())
            except ValueError:
                return self.rand_generator.choice(self.num_actions)

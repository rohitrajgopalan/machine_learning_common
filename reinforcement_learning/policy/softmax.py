import numpy as np

from .policy import Policy


class Softmax(Policy):
    # tau (float): The temperature parameter scalar.
    tau = 0.0

    def derive_policy_based_from_values(self, action_values):
        action_probs = np.exp(action_values/self.tau)
        sum_probs = np.sum(action_probs)
        return action_probs/sum_probs

    def choose_action_based_from_values(self, action_values):
        probs_batch = self.derive_policy_based_from_values(action_values)
        try:
            if self.num_actions == 1:
                return 0
            else:
                return self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
        except TypeError:
            return self.rand_generator.choice(self.num_actions, p=probs_batch)

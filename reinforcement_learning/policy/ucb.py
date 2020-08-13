import numpy as np

from .policy import Policy


class UCB(Policy):
    N = None
    time_step_counter = 1
    ucb_c = 0.0

    def __init__(self, args):
        super().__init__(args)
        self.N = np.full(self.num_actions, 0.001)

    def generate_confidence(self):
        ln_timestep = np.log(np.full(self.num_actions, self.time_step_counter))
        return self.ucb_c * np.sqrt(ln_timestep / self.N)

    def derive_policy_based_from_values(self, action_values):
        policy_probs = np.zeros(self.num_actions)
        actions_with_max = self.actions_with_max_value(action_values + self.generate_confidence())
        for action in range(self.num_actions):
            if action in actions_with_max:
                policy_probs[action] = 1 / len(actions_with_max)
        return policy_probs

    def choose_action_based_from_values(self, action_values):
        return self.argmax(action_values + self.generate_confidence()) if self.num_actions > 1 else 0

    def update(self, action, should_action_be_blocked=False, **args):
        self.N[action] += 1
        self.time_step_counter += 1

    def get_hyper_parameter(self):
        return self.ucb_c

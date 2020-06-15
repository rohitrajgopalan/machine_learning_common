import numpy as np


class Policy:
    num_actions = 0
    rand_generator = None
    min_penalty = -1

    def __init__(self, args=None):
        if args is None:
            args = {}
        for key in args:
            if key == 'random_seed':
                self.rand_generator = np.random.RandomState(args[key])
            else:
                setattr(self, key, args[key])

    def actions_with_max_value(self, action_values):
        max_value = np.max(action_values)
        ties = []
        for i in range(self.num_actions):
            if action_values[0, i] == max_value:
                ties.append(i)
        return ties

    def add_action(self):
        self.num_actions += 1

    def argmax(self, action_values):
        ties = self.actions_with_max_value(action_values)
        return self.rand_generator.choice(ties)

    def derive(self, state, network=None, coin_side=0):
        return self.derive_policy_based_from_values(network.get_action_values(state, coin_side))

    def derive_policy_based_from_values(self, action_values):
        return np.zeros(self.num_actions)

    def choose_action(self, state, network=None, coin_side=0):
        return self.choose_action_based_from_values(network.get_action_values(state, coin_side))

    def choose_action_based_from_values(self, action_values):
        return 0

    def update(self, action, reward):
        pass

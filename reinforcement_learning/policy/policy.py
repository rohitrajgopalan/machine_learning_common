import numpy as np
from reinforcement_learning.network.action_value_network import ActionValueNetwork


class Policy:
    num_actions = 0
    rand_generator = None

    def __init__(self, args):
        for key in args:
            if key == 'random_seed':
                self.rand_generator = np.random.RandomState(args[key])
            else:
                setattr(self, key, args[key])

    def actions_with_max_value(self, action_values):
        max_value = np.max(action_values)
        ties = []
        for i in range(self.num_actions):
            if action_values.shape == (1, self.num_actions):
                if action_values[0, i] == max_value:
                    ties.append(i)
            else:
                if action_values[i] == max_value:
                    ties.append(i)
        return ties

    def argmax(self, action_values):
        ties = self.actions_with_max_value(action_values)
        return self.rand_generator.choice(ties)

    def derive(self, state, **args):
        network = args['network']
        use_target = args['use_target']
        action_values = network.get_target_action_values(state) if use_target else network.get_action_values(state)
        return self.derive_policy_based_from_values(action_values)

    def derive_policy_based_from_values(self, action_values):
        return np.zeros((1, self.num_actions))

    def choose_action(self, state, **args):
        network = args['network']
        use_target = args['use_target'] if 'use_target' in args else False
        action_values = network.get_target_action_values(state) if use_target else network.get_action_values(state)
        return self.choose_action_based_from_values(action_values), 0

    def choose_action_based_from_values(self, action_values):
        return 0

    def update(self, action, should_action_be_blocked=False, **args):
        pass

    def get_hyper_parameter(self):
        return 0

import numpy as np
import tensorflow as tf

from neural_network.keras_neural_network import KerasNeuralNetwork
from .policy import Policy


class NetworkPolicy(Policy):
    policy_network = None
    use_td_error_in_function = False

    def __init__(self, args):
        super().__init__(args)
        super().__init__(args)
        self.action_dim = args['action_dim'] if 'action_dim' in args else 0
        network_args = args['network_args'] if 'network_args' in args else {}
        self.use_td_error_in_function = args[
            'use_td_error_in_function'] if 'use_td_error_in_function' in args else False
        network_args.update({'num_outputs': self.num_actions})
        if self.use_td_error_in_function:
            pass
        else:
            network_args.update({'loss_function': 'categorical_crossentropy'})
        self.policy_network = KerasNeuralNetwork(network_args)

    def derive(self, state, **args):
        if state is None:
            action_probs = np.full(self.num_actions, 1 / self.num_actions)
        else:
            try:
                action_probs = self.policy_network.predict(np.array([state]))[0]
            except:
                action_probs = np.full(self.num_actions, 1 / self.num_actions)
        if not type(action_probs) == np.ndarray:
            action_probs = action_probs.numpy()
        if not np.sum(action_probs) == 1 or np.isnan(action_probs.any()):
            action_probs = np.full(self.num_actions, 1 / self.num_actions)
        return action_probs

    def choose_action(self, state, **args):
        action_probs = self.derive(state)
        chosen_action = np.random.choice(self.num_actions, p=action_probs)
        picked_action_prob = action_probs[chosen_action]
        return chosen_action, picked_action_prob

    def update(self, action, should_action_be_blocked=False, **args):
        td_error = args['td_error'] if 'td_error' in args else 0
        state = args['state'] if 'state' in args else None
        assert state is not None
        states = np.array([state])
        if not type(td_error) == np.ndarray:
            td_error = np.array([td_error]).reshape((-1, 1))
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        a_indices_one_hot = np.array([action_one_hot]).reshape((-1, self.num_actions))

        if self.use_td_error_in_function:
            picked_action_prob = args['picked_action_prob'] if 'picked_action_prob' in args else 0
            picked_action_probs = np.array([picked_action_prob]).reshape((-1, 1))
            self.optimize_network(states, td_error, picked_action_probs=picked_action_probs)
        else:
            self.optimize_network(states, td_error, a_indices_one_hot=a_indices_one_hot)

    def optimize_network(self, states, td_error, **args):
        if self.use_td_error_in_function:
            pass
        else:
            a_indices_one_hot = args['a_indices_one_hot']
            self.policy_network.fit(states, a_indices_one_hot, advantages=td_error)

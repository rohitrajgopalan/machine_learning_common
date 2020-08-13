from .policy import Policy
from neural_network.neural_network import NeuralNetwork
from neural_network.network_types import *
import numpy as np


class NetworkPolicy(Policy):
    network = None

    def __init__(self, args):
        super().__init__(args)
        network_args = args['network_args'] if 'network_args' in args else {}
        state_dim = args['state_dim'] if 'state_dim' in args else 0
        optimizer_type = args['optimizer_type'] if 'optimizer_type' in args else NetworkOptimizer.ADAM
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        if type(state_dim) == tuple and len(state_dim) > 1:
            add_pooling = args['add_pooling'] if 'add_pooling' in args else False
            convert_to_grayscale = args['convert_to_grayscale'] if 'convert_to_grayscale' in args else False
        else:
            add_pooling = False
            convert_to_grayscale = False

        if network_args == {}:
            network_args = {
                'optimizer_type': optimizer_type,
                'optimizer_args': optimizer_args,
                'num_inputs' if type(state_dim) == int else 'input_shape': state_dim,
                'num_outputs': self.num_actions,
                'loss_function': 'categorical_crossentropy',
                'dense_layer_info_list': [
                    {'activation_function': NetworkActivationFunction.RELU,
                     'num_units': 'auto'},
                    {'activation_function': NetworkActivationFunction.SOFTMAX}
                ]
            }
            if type(state_dim) == tuple and len(state_dim) > 1:
                network_args.update({'add_pooling': add_pooling,
                                     'convert_to_grayscale': convert_to_grayscale,
                                     'conv_layer_info_list': [
                                         {'num_dimensions': 2, 'filters': 32, 'kernel_size': (8, 8), 'strides': 4,
                                          'activation_function': NetworkActivationFunction.RELU},
                                         {'num_dimensions': 2, 'filters': 64, 'kernel_size': (4, 4), 'strides': 2,
                                          'activation_function': NetworkActivationFunction.RELU},
                                         {'num_dimensions': 2, 'filters': 128, 'kernel_size': (3, 3), 'strides': 1,
                                          'activation_function': NetworkActivationFunction.RELU}
                                     ]})
        else:
            if 'num_inputs' not in network_args or 'input_shape' not in network_args:
                network_args.update({'num_inputs' if type(state_dim) == int else 'input_shape': state_dim})
            if 'loss_function' not in network_args:
                network_args.update({'loss_function': 'categorical_crossentropy'})
            if type(state_dim) == tuple and len(state_dim) > 1:
                if 'add_pooling' not in network_args:
                    network_args.update({'add_pooling': add_pooling})
                if 'convert_to_grayscale' not in network_args:
                    network_args.update({'convert_to_grayscale': convert_to_grayscale})
            network_args.update({'num_outputs': self.num_actions})
        self.network = NeuralNetwork.choose_neural_network(network_args)

    def derive(self, state, network, use_target=False):
        prediction = self.network.predict(np.array([state]))
        return prediction.ravel()

    def choose_action(self, state, network, use_target=False):
        policy = self.derive(state, network, use_target)
        return np.random.choice(self.num_actions, p=policy)

    def update(self, action, should_action_be_blocked=False, **args):
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1

        advantages = args['td_error'] if 'td_error' in args else 0
        state = args['state'] if 'state' in args else None

        assert state is not None
        self.network.fit(np.array([state]), np.array([action_one_hot]), sample_weight=np.array([advantages]))

import numpy as np
import tensorflow as tf

from neural_network.network_types import NetworkActivationFunction
from neural_network.keras_neural_network import KerasNeuralNetwork

TAU = 0.001


class DDPGActorNetwork:
    """
    Reference: https://github.com/maxkferg/DDPG/blob/master/ddpg/actor_network.py
    """

    def __init__(self, state_dim, action_dim, layer_sizes, optimizer_type, optimizer_args={}, tau=TAU,
                 conv_layer_info_list=[],
                 add_pooling=False, convert_to_grayscale=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau

        network_args = {
            'add_pooling': add_pooling,
            'convert_to_grayscale': convert_to_grayscale,
            'optimizer_type': optimizer_type,
            'optimizer_args': optimizer_args,
            'num_inputs' if type(self.state_dim) == int else 'input_shape': self.state_dim,
            'num_outputs': self.action_dim,
            'use_gradients': True,
            'is_sequential': False
        }

        for conv_layer_info in conv_layer_info_list:
            conv_layer_info.update({'add_batch_norm': True})
        network_args.update({'conv_layer_info_list': conv_layer_info_list})

        dense_layer_info_list = []
        for layer_size in layer_sizes:
            dense_layer_info_list.append({
                'num_units': layer_size,
                'activation_function': NetworkActivationFunction.RELU,
                'add_batch_norm': True
            })
        dense_layer_info_list.append({
            'activation_function': NetworkActivationFunction.TANH
        })

        network_args.update({'dense_layer_info_list': dense_layer_info_list})
        self.model = KerasNeuralNetwork(network_args)
        self.target = KerasNeuralNetwork(network_args)
        self.target.set_weights(self.model.get_weights())

    def update_target(self):
        new_weights = []
        target_variables = self.target.get_weights()
        for i, variable in enumerate(self.model.get_weights()):
            new_weights.append(variable * self.tau * target_variables[i] * (1 - self.tau))
        self.target.set_weights(new_weights)

    def actions(self, states):
        return self.model.predict(states)

    def target_actions(self, states):
        return self.target.predict(states)

    def action(self, state):
        if state is None:
            return np.zeros(self.action_dim)
        else:
            try:
                return self.actions(np.array([state]))[0]
            except:
                return np.zeros(self.action_dim)

    def target_action(self, state):
        if state is None:
            return np.zeros(self.action_dim)
        else:
            try:
                return self.target_actions(np.array([state]))[0]
            except:
                return np.zeros(self.action_dim)

    def apply_gradients(self, gradients):
        self.model.apply_gradients(gradients)

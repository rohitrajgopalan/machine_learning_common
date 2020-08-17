import math

import numpy as np
import tensorflow as tf
from keras import layers
from neural_network.neural_network import NeuralNetwork

from neural_network.network_types import NetworkOptimizer, NetworkActivationFunction

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
        # learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        # beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        # beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        # epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-07
        #
        # if optimizer_type == NetworkOptimizer.ADAMAX:
        #     self.optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_m, beta_v, epsilon)
        # elif optimizer_type == NetworkOptimizer.ADAM:
        #     self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_m, beta_v, epsilon)
        # elif optimizer_type == NetworkOptimizer.NADAM:
        #     self.optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_m, beta_v, epsilon)
        # else:
        #     self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        #
        # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # for i, layer_size in enumerate(layer_sizes):
        #     if layer_size == 'auto':
        #         layer_sizes[i] = int(math.sqrt(self.state_dim * self.action_dim))
        # inputs = layers.Input(shape=(self.state_dim,)) if type(self.state_dim) == int else layers.Input(
        #     shape=self.state_dim)
        # if convert_to_grayscale:
        #     inputs = tf.keras.layers.Lambda(lambda img: img / 255.0)(inputs)
        # out = inputs
        #
        # if len(conv_layer_info_list) > 1:
        #     for conv_layer_info in conv_layer_info_list:
        #         num_filters = conv_layer_info['num_filters']
        #         kernel_size = conv_layer_info['kernel_size']
        #         strides = conv_layer_info['strides'] if 'strides' in conv_layer_info else (1, 1)
        #         padding = conv_layer_info['padding'].lower() if 'padding' in conv_layer_info else 'same'
        #         activation_function = conv_layer_info[
        #             'activation_function'] if 'activation_function' in conv_layer_info else 'relu'
        #         if type(activation_function) == NetworkActivationFunction:
        #             activation_function = activation_function.name.lower()
        #         out = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides,
        #                                      padding=padding)(out)
        #         if add_pooling:
        #             pool_size = conv_layer_info['pool_size'] if 'pool_size' in conv_layer_info else (2, 2)
        #             out = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=conv_layer_info[
        #                 'strides'] if 'strides' in conv_layer_info else None)(out)
        #         out = tf.keras.layers.BatchNormalization()(out)
        #         out = tf.keras.layers.Activation(activation=activation_function)(out)
        #     out = tf.keras.layers.Flatten()(out)
        #
        # for layer_size in layer_sizes:
        #     out = layers.Dense(layer_size, activation='relu')(out)
        #     out = layers.BatchNormalization()(out)
        # outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)
        #
        # self.model = tf.keras.Model(inputs, outputs)
        # self.target = tf.keras.Model(inputs, outputs)
        self.model = NeuralNetwork(network_args)
        self.target = NeuralNetwork(network_args)
        self.target.set_weights(self.model.get_weights())

    def update_target(self):
        new_weights = []
        #target_variables = self.target.weights
        target_variables = self.target.get_weights()
        #for i, variable in enumerate(self.model.weights):
        for i, variable in enumerate(self.model.get_weights()):
            new_weights.append(variable * self.tau * target_variables[i] * (1 - self.tau))
        self.target.set_weights(new_weights)

    def actions(self, states):
        # return self.model(states)
        return self.model.predict(states)

    def target_actions(self, states):
        # return self.target(states)
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

    def generate_and_apply_gradients(self, actor_loss, tape=tf.GradientTape()):
        self.model.generate_and_apply_gradients(actor_loss, tape)
        # gradients = tape.gradient(actor_loss, self.model.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

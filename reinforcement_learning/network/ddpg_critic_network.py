import math

import numpy as np
import tensorflow as tf

from neural_network.network_types import NetworkOptimizer, NetworkActivationFunction

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01


class DDPGCriticNetwork:
    """
    Reference: https://github.com/maxkferg/DDPG/blob/master/ddpg/critic_network.py
    """

    def __init__(self, state_dim, state_layer_sizes, action_dim, action_layer_size, network_layer_sizes, optimizer_type,
                 optimizer_args={}, tau=TAU,
                 conv_layer_info_list=[], add_pooling=False, convert_to_grayscale=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau

        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-07

        if optimizer_type == NetworkOptimizer.ADAMAX:
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_m, beta_v, epsilon)
        elif optimizer_type == NetworkOptimizer.ADAM:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_m, beta_v, epsilon)
        elif optimizer_type == NetworkOptimizer.NADAM:
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_m, beta_v, epsilon)
        else:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)

        for i, layer_size in enumerate(network_layer_sizes):
            if layer_size == 'auto':
                network_layer_sizes[i] = int(math.sqrt(self.state_dim * self.action_dim))

        state_input = tf.keras.layers.Input(shape=(self.state_dim,)) if type(self.state_dim) == int else tf.keras.layers.Input(
            shape=self.state_dim)
        if convert_to_grayscale:
            state_input = tf.keras.layers.Lambda(lambda img: img/255.0)(state_input)
        state_out = state_input

        if len(conv_layer_info_list) > 1:
            for conv_layer_info in conv_layer_info_list:
                num_filters = conv_layer_info['num_filters']
                kernel_size = conv_layer_info['kernel_size']
                strides = conv_layer_info['strides'] if 'strides' in conv_layer_info else (1, 1)
                padding = conv_layer_info['padding'].lower() if 'padding' in conv_layer_info else 'same'
                activation_function = conv_layer_info['activation_function'] if 'activation_function' in conv_layer_info else 'relu'
                if type(activation_function) == NetworkActivationFunction:
                    activation_function = activation_function.name.lower()
                state_out = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding)(state_out)
                if add_pooling:
                    pool_size = conv_layer_info['pool_size'] if 'pool_size' in conv_layer_info else (2, 2)
                    state_out = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=conv_layer_info['strides'] if 'strides' in conv_layer_info else None)(state_out)
                state_out = tf.keras.layers.BatchNormalization()(state_out)
                state_out = tf.keras.layers.Activation(activation=activation_function)(state_out)
            state_out = tf.keras.layers.Flatten()(state_out)

        for layer_size in state_layer_sizes:
            state_out = tf.keras.layers.Dense(layer_size, activation='relu')(state_out)
            state_out = tf.keras.layers.BatchNormalization()(state_out)

        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        action_out = tf.keras.layers.Dense(action_layer_size, activation='relu')(action_input)
        action_out = tf.keras.layers.BatchNormalization()(action_out)

        concat = tf.keras.layers.Concatenate()([state_out, action_out])

        out = concat
        for layer_size in network_layer_sizes:
            out = tf.keras.layers.Dense(layer_size, activation='relu')(out)
            out = tf.keras.layers.BatchNormalization()(out)
        outputs = tf.keras.layers.Dense(1)(out)
        self.model = tf.keras.Model([state_input, action_input], outputs)
        self.target = tf.keras.Model([state_input, action_input], outputs)
        self.target.set_weights(self.model.get_weights())

    def update_target(self):
        new_weights = []
        target_variables = self.target.get_weights()
        for i, variable in enumerate(self.model.get_weights()):
            new_weights.append(variable * self.tau * target_variables[i] * (1 - self.tau))
        self.target.set_weights(new_weights)

    def q_values(self, states, actions):
        return self.model([states, actions])

    def target_qs(self, states, actions):
        return self.target([states, actions])

    def q_value(self, state, action):
        if state is None or action is None:
            return 0
        else:
            try:
                return self.q_values(np.array([state]), np.array([action]))[0]
            except:
                return 0

    def target_q(self, state, action):
        if state is None or action is None:
            return 0
        else:
            try:
                return self.target_qs(np.array([state]), np.array([action]))[0]
            except:
                return 0

    def generate_and_apply_gradients(self, critic_loss, tape=tf.GradientTape()):
        gradients = tape.gradient(critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

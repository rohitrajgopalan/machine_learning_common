from .network_layer import NetworkLayer
from .network_types import NetworkOptimizer

import numpy as np
import tensorflow as tf


class NeuralNetwork:
    model = None
    network_layers = None
    optimizer = None

    def __init__(self, num_inputs, num_outputs, optimizer_type, optimizer_args={},
                 network_layer_info_list=[]):
        self.network_layers = []
        self.optimizer_init(optimizer_type, optimizer_args)
        for idx, network_layer_info in enumerate(network_layer_info_list):
            if idx == 0:
                network_layer_info.update({'num_inputs': num_inputs})
            elif idx == len(network_layer_info_list) - 1:
                network_layer_info.update({'num_units': num_outputs})
            if 'num_units' not in network_layer_info or network_layer_info['num_units'].lower() == 'auto':
                network_layer_info.update({'num_units': int(np.sqrt(num_inputs * num_outputs))})
            network_layer = NetworkLayer(network_layer_info)
            self.network_layers.append(network_layer)
        self.build_model()

    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes, activation_function, kernel_initializer,
                 bias_initializer, optimizer_type, optimizer_args={}, use_bias=True):
        network_layer_info_list = []
        common_dict = {
            'activation_function': activation_function,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'use_bias': use_bias
        }
        for idx, hidden_layer_size in enumerate(hidden_layer_sizes):
            hidden_layer_info = common_dict
            hidden_layer_info.update({'num_units': hidden_layer_size})
            network_layer_info_list.append(hidden_layer_info)
        network_layer_info_list.append(common_dict)
        self.__init__(num_inputs, num_outputs, optimizer_type, optimizer_args, network_layer_info_list)

    def __init__(self, args={}):
        num_inputs = args['num_inputs']
        num_outputs = args['num_outputs']
        optimizer_type = args['optimizer_type']
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        if 'network_layer_info_list' in args:
            network_layer_info_list = args['network_layer_info_list']
            self.__init__(num_inputs, num_outputs, optimizer_type, optimizer_args, network_layer_info_list)
        else:
            hidden_layer_sizes = args['hidden_layer_sizes'] if 'hidden_layer_sizes' in args else []
            activation_function = args['activation_function'] if 'activation_function' in args else None
            kernel_initializer = args['kernel_initializer'] if 'kernel_initializer' in args else None
            bias_initializer = args['bias_initializer'] if 'bias_initializer' in args else None
            use_bias = args['use_bias'] if 'use_bias' in args else True
            self.__init__(num_inputs, num_outputs, hidden_layer_sizes, activation_function, kernel_initializer,
                          bias_initializer, optimizer_type, optimizer_args, use_bias)

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        for network_layer in self.network_layers:
            self.model.add(network_layer.dense_layer)
        self.model.compile(self.optimizer, loss='mse')

    def optimizer_init(self, optimizer_type, optimizer_args={}):
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        epsilon = optimizer_args['epsilon'] if 'epslion' in optimizer_args else 1e-07

        if optimizer_type == NetworkOptimizer.ADAMAX:
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_m, beta_v, epsilon)
        elif optimizer_type == NetworkOptimizer.NADAM:
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_m, beta_v, epsilon)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_m, beta_v, epsilon)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def update_network(self, inputs, outputs):
        self.model.fit(inputs, outputs, verbose=0)

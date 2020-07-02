import numpy as np

from .dense_network_layer import DenseNetworkLayer
from .neural_network import NeuralNetwork


class ObservationNeuralNetwork(NeuralNetwork):
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
            network_layer = DenseNetworkLayer(network_layer_info)
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

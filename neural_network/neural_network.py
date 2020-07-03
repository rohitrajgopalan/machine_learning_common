from supervised_learning.common import load_from_directory
from .network_layer import *
from .network_types import NetworkOptimizer
import numpy as np


class NeuralNetwork:
    model = None
    network_layers = None
    optimizer = None

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        for network_layer in self.network_layers:
            self.model.add(network_layer.layer)
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

    def load_data_from_directory(self, csv_dir, cols=[]):
        all_data = load_from_directory(csv_dir=csv_dir, cols=cols, concat=True)
        columns = all_data.columns
        features = columns[:len(cols) - 1]
        label = columns[-1]
        x = all_data[features]
        y = all_data[label]
        self.update_network(x, y)

    @staticmethod
    def choose_neural_network(args={}):
        if 'conv_layer_info_list' in args:
            return ImageFrameNeuralNetwork(args)
        else:
            return ObservationNeuralNetwork(args)


class ObservationNeuralNetwork(NeuralNetwork):
    def __init__(self, args={}):
        num_inputs = args['num_inputs']
        num_outputs = args['num_outputs']
        optimizer_type = args['optimizer_type']
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        if 'network_layer_info_list' in args:
            network_layer_info_list = args['network_layer_info_list']
            self.network_init(num_inputs, num_outputs, optimizer_type, optimizer_args, network_layer_info_list)
        else:
            hidden_layer_sizes = args['hidden_layer_sizes'] if 'hidden_layer_sizes' in args else []
            activation_function = args['activation_function'] if 'activation_function' in args else None
            kernel_initializer = args['kernel_initializer'] if 'kernel_initializer' in args else None
            bias_initializer = args['bias_initializer'] if 'bias_initializer' in args else None
            use_bias = args['use_bias'] if 'use_bias' in args else True
            self.network_init(num_inputs, num_outputs, hidden_layer_sizes, activation_function, kernel_initializer,
                              bias_initializer, optimizer_type, optimizer_args, use_bias)

    def network_init(self, num_inputs, num_outputs, optimizer_type, optimizer_args={},
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

    def network_init(self, num_inputs, num_outputs, hidden_layer_sizes, activation_function, kernel_initializer,
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


class ImageFrameNeuralNetwork(NeuralNetwork):

    def __init__(self, args={}):
        num_inputs = args['num_inputs']
        num_outputs = args['num_outputs']
        optimizer_type = args['optimizer_type']
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        conv_layer_info_list = args['conv_layer_info_list'] if 'conv_layer_info_list' else []
        network_layer_info_list = args['network_layer_info_list'] if 'network_layer_info_list' else []

        self.network_init(num_inputs, num_outputs, optimizer_type, optimizer_args, network_layer_info_list,
                          conv_layer_info_list)

    def network_init(self, num_inputs, num_outputs, optimizer_type, optimizer_args={},
                     network_layer_info_list=[], conv_layer_info_list=[]):
        self.network_layers = []
        self.optimizer_init(optimizer_type, optimizer_args)
        for idx, conv_layer_info in enumerate(conv_layer_info_list):
            if idx == 0:
                conv_layer_info.update({'num_inputs': num_inputs})
            conv_layer = ConvNetworkLayer(conv_layer_info)
            self.network_layers.append(conv_layer)
        self.network_layers.append(Flatten())
        for idx, network_layer_info in enumerate(network_layer_info_list):
            if idx == len(network_layer_info_list) - 1:
                network_layer_info.update({'num_units': num_outputs})
            if 'num_units' not in network_layer_info or network_layer_info['num_units'].lower() == 'auto':
                network_layer_info.update({'num_units': int(np.sqrt(num_inputs * num_outputs))})
            dense_layer = DenseNetworkLayer(network_layer_info)
            self.network_layers.append(dense_layer)
        self.build_model()

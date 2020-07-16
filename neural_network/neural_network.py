import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .network_layer import *
from .network_types import NetworkOptimizer


class NeuralNetwork:
    model = None
    network_layers = None
    optimizer = None
    loss_function = ''
    enable_scaling = False
    scaler = MinMaxScaler()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        for network_layer in self.network_layers:
            self.model.add(network_layer.layer)
        self.model.compile(self.optimizer, loss=self.loss_function)

    def optimizer_init(self, optimizer_type, optimizer_args={}):
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-07

        if optimizer_type == NetworkOptimizer.ADAMAX:
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_m, beta_v, epsilon)
        elif optimizer_type == NetworkOptimizer.NADAM:
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_m, beta_v, epsilon)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_m, beta_v, epsilon)

    def predict(self, inputs):
        if self.enable_scaling:
            inputs = self.scaler.transform(inputs)
        return self.model.predict(inputs)

    def fit(self, inputs, outputs):
        if self.enable_scaling:
            inputs = self.scaler.fit_transform(inputs)
        self.model.fit(inputs, outputs, verbose=0)

    def parse_dense_layer_info(self, num_inputs, num_outputs, dense_layer_info_list=[], set_input_shape=True):
        for idx, dense_layer_info in enumerate(dense_layer_info_list):
            if idx == len(dense_layer_info_list) - 1:
                num_units = num_outputs
            elif 'num_units' not in dense_layer_info or dense_layer_info['num_units'] == 'auto':
                num_units = int(np.sqrt(num_inputs * num_outputs))
            else:
                num_units = dense_layer_info['num_units']
            activation_function = dense_layer_info[
                'activation_function'] if 'activation_function' in dense_layer_info else None
            if type(activation_function) == str:
                activation_function = NetworkActivationFunction.get_type_by_name(activation_function)
            kernel_initializer = dense_layer_info[
                'kernel_initializer'] if 'kernel_initializer' in dense_layer_info else NetworkInitializationType.ZEROS
            if type(kernel_initializer) == str:
                kernel_initializer = NetworkInitializationType.get_type_by_name(kernel_initializer)
            bias_initializer = dense_layer_info[
                'bias_initializer'] if 'bias_initializer' in dense_layer_info else NetworkInitializationType.ZEROS
            if type(bias_initializer) == str:
                bias_initializer = NetworkInitializationType.get_type_by_name(bias_initializer)
            use_bias = dense_layer_info['use_bias'] if 'use_bias' in dense_layer_info else True
            input_shape = (num_inputs,) if idx == 0 and set_input_shape else None
            self.network_layers.append(
                DenseNetworkLayer(num_units, activation_function, kernel_initializer, bias_initializer, use_bias,
                                  input_shape))

    @staticmethod
    def choose_neural_network(args={}):
        if 'conv_layer_info_list' in args:
            return ImageFrameNeuralNetwork(args)
        else:
            return ObservationNeuralNetwork(args)


class ObservationNeuralNetwork(NeuralNetwork):
    def __init__(self, args={}):
        self.network_layers = []
        num_inputs = args['num_inputs']
        num_outputs = args['num_outputs']
        optimizer_type = args['optimizer_type']
        if type(optimizer_type) == str:
            optimizer_type = NetworkOptimizer.get_type_by_name(optimizer_type)
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        if 'loss_function' in args:
            self.loss_function = args['loss_function']
        self.optimizer_init(optimizer_type, optimizer_args)
        if 'dense_layer_info_list' in args:
            self.parse_dense_layer_info(num_inputs, num_outputs, args['dense_layer_info_list'])
        else:
            hidden_layer_sizes = args['hidden_layer_sizes']
            activation_function = args['activation_function']
            if type(activation_function) == str:
                activation_function = NetworkActivationFunction.get_type_by_name(activation_function)
            kernel_initializer = args['kernel_initializer']
            if type(kernel_initializer) == str:
                kernel_initializer = NetworkInitializationType.get_type_by_name(kernel_initializer)
            bias_initializer = args['bias_initializer']
            if type(bias_initializer) == str:
                bias_initializer = NetworkInitializationType.get_type_by_name(bias_initializer)
            use_bias = args['use_bias'] if 'use_bias' in args else True
            for idx, hidden_layer_size in enumerate(hidden_layer_sizes):
                if hidden_layer_size == 'auto':
                    num_units = int(np.sqrt(num_inputs * num_outputs))
                else:
                    num_units = hidden_layer_size
                input_shape = (num_inputs,) if idx == 0 else None
                self.network_layers.append(
                    DenseNetworkLayer(num_units, activation_function, kernel_initializer, bias_initializer, use_bias,
                                      input_shape))
            self.network_layers.append(
                DenseNetworkLayer(num_outputs, activation_function, kernel_initializer, bias_initializer, use_bias))
        self.enable_scaling = args['enable_scaling'] if 'enable_scaling' in args else False
        self.build_model()


class ImageFrameNeuralNetwork(NeuralNetwork):
    def __init__(self, args={}):
        self.network_layers = []
        num_inputs = args['num_inputs']
        num_outputs = args['num_outputs']
        optimizer_type = args['optimizer_type']
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        conv_layer_info_list = args['conv_layer_info_list'] if 'conv_layer_info_list' else []
        self.network_layers = []
        self.optimizer_init(optimizer_type, optimizer_args)
        if 'loss_function' in args:
            self.loss_function = args['loss_function']
        for idx, conv_layer_info in enumerate(conv_layer_info_list):
            input_shape = (num_inputs,) if idx == 0 else None
            num_dimensions = conv_layer_info['num_dimensions']
            num_filters = conv_layer_info['num_filters']
            kernel_size = conv_layer_info['kernel_size']
            strides = conv_layer_info['strides']
            is_transpose = conv_layer_info['is_transpose'] if 'is_transpose' in conv_layer_info else False
            activation_function = conv_layer_info[
                'activation_function'] if 'activation_function' in conv_layer_info else None
            if type(activation_function) == str:
                activation_function = NetworkActivationFunction.get_type_by_name(activation_function)
            kernel_initializer = conv_layer_info[
                'kernel_initializer'] if 'kernel_initializer' in conv_layer_info else NetworkInitializationType.ZEROS
            if type(kernel_initializer) == str:
                kernel_initializer = NetworkInitializationType.get_type_by_name(kernel_initializer)
            bias_initializer = conv_layer_info[
                'bias_initializer'] if 'bias_initializer' in conv_layer_info else NetworkInitializationType.ZEROS
            if type(bias_initializer) == str:
                bias_initializer = NetworkInitializationType.get_type_by_name(bias_initializer)
            use_bias = conv_layer_info['use_bias'] if 'use_bias' in args else True
            self.network_layers.append(
                ConvNetworkLayer(num_dimensions, num_filters, kernel_size, strides, is_transpose, activation_function,
                                 kernel_initializer, bias_initializer, use_bias, input_shape))
        self.network_layers.append(Flatten())
        self.parse_dense_layer_info(num_inputs, num_outputs,
                                    args['dense_layer_info_list'] if 'dense_layer_info_list' else [], False)
        self.build_model()

import numpy as np
from sklearn.preprocessing import RobustScaler, Normalizer

from .network_layer import *
from .network_types import NetworkOptimizer


class NeuralNetwork:
    model = None
    network_layers = None
    optimizer = None
    loss_function = ''
    enable_scaling = False
    enable_normalization = False
    scaler = RobustScaler()
    normalizer = Normalizer()
    use_gradients = False
    is_sequential = True
    input_shape = None
    num_inputs = 0
    num_outputs = 0
    convert_to_grayscale = False
    add_pooling = False

    def __init__(self, args):
        self.network_layers = []
        self.convert_to_grayscale = args['convert_to_grayscale'] if 'convert_to_grayscale' in args else False
        self.add_pooling = args['add_pooling'] if 'add_pooling' in args else False
        self.is_sequential = args['is_sequential'] if 'is_sequential' in args else True
        self.use_gradients = args['use_gradients'] if 'use_gradients' in args else False
        if 'num_inputs' in args:
            self.num_inputs = args['num_inputs']
            self.input_shape = (self.num_inputs,)
        elif 'input_shape' in args:
            self.input_shape = args['input_shape']
            self.num_inputs = 1
            for s in self.input_shape:
                self.num_inputs *= s
        self.num_outputs = args['num_outputs']
        if 'optimizer_type' in args:
            optimizer_type = args['optimizer_type']
            if type(optimizer_type) == str:
                optimizer_type = NetworkOptimizer.get_type_by_name(optimizer_type)
        else:
            optimizer_type = None
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        if optimizer_type is not None:
            self.optimizer_init(optimizer_type, optimizer_args)
        if 'loss_function' in args:
            self.loss_function = args['loss_function']
        self.enable_scaling = args['enable_scaling'] if 'enable_scaling' in args else False
        self.enable_normalization = args['enable_normalization'] if 'enable_normalization' in args else False
        if 'conv_layer_info_list' in args:
            self.parse_conv_layer_info(args['conv_layer_info_list'])
            self.network_layers.append(FlattenNetworkLayer())
        if 'dense_layer_info_list' in args:
            self.parse_dense_layer_info(args['dense_layer_info_list'])
        self.build_model()

    def build_model(self):
        for network_layer in self.network_layers:
            if network_layer.add_batch_norm:
                self.is_sequential = False
                break
        if self.is_sequential:
            self.model = tf.keras.models.Sequential()
            for network_layer in self.network_layers:
                self.model.add(network_layer.layer)
        else:
            inp = tf.keras.layers.Input(shape=self.input_shape)
            if self.convert_to_grayscale:
                inp = tf.keras.layers.Lambda(lambda img: img / 255.0)(inp)
            output = inp
            for network_layer in self.network_layers:
                output = network_layer.get_layer(inp=output)
            self.model = tf.keras.Model(inp, output)
        if not self.use_gradients:
            self.model.compile(self.optimizer, loss=self.loss_function)

    def optimizer_init(self, optimizer_type, optimizer_args):
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
        elif optimizer_type == NetworkOptimizer.RMSPROP:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    def predict(self, inputs):
        if self.convert_to_grayscale and self.is_sequential:
            inputs /= 255.0
        if self.enable_scaling:
            inputs = self.scaler.transform(inputs)
        if self.is_sequential:
            return self.model.predict(inputs)
        else:
            return self.model(inputs)

    def fit(self, inputs, outputs, sample_weight=None):
        if not self.is_sequential:
            pass
        if self.convert_to_grayscale:
            inputs /= 255.0
        if self.enable_scaling:
            inputs = self.scaler.fit_transform(inputs)
        if sample_weight is None:
            self.model.fit(inputs, outputs, epochs=1, verbose=0)
        else:
            self.model.fit(inputs, outputs, sample_weight=sample_weight, epochs=1, verbose=0)

    def generate_and_apply_gradients(self, loss, tape=tf.GradientTape()):
        if not self.use_gradients:
            pass
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def parse_dense_layer_info(self, dense_layer_info_list):
        for idx, dense_layer_info in enumerate(dense_layer_info_list):
            if idx == len(dense_layer_info_list) - 1:
                num_units = self.num_outputs
            elif 'num_units' not in dense_layer_info or dense_layer_info['num_units'] == 'auto':
                num_units = int(np.sqrt(self.num_inputs * self.num_outputs))
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
            add_batch_norm = dense_layer_info['add_batch_norm'] if 'add_batch_norm' in dense_layer_info else self.enable_normalization
            self.network_layers.append(
                DenseNetworkLayer(num_units, activation_function, kernel_initializer, bias_initializer, use_bias,
                                  self.input_shape if idx == 0 else None, add_batch_norm))

    def parse_conv_layer_info(self, conv_layer_info_list):
        for idx, conv_layer_info in enumerate(conv_layer_info_list):
            padding = conv_layer_info['padding'] if 'padding' in conv_layer_info else 'same'
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
            use_bias = conv_layer_info['use_bias'] if 'use_bias' in conv_layer_info else True
            if 'pool_size' in conv_layer_info:
                pool_size = conv_layer_info['pool_size']
            else:
                if num_dimensions > 1:
                    pool_size = [2] * num_dimensions
                    pool_size = tuple(pool_size)
                else:
                    pool_size = 2
            add_batch_norm = conv_layer_info['add_batch_norm'] if 'add_batch_norm' in conv_layer_info else self.enable_normalization
            self.network_layers.append(
                ConvNetworkLayer(num_dimensions, num_filters, kernel_size, strides, is_transpose, activation_function,
                                 kernel_initializer, bias_initializer, use_bias, padding,
                                 self.input_shape if idx == 0 else None, add_batch_norm, self.add_pooling, pool_size))
            if self.add_pooling and self.is_sequential:
                self.network_layers.append(MaxPoolLayer(num_dimensions, pool_size, strides, padding))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


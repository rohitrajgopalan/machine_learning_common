import tensorflow as tf
import torch as T
from .network_types import NetworkInitializationType, NetworkActivationFunction
import numpy as np


class NetworkLayer:
    layer = None

    def __init__(self, activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, input_shape=None,
                 add_batch_norm=False, use_keras=True):
        self.activation_function = activation_function
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.input_shape = input_shape
        self.add_batch_norm = add_batch_norm
        self.use_keras = use_keras
        if self.use_keras:
            self.build_keras()

    def build_keras(self):
        pass

    def get_keras(self, inp=None):
        pass

    def get_torch(self, **args):
        return {}

    def initialize_tensor(self, tensor, initialize_type, units=0):
        if initialize_type == NetworkInitializationType.ORTHOGONAL:
            T.nn.init.orthogonal_(tensor)
        elif initialize_type == NetworkInitializationType.ZEROS:
            T.nn.init.zeros_(tensor)
        elif initialize_type == NetworkInitializationType.ONES:
            T.nn.init.ones_(tensor)
        elif initialize_type == NetworkInitializationType.HE_NORMAL:
            T.nn.init.kaiming_normal_(tensor)
        elif initialize_type == NetworkInitializationType.HE_UNIFORM:
            T.nn.init.kaiming_uniform_(tensor)
        elif initialize_type == NetworkInitializationType.GLOROT_UNIFORM:
            T.nn.init.xavier_uniform_(tensor)
        elif initialize_type == NetworkInitializationType.GLOROT_NORMAL:
            T.nn.init.xavier_normal_(tensor)
        else:
            f = 1 / np.sqrt(units)
            T.nn.init.uniform_(tensor, -f, f)

    def get_torch_activation(self):
        if self.activation_function == NetworkActivationFunction.RELU:
            return T.nn.ReLU()
        elif self.activation_function == NetworkActivationFunction.LEAKY_RELU:
            return T.nn.LeakyReLU()
        elif self.activation_function == NetworkActivationFunction.SIGMOID:
            return T.nn.Sigmoid()
        elif self.activation_function == NetworkActivationFunction.ELU:
            return T.nn.ELU()
        elif self.activation_function == NetworkActivationFunction.SOFTMAX:
            return T.nn.Softmax()
        elif self.activation_function == NetworkActivationFunction.SOFTPLUS:
            return T.nn.Softplus()
        elif self.activation_function == NetworkActivationFunction.TANH:
            return T.nn.Tanh()
        else:
            return None


class ConvNetworkLayer(NetworkLayer):
    def __init__(self, num_dimensions, num_filters, kernel_size, strides, is_transpose=False,
                 activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, padding='same', input_shape=None,
                 add_batch_norm=False, add_pooling=False, pool_size=None, use_keras=True):
        self.num_dimensions = num_dimensions
        self.num_filters = num_filters
        # Can be a single integer or a tuple of 2 integers
        self.kernel_size = kernel_size
        # Can be a single integer or a tuple of 2 integers
        self.strides = strides
        self.is_transpose = is_transpose
        self.padding = padding.lower()
        self.add_pooling = add_pooling
        if pool_size is None:
            if self.num_dimensions > 1:
                pool_size = [2] * self.num_dimensions
                self.pool_size = tuple(pool_size)
            else:
                self.pool_size = 2
        else:
            self.pool_size = pool_size
        assert self.padding in ['same', 'valid']
        if activation_function is None:
            activation_function = NetworkActivationFunction.RELU
        super().__init__(activation_function, kernel_initializer, bias_initializer, use_bias, input_shape,
                         add_batch_norm, use_keras)

    def build_keras(self):
        if self.input_shape is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
        else:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding,
                                                                 input_shape=self.input_shape)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding,
                                                                 input_shape=self.input_shape)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding,
                                                        input_shape=self.input_shape)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding,
                                                        input_shape=self.input_shape)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding,
                                                        input_shape=self.input_shape)

    def get_keras(self, inp=None):
        if inp is None and self.input_shape is not None:
            inp = tf.keras.layers.Input(shape=self.input_shape)

        if inp is None:
            return None
        else:
            layer = None
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)(inp)
                elif self.num_dimensions == 3:
                    layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                            activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                            kernel_initializer=self.kernel_initializer.name.lower(),
                                                            bias_initializer=self.bias_initializer.name.lower(),
                                                            use_bias=self.use_bias, padding=self.padding)(inp)
            else:
                if self.num_dimensions == 1:
                    layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                   activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                   kernel_initializer=self.kernel_initializer.name.lower(),
                                                   bias_initializer=self.bias_initializer.name.lower(),
                                                   use_bias=self.use_bias, padding=self.padding)(inp)
                elif self.num_dimensions == 2:
                    layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                   activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                   kernel_initializer=self.kernel_initializer.name.lower(),
                                                   bias_initializer=self.bias_initializer.name.lower(),
                                                   use_bias=self.use_bias, padding=self.padding)(inp)
                elif self.num_dimensions == 3:
                    layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                   activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                                   kernel_initializer=self.kernel_initializer.name.lower(),
                                                   bias_initializer=self.bias_initializer.name.lower(),
                                                   use_bias=self.use_bias, padding=self.padding)(inp)
            if self.add_pooling:
                if self.num_dimensions == 1:
                    layer = tf.keras.layers.MaxPool1D(self.pool_size, self.strides, self.padding)(layer)
                elif self.num_dimensions == 2:
                    layer = tf.keras.layers.MaxPool2D(self.pool_size, self.strides, self.padding)(layer)
                elif self.num_dimensions == 3:
                    layer = tf.keras.layers.MaxPool3D(self.pool_size, self.strides, self.padding)(layer)
            if self.add_batch_norm:
                layer = tf.keras.layers.BatchNormalization()(layer)
            if self.activation_function is not None:
                layer = tf.keras.layers.Activation(activation=self.activation_function.name.lower())(layer)
            return layer

    def get_torch(self, **args):
        num_input_channels = args['num_input_channels'] if 'num_input_channels' in args else 1
        num_output_channels = self.num_filters
        result = {}
        if self.is_transpose:
            if self.num_dimensions == 1:
                self.layer = T.nn.ConvTranspose1d(num_input_channels, num_output_channels, self.kernel_size,
                                                  self.strides, bias=self.use_bias)
            elif self.num_dimensions == 2:
                self.layer = T.nn.ConvTranspose2d(num_input_channels, num_output_channels, self.kernel_size,
                                                  self.strides, bias=self.use_bias)
            elif self.num_dimensions == 3:
                self.layer = T.nn.ConvTranspose3d(num_input_channels, num_output_channels, self.kernel_size,
                                                  self.strides, bias=self.use_bias)
        else:
            if self.num_dimensions == 1:
                self.layer = T.nn.Conv1d(num_input_channels, num_output_channels, self.kernel_size, self.strides,
                                         bias=self.use_bias)
            elif self.num_dimensions == 2:
                self.layer = T.nn.Conv2d(num_input_channels, num_output_channels, self.kernel_size,
                                         self.strides, bias=self.use_bias)
            elif self.num_dimensions == 3:
                self.layer = T.nn.Conv3d(num_input_channels, num_output_channels, self.kernel_size,
                                         self.strides, bias=self.use_bias)
        result.update({'layer': self.layer})
        if self.add_pooling:
            pooling = None
            if self.num_dimensions == 1:
                pooling = T.nn.MaxPool1d(self.pool_size, self.strides)
            elif self.num_dimensions == 2:
                pooling = T.nn.MaxPool2d(self.pool_size, self.strides)
            elif self.num_dimensions == 3:
                pooling = T.nn.MaxPool3d(self.pool_size, self.strides)
            result.update({'pooling': pooling})
        if self.add_batch_norm:
            result.update({'bn': T.nn.BatchNorm2d(num_output_channels)})
        activation = self.get_torch_activation()
        if activation is not None:
            result.update({'activation': activation})
        return result


class MaxPoolLayer(NetworkLayer):
    def __init__(self, num_dimensions, pool_size, strides=None, padding='same', add_batch_norm=False, use_keras=True):
        self.num_dimensions = num_dimensions
        assert (type(pool_size) == int and num_dimensions == 1) or (
                type(pool_size) == tuple and 1 < num_dimensions == len(pool_size))
        self.pool_size = pool_size
        self.strides = strides
        if self.strides is not None:
            if self.num_dimensions == 1:
                assert (type(self.strides) == int)
            elif self.num_dimensions == 2:
                assert (type(self.strides) == int) or (type(self.strides) == tuple and len(self.strides) == 2)
            elif self.num_dimensions == 3:
                assert type(self.strides) == tuple and len(self.strides) == 3
        self.padding = padding.lower()
        assert self.padding in ['same', 'valid']
        super().__init__(add_batch_norm=add_batch_norm, use_keras=use_keras)

    def build_keras(self):
        if self.num_dimensions == 1:
            self.layer = tf.keras.layers.MaxPool1D(self.pool_size, self.strides, self.padding)
        elif self.num_dimensions == 2:
            self.layer = tf.keras.layers.MaxPool2D(self.pool_size, self.strides, self.padding)
        elif self.num_dimensions == 3:
            self.layer = tf.keras.layers.MaxPool3D(self.pool_size, self.strides, self.padding)

    def get_torch(self, **args):
        if self.num_dimensions == 1:
            self.layer = T.nn.MaxPool1d(self.pool_size, self.strides)
        elif self.num_dimensions == 2:
            self.layer = T.nn.MaxPool2d(self.pool_size, self.strides)
        elif self.num_dimensions == 3:
            self.layer = T.nn.MaxPool3d(self.pool_size, self.strides)

        return {'layer': self.layer}

    def get_keras(self, inp=None):
        if inp is None:
            return None
        else:
            layer = None
            if self.num_dimensions == 1:
                layer = tf.keras.layers.MaxPool1D(self.pool_size, self.strides, self.padding)(inp)
            elif self.num_dimensions == 2:
                layer = tf.keras.layers.MaxPool2D(self.pool_size, self.strides, self.padding)(inp)
            elif self.num_dimensions == 3:
                layer = tf.keras.layers.MaxPool3D(self.pool_size, self.strides, self.padding)(inp)
            if self.add_batch_norm:
                return tf.keras.layers.BatchNormalization()(layer)
            else:
                return layer


class DenseNetworkLayer(NetworkLayer):
    num_units = 0

    def __init__(self, num_units, activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, input_shape=None,
                 add_batch_norm=False, use_keras=True):
        self.num_units = num_units
        super().__init__(activation_function, kernel_initializer, bias_initializer, use_bias, input_shape,
                         add_batch_norm, use_keras)

    def build_keras(self):
        if self.input_shape is None:
            self.layer = tf.keras.layers.Dense(self.num_units,
                                               activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias)
        else:
            self.layer = tf.keras.layers.Dense(self.num_units,
                                               activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias,
                                               input_shape=self.input_shape)

    def get_keras(self, inp=None):
        if inp is None and self.input_shape is not None:
            inp = tf.keras.layers.Input(shape=self.input_shape)

        if inp is None:
            return None
        else:
            layer = tf.keras.layers.Dense(self.num_units,
                                          activation=None if self.activation_function is None else self.activation_function.name.lower(),
                                          kernel_initializer=self.kernel_initializer.name.lower(),
                                          bias_initializer=self.bias_initializer.name.lower(),
                                          use_bias=self.use_bias)(inp)
            if self.add_batch_norm:
                return tf.keras.layers.BatchNormalization()(layer)
            else:
                return layer

    def get_torch(self, **args):
        if self.input_shape is not None:
            num_inputs = self.input_shape[0]
        elif 'num_inputs' in args:
            num_inputs = args['num_inputs']
        else:
            num_inputs = None

        if num_inputs is None:
            return {}
        else:
            result = {}
            self.layer = T.nn.Linear(num_inputs, self.num_units, bias=self.use_bias)
            self.initialize_tensor(self.layer.weight.data, self.kernel_initializer, self.num_units)
            self.initialize_tensor(self.layer.bias.data, self.kernel_initializer, self.num_units)
            result.update({'layer': self.layer})
            if self.add_batch_norm:
                result.update({'bn': T.nn.BatchNorm1d(self.num_units)})
            activation = self.get_torch_activation()
            if activation is not None:
                result.update({'activation': activation})
            return result


class FlattenNetworkLayer(NetworkLayer):
    def __init__(self, use_keras=True):
        super().__init__(use_keras=use_keras)

    def build_keras(self):
        self.layer = tf.keras.layers.Flatten()

    def get_keras(self, inp=None):
        if inp is None:
            return None
        else:
            return tf.keras.layers.Flatten()(inp)

    def get_torch(self, **args):
        return {'layer': T.nn.Flatten()}

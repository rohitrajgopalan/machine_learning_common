import tensorflow as tf
import tensorflow_probability as tfp
from .network_types import NetworkInitializationType, NetworkActivationFunction


class NetworkLayer:
    layer = None

    def __init__(self, activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, input_shape=None):
        self.activation_function = activation_function
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.input_shape = input_shape
        self.build_layer()

    def build_layer(self):
        pass

    def add_unit(self):
        pass


class ConvNetworkLayer(NetworkLayer):
    def __init__(self, num_dimensions, num_filters, kernel_size, strides, is_transpose=False,
                 activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, padding='same', input_shape=None):
        self.num_dimensions = num_dimensions
        self.num_filters = num_filters
        # Can be a single integer or a tuple of 2 integers
        self.kernel_size = kernel_size
        # Can be a single integer or a tuple of 2 integers
        self.strides = strides
        self.is_transpose = is_transpose
        self.padding = padding.lower()
        assert self.padding in ['same', 'valid']
        super().__init__(activation_function, kernel_initializer, bias_initializer, use_bias, input_shape)

    def build_layer(self):
        if self.activation_function is None and self.input_shape is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
        elif self.activation_function is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape,
                                                                 padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape,
                                                                 padding=self.padding)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape,
                                                        padding=self.padding)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape,
                                                        padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape,
                                                        padding=self.padding)
        elif self.input_shape is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, padding=self.padding)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, padding=self.padding)
        else:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape,
                                                                 padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape,
                                                                 padding=self.padding)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape,
                                                        padding=self.padding)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape,
                                                        padding=self.padding)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape,
                                                        padding=self.padding)


class MaxPoolLayer(NetworkLayer):
    def __init__(self, num_dimensions, pool_size, strides=None, padding='same'):
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
        super().__init__()

    def build_layer(self):
        if self.num_dimensions == 1:
            self.layer = tf.keras.layers.MaxPool1D(self.pool_size, self.strides, self.padding)
        elif self.num_dimensions == 2:
            self.layer = tf.keras.layers.MaxPool2D(self.pool_size, self.strides, self.padding)
        elif self.num_dimensions == 3:
            self.layer = tf.keras.layers.MaxPool3D(self.pool_size, self.strides, self.padding)


class DenseNetworkLayer(NetworkLayer):
    num_units = 0

    def __init__(self, num_units, activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, input_shape=None):
        self.num_units = num_units
        super().__init__(activation_function, kernel_initializer, bias_initializer, use_bias, input_shape)

    def build_layer(self):
        if self.activation_function is None and self.input_shape is None:
            self.layer = tf.keras.layers.Dense(self.num_units,
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias)
        elif self.activation_function is None:
            self.layer = tf.keras.layers.Dense(self.num_units,
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias, input_shape=self.input_shape)
        elif self.input_shape is None:
            self.layer = tf.keras.layers.Dense(self.num_units, activation=self.activation_function.name.lower(),
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias)
        else:
            self.layer = tf.keras.layers.Dense(self.num_units, activation=self.activation_function.name.lower(),
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias, input_shape=self.input_shape)


class FlattenNetworkLayer(NetworkLayer):
    def build_layer(self):
        self.layer = tf.keras.layers.Flatten()


class NormalDistributionNetworkLayer(NetworkLayer):
    std_dev = None

    def __init__(self, std_dev):
        self.std_dev = std_dev
        super().__init__()

    def build_layer(self):
        self.layer = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t, scale=self.std_dev))

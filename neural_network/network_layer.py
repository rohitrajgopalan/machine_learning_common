import tensorflow as tf

from .network_types import NetworkInitializationType, NetworkActivationFunction


class KerasNetworkLayer:
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


class ConvKerasNetworkLayer(KerasNetworkLayer):
    def __init__(self, num_dimensions, num_filters, kernel_size, strides, is_transpose=False, activation_function: NetworkActivationFunction = None,
                 kernel_initializer: NetworkInitializationType = None,
                 bias_initializer: NetworkInitializationType = None, use_bias=True, input_shape=None):
        self.num_dimensions = num_dimensions
        self.num_filters = num_filters
        # Can be a single integer or a tuple of 2 integers
        self.kernel_size = kernel_size
        # Can be a single integer or a tuple of 2 integers
        self.strides = strides
        self.is_transpose = is_transpose
        super().__init__(activation_function, kernel_initializer, bias_initializer, use_bias, input_shape)

    def build_layer(self):
        if self.activation_function is None and self.input_shape is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias)
        elif self.activation_function is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape)
        elif self.input_shape is None:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias)
        else:
            if self.is_transpose:
                if self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3DTranspose(self.num_filters, self.kernel_size, self.strides,
                                                                 activation=self.activation_function.name.lower(),
                                                                 kernel_initializer=self.kernel_initializer.name.lower(),
                                                                 bias_initializer=self.bias_initializer.name.lower(),
                                                                 use_bias=self.use_bias, input_shape=self.input_shape)
            else:
                if self.num_dimensions == 1:
                    self.layer = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape)
                elif self.num_dimensions == 2:
                    self.layer = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape)
                elif self.num_dimensions == 3:
                    self.layer = tf.keras.layers.Conv3D(self.num_filters, self.kernel_size, self.strides,
                                                        activation=self.activation_function.name.lower(),
                                                        kernel_initializer=self.kernel_initializer.name.lower(),
                                                        bias_initializer=self.bias_initializer.name.lower(),
                                                        use_bias=self.use_bias, input_shape=self.input_shape)


class DenseKerasNetworkLayer(KerasNetworkLayer):
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

    def add_unit(self):
        self.num_units += 1
        self.build_layer()


class KerasFlatten(KerasNetworkLayer):

    def __init__(self):
        super().__init__()

    def build_layer(self):
        self.layer = tf.keras.layers.Flatten()

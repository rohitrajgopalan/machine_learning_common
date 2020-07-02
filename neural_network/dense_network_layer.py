from .network_layer import NetworkLayer
import tensorflow as tf


class DenseNetworkLayer(NetworkLayer):
    num_units = 0

    def __init__(self, num_units, activation_function=None, kernel_initializer=None, bias_initializer=None,
                 use_bias=True, num_inputs=None):
        super().__init__(activation_function, kernel_initializer, bias_initializer, use_bias, num_inputs)
        self.num_units = num_units
        self.build_layer()

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
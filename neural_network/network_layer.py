import tensorflow as tf
from .network_types import NetworkInitializationType, NetworkActivationFunction


class NetworkLayer:
    layer = None
    num_units = 0
    activation_function = None
    kernel_initializer = None
    bias_initializer = None
    use_bias = True
    num_inputs = None

    def __init__(self, args={}):
        for key in args:
            setattr(self, key, args[key])
        self.build_layer()

    def __init__(self, num_units, activation_function=None, kernel_initializer=None, bias_initializer=None,
                 use_bias=True, num_inputs=None):
        self.num_units = num_units
        self.activation_function = activation_function
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.num_inputs = num_inputs
        self.build_layer()

    def build_layer(self):
        if self.activation_function is None and self.num_inputs is None:
            self.layer = tf.keras.layers.Dense(self.num_units,
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias)
        elif self.activation_function is None:
            self.layer = tf.keras.layers.Dense(self.num_units,
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias, input_shape=(self.num_inputs,))
        elif self.num_inputs is None:
            self.layer = tf.keras.layers.Dense(self.num_units, activation=self.activation_function.name.lower(),
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias)
        else:
            self.layer = tf.keras.layers.Dense(self.num_units, activation=self.activation_function.name.lower(),
                                               kernel_initializer=self.kernel_initializer.name.lower(),
                                               bias_initializer=self.bias_initializer.name.lower(),
                                               use_bias=self.use_bias, input_shape=(self.num_inputs,))

    def add_unit(self):
        self.num_units += 1
        self.build_layer()

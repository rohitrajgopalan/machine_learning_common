class NetworkLayer:
    layer = None
    activation_function = None
    kernel_initializer = None
    bias_initializer = None
    use_bias = True
    input_shape = None

    def __init__(self, args={}):
        for key in args:
            setattr(self, key, args[key])
        self.build_layer()

    def __init__(self, activation_function=None, kernel_initializer=None, bias_initializer=None,
                 use_bias=True, input_shape=None):
        self.activation_function = activation_function
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.input_shape = input_shape

    def build_layer(self):
        pass

    def add_unit(self):
        pass

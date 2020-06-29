from neural_network.neural_network import NeuralNetwork
import numpy as np


class ActionValueNetwork(NeuralNetwork):
    model = None
    network_layers = None
    state_dim = 0
    num_actions = 0
    optimizer = None

    def __init__(self, num_inputs, num_outputs, optimizer_type, optimizer_args={},
                 network_layer_info_list=[]):
        super().__init__(num_inputs, num_outputs, optimizer_type, optimizer_args,
                         network_layer_info_list)
        self.state_dim = num_inputs
        self.num_actions = num_outputs

    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes, activation_function, kernel_initializer,
                 bias_initializer, optimizer_type, optimizer_args={}, use_bias=True):
        super().__init__(num_inputs, num_outputs, hidden_layer_sizes, activation_function, kernel_initializer,
                         bias_initializer, optimizer_type, optimizer_args, use_bias)
        self.state_dim = num_inputs
        self.num_actions = num_outputs

    def get_action_values(self, s):
        if s is None:
            return np.zeros((1, self.num_actions))
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        try:
            return super().predict(np.array([s])).reshape(1, self.num_actions)
        except ValueError as e:
            return np.zeros((1, self.num_actions))

    def add_action(self):
        self.num_actions += 1
        self.network_layers[len(self.network_layers) - 1].add_unit()
        self.build_model()

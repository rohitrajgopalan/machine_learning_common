import numpy as np

from neural_network.keras_neural_network import KerasNeuralNetwork


class ActionValueNetwork:
    num_actions = 0
    model_network = None
    target_network = None
    is_double = False

    def __init__(self, args={}, is_double=False):
        args.update({'loss_function': 'mse'})
        self.model_network = KerasNeuralNetwork(args)
        self.num_actions = args['num_outputs']
        self.is_double = is_double
        if self.is_double:
            self.target_network = KerasNeuralNetwork(args)

    def get_action_values(self, s):
        return self.get_action_values_with_network(s, self.model_network)

    def get_target_action_values(self, s):
        return self.get_action_values_with_network(s, self.target_network)

    def get_action_values_with_network(self, s, network):
        if s is None:
            return np.zeros((1, self.num_actions))

        s = np.array([s])
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        try:
            return network.predict(s)
        except ValueError:
            return np.zeros((1, self.num_actions))

    def update_network(self, inputs, outputs):
        self.model_network.fit(inputs, outputs)

    def update_target_weights(self):
        if self.is_double:
            self.target_network.model.set_weights(self.model_network.model.get_weights())

import numpy as np

from neural_network.neural_network import NeuralNetwork


class ActionValueNetwork:
    num_actions = 0
    neural_network = None

    def __init__(self, args={}):
        args.update({'loss_function': 'mse'})
        self.neural_network = NeuralNetwork.choose_neural_network(args)

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
            initial_prediction = self.neural_network.predict(np.array([s]))
            if initial_prediction.shape[1] == 0:
                return np.zeros((1, self.num_actions))
            else:
                return initial_prediction
        except ValueError as e:
            return np.zeros((1, self.num_actions))

    def add_action(self):
        self.num_actions += 1
        self.neural_network.network_layers[len(self.neural_network.network_layers) - 1].add_unit()
        self.neural_network.build_model()

    def update_network(self, inputs, outputs):
        self.neural_network.update_network(inputs, outputs)

    def get_weights(self):
        return self.neural_network.model.get_weights()

    def set_weights(self, weights):
        self.neural_network.model.set_weights(weights)

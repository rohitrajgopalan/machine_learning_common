import numpy as np

from neural_network.neural_network import NeuralNetwork


class ActionValueNetwork:
    num_actions = 0
    model_network = None
    target_network = None
    is_double = False

    def __init__(self, args={}, is_double=False):
        args.update({'loss_function': 'mse'})
        self.model_network = NeuralNetwork.choose_neural_network(args)
        self.num_actions = args['num_outputs']
        self.is_double = is_double
        if self.is_double:
            self.target_network = NeuralNetwork.choose_neural_network(args)

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
            initial_prediction = network.predict(s)
            if initial_prediction.shape[1] == 0:
                return np.zeros((1, self.num_actions))
            else:
                return initial_prediction
        except ValueError as e:
            return np.zeros((1, self.num_actions))

    def update_network(self, inputs, outputs):
        self.model_network.fit(inputs, outputs)

    def update_target_weights(self):
        if self.is_double:
            self.target_network.model.set_weights(self.model_network.model.get_weights())

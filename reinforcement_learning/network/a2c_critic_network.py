from neural_network.neural_network import NeuralNetwork
import numpy as np


class A2CCriticNetwork:
    network = None

    def __init__(self, args):
        args.update({'loss_function': 'mse'})
        self.network = NeuralNetwork.choose_neural_network(args)

    def get_values(self, states):
        return self.network.predict(states)

    def get_value(self, state):
        if state is None:
            return 0

        values = self.get_values(np.array([state]))
        try:
            return values[0, 0]
        except IndexError:
            return values[0]

    def update(self, states, discounted_rewards):
        self.network.fit(states, discounted_rewards)

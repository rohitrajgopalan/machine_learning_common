import numpy as np

from neural_network.neural_network import NeuralNetwork


class PGCriticNetwork:
    network = None
    state_dim = 0

    def __init__(self, args):
        self.state_dim = args['num_inputs']
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
        except ValueError:
            return 0

    def update(self, states, target_values):
        self.network.fit(states, target_values)

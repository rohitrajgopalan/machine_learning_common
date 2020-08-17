import numpy as np

from neural_network.neural_network import NeuralNetwork


class CACActorNetwork:
    network = None
    action_dim = 0
    std_dev = 0.0

    def __init__(self, args, normal_dist_std_dev):
        self.action_dim = args['num_outputs']
        assert type(normal_dist_std_dev) == float or (
                type(normal_dist_std_dev) == np.ndarray and normal_dist_std_dev.shape[0] == self.action_dim)
        args.update({'loss_function': 'categorical_crossentropy'})
        self.std_dev = normal_dist_std_dev
        self.network = NeuralNetwork(args)

    def actions(self, states):
        predictions = self.network.predict(states)
        return np.random.normal(loc=predictions, scale=self.std_dev)

    def action(self, state):
        if state is None:
            return np.zeros(self.action_dim)
        else:
            try:
                return self.actions(np.array([state]))[0]
            except:
                return np.zeros(self.action_dim)

    def update(self, states, actions, advantages):
        self.network.fit(states, actions, advantages)

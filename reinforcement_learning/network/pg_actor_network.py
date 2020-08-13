import numpy as np

from neural_network.neural_network import NeuralNetwork


class PGActorNetwork:
    network = None
    action_dim = 0
    std_dev = 0.0

    def __init__(self, args, normal_dist_std_dev):
        self.action_dim = args['num_outputs']
        assert type(normal_dist_std_dev) == float or (
                type(normal_dist_std_dev) == np.ndarray and normal_dist_std_dev.shape[0] == self.action_dim)
        args.update({'loss_function': 'categorical_crossentropy',
                     'normal_dist_std_dev': normal_dist_std_dev})
        self.network = NeuralNetwork.choose_neural_network(args)

    def actions(self, states):
        return self.network.predict(states)

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

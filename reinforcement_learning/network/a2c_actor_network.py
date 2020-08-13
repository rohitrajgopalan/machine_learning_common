from neural_network.neural_network import NeuralNetwork
import numpy as np


class A2CActorNetwork:
    network = None
    num_actions = 0

    def __init__(self, network_args):
        network_args.update({'loss_function': 'categorical_crossentropy'})
        self.network = NeuralNetwork.choose_neural_network(network_args)
        self.num_actions = network_args['num_outputs']

    def action(self, state):
        prediction = self.network.predict(np.array([state]))
        prediction = prediction.ravel()
        return np.random.choice(self.num_actions, p=prediction)

    def best_actions(self, state):
        prediction = self.network.predict(np.array([state]))
        prediction = prediction.ravel()
        max_value = np.max(prediction)
        actions_with_max = []
        for i in range(self.num_actions):
            if prediction[i] == max_value:
                actions_with_max.append(i)
        return actions_with_max

    def update(self, states, actions, advantages):
        self.network.fit(states, actions, advantages)

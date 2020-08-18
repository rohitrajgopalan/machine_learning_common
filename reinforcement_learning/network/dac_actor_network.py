import numpy as np
import tensorflow as tf
from neural_network.neural_network import NeuralNetwork


class DACActorNetwork:
    network = None
    num_actions = 0
    use_gradients = False

    def __init__(self, network_args, use_gradients=False):
        self.use_gradients = use_gradients
        network_args.update({'use_gradients': use_gradients})
        if not self.use_gradients:
            network_args.update({'loss_function': 'categorical_crossentropy'})
        self.network = NeuralNetwork(network_args)
        self.num_actions = network_args['num_outputs']

    def multiple_action_probabilities(self, states):
        return self.network.predict(states)

    def action_probabilites(self, state):
        return self.multiple_action_probabilities(np.array([state])).ravel()

    def action(self, state):
        prediction = self.action_probabilites(state)
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

    def update(self, states, actions, td_error):
        if not self.use_gradients:
            self.network.fit(states, actions, td_error)
        else:
            y_true = actions
            y_pred = self.multiple_action_probabilities(states)
            y_pred_clipped = np.clip(y_pred, 1e-08, 1 - 1e-08)
            log_like = y_true * np.log(y_pred_clipped)
            log_like = tf.convert_to_tensor(log_like)
            td_error = tf.convert_to_tensor(td_error)
            with tf.GradientTape() as tape:
                actor_loss = tf.reduce_sum(-log_like * td_error)
            self.network.generate_and_apply_gradients(actor_loss, tape)

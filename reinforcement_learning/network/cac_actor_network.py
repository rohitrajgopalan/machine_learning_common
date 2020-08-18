import numpy as np

from neural_network.neural_network import NeuralNetwork


class CACActorNetwork:
    network = None
    action_dim = 0
    std_dev = 0.0
    use_gradients = False

    def __init__(self, args, normal_dist_std_dev, use_gradients=False):
        self.action_dim = args['num_outputs']
        assert type(normal_dist_std_dev) == float or (
                type(normal_dist_std_dev) == np.ndarray and normal_dist_std_dev.shape[0] == self.action_dim)
        args.update({'use_gradients': use_gradients})
        if not use_gradients:
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

    def update(self, states, actions, td_error):
        if not self.use_gradients:
            self.network.fit(states, actions, td_error)
        else:
            y_true = actions
            y_pred = self.actions(states)
            y_pred_clipped = np.clip(y_pred, 1e-08, 1 - 1e-08)
            log_like = y_true * np.log(y_pred_clipped)
            actor_loss = np.sum(-log_like * td_error)
            self.network.generate_and_apply_gradients(actor_loss)

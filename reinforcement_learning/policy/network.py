import numpy as np
import torch as T
import tensorflow_probability as tfp
from neural_network.keras_neural_network import KerasNeuralNetwork
from neural_network.torch_neural_network import TorchNeuralNetwork
from .policy import Policy
import keras.backend as K


class NetworkPolicy(Policy):
    policy_network = None
    using_keras = True
    action_distribution = None
    is_continuous = False

    def __init__(self, args):
        super().__init__(args)
        super().__init__(args)
        self.action_dim = args['action_dim'] if 'action_dim' in args else 0
        self.is_continuous = self.action_dim > 0
        network_args = args['network_args'] if 'network_args' in args else {}
        self.using_keras = not args['use_td_error_in_function'] if 'use_td_error_in_function' in args else False
        network_args.update({'num_outputs': 2 if self.is_continuous else self.num_actions})
        if self.using_keras:
            network_args.update({'loss_function': 'categorical_crossentropy'})
            self.policy_network = KerasNeuralNetwork(network_args)
        else:
            self.policy_network = TorchNeuralNetwork(network_args)

    def derive(self, state, **args):
        if self.is_continuous:
            pass
        if state is None:
            action_probs = np.full(self.num_actions, 1 / self.num_actions)
        else:
            try:
                action_probs = self.policy_network.predict(np.array([state]))[0]
            except:
                action_probs = np.full(self.num_actions, 1 / self.num_actions)

        if not np.sum(action_probs) == 1 or np.isnan(action_probs.any()):
            action_probs = np.full(self.num_actions, 1 / self.num_actions)
        if not self.using_keras:
            self.action_distribution = T.distributions.Categorical(T.as_tensor(action_probs))
        return action_probs

    def choose_action(self, state, **args):
        if self.is_continuous:
            if state is None:
                actor_values = np.array([0, 1])
            else:
                try:
                    actor_values = self.policy_network.predict(np.array([state]))[0]
                except:
                    actor_values = np.array([0, 1])
            mu, sigma_unactivated = actor_values
            if self.using_keras:
                sigma = K.softplus(sigma_unactivated) + K.variable(1e-5)
                action_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
                action_as_tensor = action_dist.sample(sample_shape=[self.action_dim])
                action_as_tensor = K.tanh(action_as_tensor)
                a = action_as_tensor.numpy()
            else:
                sigma = T.nn.functional.softplus(T.as_tensor(sigma_unactivated, dtype=T.float)) + T.as_tensor(1e-5,
                                                                                                              dtype=T.float)
                self.action_distribution = T.distributions.Normal(T.as_tensor(mu, dtype=T.float), sigma)
                action_as_tensor = self.action_distribution.sample(sample_shape=T.Size([self.action_dim]))
                action_as_tensor = T.tanh(action_as_tensor)
                a = action_as_tensor.detach().numpy()
            return a, 0
        else:
            action_probs = self.derive(state)
            chosen_action = np.random.choice(self.num_actions, p=action_probs)
            picked_action_prob = action_probs[chosen_action]
            return chosen_action, picked_action_prob

    def update(self, action, should_action_be_blocked=False, **args):
        td_error = args['td_error'] if 'td_error' in args else 0
        state = args['state'] if 'state' in args else None
        assert state is not None
        states = np.array([state])
        if not type(td_error) == np.ndarray:
            td_error = np.array([td_error]).reshape((-1, 1))
        if self.is_continuous:
            actions = np.array([action]).reshape((-1, self.action_dim))
            self.optimize_network(states, td_error, actions=actions)
        else:
            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[action] = 1
            a_indices_one_hot = np.array([action_one_hot]).reshape((-1, self.num_actions))

            if self.using_keras:
                self.optimize_network(states, td_error, a_indices_one_hot=a_indices_one_hot)
            else:
                picked_action_prob = args['picked_action_prob'] if 'picked_action_prob' in args else 0
                picked_action_probs = np.array([picked_action_prob]).reshape((-1, 1))
                self.optimize_network(states, td_error, picked_action_probs=picked_action_probs)

    def optimize_network(self, states, td_error, **args):
        if self.using_keras:
            if self.is_continuous:
                actions = args['actions']
                self.policy_network.fit(states, actions, advantages=td_error)
            else:
                a_indices_one_hot = args['a_indices_one_hot']
                self.policy_network.fit(states, a_indices_one_hot, advantages=td_error)
        else:
            if self.is_continuous:
                actions = args['actions']
                for i, action in enumerate(actions):
                    loss = -T.sum(self.action_distribution.log_prob(T.as_tensor(action, dtype=T.float)),
                                  dtype=T.float) * td_error[i]
                    loss -= 1e-1 * self.action_distribution.entropy()
                    self.policy_network.update(loss)
            else:
                picked_action_probs = args['picked_action_probs']
                for i, picked_action_prob in enumerate(picked_action_probs):
                    loss = -self.action_distribution.log_prob(T.as_tensor(picked_action_prob, dtype=T.float)) * \
                           td_error[i]
                    self.policy_network.update(loss)

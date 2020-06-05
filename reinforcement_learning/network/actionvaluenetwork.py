import enum
from copy import deepcopy

import numpy as np


class NetworkType(enum.Enum):
    SINGLE = 1
    DOUBLE = 2


class ActionValueNetwork:
    weights = []

    def __init__(self, network_type, state_dim, num_hidden_units, num_actions, random_seed):
        self.network_type = network_type
        self.num_dim_state = state_dim
        self.num_hidden_units = num_hidden_units
        self.num_actions = num_actions
        self.rand_generator = np.random.RandomState(random_seed)
        self.layer_sizes = np.array([self.num_dim_state, self.num_hidden_units, self.num_actions])
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = []
        for _ in range(self.network_type.value):
            self.weights.append([{'W': self.init_saxe(self.num_dim_state, self.num_hidden_units), 'b': np.zeros((1, self.num_hidden_units))}, {'W': self.init_saxe(self.num_hidden_units, self.num_actions), 'b': np.zeros((1, self.num_actions))}])

    def add_action(self):
        self.num_actions += 1
        self.layer_sizes[2] += 1
        self.initialize_weights()

    # Developing this function to decide which index of weights to use
    def determine_coin_side(self):
        return self.rand_generator.choice(self.network_type.value)

    def get_action_values(self, s, coin_side=None):
        if s is None:
            return np.zeros((1, self.num_actions))
        if coin_side is None:
            q_vals = np.zeros((self.network_type.value, self.num_actions))
            for i in range(self.network_type.value):
                q_vals[i] = self.get_action_values_with_weights(s, self.weights[i])
            return np.mean(q_vals, axis=0).reshape(1, self.num_actions)
        else:
            return self.get_action_values_with_weights(s, self.weights[coin_side])

    def get_action_values_with_weights(self, s, weights):
        if s is None:
            return np.zeros((1, self.num_actions))
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        s = np.array([s])
        w0, b0 = weights[0]['W'], weights[0]['b']
        psi = np.dot(s, w0) + b0
        x = np.maximum(psi, 0)

        w1, b1 = weights[1]['W'], weights[1]['b']

        action_values = np.dot(x, w1) + b1

        return action_values.reshape(1, self.num_actions)

    def get_target_update(self, s, delta_mat, coin_side):
        return self.get_target_update_with_weights(s, delta_mat, self.weights[coin_side])

    def get_target_update_with_weights(self, states, delta_mat, weights):
        """
        Args:
            states  (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions). Each row of delta_mat  
            correspond to one state in the batch. Each row has only one non-zero element 
            which is the TD-error corresponding to the action taken.
            weights (Dictionary): Weights to get the target update with.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """

        num_states = states.shape[0]
        w0, b0 = weights[0]['W'], weights[0]['b']
        w1, b1 = weights[1]['W'], weights[1]['b']

        psi = np.dot(states, w0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        # td_update has the same structure as self.weights, that is an array of dictionaries.
        # td_update[0]["W"], td_update[0]["b"], td_update[1]["W"], and td_update[1]["b"] have the same shape as 
        # self.weights[0]["W"], self.weights[0]["b"], self.weights[1]["W"], and self.weights[1]["b"] respectively
        td_update = [dict() for _ in range(len(weights))]

        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / num_states
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / num_states

        v = np.dot(v, w1.T) * dx
        td_update[0]['W'] = np.dot(states.T, v) * 1. / num_states
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / num_states

        return td_update

    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    def get_weights(self):
        """
        Returns: 
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)

    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)

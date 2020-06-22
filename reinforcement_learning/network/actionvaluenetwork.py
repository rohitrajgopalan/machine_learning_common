import enum
import math
from copy import deepcopy

import numpy as np


class NetworkType(enum.Enum):
    SINGLE = 1
    DOUBLE = 2

    @staticmethod
    def all():
        return [NetworkType.SINGLE, NetworkType.DOUBLE]


class NetworkInitializationType(enum.Enum):
    XAVIER = 1,
    HE = 2,
    SAXE = 3,

    @staticmethod
    def all():
        return [NetworkInitializationType.XAVIER, NetworkInitializationType.HE, NetworkInitializationType.SAXE]


class NetworkActivationFunction(enum.Enum):
    ARCTAN = 1,
    ELU = 2,
    IDENTITY = 3,
    PRELU = 4,
    RELU = 5,
    SIGMOID = 6,
    SOFTPLUS = 7,
    TANH = 8

    @staticmethod
    def all():
        return [NetworkActivationFunction.ARCTAN, NetworkActivationFunction.ELU, NetworkActivationFunction.IDENTITY,
                NetworkActivationFunction.PRELU, NetworkActivationFunction.RELU, NetworkActivationFunction.SIGMOID,
                NetworkActivationFunction.SOFTPLUS, NetworkActivationFunction.TANH]


class ActionValueNetwork:
    weights = []

    # A flag to indicate whether hidden units was specified in the constructor or not
    did_set_hidden_units = True

    def __init__(self, network_type, initialization_type, activation_function, alpha, state_dim, num_hidden_units,
                 num_actions, random_seed):
        self.network_type = network_type
        self.initialization_type = initialization_type
        self.activation_function = activation_function
        self.alpha = alpha
        self.num_dim_state = state_dim
        self.num_hidden_units = num_hidden_units
        self.num_actions = num_actions
        self.rand_generator = np.random.RandomState(random_seed)
        self.layer_sizes = np.array([self.num_dim_state, self.num_hidden_units, self.num_actions])
        if self.num_hidden_units == int(math.sqrt(state_dim * num_actions)):
            self.did_set_hidden_units = False
        self.initialize_weights()

    """
    There is a fundamental approach to automatically determine number of hidden units in a NN
    See: Masters, Timothy. Practical neural network recipes in C++. Morgan Kaufmann, 1993.
    and http://www.iitbhu.ac.in/faculty/min/rajesh-rai/NMEICT-Slope/lecture/c14/l1.html
    """

    def __init__(self, network_type, initialization_type, activation_function, alpha, state_dim, num_actions,
                 random_seed):
        self.__init__(network_type, initialization_type, activation_function, alpha, state_dim,
                      int(math.sqrt(state_dim * num_actions)), num_actions, random_seed)

    def initialize_weights(self):
        self.weights = []
        for _ in range(self.network_type.value):
            self.weights.append([{'W': self.init_array(self.num_dim_state, self.num_hidden_units),
                                  'b': np.zeros((1, self.num_hidden_units))},
                                 {'W': self.init_array(self.num_hidden_units, self.num_actions),
                                  'b': np.zeros((1, self.num_actions))}])

    def add_action(self):
        self.num_actions += 1
        self.layer_sizes[2] += 1
        if not self.did_set_hidden_units:
            self.layer_sizes[1] = int(math.sqrt(self.num_dim_state * self.num_actions))
            self.num_hidden_units = int(math.sqrt(self.num_dim_state * self.num_actions))
        self.initialize_weights()

    # Developing this function to decide which index of weights to use
    # This is more useful for double networks
    def determine_coin_side(self):
        return self.rand_generator.choice(self.network_type.value)

    # Perform the appropriate activation based on the assigned Activation Function
    def perform_activation(self, psi):
        if self.activation_function == NetworkActivationFunction.ARCTAN:
            return np.arctan(psi)
        elif self.activation_function == NetworkActivationFunction.ELU:
            return np.where(psi < 0, self.alpha * (np.exp(psi) - 1), psi)
        elif self.activation_function == NetworkActivationFunction.PRELU:
            return np.where(psi < 0, self.alpha * psi, psi)
        elif self.activation_function == NetworkActivationFunction.RELU:
            return np.maximum(psi, 0)
        elif self.activation_function == NetworkActivationFunction.SIGMOID:
            return 1 / (1 + np.exp(psi * -1))
        elif self.activation_function == NetworkActivationFunction.SOFTPLUS:
            return np.log(1 + np.exp(psi))
        elif self.activation_function == NetworkActivationFunction.TANH:
            return np.tanh(psi)
        else:
            return psi

    # Based on the assigned activation function, perform the derative.
    def perform_derative(self, psi):
        if self.activation_function == NetworkActivationFunction.ARCTAN:
            return 1 / ((psi * psi) + 1)
        elif self.activation_function == NetworkActivationFunction.ELU:
            return np.where(psi < 0, self.alpha * np.exp(psi), psi)
        elif self.activation_function == NetworkActivationFunction.IDENTITY:
            return np.ones(psi.shape())
        elif self.activation_function == NetworkActivationFunction.PRELU:
            return np.where(psi > 0, 1, self.alpha)
        elif self.activation_function == NetworkActivationFunction.RELU:
            return np.where(psi > 0, 1, 0)
        elif self.activation_function == NetworkActivationFunction.SIGMOID:
            x = 1 / (1 + np.exp(psi * -1))
            return x * (1 - x)
        elif self.activation_function == NetworkActivationFunction.SOFTPLUS:
            return 1 / (1 + np.exp(psi * -1))
        elif self.activation_function == NetworkActivationFunction.TANH:
            x = np.tanh(psi)
            return 1 - (x * x)
        return psi

    # If we specify a coin side (useful in double networks), get the action values based on the weights from that coin side
    # Otherwise, derive the average of the action values (in double networks, divide by 2)
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
        x = self.perform_activation(psi)

        w1, b1 = weights[1]['W'], weights[1]['b']

        action_values = np.dot(x, w1) + b1

        return action_values.reshape(1, self.num_actions)

    # Get the target update using the weights of a particular coin side.
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
        x = self.perform_activation(psi)
        dx = self.perform_derative(psi)

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

    # Based on the specified initializer, initialise the weight array
    def init_array(self, rows, cols):
        if self.initialization_type == NetworkInitializationType.XAVIER:
            return self.init_xavier(rows, cols)
        elif self.initialization_type == NetworkInitializationType.HE:
            return self.init_he(rows, cols)
        elif self.initialization_type == NetworkInitializationType.SAXE:
            return self.init_saxe(rows, cols)
        else:
            return np.zeros((rows, cols))

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

    def init_scalar(self, rows, cols, scalar):
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        return tensor * np.sqrt(scalar / rows)

    def init_xavier(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on Xavier initialization.
        """
        return self.init_scalar(rows, cols, 1)

    def init_he(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in He et al.
        """
        return self.init_scalar(rows, cols, 2)

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

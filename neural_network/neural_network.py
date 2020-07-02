from .network_layer import NetworkLayer
from .network_types import NetworkOptimizer
from .image_frame_neural_network import ImageFrameNeuralNetwork
from .observation_neural_network import ObservationNeuralNetwork

import numpy as np
import tensorflow as tf

from supervised_learning.common import load_from_directory


class NeuralNetwork:
    model = None
    network_layers = None
    optimizer = None

    def __init__(self, args={}):
        pass

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        for network_layer in self.network_layers:
            self.model.add(network_layer.layer)
        self.model.compile(self.optimizer, loss='mse')

    def optimizer_init(self, optimizer_type, optimizer_args={}):
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        epsilon = optimizer_args['epsilon'] if 'epslion' in optimizer_args else 1e-07

        if optimizer_type == NetworkOptimizer.ADAMAX:
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_m, beta_v, epsilon)
        elif optimizer_type == NetworkOptimizer.NADAM:
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_m, beta_v, epsilon)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_m, beta_v, epsilon)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def update_network(self, inputs, outputs):
        self.model.fit(inputs, outputs, verbose=0)

    def load_data_from_directory(self, csv_dir, cols=[]):
        all_data = load_from_directory(csv_dir=csv_dir, cols=cols, concat=True)
        columns = all_data.columns
        features = columns[:len(cols) - 1]
        label = columns[-1]
        x = all_data[features]
        y = all_data[label]
        self.update_network(x, y)

    @staticmethod
    def choose_neural_network(args={}):
        if 'conv_layer_info_list' in args:
            return ImageFrameNeuralNetwork(args)
        else:
            return ObservationNeuralNetwork(args)

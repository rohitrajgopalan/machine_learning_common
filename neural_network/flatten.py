from .network_layer import NetworkLayer
import tensorflow as tf


class Flatten(NetworkLayer):

    def __init__(self):
        super().__init__()

    def build_layer(self):
        self.layer = tf.keras.layers.Flatten()

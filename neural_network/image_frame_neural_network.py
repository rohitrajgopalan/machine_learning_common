import numpy as np

from .flatten import Flatten
from .neural_network import NeuralNetwork
from .conv_network_layer import ConvNetworkLayer
from .dense_network_layer import DenseNetworkLayer


class ImageFrameNeuralNetwork(NeuralNetwork):

    def __init__(self, num_inputs, num_outputs, optimizer_type, optimizer_args={},
                 network_layer_info_list=[], conv_layer_info_list=[]):
        self.network_layers = []
        self.optimizer_init(optimizer_type, optimizer_args)
        for idx, conv_layer_info in enumerate(conv_layer_info_list):
            if idx == 0:
                conv_layer_info.update({'num_inputs': num_inputs})
            conv_layer = ConvNetworkLayer(conv_layer_info)
            self.network_layers.append(conv_layer)
        self.network_layers.append(Flatten())
        for idx, network_layer_info in enumerate(network_layer_info_list):
            if idx == len(network_layer_info_list) - 1:
                network_layer_info.update({'num_units': num_outputs})
            if 'num_units' not in network_layer_info or network_layer_info['num_units'].lower() == 'auto':
                network_layer_info.update({'num_units': int(np.sqrt(num_inputs * num_outputs))})
            dense_layer = DenseNetworkLayer(network_layer_info)
            self.network_layers.append(dense_layer)
        self.build_model()

    def __init__(self, args={}):
        num_inputs = args['num_inputs']
        num_outputs = args['num_outputs']
        optimizer_type = args['optimizer_type']
        optimizer_args = args['optimizer_args'] if 'optimizer_args' in args else {}
        conv_layer_info_list = args['conv_layer_info_list'] if 'conv_layer_info_list' else []
        network_layer_info_list = args['network_layer_info_list'] if 'network_layer_info_list' else []

        self.__init__(num_inputs, num_outputs, optimizer_type, optimizer_args, network_layer_info_list, conv_layer_info_list)
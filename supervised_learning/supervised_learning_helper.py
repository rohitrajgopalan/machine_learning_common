import pandas as pd
import numpy as np
from neural_network.neural_network import NeuralNetwork

from .common import select_regressor, select_classifier, MethodType


class SupervisedLearningHelper:
    method_type = None
    state_dim = 0
    csv_dir = ''
    filters = {}
    features = []
    label = ''
    historical_data = None

    def __init__(self, method_type, csv_dir, features, label, filters={}):
        self.method_type = method_type
        self.csv_dir = csv_dir
        self.features = features
        self.label = label
        self.filters = filters

    def add(self, feature_values_dict, target_value):
        new_data = {self.label: target_value}
        for feature in feature_values_dict:
            if feature in self.features:
                new_data.update({feature: feature_values_dict[feature]})
        self.historical_data = self.historical_data.append(new_data, ignore_index=True)
        x = self.historical_data[self.features]
        y = self.historical_data[self.label]
        self.fit(x, y)

    def fit(self, x, y):
        pass

    def predict(self, inputs):
        predictions = self.get_predictions(inputs)
        try:
            return predictions[0,0]
        except ValueError:
            return predictions[0]

    def get_predictions(self, inputs):
        return None

    @staticmethod
    def choose_helper(method_type, csv_dir, features, label, filters={}, dl_args=None):
        if dl_args is None:
            return ScikitLearnHelper(method_type, csv_dir, features, label, filters)
        else:
            return DeepLearningHelper(method_type, csv_dir, features, label, filters, dl_args)


class ScikitLearnHelper(SupervisedLearningHelper):
    method = None
    
    def __init__(self, choosing_method, method_type, csv_dir, features, label, filters={}):
        super().__init__(method_type, csv_dir, features, label)
        if self.method_type == MethodType.Regression:
            self.method, self.historical_data = select_regressor(self.csv_dir, choosing_method, self.features, self.label, self.filters)
        else:
            self.method, self.historical_data = select_classifier(self.csv_dir, choosing_method, self.features, self.label, self.filters)
    
    def fit(self, x, y):
        self.method.fit(x, y)

    def get_predictions(self, inputs):
        return self.method.predict(inputs)


class DeepLearningHelper(SupervisedLearningHelper):
    model = None

    def __init__(self, method_type, csv_dir, features, label, filters={}, dl_args={}):
        dl_args.update({'num_inputs': len(features), 'num_outputs': 1})
        self.model = NeuralNetwork.choose_neural_network(dl_args)
        columns = features
        columns.append(label)
        self.historical_data = self.model.load_data_from_directory(csv_dir, columns)
        super().__init__(method_type, csv_dir, features, label, filters)

    def fit(self, x, y):
        self.model.update_network(x, y)

    def get_predictions(self, inputs):
        return self.model.predict(inputs)

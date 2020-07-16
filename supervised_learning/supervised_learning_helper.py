from sklearn.preprocessing import MinMaxScaler

from neural_network.neural_network import NeuralNetwork

from .common import select_method, load_from_directory
import numpy as np
import pandas as pd


class SupervisedLearningHelper:
    method_type = None
    state_dim = 0
    csv_dir = ''
    filters = {}
    features = []
    label = ''
    historical_data = None
    model = None

    def __init__(self, method_type, csv_dir, features, label, filters={}):
        self.method_type = method_type
        self.csv_dir = csv_dir
        self.features = features
        self.label = label
        self.filters = filters
        cols = [feature for feature in self.features]
        cols.append(self.label)
        self.historical_data = load_from_directory(csv_dir, cols, filters, concat=True)
        if self.historical_data is None:
            self.historical_data = pd.DataFrame(columns=cols)
        else:
            assert self.label not in self.features
            x = self.historical_data[self.features]
            y = self.historical_data[self.label]
            self.fit(x, y)

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
        self.model.fit(x, y)

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        try:
            return predictions[0, 0]
        except IndexError:
            return predictions[0]

    @staticmethod
    def choose_helper(method_type, csv_dir, features, label, filters={}, dl_args=None, choosing_method='best'):
        if dl_args is None:
            return ScikitLearnHelper(choosing_method, method_type, csv_dir, features, label, filters)
        else:
            return DeepLearningHelper(method_type, csv_dir, features, label, filters, dl_args)


class ScikitLearnHelper(SupervisedLearningHelper):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    def __init__(self, choosing_method, method_type, csv_dir, features, label, filters={}):
        self.model = select_method(csv_dir, choosing_method, features, label, filters,
                                   method_type)
        super().__init__(method_type, csv_dir, features, label)

    def fit(self, x, y):
        self.scaler_x.fit(x)
        x = self.scaler_x.transform(x)
        y = np.array([y]).reshape(-1, 1)
        self.scaler_y.fit(y)
        y = self.scaler_y.transform(y)
        super().fit(x, y)

    def predict(self, inputs):
        self.scaler_x.fit(inputs)
        inputs = self.scaler_x.transform(inputs)
        return super().predict(inputs)


class DeepLearningHelper(SupervisedLearningHelper):

    def __init__(self, method_type, csv_dir, features, label, filters={}, dl_args={}):
        dl_args.update({'num_inputs': len(features), 'num_outputs': 1})
        self.model = NeuralNetwork.choose_neural_network(dl_args)
        super().__init__(method_type, csv_dir, features, label, filters)

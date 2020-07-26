import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer

from neural_network.neural_network import NeuralNetwork
from .common import select_method, load_from_directory


class SupervisedLearningHelper:
    method_type = None
    state_dim = 0
    files_dir = ''
    filters = {}
    features = []
    label = ''
    historical_data = None
    model = None
    enable_scaling = False
    enable_normalization = False

    def __init__(self, method_type, files_dir, features, label, filters={}, enable_scaling=False, sheet_name='', enable_normalization=True):
        self.method_type = method_type
        self.files_dir = files_dir
        self.features = features
        self.label = label
        self.filters = filters
        self.enable_scaling = enable_scaling
        self.enable_normalization = enable_normalization
        cols = [feature for feature in self.features]
        cols.append(self.label)
        self.historical_data = load_from_directory(files_dir, cols, filters, concat=True, sheet_name=sheet_name)
        if self.historical_data is None:
            self.historical_data = pd.DataFrame(columns=cols)
        else:
            if len(cols) == 1:
                cols = self.historical_data.columns
                self.features = cols[:len(cols) - 1]
                self.label = cols[-1]
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
        return self.model.predict(inputs)

    @staticmethod
    def choose_helper(method_type, files_dir, features, label, filters={}, enable_scaling=False, dl_args=None,
                      choosing_method='best', sheet_name='', enable_normalization=True):
        if dl_args is None:
            return ScikitLearnHelper(choosing_method, method_type, files_dir, features, label, filters, enable_scaling,
                                     sheet_name, enable_normalization)
        else:
            return DeepLearningHelper(method_type, files_dir, features, label, filters, enable_scaling, sheet_name,
                                      dl_args, enable_normalization)


class ScikitLearnHelper(SupervisedLearningHelper):
    # scaler = MinMaxScaler()
    scaler = RobustScaler()
    normalizer = Normalizer()

    def __init__(self, choosing_method, method_type, files_dir, features, label, filters={}, enable_scaling=False,
                 sheet_name='', enable_normalization=True):
        self.model = select_method(files_dir, choosing_method, features, label, filters,
                                   method_type, enable_scaling, sheet_name, enable_normalization)
        super().__init__(method_type, files_dir, features, label, filters, enable_scaling, sheet_name, enable_normalization)

    def fit(self, x, y):
        if self.enable_scaling:
            x = self.scaler.fit_transform(x)
        if self.enable_normalization:
            x = self.normalizer.fit_transform(x)
        super().fit(x, y)

    def predict(self, inputs):
        if self.enable_scaling:
            inputs = self.scaler.transform(inputs)
        if self.enable_normalization:
            inputs = self.normalizer.transform(inputs)
        return super().predict(inputs)


class DeepLearningHelper(SupervisedLearningHelper):

    def __init__(self, method_type, files_dir, features, label, filters={}, enable_scaling=False, sheet_name='',
                 dl_args={}, enable_normalization=True):
        dl_args.update({'num_inputs': len(features), 'num_outputs': 1, 'enable_scaling': enable_scaling, 'enable_normalization': enable_normalization})
        self.model = NeuralNetwork.choose_neural_network(dl_args)
        super().__init__(method_type, files_dir, features, label, filters, enable_scaling, sheet_name, enable_normalization)

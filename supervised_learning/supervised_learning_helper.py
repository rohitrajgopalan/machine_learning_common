import pandas as pd
from sklearn.preprocessing import RobustScaler, Normalizer

from neural_network.neural_network import NeuralNetwork
from .common import select_method, load_from_directory


class SupervisedLearningHelper:
    method_type = None
    state_dim = 0
    files_dir = ''
    features = []
    label = ''
    historical_data = None
    model = None
    enable_scaling = False
    enable_normalization = False

    def __init__(self, method_type, enable_scaling=False, enable_normalization=False, **args):
        self.method_type = method_type
        self.enable_scaling = enable_scaling
        self.enable_normalization = enable_normalization

        cols = []

        if 'data' in args:
            self.historical_data = args['data']
        else:
            self.features = args['features']
            self.label = args['label']
            cols = [feature for feature in self.features]
            cols.append(self.label)
            self.historical_data = load_from_directory(args['files_dir'], cols, concat=True,
                                                       sheet_name=args['sheet_name'] if 'sheet_name' in args else '',
                                                       header_index=args['header_index'] if 'header_index' in args else 0)
        if type(self.historical_data) == list:
            self.historical_data = pd.concat(self.historical_data, ignore_index=True)

        if len(cols) == 0:
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
    def choose_helper(method_type, enable_scaling=False, enable_normalization=False, **args):
        if 'dl_args' in args and args['dl_args'] is not None:
            return DeepLearningHelper(method_type, enable_scaling, enable_normalization, **args)
        else:
            return ScikitLearnHelper(method_type, enable_scaling, enable_normalization, **args)


class ScikitLearnHelper(SupervisedLearningHelper):
    scaler = RobustScaler()
    normalizer = Normalizer()

    def __init__(self, method_type, enable_scaling=False, enable_normalization=False, **args):
        self.model = select_method(args['choosing_method'], method_type)
        super().__init__(method_type, enable_scaling, enable_normalization, **args)

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
    def __init__(self, method_type, enable_scaling=False, enable_normalization=False, **args):
        dl_args = args['dl_args']
        dl_args.update({'num_inputs': args['num_inputs'], 'num_outputs': 1, 'enable_scaling': enable_scaling, 'enable_normalization': enable_normalization})
        self.model = NeuralNetwork(dl_args)
        super().__init__(method_type, enable_scaling, enable_normalization, **args)

import pandas as pd
from sklearn.preprocessing import Normalizer, PolynomialFeatures

from .common import select_method, load_from_directory, ScalingType, get_scaler_by_type


class SupervisedLearningHelper:
    method_type = None
    state_dim = 0
    files_dir = ''
    features = []
    label = ''
    historical_data = None
    model = None
    scaling_type = None
    enable_normalization = False

    def __init__(self, method_type, enable_normalization=False, scaling_type=ScalingType.NONE, **args):
        self.method_type = method_type
        self.scaling_type = scaling_type
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
                                                       header_index=args['header_index']
                                                       if 'header_index' in args else 0)
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
    def choose_helper(method_type, enable_normalization=False, scaling_type=ScalingType.NONE, **args):
        return ScikitLearnHelper(method_type, enable_normalization, scaling_type, **args)


class ScikitLearnHelper(SupervisedLearningHelper):
    scaler = None
    normalizer = Normalizer()
    poly = PolynomialFeatures()

    def __init__(self, method_type, enable_normalization=False, scaling_type=ScalingType.NONE, **args):
        use_grid_search = args['use_grid_search'] if 'use_grid_search' in args else False
        self.model = select_method(args['choosing_method'], method_type, use_grid_search, enable_normalization, args['cv'])
        self.scaler = get_scaler_by_type(scaling_type)
        if args['choosing_method'] in ['Linear Regression', 'Lasso', 'Ridge', 'Elastic Net']:
            enable_normalization = False
        super().__init__(method_type, enable_normalization, scaling_type, **args)

    def fit(self, x, y):
        if len(self.features) > 1:
            x = self.poly.fit_transform(x, y)
        if self.scaler is not None:
            x = self.scaler.fit_transform(x, y)
        if self.enable_normalization:
            x = self.normalizer.fit_transform(x, y)
        self.model.fit(x, y)

    def predict(self, inputs):
        if len(self.features) > 1:
            inputs = self.poly.transform(inputs)
        if self.scaler is not None:
            inputs = self.scaler.transform(inputs)
        if self.enable_normalization:
            inputs = self.normalizer.transform(inputs)
        return self.model.predict(inputs)

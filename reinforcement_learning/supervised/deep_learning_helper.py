from neural_network.neural_network import NeuralNetwork
from .supervised_learning_helper import SupervisedLearningHelper
from supervised_learning.common import load_from_directory
import pandas as pd


class DeepLearningHelper(SupervisedLearningHelper):
    model = None

    def __init__(self, method_type, csv_dir, state_dim, label, filters={}, dl_args={}):
        dl_args.update({'num_inputs': state_dim, 'num_outputs': 1})
        self.model = NeuralNetwork(dl_args)
        df_from_each_file = load_from_directory(self.csv_dir, self.features, self.label, self.filters)
        if len(df_from_each_file) > 0:
            self.historical_data = pd.concat(df_from_each_file, ignore_index=True)
        if self.historical_data is not None:
            x = self.historical_data[self.features]
            y = self.historical_data[self.label]
            self.fit(x, y)
        super().__init__(method_type, csv_dir, state_dim, label, filters)

    def fit(self, x, y):
        self.model.update_network(x, y)

    def get_predictions(self, inputs):
        self.model.predict(inputs)

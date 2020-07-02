from neural_network.neural_network import NeuralNetwork
from .supervised_learning_helper import SupervisedLearningHelper


class DeepLearningHelper(SupervisedLearningHelper):
    model = None

    def __init__(self, method_type, csv_dir, state_dim, label, filters={}, dl_args={}):
        dl_args.update({'num_inputs': state_dim, 'num_outputs': 1})
        self.model = NeuralNetwork(dl_args)
        columns = self.features
        columns.append(self.label)
        self.model.load_data_from_directory(self.csv_dir, columns)
        super().__init__(method_type, csv_dir, state_dim, label, filters)

    def fit(self, x, y):
        self.model.update_network(x, y)

    def get_predictions(self, inputs):
        self.model.predict(inputs)

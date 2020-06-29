from supervised_learning.common import select_best_regressor, randomly_select_regressor, randomly_select_classifier, \
    select_best_classifier
from .supervised_learning_helper import SupervisedLearningHelper, MethodType
import pandas as pd


class ScikitLearnHelper(SupervisedLearningHelper):
    method = None

    def update(self):
        if self.method_type == MethodType.Regression:
            self.method, self.historical_data = select_best_regressor(self.csv_dir, features=self.features,
                                                                      label=self.label, filters=self.filters)
            if self.method is None:
                self.method = randomly_select_regressor()
        else:
            self.method, self.historical_data = select_best_classifier(self.csv_dir, features=self.features,
                                                                       label=self.label)
            if self.method is None:
                self.method = randomly_select_classifier()

    def fit(self, x, y):
        self.method.fit(x, y)

    def get_predictions(self, inputs):
        return self.method.predict(inputs)

from reinforcement_learning.supervised.supervised_learning_helper import SupervisedLearningHelper, MethodType


class NegativeActionBlocker(SupervisedLearningHelper):
    def __init__(self, csv_dir, state_dim):
        super().__init__(MethodType.Classification, csv_dir, state_dim, 'BLOCKED?')

    def add(self, state, action, blocked_boolean):
        super().add(state, action, 1 if blocked_boolean else 0)

    def should_we_block_action(self, state, action):
        return super().predict(state, action) == 1

from reinforcement_learning.supervised.supervised_learning_helper import SupervisedLearningHelper, MethodType


class NegativeActionBlocker:
    supervised_learning_helper = None

    def __init__(self, csv_dir, state_dim, dl_args=None):
        self.supervised_learning_helper = SupervisedLearningHelper.choose_helper(MethodType.Classification, csv_dir, state_dim, 'BLOCKED?', {}, dl_args)

    def add(self, state, action, blocked_boolean):
        self.supervised_learning_helper.add(state, action, 1 if blocked_boolean else 0)

    def should_we_block_action(self, state, action):
        return self.supervised_learning_helper.predict(state, action) == 1

from supervised_learning.supervised_learning_helper import MethodType
from .rl_supervised_helper import RLSupervisedHelper


class NegativeActionBlocker(RLSupervisedHelper):
    def __init__(self, csv_dir, state_dim, dl_args=None):
        super().__init__(MethodType.Classification, csv_dir, state_dim, 'BLOCKED?', {}, dl_args)

    def add(self, state, action, blocked_boolean):
        super().add(state, action, 1 if blocked_boolean else 0)

    def should_we_block_action(self, state, action):
        return super().predict(state, action) == 1

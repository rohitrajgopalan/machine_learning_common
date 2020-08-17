from supervised_learning.common import MethodType
from .rl_supervised_helper import RLSupervisedHelper


class NegativeActionBlocker(RLSupervisedHelper):
    def __init__(self, csv_dir, state_dim, action_dim, dl_args=None):
        if dl_args is not None:
            dl_args.update({'loss_function': 'binary_crossentropy'})
        super().__init__(MethodType.Classification, csv_dir, state_dim, action_dim, 'BLOCKED?', dl_args)

    def add(self, state, action, blocked_boolean):
        super().add(state, action, 1 if blocked_boolean else 0)

    def should_we_block_action(self, state, action):
        return super().predict(state, action) == 1

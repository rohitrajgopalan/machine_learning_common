from .base_policy import Policy
from .epsilon_greedy import EpsilonGreedy
from .softmax import Softmax
from .static import StaticPolicy
from .thompson_sampling import ThompsonSampling
from .ucb import UCB


def choose_policy(policy_name, args=None):
    if args is None:
        args = {}
    if policy_name == 'epsilon_greedy':
        return EpsilonGreedy(args)
    elif policy_name == 'softmax':
        return Softmax(args)
    elif policy_name == 'thompson_sampling':
        return ThompsonSampling(args)
    elif policy_name == 'ucb':
        return UCB(args)
    elif policy_name == 'static':
        return StaticPolicy(args)
    else:
        return Policy(args)

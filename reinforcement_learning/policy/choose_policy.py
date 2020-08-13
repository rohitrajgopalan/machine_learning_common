from .policy import Policy
from .epsilon_greedy import EpsilonGreedy
from .softmax import Softmax
from .static import StaticPolicy
from .thompson_sampling import ThompsonSampling
from .network import NetworkPolicy
from .ucb import UCB


def choose_policy(policy_name, args={}):
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
    elif policy_name == 'network':
        return NetworkPolicy(args)
    else:
        return Policy(args)

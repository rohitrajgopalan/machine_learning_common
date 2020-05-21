from reinforcement_learning.algorithm.base_algorithm import Algorithm
from reinforcement_learning.algorithm.td import SARSA, Q, ExpectedSARSA, MCQL
from reinforcement_learning.algorithm.td_lambda import SARSALambda, QLambda, ExpectedSARSALambda, MCQLambda


def choose_algorithm(algorithm_name, args=None):
    if args is None:
        args = {}
    if algorithm_name == 'sarsa':
        return SARSA(args)
    elif algorithm_name == 'sarsa_lambda':
        return SARSALambda(args)
    elif algorithm_name == 'q':
        return Q(args)
    elif algorithm_name == 'q_lambda':
        return QLambda(args)
    elif algorithm_name == 'expected_sarsa':
        return ExpectedSARSA(args)
    elif algorithm_name == 'expected_sarsa_lambda':
        return ExpectedSARSALambda(args)
    elif algorithm_name == 'mcql':
        return MCQL(args)
    elif algorithm_name == 'mcq_lambda':
        return MCQLambda(args)
    else:
        return Algorithm(args)
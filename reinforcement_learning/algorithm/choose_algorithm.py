from reinforcement_learning.algorithm.base_algorithm import *


def choose_algorithm(name, args={}):
    if name == 'td':
        return TDAlgorithm(args)
    elif name == 'td_lambda':
        return TDLambdaAlgorithm(args)
    else:
        return Algorithm(args)

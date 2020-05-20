from algorithm import Algorithm
from td import *
from td_lambda import *
from delayed import *

def choose_algorithm(algorithm_name,**args):
    if algorithm_name == 'sarsa':
        return SARSA(args)
    elif algorithm_name == 'sarsa_lambda':
        return SARSALambda(args)
    elif algorithm_name == 'delayed_sarsa':
        return DelayedSARSA(args)
    elif algorithm_name == 'q':
        return Q(args)
    elif algorithm_name == 'q_lambda':
        return QLambda(args)
    elif algorithm_name == 'delayed_q':
        return DelayedQ(args)
    elif algorithm_name == 'expected_sarsa':
        return ExpectedSARSA(args)
    elif algorithm_name == 'expected_sarsa_lambda':
        return ExpectedSARSALambda(args)
    elif algorithm_name == 'delayed_expected_sarsa':
        return DelayedExpectedSARSA(args)
    elif algorithm_name == 'mcql':
        return MCQL(args)
    elif algorithm_name == 'mcq_lambda':
        return MCQLambda(args)
    elif algorithm_name == 'delayed_mcql':
        return DelayedMCQL(args)
    else:
        return Algorithm(args)

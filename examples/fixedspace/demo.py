from os.path import dirname, realpath

from examples.fixedspace.fixedspaceenvironment import FixedSpaced
# from examples.common_online_demo import run_demo
from examples.simple_online_demo import run_demo

environment = FixedSpaced(5)

agent_info_list = [{'initial_state': (0, 0), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (0, 2), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (0, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (2, 0), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (2, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (4, 0), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (4, 2), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
    , {'initial_state': (4, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}]

run_demo(environment, agent_info_list, dirname(realpath('__file__')))

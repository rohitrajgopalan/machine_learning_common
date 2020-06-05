from os.path import dirname, realpath

# from examples.common_online_demo import run_demo
from examples.simple_online_demo import run_demo
from examples.fixedspace.fixedspaceenvironment import FixedSpaced

environment = FixedSpaced(4)

agent_info_list = [{'initial_state': (1, 1), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   , {'initial_state': (1, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   , {'initial_state': (4, 1), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}
                   , {'initial_state': (4, 4), 'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT']}]

run_demo(environment, agent_info_list, dirname(realpath('__file__')))

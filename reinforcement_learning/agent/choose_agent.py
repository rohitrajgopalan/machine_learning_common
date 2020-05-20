from agent import agent
from dyna import DynaAgent
from dyna_plus import DynaPlusAgent
from prioritized_sweeping import PrioritizedSweepingAgent

def choose_agent(agent_name,**args):
    if agent_name == 'dyna':
        return DynaAgent(args)
    elif agent_name == 'dyna+':
        return DynaPlusAgent(args)
    elif agent_name == 'prioritized_sweeping':
        return PrioritizedSweeping(args)
    else:
        return Agent(args)

from reinforcement_learning.agent.base_agent import Agent
from reinforcement_learning.agent.dyna import DynaAgent
from reinforcement_learning.agent.dyna_plus import DynaPlusAgent
from reinforcement_learning.agent.prioritized_sweeping import PrioritizedSweepingAgent


def choose_agent(agent_name, args=None):
    if agent_name == 'dyna':
        return DynaAgent(args)
    elif agent_name == 'dyna+':
        return DynaPlusAgent(args)
    elif agent_name == 'prioritized_sweeping':
        return PrioritizedSweepingAgent(args)
    else:
        return Agent(args)

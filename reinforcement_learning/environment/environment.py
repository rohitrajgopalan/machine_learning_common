import numpy as np
import enum
from agent import Agent

class RewardType(enum.Enum):
    Immediate = 1
    Delayed = 2

class Environment:
    agents = []
    reward_type = None

    def __init__(self,reward_type):
        self.reward_type = reward_type

    def calculate_reward(self,agent):
        return (0,False)

    def determine_next_state(self,agent):
        pass

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def is_complete(self):
        return False

    def step(self):
        active_agents = [agent for agent in self.agents if agent.active]
        if len(active_agents) == 0:
            return True
        
        for agent in active_agents:
            agent.choose_next_action(agent.current_state)
            self.determine_next_state(agent)
            if self.reward_type == RewardType.Immediate:
                r,terminal = self.calculate_reward(agent)
                agent.step(r,terminal)
                if self.is_complete():
                    return True
        
        if self.reward_type == RewardType.Delayed:
            for agent in active_agents:
                r,terminal = self.calculate_reward(agent)
                agent.step(r,terminal)
            return self.is_complete()
        else:
            return False

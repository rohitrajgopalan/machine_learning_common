import enum


class RewardType(enum.Enum):
    Immediate = 1
    Delayed = 2


class Environment:
    agents = []
    reward_type = None
    required_state_dim = 0

    def __init__(self, reward_type, required_state_dim):
        self.reward_type = reward_type
        self.required_state_dim = required_state_dim

    def calculate_reward(self, agent):
        return 0

    def determine_next_state(self, agent):
        pass

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def is_complete(self):
        active_agents = [agent for agent in self.agents if agent.active]
        return len(active_agents) == 0

    def step(self):
        if self.is_complete():
            return True
        active_agents = [agent for agent in self.agents if agent.active]
        for agent in active_agents:
            agent.choose_next_action()
            self.determine_next_state(agent)
            if self.reward_type == RewardType.Immediate:
                r = self.calculate_reward(agent)
                agent.step(r)
                if self.is_complete():
                    return True

        if self.reward_type == RewardType.Delayed:
            for agent in active_agents:
                r = self.calculate_reward(agent)
                agent.step(r)
            return self.is_complete()
        else:
            return False

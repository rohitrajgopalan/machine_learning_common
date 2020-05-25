import numpy as np
from reinforcement_learning.environment.environment import Environment, RewardType


class FixedSpaced(Environment):
    square_length = 0
    final_location = (0, 0)

    def __init__(self, square_length):
        super().__init__(RewardType.Delayed, 2)
        self.square_length = square_length
        self.final_location = (square_length / 2, square_length / 2)

    def calculate_reward(self, agent):
        terminal = False

        if agent.next_state == self.final_location:
            reward = 100
            terminal = True
        else:
            num_collisions = 0
            other_agents = [other for other in self.agents if not other.agent_id == agent.agent_id]
            for other in other_agents:
                if agent.next_state == other.next_state:
                    num_collisions += 1

            if num_collisions == 0:
                if agent.next_state == agent.current_state:
                    reward = 0
                else:
                    reward = 1
            else:
                agent.next_state = agent.current_state
                reward = -10 * num_collisions
        if terminal:
            agent.active = False
        return reward

    def determine_next_state(self, agent):
        chosen_action = agent.actions[agent.actual_action]

        x, y = agent.current_state
        if chosen_action == 'UP':
            x = max(x - 1, 1)
        elif chosen_action == 'DOWN':
            x = min(x + 1, self.square_length)
        elif chosen_action == 'LEFT':
            y = max(y - 1, 1)
        elif chosen_action == 'RIGHT':
            y = min(y + 1, self.square_length)
        agent.next_state = (x, y)

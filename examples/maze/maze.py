import numpy as np

from reinforcement_learning.environment.environment import Environment, RewardType


class MazeEnvironment(Environment):
    def __init__(self, maze_dim, obstacles, start_location, final_location):
        super().__init__(RewardType.Immediate, 2)
        self.maze_dim = maze_dim
        self.obstacles = obstacles
        self.start_location = start_location
        self.final_location = final_location

    def determine_next_state(self, agent):
        chosen_action = agent.actions[agent.actual_action]

        x = agent.current_state[0, 0]
        y = agent.current_state[0, 1]
        if chosen_action == 'UP':
            x = x - 1
        elif chosen_action == 'DOWN':
            x = x + 1
        elif chosen_action == 'LEFT':
            y = y - 1
        elif chosen_action == 'RIGHT':
            y = y + 1
        agent.next_state = np.array([x, y])

    def out_of_bounds(self, row, col):
        return row < 0 or row > self.maze_dim[0] or col < 0 or col > self.maze_dim[1]

    def is_obstacle(self, row, col):
        return (row, col) in self.obstacles

    def calculate_reward(self, agent):
        terminal = False

        row, col = agent.next_state
        if agent.next_state == self.final_location:
            reward = 100
            terminal = True
        elif self.out_of_bounds(row, col) or self.is_obstacle(row, col):
            reward = -10
            agent.next_state = agent.current_state
        else:
            reward = 1

        return reward, terminal

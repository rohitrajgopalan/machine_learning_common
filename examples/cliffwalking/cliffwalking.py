from reinforcement_learning.environment.environment import Environment, RewardType


class CliffWalkingEnvironment(Environment):
    grid_h = 0
    grid_w = 0
    goal_loc = (0, 0)
    cliff = []

    def __init__(self, grid_h, grid_w):
        super().__init__(RewardType.Immediate, 2)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.goal_loc = (self.grid_h, self.grid_w)
        self.cliff = [(self.grid_h, i) for i in range(1, self.grid_w+1)]

    def determine_next_state(self, agent):
        chosen_action = agent.actions[agent.actual_action]
        x, y = agent.current_state
        if chosen_action == 'UP':
            x = max(x - 1, 1)
        elif chosen_action == 'DOWN':
            x = min(x + 1, self.grid_h)
        elif chosen_action == 'LEFT':
            y = max(y - 1, 1)
        elif chosen_action == 'RIGHT':
            y = min(y + 1, self.grid_w)
        agent.next_state = (x, y)

    def calculate_reward(self, agent):
        reward = -1
        terminal = False

        if agent.next_state == self.goal_loc:
            reward = 1000
            terminal = True
        elif agent.next_state in self.cliff:
            reward = -100
            agent.next_state = agent.initial_state
        elif agent.current_state == agent.next_state:
            reward = -2
        return reward, terminal

from environment import Environment,RewardType
import numpy as np

class CliffWalkingEnvironment(Environment):
    grid_h = 0
    grid_w = 0
    goal_loc = (0,0)
    cliff = []

    def __init__(self,grid_h,grid_w):
        super().__init__(RewardType.Immediate)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.goal_loc = (self.grid_h,self.grid_w)
        self.cliff = [(self.grid_h,i) for i range (1,(self.grid_w))]

    def determine_next_state(self,agent):
        state_as_list = list(np.array([agent.current_state]))
        if len(state_as_list) < 2:
            for _ in range(len(state_as_list),2):
                state_as_list.append(1)
            agent.current_state = tuple(state_as_list)
        elif len(state_as_list) > 2:
            agent.current_state =(state_as_list[0],state_as_list[1])

        chosen_action = agent.actions[agent.actual_action]
        x,y = agent.current_state
        if chosen_action == 'UP':
            x = max(x-1,1)
        elif chosen_action == 'DOWN':
            x = min(x+1,self.square_length)
        elif chosen_action == 'LEFT':
            y = max(y-1,1)
        elif chosen_action == 'RIGHT':
            y = max(y+1,self.square_length)
        agent.next_state = (x,y)

    def calculate_reward(self,agent):
        reward = -1
        terminal = False

        if agent.next_state == self.goal_loc:
            reward = 1000
            terminal = True
        elif agent.next_state in self.cliff:
            reward = -100
            agent.next_state = agent.initial_state
        return (reward,terminal)

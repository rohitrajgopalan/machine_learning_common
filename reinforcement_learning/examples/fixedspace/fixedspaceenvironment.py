import numpy as np
from environment import Environment,RewardType

class FixedSpaced(Environment):
    square_length = 0
    final_location = (0,0)

    def __init__(self,square_length):
        super().__init__(RewardType.Delayed)
        self.square_length = square_length
        self.final_location = (square_length/2,square_length/2)

    def calculate_reward(self,agent):
        reward = 0
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
                reward = 1
            else:
                reward = -10*num_collisions
        return (reward,terminal)

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
        
    def is_complete(self):
        active_agents = [agent for agent in self.agents if agent.active]
        return len(active_agents) == 0
    
        
        

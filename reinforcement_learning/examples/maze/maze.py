from environment import Environment,RewardType

class MazeEnvironment(Environment):
    maze_dim = (6,9)
    obstacles = [(2, 3), (3, 3), (4, 3), (5, 6), (1, 8), (2, 8), (3, 8)]
    final_location = (1,9)
    start_location = (3,1)

    def __init__(self,maze_dim,obstacles,start_location,final_location):
        super().__init__(RewardType.Immediate)
        self.maze_dim = maze_dim
        self.obstacles = obstacles
        self.start_location = start_location
        self.final_location = final_location

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
            x = x-1
        elif chosen_action == 'DOWN':
            x = x+1
        elif chosen_action == 'LEFT':
            y = y-1
        elif chosen_action == 'RIGHT':
            y = y+1
        agent.next_state = (x,y)
        
    def out_of_bounds(self,row,col):
        return row < 0 or row > self.maze_dim[0] or col < 0 or col > self.maze_dim[1]

    def is_obstacle(self,row,col):
        return (row,col) in self.obstacles

    def calculate_reward(self,agent):
        reward = 0
        terminal = False

        row,col = agent.next_state
        if agent.next_state == self.final_location:
            reward = 100
            terminal = True
        elif self.out_of_bounds(row,col) or self.is_obstacle(row,col):
            reward = -10
            agent.next_state = agent.current_state
        else:
            reward = 1
        
        return (reward,terminal)

        

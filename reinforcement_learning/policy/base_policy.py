import numpy as np

class Policy:
    num_actions = 0
    rand_generator = None

    def __init__(self,args={}):
        for key in args:
            if key == 'random_seed':
                self.rand_generator = np.random.RandomState(args[key])
            else:
                setattr(self,key,args[key])

    def actions_with_max(self,action_values):
        max_value = np.max(action_values)
        ties = []
        for i in range(num_actions):
            if action_values[i] > max_value:
                max_value = action_values[0,i]
                ties = []
                ties.add(i)
            elif action_values[i] == max_value:
                ties.add(i)
        return ties
    
    def argmax(self,action_values):
        ties = actions_with_max(action_values)
        return self.rand_generator.choice(ties)

    def derive(self,action_values):
        return np.zeros(self.num_actions)

    def choose_action(self,action_values):
        return 0

    def update(self,reward,action):
        pass

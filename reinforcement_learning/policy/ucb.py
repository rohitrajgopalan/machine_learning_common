from .base_policy import Policy
import numpy as np

class UCB(Policy):
    N = None
    time_step_counter = 1
    ucb_c = 0.0

    def __init__(self,args={}):
        super().__init__(args)
        self.N = np.zeros(self.num_actions)+0.001

    def generate_confidence(self):
        ln_timestep = np.log(np.full(self.num_actions,self.time_step_counter))
        return self.ucb_c * np.sqrt(ln_timestep/self.N)

    def derive(self,action_values):
        policy_probs = np.zeros(self.num_actions)
        actions_with_max = self.actions_with_max(actions_values+self.generate_confidence())
        for action in range(self.num_actions):
            if action in actions_with_max:
                policy_probs[action] = 1/len(actions_with_max)
        return policy_probs

    def choose_action(self,action_values):
        return self.argmax(action_values+self.generate_confidence())

    def update(self,reward,action):
        self.N[action] += 1
        time_step_counter += 1

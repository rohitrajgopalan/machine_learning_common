from dyna import DynaAgent
import numpy as np

class DynaPlusAgent(DynaAgent):
    kappa = 0.0
    tau = None

    def __init__(self,**args):
        self.tau = np.empty((0,len(self.actions)))
        super().__init__(args)

    def add_state(self,s):
        if self.get_state_index(s) == -1:
            super().add_state(s)
            self.tau = np.append(self.tau,np.zeros((1,len(self.actions))),axis=0)

    def step(self,r,terminal=False):
        self.tau += 1
        super().step(r,terminal)
        self.tau[self.get_state_index(self.last_state),self.last_action] = 0
        
    def refine_reward(self,s,a,r):
        return r + (self.kappa * np.sqrt(self.tau[self.get_state_index(s),a]))

    def update_model(r,terminal=False):
        state_index = self.get_state_index(self.last_state)
        state_index_ = self.get_state_index(self.current_state)

        if not state_index in self.model:
            self.model[state_index] = {self.last_action: (state_index_,r,int(terminal))}
            for action in range(len(self.actions)):
                if not action == self.last_action:
                    self.model[state_index].update({action: (state_index,0,0)})
        else:
            self.model[state_index][self.last_action] = (state_index_,r,int(terminal))

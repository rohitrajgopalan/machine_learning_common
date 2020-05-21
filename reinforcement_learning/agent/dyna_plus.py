import numpy as np

from reinforcement_learning.agent.dyna import DynaAgent


class DynaPlusAgent(DynaAgent):
    kappa = 0.0
    tau = {}

    def __init__(self, args=None):
        super().__init__(args)

    def step(self, r, terminal=False):
        if self.algorithm.get_state_index(self.next_state) == -1:
            self.tau.update({self.last_state: [0]*len(self.actions)})
        for s in self.tau:
            for a in range(len(self.actions)):
                self.tau[s][a] += 1
        super().step(r, terminal)
        self.tau[self.state_space.index(self.last_state)][self.last_action] = 0

    def refine_reward(self, s, a, r):
        return r + (self.kappa * np.sqrt(self.tau[self.algorithm.get_state_index(s), a]))

    def update_model(self, r, terminal=False):
        if self.last_state not in self.model:
            self.model[self.last_state] = {self.last_action: (self.current_state, r, int(terminal))}
            for action in range(len(self.actions)):
                if not action == self.last_action:
                    self.model[self.last_state].update({action: (self.current_state, 0, 0)})
        else:
            self.model[self.last_state][self.last_action] = (self.current_state, r, int(terminal))

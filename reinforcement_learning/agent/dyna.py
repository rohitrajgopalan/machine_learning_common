from agent import *
import numpy as np

class DynaAgent(Agent):
    model = {}
    num_planning_steps = 0
    plan_rand_generator = None
    last_state = None
    last_action = -1

    def step(self,r,terminal=False):
        self.last_state = self.current_state
        self.last_action = self.initial_action
        super().step(self.refined_reward(self.last_state,self.last_action,r),terminal)
        self.update_model(r,terminal)
        self.planning_step()

    def refine_reward(self,s,a,r):
        return r

    def update_model(self,r,terminal=False):
        state_index = self.get_state_index(self.last_state)
        state_index_ = self.get_state_index(self.current_state)

        if not state_index in self.model:
            self.model[state_index] = {}

        self.model[state_index].update({self.last_action: (self.current_state,r,int(terminal))})

    def planning_step(self):
        experiences = []
        current_q = deepcopy(self.network)
        for _ in range(self.num_planning_steps):
            state_index = self.plan_rand_generator.choice(list(self.model.keys()))
            a = self.plan_rand_generator.choice(list(self.model[state_index].keys()))

            (state_index_,r,terminal) = self.model[state_index][a]

            s = self.state_space[state_index]
            s_= self.state_space[state_index_]

            r = self.refine_reward(s,a,r)

            if self.learning_type == LearningType.Replay:
                experiences.append((s,a,r,terminal,s_))
            else:
                self.optimize_network(s,a,s_,r,terminal,current_q)

        if self.learning_type == LearningType.Replay:
            self.optimize_network(experiences,current_q)

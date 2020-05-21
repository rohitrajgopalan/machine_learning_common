from copy import deepcopy

from reinforcement_learning.agent.base_agent import Agent, LearningType


class DynaAgent(Agent):
    model = {}
    num_planning_steps = 0
    plan_rand_generator = None
    last_state = None
    last_action = -1

    def step(self, r, terminal=False):
        self.last_state = self.current_state
        self.last_action = self.initial_action
        super().step(self.refine_reward(self.last_state, self.last_action, r), terminal)
        self.update_model(r, terminal)
        self.planning_step()

    def refine_reward(self, s, a, r):
        return r

    def update_model(self, r, terminal=False):
        if self.last_state not in self.model:
            self.model[self.last_state] = {}

        self.model[self.last_state].update({self.last_action: (self.current_state, r, int(terminal))})

    def planning_step(self):
        experiences = []
        current_q = deepcopy(self.network)
        for _ in range(self.num_planning_steps):
            s = self.plan_rand_generator.choice(list(self.model.keys()))
            a = self.plan_rand_generator.choice(list(self.model[s].keys()))

            (s_, r, terminal) = self.model[s][a]

            r = self.refine_reward(s, a, r)

            if self.learning_type == LearningType.Replay:
                experiences.append((s, a, r, terminal, s_))
            else:
                self.optimize_network(s, a, s_, r, terminal, current_q)

        if self.learning_type == LearningType.Replay:
            self.optimize_network(experiences, current_q)

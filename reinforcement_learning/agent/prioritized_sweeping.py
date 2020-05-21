from copy import deepcopy
from queue import PriorityQueue

from reinforcement_learning.agent.base_agent import Agent, LearningType


class PrioritizedSweepingAgent(Agent):
    queue = PriorityQueue()
    predecessors = {}
    theta = 0.0
    model = {}
    num_planning_steps = 0
    plan_rand_generator = None

    def add_to_queue(self, coin_side, s, a, s_, r, terminal):
        priority = self.algorithm.get_target_error(s, a, s_, r, int(terminal), self.network, deepcopy(self.network))
        if priority > self.theta:
            self.queue.put((-priority, (s, a, coin_side)))

    def update_model(self, r, terminal=False):
        if self.current_state not in self.model:
            self.model[self.current_state] = {}

        self.model[self.current_state].update({self.initial_action: (self.next_state, r, int(terminal))})

    def step(self, r, terminal=False):
        coin_side = self.network.determine_coin_side()
        if not self.initial_action == self.actual_action:
            r *= -1

        self.active = not terminal

        # TODO: Add data to supervised learning
        self.add_to_supervised_learning(r)

        if r < 0:
            self.next_state = self.current_state

        self.add_to_state_space(self.next_state)

        self.add_to_queue(coin_side, self.current_state, self.initial_action, self.next_state, r, terminal)
        self.update_model(r, terminal)

        if self.next_state not in self.predecessors:
            self.predecessors[self.next_state] = []
            self.predecessors[self.next_state].append((self.current_state, self.initial_action, coin_side))

        if self.active:
            self.current_state = self.next_state

        self.planning_step()

    def planning_step(self):
        experiences = []
        current_q = deepcopy(self.network)
        for _ in range(self.num_planning_steps):
            if self.queue.empty():
                break
            (s, a, coin_side) = self.queue.get()[1]
            (s_, r, terminal) = self.model[s][a]

            if self.learning_type == LearningType.Replay:
                experiences.append((s, a, r, terminal, s_))
            else:
                self.optimize_network(s, a, s_, r, terminal, current_q)

            if s_ not in self.predecessors:
                continue

            for predecessor in self.predecessors[s_]:
                pre_state, pre_action, pre_coin_side = predecessor
                _, pre_reward, pre_terminal = self.model[pre_state][pre_action]
                self.add_to_queue(pre_coin_side, pre_state, pre_action, s, pre_reward, pre_terminal)

        if self.learning_type == LearningType.Replay:
            self.optimize_network(experiences, current_q)

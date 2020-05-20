from agent import Agent
from queue import PriorityQueue
import numpy as np

class PrioritizedSweepingAgent(Agent):
    queue = PriorityQueue()
    predecessors = {}
    theta = 0.0
    model = {}
    num_planning_steps = 0
    plan_rand_generator = None

    def add_to_queue(self,coin_side,s,a,s_,r,terminal):
        state_index = self.get_state_index(s)
        self.e_traces[state_index,a] += 1
        q_values = self.network.get_action_values(s,coin_side)
        q_values_ = self.network.get_action.values(s_,self.network.network_type.value-1-coin_side)

        priority = self.algorithm.get_target_error(a,r,int(terminal),self.e_traces[state_index],q_values,q_values_)
        if priority > self.theta:
            self.queue.put((-priority,(state_index,a,coin_side)))

    def update_model(self,r,terminal=False):
        state_index = self.get_state_index(self.current_state)
        state_index_ = self.get_state_index(self.next_state)

        if not state_index in self.model:
            self.model[state_index] = {}

        self.model[state_index].update({self.initial_action: (self.next_state,r,int(terminal))})

    def step(self,r,terminal=False):
        coin_side = self.network.determine_coin_side()
        state_index = self.get_state_index(self.current_state)
        state_index_ = self.get_state_index(self.next_state)
        if not self.initial_action == self.actual_action:
            r *= -1

        self.active = not terminal

        #TODO: Add data to supervised learning

        if r < 0:
            self.next_state = self.current_state

        self.add_to_queue(coin_side,self.current_state,self.initial_action,self.next_state,r,terminal)
        self.update_model(r,terminal)

        if not state_index_ in self.predecessors:
            self.predecessors[state_index_] = []
        self.predecessors[state_index_].append((state_index,a,coin_side))

        if self.active:
            self.current_state = self.next_state
            self.add_state(self.next_state)
        
        self.planning_step()

    def planning_step(self):
        experiences = []
        current_q = deepcopy(self.network)
        for _ in range(self.num_planning_steps):
            if self.queue.empty():
                break
            (state_index,a,coin_side) = self.queue.get()[1]
            (state_index_,r,terminal) = self.model[state_index][a]

            s = self.state_space[state_index]
            s_ = self.state_space[state_index_]

            if self.learning_type == LearningType.Replay:
                experiences.append((s,a,r,terminal,s_))
            else:
                self.optimize_network(s,a,s_,r,terminal,current_q)

            if not state_index_ in self.predecessors:
                continue

            for predecessor in self.predecessors[state_index_]:
                pre_state_index,pre_action,pre_coin_side = predecessor
                _,pre_reward,pre_terminal = self.model[pre_state_index][pre_action]

                pre_state = self.state_space[pre_state_index]
                self.add_to_queue(pre_coin_side,pre_state,pre_action,s,pre_reward,pre_terminal)

        if self.learning_type == LearningType.Replay:
            self.optimize_network(experiences,current_q)
            

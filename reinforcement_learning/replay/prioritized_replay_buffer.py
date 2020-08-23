from queue import PriorityQueue
from .replay_buffer import ReplayBuffer
import math


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, minibatch_size, **args):
        super().__init__(size, minibatch_size)
        self.buffer = PriorityQueue(maxsize=size)

    def append(self, state, action, next_state, reward, terminal, picked_action_prob=0, **args):
        if self.buffer.full():
            _ = self.buffer.get()
        priority = 0
        if 'target_error' in args:
            priority = args['target_error']
        elif 'td_error' in args:
            priority = args['td_error']
        self.buffer.put((math.fabs(priority), [state, action, next_state, reward, terminal, picked_action_prob]))

    def sample(self):
        num_samples = 0
        samples = []
        while not self.buffer.empty() and num_samples < self.minibatch_size:
            samples.append(self.buffer.get())
            num_samples += 1
        return samples

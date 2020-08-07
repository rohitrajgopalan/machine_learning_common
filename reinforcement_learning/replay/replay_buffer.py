class ReplayBuffer:
    def __init__(self, size, minibatch_size, **args):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (float/integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        if type(minibatch_size) == int:
            self.minibatch_size = minibatch_size
        elif type(minibatch_size) == float and 0 < minibatch_size <= 1.0:
            self.minibatch_size = int(minibatch_size * size)
        self.max_size = size

    def append(self, state, action, next_state, reward, terminal, **args):
        pass

    def sample(self):
        return []

    def size(self):
        return len(self.buffer)

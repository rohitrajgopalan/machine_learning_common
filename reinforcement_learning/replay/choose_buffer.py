from .basic_replay_buffer import BasicReplayBuffer
from .randomized_replay_buffer import RandomizedReplayBuffer
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .buffer_type import BufferType


def choose_buffer(buffer_type, size, minibatch_size, seed=None):
    if type(buffer_type) == str:
        buffer_type = BufferType.get_type_by_name(buffer_type)
    if buffer_type == BufferType.RANDOMIZED or seed is not None:
        return RandomizedReplayBuffer(size, minibatch_size, seed=seed)
    else:
        return PrioritizedReplayBuffer(size,
                                       minibatch_size) if buffer_type == BufferType.PRIORITIZED else BasicReplayBuffer(
            size, minibatch_size)

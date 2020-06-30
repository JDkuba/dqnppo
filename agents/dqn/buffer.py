from collections import deque
import random as rd
from agents.dqn.sum_tree import SumTree


class SimpleBuffer:
    def __init__(self, memory_size=10000):
        self.memory = deque(maxlen=memory_size)

    def sample_batch(self, sample_size):
        batch = rd.sample(self.memory, sample_size)
        return batch

    def length(self):
        return len(self.memory)

    def update_memory(self, element):
        self.memory.append(element)


class PrioritizedBuffer:
    def __init__(self, memory_size=10000, epsilon=0.01, alpha=0.6):
        self.tree = SumTree(max_size=memory_size)
        self.max_size = memory_size
        self.epsilon = epsilon
        self.alpha = alpha

    def length(self):
        return self.tree.size

    def update_memory(self, sample, error):
        self.tree.push(sample, self.__priority(error))

    def sample_batch(self, sample_size):
        batch, indices = [], []
        s = self.tree.top() / sample_size
        left, right = 0, s

        for _ in range(sample_size):
            p = rd.uniform(left, right)
            idx, sample = self.tree.get(p)
            batch.append(sample)
            indices.append(idx)
            left, right = left + s, right + s

        return batch, indices

    def update_priorities(self, idx, error):
        self.tree.update_priority(idx, self.__priority(error))

    def __priority(self, error):
        return (error + self.epsilon) ** self.alpha

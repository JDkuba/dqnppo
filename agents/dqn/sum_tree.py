import numpy as np
from math import floor


class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree = np.zeros(2 * max_size - 1)
        self.memory = np.empty(max_size, dtype=tuple)
        self.size = 0
        self.current = 0

    def top(self):
        return self.tree[0]

    def push(self, sample, priority):
        self.memory[self.current] = sample
        h = self.current + self.max_size - 1
        self.update_priority(h, priority)
        self.current = (self.current + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_priority(self, level, priority):
        shift = priority - self.tree[level]
        self.tree[level] = priority
        self.__propagate(level, shift)

    def get(self, p):
        idx = self.__retrieve(0, p)
        i = idx - self.max_size + 1
        return idx, self.memory[i]

    def __propagate(self, level, shift):
        parent = floor((level - 1) / 2)
        self.tree[parent] += shift
        if parent:
            self.__propagate(parent, shift)

    def __retrieve(self, level, p):
        left = 2 * level + 1
        if left >= len(self.tree):
            return level
        if p <= self.tree[left]:
            return self.__retrieve(left, p)
        return self.__retrieve(left + 1, p - self.tree[left])

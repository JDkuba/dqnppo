import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch
from .buffer import SimpleBuffer
from agents.dqn.network_models import default_network


class DQN:
    def __init__(self, action_space, obs_dim, gamma=0.99, epsilon=0.1,
                 create_network=lambda inp, out: default_network(inp, out)):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space
        self.experience = SimpleBuffer()

        self.neural_network = create_network(obs_dim, action_space.n)

        self.optimizer = Adam(self.neural_network.parameters())
        self.get_loss = nn.MSELoss()

    def sample_batch(self, sample_size):
        batch = self.experience.sample_batch(sample_size)
        states, actions, rewards, next_states, dones = (torch.Tensor(obs) for obs in zip(*batch))
        return states, actions.long(), rewards, next_states, dones

    def q_values(self, states, actions, rewards, next_states, dones):
        q_values = self.neural_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        max_q_next = self.neural_network(next_states).max(dim=1).values
        q_expected = rewards + (1 - dones) * self.gamma * max_q_next
        return q_values, q_expected

    def learn(self, sample_size, _):
        if sample_size > self.experience.length():
            return "Not learning yet"

        states, actions, rewards, next_states, dones = self.sample_batch(sample_size)
        q_values, q_expected = self.q_values(states, actions, rewards, next_states, dones)
        loss = self.get_loss(q_values, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def update_experience(self, state, action, reward, next_state, done, _):
        self.experience.update_memory((state, action, reward, next_state, done))

    def __predict(self, tensor_state):
        with torch.no_grad():
            return self.neural_network(tensor_state)

    def action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample(), None

        state = torch.Tensor(state)
        q_value = self.__predict(state)
        action = q_value.argmax().item()
        return action, None


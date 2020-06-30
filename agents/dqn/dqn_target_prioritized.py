import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch
from .buffer import PrioritizedBuffer
from agents.dqn.network_models import default_network


class DQNTargetPrioritized:
    def __init__(self, action_space, obs_dim, gamma=0.99, epsilon=0.1, tau=100, alpha=0.6,
                 create_network=lambda inp, out: default_network(inp, out)):
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.action_space = action_space
        self.experience = PrioritizedBuffer(alpha=alpha)

        self.neural_network = create_network(obs_dim, action_space.n)
        self.target_network = create_network(obs_dim, action_space.n)

        self.target_network.load_state_dict(self.neural_network.state_dict())
        self.optimizer = Adam(self.neural_network.parameters())
        self.get_loss = nn.MSELoss()

    def sample_batch(self, sample_size):
        batch, indices = self.experience.sample_batch(sample_size)
        states, actions, rewards, next_states, dones = (torch.Tensor(obs) for obs in zip(*batch))
        return states, actions.long(), rewards, next_states, dones, indices

    def q_values(self, states, actions, rewards, next_states, dones):
        q_values = self.neural_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        max_q_next = self.target_network(next_states).max(dim=1).values
        q_expected = rewards + (1 - dones) * self.gamma * max_q_next
        return q_values, q_expected

    def learn(self, sample_size, time):
        if sample_size > self.experience.length():
            return "Not learning yet"

        states, actions, rewards, next_states, dones, indices = self.sample_batch(sample_size)
        q_values, q_expected = self.q_values(states, actions, rewards, next_states, dones)

        errors = torch.abs(q_values - q_expected).detach().numpy()
        for i in range(sample_size):
            idx = indices[i]
            self.experience.update_priorities(idx, errors[i])

        loss = self.get_loss(q_values, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if time % self.tau == 0:
            self.update_target()
        return loss

    def update_experience(self, state, action, reward, next_state, done, _):
        new = reward + self.gamma * self.__predict(torch.Tensor(next_state)).argmax().item()
        old = self.__predict(torch.Tensor(state)).numpy()[action]
        error = np.abs(new - old)
        self.experience.update_memory((state, action, reward, next_state, done), error)

    def update_target(self):
        self.target_network.load_state_dict(self.neural_network.state_dict())

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

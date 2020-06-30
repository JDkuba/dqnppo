import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch

from agents.ppo.network_models import default_actor, default_critic


class Policy(nn.Module):
    def __init__(self, action_dim, obs_dim, create_actor, create_critic):
        super().__init__()
        self.actor = create_actor(obs_dim, action_dim)
        self.critic = create_critic(obs_dim)

    def __predict(self, tensor_state):
        with torch.no_grad():
            return self.actor(tensor_state)

    def action(self, tensor_state):
        q_value = self.__predict(tensor_state)
        distribution = Categorical(q_value)
        action = distribution.sample()

        logprob = distribution.log_prob(action)
        return action.item(), logprob

    def values(self, tensor_states, tensor_actions):
        q_value = self.actor(tensor_states)
        distribution = Categorical(q_value)

        logprobs = distribution.log_prob(tensor_actions)
        v_value = self.critic(tensor_states)
        return torch.squeeze(v_value), logprobs


class PPO:
    def __init__(self, action_space, obs_dim, gamma=0.99, epsilon=0.1, horizon_range=200,
                 create_actor=lambda inp, out: default_actor(inp, out), create_critic=lambda inp: default_critic(inp)):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon_range = horizon_range
        self.experience = []

        self.policy = Policy(action_space.n, obs_dim, create_actor, create_critic)
        self.old_policy = Policy(action_space.n, obs_dim, create_actor, create_critic)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = Adam(self.policy.parameters())
        self.get_loss = nn.MSELoss()

    def update_experience(self, state, action, reward, _, done, logprob):
        self.experience.append((state, action, reward, done, logprob))

    @staticmethod
    def normalize(vector):
        return (vector - vector.mean()) / vector.std()

    def __loss(self, tensor_rewards, ratio, v_value):
        advantage = tensor_rewards - v_value.detach()
        simple_surr = ratio * advantage
        clamp_surr = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        return -torch.min(simple_surr, clamp_surr) + 0.5 * self.get_loss(v_value, tensor_rewards)

    def learn(self, batch_size, time):
        if time % self.horizon_range == 0:
            self.step(batch_size)
            self.clear_experience()

    def step(self, batch_size):
        states, actions, rewards, dones, logprobs = (torch.Tensor(obs) for obs in zip(*self.experience))

        discounted_rewards = []
        discounted = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted = 0
            discounted = reward + self.gamma * discounted
            discounted_rewards.append(discounted)

        discounted_rewards = self.normalize(reversed(torch.Tensor(discounted_rewards)))

        for _ in range(batch_size):
            v_value, new_logprobs = self.policy.values(states, actions)
            ratio = torch.exp(new_logprobs - logprobs.detach())

            loss = self.__loss(discounted_rewards, ratio, v_value)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

    def action(self, state):
        return self.old_policy.action(torch.Tensor(state))

    def clear_experience(self):
        del self.experience[:]

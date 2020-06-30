import gym
import matplotlib.pyplot as plt

from agents.dqn.dqn import DQN
from agents.dqn.dqn_prioritized import DQNPrioritized
from agents.dqn.dqn_target_prioritized import DQNTargetPrioritized
from utils import save_file, plot_directory
from play import play
from agents.ppo.ppo import PPO
from agents.dqn.dqn_target import DQNTarget
from agents.dqn.network_models import *
from agents.ppo.network_models import *

env_name = "CartPole-v0"
env = gym.make(env_name)
if not isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) > 1 or \
        not isinstance(env.action_space, gym.spaces.Discrete):
    raise NotImplementedError

no_games = 300

agent = DQN(action_space=env.action_space, obs_dim=env.observation_space.shape[0])
mean_rewards = play(agent, no_games, env, batch_size=32)
save_file('./saved_games/dqn', env_name, mean_rewards)
print()

agent = DQNTarget(action_space=env.action_space, obs_dim=env.observation_space.shape[0], create_network=nn_dense_32_32)
mean_rewards = play(agent, no_games, env, batch_size=32)
save_file('./saved_games/ddqn', env_name, mean_rewards)
print()

agent = DQNPrioritized(action_space=env.action_space, obs_dim=env.observation_space.shape[0])
mean_rewards = play(agent, no_games, env, batch_size=32)
save_file('./saved_games/ddqn_priori', env_name, mean_rewards)
print()

agent = PPO(action_space=env.action_space, obs_dim=env.observation_space.shape[0], horizon_range=750)
mean_rewards = play(agent, no_games, env, batch_size=32)
save_file('./saved_games/ppo', env_name, mean_rewards)
print()

agent = PPO(action_space=env.action_space, obs_dim=env.observation_space.shape[0], horizon_range=1500,
            create_critic=simple_critic, create_actor=simple_actor)
mean_rewards = play(agent, no_games, env, batch_size=32)
save_file('./saved_games/ppo_two_hidden', env_name, mean_rewards)

directories = [('dqn', 'DQN'), ('ddqn', 'DDQN'), ('ddqn_priori', 'DDQN priori.'), ('ppo', 'PPO'),
               ('ppo_two_hidden', 'PPO two hidden layers')]
ax = plot_directory(path='./saved_games', env_name=env_name, directories=directories, ci=None)

plt.show()

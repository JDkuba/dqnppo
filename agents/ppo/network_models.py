import torch.nn as nn


def simple_actor(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, output_dim),
        nn.Softmax(dim=-1)
    )


def simple_critic(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )


def default_actor(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.Tanh(),
        nn.Linear(128, output_dim),
        nn.Softmax(dim=-1)
    )


def default_critic(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
    )

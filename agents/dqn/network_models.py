import torch.nn as nn


def default_network(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 128),
        nn.LeakyReLU(),
        nn.Linear(128, output_dim)
    )


def nn_dense_32_32(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 32),
        nn.LeakyReLU(),
        nn.Linear(32, output_dim)
    )

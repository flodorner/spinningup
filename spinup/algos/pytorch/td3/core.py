import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class split_model(nn.Module):
    def __init__(self,sizes, activation, output_activation=nn.Identity,split_index=-1):
        super(split_model, self).__init__()
        self.mlp1 = mlp(sizes, activation, output_activation)
        self.mlp2 = mlp(sizes, activation, output_activation)
        self.split_index = split_index
    def forward(self, x):
        if len(list(x.size()))==1:
            x = x.unsqueeze(0)
        split = torch.eq(x[:,self.split_index],1).unsqueeze(1)
        x = split.float()*self.mlp1(x)+torch.logical_not(split).float()*self.mlp2(x)
        return x

def mlp_switch(sizes, activation, output_activation=nn.Identity,split_index=-1):
    return split_model(sizes, activation, output_activation,split_index=split_index)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit,use_split=False):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        if not use_split:
            self.pi = mlp(pi_sizes, activation, nn.Tanh)
        else:
            self.pi = mlp_switch(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,use_split=False):
        super().__init__()
        if not use_split:
            self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        else:
            self.q = mlp_switch([obs_dim + act_dim] + list(hidden_sizes) + [1], activation,split_index=-1-act_dim)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

class MLPActorCriticSplit(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit,use_split=True)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation,use_split=True)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation,use_split=True)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

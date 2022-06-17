import numpy as np
import torch
import torch.nn as nn

from src.a2c_ppo_acktr.agents.distributions import FixedBernoulli, FixedCategorical, FixedDiagGaussian
from .base.cnn_base import CNNBase
from .base.mlp_base import MLPBase

device = torch.device("cuda:0")


class Policy(nn.Module):

    def __init__(self, policy, obs_shape, action_space, base=None, base_kwargs=None) -> object:
        super(Policy, self).__init__()
        base = CNNBase

        self.base = base(config, obs_shape[0], **base_kwargs)

        num_outputs = action_space.n
        self.dist = FixedCategorical(self.base.output_size, num_outputs)

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, trace, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, trace, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, trace, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, trace, rnn_hxs, masks):
        value, _, _ = self.base(inputs, trace, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, trace, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, trace, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)

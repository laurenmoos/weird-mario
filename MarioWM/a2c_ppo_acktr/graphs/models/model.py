import numpy as np
import torch
import torch.nn as nn

from ...agents.distributions import FixedBernoulli, FixedCategorical, FixedDiagGaussian
from .base.cnn_base import CNNBase
from .base.mlp_base import MLPBase

device = torch.device("cuda:0")


class Policy(nn.Module):

    # TODO: probably should just always pass base if we can
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None) -> object:
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = FixedCategorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = FixedDiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = FixedBernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, trace, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, trace, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, trace, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
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
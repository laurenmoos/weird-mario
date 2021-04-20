import torch.nn as nn

import torch

from .LSTM import LSTM
from ....utils.train_utils import init
from .nn_base import NNBase


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(NNBase):

    def __init__(self, config, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.model = config.base_model

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        if self.model != 0:
            self.lstm = LSTM()

        if self.model != 1:
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
                init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        if self.model == 2:
            self.blend = init_(nn.Linear(hidden_size * 2, hidden_size))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, obs, trace, rnn_hxs, masks):

        if self.model == 0:
            x = self.main(obs / 255.0)
        elif self.model == 1:
            x = trace.clone().detach().long()
            x = self.lstm(x)
        elif self.model == 2:
            z = trace.clone().detach().long()
            z = self.lstm(z)
            x = self.main(obs / 255.0)
            x = torch.cat((x, z), 1)
            x = self.blend(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

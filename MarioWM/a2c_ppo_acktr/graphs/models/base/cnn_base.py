import torch.nn as nn

import torch

import LSTM
from nn_base import NNBase


class CNNBase(NNBase):

    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        if args.model != 0:
            self.lstm = LSTM()

        if args.model != 1:
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
                init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        if args.model == 2:
            self.blend = init_(nn.Linear(hidden_size * 2, hidden_size))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, obs, trace, rnn_hxs, masks):

        if args.model == 0:
            x = self.main(obs / 255.0)
        elif args.model == 1:
            x = trace.clone().detach().long()
            x = self.lstm(x)
        elif args.model == 2:
            z = trace.clone().detach().long()
            z = self.lstm(z)
            x = self.main(obs / 255.0)
            x = torch.cat((x, z), 1)
            x = self.blend(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

    """
    #########################
    Architecture:
    #########################
    Input: (N, 3, 40, 80)
    Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))
    Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2))
    Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
    Linear(in_features=448, out_features=2, bias=True)
    ----
    torch.Size([128, 3, 40, 80])
    torch.Size([128, 16, 18, 38])
    torch.Size([128, 32, 7, 17])
    torch.Size([128, 32, 2, 7]) --> 32x2x7 = 448
    torch.Size([128, 2])
    """

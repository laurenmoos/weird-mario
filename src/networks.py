import torch.nn as nn

import torch

from .LSTM import LSTM
from ....utils.train_utils import init
from .nn_base import NNBase

class LSTM(nn.Module):

    def __init__(self, embedding_dim=100, hidden_dim=512, vocab_size=50000):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        embeds = torch.transpose(embeds, 0, 1)
        lstm_out, h = self.lstm(embeds)
        return lstm_out[-1]

class GamePhysicsModel(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout, batch_first):
        super(GamePhysicsModel, self).__init__(hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        if self.model == 2:
            self.blend = init_(nn.Linear(hidden_size * 2, hidden_size))
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_in
        puts, 32, 8, stride = 4)), nn.ReLU(),
                                   init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                                   init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
                                   init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,  num_layers=num_layers, dropout=dropout, batch_first=True)
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, obs, trace, rnn_hxs, masks):
        z = trace.clone().detach().long()
        z = self.lstm(z)
        x = self.main(obs / 255.0)
        x = torch.cat((x, z), 1)
        x = self.blend(x)

        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        #TODO: I don't think the critic network should be here
        return self.critic_linear(x), x, rnn_hxs











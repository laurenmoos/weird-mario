import torch
import torch.nn as nn

'''
Long-Short Term Memory Base Network
'''


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

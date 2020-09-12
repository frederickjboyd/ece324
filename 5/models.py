import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0)  # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)

        # Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return output


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.convA = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(2, embedding_dim))
        self.convB = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(4, embedding_dim))
        self.poolA = nn.MaxPool2d(kernel_size=(1, 1))
        self.poolB = nn.MaxPool2d(kernel_size=())
        self.fc1 = nn.Linear(100, 1)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        embedded = embedded.unsqueeze(1)
        x1 = F.relu(self.convA(embedded))
        x2 = F.relu(self.convB(embedded))
        self.poolA = nn.MaxPool2d(kernel_size=(x1.shape[2], 1))
        self.poolB = nn.MaxPool2d(kernel_size=(x2.shape[2], 1))
        x1 = self.poolA(x1)
        x2 = self.poolB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = x.squeeze(3)
        x = x.squeeze(2)
        x = torch.sigmoid(self.fc1(x))
        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths=lengths)
        _, x = self.rnn(x)
        x = F.sigmoid(self.fc1(x))

        return x

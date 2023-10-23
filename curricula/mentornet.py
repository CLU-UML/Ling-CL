import torch
import torch.nn as nn

class MentorNet(nn.Module):
    def __init__(self, num_labels, num_epochs,
            decay = 0.9, percentile = 0.7,
            label_embed_dim = 2, epoch_embed_dim = 5,
            fc_dim = 20, lstm_dim = 1):
        super().__init__()

        self.register_buffer('avg', None)
        self.decay = decay
        self.percentile = percentile

        self.lstm = nn.LSTM(2, lstm_dim, batch_first = True, bidirectional = True)

        self.label_embed = nn.Embedding(num_labels, label_embed_dim)
        self.epoch_embed = nn.Embedding(num_epochs, epoch_embed_dim)
        self.epoch_embed.weight.requires_grad = False

        self.fc = nn.Sequential(
                nn.Linear(2*lstm_dim + label_embed_dim + epoch_embed_dim, fc_dim),
                nn.Tanh(),
                nn.Linear(fc_dim, 1),
                nn.Sigmoid()
                )
                  
    def forward(self, loss, labels, epoch, *args):
        with torch.no_grad():
            if self.avg is None:
                self.avg = torch.quantile(loss, self.percentile)
            else:
                self.avg = self.decay * self.avg + (1 - self.decay) * torch.quantile(loss, self.percentile)

        lossdiff = loss - self.avg
        
        lstm_input = torch.stack([loss, lossdiff], 1).unsqueeze(1)

        _, (h, _) = self.lstm(lstm_input)
        h = torch.cat([d for d in h], -1)

        epochs = torch.ones_like(loss).long() * epoch
        epoch_embed = self.epoch_embed(epochs)
        label_embed = self.label_embed(labels)

        feats = torch.cat([h, epoch_embed, label_embed], 1)

        confs = self.fc(feats)

        return confs

from typing import Dict

import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout

class SeqClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm = nn.LSTM(input_size = embeddings.size(1),
                    hidden_size = hidden_size,
                    num_layers = num_layers,
                    dropout = dropout,
                    bidirectional = bidirectional,
                    batch_first = True
                   )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return num_class

    def forward(self, data) -> Dict[str, torch.Tensor]:
        #print('data', data.size())
        X = self.embed(data)
        #print('Embed', X[0][0])
        out, H = self.lstm(X)
        #print('lstm', out.size())
        out = out[:, 0,:]
        #print('out', out.size())
        pred = self.fc(out)
        #print('fc', pred.size())
        return pred


class TagClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(TagClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        
        self.lstm = nn.LSTM(input_size = embeddings.size(1),
                   hidden_size = hidden_size,
                   num_layers = num_layers,
                   dropout = dropout,
                   bidirectional = bidirectional,
                   batch_first = True,
                  )

        self.fc = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return num_class

    def forward(self, data) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #print('data', data.size())
        X = self.embed(data)
        #print('Embed', X.size())
        out, (H, C) = self.lstm(X)
        #print('out', out.size())
        pred = self.fc(out)
        #print('fc', pred.size())
        return pred.permute(0, 2, 1)
    
    
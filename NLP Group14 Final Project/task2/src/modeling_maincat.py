
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModel

from model_output import SequenceClassifierOutput

class BertForMultiLabelMultiClassClassification(nn.Module):
    def __init__(self, model_name_or_path, config, ignore_mismatched_sizes, num_labels=3, num_classes=1):
        super(BertForMultiLabelMultiClassClassification, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.loss_weights = torch.Tensor([1.0, 1.5, 1.5, 1.0]).to(self.device)
        self.num_bert = 1
        self.dropout = 0.3

        self.bert = nn.ModuleList(
            [AutoModel.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            ) for _ in range(self.num_bert)]
        )

        self.shared_classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(768*self.num_bert, 768*self.num_bert, bias=True),
            nn.Tanh(),
        ) 
        self.classifier = nn.ModuleList(
            [nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(768*self.num_bert, num_labels, bias=True)
            ) for _ in range(num_classes)]
        )

    def forward(self, **kwargs):

        # print("kwargs: ", kwargs)
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        token_type_ids = kwargs["token_type_ids"]
        all_labels = kwargs["labels"]

        outputs_bert = [self.bert[i](input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) for i in range(self.num_bert)]
        logits_bert = [output.last_hidden_state[:, 0, :] for output in outputs_bert]

        # logits_concat = torch.concat((logits_a, logits_b), dim=1)
        logits_concat = torch.concat(tuple(logits_bert), dim=1)
        logits_shared = self.shared_classifier(logits_concat)
        all_logits = [classifier(logits_shared) for classifier in self.classifier]
        all_logits = [classifier(logits_concat) for classifier in self.classifier]
        
        total_loss: torch.Tensor = torch.Tensor().to(self.device)
        for idx_class in range(self.num_classes):
            logits = all_logits[idx_class]
            labels = all_labels[:,idx_class]
            # print(logits.shape, labels.shape)
            loss = None
            if labels is not None:

                # Single Label Classification
                loss_fct = CrossEntropyLoss(ignore_index=0)
                loss: torch.Tensor = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if torch.isnan(loss):
                    loss = torch.tensor(0).to(self.device)
                # print(loss)
                total_loss = torch.cat((total_loss, loss.unsqueeze(-1)))

        # print(total_loss, total_loss.mean())

        return SequenceClassifierOutput(
            loss=total_loss.mean(),
            logits=all_logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
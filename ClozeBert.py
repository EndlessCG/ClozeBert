import math

import torch
from torch import nn
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import AlbertConfig, AlbertModel

# from bert_basics import *
from transformers.models.albert.modeling_albert import AlbertMLMHead

from ClozeBert_utils import bert_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BERT_LM_LOSS = False


class ClozeBertModelForTransformers(nn.Module):
    def __init__(self, config):
        # self.bert = BertForMaskedLM(self.config)
        super(ClozeBertModelForTransformers, self).__init__()
        self.bert = AlbertModel.from_pretrained(bert_path)
        self.cls = AlbertMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()

    def accuracy(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).sum().float() / out.shape[0]

    def forward(self, input_ids, masks, token_type_ids, mask_ids, options, answers):
        output = self.bert(input_ids, attention_mask=masks, token_type_ids=token_type_ids)
        sequence_output = output[0]
        logits = self.cls(sequence_output)
        num_sentence = mask_ids.shape[2]
        mask_output = logits[0, mask_ids.squeeze(0).squeeze(1), :].squeeze(0)
        option_indices = torch.arange(0, num_sentence, device=DEVICE).repeat(4, 1).T.reshape(-1)

        # @TODO: 1. build a better tokenizer 2. fix here
        try:
            option_vals = mask_output[option_indices, options.reshape(-1)].reshape(-1, 4)
        except IndexError:
            pass

        option_opts = torch.argmax(option_vals, dim=1)
        loss = self.criterion(option_vals, answers.squeeze(0))

        if USE_BERT_LM_LOSS:
            return logits, option_opts, output.loss
        else:
            return logits, option_opts, loss

    def summary_net(self):
        writer = SummaryWriter()
        writer.add_graph(self, [torch.zeros([20, 40]), torch.zeros([20, 40]), torch.zeros([20, 40]), torch.zeros([20, 4]),
                                torch.zeros([20, 1]), torch.zeros([20, 1]), torch.zeros(20, 40)])
        writer.close()

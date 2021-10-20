import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AlbertModel, BertForMaskedLM, AutoModelForMaskedLM
from transformers.models.albert.modeling_albert import AlbertMLMHead

from ClozeBert_utils import bert_path, tokenizer  # for debug use

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BERT_LM_LOSS = False


class ClozeBertModelForTransformers(nn.Module):
    def __init__(self, config):
        super(ClozeBertModelForTransformers, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        # self.bert = AlbertModel.from_pretrained(bert_path)
        # self.cls = AlbertMLMHead(config)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def accuracy(self, out, tgt):
        out = torch.argmax(out, -1)
        return (out == tgt).sum().float() / out.shape[0]

    def forward(self, input_ids, masks, token_type_ids, mask_ids, options, answers):
        # output = self.bert(input_ids, attention_mask=masks, token_type_ids=token_type_ids)
        # sequence_output = output[0]
        # logits = self.cls(sequence_output)
        logits = self.bert(input_ids, masks, token_type_ids).logits
        num_question = answers.shape[1]
        mask_ids = torch.nonzero(mask_ids != -100)[:, 2].to(DEVICE)
        mask_output = logits[0, mask_ids, :].squeeze(0)
        option_indices = torch.arange(0, num_question, device=DEVICE).repeat(4, 1).T.reshape(-1)
        option_vals = mask_output[option_indices, options.reshape(-1)].reshape(-1, 4)

        option_opts = torch.argmax(option_vals, dim=1)
        loss = self.criterion(option_vals, answers.squeeze(0)).sum()
        return logits, option_opts, loss

    def summary_net(self):
        writer = SummaryWriter()
        writer.add_graph(self,
                         [torch.zeros([20, 40]), torch.zeros([20, 40]), torch.zeros([20, 40]), torch.zeros([20, 4]),
                          torch.zeros([20, 1]), torch.zeros([20, 1]), torch.zeros(20, 40)])
        writer.close()

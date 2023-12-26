import torch
from typing import Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm
from transformers import AlbertForQuestionAnswering
from transformers import AlbertModel

def load_albertQA():
    return AlbertForQuestionAnswering.from_pretrained("twmkn9/albert-base-v2-squad2")


class AlbertSquad(torch.nn.Module):
    def __init__(self, hidden_size=128, nhead=8):
        super().__init__()
        self.albert_base = AlbertModel.from_pretrained("albert-base-v2")
        self.albert_base.eval()
        self.bigru_encoder_context = torch.nn.GRU(768, hidden_size, batch_first=True, bidirectional=True)  # output size is 2 * hidden_size
        self.bigru_encoder_query = torch.nn.GRU(768, hidden_size, batch_first=True, bidirectional=True)  # output size is 2 * hidden_size

        self.query2context = torch.nn.TransformerDecoderLayer(2 * hidden_size, nhead, batch_first=True)

        self.bigru_decoder = torch.nn.GRU(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True) # output size is 2 * hidden_size

        self.linear_attention = torch.nn.Linear(2 * hidden_size, 1)
        self.linear_decoder = torch.nn.Linear(2 * hidden_size, 1)
        self.bigru_end = torch.nn.GRU(2 * hidden_size, 1, batch_first=True)

    def forward(self, context, query):
        with torch.no_grad():
            context = self.albert_base(**context)
            query = self.albert_base(**query)

        context = self.bigru_encoder_context(context)
        query = self.bigru_encoder_query(query)

        attention = self.query2context(context, query)
        decoded_attention = self.bigru_decoder(attention)

        linear_attention = self.linear_attention(attention).squeeze(dim=-1)
        linear_decoder = self.linear_decoder(decoded_attention).squeeze(dim=-1)
        bigru_end = self.bigru_end(decoded_attention).squeeze(dim=-1)

        start_logits = linear_attention + linear_decoder
        end_logits = linear_attention + bigru_end

        return (start_logits, end_logits)
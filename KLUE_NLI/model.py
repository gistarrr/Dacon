import torch
import torch.nn as nn

from transformers import RobertaPreTrainedModel, RobertaModel

class ExplainableModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        print("######### Loading Custom Model #########")
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.span_info_collect = SICModel(config.hidden_size)
        self.interpretation = InterpretationModel(config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, start_indexs=None, end_indexs=None, span_masks=None, labels=None, length=None):
        # generate mask
        attention_mask = (input_ids != 1).long()
        # intermediate layer
        hidden_states = self.roberta(input_ids, attention_mask=attention_mask)[0]  # output.shape = (bs, length, hidden_size)
        hidden_states = self.dropout(hidden_states)
        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(hidden_states, start_indexs, end_indexs)
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, span_masks)
        # output layer
        logits = self.classifier(self.dropout(H))
        return logits, a_ij

class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indexs)  # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indexs)

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        h_ij = torch.tanh(span)
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1)  # (ba, span_num)
        # mask illegal span
        o_ij = o_ij - span_masks
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # weight average span representation to get H
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, hidden_size)
        return H, a_ij

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

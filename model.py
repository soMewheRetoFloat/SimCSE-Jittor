import jittor
import jittor.nn as nn
from bert_jt import BertModel, BertBase

        
def j_cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    numerator = (x1 * x2).sum(dim=dim)
    x1_norm = jittor.sqrt((x1 ** 2).sum(dim=dim))
    x2_norm = jittor.sqrt((x2 ** 2).sum(dim=dim))
    return numerator / ((x1_norm * x2_norm) + eps)


class SimcseModel(BertBase):
    def __init__(self, config, pooling, dropout=0.3):
        super(SimcseModel, self).__init__(config)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel(config)
        self.pooling = pooling
        self.init_weights()

    def execute(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        # return out[1]
        if self.pooling == 'cls':
            return out["last_hidden_state"][:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out["pooler_output"]  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out["last_hidden_state"].transpose(1, 2)  # [batch, 768, seqlen]
            return jittor.squeeze(nn.pool(last, kernel_size=last.shape[-1], op="avg"), -1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out["hidden_states"][1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out["hidden_states"][-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = jittor.squeeze(nn.pool(first, kernel_size=last.shape[-1], op="avg"), -1)  # [batch, 768]
            last_avg = jittor.squeeze(nn.pool(last, kernel_size=last.shape[-1], op="avg"), -1)  # [batch, 768]
            two_avg = jittor.concat((jittor.unsqueeze(first_avg, 1), jittor.unsqueeze(last_avg, 1)), dim=1)  # [batch, 2, 768]
            return jittor.squeeze(nn.pool(two_avg.transpose(1, 2), kernel_size=2, op="avg"), -1)  # [batch, 768]


def simcse_unsup_loss(y_pred, temp=0.05):
    y_true = jittor.arange(y_pred.shape[0])
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = j_cosine_similarity(jittor.unsqueeze(y_pred, 1), jittor.unsqueeze(y_pred, 0), dim=-1)
    sim = sim - jittor.init.eye(y_pred.shape[0]) * 1e12
    sim = sim / temp
    loss = nn.cross_entropy_loss(sim, y_true)
    return jittor.mean(loss)


def simcse_sup_loss(y_pred, lamda=0.05):
    similarities = j_cosine_similarity(jittor.unsqueeze(y_pred, 0), jittor.unsqueeze(y_pred, 1), dim=2)
    row = jittor.arange(0, y_pred.shape[0], 3)
    col = jittor.arange(0, y_pred.shape[0])
    col = col[col % 3 != 0]
    similarities = similarities[row, :]
    similarities = similarities[:, col]
    similarities = similarities / lamda
    y_true = jittor.arange(0, len(col), 2)
    loss = nn.cross_entropy_loss(similarities, y_true)
    return loss

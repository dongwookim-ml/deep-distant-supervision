import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging
import numpy as np
from tqdm import tqdm
from dds import data_fetcher

# Logger
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger = logging.getLogger()
logger.addHandler(ch)
logger.setLevel('INFO')


def new_parameter(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal(out)
    return out


class SentenceAttention(nn.Module):
    def __init__(self, input_dim):
        super(SentenceAttention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, hidden_dim) * (hidden_dim, 1) -> (1, hidden_dim)
        attn = torch.matmul(input, self.w1)
        norm_attn = F.softmax(attn, 0)  # (batch_size)
        weighted = torch.mul(input, norm_attn.unsqueeze(-1).expand_as(input))
        summary = weighted.sum(0).squeeze()
        return summary


class WordAttention(nn.Module):
    """
    Simple attention layer
    """

    def __init__(self, hidden_dim):
        super(WordAttention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_dim, requires_grad=True))

    def forward(self, input, seq_len):
        # shape (batch, seq_len, hidden_dim) * (hidden_dim, 1) -> (batch, hidden_dim)
        # attention with masked softmax
        attn = torch.einsum('ijk,k->ij', [input, self.w1])
        attn_max, _ = torch.max(attn, dim=1, keepdim=True)
        attn_exp = torch.exp(attn - attn_max)
        attn_exp = attn_exp * (attn != 0).float()
        norm_attn = attn_exp / (torch.sum(attn_exp, dim=1, keepdim=True))
        summary = torch.einsum("ijk,ij->ik", [input, norm_attn])
        return summary


class DDS(nn.Module):
    """
    Deep distant supervision model
    """

    def __init__(self, embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, embed=None,
                 dropout_prob=0.5, activation_fn=nn.ReLU):
        logger.info('Initialize DDS model')
        super(DDS, self).__init__()

        if embed is not None and type(embed) == np.ndarray:
            embed = torch.from_numpy(embed)

        if embed is not None:
            self.w2v = nn.Embedding(num_embeddings=num_voca, embedding_dim=embed_dim, _weight=embed)
        else:
            self.w2v = nn.Embedding(num_embeddings=num_voca, embedding_dim=embed_dim)

        self.pos1vec = nn.Embedding(num_voca, pos_dim)
        self.pos2vec = nn.Embedding(num_voca, pos_dim)
        self.input_dim = embed_dim + 2 * pos_dim
        self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout_prob)
        self.word_attn = WordAttention(hidden_dim * 2)
        self.sen_attn = SentenceAttention(hidden_dim * 2)
        self.inter_linear = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.output_linear = nn.Linear(hidden_dim, num_relations, bias=True)
        self.activation_fn = activation_fn()
        logger.info('Done')

    def forward(self, x):
        x = sorted(x, key=lambda x: len(x[0]))
        x.reverse()  # the longest sequence should be placed in the first of the list

        seq_len = list()
        max_seq = len(x[0][0])
        batch_in = torch.zeros((len(x), max_seq, self.input_dim))
        for i, (sen, pos1, pos2) in enumerate(x):
            seq_len.append(len(sen))
            _word = self.w2v(torch.from_numpy(sen))
            _pos1 = self.pos1vec(torch.from_numpy(pos1))
            _pos2 = self.pos2vec(torch.from_numpy(pos2))
            combined = torch.cat((_word, _pos1, _pos2), 1)
            batch_in[i, :len(sen)] = combined

        packed_batch_in = nn.utils.rnn.pack_padded_sequence(batch_in, seq_len, batch_first=True)
        rnn_output, _ = self.rnn(packed_batch_in)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        sen_embeds = self.word_attn(unpacked, seq_len)
        pair_embeds = self.sen_attn(sen_embeds)
        hidden_output = self.activation_fn(self.inter_linear(pair_embeds))
        return self.output_linear(hidden_output)


def evaluation(all_prob, target_y):
    target_prob = np.reshape(all_prob[:, 1:], (-1))  # note that the relation of the first column is NA
    target_y = np.array(target_y)
    target_y = np.reshape(target_y[:, 1:], (-1))
    ordered_idx = np.argsort(-target_prob)
    top_n = [100, 200, 300]
    prec_at_n = np.zeros(len(top_n))
    for k, top_k in enumerate(top_n):
        prec_at_n[k] = np.sum(target_y[ordered_idx][:top_k], dtype=float) / float(top_k)
        logger.info("Precision @ %d: %f", top_k, prec_at_n[k])

    roc_auc = roc_auc_score(target_y, target_prob)
    logger.info("ROC-AUC score: %f", roc_auc)
    ap = average_precision_score(target_y, target_prob)
    logger.info("Average Precision: %f", ap)


def test(test_data, model, loss_fn):
    logger.info('Validation ...')
    all_y = list()
    all_predicted_y = list()
    loss_sum = 0
    for x, y in test_data:
        output = model(x)
        predicted_y = output.data.numpy()
        _y = torch.from_numpy(y).float()
        loss_sum += loss_fn(output, _y)
        all_y.append(y)
        all_predicted_y.append(predicted_y)
    logger.info("Loss sum : %f", loss_sum)
    evaluation(np.array(all_predicted_y), np.array(all_y))
    logger.info('Done ...')


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score, average_precision_score

    logger.info('Loading word2vec')
    word2id, embedding = data_fetcher.load_w2v('../../data/word2vec.txt')
    logger.info('... Done')
    logger.info('Loading relations')
    rel2id = data_fetcher.load_relations('../../data/nyt/relation2id.txt', True)
    logger.info('... Done')
    logger.info('Loading dataset')
    triple, sen_col = data_fetcher.loadnyt('../../data/nyt/train.txt', word2id)
    logger.info('... Done')

    embed_dim = 50
    hidden_dim = 128
    num_valid = 500
    valid_cycle = 1000
    num_layers = 2
    seq_len = 70
    pos_dim = 5
    num_epoch = 3
    num_voca = len(word2id)
    num_relations = len(rel2id)

    model = DDS(embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('Param %s : %s', name, param.shape)

    for epoch in range(num_epoch):
        fetcher = data_fetcher.fetch_sentences_nyt(triple, sen_col, rel2id)

        valid_data = list()
        for i in range(num_valid):
            x, y = next(fetcher)
            valid_data.append((x, y))

        for i, (x, y) in enumerate(tqdm(fetcher, initial=epoch*len(triple), total=(len(triple)-num_valid)*num_epoch)):
            _y = torch.from_numpy(y).float()

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, _y)
            loss.backward()
            optimizer.step()

            logger.debug('%d Done, loss = %f', i, loss)

            if i % valid_cycle == 0:
                test(valid_data, model, loss_fn)

    logger.info('Loading test dataset')
    test_triple, test_sen_col = data_fetcher.loadnyt('../../data/nyt/test.txt', word2id)
    logger.info('... Done')

    test_fetcher = data_fetcher.fetch_sentences_nyt(test_triple, test_sen_col, rel2id)
    test(test_fetcher, model, loss_fn)

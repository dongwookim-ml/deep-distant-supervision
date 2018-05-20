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


class SentenceAttention(nn.Module):
    def __init__(self, input_dim):
        super(SentenceAttention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, hidden_dim) * (hidden_dim, 1) -> (1, hidden_dim)
        attn = torch.einsum('ik,k->i', [input, self.w1])
        norm_attn = F.softmax(attn).clone()
        summary = torch.einsum("ik,i->k", [input, norm_attn])
        return summary


class WordAttention(nn.Module):
    """
    Simple attention layer
    """

    def __init__(self, input_dim):
        super(WordAttention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, seq_len, hidden_dim) * (hidden_dim, 1) -> (batch, hidden_dim)
        attn = torch.einsum('ijk,k->ij', [input, self.w1])
        norm_attn = F.softmax(attn, 1).clone()
        summary = torch.einsum("ijk,ij->ik", [input, norm_attn])
        return summary


class DDS(nn.Module):
    """
    Deep distant supervision model
    """

    def __init__(self, embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, embed):
        logger.info('Initialize DDS model')
        super(DDS, self).__init__()
        if type(embed) == np.ndarray:
            embed = torch.from_numpy(embed)
        self.w2v = nn.Embedding(num_embeddings=num_voca, embedding_dim=embed_dim, _weight=embed)
        self.pos1vec = nn.Embedding(num_voca, pos_dim)
        self.pos2vec = nn.Embedding(num_voca, pos_dim)
        self.input_dim = embed_dim + 2 * pos_dim
        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True)
        self.word_attn = WordAttention(hidden_dim * 2)
        self.sen_attn = SentenceAttention(hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, num_relations, bias=True)
        logger.info('Done')

    def forward(self, x):
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

        batch_in = nn.utils.rnn.pack_padded_sequence(batch_in, seq_len, batch_first=True)
        rnn_output, _ = self.rnn(batch_in)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)
        sen_embeds = self.word_attn(unpacked)
        pair_embeds = self.sen_attn(sen_embeds)
        output = self.linear(pair_embeds)
        return output


if __name__ == '__main__':
    logger.info('Loading word2vec')
    word2id, embedding = data_fetcher.load_w2v('../../data/word2vec.txt')
    logger.info('... Done')
    logger.info('Loading relations')
    rel2id = data_fetcher.load_relations('../../data/nyt/relation2id.txt', True)
    logger.info('... Done')
    logger.info('Loading dataset')
    triple, sen_col = data_fetcher.loadnyt('../../data/nyt/train.txt', word2id)
    logger.info('... Done')
    fetcher = data_fetcher.fetch_sentences_nyt(triple, sen_col, rel2id)

    embed_dim = 50
    hidden_dim = 32
    num_layers = 2
    seq_len = 70
    pos_dim = 5
    num_voca = len(word2id)
    num_relations = len(rel2id)

    model = DDS(embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, embedding)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for i, (x, y) in enumerate(tqdm(fetcher, total=len(triple))):
        x = sorted(x, key=lambda x: len(x[0]))
        x.reverse()
        print([len(k[0]) for k in x])
        _y = torch.from_numpy(y).float()

        output = model(x)
        loss = loss_fn(output, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.debug('%d Done, loss = %f' % (i, loss))

        if i % 10000 == 0:
            logger.info('%d entity pairs are processed' % i)

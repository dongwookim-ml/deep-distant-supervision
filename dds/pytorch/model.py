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
logger.setLevel('DEBUG')


class Attention(nn.Module):
    """
    Simple attention layer
    """

    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, seq_len, hidden_dim) * (hidden_dim, 1) -> (batch, seq_len, 1)
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
        input_dim = embed_dim + 2 * pos_dim
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, num_relations, bias=True)
        logger.info('Done')

    def forward(self, sen, pos1, pos2):
        word_embedding = self.w2v(sen)
        pos1_embedding = self.pos1vec(pos1)
        pos2_embedding = self.pos2vec(pos2)

        combined = torch.cat((word_embedding, pos1_embedding, pos2_embedding), 1)
        unsqueezed = combined.unsqueeze(0)

        rnn_output, _ = self.rnn(unsqueezed)
        sen_embedding = self.attn(rnn_output)
        output = self.linear(sen_embedding)
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
        unsqueezed_y = torch.from_numpy(y).float().unsqueeze(0)
        for sentence in x:
            sen, pos1, pos2 = sentence
            sen = torch.from_numpy(sen)
            pos1 = torch.from_numpy(pos1)
            pos2 = torch.from_numpy(pos2)
            # feed each sentence to model
            output = model(sen, pos1, pos2)

            loss = loss_fn(output, unsqueezed_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 10000 == 0:
            logger.info('%d entity pairs are processed')

import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import logging
import numpy as np
from dds.data_fetcher import NYTFetcher

# Logger
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger = logging.getLogger()
logger.addHandler(ch)
logger.setLevel('INFO')

max_sen_len = 70  # Predefined maximum length of sentence, need for position embedding
max_sens = 100  # maximum number of sentences for an entity pair


def new_parameter(*size):
    out = nn.Parameter(FloatTensor(*size))
    torch.nn.init.xavier_normal(out)
    return out


class NYTData(torch.utils.data.Dataset):
    """
    use Dataset class to prefetch data into GPUs
    """

    def __init__(self, nyt_fetcher, device):
        """
        Initialise dataset
        :param nyt_fetcher: nyt dataset fetcher
        """
        self.data = [(xs, y) for (xs, y) in nyt_fetcher]
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = sorted(x, key=lambda x: len(x[0]))
        x.reverse()  # the longest sequence should be placed in the first of the list
        new_x = list()
        y = FloatTensor(y).to(self.device)
        for a, b, c in x:
            new_x.append((LongTensor(a).to(self.device),
                          LongTensor(b).to(self.device),
                          LongTensor(c).to(self.device)))
        return new_x, y


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

    def __init__(self, embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, device, embed=None,
                 dropout_prob=0.5, activation_fn=nn.ReLU):
        logger.info('Initialize DDS model')
        super(DDS, self).__init__()
        self.device = device

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
        seq_len = list()
        batch_in = list()
        for i, (sen, pos1, pos2) in enumerate(x):
            seq_len.append(len(sen))
            _word = self.w2v(sen.squeeze())
            _pos1 = self.pos1vec(pos1.squeeze())
            _pos2 = self.pos2vec(pos2.squeeze())
            combined = torch.cat((_word, _pos1, _pos2), 1)
            batch_in.append(combined)

        stacked_batch_in = nn.utils.rnn.pad_sequence(batch_in, batch_first=True)
        packed_batch_in = nn.utils.rnn.pack_padded_sequence(stacked_batch_in, seq_len, batch_first=True)
        rnn_output, _ = self.rnn(packed_batch_in)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        sen_embeds = self.word_attn(unpacked, seq_len)
        pair_embeds = self.sen_attn(sen_embeds)
        hidden_output = self.activation_fn(self.inter_linear(pair_embeds))
        return self.output_linear(hidden_output)


def evaluation(prob_y, target_y):
    target_prob = np.reshape(prob_y[:, 1:], (-1))  # note that the relation of the first column is NA
    target_y = np.array(target_y)
    target_y = np.reshape(target_y[:, 1:], (-1))
    ordered_idx = np.argsort(-target_prob)
    logger.info('Total validation count %d', np.sum(target_y))
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
        loss_sum += loss_fn(output.squeeze(), y.squeeze())
        all_y.append(y.squeeze().data.cpu().numpy())
        all_predicted_y.append(output.squeeze().data.cpu().numpy())
    logger.info("Loss sum : %f", loss_sum)
    evaluation(np.array(all_predicted_y), np.array(all_y))
    logger.info('Done ...')


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score, average_precision_score

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')

    print('Current device :', device)

    embed_dim = 50
    logger.info('Loading Fetcher')
    w2v_path = '../../data/word2vec.txt'
    rel_path = '../../data/nyt/relation2id.txt'
    sen_path = '../../data/nyt/train.txt'
    fetcher = NYTFetcher(w2v_path, rel_path, embed_dim, sen_path)

    hidden_dim = 128
    num_valid = 500
    valid_cycle = 10000
    num_layers = 2
    seq_len = 70
    pos_dim = 5
    num_epoch = 1
    batch_size = 1
    num_voca = len(fetcher.word2id)
    num_relations = len(fetcher.rel2id)

    model = DDS(embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, device,
                embed=fetcher.word_embedding)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('Param %s : %s', name, param.shape)

    nyt_dataset = NYTData(fetcher, device)
    dataset_loader = DataLoader(nyt_dataset, batch_size=1, shuffle=True, drop_last=True)

    for epoch in range(num_epoch):
        fetcher.reset()

        valid_data = list()
        for i, (x, y) in enumerate(dataset_loader):
            valid_data.append((x, y.squeeze()))
            if i == num_valid:
                break

        optimizer.zero_grad()
        for i, (x, y) in enumerate(tqdm(dataset_loader, initial=epoch * (len(nyt_dataset) - num_valid),
                                        total=(len(nyt_dataset) - num_valid) * num_epoch)):
            loss = loss_fn(model(x), y.squeeze()) / batch_size
            loss.backward()
            if i % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.debug('%d Done, loss = %f', i, loss)

            if i % valid_cycle == 0 and i != 0:
                test(valid_data, model, loss_fn)

        torch.save(model.state_dict(), 'saved_model.tmp')

    logger.info('Save Model ...')
    torch.save(model.state_dict(), 'saved_model.tmp')
    logger.info('Done')

    test_path = '../../data/nyt/test.txt'
    test_fetcher = NYTFetcher(w2v_path, rel_path, embed_dim, test_path)
    test(DataLoader(NYTData(test_fetcher, device), batch_size=1), model, loss_fn)

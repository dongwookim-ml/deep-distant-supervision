import sys
import logging
from tqdm import tqdm
from dds.pytorch.model import DDS, test
from dds.data_fetcher import FreebaseFetcher
import torch
import torch.nn as nn

# Logger
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger = logging.getLogger()
logger.addHandler(ch)
logger.setLevel('INFO')

if __name__ == '__main__':
    embed_dim = 50
    logger.info('Loading Fetcher')
    w2v_path = '../../data/word2vec.txt'
    rel_path = '../../data/freebase/relations.txt'
    fetcher = FreebaseFetcher(w2v_path, rel_path, embed_dim, include_rel_id=False)

    hidden_dim = 256
    num_valid = 5000
    valid_cycle = 100000
    num_layers = 2
    seq_len = 70
    pos_dim = 5
    num_epoch = 3
    batch_size = 4
    num_voca = len(fetcher.word2id)
    num_relations = len(fetcher.rel2id)

    model = DDS(embed_dim, hidden_dim, num_layers, num_relations, num_voca, pos_dim, fetcher.word_embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('Param %s : %s', name, param.shape)

    logger.info('Number of relations: %d' % (len(fetcher.rel2id)))
    logger.info('Number of entity pairs: %d' % fetcher.num_pairs)

    for epoch in range(num_epoch):
        fetcher.reset()

        valid_data = list()
        for i in range(num_valid):
            x, y = next(fetcher)
            valid_data.append((x, y))

        optimizer.zero_grad()
        for i, (x, y) in enumerate(
                tqdm(fetcher, initial=epoch * fetcher.num_pairs, total=(fetcher.num_pairs - num_valid) * num_epoch)):

            loss = loss_fn(model(x), torch.from_numpy(y).float()) / batch_size
            loss.backward()
            if i % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.debug('%d Done, loss = %f', i, loss)

            if i % valid_cycle == 0 and i != 0:
                test(valid_data, model, loss_fn)

            if i % 1000000 == 0 and i != 0:
                logger.info('Save Model %d ...' % i)
                torch.save(model.state_dict(), 'saved_model_%d.tmp' % i)
                logger.info('Done')

    logger.info('Save Model ...')
    torch.save(model.state_dict(), 'saved_model_final.tmp')
    logger.info('Done')

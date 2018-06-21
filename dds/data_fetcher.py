"""
Data fetcher for deep distant supervision models

DataFetcher is an abstract class handling dataset.

Each data point feed into model will consist of
    (first_entity, second_entity, bag_of_sentences, set_of_possible_relations)
    - bag_of_sentences: collection of all sentences which contains both first_entity and second_entity
    - set_of_possible_relations: set of all possible relations between first_entity and second_entity, retrieved from existing knowledge base
"""
import logging
from abc import abstractmethod
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# process draft
OOV = 'OOV'
BLANK = 'BLANK'
max_sen_len = 70  # Predefined maximum length of sentence, need for position embedding


class DataFetcher():
    def __init__(self, w2v_path='', rel_path='', emb_dim='', data_path='', is_shuffle=True, max_sen_len=70):
        self.w2v_path = w2v_path
        self.rel_path = rel_path
        self.data_path = data_path
        self.word2id = {}
        self.rel2id = {}
        self.is_shuffle = is_shuffle
        self.max_sen_len = max_sen_len
        self.current_pos = 0
        self.emb_dim = emb_dim
        logger.info('Loading w2v')
        self.word2id, self.word_embedding = self.load_w2v(w2v_path, emb_dim)
        logger.info('Loading relations')
        self.rel2id, self.id2rel = self.load_relations(rel_path)
        self.num_voca = len(self.word2id)
        self.num_rel = len(self.rel2id)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abstractmethod
    def next(self):
        """This will return next training pair x, y"""

    def reset(self):
        """Reset the generator position to start"""
        self.current_pos = 0

    def load_w2v(self, path, dim):
        vec = []
        word2id = {}
        word2id[BLANK] = len(word2id)
        word2id[OOV] = len(word2id)
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                word2id[tokens[0]] = len(word2id)
                vec.append(np.array([float(x) for x in tokens[1:]]))

        word_embeddings = np.array(vec, dtype=np.float32)
        return word2id, word_embeddings

    @abstractmethod
    def _load_relations(self, path):
        """ return relations from relation file"""

    # 2. Load target relations including NA
    def load_relations(self, path):
        return self._load_relations(path)


class NYTFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):
        super(NYTFetcher, self).__init__(*args, **kwargs)
        logger.info('Loading sentences')
        self.pairs, self.sen_col = self._load_triple(self.data_path)
        self.keys = list(self.pairs.keys())
        self.num_pairs = len(self.pairs)
        if self.is_shuffle:
            np.random.shuffle(self.keys)
        logger.info('Loading fetcher done')

    def _load_relations(self, path):
        rel2id = dict()
        id2rel = dict()

        with open(path, 'r') as f:
            for line in f:
                rel, id = line.strip().split()
                rel2id[rel] = int(id)
                id2rel[int(id)] = rel
        return rel2id, id2rel

    def _load_triple(self, datapath):
        """
        extract knowledge graph and sentences from nyt dataset
        :return:
        """
        triples = defaultdict(set)
        sen_col = defaultdict(list)
        with open(datapath, 'r') as f:
            for line in f:
                tokens = line.split('\t')
                # the number of tokens are not the same for train.txt and test.txt, cannot unpack directly from split
                en1id, en2id, en1token, en2token, rel, sen = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], \
                                                             tokens[5]

                # add relation
                triples[(en1id, en2id)].add(rel)

                # find entity positions
                tokens = sen.split()
                if len(tokens) < self.max_sen_len:
                    sen2tid = list()
                    en1pos, en2pos = -1, -1
                    for pos, token in enumerate(tokens):
                        if token == en1token:
                            en1pos = pos
                        if token == en2token:
                            en2pos = pos

                        if token in self.word2id:
                            sen2tid.append(self.word2id[token])
                        else:
                            sen2tid.append(self.word2id[OOV])

                    # add parsed sentence
                    sen_col[(en1id, en2id)].append((sen2tid, en1pos, en2pos))

        for key, value in triples.items():
            # if entity pair has any relation except 'NA', remove 'NA' from relation set
            if len(value) > 1:
                if 'NA' in value:
                    value.remove('NA')

        return triples, sen_col

    def next(self):
        """
        Generate batch of sentences from a randomly sampled entity pair
        :param triples: dictionary, key=(en1id, en2id), value=(rel1, rel2, ...)
        :param sen_col: dictionary, key=(en1id, en2id), value=((sen2tid, en1pos, en2pos), ...)
        :param rel2id: dictionary, key=relation, value=id
        :return:
        """
        while True:
            if self.current_pos >= self.num_pairs:
                raise StopIteration

            key = self.keys[self.current_pos]
            self.current_pos += 1
            x = list()
            y = np.zeros(self.num_rel)
            for rel in self.pairs[key]:
                if rel in self.rel2id:
                    y[self.rel2id[rel]] = 1
                else:
                    y[0] = 1
            sentences = self.sen_col[key]
            for sen2tid, en1pos, en2pos in sentences:
                sen_len = len(sen2tid)
                sen2tid = np.array(sen2tid)
                pos1vec = np.arange(max_sen_len - en1pos, max_sen_len - en1pos + sen_len)
                pos2vec = np.arange(max_sen_len - en2pos, max_sen_len - en2pos + sen_len)
                x.append((sen2tid, pos1vec, pos2vec))

            if len(x) > 0:
                return x, y


class FreebaseFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):
        from dds.data_utils.db_helper import pair_collection, pair_count, relation_collection, sentence_collection

        super(FreebaseFetcher, self).__init__(*args, **kwargs)
        self.pair_count = pair_count
        self.pair_collection = pair_collection
        self.relation_collection = relation_collection
        self.sentence_collection = sentence_collection

        self.cur = pair_count.find()
        self.current_pos = 0
        self.num_pairs = self.pair_collection.count()

    def _load_relations(self, path):
        rel2id = dict()
        id2rel = dict()

        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 3:
                    if int(tokens[2]) > 100:
                        #minimum number of sentences to be trained
                        rel2id[line.strip().split()[0]] = len(rel2id)
                        id2rel[len(rel2id) - 1] = line.strip()
                else:
                    # NA
                    rel2id[line.strip().split()[0]] = len(rel2id)
                    id2rel[len(rel2id) - 1] = line.strip()

        return rel2id, id2rel

    def _sentence2id(self, sentence):
        sen2tid = list()  # sentence to list of token ids
        for token in sentence:
            if type(token) == list:
                entity = ' '.join(token)
                if entity in self.word2id:
                    token_id = self.word2id[entity]
                else:
                    token_id = self.word2id[OOV]
            else:
                if token in self.word2id:
                    token_id = self.word2id[token]
                else:
                    token_id = self.word2id[OOV]
            sen2tid.append(token_id)
        return np.array(sen2tid)

    def reset(self):
        self.cur = self.pair_count.find()
        self.current_pos = 0

    def next(self):
        # retrieve entity pairs from beginning to end
        # random sample from collection of entity pairs
        while True:
            if self.current_pos > self.num_pairs:
                raise StopIteration

            en_pair = self.cur[self.current_pos]
            self.current_pos += 1
            # for each entity pair in the dataset
            en1id = en_pair['en1id']
            en2id = en_pair['en2id']
            query = {'en1id': en1id, 'en2id': en2id}

            pair_cur = self.pair_collection.find(query)
            x = list()
            for pair in pair_cur:
                # construct inputs for the next batch
                sen_cur = self.sentence_collection.find({'_id': pair['sid']})
                sen_row = sen_cur.next()  # unique
                en1pos = pair['en1pos']
                en2pos = pair['en2pos']
                sen_len = len(sen_row['sentence'])
                if sen_len < max_sen_len:
                    pos1vec = np.arange(max_sen_len - en1pos, max_sen_len - en1pos + sen_len)
                    pos2vec = np.arange(max_sen_len - en2pos, max_sen_len - en2pos + sen_len)
                    sen2tid = self._sentence2id(sen_row['sentence'])
                    x.append((sen2tid, pos1vec, pos2vec))

            rel_cur = self.relation_collection.find(query)
            y = np.zeros(self.num_rel)
            if rel_cur.count() != 0:
                for row in rel_cur:
                    if row['rel'] in self.rel2id:
                        # we only consider predefined relation
                        y[self.rel2id[row['rel']]] = 1
                    else:
                        y[self.rel2id['NA']] = 1
            else:
                y[self.rel2id['NA']] = 1

            if len(x) > 0:
                return x, y


if __name__ == '__main__':
    # test data fetcher for nyt dataset
    def compute_stats(fetcher):
        num_train_rel = 0
        num_train_sen = 0
        num_valid_train_sen = 0
        rel_cnt = defaultdict(int)
        for batch_x, batch_y in fetcher:
            num_train_rel += np.sum(batch_y[1:])
            num_train_sen += len(batch_x)
            if np.sum(batch_y[1:]) != 0:
                num_valid_train_sen += len(batch_x)
            for i in range(1, fetcher.num_rel):
                if batch_y[i] == 1:
                    rel_cnt[fetcher.id2rel[i]] = len(batch_x)
        return num_train_rel, num_train_sen, num_valid_train_sen, rel_cnt


    embed_dim = 50
    w2v_path = '../data/word2vec.txt'
    rel_path = '../data/nyt/relation2id.txt'
    train_path = '../data/nyt/train.txt'
    test_path = '../data/nyt/test.txt'
    train_fetcher = NYTFetcher(w2v_path, rel_path, embed_dim, train_path)
    test_fetcher = NYTFetcher(w2v_path, rel_path, embed_dim, test_path)

    num_tr, num_ts, num_vts, rel_dict = compute_stats(train_fetcher)
    t_num_tr, t_num_ts, t_num_vts, t_rel_dict = compute_stats(test_fetcher)

    print('Number of triples in the training/test set:', num_tr, t_num_tr)
    print('Number of sentences included in the training/test triples:', num_ts, t_num_ts)
    print('Number of valid sentences (sentences whose relation is not NA) included in the training/test triples:',
          num_vts, t_num_vts)
    print('Number of training/test sentences for each relation')
    for key, value in rel_dict.items():
        print('\tRelation %s: %d, %d' % (key, value, t_rel_dict[key]))

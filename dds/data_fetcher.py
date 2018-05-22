"""
Pytorch version of deep-distant supervision algorithm.

What we need as input files:

0. Predefined set of relations
1. Collection of sentences (sentence (tokenized), entity1, entity2, entity1_position, entity2_position
2. Knowledge graph - query from database. When issue query on pair of entities, db returns the collection of relations
3. word2vec (or predefined vocabulary) - convert sentences to a list of token ids

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

    # 2. Load target relations including NA
    def load_relations(self, path, include_id=True):
        rel2id = dict()
        id2rel = dict()

        with open(path, 'r') as f:
            for line in f:
                if include_id:
                    rel, id = line.strip().split()
                    rel2id[rel] = int(id)
                    id2rel[int(id)] = rel
                else:
                    rel2id[line.strip()] = len(rel2id)
                    id2rel[len(rel2id) - 1] = line.strip()
        return rel2id, id2rel


class NYTFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):
        super(NYTFetcher, self).__init__(*args, **kwargs)
        logger.info('Loading sentences')
        self.pairs, self.sen_col = self.load_triple(self.data_path)
        self.keys = list(self.pairs.keys())
        self.num_pairs = len(self.pairs)
        if self.is_shuffle:
            np.random.shuffle(self.keys)
        logger.info('Loading fetcher done')

    def load_triple(self, datapath):
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
        self.pair_collection = pair_collection
        self.relation_collection = relation_collection
        self.sentence_collection = sentence_collection

        self.cur = pair_count.find()
        self.current_pos = 0
        self.num_pairs = self.pair_collection.count()
        self.ret_order = np.arange(self.num_pairs)
        if self.is_shuffle:
            np.random.shuffle(self.ret_order)

    def sentence2id(self, sentence):
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
        return sen2tid

    def fetch_sentences_wiki(self, word2id, rel2id):
        # retrieve entity pairs from beginning to end
        # random sample from collection of entity pairs

        while True:
            if self.current_pos > self.num_pairs:
                raise StopIteration

            en_pair = self.cur[self.ret_order[self.current_pos]]  # is this efficient enough?
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
                    sen2tid = self.sentence2id(sen_row['sentence'])
                    x.append((sen2tid, pos1vec, pos2vec))

            rel_cur = self.relation_collection.find(query)
            y = np.zeros(len(rel2id))
            if rel_cur.count() != 0:
                for row in rel_cur:
                    y[rel2id[row['rel']]] = 1
            else:
                y[rel2id['NA']] = 1

            if len(x) > 0:
                return x, y

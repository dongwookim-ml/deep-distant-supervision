"""
Pytorch version of deep-distant supervision algorithm.

What we need as input files:

0. Predefined set of relations
1. Collection of sentences (sentence (tokenized), entity1, entity2, entity1_position, entity2_position
2. Knowledge graph - query from database. When issue query on pair of entities, db returns the collection of relations
3. word2vec (or predefined vocabulary) - convert sentences to a list of token ids

"""
import numpy as np
from collections import defaultdict

# process draft
OOV = 'OOV'
BLANK = 'BLANK'
max_sen_len = 70  # Predefined maximum length of sentence, need for position embedding


# 1. Load word2vec
def load_w2v(path):
    dim = 50
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
def load_relations(path, include_id):
    rel2id = {}
    with open(path, 'r') as f:
        for line in f:
            if include_id:
                rel, id = line.strip().split()
                rel2id[rel] = int(id)
            else:
                rel2id[line.strip()] = len(rel2id)
    return rel2id


# 3. Select random batch of sentences of entity pairs (DB conn required, wikipedia/pair_col, wikipedia/sentence_col)
# 4. Retrieve a set of possible relations given entity pairs from #3 (DB conn required, fb_db/relation_col)
# 5. Convert sentences into list of tokens and relative position w.r.t. each entity
def sentence2id(sentence, word2id):
    sen2tid = list()  # sentence to list of token ids
    for token in sentence:
        if type(token) == list:
            entity = ' '.join(token)
            if entity in word2id:
                token_id = word2id[entity]
            else:
                token_id = word2id[OOV]
        else:
            if token in word2id:
                token_id = word2id[token]
            else:
                token_id = word2id[OOV]
        sen2tid.append(token_id)
    return sen2tid


def fetch_sentences_wiki(word2id, rel2id):
    from dds.data_utils.db_helper import pair_collection, pair_count, relation_collection, sentence_collection

    # retrieve entity pairs from beginning to end
    # cur = pair_count.find()
    # random sample from collection of entity pairs
    cur = pair_count.aggregate([{'$sample': {'size': 1}}])

    for en_pair in cur:
        # for each entity pair in the dataset
        en1id = en_pair['en1id']
        en2id = en_pair['en2id']
        query = {'en1id': en1id, 'en2id': en2id}

        pair_cur = pair_collection.find(query)
        x = list()
        for pair in pair_cur:
            # construct inputs for the next batch
            sen_cur = sentence_collection.find({'_id': pair['sid']})
            sen_row = sen_cur.next()  # unique
            en1pos = pair['en1pos']
            en2pos = pair['en2pos']
            sen_len = len(sen_row['sentence'])
            if sen_len < max_sen_len:
                pos1vec = np.arange(max_sen_len - en1pos, max_sen_len - en1pos + sen_len)
                pos2vec = np.arange(max_sen_len - en2pos, max_sen_len - en2pos + sen_len)
                sen2tid = sentence2id(sen_row['sentence'], word2id)
                x.append((sen2tid, pos1vec, pos2vec))

        rel_cur = relation_collection.find(query)
        y = np.zeros(len(rel2id))
        if rel_cur.count() != 0:
            for row in rel_cur:
                y[rel2id[row['rel']]] = 1
        else:
            y[rel2id['NA']] = 1

        if len(x) > 0:
            yield (x, y)


# 6. Feed the data into the model and train
# word2id, embedding = load_w2v()
# rel2id = load_relations()
# fetch_sentences(word2id, rel2id)


def loadnyt(datapath, word2id):
    """
    extract knowledge graph and sentences from nyt dataset
    :return:
    """
    triples = defaultdict(set)
    sen_col = defaultdict(list)
    with open(datapath, 'r') as f:
        for line in f:
            en1id, en2id, en1token, en2token, rel, sen = line.split('\t')

            # add relation
            triples[(en1id, en2id)].add(rel)

            # find entity positions
            tokens = sen.split()
            if len(tokens) < max_sen_len:
                sen2tid = list()
                en1pos, en2pos = -1, -1
                for pos, token in enumerate(tokens):
                    if token == en1token:
                        en1pos = pos
                    if token == en2token:
                        en2pos = pos

                    if token in word2id:
                        sen2tid.append(word2id[token])
                    else:
                        sen2tid.append(word2id[OOV])

                # add parsed sentence
                sen_col[(en1id, en2id)].append((sen2tid, en1pos, en2pos))

    return triples, sen_col


def fetch_sentences_nyt(triples, sen_col, rel2id):
    """
    Generate batch of sentences from a randomly sampled entity pair
    :param triples: dictionary, key=(en1id, en2id), value=(rel1, rel2, ...)
    :param sen_col: dictionary, key=(en1id, en2id), value=((sen2tid, en1pos, en2pos), ...)
    :param rel2id: dictionary, key=relation, value=id
    :return:
    """
    keys = list(triples.keys())
    np.random.shuffle(keys)
    num_rel = len(rel2id)
    for key in keys:
        x = list()
        y = np.zeros(num_rel)
        for rel in triples[key]:
            if rel in rel2id:
                y[rel2id[rel]] = 1
            else:
                y[0] = 1
        sentences = sen_col[key]
        for sen2tid, en1pos, en2pos in sentences:
            sen_len = len(sen2tid)
            pos1vec = np.arange(max_sen_len - en1pos, max_sen_len - en1pos + sen_len)
            pos2vec = np.arange(max_sen_len - en2pos, max_sen_len - en2pos + sen_len)
            x.append((sen2tid, pos1vec, pos2vec))
        if len(x) > 0:
            yield (x, y)

"""
Script to parse fb15k file
"""
import os, sys
import logging
from collections import defaultdict

# Logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(ch)
logger.setLevel('INFO')

fb15k_path = '../../data/FB15k-237.2/'
train_file = 'train.txt'
valid_file = 'valid.txt'
test_file = 'test.txt'

DELIM = '_'


def read_triples(file):
    """ Read file and return dictionary (triple) whose key is a pair of entities and value is a list of relations """
    triple = defaultdict(list)

    with open(file, 'r') as f:
        for line in f:
            tokens = line.split('\t')
            en1 = tokens[0].strip()[1:].replace('/', '.')
            en2 = tokens[2].strip()[1:].replace('/', '.')
            rel = tokens[1].strip()
            triple[(en1, en2)].append(rel)

    return triple


def merge_triples(triples):
    """ merge multiple triples into single triple """
    new_dict = defaultdict(list)
    en_dict = dict()
    rel_dict = dict()
    for triple in triples:
        for key, value in triple.items():
            if key not in new_dict:
                new_dict[key] = value
            else:
                for rel in value:
                    if rel not in new_dict[key]:
                        new_dict[key].append(rel)
            for en in key:
                if en not in en_dict:
                    en_dict[en] = len(en_dict)
            for rel in value:
                if rel not in rel_dict:
                    rel_dict[rel] = len(rel_dict)

    return new_dict, en_dict, rel_dict


if __name__ == '__main__':
    tr_triple = read_triples(os.path.join(fb15k_path, train_file))
    val_triple = read_triples(os.path.join(fb15k_path, valid_file))
    t_triple = read_triples(os.path.join(fb15k_path, test_file))

    all_triples, fb15k_en_dict, fb15k_rel_dict = merge_triples([tr_triple, val_triple, t_triple])

    logger.info('# of entities in fb15k: {}'.format(len(fb15k_en_dict)))
    logger.info('# of relations in fb15k: {}'.format(len(fb15k_rel_dict)))


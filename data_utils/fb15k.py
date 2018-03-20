"""
Script to parse fb15k file
"""
import os
import data_utils
import itertools
import wikipedia
from collections import defaultdict
from nltk.tag.stanford import CoreNLPNERTagger

fb15k_path = '../data/FB15k-237.2/'
train_file = 'train.txt'
valid_file = 'valid.txt'
test_file = 'test.txt'


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

    print('# of entities in fb15k: {}'.format(len(fb15k_en_dict)))
    print('# of relations in fb15k: {}'.format(len(fb15k_rel_dict)))

    name2id, id2name = data_utils.load_freebase_entity()

    fb15k_id2name = dict()
    fb15k_name2id = dict()

    not_in_fb = 0
    for key in fb15k_en_dict.keys():
        if key in id2name:
            fb15k_id2name[key] = id2name[key]
            fb15k_name2id[id2name[key]] = key
        else:
            not_in_fb += 1

    print('{} entities of fb15k are not in Freebase'.format(not_in_fb))

    server_url = 'http://localhost:9000'  # Stanford corenlp server address
    stream = wikipedia.iter_wiki()
    tagger = CoreNLPNERTagger(url=server_url)
    # for title, tokens in itertools.islice(stream, 10):
    for title, tokens in stream:
        try:
            tagged_text = tagger.tag(tokens)
            ners = data_utils.extract_ners(tagged_text)

            cnt = 0
            valid_ners = list()
            for ner in ners:
                name = ' '.join(ner[0])
                if name in fb15k_name2id:
                    print('NER in FB', name)
                    valid_ners.append(name)
                    cnt += 1

            if cnt > 2:
                # there are more than 2 entities in this sentence.
                print(' '.join(tokens))
                print(valid_ners)

        except:
            pass


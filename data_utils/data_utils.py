from collections import defaultdict
from scipy.sparse import csc_matrix

IGNORE_NER_TAG = ('O', 'DATE', 'NUMBER', 'ORDINAL')  # Non-NER tag


def extract_ners(tokens):
    """
    Extract consecutive ners from the result of CoreNLPNERTagger
    :param tokens: list of tuple with token and tag
    :return: list of tuple with list of tokens and corresponding tag
    """
    ners = list()
    new_tokens = list()

    candid_entity = list()
    keep = False
    prev_tag = 'O'

    for i, (token, tag) in enumerate(tokens):
        if keep:
            if tag not in IGNORE_NER_TAG:
                if prev_tag == tag:
                    candid_entity.append(token)
                    keep = True
                else:
                    ners.append((candid_entity, prev_tag))
                    candid_entity = list()
                    candid_entity.append(token)
                    keep = True
            else:
                ners.append((candid_entity, prev_tag))
                keep = False
        else:
            if tag not in IGNORE_NER_TAG:
                candid_entity = list()
                candid_entity.append(token)
                keep = True
        prev_tag = tag

    return ners


def lookup_freebase(ner, en_dict):
    """
    Return existence of ner from Freebase entity dictionary
    :param ner:
    :return:
    """
    pass


def load_freebase_entity(path="../data/freebase/dict.txt"):
    """
    Load freebase entity dictionary from saved dict
    :return:
    """
    print('Loading freebase entity dictionary...')

    name2id = dict()
    id2name = dict()
    with open(path, 'r', buffering=1024*1024*100) as f:
        for line in f:
            tokens = line.split('\t')
            _name = tokens[0].strip()
            _id = tokens[1].strip()
            name2id[_name] = _id
            id2name[_id] = _name

    print('Successfully loaded {} entities from file'.format(len(name2id)))

    return name2id, id2name


if __name__ == '__main__':
    pass

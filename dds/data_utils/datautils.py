""" Utils for the corpus generation """
import logging
import json
from dds.data_utils.config import IGNORE_NER_TAG

logger = logging.getLogger(__name__)


def encode_rds(obj):
    return json.dumps(obj)


def decode_rds(obj):
    return json.loads(obj)


def merge_ners(tokens):
    """
    Extract consecutive ners from the result of CoreNLPNERTagger
    :param tokens: list of tuple with token and tag
    :return: list of tuple with list of tokens and corresponding tag
    """
    ners = list()
    merged_tokens = list()

    candid_entity = list()
    keep = False
    prev_tag = 'O'

    tokens.append(('NA', 'O'))

    for i, (token, tag) in enumerate(tokens):
        if keep:
            if tag not in IGNORE_NER_TAG:
                candid_entity.append(token)
                keep = True
            else:
                # ner ends in prev step
                merged_tokens.append(candid_entity)
                merged_tokens.append(token)
                ners.append((candid_entity, prev_tag))
                keep = False
        else:
            if tag not in IGNORE_NER_TAG:
                # new ner
                candid_entity = list()
                candid_entity.append(token)
                keep = True
            else:
                # not ner token
                merged_tokens.append(token)
        prev_tag = tag

    return ners, merged_tokens


def load_freebase_entity(path="../data/freebase/dict.txt"):
    """
    Load freebase entity dictionary from saved dict
    :return:
    """
    logger.info('Loading freebase entity dictionary...')

    name2id = dict()
    id2name = dict()
    with open(path, 'r', buffering=1024 * 1024 * 100) as f:
        for line in f:
            tokens = line.split('\t')
            _name = tokens[0].strip()
            _id = tokens[1].strip()
            name2id[_name] = _id
            id2name[_id] = _name

    logger.info('Successfully loaded {} entities from file'.format(len(name2id)))

    return name2id, id2name


def get_nerpos(tokens, ner):
    """
    Return position of ner in list of tokens

    :param tokens: list of tokens
    :param ner: list of tokens
    :return: list, 0-indexed position of ner
    """

    loc = list()
    for i, token in enumerate(tokens):
        if token == ner:
            loc.append(i)
    return loc


def get_nerspos(tokens, ners):
    """
    Return positions of ners in list of tokens

    :param tokens:
    :param ners:
    :return:
    """
    pos_list = list()
    for ner in ners:
        pos = get_nerpos(tokens, ner)
        pos_list.append(pos)

    return pos_list


if __name__ == '__main__':
    pass

import sys, traceback
from data_utils import *
import itertools
import logging
import redis
import uuid
from nltk.tag.stanford import CoreNLPNERTagger
import concurrent.futures as confu

# Logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(ch)
logger.setLevel('INFO')


def extract_ners(tokens, tagger, rdb):
    """
    Extract freebase entities (NERs) from token using NER tagger.

    :param tokens: list of tokens forming a sentence
    :param tagger: NER tagger return list of tokens with corresponding NER tags
    :param rdb: redis database which contains name2id
    :return: if there are more than two freebase entities in the sentence, then return list of tokens and ners.
    each ner is again a list of tokens.
    """
    try:
        tagged_text = tagger.tag(tokens)
    except:
        logger.info('NER error {}'.format(len(tokens)))
        return None

    ners, merged_tokens = datautils.extract_ners(tagged_text)

    valid_ners = list()
    for ner in ners:
        # ner = ((tokens), tag)
        if lookup_fb(ner[0], rdb) is not None:
            if ner[0] not in valid_ners:
                valid_ners.append(ner[0])

    if len(valid_ners) >= 2:
        return merged_tokens, valid_ners

    return None


def process_sentence(tokens, ners, sdb):
    """
    Find position (starting from 0) of NERs in the list of tokens.
    For all possible pairs in NERs, yield their positions in the token list
    :param tokens: list of tokens forming a sentence
    :param ners: list of ners in the token list
    :param sdb: sentence database
    """
    ner_pos = datautils.get_nerspos(tokens, ners)

    sid = uuid.uuid1()
    sdb.set(sid, encode_rds(tokens))

    num_ner = len(ner_pos)
    for first_ner in range(num_ner - 1):
        for second_ner in range(1, num_ner):
            for pos1, pos2 in itertools.product(ner_pos[first_ner], ner_pos[second_ner]):
                yield (ners[first_ner], ners[second_ner], pos1, pos2, sid)


def update_db(rdb, en1, en2, en1pos, en2pos, sid):
    """
    Update database to add a sentence along with the position of two entities in the sentence.

    :param rdb: Redis DB containing wiki sentence
    :param en1: list, The first entity
    :param en2: list, The second entity
    :param en1pos: The position of the first entity in the list of tokens
    :param en2pos: The position of the second entity in the list of tokens
    :param sid: id of sentence in sentence database
    :return:
    """
    key = encode_rds((en1, en2))
    cnt = rdb.hincrby(key, "cnt")  # the number of sentences for this key

    rdb.hset(key, (cnt, "en1pos"), en1pos)
    rdb.hset(key, (cnt, "en2pos"), en2pos)
    rdb.hset(key, (cnt, "sentence"), sid)


if __name__ == '__main__':
    num_thread = 8
    server_url = 'http://localhost:9000'  # Stanford corenlp server address
    stream = wikipedia.iter_wiki()
    tagger = CoreNLPNERTagger(url=server_url)

    token_list = list()

    meta_db = redis.Redis(host=REDIS_HOST, db=0, port=REDIS_PORT, decode_responses=True)
    name2id_db = redis.Redis(host=REDIS_HOST, db=NAME2ID, port=REDIS_PORT, decode_responses=True)
    wiki_db = redis.Redis(host=REDIS_HOST, db=WIKI_NER, port=REDIS_PORT, decode_responses=True)
    sentence_db = redis.Redis(host=REDIS_HOST, db=SENTENCE, port=REDIS_PORT, decode_responses=True)

    _skip = int(meta_db.get('stat:wiki_parsed'))

    with confu.ThreadPoolExecutor(num_thread) as executor:
        try:
            for i, (title, tokens) in enumerate(stream):
                if _skip is not None and i < _skip:
                    continue

                token_list.append(tokens)

                if len(token_list) == num_thread:
                    futures = [executor.submit(extract_ners, tokens, tagger, name2id_db) for tokens in token_list]

                    for future in confu.as_completed(futures):
                        result = future.result()
                        try:
                            if result is not None:
                                tokens = result[0]
                                ners = result[1]
                                logger.info('Sentence {}'.format(tokens))
                                logger.info('NERS {}'.format(ners))
                                for en1, en2, en1pos, en2pos, sid in process_sentence(tokens, ners, sentence_db):
                                    update_db(wiki_db, en1, en2, en1pos, en2pos, sid)

                        except ValueError as e:
                            logger.debug(e)

                    token_list = list()

                if i % 100 == 0:
                    logger.info('{} lines processed'.format(i))

                if i > _skip:
                    meta_db.set('stat:wiki_parsed', i)

        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            meta_db.set('stat:wiki_parsed', i)

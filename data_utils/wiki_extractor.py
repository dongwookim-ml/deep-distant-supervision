import sys, traceback
from data_utils import *
import itertools
import logging
import redis
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
    each ner is joined by DELIM.
    """
    try:
        tagged_text = tagger.tag(tokens)
    except:
        logger.info('NER error {}'.format(len(tokens)))
        return None

    ners, merged_tokens = datautils.extract_ners(tagged_text)

    valid_ners = list()
    for ner in ners:
        if lookup_freebase(ner, rdb) is not None:
            if ner not in valid_ners:
                valid_ners.append(ner)

    if len(valid_ners) >= 2:
        return merged_tokens, valid_ners

    return None


def process_sentence(tokens, ners):
    """
    Find position (starting from 0) of NERs in the list of tokens.
    For all possible pairs in NERs, yield their positions in the token list
    :param tokens: list of tokens forming a sentence
    :param ners: list of ners in the token list
    """
    ner_pos = datautils.get_nerspos(tokens, ners)

    num_ner = len(ner_pos)
    for first_ner in range(num_ner - 1):
        for second_ner in range(1, num_ner):
            for pos1, pos2 in itertools.product(ner_pos[first_ner], ner_pos[second_ner]):
                yield (ners[first_ner], ners[second_ner], pos1, pos2, tokens)


def update_db(rdb, en1, en2, en1pos, en2pos, tokens):
    """
    Update database to add a sentence along with the position of two entities in the sentence.

    :param rdb: Redis DB containing wiki sentence
    :param en1: The first entity
    :param en2: The second entity
    :param en1pos: The position of the first entity in the list of tokens
    :param en2pos: The position of the second entity in the list of tokens
    :param tokens: List of tokens
    :return:
    """
    key = (en1, en2)
    cnt = rdb.hincrby(key, "cnt")

    rdb.hset(key, (cnt, "en1pos"), en1pos)
    rdb.hset(key, (cnt, "en2pos"), en2pos)
    rdb.hset(key, (cnt, "sentence"), ' '.join(tokens))


if __name__ == '__main__':
    num_thread = 8
    server_url = 'http://localhost:9000'  # Stanford corenlp server address
    stream = wikipedia.iter_wiki()
    tagger = CoreNLPNERTagger(url=server_url)

    token_list = list()

    meta_db = redis.Redis(host=REDIS_HOST, db=0, port=REDIS_PORT, decode_responses=True)
    name2id_db = redis.Redis(host=REDIS_HOST, db=NAME2ID, port=REDIS_PORT, decode_responses=True)
    wiki_db = redis.Redis(host=REDIS_HOST, db=WIKI_NER, port=REDIS_PORT, decode_responses=True)

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
                                logger.info('Sentence {}'.format(' '.join(tokens)))
                                logger.info('NERS {}'.format(ners))
                                for en1, en2, en1pos, en2pos, tokens in process_sentence(tokens, ners):
                                    update_db(wiki_db, en1, en2, en1pos, en2pos, tokens)


                        except ValueError as e:
                            logger.debug(e)

                    token_list = list()

                if i % 1000 == 0:
                    logger.info('{} lines processed'.format(i))
                    meta_db.set('stat:wiki_parsed', i)

        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            meta_db.set('stat:wiki_parsed', i)

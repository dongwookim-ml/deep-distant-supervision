import os, sys, traceback
from data_utils import wikipedia
from data_utils import datautils
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

DELIM = '_'

REDIS_HOST = 'localhost'
REDIS_PORT = 6379


def extract_ners(tokens, rdb):
    try:
        tagged_text = tagger.tag(tokens)
    except:
        logger.info('NER error {}'.format(len(tokens)))
        return None

    ners, merged_tokens = datautils.extract_ners(tagged_text)

    valid_ners = list()
    for ner in ners:
        name = DELIM.join(ner[0])
        rval = rdb.get(name)
        if rval is not None:
            if name not in valid_ners:
                valid_ners.append(name)

    if len(valid_ners) >= 2:
        return merged_tokens, valid_ners

    return None


def process_sentence(tokens, ners):
    ner_pos = datautils.get_nerspos(tokens, ners)

    num_ner = len(ner_pos)
    for first_ner in range(num_ner - 1):
        for second_ner in range(1, num_ner):
            for pos1, pos2 in itertools.product(ner_pos[first_ner], ner_pos[second_ner]):
                yield (ners[first_ner], ners[second_ner], pos1, pos2, tokens)


def update_db(rdb, en1, en2, en1pos, en2pos, tokens):
    key = (en1, en2)
    cnt = rdb.hincrby(key, "cnt")

    rdb.hset(key, (cnt, "en1"), en1)
    rdb.hset(key, (cnt, "en2"), en1)
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
    db_no = meta_db.get('wiki_ner_db_no')
    _skip = int(meta_db.get('wiki_parsed'))
    name2id_db_no = meta_db.get('name2id_db_no')
    name2id_db = redis.Redis(host=REDIS_HOST, db=name2id_db_no, port=REDIS_PORT, decode_responses=True)
    wiki_db = redis.Redis(host=REDIS_HOST, db=db_no, port=REDIS_PORT, decode_responses=True)

    with confu.ThreadPoolExecutor(num_thread) as executor:
        try:
            for i, (title, tokens) in enumerate(stream):
                if _skip is not None and i < _skip:
                    continue

                token_list.append(tokens)

                if len(token_list) == num_thread:
                    futures = [executor.submit(extract_ners, tokens, name2id_db) for tokens in token_list]

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
                    meta_db.set('wiki_parsed', i)

        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            meta_db.set('wiki_parsed', i)

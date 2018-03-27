"""
Database operations
"""
import redis
import pymongo
from data_utils import *

# connect to mongodb
client = pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
db = client[MG_WIKI_DB]
sentence_collection = db[MG_SENTENCE_COL]
pair_collection = db[MG_PAIR_COL]
pair_count = db[MG_PAIR_CNT]

meta_db = redis.Redis(host=REDIS_HOST, db=METADB, port=REDIS_PORT, decode_responses=True)
# surface name to freebase id
name2id_db = redis.Redis(host=REDIS_HOST, db=NAME2ID, port=REDIS_PORT, decode_responses=True)
# freebase id to surface name
id2name_db = redis.Redis(host=REDIS_HOST, db=ID2NAME, port=REDIS_PORT, decode_responses=True)


def get_wiki_parsed_stat():
    """
    :return: The number of sentences parsed so far
    """
    return int(meta_db.get('stat:wiki_parsed'))


def set_wiki_parsed_cnt(cnt):
    """
    Set the count to cnt
    :param cnt: the number of sentences parsed so far
    :return:
    """
    meta_db.set('stat:wiki_parsed', cnt)


def lookup_fb(ner):
    """
    Search NER in freebase via exact matching
    :param ner: str or list of str
    :return:
    Return freebase_id (starting with m.) of ner from Freebase entity dictionary,
    otherwise return None.
    """
    if type(ner) == str:
        return name2id_db.get(ner)
    return name2id_db.get(' '.join(ner))


def add_sentence(sentence, ners=None, ner_pos=None):
    """
    Add sentence to db
    :param sentence: list of tokens, each token is a string or a list (if ner)
    :param ners: list of ners, each ner is a list of tokens
    :param ner_pos: list of position of ner in the sentence
    :return:
    """
    if ners is not None:
        post = {'sentence': sentence,
                'ners': ners,
                'ner_pos': ner_pos}
    else:
        post = {'sentence': sentence}

    return sentence_collection.insert_one(post).inserted_id


def add_pair(en1id, en2id, en1, en2, en1pos, en2pos, sid):
    """
    Add an entity pair into db
    :param en1id: freebase id of the first entity
    :param en2id: freebase id of the second entity
    :param en1: list, list of tokens of the first entity
    :param en2: list, list of tokens of the second entity
    :param en1pos: the position of the first entity in the sentence[sid]
    :param en2pos: the position of the second entity in the sentence[sid]
    :param sid: unique sentence id
    :return:
    """
    post = {'en1id': en1id,
            'en2id': en2id,
            'en1': en1,
            'en2': en2,
            'en1pos': en1pos,
            'en2pos': en2pos,
            'sid': sid
            }
    pair_collection.insert_one(post)
    pair_count.update_one({'en1id': en1id, 'en2id': en2id},
                          {'$inc': {'cnt': 1}}, upsert=True)


def lookup_sentence(sid):
    """
    :param sid: sentence id
    :return: Return sentence (list of tokens)
    """
    return sentence_collection.find({"_id": sid})

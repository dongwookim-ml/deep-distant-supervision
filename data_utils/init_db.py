import redis
import data_utils

METADB = data_utils.METADB
NAME2ID = data_utils.NAME2ID
ID2NAME = data_utils.METADB
WIKI_NER = data_utils.WIKI_NER

HOST = data_utils.REDIS_HOST
PORT = data_utils.REDIS_PORT

metadb = redis.Redis(host=HOST, db=METADB, port=PORT, decode_responses=True)
metadb.flushall()

metadb.set('db:name2id', NAME2ID)
metadb.set('db:id2name', ID2NAME)
metadb.set('db:wiki_ner', WIKI_NER)
metadb.set('db:id2uid', WIKI_NER)


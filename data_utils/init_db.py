import redis
import data_utils

METADB = data_utils.METADB
NAME2ID = data_utils.NAME2ID
ID2NAME = data_utils.METADB
WIKI_NER = data_utils.WIKI_NER

HOST = data_utils.REDIS_HOST
PORT = data_utils.REDIS_PORT

metadb = redis.Redis(host=HOST, db=METADB, port=PORT, decode_responses=True)
metadb.set('name2id_db_no', NAME2ID)
metadb.set('id2name_db_no', ID2NAME)
metadb.set('wiki_ner_db_no', WIKI_NER)

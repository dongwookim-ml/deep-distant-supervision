import redis

METADB = 0
NAME2ID = 1
ID2NAME = 2
WIKI_NER = 3

HOST = 'localhost'
PORT = 6379

metadb = redis.Redis(host=HOST, db=METADB, port=PORT, decode_responses=True)
metadb.set('name2id_db_no', NAME2ID)
metadb.set('id2name_db_no', ID2NAME)
metadb.set('wiki_ner_db_no', WIKI_NER)

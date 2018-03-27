"""
Initialise databases
"""
import pymongo
import data_utils
from data_utils.db_helper import meta_db, pair_collection, pair_count

# Redis

METADB = data_utils.METADB
NAME2ID = data_utils.NAME2ID
ID2NAME = data_utils.METADB

HOST = data_utils.REDIS_HOST
PORT = data_utils.REDIS_PORT

# meta_db.flushall()     # be careful this will remove the current databases

meta_db.set('db:name2id', NAME2ID)
meta_db.set('db:id2name', ID2NAME)
meta_db.set('stat:wiki_parsed', 0)

# MongoDB

pair_collection.create_index([('en1id', pymongo.ASCENDING), ('en2id', pymongo.ASCENDING)])
pair_count.create_index([('en1id', pymongo.ASCENDING), ('en2id', pymongo.ASCENDING)], unique=True)

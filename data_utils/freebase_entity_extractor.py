"""
Script for extracting entities from Freebase dump.
"""

import io
import gzip
import redis
from data_utils import *

cnt = 0

metadb = redis.Redis(host=REDIS_HOST, db=METADB, port=REDIS_PORT, decode_responses=True)
name2id_db = redis.Redis(host=REDIS_HOST, db=NAME2ID, port=REDIS_PORT, decode_responses=True)
id2name_db = redis.Redis(host=REDIS_HOST, db=ID2NAME, port=REDIS_PORT, decode_responses=True)

with io.TextIOWrapper(gzip.open("../data/freebase/freebase-rdf-latest.gz", "r")) as freebase:
    for line in freebase:
        if "type.object.name" in line:
            tokens = line.split('\t')
            mid = tokens[0].split('/')[-1][:-1]
            name = tokens[2]
            if mid.startswith('m.'):
                if "@en" in name:
                    name = name.replace("@en", "")
                    name = name.replace("\"", "")

                    # insert into dbs
                    name2id_db.set(name, mid)
                    name2id_db.set(name.replace(' ', DELIM), mid)
                    id2name_db.set(mid, name)

        cnt += 1

        if cnt % 10000000 == 0:

            print("{} lines processed, dict_size={}".format(cnt, name2id_db.dbsize()))


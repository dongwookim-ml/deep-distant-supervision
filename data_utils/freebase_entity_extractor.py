"""
Script for extracting entities from Freebase dump.
"""

import io
import gzip
import redis

META_DB = 0
HOST = 'localhost'
PORT = 6379

cnt = 0

metadb = redis.Redis(host=HOST, db=META_DB, port=PORT, decode_responses=True)
NAME2ID = metadb.get('name2id_db_no')
ID2NAME = metadb.get('id2name_db_no')

name2id_db = redis.Redis(host=HOST, db=NAME2ID, port=PORT, decode_responses=True)
id2name_db = redis.Redis(host=HOST, db=ID2NAME, port=PORT, decode_responses=True)

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
                    id2name_db.set(mid, name)

        cnt += 1

        if cnt % 10000000 == 0:

            print("{} lines processed, dict_size={}".format(cnt, name2id_db.dbsize()))


"""
Script for extracting relations from Freebase dump.
Use mongodb to index the extracted relations
"""

import io
import gzip
import logging
from dds.data_utils.db_helper import relation_collection, pair_collection

logger = logging.getLogger(__name__)


def get_fb_relations():
    cnt = 0

    with io.TextIOWrapper(gzip.open("../data/freebase/freebase-rdf-latest.gz", "r")) as freebase:
        for line in freebase:
            tokens = line.split('\t')

            en1 = tokens[0].split('/')[-1][:-1]
            en2 = tokens[2].split('/')[-1][:-1]
            if en1.startswith('m.') and en2.startswith('m.'):
                rel = tokens[1].split('/')[-1][:-1]
                post = {'en1id': en1,
                        'en2id': en2,
                        'rel': rel
                        }
                # only add relation if entities are already in the pair collection
                cur = pair_collection.find({'en1id': en1, 'en2id': en2})
                if cur.count() > 0:
                    # add relation to collection and increase count if the triple has been added already
                    relation_collection.update_one(post, {'$inc': {'cnt': 1}}, upsert=True)
                    logger.debug('En1 {}, En2 {}, Rel {}'.format(en1, en2, rel))

            cnt += 1

            if cnt % 10000000 == 0:
                logger.info("{} lines processed, dict_size={}".format(cnt, relation_collection.count()))


def retrieve_db():
    cur = relation_collection.find()
    relations = set()
    for record in cur:
        relations.add(record['rel'])

    with open('../data/freebase/relations.txt') as f:
        f.write('NA\n')
        for rel in relations:
            f.write('%s\n' % (rel))
    return relations


if __name__ == '__main__':
    get_fb_relations()
    retrieve_db()

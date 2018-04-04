"""
Script for extracting relations from Freebase dump.
Use mongodb to index the extracted relations
"""

import io
import gzip
import logging
from db_helper import relation_collection

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


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
                relation_collection.update_one(post, {'$inc': {'cnt': 1}}, upsert=True)
                logger.debug('En1 {}, En2 {}, Rel {}'.format(en1, en2, rel))
                print('En1 {}, En2 {}, Rel {}'.format(en1, en2, rel))

            cnt += 1

            if cnt % 10000000 == 0:
                logger.info("{} lines processed, dict_size={}".format(cnt, relation_collection.count()))


if __name__ == '__main__':
    get_fb_relations()

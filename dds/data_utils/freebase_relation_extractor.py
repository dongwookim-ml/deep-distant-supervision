"""
Script for extracting relations from Freebase dump.
Use mongodb to index the extracted relations
"""

import io
import gzip
import logging
from collections import defaultdict
from dds.data_utils.db_helper import relation_collection, pair_collection, id2name_db, pair_count

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
    rel_count = defaultdict(int)
    rel_num_sen = defaultdict(int)

    with open('../../data/freebase/triples.txt', 'w') as f:
        for record in cur:
            en1id = record['en1id']
            en2id = record['en2id']
            rel = record['rel']
            cur2 = pair_count.find({'en1id':en1id, 'en2id':en2id})
            rel_count[rel] += 1
            for _record in cur2:
                rel_num_sen[rel] += _record['cnt']
            en1name = id2name_db.get(en1id)
            en2name = id2name_db.get(en2id)
            f.write('%s\t%s\t%s\t%s\t%s\n' % (en1id, en2id, en1name, en2name, rel))
            print('%s\t%s\t%s\t%s\t%s\n' % (en1id, en2id, en1name, en2name, rel))

    with open('../../data/freebase/relations.txt', 'w') as f:
        f.write('NA\n')
        for key, value in rel_count.items():
            f.write('%s %d %d\n' % (key, value, rel_num_sen[key]))
    return rel_count


if __name__ == '__main__':
    # get_fb_relations()
    retrieve_db()


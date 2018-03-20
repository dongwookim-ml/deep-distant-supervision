"""
Script for extracting entities from Freebase dump.
"""

import io
import gzip

cnt = 0
en_dict = dict()

old_dicts = list()

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
                    en_dict[name] = mid

        cnt += 1

        if cnt % 10000000 == 0:

            if len(en_dict) > 500000:
                # create new dictionary if it has more than 100000 entries
                old_dicts.append(en_dict)
                en_dict = dict()

            print("{} lines processed, num_dict={}, dict_size={}".format(cnt, len(old_dicts), len(en_dict)))

        if cnt > 3160000000:
            break

size = len(en_dict)
for old_dict in old_dicts:
    size += len(old_dict)
    en_dict.update(old_dict)

print('Pre-merged dict size {}'.format(size))
print('Final dict size {}'.format(len(en_dict)))

with io.TextIOWrapper(open('../data/freebase/dict.txt', 'wb')) as f:
    for key, value in en_dict.items():
        f.write("{}\t{}\n".format(key, value))



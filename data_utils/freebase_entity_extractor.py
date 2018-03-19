"""
Script for extracting entities from Freebase dump.
Extremely slow. It would be better to stop the process after obtaining 20 million entities.
    (It seems that the total number of entities are around 20 millions.)
"""

import io
import gzip
from collections import defaultdict

# constructor of nested default dictionary
nested_dd = lambda: defaultdict(nested_dd)

cnt = 0
en_dict = nested_dd()

with io.TextIOWrapper(gzip.open("freebase-rdf-latest.gz", "r")) as freebase:
    with io.TextIOWrapper(open("freebase_dict.txt", 'wb'), encoding="utf-8") as f:
        line = freebase.readline()
        tmp = ""
        while line is not None:
            if "type.object.name" in line:
                tokens = line.split('\t')
                mid = tokens[0].split('/')[-1][:-1]
                name = tokens[2]
                if "@en" in name:
                    name = name.replace("@en", "")
                    name = name.replace("\"", "")
                    # use the first two characters to access the first level dict
                    en_dict[name[:2]][name] = mid
                    # en_dict[name]=mid
            line = freebase.readline()

            cnt += 1

            if cnt % 10000000 == 0:
                num_dict = 0
                size = 0
                for key, value in en_dict.items():
                    size += len(en_dict[key])
                    num_dict += 1
                print("{} lines processed, num_dict={}, dict_size={}".format(cnt, num_dict, size))

        for key, _ in en_dict.items():
            for key2, value2 in en_dict[key].items():
                f.write("{}\t{}\n".format(key2, value2))

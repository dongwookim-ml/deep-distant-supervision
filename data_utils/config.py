"""
List of global variables used to configure the project
"""
METADB = 0
NAME2ID = 1
ID2NAME = 2

REDIS_HOST = 'localhost'
REDIS_PORT = 6379

MONGO_HOST = 'localhost'
MONGO_PORT = 27017

MG_FB = 'wiki-freebase'
MG_FBDICT = 'fb_dict'
MG_WIKI_DB = 'wikipedia'
MG_SENTENCE_COL = 'sentence_col'
MG_PAIR_COL = 'pair_col'
MG_PAIR_CNT = 'pair_cnt'

IGNORE_NER_TAG = ('O', 'DATE', 'NUMBER', 'ORDINAL', 'PERCENT', 'MONEY', 'DURATION')  # Non-NER tag


MIN_SENTENCE_LENGTH = 20

"""
Parsing Wikipedia corpus to obtain training set for distant supervision,

reference: https://radimrehurek.com/topic_modeling_tutorial/2 - Topic Modeling.html
"""

import logging
import itertools

import numpy as np
import gensim

from gensim.utils import smart_open
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki

from nltk.tag.stanford import CoreNLPNERTagger
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore


def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))


def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        sentences = sent_tokenize(text)
        for sent in sentences:
            tokens = word_tokenize(sent)
            if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
                continue  # ignore short articles and various meta-articles
            yield title, tokens


raw_path = "../data/wikipedia/enwiki-20180301-pages-articles.xml.bz2"
server_url = 'http://localhost:9000'

stream = iter_wiki(raw_path)
for title, tokens in itertools.islice(stream, 8):
    tagger = CoreNLPNERTagger(url=server_url)
    tagged_text = tagger.tag(tokens)
    print(tagged_text)

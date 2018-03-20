"""
Utils for parsing Wikipedia corpus to obtain training set for distant supervision

"""

import logging
import itertools
import data_utils

from gensim.utils import smart_open
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki

from nltk.tag.stanford import CoreNLPNERTagger
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

raw_path = "../data/wikipedia/enwiki-20180301-pages-articles.xml.bz2"
server_url = 'http://localhost:9000'  # Stanford corenlp server address


def iter_wiki(dump_file=raw_path):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple.
    Reference: https://radimrehurek.com/topic_modeling_tutorial/2 - Topic Modeling.html
    """
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        sentences = sent_tokenize(text)
        for sent in sentences:
            tokens = word_tokenize(sent)
            if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
                continue  # ignore short articles and various meta-articles
            yield title, tokens

if __name__ == '__main__':
    stream = iter_wiki(raw_path)
    tagger = CoreNLPNERTagger(url=server_url)
    for title, tokens in itertools.islice(stream, 8):
        tagged_text = tagger.tag(tokens)
        print(title, tagged_text)
        data_utils.extract_ners(tagged_text)


"""
Utils for enumerating Wikipedia corpus
"""

import json
import logging
import itertools
from config import *
from datautils import extract_ners
from gensim.utils import smart_open

from nltk.tag.stanford import CoreNLPNERTagger
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

# run `python -m gensim.scripts.segment_wiki -i -f enwiki-20180301-pages-articles.xml.bz2 -o enwiki.json.gz`
# raw_path = "../data/wikipedia/enwiki-20180301-pages-articles.xml.bz2"
raw_path = "../data/wikipedia/enwiki.json.gz"
server_url = 'http://localhost:9000'  # Stanford corenlp server address


def iter_wiki(dump_file=raw_path):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple.
    Reference: https://radimrehurek.com/topic_modeling_tutorial/2 - Topic Modeling.html
    """
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for line in smart_open(dump_file):
        article = json.loads(line)

        title = article['title']
        if any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue
        for section in article['section_texts']:
            for sentence in sent_tokenize(section):
                tokens = word_tokenize(sentence)
                if len(tokens) > MIN_SENTENCE_LENGTH:
                    yield title, tokens


if __name__ == '__main__':
    stream = iter_wiki(raw_path)
    tagger = CoreNLPNERTagger(url=server_url)
    for title, tokens in itertools.islice(stream, 8):
        tagged_text = tagger.tag(tokens)
        print(title, tagged_text)
        extract_ners(tagged_text)

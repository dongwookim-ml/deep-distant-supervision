"""
Script for extracting sentences along with freebase entities in the sentences.

Redis key-value database is used to store and retrieve freebase entity by freebase id and surface form
MongoDB is used to store extracted sentences and entities

"""
import sys, traceback
from dds.data_utils.datautils import merge_ners, get_nerspos
from dds.data_utils.wikipedia import iter_wiki
from dds.data_utils.db_helper import add_sentence, add_pair, lookup_fb, get_wiki_parsed_stat, set_wiki_parsed_cnt
import itertools
import logging
from nltk.tag.stanford import CoreNLPNERTagger
import concurrent.futures as confu

# Logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(ch)
logger.setLevel('INFO')


def extract_ners(tokens, tagger):
    """
    Extract freebase entities (NERs) from token using NER tagger.

    :param tokens: list of tokens forming a sentence
    :param tagger: NER tagger return list of tokens with corresponding NER tags
    :return: if there are more than two freebase entities in the sentence, then return list of tokens,
    ners, and freebase ner ids. each ner is again a list of tokens.
    """
    try:
        tagged_text = tagger.tag(tokens)
    except:
        logger.info('NER error {}'.format(len(tokens)))
        return None

    ners, merged_tokens = merge_ners(tagged_text)

    valid_ners = list()
    valid_ner_id = list()
    for ner in ners:
        # ner = ((tokens), tag)
        rval = lookup_fb(ner[0])
        if rval is not None:
            if ner[0] not in valid_ners:
                valid_ners.append(ner[0])
                valid_ner_id.append(rval)

    if 2 <= len(valid_ners) <= 4:
        # we only take into account the sentences with a few number of entities
        return merged_tokens, valid_ners, valid_ner_id

    return None


def process_sentence(tokens, ners, ner_id):
    """
    Find position (starting from 0) of NERs in the list of tokens.
    For all possible pairs in NERs, yield their positions in the token list
    :param tokens: list of tokens forming a sentence
    :param ners: list of ners in the token list
    :param ner_id: list of freebase id for each ner
    """
    ner_pos = get_nerspos(tokens, ners)

    sid = add_sentence(tokens, ners, ner_pos)

    num_ner = len(ner_pos)
    for first_ner, second_ner in itertools.product(range(num_ner - 1), range(1, num_ner)):
        for pos1, pos2 in itertools.product(ner_pos[first_ner], ner_pos[second_ner]):
            yield (ner_id[first_ner], ner_id[second_ner], pos1, pos2, sid)


if __name__ == '__main__':
    num_thread = 8
    server_url = 'http://localhost:9000'  # Stanford corenlp server address
    stream = iter_wiki()
    tagger = CoreNLPNERTagger(url=server_url)

    token_list = list()

    _skip = get_wiki_parsed_stat()

    with confu.ThreadPoolExecutor(num_thread) as executor:
        try:
            for i, (title, tokens) in enumerate(stream):
                if i < _skip:
                    continue
                elif i == _skip:
                    logger.info('{} Lines are passed'.format(i))

                token_list.append(tokens)

                if len(token_list) == num_thread:
                    futures = [executor.submit(extract_ners, tokens, tagger) for tokens in token_list]

                    for future in confu.as_completed(futures):
                        result = future.result()
                        if result is not None:
                            tokens = result[0]
                            ners = result[1]
                            ner_id = result[2]
                            logger.debug('Sentence {}'.format(tokens))
                            logger.debug('NERS {}'.format(ners))
                            for en1id, en2id, en1pos, en2pos, sid in process_sentence(tokens, ners, ner_id):
                                add_pair(en1id, en2id, en1pos, en2pos, sid)

                    token_list = list()

                if i % 1000 == 0:
                    logger.info('{} lines processed'.format(i))

                set_wiki_parsed_cnt(i)

        except KeyboardInterrupt:
            traceback.print_exc(file=sys.stdout)
            set_wiki_parsed_cnt(i)
            sys.exit(0)

        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            set_wiki_parsed_cnt(i)

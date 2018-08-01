import logging, sys, itertools
import numpy as np
import torch
from torch import nn
from dds.pytorch.model import DDS
from dds.data_fetcher import NYTFetcher, OOV, max_sen_len, max_sens
from dds.data_utils.wiki_extractor import extract_ners
from dds.data_utils.datautils import get_nerspos
from nltk.tag.stanford import CoreNLPNERTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger = logging.getLogger()
logger.addHandler(ch)
logger.setLevel('INFO')


class Predictor:
    def __init__(self, model_path, embed_dim, w2v_path, rel_path, sen_path, ner_tagger):
        self.fetcher = NYTFetcher(w2v_path, rel_path, embed_dim, sen_path)
        self.model = torch.load(model_path)
        self.tagger = ner_tagger
        self.sentence_collection = dict()
        self.entity_pair_dict = defaultdict(list)
        self.word2id = self.fetcher.word2id

    def extract_corpus(self, corpus):
        """
        enumerate corpus and generate all pair of entities and corresponding sentences.

        :param corpus: iterable, list of documents
        :return:
        """

        for document in corpus:
            for sentence in sent_tokenize(document):
                tokens = word_tokenize(sentence)
                rval = extract_ners(tokens, self.tagger, lookup=False)
                if rval is not None:
                    sid = len(self.sentence_collection)
                    merged_tokens, ners, _ = rval
                    self.sentence_collection[sid] = merged_tokens
                    ner_pos = get_nerspos(merged_tokens, ners)

                    num_ner = len(ner_pos)
                    for first_ner, second_ner in itertools.product(range(num_ner - 1), range(1, num_ner)):
                        for pos1, pos2 in itertools.product(ner_pos[first_ner], ner_pos[second_ner]):
                            self.entity_pair_dict[(ners[first_ner], ners[second_ner])].append((pos1,pos2,sid))


    def convert_datapoint(self, collection):
        x = list()
        for en1pos, en2pos, sid in collection:
            sentence = self.sentence_collection[sid]
            sen2id = list()
            for token in sentence:
                if type(token) == list:
                    entity = ' '.join(token)
                    if entity in self.word2id:
                        token_id = self.word2id[entity]
                    else:
                        token_id = self.word2id[OOV]
                else:
                    if token in self.word2id:
                        token_id = self.word2id[token]
                    else:
                        token_id = self.word2id[OOV]
                sen2id.append(token_id)

            sen2id = np.array(sen2id)
            sen_len = len(sen2id)
            pos1vec = np.arange(max_sen_len - en1pos, max_sen_len - en1pos + sen_len)
            pos2vec = np.arange(max_sen_len - en2pos, max_sen_len - en2pos + sen_len)
            x.append((sen2id, pos1vec, pos2vec))
        return x



    def test(self, corpus):
        self.extract_corpus(corpus)
        with torch.no_grad():
            predicted_y = dict()
            for entity_pair, collection in self.entity_pair_dict.items():
                x = self.convert_datapoint(collection)
                output = self.model(x)
                y = output.data.numpy()
                predicted_y[entity_pair] = y

        # do something with dictionary
        return predicted_y


    def extract_topk(self, result, k):
        score_matrix = np.zeros((len(result), self.fetcher.num_rel))

        pair_list = list()
        for i, (en_pair, scores) in enumerate(result.items()):
            pair_list.append(en_pair)
            score_matrix[i] = scores

        score_matrix = score_matrix[:, 1:]

        top_index = np.unravel_index(np.argsort(np.ravel(score_matrix))[::-1][:k], score_matrix.shape)

        rval = list()
        for item in np.transpose(top_index):
            pair_no = item[0]
            rel_no = item[1]
            logger.info(pair_list[pair_no], self.fetcher.id2rel[rel_no+1])
            rval.append((pair_list[pair_no], self.fetcher.id2rel[rel_no+1]))

        return rval


if __name__ == '__main__':
    embed_dim = 50
    logger.info('Loading Fetcher')
    w2v_path = '../../data/word2vec.txt'
    rel_path = '../../data/nyt/relation2id.txt'
    sen_path = '../../data/nyt/train.txt'
    model_path = 'saved_model_0.tmp'
    ner_server_url = 'http://localhost:9000'  # Stanford corenlp server address
    ner_tagger = CoreNLPNERTagger(url=ner_server_url)

    predictor = Predictor(model_path, embed_dim, w2v_path, rel_path, sen_path, ner_tagger)
    corpus = [('hello my name is Dongwoo, and your name is Minjeong'), ('This is test document written in New York and Atlanta')]
    result = predictor.test(corpus)
    predictor.extract_topk(result, 10)



import os
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


def get_model_dir(config, exceptions=('help', 'helpful', 'helpshort')):
    """ Return model save path with model parameters
    :param config: tf.app.flags.FLAGS, configuration values will be used to generate the path where the model is saved
    :param exceptions: iterable, will not be included in the saving path
    :return: model save path
    """

    attrs = config.__dict__['__wrapped']

    names = []
    keys = dir(attrs)
    for key in keys:
        if key not in exceptions:
            names.append("%s=%s" % (key, ",".join([str(i) for i in config[key].value]) if type(
                config[key].value) == list else config[key].value))
    return os.path.join('checkpoints', *names) + '/'


def unstack_next_batch(model, triple_batch, pos1_batch, pos2_batch, y_batch):
    """
    Unstack original triple data into feeddict
    :param triple_batch: batch of triples containing corresponding sentences
    :param pos1_batch: batch of triples containing relative position w.r.t the first entity
    :param pos2_batch: batch of triples containing relative position w.r.t the first entity
    :param y_batch: batch of one-hot vector of triples
    :param batch_size: the number of triples to be fetched
    :return: feed_dict
    """
    num_sentences = 0
    feed_sentences = []
    feed_pos1 = []
    feed_pos2 = []
    triple_index = np.zeros(len(triple_batch) + 1)

    for i in range(len(triple_batch)):
        triple_index[i] = num_sentences
        num_sentences += len(triple_batch[i])
        for sentence, pos1, pos2 in zip(triple_batch[i], pos1_batch[i], pos2_batch[i]):
            feed_sentences.append(sentence)
            feed_pos1.append(pos1)
            feed_pos2.append(pos2)

    triple_index[-1] = num_sentences
    feed_sentences = np.array(feed_sentences)
    feed_pos1 = np.array(feed_pos1)
    feed_pos2 = np.array(feed_pos2)
    feed_y = y_batch

    feed_dict = {}
    feed_dict[model.input_sentences] = feed_sentences
    feed_dict[model.input_pos1] = feed_pos1
    feed_dict[model.input_pos2] = feed_pos2
    feed_dict[model.input_y] = feed_y
    feed_dict[model.input_triple_index] = triple_index

    return feed_dict

import os
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


def get_model_dir(config, exceptions=('help', 'helpfull', 'helpshort')):
    """ Return model save path with model parameters
    :param config: tf.app.flags.FLAGS, configuration values will be used to generate the path where the model is saved
    :param exceptions: iterable, will not be included in the saving path
    :return: model save path
    """

    try:
        attrs = config.flag_values_dict()
    except:
        # for tensorflow 1.2 (not sure which other versions use this structure except 1.2)
        attrs = config.__flags

    names = []
    for key, value in attrs.items():
        if key not in exceptions:
            names.append("%s=%s" % (key, ",".join([str(i) for i in value]) if type(
                value) == list else value))
    return os.path.join('./checkpoints', *names) + '/'

def unstack_data(train_x, train_y, max_sen):
    """ If a certain triple has more than max_sen sentences divide it into small pieces """
    unstacked_x = list()
    unstacked_y = list()

    for x, y in zip(train_x, train_y):
        if len(x) > max_sen:
            total_sen = len(x)
            for i in range(int(total_sen/max_sen)+1):
                if i*max_sen != min((i+1)*max_sen, total_sen):
                    unstacked_x.append(x[i*max_sen:min((i+1)*max_sen, total_sen)])
                    unstacked_y.append(y)
        else:
            unstacked_x.append(x)
            unstacked_y.append(y)

    return np.array(unstacked_x), np.array(unstacked_y)

def unstack_next_batch(model, x_batch, y_batch):
    """
    Unstack original triple data into feeddict
    :param x_batch: tuple of list of list, batch of sentences of triples
    :param y_batch: batch of one-hot vector of triples
    :return: feed_dict
    """
    cumsum_sentences = 0
    feed_sentences = []
    feed_pos1 = []
    feed_pos2 = []
    triple_index = np.zeros(len(x_batch) + 1)

    for i in range(len(x_batch)):
        triple_index[i] = cumsum_sentences
        num_sen = len(x_batch[i])
        cumsum_sentences += num_sen

        for j in range(num_sen):
            # TODO: Positions can be computed on the fly if the position of two entities are given.
            # TODO: This will reduce IO time to load dataset
            word_list = []
            pos1_list = []
            pos2_list = []
            for word, pos1, pos2 in x_batch[i][j]:
                word_list.append(word)
                pos1_list.append(pos1)
                pos2_list.append(pos2)
            feed_sentences.append(word_list)
            feed_pos1.append(pos1_list)
            feed_pos2.append(pos2_list)

    triple_index[-1] = cumsum_sentences
    feed_sentences = np.array(feed_sentences)
    feed_pos1 = np.array(feed_pos1)
    feed_pos2 = np.array(feed_pos2)
    feed_y = y_batch

    feed_dict = dict()
    feed_dict[model.input_sentences] = feed_sentences
    feed_dict[model.input_pos1] = feed_pos1
    feed_dict[model.input_pos2] = feed_pos2
    feed_dict[model.input_y] = feed_y
    feed_dict[model.input_triple_index] = triple_index

    return feed_dict

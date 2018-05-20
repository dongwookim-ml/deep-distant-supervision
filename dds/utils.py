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


def unstack_next_batch(model, fetcher, conf):
    """
    Obtain training data point from fetcher and construct feed dict
    """
    batch_size = conf.batch_size
    max_sen = conf.max_batch_sentences
    cumsum_sentences = 0
    feed_sentences = []
    feed_pos1 = []
    feed_pos2 = []
    triple_index = np.zeros(batch_size + 1)
    feed_y = []

    for i in range(batch_size):
        triple_index[i] = cumsum_sentences

        try:
            x, y = next(fetcher)
        except StopIteration:
            return None

        if len(x) > max_sen:
            np.random.shuffle(x)
            x = x[:max_sen]
        for sen, pos1, pos2 in x:
            sen = list(sen)
            pos1 = list(pos1)
            pos2 = list(pos2)
            if len(sen) < conf.len_sentence:
                for padding in range(conf.len_sentence - len(sen)):
                    # padding sentence and position vectors for static computational graph model with fixed sentence len
                    sen.append(0)
                    pos1.append(pos1[-1]+1)
                    pos2.append(pos2[-1]+1)
            feed_sentences.append(sen)
            feed_pos1.append(pos1)
            feed_pos2.append(pos2)
        feed_y.append(y)
        cumsum_sentences += len(x)

    triple_index[-1] = cumsum_sentences

    feed_dict = dict()
    feed_dict[model.input_sentences] = feed_sentences
    feed_dict[model.input_pos1] = feed_pos1
    feed_dict[model.input_pos2] = feed_pos2
    feed_dict[model.input_y] = np.array(feed_y)
    feed_dict[model.input_triple_index] = triple_index

    return feed_dict


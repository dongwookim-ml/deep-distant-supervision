import logging
import tensorflow as tf
import numpy as np
import model
import os
import sys
from utils import get_model_dir, unstack_next_batch

flags = tf.app.flags

# Model
flags.DEFINE_boolean('word_attn', True, 'Whether to use word-level attention')
flags.DEFINE_boolean('sent_attn', True, 'Whether to use sentence-level attention')
flags.DEFINE_integer('num_hidden', 256, 'The number of hidden unit')
flags.DEFINE_integer('num_filter', 32, 'The number of filter in cnn (if network_type=cnn)')
flags.DEFINE_integer('pos_dim', 5, 'The dimensionality of position embedding')
flags.DEFINE_boolean('bidirectional', True, 'Whether to define bidirectional rnn')
flags.DEFINE_integer('num_relation', 53, 'The number of relations to be classified')
flags.DEFINE_integer('max_position', 123, 'The upper bound on relative position')
flags.DEFINE_integer('len_sentence', 70, 'The upper bound on sentence length')
flags.DEFINE_boolean('dropout', True, 'If true, apply dropout layer after rnn layer')
flags.DEFINE_float('keep_prob', 0.5, 'Dropout: probability of keeping a variable')
flags.DEFINE_string('save_path', '', 'Model save path')
flags.DEFINE_string('network_type', 'rnn', 'Model type [rnn, cnn]')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_boolean('load_prev', True, 'Restore previously trained model if True')
flags.DEFINE_integer('batch_size', 48,
                     'The size of batch for minibatch training (number of triples to be mini-batched)')
flags.DEFINE_boolean('pretrained_w2v', True,
                     'Use pretrained word2vec if True, note that the word id should be aligned with the word2vec id')
flags.DEFINE_string('w2v_path', 'data/vec.npy', 'Path to the pretrained word2vec')
flags.DEFINE_integer('num_epoch', 10, 'The number of epochs used for training')
flags.DEFINE_integer('max_batch_sentences', 1500, 'The maximum number of sentences to be batched for each triple')
flags.DEFINE_string('dataset', 'data/nyt', 'path to the dataset')
flags.DEFINE_integer('print_gap', 50, 'Print status every print_gap iteration')
flags.DEFINE_integer('save_gap', 1000, 'Save model every save_gap iteration to save_path')

# Optimizer
flags.DEFINE_float('reg_weight', 0.0001, 'Weight of regularizer on model parameters')
flags.DEFINE_float('learning_rate', 0.001, 'The learning rate of training')
flags.DEFINE_float('beta1', 0.9, 'Beta 1 of Adam optimizer')
flags.DEFINE_float('beta2', 0.999, 'Beta 2 of Adam optimizer')
flags.DEFINE_float('epsilon', 1e-8, 'Epsilon of Adam optimizer')

# Debug
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

conf = flags.FLAGS

# These fields will be excluded in the automated model path. see utils.get_model_dir
exclude_list = ['save_path',
                'num_relation',
                'num_epoch',
                'load_prev',
                'h',
                'help',
                'helpfull',
                'helpshort',
                'save_gap',
                'print_gap'
                ]

# Logger
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger = logging.getLogger()
logger.addHandler(ch)
logger.setLevel(conf.log_level)


def train(train_x, train_y, conf, save_path):
    with tf.Session() as sess:
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            if conf.pretrained_w2v:
                word_embedding = np.load(conf.w2v_path)
                nre = model.NRE(conf, word_embedding)
            else:
                nre = model.NRE(conf)

        num_triples = len(train_x)  # total number of triples to be trained
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate, beta1=conf.beta1, beta2=conf.beta2,
                                           epsilon=conf.epsilon)

        train_op = optimizer.minimize(nre.final_loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs/%s' % (save_path), graph=sess.graph)

        saver = tf.train.Saver()
        last_ckpt = tf.train.latest_checkpoint(save_path)
        if last_ckpt and conf.load_prev:
            logger.info("Restore last session from {}".format(last_ckpt))
            saver.restore(sess, last_ckpt)
            logger.info("Last Session Restored")

        for one_epoch in range(conf.num_epoch):
            # randomly shuffle index of training set
            random_ordered_idx = np.arange(num_triples)
            np.random.shuffle(random_ordered_idx)
            total_batch = int(num_triples / float(conf.batch_size))

            for i in range(total_batch):
                random_idx = random_ordered_idx[i * conf.batch_size: (i + 1) * conf.batch_size]
                batch_x = train_x[random_idx]
                batch_y = train_y[random_idx]

                feed_dict = unstack_next_batch(nre, batch_x, batch_y, conf.max_batch_sentences)

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, nre.total_loss, nre.accuracy, merged_summary, nre.l2_loss,
                     nre.final_loss], feed_dict)

                summary_writer.add_summary(summary, global_step=step)

                if step % conf.print_gap == 0:
                    acc = np.mean(np.array(accuracy))
                    logger.info("step {}, softmax_loss {:g}, acc {:g}".format(step, loss, acc))
                if step % conf.save_gap == 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    saver.save(sess, save_path, global_step=step)
                    logger.info("Model saved at step {}".format(step))


def main(_):
    dataset = conf.dataset

    save_path = get_model_dir(conf, exclude_list)
    if conf.save_path != '':
        save_path = conf.save_path
    logger.info("Model path {}".format(save_path))

    if conf.is_train:
        train_x = np.load(dataset + '/train_x.npy')
        train_y = np.load(dataset + '/train_y.npy')
        train(train_x, train_y, conf, save_path)
    else:
        pass


if __name__ == '__main__':
    tf.app.run()

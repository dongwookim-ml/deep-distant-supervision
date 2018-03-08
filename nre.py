import logging
import tensorflow as tf
from tensorflow.contrib import rnn

logger = logging.getLogger(__name__)


class NRE:
    """Neural Relation Extractor
    """

    def __init__(self, conf, pre_word2vec=None, activate_fn=tf.nn.tanh):
        pretrained_w2v = conf.pretrained_w2v
        max_position = conf.max_position
        pos_dim = conf.pos_dim
        num_relation = conf.num_relation
        len_sentence = conf.len_sentence
        num_hidden = conf.num_hidden
        batch_size = conf.batch_size
        reg_weight = conf.reg_weight

        self.input_sentences = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_sentence')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_position1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_position2')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, num_relation], name='input_y')
        self.input_triple_index = tf.placeholder(dtype=tf.int32, shape=[None])

        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []

        num_sentences = self.input_triple_index[-1]

        if pretrained_w2v:
            self.word2vec = tf.get_variable(initializer=pre_word2vec, name="word_embedding")
        else:
            self.word2vec = tf.get_variable(shape=[conf.voca_size, conf.word_embedding_dim], name="word_embedding")

        self.pos2vec1 = tf.get_variable(shape=[max_position, pos_dim], name="pos2vec1")
        self.pos2vec2 = tf.get_variable(shape=[max_position, pos_dim], name="pos2vec2")

        # concatenate word embedding + position embeddings
        # input_forward & backward = [num_sentence, len_sentence, w2v_dim+2*p2v_dim]
        input_forward = tf.concat([tf.nn.embedding_lookup(self.word2vec, self.input_sentences),
                                   tf.nn.embedding_lookup(self.pos2vec1, self.input_pos1),
                                   tf.nn.embedding_lookup(self.pos2vec2, self.input_pos2)], 2)

        input_forward = tf.unstack(input_forward, len_sentence, 1)

        # forward and backward cell
        rnn_fw_cell = rnn.GRUCell(num_hidden, name='forward-gru')
        rnn_bw_cell = rnn.GRUCell(num_hidden, name='backward-gru')
        num_hidden_rnn = 2 * num_hidden

        if conf.dropout and conf.is_train:
            rnn_fw_cell = rnn.DropoutWrapper(rnn_fw_cell, output_keep_prob=conf.keep_prob)
            rnn_bw_cell = rnn.DropoutWrapper(rnn_bw_cell, output_keep_prob=conf.keep_prob)

        with tf.variable_scope("RNN"):
            output_rnn, _, _ = rnn.static_bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, input_forward,
                                                            dtype=tf.float32)
            output_rnn = tf.reshape(output_rnn, [num_sentences, len_sentence, num_hidden_rnn])

        # word-level attention layer, represent a sentence as a weighted sum of word vectors
        with tf.variable_scope("word-attn"):
            if conf.word_attn:
                word_attn = tf.get_variable('W', shape=[num_hidden_rnn, 1])
                word_weight = tf.matmul(
                    tf.reshape(activate_fn(output_rnn), [num_sentences * len_sentence, num_hidden_rnn]), word_attn)
                word_weight = tf.reshape(word_weight, [num_sentences, len_sentence])
                sentence_embedding = tf.matmul(
                    tf.reshape(tf.nn.softmax(word_weight, axis=1), [num_sentences, 1, len_sentence]),
                    output_rnn)
                sentence_embedding = tf.reshape(sentence_embedding, [num_sentences, num_hidden_rnn])
            else:
                sentence_embedding = tf.reduce_mean(output_rnn, 1)

        # sentence-level attention layer, represent a triple as a weighted sum of sentences
        with tf.variable_scope("sentence-attn"):
            rel_embedding = tf.get_variable('R', shape=[num_relation, num_hidden_rnn])
            rel_bias = tf.get_variable('R_bias', shape=num_relation)
            weight = tf.get_variable("W", shape=[num_hidden_rnn, 1])

            for i in range(batch_size):
                target_sentences = sentence_embedding[self.input_triple_index[i]:self.input_triple_index[i + 1]]

                if conf.sent_attn:
                    num_triple_sentence = self.input_triple_index[i + 1] - self.input_triple_index[i]
                    triple_sentences = activate_fn(target_sentences)
                    sentence_weight = tf.reshape(
                        tf.nn.softmax(tf.reshape(tf.matmul(triple_sentences, weight), [num_triple_sentence])),
                        [1, num_triple_sentence])
                    triple_embedding = tf.reshape(tf.matmul(sentence_weight, target_sentences), [num_hidden_rnn, 1])
                else:
                    triple_embedding = tf.reduce_mean(target_sentences, 0)

                triple_output = tf.add(tf.reshape(tf.matmul(rel_embedding, triple_embedding), [num_relation]), rel_bias)

                self.prob.append(tf.nn.softmax(triple_output))
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))
                self.loss.append(
                    tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=triple_output, labels=self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

        tf.summary.scalar("loss", self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(reg_weight),
            weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar("l2_loss", self.l2_loss)
        tf.summary.scalar("final_loss", self.final_loss)

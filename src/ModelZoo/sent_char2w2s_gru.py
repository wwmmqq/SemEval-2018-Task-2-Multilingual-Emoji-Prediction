# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from encoder import char2word_rnn, word2sent


class Model(object):
    def __init__(self, config):
        # base config
        self.name = 'c2w2s'
        self.max_len_sent = config.max_len_sent
        self.max_len_word = config.max_len_word
        self.dim_char = config.dim_char
        self.dim_word = config.dim_word
        self.n_rnn = config.n_rnn
        self.n_mlp = config.n_mlp
        self.n_class = config.n_class
        self.max_gradient_norm = config.max_gradient_norm

        self.global_step = tf.Variable(
            0, name="g_step", trainable=False)
        self.learning_rate = tf.Variable(
            config.learning_rate, name='lr', trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate.value()*0.1)
        self.drop_keep_rate = tf.placeholder(
            tf.float32, name='keep_prob')
        self.bug = None

        # Token Embedding
        if config.ce is not None:
            self.ce = tf.Variable(config.ce, name='char_emb')
        else:
            self.ce = tf.get_variable(
                'char_embedding',
                [config.char_size, self.dim_char],
                tf.float32,
                initializer=tf.random_uniform_initializer(-0.25, 0.25))
            print('get random ce (char embedding).')

        # Build the Computation Graph
        self._build_model()
        # # Set loss
        self._set_cost_and_optimize()
        # # Set prediction and acc
        self._set_predict()
        # # add tensor board
        self._log_summaries()
        # # ModelZoo parameter saver
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        # Model PlaceHolder for input
        # shape: [batch, max_seq, max_len_word]
        self.in_chars = tf.placeholder(
            tf.int32,
            [None, self.max_len_sent, self.max_len_word])

        self.in_len_sent = tf.placeholder(tf.int32, [None])
        self.in_len_word = tf.placeholder(
            tf.int32, [None, self.max_len_sent], name='in_len_w')
        self.in_y = tf.placeholder(tf.int32, [None])

        seqs_embed = char2word_rnn(
            self.in_chars, self.in_len_word, self.ce, 200, self.dim_word)
        sent = word2sent(seqs_embed, self.in_len_sent, self.n_rnn)

        with tf.name_scope('logits'), tf.variable_scope('logits'):
            self.logits = tf.layers.dense(sent, self.n_class)

    def _set_cost_and_optimize(self):
        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.in_y))
        self.cost = softmax_cost
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, train_vars), self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(
            list(zip(grads, train_vars)),
            global_step=self.global_step)

    def _set_predict(self):
        self.y_prob = tf.nn.softmax(self.logits)
        self.y_p = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        # Accuracy
        check_prediction = tf.equal(self.y_p, self.in_y)
        self.acc_num = tf.reduce_sum(tf.cast(check_prediction, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(check_prediction, tf.float32))

    def _log_summaries(self):
        """
        Adds summaries for the following variables to the graph and returns
        an operation to evaluate them.
        """
        cost = tf.summary.scalar("loss", self.cost)
        acc = tf.summary.scalar("acc", self.acc)
        self.merged = tf.summary.merge([cost, acc])

    def model_debug(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_len_sent: batch[2],
            self.in_y: batch[3],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.bug]

        return sess.run(return_list, feed_dict)

    def model_train(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_y: batch[1],
            self.in_len_word: batch[2],
            self.in_len_sent: batch[3],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [
            self.train_op, self.merged,
            self.global_step, self.cost, self.acc_num]

        return sess.run(return_list, feed_dict)

    def model_test(self, sess, batch):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_y: batch[1],
            self.in_len_word: batch[2],
            self.in_len_sent: batch[3],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.acc_num, feed_dict)

    def mode_predict(self, sess, batch):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_len_word: batch[2],
            self.in_len_sent: batch[3],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.y_p, feed_dict)

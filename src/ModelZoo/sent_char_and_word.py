# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import cnn2d


class Model(object):
    def __init__(self, config):
        # base config
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

        # Setup Model Parameters
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

        # Word Embedding
        if config.we is not None:
            self.we = tf.Variable(config.we, name='word_emb')
        else:
            self.we = tf.get_variable(
                'word_embedding',
                [config.vocab_size, self.dim_word],
                tf.float32,
                initializer=tf.random_uniform_initializer(-0.25, 0.25))
            print('get random we (word embedding).')

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
        self.in_words = tf.placeholder(
            tf.int32,
            [None, self.max_len_sent])

        self.in_len_sent = tf.placeholder(tf.int32, [None])
        self.in_y = tf.placeholder(tf.int32, [None])

        seq_char_word = []
        char_indices = tf.split(
            self.in_chars, self.max_len_sent, 1)
        word_indices = tf.split(
            tf.expand_dims(self.in_words, -1),
            self.max_len_sent, 1)

        with tf.variable_scope('char_word') as scope:
            for idx in range(self.max_len_sent):
                if idx != 0:
                    scope.reuse_variables()
                # [batch, 1, max_len_word] => [batch, max_len_word]
                char_index = tf.reshape(char_indices[idx], [-1, self.max_len_word])
                # [batch x max_len_sent, char_embed]
                char_embed = tf.nn.embedding_lookup(self.ce, char_index)
                char_cnn = cnn(
                    char_embed,
                    k_h_list=[1, 2, 3, 4, 5, 6],
                    k_w=self.dim_char,
                    k_nums=[50, 50, 50, 50, 50, 50])

                w_c = tf.get_variable(
                    'w_char',
                    shape=[300, self.dim_word],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(0.02))
                w_c_b = tf.get_variable(
                    'w_char_b',
                    [self.dim_word],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)
                char2w = tf.tanh(tf.nn.xw_plus_b(char_cnn, w_c, w_c_b))
                word_index = tf.reshape(word_indices[idx], [-1, 1])
                word_embed = tf.nn.embedding_lookup(self.we, word_index)
                char_with_word = tf.concat([char2w, tf.squeeze(word_embed, [1])], 1)
                seq_char_word.append(char_with_word)
        # seq = tf.stack(seq_char_word)
        seq = tf.reshape(seq_char_word, [-1, self.max_len_sent, 2*self.dim_word])
        with tf.name_scope('rnn'), tf.variable_scope('rnn'):
            cell_fw = tf.nn.rnn_cell.GRUCell(self.n_rnn)
            cell_bw = tf.nn.rnn_cell.GRUCell(self.n_rnn)
            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                seq,
                self.in_len_sent,
                dtype=tf.float32)
            out_rnn = tf.concat([b_states[0], b_states[1]], axis=-1)

        with tf.name_scope('mlp'), tf.variable_scope('mlp'):
            mlp = tf.layers.dense(
                out_rnn, self.n_mlp, activation=tf.nn.relu)

        with tf.name_scope('logits'), tf.variable_scope('logits'):
            self.logits = tf.layers.dense(mlp, self.n_class)

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
            self.in_words: batch[1],
            self.in_len_sent: batch[2],
            self.in_y: batch[3],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.bug]

        return sess.run(return_list, feed_dict)

    def model_train(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_words: batch[1],
            self.in_y: batch[2],
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
            self.in_words: batch[1],
            self.in_y: batch[2],
            self.in_len_sent: batch[3],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.acc_num, feed_dict)

    def mode_predict(self, sess, batch):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_words: batch[1],
            self.in_len_sent: batch[2],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.y_p, feed_dict)

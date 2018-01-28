# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .ops import lstm, bilstm, peephole_lstm


class Model(object):
    def __init__(self, config):
        # base config
        self.name = config.model_name
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

        self.use_tf_lstm_api = config.use_tf_lstm_api

        # regular
        self.l2_norm = config.l2_norm
        self.we_reg = False
        self.params_reg = True

        # Setup Model Parameters
        self.max_len_sent = config.max_len_sent
        self.bi = config.bi
        self.n_rnn = config.n_rnn
        # self.vocab_size = config.vocab_size
        self.dim_word = config.dim_word
        self.n_mlp = config.n_mlp
        self.n_class = config.n_class
        self.clipper = config.clipper
        self.g_step = tf.Variable(0, name='g_step', trainable=False)

        self.lr = tf.Variable(config.lr, name='lr', trainable=False)
        self.lr_decay_op = self.lr.assign(self.lr.value() * 0.1)

        self.keep_rate = tf.placeholder(tf.float32, name='keep_prob')

        # Word Embedding
        with tf.variable_scope('word_embedding'):
            if config.we is not None:
                self.we = tf.Variable(config.we, name='word_embedding', dtype=tf.float32)
            else:
                self.we = tf.get_variable(
                    'word_embedding',
                    [config.vocab_size, self.dim_word],
                    tf.float32,
                    initializer=tf.random_uniform_initializer(-0.25, 0.25))
                print('get random we (word embedding).')
        if self.we_reg:
            self.we_init = tf.constant(config.we, dtype=tf.float32)

        # Build the Computation Graph
        self._build_model()
        # Set loss
        self._set_cost_and_optimize()
        # Set prediction and acc
        self._set_predict()
        # add tensor board
        self._log_summaries()
        # ModelZoo parameter saver
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        # Model PlaceHolder for input
        self.in_x = tf.placeholder(tf.int32, [None, self.max_len_sent])  # [batch, sent]
        self.in_y = tf.placeholder(tf.int32, [None])
        self.in_len = tf.placeholder(tf.int32, [None])

        # Embedding layer
        # [batch, sent, dim_word]
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)

        self.f_gate = None
        self.f_gate_mean = None
        self.f_gate_var = None
        self.f_gate_h_mean = None

        with tf.variable_scope("Model"):
            if not self.use_tf_lstm_api:
                print("standard lstm by my own implement ... ")
                if self.bi:
                    print ("bi lstm")
                    hs_fw, h_fw, hs_bw, h_bw = bilstm(
                        embedded_seq, self.dim_word, self.n_rnn,
                        self.in_len, self.max_len_sent, tf.float32)
                    rnn_out = tf.concat([h_fw, h_bw], axis=-1)
                else:
                    print ("single direction peephole_lstm")
                    hs, h, fs = peephole_lstm(
                        embedded_seq, self.dim_word, self.n_rnn,
                        self.in_len, self.max_len_sent, tf.float32)
                    self.f_gate = fs  # batch x max_len x n_rnn_word
                    self.f_gate_mean = tf.reduce_mean(fs, axis=2)  # batch x max_len
                    # tf.expand_dims(self.in_lens_word, axis=1)
                    # self.f_gate_mean = tf.reduce_mean(self.f_gate_mean)
                    f_gate_sum_1 = tf.reduce_sum(fs, axis=1)  # batch x n_rnn_word
                    f_gate_sum_2 = tf.reduce_sum(fs, axis=2, keep_dims=True)  # batch x max_len x 1
                    var = tf.reduce_mean(tf.square(fs - f_gate_sum_2), axis=2) # batch x max_len
                    var = var / tf.expand_dims(self.in_len, axis=1)
                    self.f_gate_var = tf.reduce_mean(var, axis=0)
                    f_gate_h_mean = f_gate_sum_1 / tf.expand_dims(self.in_len, axis=1)  # batch x n_rnn_word
                    self.f_gate_h_mean = tf.reduce_mean(f_gate_h_mean, axis=0)  # n_rnn_word
                    rnn_out = h
            else:
                print("standard lstm use tensorflow lstm api ... ")
                rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.n_rnn, use_peepholes=True)
                if self.bi:
                    print("bi lstm ... ")
                    rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.n_rnn)
                    # # outputs: A tuple (output_fw, output_bw)
                    # # output_fw: [batch_size, max_time, cell_bw.output_size]
                    b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,
                                                                          rnn_cell_bw,
                                                                          embedded_seq, self.in_len,
                                                                          dtype=tf.float32)
                    # [batch_size, cell.state_size x 2]
                    rnn_out = tf.concat([b_states[0][0], b_states[1][1]], axis=-1)
                else:
                    outputs, state = tf.nn.dynamic_rnn(
                        rnn_cell_fw, embedded_seq, self.in_len, dtype=tf.float32)
                    rnn_out = state[1]

            rnn_out = tf.nn.dropout(rnn_out, keep_prob=self.keep_rate)
            mlp = tf.layers.dense(rnn_out, self.n_mlp, activation=tf.nn.relu)
            self.logits = tf.layers.dense(mlp, self.n_class)

    def _set_cost_and_optimize(self):

        # loss
        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.in_y))
        self.cost = softmax_cost

        if self.params_reg:
            model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Model")
            for p in model_params:
                self.cost += self.l2_norm * tf.nn.l2_loss(p)
        if self.we_reg:
                self.cost += self.l2_norm * tf.nn.l2_loss(tf.subtract(self.we, self.we_init))

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)  # .minimize(self.loss)
        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), self.clipper)
        self.train_op = optimizer.apply_gradients(list(zip(grads, train_vars)),
                                                  global_step=self.g_step)

    def _set_predict(self):
        y_prob = tf.nn.softmax(self.logits)
        self.y_p = tf.cast(tf.argmax(y_prob, 1), tf.int32)
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
        # f_gate_mean = tf.summary.scalar("f_gate_mean", self.f_gate_mean)
        # f_gate_var = tf.summary.histogram("f_gate_var", self.f_gate_var)
        # f_gate_h_mean = tf.summary.histogram("f_gate_h_mean", self.f_gate_h_mean)
        # gate = tf.summary.scalar("gate", self.gate)
        # self.merged = tf.summary.merge([loss, acc, f_gate_mean, f_gate_var, f_gate_h_mean])

    def train(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.keep_rate: drop_keep_rate
        }
        return_list = [self.train_op, self.merged, self.g_step, self.cost, self.acc_num]

        return sess.run(return_list, feed_dict)

    def model_test(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.keep_rate: 1.0
        }
        return sess.run(self.acc_num, feed_dict)

    def predict(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_len: batch[2],
            self.keep_rate: 1.0
        }
        return sess.run(self.y_p, feed_dict)

    def model_predict_f_gate(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.keep_rate: 1.0
        }
        return sess.run([self.y_p, self.f_gate_mean], feed_dict)

    def lr_decay(self, sess):
        old_lr = sess.run(self.lr)
        sess.run(self.lr_decay_op)
        new_lr = sess.run(self.lr)
        return old_lr, new_lr

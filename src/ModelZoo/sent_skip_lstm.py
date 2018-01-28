# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import skip_bilstm, skip_lstm


class Model(object):
    def __init__(self, config):
        # base config
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

        # regular
        self.l2_norm = config.l2_norm
        self.we_reg = False
        self.params_reg = True

        # ModelZoo
        self.bi = config.bi
        # Setup Model Parameters
        self.max_seq_len = config.max_seq_len
        self.rnn_size = config.rnn_size
        # self.vocab_size = config.vocab_size
        self.word_dim = config.word_dim
        self.mlp_size = config.mlp_size
        self.class_num = config.class_num
        self.max_gradient_norm = config.max_gradient_norm
        self.global_step = tf.Variable(0, name="g_step", trainable=False)

        self.learning_rate = tf.Variable(config.learning_rate, name='lr', trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate.value()*0.1)

        self.drop_keep_rate = tf.placeholder(tf.float32, name='keep_prob')

        # Word Embedding
        with tf.variable_scope("Embeddings"):
            self.we = tf.Variable(config.we, name='emb', dtype=tf.float32)
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
        self.in_len = tf.placeholder(tf.int32, [None])
        self.in_x = tf.placeholder(tf.int32, [None, self.max_seq_len])  # shape: (batch x seq)
        self.in_y = tf.placeholder(tf.int32, [None])
        # Embedding layer
        # shape: (batch x seq x dim_word)
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)

        with tf.variable_scope("Model"):
            if self.bi:
                print ("bi skip lstm ...")
                hs_fw, h_fw, fs_fw, hs_bw, h_bw, fs_bw = skip_bilstm(
                    embedded_seq, self.word_dim, self.rnn_size, self.in_len, self.max_seq_len, dtype=tf.float32)
                rnn_out = tf.concat([h_fw, h_bw], axis=1)
                f_gate = tf.concat([fs_bw, fs_fw], axis=1)
                f_gate = tf.reduce_mean(f_gate, axis=-1)
            else:
                print("skip lstm ...")
                hs, h, fs = skip_lstm(
                    embedded_seq, self.word_dim, self.rnn_size, self.in_len, self.max_seq_len, dtype=tf.float32)
                # fs [batch x max_len x n_rnn_word]
                rnn_out = h
                f_gate = tf.reduce_mean(fs, axis=-1)

            self.debug = f_gate
            rnn_out = tf.nn.dropout(rnn_out, keep_prob=self.drop_keep_rate)
            mlp = tf.layers.dense(rnn_out, self.mlp_size, activation=tf.nn.relu)
            self.logits = tf.layers.dense(mlp, self.class_num)

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
        optimizer = tf.train.AdamOptimizer(self.learning_rate)  # .minimize(self.loss)

        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(list(zip(grads, train_vars)),
                                                  global_step=self.global_step)

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
        # gate = tf.summary.scalar("gate", self.gate)
        self.merged = tf.summary.merge([cost, acc])

    def model_train(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: drop_keep_rate,
        }
        return_list = [self.train_op, self.merged, self.global_step, self.cost, self.acc_num]

        return sess.run(return_list, feed_dict)

    def model_test(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0,
        }
        return sess.run(self.acc_num, feed_dict)

    def mode_predict(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0,
        }
        return sess.run(self.y_p, feed_dict)

    def model_debug(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.train_op, self.merged, self.global_step, self.cost, self.debug]

        return sess.run(return_list, feed_dict)

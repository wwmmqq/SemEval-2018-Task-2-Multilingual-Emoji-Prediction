# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .ops_attention import structured_self_attention


class Model(object):
    def __init__(self, cfg):
        self.name = 'multi_att_we'
        # attention number
        self.structure_punish = False
        self.concat_we = True
        self.att_num = cfg.att_num

        # Setup Model Parameters
        self.max_len_sent = cfg.max_len_sent
        self.n_rnn = cfg.n_rnn
        # self.vocab_size = cfg.vocab_size
        self.dim_word = cfg.dim_word
        self.n_mlp = cfg.n_mlp
        self.n_class = cfg.n_class
        self.max_gradient_norm = cfg.max_gradient_norm
        self.global_step = tf.Variable(0, name="g_step", trainable=False)

        self.learning_rate = tf.Variable(
            cfg.learning_rate, name='lr', trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate.value()*0.1)

        self.drop_keep_rate = tf.placeholder(tf.float32, name='keep_prob')

        # opt [regular]
        self.l2_norm = cfg.l2_norm
        self.we_reg = False
        self.params_reg = True
        self.bug = None

        # Word Embedding
        with tf.variable_scope("word_embedding"):
            self.we = tf.Variable(cfg.we, name='emb', dtype=tf.float32)
        if self.we_reg:
            self.we_init = tf.constant(cfg.we, dtype=tf.float32)

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
        self.in_x = tf.placeholder(tf.int32, [None, self.max_len_sent])
        self.in_y = tf.placeholder(tf.int32, [None])

        # Embedding layer
        # shape: (batch x seq x dim_word)
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_x)

        with tf.variable_scope("Model"):
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.n_rnn)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.n_rnn)
            # outputs: A tuple (output_fw, output_bw)
            # output_fw: [batch_size, max_time, cell_bw.output_size]
            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell_fw, rnn_cell_bw, embedded_seq,
                self.in_len, dtype=tf.float32)
            # [batch_size, max_time, cell_bw.output_size x 2]

            if self.concat_we:
                print ('concat we in bi_rnn_out')
                bi_rnn_out = tf.concat([b_outputs[0], embedded_seq, b_outputs[1]], axis=-1)
                context_size = self.n_rnn * 2 + self.dim_word
            else:
                bi_rnn_out = tf.concat([b_outputs[0], b_outputs[1]], axis=-1)
                context_size = self.n_rnn * 2
            # structured self attention
            # [batch, max_time, att_num]
            self.att_prob_s = structured_self_attention(
                bi_rnn_out, self.in_len, context_size, att_num=self.att_num)
            # [batch, cell_bw.output_size x 2, max_time]
            h_t = tf.transpose(bi_rnn_out, perm=[0, 2, 1], name='h_T')

            # [batch, cell_bw.output_size x 2, r]
            sent_rep_2d = tf.matmul(h_t, self.att_prob_s)
            # flatten
            sent_rep = tf.reshape(sent_rep_2d, shape=[-1, context_size*self.att_num])

        with tf.name_scope('mlp'), tf.variable_scope('mlp'):
            fc1 = tf.layers.dense(sent_rep, context_size, activation=tf.nn.relu)
            fc1_drop = tf.nn.dropout(fc1, keep_prob=self.drop_keep_rate)
            mlp = tf.layers.dense(fc1_drop, self.n_mlp, activation=tf.nn.relu)

        with tf.name_scope('logits'), tf.variable_scope('logits'):
            self.logits = tf.layers.dense(mlp, self.n_class)

    def _set_cost_and_optimize(self):
        # loss
        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.in_y))
        self.cost = softmax_cost

        if self.params_reg:
            model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Model")
            for p in model_params:
                self.cost += self.l2_norm * tf.nn.l2_loss(p)
        if self.we_reg:
                self.cost += self.l2_norm * tf.nn.l2_loss(tf.subtract(self.we, self.we_init))

        if self.structure_punish:
            # structured self attentive loss
            # Compute penalization term
            with tf.variable_scope("penalization_term"):
                # [batch x r x max_time]
                A = tf.transpose(self.att_prob_s, perm=[0, 2, 1], name="A")
                # [batch x r x r]
                A_A_T = tf.matmul(A, self.att_prob_s, name="AA_T")
                diag = tf.diag(tf.ones([self.att_num]), name="diag_identity")
                # [batch x (r x r)] ==> [batch x r x r]
                identity = tf.tile(diag, [tf.shape(self.in_x)[0], 1])
                identity = tf.reshape(
                    identity,
                    [tf.shape(self.in_x)[0], self.att_num, self.att_num])
                penalized_term = tf.square(
                    tf.norm(A_A_T - identity, ord='euclidean', axis=[1, 2], name="frobenius_norm"))
                self.penalized_term = tf.reduce_mean(penalized_term)
            self.cost += self.penalized_term

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, train_vars), self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(
            list(zip(grads, train_vars)),
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
        self.merged = tf.summary.merge([cost, acc])

    def model_debug(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.train_op, self.cost, self.bug]

        return sess.run(return_list, feed_dict)

    def model_train(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: drop_keep_rate
        }
        return_list = [self.train_op, self.merged,
                       self.global_step, self.cost, self.acc_num]

        return sess.run(return_list, feed_dict)

    def model_test(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_y: batch[1],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.acc_num, feed_dict)

    def mode_predict(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0
        }
        return sess.run(self.y_p, feed_dict)

    def mode_predict_detail(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0
        }
        return_list = [self.y_p, self.att_prob_s]
        return sess.run(return_list, feed_dict)

    def mode_predict_with_att_prob(self, sess, batch):
        feed_dict = {
            self.in_x: batch[0],
            self.in_len: batch[2],
            self.drop_keep_rate: 1.0
        }
        return_list = [self.y_p, self.att_prob_s]
        return sess.run(return_list, feed_dict)

# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from template import BaseModel
import tf_encoder
import tf_ops


class Model(BaseModel):
    def __init__(self, cfg, we):
        super(Model, self).__init__(cfg)
        self.name = "self_att"
        self.dim_word = cfg.dim_word
        self.max_len_sent = cfg.max_len_sent
        self.n_rnn = cfg.n_rnn
        self.att_size = cfg.att_size
        self.n_mlp = cfg.n_mlp
        self.n_class = cfg.n_class
        self.clipper = cfg.clipper

        # opt [regular]
        self.l2_norm = cfg.l2_norm
        self.params_reg = False
        self.inference(we=we)

    def _build_model(self):
        self.in_x = tf.placeholder(tf.int32, [None, self.max_len_sent])
        self.in_y = tf.placeholder(tf.int32, [None])
        self.in_len = tf.placeholder(tf.int32, [None])

        seq_words = tf.nn.embedding_lookup(self.we, self.in_x)  # [B, T, D]
        with tf.name_scope("rnn"):
            rnn_out = tf_encoder.bi_rnn_std(seq_words, self.in_len, self.n_rnn, return_hs=True)
        with tf.name_scope("att"):
            sent = tf_ops.self_attention(rnn_out, self.in_len, att_size=self.att_size)
            sent_dim = 2*self.n_rnn
        with tf.name_scope("mlp"):
            mlp = tf_ops.dense_layer(sent, sent_dim, self.n_mlp, activation=tf.nn.relu)
        with tf.name_scope("logits"):
            self.logits = tf_ops.dense_layer(mlp, self.n_mlp, self.n_class, scope='logits')

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_y: batch[0],
            self.in_x: batch[1],
            self.in_len: batch[2],
            self.keep_prob: keep_rate
        }
        _, g_step, loss, y = sess.run(
            [self.train_op, self.g_step, self.loss, self.y], feed_dict)
        return g_step, loss, y

    def predict(self, sess, batch):
        feed_dict = {
            self.in_x: batch[1],
            self.in_len: batch[2],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

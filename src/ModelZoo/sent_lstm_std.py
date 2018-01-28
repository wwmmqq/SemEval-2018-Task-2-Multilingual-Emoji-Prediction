# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from template import BaseModel
import tf_encoder
import tf_ops


class Model(BaseModel):
    def __init__(self, cfg, we=None):
        super(Model, self).__init__(cfg)
        self.name = 'lstm_std'
        # Setup Model Parameters
        self.dim_word = cfg.dim_word
        self.max_len_sent = cfg.max_len_sent
        self.n_rnn = cfg.n_rnn
        self.n_mlp = cfg.n_mlp
        self.n_class = cfg.n_class
        self.clipper = cfg.clipper

        self.inference(we=we)

    def _build_model(self):
        # Model PlaceHolder for input
        # [batch, sent]
        self.in_words = tf.placeholder(tf.int32, [None, self.max_len_sent])
        self.in_lens_sent = tf.placeholder(tf.int32, [None])
        self.in_y = tf.placeholder(tf.int32, [None])

        # Embedding layer
        # [batch, sent, dim_word]
        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_words)

        # with tf.name_scope("lstm"):
        #     rnn_h_out = tf_encoder.bi_rnn_std(
        #         embedded_seq, self.in_lens_word, self.n_rnn, return_hs=True)
        #     rnn_out = tf_ops.max2d_pooling(rnn_h_out, self.max_len_sent)
        #
        # with tf.name_scope("mlp"):
        #     mlp = tf_ops.dense_layer(
        #         rnn_out, 2*self.n_rnn, self.n_mlp, activation=tf.nn.relu, scope="mlp")

        with tf.name_scope("lstm"):
            rnn_out = tf_encoder.bi_rnn_std(
                embedded_seq, self.in_lens_word, self.n_rnn)

        with tf.name_scope("mlp"):
            mlp = tf_ops.dense_layer(
                rnn_out, 2*self.n_rnn, self.n_mlp, activation=tf.nn.relu, scope="mlp")

        with tf.name_scope("logits"):
            self.logits = tf_ops.dense_layer(
                mlp, self.n_mlp, self.n_class, scope="logits")
            self.logits = tf.layers.dense(mlp, self.n_class)

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_y: batch[0],
            self.in_words: batch[1],
            self.in_lens_sent: batch[2],
            self.keep_prob: keep_rate
        }

        _, g_step, loss, y = sess.run(
            [self.train_op, self.g_step, self.loss, self.y], feed_dict)
        return g_step, loss, y

    def predict(self, sess, batch):
        feed_dict = {
            self.in_words: batch[1],
            self.in_lens_sent: batch[2],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

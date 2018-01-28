# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from template import BaseModel
import tf_ops


class Model(BaseModel):
    def __init__(self, cfg, we):
        super(Model, self).__init__(cfg)
        self.name = 'nbow'
        self.dim_word = cfg.dim_word
        self.max_len_sent = cfg.max_len_sent
        self.n_mlp = cfg.n_mlp
        self.n_class = cfg.n_class
        self.clipper = cfg.clipper
        self.inference(we=we)

    def _build_model(self):
        # Model PlaceHolder for input
        self.in_words = tf.placeholder(tf.int32, [None, self.max_len_sent])
        self.in_y = tf.placeholder(tf.int32, [None])
        self.in_lens_word = tf.placeholder(tf.int32, [None])

        embedded_seq = tf.nn.embedding_lookup(self.we, self.in_words)

        with tf.name_scope('nbow'):
            avg_out = tf_ops.avg_pool(embedded_seq, self.in_lens_word)

        with tf.name_scope("mlp"):
            mlp = tf_ops.dense_layer(
                avg_out, self.dim_word, self.n_mlp, activation=tf.nn.relu, scope="mlp")

        with tf.name_scope('logits'):
            self.logits = tf_ops.dense_layer(mlp, self.n_mlp, self.n_class, scope="logits")

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_y: batch[0],
            self.in_words: batch[1],
            self.in_lens_word: batch[2],
            self.keep_prob: keep_rate
        }
        _, step, cost, y = sess.run(
            [self.train_op, self.g_step, self.loss, self.y],
            feed_dict)
        return step, cost, y

    def predict(self, sess, batch):
        feed_dict = {
            self.in_words: batch[1],
            self.in_lens_word: batch[2],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

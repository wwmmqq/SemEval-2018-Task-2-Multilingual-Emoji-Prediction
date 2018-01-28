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
        self.name = "cnn"
        self.dim_word = cfg.dim_word
        self.max_len_sent = cfg.max_len_sent
        self.n_mlp = cfg.n_mlp
        self.n_class = cfg.n_class
        self.clipper = cfg.clipper
        self.inference(we=we)

    def _build_model(self):
        self.in_words = tf.placeholder(tf.int32, [None, self.max_len_sent])
        self.in_y = tf.placeholder(tf.int32, [None])

        seq_words = tf.nn.embedding_lookup(self.we, self.in_words)

        cnn_out = tf_encoder.cnn2d(
            seq_words, [3, 4, 5], self.dim_word, k_nums=[100, 100, 100])
        cnn_out_dim = 300
        cnn_out = tf.nn.dropout(cnn_out, self.keep_prob)

        with tf.name_scope("logits"):
            self.logits = tf_ops.dense_layer(
                cnn_out, cnn_out_dim, self.n_class, scope="logits")

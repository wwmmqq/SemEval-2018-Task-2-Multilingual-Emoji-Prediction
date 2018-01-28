# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from template import BaseModel
import tf_ops
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"


class Model(BaseModel):
    """
    Character-level Convolutional Networks for Text Classification
    """
    def __init__(self, cfg, ce):
        super(Model, self).__init__(cfg)
        self.name = "char_cnn"
        self.dim_char = cfg.dim_char
        self.max_len = 120
        self.n_class = cfg.n_class
        self.clipper = cfg.clipper
        self.filter_sizes = (7, 3, 3,)  # (7, 7, 3, 3, 3, 3)
        self.inference(ce=ce)

    def _build_model(self):
        self.in_chars = tf.placeholder(tf.int32, [None, self.max_len])
        self.in_y = tf.placeholder(tf.int32, [None])

        seq_chars = tf.nn.embedding_lookup(self.ce, self.in_chars)
        x = tf.expand_dims(seq_chars, -1)   # [B, T, D, 1]

        with tf.name_scope("conv1-max-pooling"):
            # [filter_height, filter_width, in_channels, out_channels]
            filter_shape = [7, self.dim_char, 1, 256]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            x = tf.nn.max_pool(h, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding="VALID", name="pool1")

        with tf.name_scope("conv2-max-pooling"):
            filter_shape = [3, 1, 256, 256]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            x = tf.nn.max_pool(h,
                               ksize=[1, 3, 1, 1],
                               strides=[1, 3, 1, 1],
                               padding="VALID", name="pool2")

        num_features_total = 12 * 256
        x = tf.reshape(x, [-1, num_features_total])
        with tf.name_scope("dropout-1"):
            x = tf.nn.dropout(x, self.keep_prob)

        x = tf_ops.dense_layer(x, num_features_total, 1024, activation=tf.nn.relu, scope="fc1")
        with tf.name_scope("dropout-1"):
            x = tf.nn.dropout(x, self.keep_prob)
        x = tf_ops.dense_layer(x, 1024, 256, activation=tf.nn.relu, scope="fc2")
        self.logits = tf_ops.dense_layer(x, 256, self.n_class, scope="logits")

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_y: batch[0],
            self.in_chars: batch[1],
            self.keep_prob: keep_rate
        }
        _, step, cost, y = sess.run(
            [self.train_op, self.g_step, self.loss, self.y],
            feed_dict)
        return step, cost, y

    def predict(self, sess, batch):
        feed_dict = {
            self.in_chars: batch[1],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

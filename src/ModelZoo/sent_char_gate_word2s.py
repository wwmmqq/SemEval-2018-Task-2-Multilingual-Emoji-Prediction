# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from template import BaseModel
import tf_ops
import tf_encoder


class Model(BaseModel):
    def __init__(self, cfg, ce, we):
        super(Model, self).__init__(cfg)
        self.name = 'char_gate_word'
        self.dim_word = cfg.dim_word
        self.max_len_word = cfg.max_len_word
        self.max_len_sent = cfg.max_len_sent
        self.n_rnn_char = cfg.n_rnn_char
        self.n_rnn_word = cfg.n_rnn_word
        self.n_mlp = cfg.n_mlp
        self.n_class = cfg.n_class
        self.clipper = cfg.clipper

        self.inference(ce=ce, we=we)

    def _build_model(self):
        self.in_words = tf.placeholder(tf.int32, [None, self.max_len_sent])
        self.in_chars = tf.placeholder(tf.int32, [None, self.max_len_sent, self.max_len_word])
        self.in_y = tf.placeholder(tf.int32, [None])
        self.in_lens_sent = tf.placeholder(tf.int32, [None])
        self.in_lens_word = tf.placeholder(
            tf.int32, [None, self.max_len_sent], name="in_lens_word")

        seqs_words_embed = tf.nn.embedding_lookup(self.we, self.in_words)
        # seqs_words_embed = tf.nn.dropout(seqs_words_embed, self.keep_prob)

        c_gate_w = tf_encoder.char2word_gate_word(
            self.in_chars, self.in_lens_word, self.ce, self.n_rnn_char, seqs_words_embed)

        with tf.name_scope("rnn"):
            rnn_out = tf_encoder.bi_rnn_std(c_gate_w, self.in_lens_sent, self.n_rnn_word, return_hs=True)
        with tf.name_scope("att"):
            sent = tf_ops.self_attention(rnn_out, self.in_lens_sent, att_size=self.n_rnn_word)
            sent_dim = 2*self.n_rnn_word

        # sent = tf_encoder.word2sent(c_gate_w, self.in_lens_sent, self.n_rnn_word)
        # dim_sent = 2*self.n_rnn_word

        with tf.name_scope("mlp"):
            # sent = tf.nn.dropout(sent, self.keep_prob)
            mlp = tf_ops.dense_layer(
                sent, sent_dim, self.n_mlp, activation=tf.nn.relu, scope="mlp")

        with tf.name_scope("logits"):
            self.logits = tf_ops.dense_layer(mlp, self.n_mlp, self.n_class, scope="logits")

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_y: batch[0],
            self.in_words: batch[1],
            self.in_lens_sent: batch[2],
            self.in_chars: batch[3],
            self.in_lens_word: batch[4],
            self.keep_prob: keep_rate
        }
        _, g_step, loss, y = sess.run(
            [self.train_op, self.g_step, self.loss, self.y], feed_dict)
        return g_step, loss, y

    def predict(self, sess, batch):
        feed_dict = {
            self.in_words: batch[1],
            self.in_lens_sent: batch[2],
            self.in_chars: batch[3],
            self.in_lens_word: batch[4],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

# coding: utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from template import BaseModel
from encoder import char2word_rnn, word2sent, char2word_cnn


class Model(BaseModel):

    def __init__(self, cfg, ce=None, we=None):
        BaseModel.__init__(self, cfg)
        # base cfg
        self.name = 'c2w2s'
        self.max_len_sent = cfg.max_len_sent
        self.max_len_word = cfg.max_len_word
        self.dim_char = cfg.dim_char
        self.dim_word = cfg.dim_word

        self.inference(ce=ce, we=we)

    def _build_model(self):
        # Model PlaceHolder for input
        # shape: [batch, max_seq, max_len_word]
        self.in_chars = tf.placeholder(
            tf.int32,
            [None, self.max_len_sent, self.max_len_word])
        self.in_words = tf.placeholder(
            tf.int32, [None, self.max_len_sent], name='in_words')

        self.in_len_sent = tf.placeholder(tf.int32, [None])
        self.in_len_word = tf.placeholder(
            tf.int32, [None, self.max_len_sent], name='in_len_sent_w')
        self.in_y = tf.placeholder(tf.int32, [None])

        seqs_embed_c2w = char2word_rnn(
            self.in_chars, self.in_len_word, self.ce, 200, self.dim_word)
        # seqs_embed_c2w = char2word_cnn(
        #     self.in_chars, self.ce, self.dim_char,
        #     k_h_n=((2, 25), (3, 25), (4, 25), (5, 25)))
        #
        # seqs_embed_w = tf.nn.embedding_lookup(self.we, self.in_words)  # [32 x 30 x 200]
        #
        # seqs_embed = tf.concat([seqs_embed_w, seqs_embed_c2w], axis=-1)  # [32 x 30 x (200+100)]

        sent = word2sent(seqs_embed_c2w, self.in_len_sent, self.n_rnn)

        self.debug = sent

        with tf.name_scope('mlp'), tf.variable_scope('mlp'):
            w = tf.get_variable(
                'w_mlp',
                [self.n_rnn, self.n_mlp], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.02))
            b = tf.get_variable(
                'b', [self.n_mlp], dtype=tf.float32,
                initializer=tf.zeros_initializer)
            mlp = tf.nn.xw_plus_b(sent, w, b)
            mlp = tf.nn.relu(mlp)

        with tf.name_scope('logits'), tf.variable_scope('logits'):
            self.logits = tf.layers.dense(mlp, self.n_class)

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_words: batch[1],
            self.in_y: batch[2],
            self.in_len_sent: batch[4],
            self.keep_prob: keep_rate
        }
        return_list = [
            self.train_op, self.merged,
            self.g_step, self.loss, self.acc_num]

        return sess.run(return_list, feed_dict)

    def predict(self, sess, batch):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_words: batch[1],
            self.in_len_word: batch[2],
            self.in_len_sent: batch[4],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

    def model_debug(self, sess, batch, drop_keep_rate=0.5):
        feed_dict = {
            self.in_chars: batch[0],
            self.in_words: batch[1],
            self.in_y: batch[2],
            self.keep_prob: drop_keep_rate
        }

        return sess.run(self.debug, feed_dict)

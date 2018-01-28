# coding: utf-8
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from data_utils import (load_word_embedding_std,
                        load_char_embedding_std,
                        sentence2_char_word_id)
data = namedtuple('data', 'x_c,x_w,y,len_word,len_sent')


class DataSet(object):
    def __init__(self, cfg, init_data=True):
        self.train_file = cfg.train_file
        self.test_file = cfg.test_file
        self.dev_file = cfg.dev_file
        self.we_file = cfg.word_embed_file
        self.ce_file = cfg.char_embed_file
        self.dim_word = cfg.dim_word
        self.dim_char = cfg.dim_char
        self.max_len_sent = cfg.max_len_sent
        self.max_len_word = cfg.max_len_word

        self.train = None
        self.dev = None
        self.test = None
        self._get_word_dict()

        if init_data:
            self._get_train_trial_test()

    def _get_word_dict(self):
        self.w2id, self.we = load_word_embedding_std(
            self.dim_word,
            self.we_file)
        self.c2id, self.ce = load_char_embedding_std(
            self.dim_char,
            self.ce_file)

    def _get_train_trial_test(self):
        # x_word, x_char, y, len_sent
        t = sentence2_char_word_id(
            self.train_file, self.c2id, self.w2id,
            self.max_len_word, self.max_len_sent)
        self.train = data(x_c=t[0], x_w=t[1], y=t[2], len_word=t[3], len_sent=t[4])

        if self.dev_file is not None:
            t = sentence2_char_word_id(
                self.dev_file, self.c2id, self.w2id,
                self.max_len_word, self.max_len_sent)
            self.dev = data(x_c=t[0], x_w=t[1], y=t[2], len_word=t[3], len_sent=t[4])

        if self.test_file is not None:
            t = sentence2_char_word_id(
                self.train_file, self.c2id, self.w2id,
                self.max_len_word, self.max_len_sent)
            self.test = data(x_c=t[0], x_w=t[1], y=t[2], len_word=t[3], len_sent=t[4])

    def info(self):
        msg = 'data info: \n'
        msg += 'max_len_sent: {}\n'.format(self.max_len_sent)
        msg += 'we shape: {}\n'.format(self.we.shape)
        msg += 'train data: x_c, x_w, y, len_word, len_sent\n'
        if self.train is not None:
            msg += ' '.join(['{}'.format(d.shape) for d in self.train])
        if self.dev is not None:
            msg += '\n dev data: x_c, x_w, y, len_word, len_sent\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.dev])
        if self.test is not None:
            msg += '\n test data: x_c, x_w, y, len_word, len_sent\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.test])
        print (msg)

    def get_batch_sample(self, batch_size=32):
        import numpy as np
        batch = data(
            x_c=np.array(
                [[[0]*self.max_len_word for _ in range(self.max_len_sent)] for _ in range(batch_size)],
                dtype=np.int32),
            x_w=np.array([[0]*self.max_len_sent for _ in range(batch_size)],
                         dtype=np.int32),
            y=np.array([0]*batch_size, dtype=np.int32),
            len_word=np.array([[6]*self.max_len_word for _ in range(self.max_len_sent)], dtype=np.int32),
            len_sent=np.array([self.max_len_sent]*batch_size, dtype=np.int32)
        )
        print ("""batch info:
                x_c: {} x_w: {}
                y: {}  len_word: {}   len_sent: {}
        """.format(batch.x_c.shape, batch.x_w.shape,
                   batch.y.shape, batch.len_word.shape, batch.len_sent.shape))
        return batch

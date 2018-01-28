# coding: utf-8
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from data_utils import (load_char_embedding_std,
                        sentence2_char_id)
data = namedtuple('data', 'x_c,y,word_len,sent_len')


class DataSet(object):
    def __init__(self, config):
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.dev_file = config.dev_file
        self.we_file = config.word_embed_file
        self.ce_file = config.char_embed_file
        self.dim_word = config.dim_word
        self.dim_char = config.dim_char
        self.max_len_sent = config.max_len_sent
        self.max_len_word = config.max_len_word

        self.train = None
        self.dev = None
        self.test = None
        self._init_data()

    def _init_data(self):
        self.c2id, self.ce = load_char_embedding_std(
            15,
            self.ce_file)

        # x_char, y, len_sent, len_word
        t = sentence2_char_id(
            self.train_file, self.c2id,
            self.max_len_sent, self.max_len_word)
        self.train = data(x_c=t[0], y=t[1], word_len=t[2], sent_len=t[3])

        if self.dev_file is not None:
            t = sentence2_char_id(
                self.dev_file, self.c2id,
                self.max_len_sent, self.max_len_word)
            self.dev = data(x_c=t[0], y=t[1], word_len=t[2], sent_len=t[3])

        if self.test_file is not None:
            t = sentence2_char_id(
                self.train_file, self.c2id,
                self.max_len_sent, self.max_len_word)
            self.test = data(x_c=t[0], y=t[1], word_len=t[2], sent_len=t[3])

    def info(self):
        msg = 'data info: \n'
        msg += 'max_len_sent: {}\n'.format(self.max_len_sent)
        msg += 'ce shape: {}\n'.format(self.ce.shape)
        msg += 'train data: x_c, x_w, y, sent_len\n'
        msg += ' '.join(['{}'.format(d.shape) for d in self.train])
        if self.dev is not None:
            msg += '\n dev data: x_c, x_w, y, sent_len\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.dev])
        if self.test is not None:
            msg += '\n test data: x_c, x_w, y, sent_len\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.test])
        print (msg)

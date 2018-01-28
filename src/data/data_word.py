# coding: utf-8
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import data_utils
data = namedtuple('data', 'x,y,sent_len')


class DataSet(object):

    def __init__(self, cfg):
        self.train_file = cfg.train_file
        self.test_file = cfg.test_file
        self.dev_file = cfg.dev_file
        self.we_file = cfg.word_embed_file
        self.dim_word = cfg.dim_word
        self.max_len_sent = cfg.max_len_sent

        self.train = None
        self.dev = None
        self.test = None
        self._init_data()

    def _init_data(self):
        self.w2id = data_utils.build_word_vocab(
            [self.train_file, self.dev_file, self.test_file],
            max_num=200000, min_cnt=2, log_words=True)
        self.we = data_utils.load_embedding_std(self.we_file, self.dim_word, self.w2id)

        # train_x, train_y, train_seq_len
        t = data_utils.sentence2id_and_pad(
            self.train_file, self.w2id, self.max_len_sent)
        self.train = data(x=t[0], y=t[1], sent_len=t[2])

        if self.dev_file is not None:
            t = data_utils.sentence2id_and_pad(
                self.dev_file, self.w2id, self.max_len_sent)
            self.dev = data(x=t[0], y=t[1], sent_len=t[2])

        if self.test_file is not None:
            t = data_utils.sentence2id_and_pad(
                self.test_file, self.w2id, self.max_len_sent)
            self.test = data(x=t[0], y=t[1], sent_len=t[2])

    def info(self):
        msg = 'data info: \n'
        msg += 'max_len_sent: {}\n'.format(self.max_len_sent)
        msg += 'we shape: {}\n'.format(self.we.shape)
        if self.dev is not None:
            msg += 'train data: x, y, sent_len\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.train])
        if self.dev is not None:
            msg += '\n dev data: x, y, sent_len\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.dev])
        if self.test is not None:
            msg += '\n test data: x, y, sent_len\n'
            msg += ' '.join(['{}'.format(d.shape) for d in self.test])
        print (msg)

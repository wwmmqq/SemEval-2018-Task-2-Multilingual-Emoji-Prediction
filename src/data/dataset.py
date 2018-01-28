# coding: utf-8
from __future__ import division
from __future__ import print_function

import logging

import data_utils
logger = logging.getLogger(__name__)


class DataSet:
    _HOME_PATH = "/home/wmq/Desktop/DeepText/SemEval2018Task2"
    train_file = _HOME_PATH + "/data/emoji/us_train_process.txt"
    test_file = _HOME_PATH + "/data/emoji/us_test_process.txt"
    dev_file = _HOME_PATH + "/data/emoji/us_trial_process.txt"
    data_pkl = _HOME_PATH + "/data/emoji/data.pkl"
    emoji_embed_file = _HOME_PATH + "/data/embed/emoji_we.txt"
    word_embed_file = _HOME_PATH + "/data/embed/tweet_we.txt"
    word_embed_file2 = _HOME_PATH + "/data/embed/swm_300_we.txt"
    word_embed_file_big = _HOME_PATH + "/data/embed/model_swm_300-6-10-low.w2v"
    vocab_file = _HOME_PATH + "/data/embed/vocab.txt"
    char_embed_file = None  # _HOME_PATH + "/data/embed/tweet_ce.txt"
    dim_word = 300  # 200
    dim_char = 25   # 25
    max_len = 120   # max char number of sentence
    max_len_sent = 30
    max_len_word = 10

    def __init__(self):
        self.w2id = None
        self.we = None
        self.c2id = None
        self.ce = None
        self.emoji_embed = None
        self.train = []
        self.dev = []
        self.test = []

    def load_we(self):
        self.w2id, self.we = data_utils.load_word_embedding_std(
            DataSet.dim_word,
            DataSet.word_embed_file)

    def load_vocab_and_we(self):
        self.w2id, self.we = data_utils.sample_embedding_from_big_embed(
            DataSet.vocab_file, n_dim=self.dim_word,
            big_emb_file=DataSet.word_embed_file2)

    def load_ce(self):
        self.c2id, self.ce = data_utils.load_char_embedding_std(
            DataSet.dim_char,
            DataSet.char_embed_file)

    def load_emoji(self, transpose):
        import emoji
        self.emoji_embed = emoji.load_emoji(self.emoji_embed_file, transpose)

    def load_data_label(self):
        self.train.append(
            data_utils.read_label(DataSet.train_file))
        self.dev.append(
            data_utils.read_label(DataSet.dev_file))
        self.test.append(
            data_utils.read_label(DataSet.test_file))

    def load_data_words(self):
        s_words, s_len = data_utils.sent_word2id_and_pad(
            DataSet.train_file, self.w2id, DataSet.max_len_sent)
        self.train.append(s_words)
        self.train.append(s_len)

        s_words, s_len = data_utils.sent_word2id_and_pad(
            DataSet.dev_file, self.w2id, DataSet.max_len_sent)
        self.dev.append(s_words)
        self.dev.append(s_len)

        s_words, s_len = data_utils.sent_word2id_and_pad(
            DataSet.test_file, self.w2id, DataSet.max_len_sent)
        self.test.append(s_words)
        self.test.append(s_len)

    def load_data_char2words(self):
        s_chars, s_len = data_utils.sent_char2id2word_and_pad(
            DataSet.train_file, self.c2id, DataSet.max_len_sent, DataSet.max_len_word)
        self.train.append(s_chars)
        self.train.append(s_len)

        s_chars, s_len = data_utils.sent_char2id2word_and_pad(
            DataSet.dev_file, self.c2id, DataSet.max_len_sent, DataSet.max_len_word)
        self.dev.append(s_chars)
        self.dev.append(s_len)

        s_chars, s_len = data_utils.sent_char2id2word_and_pad(
            DataSet.test_file, self.c2id, DataSet.max_len_sent, DataSet.max_len_word)
        self.test.append(s_chars)
        self.test.append(s_len)

    def load_data_chars(self):
        s_chars, s_len = data_utils.sent_char2id_and_pad(
            DataSet.train_file, self.c2id, DataSet.max_len)
        self.train.append(s_chars)
        self.train.append(s_len)

        s_chars, s_len = data_utils.sent_char2id_and_pad(
            DataSet.dev_file, self.c2id, DataSet.max_len)
        self.dev.append(s_chars)
        self.dev.append(s_len)

        s_chars, s_len = data_utils.sent_char2id_and_pad(
            DataSet.test_file, self.c2id, DataSet.max_len)
        self.test.append(s_chars)
        self.test.append(s_len)

    def clear(self):
        self.train = []
        self.dev = []
        self.test = []

    def save(self, save_file=None):
        if save_file is None:
            save_file = DataSet.data_pkl
        data_utils.save_params(
            [self.we, self.ce, self.train, self.dev, self.test], save_file)
        logger.info("saved [self.we, self.ce, self.train, self.dev, self.test] to %s" % save_file)

    def load(self, load_file=None):
        if load_file is None:
            load_file = DataSet.data_pkl
        self.we, self.ce, self.train, self.dev, self.test = data_utils.load_params(
            load_file)

    def show(self):
        logger.info("dim_word: %d" % DataSet.dim_word)
        logger.info("dim_char: %d" % DataSet.dim_char)
        logger.info("max_len_sent: %d" % DataSet.max_len_sent)
        logger.info("max_len_word: %d" % DataSet.max_len_word)
        if self.we is not None:
            logger.info("we shape: (%d %d)" % self.we.shape)
        if self.ce is not None:
            logger.info("ce shape: (%d %d)" % self.ce.shape)
        if len(self.train) != 0:
            logger.info("==> train :")
            for var in self.train:
                logger.info(var.shape)

        if len(self.dev) != 0:
            logger.info("==> dev :")
            for var in self.dev:
                logger.info(var.shape)

        if len(self.test) != 0:
            logger.info("==> test :")
            for var in self.test:
                logger.info(var.shape)


if __name__ == '__main__':
    data = DataSet()
    # word2id = data_utils.build_word_vocab([
    #     DataSet.train_file,
    #     DataSet.dev_file,
    #     DataSet.test_file],
    #     min_cnt=0,
    #     max_num=200000)
    #
    # print("vocab size: %d" % len(word2id))
    # fw = open(DataSet.word_embed_file+'.new', 'w')
    # cnt = 0
    # with open(DataSet.word_embed_file, 'r') as fr:
    #     fr.readline()
    #     fw.write("55831 200\n")
    #     for line in fr:
    #         if line.split()[0] in word2id:
    #             fw.write(line)
    #         else:
    #             cnt += 1
    # print("cnt: %d" % cnt)
    # fw.close()
    # data.load_we()
    # data.load_data_words()

    # data_utils.get_embedding_token(DataSet.word_embed_file_big)
    # data_utils.sample_embedding_from_big_embed(
    #     DataSet.vocab_file, n_dim=300,
    #     big_emb_file=DataSet.word_embed_file2)
    data.load_emoji(transpose=False)
    print(data.emoji_embed.shape)

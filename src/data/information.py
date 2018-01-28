from __future__ import print_function
from __future__ import division

import string
from collections import Counter
from data_utils import build_word_vocab
punctuation = set(string.punctuation)


class DataInformation:

    def __init__(self, f_train, f_dev=None, f_test=None):

        self.f_dir = '/'.join(f_train.split('/')[:-1])
        self.fps = [f_train, f_dev, f_test]

        self.most_words = 100000
        self.n_train = 0
        self.n_dev = 0
        self.n_test = 0
        self.n_words_train = 0
        self.n_words_dev = 0
        self.n_words_test = 0
        self.n_chars_train = 0
        self.n_chars_dev = 0
        self.n_chars_test = 0
        self.n_miss_words_dev = 0
        self.n_miss_chars_dev = 0
        self.n_miss_words_test = 0
        self.n_miss_chars_test = 0

        self.main()

    def main(self):
        words_train = Counter()
        chars_train = Counter()
        miss_chars = set()
        miss_words = set()

        max_len_s = 0.
        max_len_w = 0.
        with open(self.fps[0], 'r', encoding='utf-8') as fr:
            sum_len_s = 0
            sum_len_w = 0
            cnt_w = 0
            try:
                for line in fr:
                    self.n_train += 1
                    sent = line.strip().split()[1:]
                    words_train.update(sent)
                    chars_train.update(''.join(sent))
                    if len(sent) > max_len_s:
                        max_len_s = len(sent)
                    sum_len_s += len(sent)
                    for w in sent:
                        if len(w) > max_len_w:
                            max_len_w = len(w)
                        if w not in punctuation:
                            sum_len_w += len(w)
                            cnt_w += 1
            except UnicodeError:
                print('unicode error in num: %d' % self.n_train)

            self.n_words_train = len(words_train)
            self.n_chars_train = len(chars_train)
        words_total = set(words_train.keys())
        print('train instance: %d' % self.n_train)
        print('train chars num: %d' % self.n_chars_train)
        print('train words num: %d' % self.n_words_train)
        print('train max_len_s: %d' % max_len_s)
        print('train max_len_w: %d' % max_len_w)
        print('train avg_len_s: %d' % (sum_len_s / self.n_train))
        print('train avg_len_w: %d' % (sum_len_w / cnt_w))

        if self.fps[1] is not None:
            chars_dev = Counter()
            words_dev = Counter()
            with open(self.fps[1], 'r', encoding='utf-8') as fr:
                for line in fr:
                    self.n_dev += 1
                    t = line.strip().split()
                    ws = t[1:]
                    words_dev.update(ws)
                    words_total.update(ws)
                    for w in ws:
                        if w not in words_train:
                            self.n_miss_words_dev += 1
                            miss_words.add(w)
                    chars = ''.join(t[1:])
                    chars_dev.update(chars)
                    for c in chars:
                        if c not in chars_train:
                            self.n_miss_chars_dev += 1
                            miss_chars.add(c)
                self.n_chars_dev = len(chars_dev)
            print('dev instance: %d' % self.n_dev)
            print('dev chars num: %d' % self.n_chars_dev)
            print('dev oov chars num in train: %d' % self.n_miss_chars_dev)
            print('dev oov words num in train: %d' % self.n_miss_words_dev)

        if self.fps[2] is not None:
            chars_test = Counter()
            words_test = Counter()
            with open(self.fps[2], 'r', encoding='utf-8') as fr:
                for line in fr:
                    self.n_test += 1
                    t = line.strip().split()
                    ws = t[1:]
                    words_test.update(ws)
                    words_total.update(ws)
                    for w in ws:
                        if w not in words_train:
                            self.n_miss_words_test += 1
                            miss_words.add(w)

                    chars = ''.join(t[1:])
                    chars_test.update(chars)
                    for c in chars:
                        if c not in chars_train:
                            self.n_miss_chars_test += 1
                            miss_chars.add(c)

                self.n_chars_test = len(chars_test)
            print('test instance: %d' % self.n_test)
            print('test chars num: %d' % self.n_chars_test)
            print('test oov chars num in train: %d' % self.n_miss_chars_test)
            print('test oov words num in train: %d' % self.n_miss_words_test)


def sentence_length_distribution(files, seq_len=50):
    seq_len += 1
    max_len = 0
    len_dic = {i: 0 for i in range(1, seq_len)}

    total = 0
    for f in files:
        with open(f, 'r') as fr:
            for line in fr:
                total += 1
                words = line.strip().split()[1:]
                n = len(words)
                if n < seq_len:
                    len_dic[n] += 1
                if n > max_len:
                    max_len = n
                    # print(' '.join(words))
    print("total sample: %d" % total)
    print(max_len)
    for k, v in len_dic.items():
        print('%s : %s' % (k, v))

if __name__ == '__main__':
    DIR = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji'
    files = [DIR + '/us_train_process.txt',
             DIR + '/us_trial_process.txt',
             DIR + '/us_test_process.txt']
    # DataInformation(f_train=DIR + '/us_train_process.txt',
    #                 f_dev=DIR + '/us_trial_process.txt',
    #                 f_test=DIR + '/us_test_process.txt')

    # sentence_length_distribution(files)
    w2id = build_word_vocab(files, max_num=200000, min_cnt=0, log_words=True)
    print (len(w2id))
# train instance: 471455
# train chars num: 67
# train words num: 166272
# train max_len_s: 43
# train max_len_w: 64
# train avg_len_s: 13
# train avg_len_w: 4
# dev instance: 50000
# dev chars num: 65
# dev oov chars num in train: 0
# dev oov words num in train: 10264
# test instance: 50000
# test chars num: 66
# test oov chars num in train: 0
# test oov words num in train: 14293


# sentence_length_distribution
# total sample: 571455
# 43
# 1 : 54
# 2 : 516
# 3 : 3398
# 4 : 9020
# 5 : 14018
# 6 : 18414
# 7 : 23049
# 8 : 26822
# 9 : 30261
# 10 : 34305
# 11 : 38234
# 12 : 41794
# 13 : 43565
# 14 : 43338
# 15 : 42242
# 16 : 40704
# 17 : 39133
# 18 : 35499
# 19 : 29897
# 20 : 22293
# 21 : 14744
# 22 : 8757
# 23 : 4851
# 24 : 2522
# 25 : 1426
# 26 : 865
# 27 : 538
# 28 : 371
# 29 : 295
# 30 : 178
# 31 : 133
# 32 : 91
# 33 : 51
# 34 : 40
# 35 : 13
# 36 : 9
# 37 : 7
# 38 : 3
# 39 : 2
# 40 : 0
# 41 : 2
# 42 : 0
# 43 : 1

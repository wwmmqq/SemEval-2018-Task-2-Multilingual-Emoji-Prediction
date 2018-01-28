# coding: utf-8

import numpy as np
import os
import pickle
import string
from collections import Counter
g_chars = string.punctuation+string.ascii_lowercase+string.digits
g_label2id = {'__label__%d' % i: i for i in range(20)}

token_pad = '<pad>'
token_unk = '<unk>'


def save_txt(params, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'w') as fw:
        for p in params:
            fw.write('%s\n' % str(p))


def save_params(params, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as fw:
        pickle.dump(params, fw, protocol=2)


def load_params(fname):
    if not os.path.exists(fname):
        raise RuntimeError('no file: %s' % fname)
    with open(fname, 'rb') as fr:
        params = pickle.load(fr)
    return params


def build_word_vocab(files, max_num=100000, min_cnt=0, log_words=False):
    """
    :param files: list of file, eg. ['train.txt', 'test.txt']
    :param max_num: max size of vocab
    :param min_cnt: min cnt of word
    :param log_words: True / False, words_counter
    :return: word2id
    """
    counter = Counter()
    for f in files:
        with open(f, 'r') as fr:
            for line in fr:
                ws = line.strip().split()[1:]
                counter.update(ws)
    words = counter.most_common(max_num)
    word2id = {w: i+2 for i, (w, cnt) in enumerate(words) if cnt > min_cnt}
    word2id[token_pad] = 0
    word2id[token_unk] = 1

    if log_words:
        f_dir = os.path.dirname(files[0])
        save_txt(words, f_dir+'/words_count.txt')
    return word2id


def get_embedding_token(emb_file, head_line_is_info=True, out_file='vocab.txt'):
    """
    :param emb_file: string
    :param out_file: string
    :param head_line_is_info: True / False
    """
    fr = open(emb_file, 'r')
    if head_line_is_info:
        print("token info: %s" % fr.readline())
    f_dir = os.path.dirname(emb_file)
    f_out = f_dir+'/'+out_file
    with open(f_out, 'w') as fw:
        for line in fr:
            token = line.split(None, 1)[0]
            fw.write("%s\n" % token)
    fr.close()
    print("log tokens in file: %s" % f_out)


def build_word_from_vocab_file(vocab_file):
    """
    Args:
        vocab_file: string
    Returns: word2id, e.t. {'you': 45, ...}
    """
    with open(vocab_file, 'r') as fr:
        word2id = {w.strip(): i + 2 for i, w in enumerate(fr)}
    word2id[token_pad] = 0
    word2id[token_unk] = 1
    return word2id


def sample_embedding_from_big_embed(vocab_file, n_dim, big_emb_file, head_line_is_info=True, save=True):
    word2id = build_word_from_vocab_file(vocab_file)
    we = load_embedding_std(big_emb_file, n_dim, word2id, head_line_is_info=head_line_is_info)
    if save:
        f_out = os.path.dirname(big_emb_file) + "/small_we.txt"
        with open(f_out, 'w') as fw:
            fw.write("%d %d\n" % we.shape)
            for w, idx in word2id.items():
                t = " ".join(["%0.6f" % x for x in we[idx]])
                fw.write("%s %s\n" % (w, t))
    return word2id, we


def load_embedding_std(emb_file, n_dim, token2id, head_line_is_info=False):
    """
    :param emb_file: string
    :param n_dim: int
    :param token2id: dict
    :param head_line_is_info
    :return: we, 2d array
    """
    fr = open(emb_file, 'r')
    n_token = len(token2id)
    we = np.random.uniform(-0.25, 0.25, (n_token, n_dim)).astype(dtype=np.float32)
    we[0] = np.zeros(n_dim)

    if head_line_is_info:
        n_vocab, dim = fr.readline().strip().split()
        assert int(dim) == n_dim
        print("token info: %s %s" % (n_vocab, dim))
    for line in fr:
        sp = line.rstrip().split()
        idx = token2id.get(sp[0], -1)
        if idx != -1:
            we[idx] = [float(t) for t in sp[1:]]
    fr.close()
    return we


def load_word_embedding_std(n_dim=200, emb_file=None):
    fr = open(emb_file, 'r')
    n, dim = [int(t) for t in fr.readline().strip().split()]
    assert dim == n_dim
    w2id = {token_pad: 0, token_unk: 1}
    we = np.zeros(shape=[n+2, n_dim], dtype=np.float32)
    we[1] = np.random.uniform(-0.25, 0.25, (n_dim, ))
    idx = 2
    for line in fr:
        sp = line.rstrip().split()
        w2id[sp[0]] = idx
        we[idx] = [float(t) for t in sp[1:]]
        idx += 1
    fr.close()
    # print ('end idx: %s' % idx)
    print ('we shape: (%d, %d)' % we.shape)
    return w2id, we


def sentence2id_and_pad(in_file, w2id, max_sent_len):
    x = []
    y = []
    seq_len = []
    oov_cnt = 0
    # oov = set()
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.strip().split()
            y.append(g_label2id[t[0]])
            words = t[1].split()
            words = words[: max_sent_len]
            seq_len.append(len(words))
            t = []
            for i, w in enumerate(words):
                idx = w2id.get(w, 1)
                if idx == 1:
                    # oov.add(w)
                    oov_cnt += 1
                else:
                    t.append(idx)
            t += [0] * (max_sent_len - len(t))
            x.append(t)
    print ("sentence2id oov tokens cnt: %s" % oov_cnt)
    return (np.array(x, dtype=np.int32),
            np.array(y, dtype=np.int32),
            np.array(seq_len, dtype=np.float32))


def load_char_embedding_std(n_dim=30, emb_file=None):
    """ char index and char embedding
    Args:
        n_dim: char dim
        emb_file:
    Returns:
        c2id  dict
        ce  numpy
    """
    c2id = {token_pad: 0, token_unk: 1}

    if emb_file is not None:
        fr = open(emb_file, 'r')
        n, dim = [int(t) for t in fr.readline().strip().split()]
        assert dim == n_dim
        ce = np.zeros(shape=[n+1, n_dim], dtype=np.float32)
        idx = 1
        for line in fr:
            sp = line.rstrip().split()
            c2id[sp[0]] = idx
            ce[idx] = [float(t) for t in sp[1:]]
            idx += 1
        fr.close()
    else:
        c2id = {c: i+2 for i, c in enumerate(g_chars)}
        ce = np.random.uniform(
            -0.1, 0.1,
            (len(g_chars)+2, n_dim)).astype(np.float32)
        ce[0] = np.zeros(n_dim)
        print ('get random init ce.')
    print ('ce shape: (%d, %d)' % ce.shape)
    return c2id, ce


def sentence2id_word_char(sent, w2id, c2id, max_len_word, max_len_sent):
    x_word = [0]*max_len_sent
    x_char = []
    sent = sent[: max_len_sent]
    for i, w in enumerate(sent):
        x_word[i] = w2id.get(w, 1)
        c_w = [0] * max_len_word
        for j, c in enumerate(w):
            c_w[j] = c2id.get(c, 0)
        x_char.append(c_w)
    return x_word, x_char


def sentence2id(in_file, w2id, c2id, max_len_sent, max_len_word):
    g_divide = '\t\t'
    x_word = []
    x_char = []
    y = []
    len_sent = []
    with open(in_file, 'r') as fr:
        for nu, line in enumerate(fr):
            t = line.strip().split(g_divide)
            assert len(t) == 2
            sent = t[1].split()[: max_len_sent]
            v_words = [0] * max_len_sent

            v_chars = np.zeros((max_len_sent, max_len_word), dtype=np.int32)
            for i, w in enumerate(sent):
                v_words[i] = w2id.get(w, 1)
                w = w[: max_len_word]
                for j, c in enumerate(w):
                    v_chars[i][j] = c2id.get(c, 0)
            x_char.append(v_chars)
            x_word.append(v_words)
            y.append(int(t[0]))
            len_sent.append(len(sent))

    return (np.array(x_char, dtype=np.int32),
            np.array(x_word, dtype=np.int32),
            np.array(y, dtype=np.int32),
            np.array(len_sent, dtype=np.int32))


def sentence2_char_id(in_file, c2id, max_len_sent, max_len_word):
    g_divide = '\t\t'
    x_char = []
    len_word = []
    y = []
    len_sent = []
    with open(in_file, 'r') as fr:
        for nu, line in enumerate(fr):
            t = line.strip().split(g_divide)
            assert len(t) == 2
            sent = t[1].split()[: max_len_sent]

            v_chars = np.zeros((max_len_sent, max_len_word), dtype=np.int32)
            len_words = np.zeros(max_len_sent, dtype=np.int32)
            for i, w in enumerate(sent):
                w = w[: max_len_word]
                for j, c in enumerate(w):
                    v_chars[i][j] = c2id.get(c, 0)
                len_words[i] = len(w)
            x_char.append(v_chars)
            y.append(int(t[0]))
            len_word.append(len_words)
            len_sent.append(len(sent))

    return (np.array(x_char, dtype=np.int32),
            np.array(y, dtype=np.int32),
            np.array(len_word, dtype=np.int32),
            np.array(len_sent, dtype=np.int32))


def sentence2_char_word_id(in_file, c2id, w2id, max_len_word, max_len_sent):
    g_divide = '\t\t'
    x_char = []
    x_word = []
    y = []
    len_word = []
    len_sent = []
    with open(in_file, 'r') as fr:
        for nu, line in enumerate(fr):
            t = line.strip().split(g_divide)
            assert len(t) == 2
            sent = t[1].split()[: max_len_sent]

            v_chars = np.zeros((max_len_sent, max_len_word), dtype=np.int32)
            len_words = np.zeros(max_len_sent, dtype=np.int32)
            v_words = [0] * max_len_sent
            for i, w in enumerate(sent):
                w = w[: max_len_word]
                v_words[i] = w2id.get(w, 1)
                for j, c in enumerate(w):
                    v_chars[i][j] = c2id.get(c, 0)
                len_words[i] = len(w)
            x_char.append(v_chars)
            x_word.append(v_words)
            y.append(int(t[0]))
            len_word.append(len_words)
            len_sent.append(len(sent))

    return (np.array(x_char, dtype=np.int32),
            np.array(x_word, dtype=np.int32),
            np.array(y, dtype=np.int32),
            np.array(len_word, dtype=np.int32),
            np.array(len_sent, dtype=np.int32))


def sent_word2id_and_pad(in_file, words2id, max_len_sent):
    """
    :param in_file: string
    :param words2id: dict, eg. {'world': 6}
    :param max_len_sent: int , eg. 30
    :return: np.arrays, ([N x max_len_s],
                        [N])
    """

    vectors_w = []
    lens_sent = []
    n_oov = 0
    n_sent = 0.
    with open(in_file, 'r') as fr:
        for line in fr:
            n_sent += 1
            _, sent = line.strip().split(None, 1)
            sent = sent.split()[: max_len_sent]
            v_w = np.zeros(max_len_sent, dtype=np.int32)
            for i, word in enumerate(sent):
                v_w[i] = words2id.get(word, 0)
                if v_w[i] == 0:
                    n_oov += 1
            vectors_w.append(v_w)
            lens_sent.append(len(sent))
    print("oov_n: %d, oov/sent : %0.2f " % (n_oov, n_oov/n_sent))
    return (np.array(vectors_w, dtype=np.int32),
            np.array(lens_sent, dtype=np.int32))


def sent_char2id2word_and_pad(in_file, chars2id, max_len_sent, max_len_word):
    """
    :param in_file: string
    :param chars2id: dict, eg. {'a': 2}
    :param max_len_sent: int , e.g. 30
    :param max_len_word: int , eg. 10
    :return: np.arrays, ([N x max_len_s x max_len_c],
                        [max_len_s x max_len_c])
    """

    vectors_c = []
    lens_word = []
    with open(in_file, 'r') as fr:
        for line in fr:
            _, sent = line.strip().split(None, 1)
            sent = sent.split()[: max_len_sent]
            sent_w_c = np.zeros((max_len_sent, max_len_word), dtype=np.int32)
            len_ws = np.zeros(max_len_sent, dtype=np.int32)  # len of word in a sentence
            for i, word in enumerate(sent):
                word = word[: max_len_word]
                len_ws[i] = len(word)
                for j, c in enumerate(word):
                    sent_w_c[i][j] = chars2id.get(c, 1)
            vectors_c.append(sent_w_c)
            lens_word.append(len_ws)

    return (np.array(vectors_c, dtype=np.int32),
            np.array(lens_word, dtype=np.int32))


def sent_char2id_and_pad(in_file, chars2id, max_len):
    """
    :param in_file: string
    :param chars2id: dict, eg. {'a': 2}
    :param max_len: int , e.g. 1024
    :return: np.arrays, ([N x x max_len],
                        [N])
    """

    vectors_c = []
    lens = []
    with open(in_file, 'r') as fr:
        for line in fr:
            _, sent = line.strip().split(None, 1)
            sent_char = ''.join(sent.split())[:max_len]
            sent_c = np.zeros(max_len, dtype=np.int32)
            for i, char in enumerate(sent_char):
                sent_c[i] = chars2id.get(char, 1)
            vectors_c.append(sent_c)
            lens.append(len(sent_c))

    return (np.array(vectors_c, dtype=np.int32),
            np.array(lens, dtype=np.int32))


def read_label(in_file):
    """
    Args:
        in_file: string
    Returns: np.arrays, [N]

    """
    y = []
    with open(in_file, 'r') as fr:
        for line in fr:
            t = line.split()[0]
            y.append(g_label2id[t])
    return np.array(y, dtype=np.int32)


def test1():
    data_dir = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/embed'
    w2id, we = load_word_embedding_std(
        200,
        data_dir+'/tweet_we.bin')
    print (we.shape)
    c2id, ce = load_char_embedding_std(
        30,
        None)
    print ('we shape : %d %d' % we.shape)
    print ('ce shape : %d %d' % ce.shape)

    in_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/sample_train.txt'
    rst = sentence2id(in_file, w2id, c2id, 30, 10)
    print (rst[0].shape)
    print (rst[1].shape)
    print (rst[2].shape)
    print (rst[2].shape)

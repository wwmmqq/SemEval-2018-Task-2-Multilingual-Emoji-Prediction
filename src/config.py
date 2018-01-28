# coding : utf-8
import os
import logging
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # for issue: The TensorFlow library wasn't compiled to use SSE3

random_seed = 1234
tf.set_random_seed(random_seed)
np.random.seed(random_seed)
logger = logging.getLogger(__name__)


class BaseConfig(object):
    # [General]
    _HOME_PATH = "/home/wmq/Desktop/DeepText/SemEval2018Task2"

    model_name = "base"
    data_name = "emoji"
    name = "lstm"

    train_file = _HOME_PATH + "/data/{}/us_train_process.txt".format(data_name)
    test_file = _HOME_PATH + "/data/{}/us_test_process.txt".format(data_name)
    dev_file = _HOME_PATH + "/data/{}/us_trial_process.txt".format(data_name)
    word_embed_file = _HOME_PATH + "/data/embed/tweet_we.txt"
    char_embed_file = _HOME_PATH + "/data/embed/tweet_ce.txt"
    model_dir = _HOME_PATH + "/ModelZoo"
    log_dir = _HOME_PATH + "/log"
    result_dir = _HOME_PATH + "/result"
    result_file = result_dir + "/result_{}_{}.txt".format(data_name, name)
    test_error = result_dir + "/result_{}_{}_error.txt".format(data_name, name)

    def info(self):
        for k, v in vars(self).items():
            logger.info("%s : %s" % (k, v))


class NBOWConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 64
        self.dim_word = 200
        self.max_len_sent = 30

        self.n_mlp = 100
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.01
        self.l2_norm = 1e-4
        self.save_model = True


class NBOWConfig2(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 64
        self.dim_word = 300
        self.max_len_sent = 30

        self.n_mlp = 300
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.01
        self.l2_norm = 1e-4
        self.save_model = True


class CharGateWordConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 20
        self.batch_size = 32
        self.dim_char = 25
        self.dim_word = 200

        self.max_len_word = 10
        self.max_len_sent = 30

        self.n_rnn_char = 80
        self.n_rnn_word = 300
        self.n_mlp = 100
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.001
        self.l2_norm = 1e-4
        self.save_model = True


class CharGateWordConfig2(BaseConfig):
    def __init__(self):
        self.epoch_size = 5
        self.batch_size = 32
        self.dim_char = 25
        self.dim_word = 300

        self.max_len_word = 10
        self.max_len_sent = 30

        self.n_rnn_char = 80
        self.n_rnn_word = 300
        self.n_mlp = 300
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.001
        self.l2_norm = 1e-4
        self.save_model = True


class CenterConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 5
        self.batch_size = 32
        self.dim_char = 25
        self.dim_word = 300

        self.max_len_word = 10
        self.max_len_sent = 30

        self.n_rnn_char = 80
        self.n_rnn_word = 300
        self.n_mlp = 300
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.001
        self.l2_norm = 1e-4
        self.save_model = True


class LSTMConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 64
        self.dim_word = 200
        self.max_len_sent = 30
        self.n_rnn = 300
        self.n_mlp = 100

        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.01
        self.l2_norm = 1e-4
        self.save_model = True


class LSTMAttConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 20
        self.batch_size = 64
        self.dim_word = 200
        self.max_len_sent = 30
        self.n_rnn = 200
        self.att_size = 200
        self.n_mlp = 100

        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.01
        self.l2_norm = 1e-4
        self.save_model = True


class RCNNConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 64
        self.dim_word = 200
        self.max_len_sent = 30
        self.n_rnn = 200
        self.n_mlp = 100
        self.n_class = 20
        self.clipper = 5.0
        self.lr = 0.01
        self.l2_norm = 1e-4
        self.save_model = False


class CNNConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 50
        self.batch_size = 128
        self.dim_word = 200
        self.max_len_sent = 30
        self.n_mlp = 100
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.01
        self.l2_norm = 1e-4
        self.save_model = True


class CharCNNConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 50
        self.batch_size = 128
        self.dim_char = 70
        self.max_len = 120
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.01
        self.l2_norm = 1e-4


class SkipConfig(BaseConfig):
    def __init__(self):
        self.use_tf_lstm_api = False
        self.bi = True
        self.epoch_size = 5
        self.batch_size = 64
        self.word_dim = 200
        self.max_seq_len = 30
        self.rnn_size = 150
        self.mlp_size = 50

        self.class_num = 20
        self.max_gradient_norm = 10
        self.learning_rate = 0.001
        self.l2_norm = 1e-4
        self.save_model = True


class SelfAttConfig(BaseConfig):
    def __init__(self):
        self.load = False
        self.bi = False
        self.epoch_size = 10
        self.batch_size = 64
        self.word_dim = 200
        self.max_seq_len = 30
        self.rnn_size = 150
        self.mlp_size = 100

        self.class_num = 20
        self.max_gradient_norm = 10
        self.learning_rate = 0.001
        self.l2_norm = 1e-4
        self.save_model = True


class CharWordConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 32
        self.dim_char = 25
        self.dim_word = 200

        self.max_len_word = 10
        self.max_len_sent = 30

        self.n_rnn_char = 80
        self.n_rnn = 300
        self.n_mlp = 100
        self.n_class = 20
        self.clipper = 10.0
        self.lr = 0.001
        self.l2_norm = 1e-4
        self.save_model = True


class Char2w2sConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 32
        self.dim_char = 25
        self.dim_word = 200

        self.max_len_sent = 30
        self.max_len_word = 10

        self.n_rnn = 300
        self.n_mlp = 100
        self.n_class = 20
        self.clipper = 5.0
        self.lr = 1e-3
        self.l2_norm = 1e-4
        self.save_model = True


class StructuredConfig(BaseConfig):
    def __init__(self):
        self.epoch_size = 10
        self.batch_size = 64
        self.dim_word = 200
        self.max_len_sent = 30
        self.att_num = 5
        self.n_rnn = 200
        self.n_mlp = 100

        self.n_class = 20
        self.max_gradient_norm = 10
        self.learning_rate = 0.001
        self.l2_norm = 1e-4
        self.save_model = True

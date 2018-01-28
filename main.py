# coding: utf-8
import sys
sys.path.append("/home/wmq/Desktop/DeepText/SemEval2018Task2/src")

import time
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from src.data.dataset import DataSet
from src import config
from src.ModelZoo import train_tool
from src.ModelZoo import sent_nbow
from src.ModelZoo import sent_nbow_emoji
from src.ModelZoo import sent_char_gate_word2s
from src.ModelZoo import sent_center
from src.ModelZoo import sent_lstm_std
from src.ModelZoo import sent_rcnn
from src.ModelZoo import sent_self_attention
from src.ModelZoo import sent_cnn
from src.ModelZoo import sent_charcnn
from src.log import setup_logging

data = DataSet()


def main(model_type):
    # model_type = "char_gate_word_2"
    logger_file = "%s.log" % model_type
    logger = setup_logging(logger_file, stdout=True)
    model_save_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/Model5/{}_saver.ckpt'.format(model_type)
    load_data = False

    if "enhance" in model_type:
        DataSet.train_file = "/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train_enhanced.txt"

    if model_type == "char_cnn":
        char_data_file = "/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/data_char.pkl"
        if load_data:
            data.load(char_data_file)
        else:
            data.load_ce()
            data.load_data_label()
            data.load_data_chars()
            data.save(char_data_file)
    else:
        if load_data:
            data.load()
        else:
            data.load_vocab_and_we()
            # data.load_we()
            data.load_data_label()
            data.load_data_words()

            if "char" in model_type:
                data.load_ce()
                data.load_data_char2words()
            # data.save()
    data.show()

    if model_type == "nbow":
        cfg = config.NBOWConfig()
        cfg.batch_size = 128
        cfg.info()
        model = sent_nbow.Model(cfg, data.we)
    elif model_type == "nbow2":
        cfg = config.NBOWConfig2()
        cfg.batch_size = 128
        cfg.info()
        data.load_emoji(transpose=False)
        model = sent_nbow_emoji.Model(cfg, data.we, data.emoji_embed)
    elif model_type == "lstm":
        cfg = config.LSTMConfig()
        cfg.info()
        model = sent_lstm_std.Model(cfg, data.we)
    elif model_type == "lstm_att":
        cfg = config.LSTMAttConfig()
        cfg.info()
        model = sent_self_attention.Model(cfg, data.we)
    elif model_type == "char_gate_word":
        cfg = config.CharGateWordConfig()
        cfg.info()
        model = sent_char_gate_word2s.Model(cfg, data.ce, data.we)
    elif model_type == "char_gate_word_att_emoji":
        cfg = config.CharGateWordConfig2()
        cfg.info()
        model = sent_char_gate_word2s.Model(cfg, data.ce, data.we)
        model.name = "char_gate_word_att_emoji"
    elif model_type == "char_word_center":
        cfg = config.CenterConfig()
        cfg.info()
        data.load_emoji(transpose=False)
        model = sent_center.Model(cfg, data.ce, data.we, data.emoji_embed)
        model.name = "char_word_center"
    elif model_type == "cnn":
        cfg = config.CNNConfig()
        cfg.info()
        model = sent_cnn.Model(cfg, data.we)
    elif model_type == "rcnn":
        cfg = config.RCNNConfig()
        cfg.info()
        model = sent_rcnn.Model(cfg, data.we)
    elif model_type == "char_cnn":
        cfg = config.CharCNNConfig()
        cfg.info()
        model = sent_charcnn.Model(cfg, data.ce)
    else:
        raise KeyError("no such model type: %s" % model_type)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    start_t = time.time()
    # print("Epoch: %d" % 1)
    # train_tool.train_epoch(model, sess, data.train, cfg.batch_size)
    # print("### cost time: %0.2f m" % ((time.time()-start_t)/60))

    for epoch in range(0, cfg.epoch_size+1):
        logger.info("Epoch: %d" % epoch)
        if epoch < 5:
            if epoch == 1 or epoch == 2 or epoch == 3:
                old_lr, new_lr = model.lr_decay(sess)
                logger.info("change lr: %0.6f  => %0.6f" % (old_lr, new_lr))

        train_tool.train_epoch_with_test(
            model, sess, data.train, cfg.batch_size,
            dev_data=data.dev,
            test_data=data.test,
            macro_f1=0.27)
        if epoch == 1:
            # model.saver.save(sess, model_save_file, global_step=epoch)
            model.save(sess, model_save_file, global_step=epoch)
        logger.info("### cost time: %0.2f m" % ((time.time()-start_t)/60))


if __name__ == '__main__':
    args = sys.argv
    print("args: %s" % " ".join(args))
    print("model_type: %s " % args[1])
    main(args[1])

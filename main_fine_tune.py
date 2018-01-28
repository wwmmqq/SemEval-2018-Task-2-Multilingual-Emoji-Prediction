# coding: utf-8
import sys
sys.path.append("/home/wmq/Desktop/DeepText/SemEval2018Task2/src")

import time
import tensorflow as tf
from src.data.dataset import DataSet
from src import config
from src.ModelZoo import train_tool
from src.ModelZoo import sent_char_gate_word2s
from src.log import setup_logging

data = DataSet()


def main():
    model_type = "char_gate_word_att_emoji"
    logger_file = "%s_fine_tune.log" % model_type
    logger = setup_logging(logger_file, stdout=True)
    model_dir = "/home/wmq/Desktop/DeepText/SemEval2018Task2/Model5"
    model_save_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/Model5/{}_saver.ckpt'.format(model_type)

    data.load_vocab_and_we()
    data.load_data_label()
    data.load_data_words()
    data.load_ce()
    data.load_data_char2words()
    data.show()

    cfg = config.CharGateWordConfig2()
    cfg.lr = 0.01
    cfg.info()
    model = sent_char_gate_word2s.Model(cfg, data.ce, data.we)
    model.name = "char_gate_word_att_emoji"

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    model.load(sess, model_dir)
    model.set_optimize("sgd")

    start_t = time.time()
    for epoch in range(2, cfg.epoch_size+1):
        logger.info("Epoch: %d" % epoch)
        if epoch < 5:
            if epoch == 3 or epoch == 4:
                old_lr, new_lr = model.lr_decay(sess)
                logger.info("change lr: %0.6f  => %0.6f" % (old_lr, new_lr))

        train_tool.train_epoch_with_test(
            model, sess, data.train, cfg.batch_size,
            dev_data=data.dev,
            test_data=data.test,
            macro_f1=0.30)
        if epoch == 4:
            # model.saver.save(sess, model_save_file, global_step=epoch)
            model.save(sess, model_save_file, global_step=epoch)
        logger.info("### cost time: %0.2f m" % ((time.time()-start_t)/60))


if __name__ == '__main__':
    main()

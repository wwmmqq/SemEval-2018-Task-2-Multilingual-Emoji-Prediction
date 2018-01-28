from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def write_label(labels, in_file):
    with open(in_file, "w") as fw:
        for y in labels:
            fw.write("__label__%d\n" % y)


def batch_it(data, batch_size=32, shuffle=False):
    assert 1 == len(set([len(d) for d in data])), "Error: data len not same !"
    sample_num = len(data[0])
    idx = np.arange(sample_num)
    if shuffle:
        idx = np.random.permutation(sample_num)

    for start_idx in range(0, sample_num, batch_size):
        excerpt = idx[start_idx:start_idx + batch_size]
        rst = [x[excerpt] for x in data]
        yield rst


def batch_iter(x, y, seq_len=None, batch_size=None, shuffle=False):
    assert len(x) == len(y)
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(len(x))
    if seq_len is not None:
        for start_idx in range(0, len(x), batch_size):
            excerpt = idx[start_idx:start_idx + batch_size]
            yield x[excerpt], y[excerpt], seq_len[excerpt]
    else:
        for start_idx in range(0, len(x), batch_size):
            excerpt = idx[start_idx:start_idx + batch_size]
            yield x[excerpt], y[excerpt]


def train_epoch(model, sess, inputs, batch_size=32, show_step=100):
    cost_sum = 0.0
    cnt = 0
    y_true = []
    y_pred = []
    for batch in batch_it(inputs, batch_size, shuffle=True):
        step, cost, y = model.train(sess, batch)
        cnt += 1
        cost_sum += cost
        y_true += list(batch[0])
        y_pred += list(y)
        if step % show_step == 0:
            print("step_%d loss: %0.5f, acc: %0.5f" %
                  (step, cost_sum/cnt, accuracy_score(y_true, y_pred)))
            cost_sum = 0.0
            cnt = 0
            y_true = []
            y_pred = []


def train_epoch_with_test(
        model, sess, inputs,
        batch_size=32, show_step=100,
        dev_data=None, dev_step=1000,
        test_data=None,
        macro_f1=0):

    best_macro_f1 = macro_f1
    cost_sum = 0.0
    cnt = 0
    y_true = []
    y_pred = []
    for batch in batch_it(inputs, batch_size, shuffle=True):
        step, cost, y = model.train(sess, batch)
        cnt += 1
        cost_sum += cost
        y_true += list(batch[0])
        y_pred += list(y)
        if step % show_step == 0:
            msg = "step_%d loss: %0.5f, acc: %0.5f" % (
                step, cost_sum / cnt, accuracy_score(y_true, y_pred))
            logging.info(msg)
            cost_sum = 0.0
            cnt = 0
            y_true = []
            y_pred = []
        if step % dev_step == 0:
            y_pred_dev = test(model, sess, dev_data)
            macro_f1 = f1_score(dev_data[0], y_pred_dev, average='macro')
            logging.info(
                "macro f1: %0.5f (best: %0.5f)" % (macro_f1, best_macro_f1))
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                if test_data is not None:
                    y_pred_test = test(model, sess, test_data)
                    write_label(y_pred_test, "./test_result_%s_%d.txt" % (model.name, step))
                    write_label(y_pred_dev, "./dev_result_%s_%d.txt" % (model.name, step))
    logging.info(msg="End current epoch !")


def test(model, sess, inputs, load_init_model=False):
    if load_init_model:
        ckpt = tf.train.get_checkpoint_state(
            model.model_dir + '/{}'.format(model.model_name))
        print('ModelZoo file: ', ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('not exist ModelZoo ...')
    y_pred = []
    for batch in batch_it(inputs, 100, shuffle=False):
        y = model.predict(sess, batch)
        y_pred += list(y)
    return y_pred


def get_result(model, sess, data, load_model=False, result_file='result.txt'):
    if load_model:
        ckpt = tf.train.get_checkpoint_state(model.model_dir + '/{}'.format(model.model_name))
        print('ModelZoo file: ', ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('not exist ModelZoo ...')
    y_pred = []
    for batch in batch_it(data, 500):
        y_p = model.predict(sess, batch)
        y_pred += y_p.tolist()
    with open(result_file, 'w') as fw:
        for y in y_pred:
            fw.write('%s\n' % y)
    macro_fs = f1_score(data.y, y_pred, average='macro')
    print ('accuracy_score: %0.4f' % accuracy_score(data.y, y_pred))
    print ('macro_fs: %0.4f' % macro_fs)


def debug(model, sess, batch):
    return model.model_debug(sess, batch)


def train_dev(model, sess, data_train, data_dev, bs=32, es=10,
              result_dir='.', summary_writer=None):
    for epoch in range(es):
        print("Epoch {} / {} ...".format(epoch, es))
        for batch in batch_it(data_train, bs, shuffle=True):
            # print (batch[0].shape)
            # print (batch[1].shape)
            # print (batch[2].shape)
            # print (batch[3].shape)
            # input()
            _, summary, g_step, cost, acc_num = model.train(sess, batch)
            if g_step % 200 == 0:
                print ('batch loss: %0.4f,  acc: %0.4f' %
                       (cost, acc_num/len(batch[0])))

            if summary_writer is not None:
                summary_writer.add_summary(summary, g_step)

            if g_step % 1000 == 0:
                print('get result at step %s' % g_step)
                result_file = '%s/result_%s_%s.txt' % (result_dir, model.name, g_step)
                get_result(model, sess, data_dev, result_file=result_file)

            if g_step == 20000:
                old_lr, new_lr = model.lr_decay(sess)
                print ('lr %0.8f decay to ==> %0.8f' % (old_lr, new_lr))


def train_dev_test(
        model, sess, train_xy, dev_xy=None, test_xy=None,
        batch_size=32, epoch_size=10, lrd=3, save_model=False, summary_writer=None):

    save_file = model.model_dir + '/{}/{}_saver.ckpt'.format(
        model.model_name, model.model_name)
    best_test_acc = 0.0
    test_acc = 0.0

    for epoch in range(epoch_size):
        print("Epoch {} / {} ...".format(epoch, epoch_size))
        cost, acc = train_epoch(model, sess, train_xy, batch_size, summary_writer)
        print("Epoch {}: train loss: {}, acc: {}".format(epoch, cost, acc))
        if dev_xy is not None:
            dev_acc = test(model, sess, dev_xy)
            print("Epoch {}: dev acc: {}".format(epoch, dev_acc))
        if test_xy is not None:
            test_acc = test(model, sess, test_xy)
            print("Epoch {}: test acc: {}".format(epoch, test_acc))

        if save_model:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                model.saver.save(sess, save_file, global_step=epoch + 1)
        print("best test acc: {}".format(best_test_acc))
        if None is not lrd and epoch < lrd:
            sess.run(model.learning_rate_decay_op)


def predict(model, sess, test_data,
            load_init_model=True,
            result_file="result.txt"):
    if load_init_model:
        ckpt = tf.train.get_checkpoint_state(model.model_dir + '/{}'.format(model.model_name))
        print('ModelZoo file: ', ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('not exist ModelZoo ...')
    y_pred = []
    y_true = []
    for batch in batch_iter(test_data[0], test_data[1], test_data[2], batch_size=100):
        y_p = model.predict(sess, batch)
        y_pred += y_p.tolist()
        y_true += batch[1].tolist()

    with open(result_file, 'w') as fw:
        print (result_file)
        # fw.write("True    Predict\n")
        for y1, y2 in zip(y_true, y_pred):
            fw.write("{}    {}\n".format(y1, y2))


def predict_with_att_prob(model, sess, test_data,
                          load_init_model=True,
                          result_file="result.txt"):
    if load_init_model:
        ckpt = tf.train.get_checkpoint_state(model.model_dir + '/{}'.format(model.model_name))
        print('ModelZoo file: ', ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('not exist ModelZoo ...')

    y_true = test_data[1].tolist()
    y_pred, att_prob = model.mode_predict_with_att_prob(sess, test_data)
    y_pred = y_pred.tolist()
    with open(result_file, 'w') as fw:
        print (result_file)
        # fw.write("True    Predict\n")
        for y1, y2 in zip(y_true, y_pred):
            fw.write("{}    {}\n".format(y1, y2))

    # [batch, max_time, att_num]
    att_prob = att_prob.swapaxes(1, 2)
    shape = att_prob.shape
    print ("att_prob shape: {}".format(shape))
    att_prob = att_prob.tolist()
    with open(result_file + ".att_prob", "w") as fw:
        for idx, prob in enumerate(att_prob):
            for p in prob:
                rst = [str(t) for t in p[:int(test_data[2][idx])]]
                fw.write(" ".join(rst) + "\n")


def predict_f_gate(model, sess, test_data,
                   load_init_model=True):
    if load_init_model:
        ckpt = tf.train.get_checkpoint_state(model.model_dir + '/{}'.format(model.model_name))
        print('ModelZoo file: ', ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('not exist ModelZoo ...')
    y_p, fg = model.model_predict_f_gate(
        sess, [test_data[0], test_data[1], test_data[2]])
    return y_p, fg

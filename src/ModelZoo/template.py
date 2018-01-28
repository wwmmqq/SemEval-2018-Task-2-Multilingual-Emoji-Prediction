# coding: utf-8
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, cfg):
        # ------ setup ModelZoo parameters ------
        self.name = "template"
        self.vocab_size = None
        self.dim_word = None
        self.char_size = None
        self.dim_char = None

        self.ce = None
        self.we = None

        self.max_len_sent = None
        self.max_len_word = None
        self.n_mlp = None
        self.n_class = None
        self.clipper = None

        self.g_step = tf.Variable(0, name="g_step", trainable=False)
        self.lr = tf.Variable(cfg.lr, name="lr", dtype=tf.float32, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.lr_decay_op = self.lr.assign(self.lr.value() * 0.1)

        # ------ network ------
        self.in_chars = None
        self.in_words = None
        self.in_y = None
        self.in_lens_sent = None
        self.in_lens_word = None
        self.logits = None
        self.loss = None
        self.merged = None
        self.params_reg = False
        self.l2_norm = 0.0001
        self.optimizer = "sgd"
        self.train_op = None
        self.saver = None

    def inference(self, summary=False, token_trainable=True, **kwargs):
        self._token_embed(token_trainable, **kwargs)
        print("==> token embedding.")
        self._build_model()
        print("==> build model.")
        self._set_predict()
        self._set_loss()
        self.set_optimize()
        if summary:
            self._log_summaries()
            print("==> summary network.")
        self.saver = tf.train.Saver(tf.global_variables())
        print("==> finished build %s." % self.name)

    def _token_embed(self, trainable, **kwargs):

        if "we" in kwargs:
            we = kwargs.get("we", None)
            if we is not None:
                self.we = tf.Variable(we, name="word_emb", trainable=trainable)
            else:
                self.we = tf.get_variable(
                    "word_embedding",
                    [self.vocab_size, self.dim_word],
                    tf.float32,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                print("get random we (word embedding).")

        if "ce" in kwargs:
            ce = kwargs.get("ce", None)
            if ce is not None:
                self.ce = tf.Variable(ce, name="char_emb", trainable=trainable)
            else:
                self.ce = tf.get_variable(
                    "char_embedding",
                    [self.char_size, self.dim_char],
                    tf.float32,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                print("get random ce (char embedding).")

    @abstractmethod
    def _build_model(self):
        raise NotImplemented

    def _set_loss(self):
        softmax_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.in_y))
        self.loss = softmax_cost

        if self.params_reg:
            model_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "mlp")
            for p in model_params:
                self.loss += self.l2_norm * tf.nn.l2_loss(p)
            print("==> added params_reg in mlp.")

    def set_optimize(self, opt="adam"):
        if opt == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            raise KeyError("no such (%s) optimizer !" % opt)

        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, train_vars), self.clipper)
        self.train_op = optimizer.apply_gradients(
            zip(grads, train_vars),
            global_step=self.g_step)
        print("set_optimize : %s" % opt)

    def _set_predict(self):
        self.y_prob = tf.nn.softmax(self.logits)
        self.y = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        # Accuracy
        check_prediction = tf.equal(self.y, self.in_y)
        self.acc_num = tf.reduce_sum(tf.cast(check_prediction, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(check_prediction, tf.float32))

    def _log_summaries(self):
        """
        Adds summaries for the following variables to the graph and returns
        an operation to evaluate them.
        """
        cost = tf.summary.scalar("loss", self.loss)
        acc = tf.summary.scalar("acc", self.acc)
        self.merged = tf.summary.merge([cost, acc])

    def train(self, sess, batch, keep_rate=0.5):
        feed_dict = {
            self.in_y: batch[0],
            self.in_words: batch[1],
            self.keep_prob: keep_rate
        }
        _, step, cost, y = sess.run(
            [self.train_op, self.g_step, self.loss, self.y],
            feed_dict)
        return step, cost, y

    def predict(self, sess, batch):
        feed_dict = {
            self.in_words: batch[1],
            self.keep_prob: 1.0
        }
        return sess.run(self.y, feed_dict)

    def get_prob(self, sess, batch):
        feed_dict = {
            self.in_words: batch[1],
            self.keep_prob: 1.0
        }
        return sess.run(self.y_prob, feed_dict)

    def lr_decay(self, sess):
        old_lr = sess.run(self.lr)
        sess.run(self.lr_decay_op)
        new_lr = sess.run(self.lr)
        return old_lr, new_lr

    def save(self, sess, model_file, global_step):
        self.saver.save(sess, model_file, global_step=global_step)
        print("save model to file: %s" % model_file)

    def load(self, sess, model_file_dir):
        ckpt = tf.train.get_checkpoint_state(model_file_dir)
        print("ModelZoo file: ", ckpt.model_checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError("not exist Model in %s..." % model_file_dir)

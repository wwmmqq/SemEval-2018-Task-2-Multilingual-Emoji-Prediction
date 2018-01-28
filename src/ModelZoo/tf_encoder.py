# coding=utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_ops


def rnn_std(inputs, seq_len, n_rnn, rnn_type='lstm', drop_out=None, return_hs=False):
    """
    :param inputs: 3D tensor
    :param seq_len: 1D
    :param n_rnn: int
    :param rnn_type: 'gru' or 'lstm'
    :param drop_out: float or None
    :param return_hs: True / False
    :return: [batch x dim_h] or [batch x time x dim_h]
    """

    cell = tf.nn.rnn_cell.LSTMCell
    if rnn_type is 'gru':
        cell = tf.nn.rnn_cell.GRUCell
    cell_fw = cell(n_rnn)
    if drop_out is not None:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=drop_out)

    # state (c, final_state)
    outputs, state = tf.nn.dynamic_rnn(
        cell_fw, inputs, seq_len, dtype=tf.float32)
    if return_hs:
        return outputs
    return state[1]


def bi_rnn_std(inputs, seq_len, n_rnn, rnn_type='lstm', drop_out=None, return_hs=False, name='rnn'):
    """
    :param inputs: 3D tensor
    :param seq_len: 1D
    :param n_rnn: int
    :param rnn_type: 'gru' or 'lstm'
    :param drop_out: float or None
    :param return_hs: True / False
    :param name: string
    :return: [batch x 2*dim_h] or [batch x time x 2*dim_h]
    """

    with tf.variable_scope(name):
        cell = tf.nn.rnn_cell.LSTMCell
        if rnn_type is 'gru':
            cell = tf.nn.rnn_cell.GRUCell
        cell_fw = cell(n_rnn)
        cell_bw = cell(n_rnn)
    if drop_out is not None:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=drop_out)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=drop_out)

    outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
    if return_hs:
        return tf.concat(outputs, axis=-1)
    final_state = tf.concat([state_fw[1], state_bw[1]], axis=-1)
    return final_state


def rcnn(inputs, in_lens, n_rnn, name='rcnn'):
    """
    :param inputs: 3D tensor, [B, T, D]
    :param in_lens: 1D tensor, [B]
    :param n_rnn: int
    :param name string
    :return:
    """
    _, max_len_sent, _ = inputs.get_shape().as_list()
    rnn_out = bi_rnn_std(
        inputs, in_lens, n_rnn, return_hs=True, name=name)
    rcnn_in = tf.concat([rnn_out, inputs], axis=-1)
    rcnn_out = tf_ops.max2d_pooling(rcnn_in, max_len_sent)
    return rcnn_out


def lstm_std(x, x_lens, n_h, num_layers=1, bidirectional=False, return_hs=False, use_peepholes=True):
    """
    Args:
        x: 3-D tensor, [batch, max_len, token_dim]
        x_lens: integer tensor, [batch]
        n_h: int
        bidirectional: True / False
        num_layers: int
        return_hs: True/False
        use_peepholes: True/False
    Returns:
        if return is True:
            return [batch, max_len, h] or [batch, max_len, 2*h]
        else:
            return [batch, h] or [batch, 2*h]
    """
    def get_cell(rnn_size):
        return tf.nn.rnn_cell.LSTMCell(rnn_size, use_peepholes=use_peepholes)
    if num_layers > 1:
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [get_cell(n_h) for _ in range(num_layers)])
    else:
        cell_fw = get_cell(n_h)

    if bidirectional:
        if num_layers > 1:
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [get_cell(n_h) for _ in range(num_layers)])
        else:
            cell_bw = get_cell(n_h)

        b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x,
            sequence_length=x_lens,
            dtype=tf.float32)
        if return_hs:
            out = tf.concat(b_outputs, axis=2)
        else:
            if num_layers is not 1:
                out = tf.concat([b_states[-1][0][1], b_states[-1][1][1]], axis=-1)
            else:
                out = tf.concat([b_states[0][1], b_states[1][1]], axis=-1)
    else:
        outputs, state = tf.nn.dynamic_rnn(
            cell_fw, x, x_lens, dtype=tf.float32)
        if return_hs:
            out = outputs
        else:
            if num_layers is not 1:
                out = state[-1][1]
            else:
                out = state[1]
    return out


def cnn2d(x, k_hs, k_w, k_nums, stddev=0.1, activation=tf.tanh, name='cnn'):
    """
    Args:
        x: 3-D tensor [batch, max_len_sent, dim_word]
        k_hs: list of n-gram size, eg [2, 3, 4]
        k_w: word dim
        k_nums: kernel numbers of n-gram
        stddev: 0.1
        activation: tf.tanh or tf.relu ..
        name: string
    Returns: 2D [batch x seq_dim]
    """

    pool_outs = []
    # [batch, max_len_sent, dim_word, 1]
    x = tf.expand_dims(x, -1)
    max_len_sent = x.get_shape().as_list()[1]
    with tf.variable_scope(name):
        for i, (k_h, k_num) in enumerate(zip(k_hs, k_nums)):
            reduced_length = max_len_sent - k_h + 1

            # filter shape: [filter_height, filter_width, in_channels, out_channels]
            w = tf.get_variable(
                'w_%d' % i,
                shape=[k_h, k_w, 1, k_num],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev))
            b = tf.get_variable(
                name='b_%d' % i, shape=[k_num], dtype=tf.float32,
                initializer=tf.zeros_initializer)
            conv = tf.nn.conv2d(
                x, filter=w, strides=[1, 1, 1, 1], padding='VALID')

            conv = activation(tf.nn.bias_add(conv, b))
            # [batch_size x 1 x 1 x feature_map_dim]
            pool = tf.nn.max_pool(
                conv,
                [1, reduced_length, 1, 1],
                [1, 1, 1, 1], 'VALID')
            pool_outs.append(pool)

        outputs = tf.concat(pool_outs, -1)
        # # [batch x total_features]
        # outputs = tf.squeeze(outputs)
        # support for tf.layers.dense
        outputs = tf.reshape(outputs, shape=[-1, sum(k_nums)])
    return outputs


def char2word_rnn(inputs, seqs_len, ce, n_rnn, bidirectional=False, rnn_type='gru', name='c2w'):
    """
    Args:
        inputs: 3D tensor, [batch x max_len_sent x max_word_len]
        seqs_len: 2D [batch x max_len_seq]
        ce: chars embedding matrix
        n_rnn: int
        bidirectional: True/False
        rnn_type: string, 'lstm' or 'gru'
        name:
    Returns: words 3D tensor, [batch x max_len_sent x dim_word]
    """
    shape = inputs.get_shape().as_list()  # [32 x 30 x 10]
    seqs_t = tf.reshape(inputs, [-1, shape[-1]])  # [(32x30) x 10]
    seqs_len_t = tf.reshape(seqs_len, [-1])  # [(32x30)]
    seqs_embed = tf.nn.embedding_lookup(ce, seqs_t)  # [(32x30) x 10 x 300]

    cell = tf.nn.rnn_cell.GRUCell

    if rnn_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell

    words_dim = n_rnn
    with tf.variable_scope('rnn_%s' % name):
        cell_fw = cell(n_rnn)
        if bidirectional:
            cell_bw = cell(n_rnn)
            # states_fw/bw: [c, h]
            _, (states_fw,  states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, seqs_embed, seqs_len_t, dtype=tf.float32)
            words = tf.concat([states_fw[1], states_bw[1]], axis=-1)
            words_dim = 2*n_rnn
            print('==> bidirectional %s for char2word_rnn.' % rnn_type)
        else:
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs, states = tf.nn.dynamic_rnn(
                cell_fw, seqs_embed, seqs_len_t, dtype=tf.float32)
            words = states
            print('==> unidirectional %s for char2word_rnn.' % rnn_type)
    words = tf.reshape(words, [-1, shape[1], words_dim])
    return words


def char2word_cnn(inputs, ce, dim_char, k_h_n, name='c2w_cnn'):
    """
    Args:
        inputs: 3D tensor, [batch, max_len_sent, max_word_len]
        ce: chars embedding matrix
        dim_char: int
        k_h_n: kernel's height and number. eg, ((2, 50), (3, 50), (4, 50))
        name: string

    Returns: 3D tensor, [batch, max_len_sent, dim_c2w]

    """
    _, max_len_sent, max_len_word = inputs.get_shape().as_list()  # [32 x 30 x 10]
    seqs_t = tf.reshape(inputs, [-1, max_len_word])  # [(32 x 30) x 10]
    seqs_embed = tf.nn.embedding_lookup(ce, seqs_t)  # [(32 x 30) x 10 x 25]
    x = tf.expand_dims(seqs_embed, -1)  # [(32 x 30) x 10 x 25 x 1]
    pool_outs = []
    dim_c2w = 0
    with tf.variable_scope(name):
        for i, (k_h, k_n) in enumerate(k_h_n):
            dim_c2w += k_n
            reduced_length = max_len_word - k_h + 1
            # shape in w: [filter_height, filter_width, in_channels, out_channels]
            w = tf.get_variable(
                'w_%d' % i,
                shape=[k_h, dim_char, 1, k_n],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.02))
            conv = tf.nn.conv2d(
                x, filter=w, strides=[1, 1, 1, 1], padding='VALID')

            # [batch_size x 1 x 1 x feature_map_dim]
            pool = tf.nn.max_pool(
                tf.tanh(conv),
                [1, reduced_length, 1, 1],
                [1, 1, 1, 1], 'VALID')
            pool_outs.append(pool)
    print('==> char2word_cnn, dim_c2w %s' % dim_c2w)
    outputs = tf.concat(pool_outs, -1)
    outputs = tf.squeeze(outputs)  # [(32x30) x total_features]
    outputs = tf.reshape(outputs, [-1, max_len_sent, dim_c2w])  # [32 x 30 x dim_c2w]
    return outputs


def gate_char_word(in_char2w, in_words, name='gate_char_word'):
    """ g1 = sigmoid(W1 x in_words + b_1)
        g2 = sigmoid(W2 x in_char2ws + b_2)
        h = [g1*c,  g2*w]
    Args:
        in_char2w: 3D tensor, [batch, max_len_seq, dim_word]
        in_words: 3D tensor, [batch, max_len_seq, dim_word]
        name: string

    Returns: 3D tensor [batch, max_len_seq, dim_word+dim_char]
    """
    _, max_len_sent, dim_char = in_char2w.get_shape().as_list()
    _, max_len_sent, dim_word = in_words.get_shape().as_list()
    with tf.variable_scope(name):
        w1 = tf.get_variable(
            'w1',
            [dim_word, dim_char], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(0.02))
        b1 = tf.get_variable(
            'b1', [dim_char], dtype=tf.float32,
            initializer=tf.zeros_initializer)
        w2 = tf.get_variable(
            'w2',
            [dim_char, dim_word], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(0.02))
        b2 = tf.get_variable(
            'b2', [dim_word], dtype=tf.float32,
            initializer=tf.zeros_initializer)

        in_char2w = tf.reshape(in_char2w, [-1, dim_char])
        in_words = tf.reshape(in_words, [-1, dim_word])
        gate1 = tf.sigmoid(tf.nn.xw_plus_b(in_words, w1, b1))
        gate2 = tf.sigmoid(tf.nn.xw_plus_b(in_char2w, w2, b2))
        h = tf.concat([in_char2w * gate1, in_words * gate2], -1)
        print('=> gate_char_word, dim_word+dim_char: %d' % (dim_word+dim_char))
        return tf.reshape(h, [-1, max_len_sent, dim_word+dim_char])


def fine_grained_gate(in_chars, in_words, in_nlp_features):
    """ g = sigmoid(W_g x v + b_g)   # v is in_nlp_features vector
        h = g * c + (1 − g) * w

        * we assume c and w have the same dim *
    Args:
        in_chars: 3D tensor, [batch x max_len_seq x dim_char]
        in_words: 3D tensor, [batch x max_len_seq x dim_word]
        in_nlp_features: 2D tensor, [batch x nlp_feature_dim]

    Returns: 3D tensor, [batch x max_len_seq x dim_word]
    """
    _, max_len_sent, dim_char = in_chars.get_shape().as_list()
    _, max_len_sent, dim_word = in_words.get_shape().as_list()
    assert dim_char == dim_word, 'dim_char != dim_word'

    _, dim_features = in_nlp_features.get_shape().as_list()
    with tf.variable_scope('fine_grained_gate'):
        w_g = tf.get_variable(
            'w_g',
            [dim_features, dim_word], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(0.02))
        w_b = tf.get_variable(
            'b', [dim_word], dtype=tf.float32,
            initializer=tf.zeros_initializer)

        in_chars = tf.reshape(in_chars, [-1, dim_word])
        in_words = tf.reshape(in_words, [-1, dim_word])
        gate = tf.sigmoid(tf.nn.xw_plus_b(in_nlp_features, w_g, w_b))  # [(32x30) x 300]
        h = in_chars * gate + in_words * (1. - gate)
        return tf.reshape(h, [-1, max_len_sent, dim_word])


def char2word_gate_word(inputs_char, seqs_len_word, ce, n_rnn_char, inputs_word):
    """
    :return 3D tensor [batch, max_len_seq, dim_word+dim_char]
    """
    seqs_char2word = char2word_rnn(inputs_char, seqs_len_word, ce, n_rnn_char)
    return gate_char_word(seqs_char2word, inputs_word)


def word2sent(inputs, seqs_len, n_rnn, rnn_type="gru", return_hs=False, name="w2s"):
    """
    Args:
        inputs: 3D tensor, [batch, max_sent_len, dim_word]
        seqs_len: 1D [max_sent_len]
        n_rnn: int
        rnn_type: string ["gru" or "lstm"]
        return_hs: False/True
        name: string
    Returns: 2D tensor [batch, seq_dim] or 3D tensor [batch, max_sent_len, seq_dim]
    """
    cell = tf.nn.rnn_cell.GRUCell
    if rnn_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell

    with tf.variable_scope('%s_rnn_gru' % name):
        cell_bw = cell(n_rnn)
        cell_fw = cell(n_rnn)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, inputs,
            sequence_length=seqs_len,
            dtype=tf.float32)

    if return_hs:
        rst = tf.concat(outputs, -1)
    else:
        rst = tf.concat([states[0], states[1]], -1)
        # gru states: (f_s, b_s)
        # lstm states: (f_s[1], b_s[1])
    return rst

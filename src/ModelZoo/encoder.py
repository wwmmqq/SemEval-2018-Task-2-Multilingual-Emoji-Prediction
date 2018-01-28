# coding: utf-8
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    _BATCH_NORM_DECAY = 0.997
    _BATCH_NORM_EPSILON = 1e-5
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def gate_char_word(in_char2ws, in_words, name='gate_char_word'):
    """ g1 = sigmoid(W1 x in_words + b_1)
        g2 = sigmoid(W2 x in_char2ws + b_2)
        h = [g1*c,  g2*w]
    Args:
        in_char2ws: 3D tensor, [batch, max_len_seq, dim_word]
        in_words: 3D tensor, [batch, max_len_seq, dim_word]
        name: string

    Returns: 3D tensor [batch, max_len_seq, dim_word+dim_char]
    """
    _, max_len_sent, dim_char = in_char2ws.get_shape().as_list()
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

        in_char2ws = tf.reshape(in_char2ws, [-1, dim_char])
        in_words = tf.reshape(in_words, [-1, dim_word])
        gate1 = tf.sigmoid(tf.nn.xw_plus_b(in_words, w1, b1))
        gate2 = tf.sigmoid(tf.nn.xw_plus_b(in_char2ws, w2, b2))
        h = tf.concat([in_char2ws * gate1, in_words * gate2], -1)
        print ('  => gate_char_word, dim_word+dim_char: %d' % (dim_word+dim_char))
        return tf.reshape(h, [-1, max_len_sent, dim_word+dim_char])


def fine_grained_gate(in_chars, in_words, in_nlp_features):
    """ g = sigmoid(W_g x v + b_g)
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


def char2word_rnn(seqs, seqs_len, ce, n_rnn, bidirectional=False, rnn_type='gru', name='c2w'):
    """
    Args:
        seqs: 3D tensor, [batch x max_len_sent x max_word_len]
        seqs_len: 2D [batch x max_len_seq]
        ce: chars embedding matrix
        n_rnn: int
        bidirectional: True/False
        rnn_type: string, 'lstm' or 'gru'
        name:
    Returns: words 3D tensor, [batch x max_len_sent x dim_word]
    """
    shape = seqs.get_shape().as_list()  # [32 x 30 x 10]
    seqs_t = tf.reshape(seqs, [-1, shape[-1]])  # [(32x30) x 10]
    seqs_len_t = tf.reshape(seqs_len, [-1, ])  # [(32x30)]
    seqs_embed = tf.nn.embedding_lookup(ce, seqs_t)  # [(32x30) x 10 x 300]

    if rnn_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell
    elif rnn_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell
    else:
        raise TypeError('no such rnn type %s' % rnn_type)

    with tf.variable_scope('gru_%s' % name):
        cell_fw = cell(n_rnn)
        if bidirectional:
            cell_bw = cell(n_rnn)
            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, seqs_embed, seqs_len_t, dtype=tf.float32)
            words = b_states[0] + b_states[1]
            print('  ==> bidirectional gru for char2word_rnn.')
        else:
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs, states = tf.nn.dynamic_rnn(
                cell_fw, seqs_embed, seqs_len_t, dtype=tf.float32)
            words = states
            print ('  ==> unidirectional gru for char2word_rnn.')
    words = tf.reshape(words, [-1, shape[1], n_rnn])
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
    print ('  ==> char2word_cnn, dim_c2w %s' % dim_c2w)
    outputs = tf.concat(pool_outs, -1)
    outputs = tf.squeeze(outputs)  # [(32x30) x total_features]
    outputs = tf.reshape(outputs, [-1, max_len_sent, dim_c2w])  # [32 x 30 x dim_c2w]
    return outputs


def word2sent(inputs, seqs_len, n_rnn, return_hs=False, name='w2s'):
    """
    Args:
        inputs: 3D tensor, [batch, max_sent_len, dim_word]
        seqs_len: 1D [max_sent_len]
        n_rnn: int
        return_hs: False/True
        name: string
    Returns: 2D tensor [batch, seq_dim] or 3D tensor [batch, max_sent_len, seq_dim]
    """

    with tf.variable_scope('%s_rnn_gru' % name):
        cell_bw = tf.nn.rnn_cell.GRUCell(n_rnn)
        cell_fw = tf.nn.rnn_cell.GRUCell(n_rnn)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, inputs,
            sequence_length=seqs_len,
            dtype=tf.float32)

    if return_hs:
        rst = tf.concat(outputs, -1)
    else:
        rst = tf.concat([states[0], states[1]], -1)
    return rst

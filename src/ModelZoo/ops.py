# coding: utf-8
# tensorflow Version: 1.4.0
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def avg_pool(x, in_len):
    """
    Args:
        x: 3-D tensor, [batch, max_len, dim]
        in_len: integer tensor, [batch]
    Returns: 2-D tensor, [batch, dim]
    """
    # batch x dim
    x_sum = tf.reduce_sum(x, axis=1)
    in_len = tf.cast(in_len, tf.float32)
    avg = x_sum / tf.expand_dims(in_len, -1)
    return avg


def cnn2d(x, k_hs, k_w, k_nums, stddev=0.02, name='cnn'):
    """
    Args:
        x: 3-D tensor [batch, max_len_sent, dim_word]
        k_hs: list of n-gram size, eg [2, 3, 4]
        k_w: word dim
        k_nums: kernel numbers of n-gram
        stddev: 0.02
        name:
    Returns: 2D [batch x seq_dim]
    """

    pool_outs = []
    # [batch, max_len_sent, dim_word, 1]
    x = tf.expand_dims(x, -1)

    with tf.variable_scope(name):
        for i, (k_h, k_num) in enumerate(zip(k_hs, k_nums)):
            reduced_length = x.get_shape()[1] - k_h + 1

            # [filter_height, filter_width, in_channels, out_channels]
            w = tf.get_variable(
                'w_%d' % i,
                shape=[k_h, k_w, 1, k_num],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev))

            conv = tf.nn.conv2d(
                x, filter=w, strides=[1, 1, 1, 1], padding='VALID')

            # [batch_size x 1 x 1 x feature_map_dim]
            pool = tf.nn.max_pool(
                tf.tanh(conv),
                [1, reduced_length, 1, 1],
                [1, 1, 1, 1], 'VALID')
            pool_outs.append(pool)

        outputs = tf.concat(pool_outs, -1)
        # [batch x total_features]
        outputs = tf.squeeze(outputs)

        # support for tf.layers.dense
        outputs = tf.reshape(outputs, shape=[-1, sum(k_nums)])
    return outputs


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
                # out = tf.concat([b_states[0], b_states[1]], -1)
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


def lstm(in_x, in_size, out_size, seq_len, max_seq_len,
         dtype, initializer=tf.orthogonal_initializer, scope="lstm"):
    """
    Args:
        in_x: 3-D tensor [batch x max_len, dim_word]
        in_size: the last dim of in_chars
        out_size: lstm-size
        seq_len: 1-D tensor [batch x 1] for length of sentence
        max_seq_len:
        dtype:
        initializer:
        scope:

    Returns: hs [batch x max_len x out_size]   eg. (32, 53, 150)
             h [batch x out_size]
    """
    with tf.variable_scope(scope):
        w_xh = tf.get_variable(
            'w_xh', [in_size, out_size*4],
            dtype=dtype,
            initializer=initializer)
        w_hh = tf.get_variable(
            'w_hh', [out_size, out_size*4],
            dtype=dtype,
            initializer=initializer)
        b = tf.get_variable(
            'biases', [out_size*4],
            dtype=dtype,
            initializer=tf.zeros_initializer)

    def _step(x_, c_, h_, m_):
        """
        i, u, f, c, o, h have a same shape: [batch x n_rnn_word]
        m_ : [batch]
        """
        res_x = tf.matmul(x_, w_xh)
        res_h = tf.matmul(h_, w_hh)
        res = res_x + res_h + b
        # i = input_gate, u = new_input, f = forget_gate, o = output_gate
        i, u, f, o = tf.split(res, 4, axis=1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.nn.tanh(u)

        m_ = tf.expand_dims(m_, 1)  # [batch x 1]
        new_c = c_ * f + u * i
        new_c = new_c * m_ + c_ * (1.0 - m_)
        new_h = tf.nn.tanh(new_c) * o
        new_h = new_h * m_ + h_ * (1.0 - m_)
        return new_c, new_h, f*m_

    h_list = []
    f_list = []
    seqs = tf.unstack(in_x, num=in_x.get_shape().as_list()[1], axis=1)
    mask = tf.sequence_mask(seq_len, maxlen=max_seq_len, dtype=dtype)
    mask = tf.unstack(mask, num=in_x.get_shape().as_list()[1], axis=1)
    c = tf.zeros([1, out_size], dtype=dtype)
    h = tf.zeros([1, out_size], dtype=dtype)
    for x, m in zip(seqs, mask):
        """ x shape: [batch x dim_word]
        """
        c, h, f = _step(x, c, h, m)
        h_list.append(h)
        f_list.append(f)
    hs = tf.stack(h_list)
    hs = tf.transpose(hs, perm=[1, 0, 2])
    fs = tf.stack(f_list)
    fs = tf.transpose(fs, perm=[1, 0, 2])

    return hs, h, fs


def bilstm(in_x, in_size, out_size, seq_len, max_seq_len,
           dtype, initializer=tf.orthogonal_initializer, scope="lstm"):
        in_x_reverse = tf.reverse_sequence(
            input=in_x,
            seq_lengths=tf.cast(seq_len, tf.int32), seq_dim=1, batch_dim=0)
        hs_fw, h_fw, fs_fw = lstm(
            in_x, in_size, out_size, seq_len, max_seq_len,
            dtype, initializer=initializer, scope=scope+"_fw")
        hs_bw, h_bw, fs_bw = lstm(
            in_x_reverse, in_size, out_size, seq_len, max_seq_len,
            dtype, initializer=initializer, scope=scope + "_bw")
        return hs_fw, h_fw, hs_bw, h_bw


def skip_lstm(in_x, in_size, out_size, seq_len, max_seq_len,
              dtype=tf.float32, initializer=tf.orthogonal_initializer, scope="lstm"):
    """
    Args: start c, h, 2-D tensor with shape [batch x n_rnn_word]
        in_chars: 3-D tensor [batch x max_len, dim_word]
        in_size: the last dim of in_chars
        out_size: lstm-size
        seq_len: 1-D tensor [batch x 1] for length of sentence
        max_len_sent:
        dtype:
        initializer:
        scope:

    Returns: hs [batch x max_len x out_size]   eg. (32, 53, 150)
             h [batch x out_size]
    """
    with tf.variable_scope(scope):
        w_xh = tf.get_variable(
            'w_xh', [in_size, out_size*4],
            dtype=dtype,
            initializer=initializer)
        w_hh = tf.get_variable(
            'w_hh', [out_size, out_size*4],
            dtype=dtype,
            initializer=initializer)
        b = tf.get_variable(
            'biases', [out_size*4],
            dtype=dtype,
            initializer=tf.zeros_initializer)

        c_w = tf.get_variable(
            'c_w_for_gate', [out_size, out_size],
            dtype=dtype,
            initializer=initializer)
        c_w_b = tf.get_variable(
            'c_w_biases', [out_size],
            dtype=dtype,
            initializer=tf.zeros_initializer)

    def gate(c_, old_c):
        return c_ + tf.nn.relu(tf.matmul(old_c, c_w) + c_w_b)

    def _step(x_, c_, h_, m_, old_c):
        """
        i, u, f, c, o, h have a same shape: [batch x n_rnn_word]
        m_ : [batch]
        """
        res_x = tf.matmul(x_, w_xh)
        res_h = tf.matmul(h_, w_hh)
        res = res_x + res_h + b
        # i = input_gate, u = new_input, f = forget_gate, o = output_gate
        i, u, f, o = tf.split(res, 4, axis=1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.nn.tanh(u)

        m_ = tf.expand_dims(m_, 1)  # [batch x 1]
        c_ = gate(c_, old_c)
        new_c = c_ * f + u * i
        new_c = new_c * m_ + c_ * (1.0 - m_)
        new_h = tf.nn.tanh(new_c) * o
        new_h = new_h * m_ + h_ * (1.0 - m_)
        return new_c, new_h, f*m_

    c = tf.zeros([1, out_size], dtype=dtype)
    h = tf.zeros([1, out_size], dtype=dtype)

    seqs = tf.unstack(in_x, num=in_x.get_shape().as_list()[1], axis=1)
    mask = tf.sequence_mask(seq_len, maxlen=max_seq_len, dtype=dtype)
    mask = tf.unstack(mask, num=in_x.get_shape().as_list()[1], axis=1)
    h_list = []
    f_list = []
    c_list = [c, c]
    for x, m in zip(seqs, mask):
        """ x shape: [batch x dim_word]
        """
        c, h, f = _step(x, c, h, m, c_list[-2])
        h_list.append(h)
        f_list.append(f)
        c_list.append(c)
    hs = tf.stack(h_list)
    hs = tf.transpose(hs, perm=[1, 0, 2])
    fs = tf.stack(f_list)
    fs = tf.transpose(fs, perm=[1, 0, 2])

    return hs, h, fs


def skip_bilstm(in_x, in_size, out_size, seq_len, max_seq_len,
                dtype, initializer=tf.orthogonal_initializer, scope="lstm"):
    in_x_reverse = tf.reverse_sequence(
        input=in_x,
        seq_lengths=tf.cast(seq_len, tf.int32), seq_dim=1, batch_dim=0)
    hs_fw, h_fw, fs_fw = skip_lstm(
        in_x, in_size, out_size, seq_len, max_seq_len,
        dtype, initializer=initializer, scope=scope + "_fw")
    hs_bw, h_bw, fs_bw = skip_lstm(
        in_x_reverse, in_size, out_size, seq_len, max_seq_len,
        dtype, initializer=initializer, scope=scope + "_bw")
    return hs_fw, h_fw, fs_fw, hs_bw, h_bw, fs_bw


def peephole_lstm(in_x, in_size, out_size, seq_len, max_seq_len,
                  dtype, initializer=tf.orthogonal_initializer, scope="lstm"):
    """
    Args:
        in_x: 3-D tensor [batch x max_len, dim_word]
        in_size: the last dim of in_chars
        out_size: lstm-size
        seq_len: 1-D tensor [batch x 1] for length of sentence
        max_seq_len:
        dtype:
        initializer:
        scope:

    Returns: hs [batch x max_len x out_size]   eg. (32, 53, 150)
             h [batch x out_size]
    """
    with tf.variable_scope(scope):
        w_xi = tf.get_variable(
            'w_xi',
            [in_size, out_size], dtype=dtype,
            initializer=initializer)
        w_hi = tf.get_variable(
            'w_hi',
            [out_size, out_size], dtype=dtype,
            initializer=initializer)
        w_ci = tf.get_variable(
            'w_ci',
            [out_size, out_size], dtype=dtype,
            initializer=initializer)
        b_i = tf.get_variable(
            'b_i', [out_size], dtype=dtype,
            initializer=tf.zeros_initializer)

        w_xf = tf.get_variable(
            'w_xf',
            [in_size, out_size], dtype=dtype,
            initializer=initializer)
        w_hf = tf.get_variable(
            'w_hf',
            [out_size, out_size], dtype=dtype,
            initializer=initializer)
        w_cf = tf.get_variable(
            'w_cf',
            [out_size, out_size], dtype=dtype,
            initializer=initializer)
        b_f = tf.get_variable(
            'b_f',
            [out_size],  dtype=dtype,
            initializer=tf.zeros_initializer)

        w_xc = tf.get_variable(
            'w_xc',
            [in_size, out_size], dtype=dtype,
            initializer=initializer)
        w_hc = tf.get_variable(
            'w_hc',
            [out_size, out_size], dtype=dtype,
            initializer=initializer)
        b_c = tf.get_variable(
            'b_c',
            [out_size], dtype=dtype,
            initializer=tf.zeros_initializer)

        w_xo = tf.get_variable(
            'w_xo', [in_size, out_size], dtype=dtype,
            initializer=initializer)
        w_ho = tf.get_variable(
            'w_ho', [out_size, out_size], dtype=dtype,
            initializer=initializer)
        w_co = tf.get_variable(
            'w_co', [out_size, out_size], dtype=dtype,
            initializer=initializer)
        b_o = tf.get_variable(
            'b_o', [out_size], dtype=dtype,
            initializer=tf.zeros_initializer)

    def _step(x_, c_, h_, m_):
        """ i, u, f, c, o, h have a same shape: [batch x n_rnn_word]
        Args:
            x_: [batch x in_size]  ( [batch x dim_word] )
            c_: [batch x n_rnn_word]
            h_: [batch x n_rnn_word]
            m_: [batch]

        Returns: new_c , new_h , f

        """
        i = tf.nn.sigmoid(
            tf.matmul(x_, w_xi) +
            tf.matmul(h_, w_hi) +
            tf.matmul(c_, w_ci) + b_i)
        f = tf.nn.sigmoid(
            tf.matmul(x_, w_xf) +
            tf.matmul(h_, w_hf) +
            tf.matmul(c_, w_cf) + b_f)
        u = tf.nn.tanh(
            tf.matmul(x_, w_xc) +
            tf.matmul(h_, w_hc) + b_c)

        m_ = tf.expand_dims(m_, 1)  # [batch x 1]
        new_c = c_ * f + u * i
        new_c = new_c * m_ + c_ * (1.0 - m_)

        o = tf.nn.sigmoid(
            tf.matmul(x_, w_xo) +
            tf.matmul(h_, w_ho) +
            tf.matmul(new_c, w_co)
            + b_o)

        new_h = tf.nn.tanh(new_c) * o
        new_h = new_h * m_ + h_ * (1.0 - m_)

        return new_c, new_h, f*m_

    h_list = []
    f_list = []
    seqs = tf.unstack(in_x, num=in_x.get_shape().as_list()[1], axis=1)
    mask = tf.sequence_mask(seq_len, maxlen=max_seq_len, dtype=dtype)
    mask = tf.unstack(mask, num=in_x.get_shape().as_list()[1], axis=1)
    c = tf.zeros([1, out_size], dtype=dtype)
    h = tf.zeros([1, out_size], dtype=dtype)
    for x, m in zip(seqs, mask):
        """ x shape: [batch x dim_word]
        """
        c, h, f = _step(x, c, h, m)
        h_list.append(h)
        f_list.append(f)
    hs = tf.stack(h_list)
    hs = tf.transpose(hs, perm=[1, 0, 2])
    fs = tf.stack(f_list)
    fs = tf.transpose(fs, perm=[1, 0, 2])

    return hs, h, fs

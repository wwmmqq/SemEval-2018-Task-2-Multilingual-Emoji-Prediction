# coding: utf-8
from __future__ import print_function
import tensorflow as tf


def attention_self(in_x, in_len, in_dim, u_dim, name=0):
    in_x_shape = tf.shape(in_x)  # [batch_size, max_time, h_size]
    in_x_temp = tf.reshape(in_x, [in_x_shape[0]*in_x_shape[1], in_x_shape[2]])
    w_x = tf.get_variable(
        'w_self_hidden_%d' % name,
        [in_dim, u_dim],
        initializer=tf.random_normal_initializer(stddev=0.01))
    b_x = tf.get_variable(
        'b_self_hidden_%d' % name,
        [u_dim],
        initializer=tf.zeros_initializer)
    u = tf.nn.xw_plus_b(in_x_temp, w_x, b_x)  # [(batch*max_time) x u_dim]
    u = tf.tanh(u)  # [(batch x time_step) x u_dim]

    w_u = tf.get_variable(
        'w_self_u_%d' % name,
        [u_dim, 1],
        initializer=tf.random_normal_initializer(stddev=0.01))
    att = tf.matmul(u, w_u)  # [(batch x time_step)x1]
    att = tf.reshape(att, [in_x_shape[0], in_x_shape[1]])   # [batch x max_time]
    # [batch x max_time]
    mask = tf.sequence_mask(in_len, maxlen=in_x_shape[1], dtype=tf.float32)
    att = tf.exp(att) * mask
    att_sum = tf.reduce_sum(att, axis=1, keep_dims=True)  # [batch x 1]
    att_prob = att / att_sum  # [batch x time_step]
    return att_prob


def self_attention(in_x, in_len, u_size, name='att'):
    """
    Args:
        in_x:  3-D tensor [batch x max_time x h_size]
        in_len:
        u_size:
        name:

    Returns: 2-D tensor [batch x max_time]

    """
    # [batch_size, max_time, h_size]
    in_x_shape = tf.shape(in_x)
    in_x_temp = tf.reshape(
        in_x,
        [in_x_shape[0] * in_x_shape[1], in_x_shape[2]])
    w_x = tf.get_variable(
        'w_self_hidden_%s' % name,
        [in_x.get_shape().as_list()[-1], u_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
    b_x = tf.get_variable(
        'b_self_hidden_%s' % name,
        [u_size],
        initializer=tf.zeros_initializer)

    # [(batch x max_time) x u_size]
    u = tf.matmul(in_x_temp, w_x) + b_x
    u = tf.tanh(u)

    w_u = tf.get_variable(
        'w_self_u_%s' % name,
        [u_size, 1],
        initializer=tf.random_normal_initializer(stddev=0.01))
    # [(batch x max_time) x 1]
    att = tf.matmul(u, w_u)
    # [batch x max_time]
    att = tf.reshape(att, [in_x_shape[0], in_x_shape[1]])
    # [batch x max_time]
    mask = tf.sequence_mask(
        in_len, maxlen=in_x_shape[1], dtype=tf.float32)
    att = tf.exp(att) * mask
    # [batch x max_time]
    att_sum = tf.reduce_sum(att, axis=1, keep_dims=True)
    att_prob = att / att_sum

    return att_prob


def structured_self_attention(in_x, in_len, u_size, att_num, name='str_att'):
    """
    Args:
        in_x: 3-D tensor [batch x max_time x h_size]
        in_len: sequence length
        u_size:
        att_num: multi size
        name:
    Returns: att_prob, a 3-D tensor [batch x max_time x r]

    """
    # Because ce use batch, H is a 3D matrix while w_x and w_u are 2D matrix.
    # M = A * H = softmax(w_u * tanh(w_h * H^T)) * H

    # [batch_size, max_time, h_size]
    in_x_shape = tf.shape(in_x)
    # [(batch_size x max_time), h_size]
    in_x_temp = tf.reshape(
        in_x,
        [-1, in_x_shape[2]])
    w_x = tf.get_variable(
        'w_self_hidden_%s' % name,
        [in_x.get_shape().as_list()[-1], u_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
    b_x = tf.get_variable(
        'b_self_hidden_%s' % name,
        [u_size],
        initializer=tf.zeros_initializer)

    # [(batch x max_time) x u_size]
    u = tf.tanh(
        tf.matmul(in_x_temp, w_x) + b_x)

    w_u = tf.get_variable(
        'w_self_u_%s' % name,
        [u_size, att_num],
        initializer=tf.random_normal_initializer(stddev=0.01))
    # [(batch x max_time) x att_num]
    att = tf.matmul(u, w_u)

    # [batch x max_time x att_num]
    att = tf.reshape(att, [in_x_shape[0], in_x_shape[1], att_num])

    # [batch x max_time]
    mask = tf.sequence_mask(
        in_len, maxlen=in_x_shape[1], dtype=tf.float32)
    # [batch x max_time x att_num]
    # att = tf.exp(att) * tf.expand_dims(mask, axis=2)
    # [batch x max_time x 1]
    # att_sum = tf.reduce_sum(att, axis=2, keep_dims=True)

    # [batch x max_time x att_num]
    # att_prob = att / att_sum

    att_prob = tf.nn.softmax(att, dim=1)
    att_prob = att_prob * tf.expand_dims(mask, axis=2)
    return att_prob


def self_attention_topk(in_x, in_len, u_size, name='att'):
    """
    Args:
        in_x:  3-D tensor [batch x max_time x h_size]
        in_len:
        u_size:
        name:

    Returns: 2-D tensor [batch x max_time]

    """
    in_x_shape = tf.shape(in_x)  # [batch_size, max_time, h_size]
    in_x_temp = tf.reshape(
        in_x, [in_x_shape[0] * in_x_shape[1], in_x_shape[2]])
    w_x = tf.get_variable(
        'w_self_hidden_%s' % name,
        [in_x.get_shape().as_list()[-1], u_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
    b_x = tf.get_variable(
        'b_self_hidden_%s' % name,
        [u_size],
        initializer=tf.zeros_initializer)

    # [(batch x max_time) x u_size]
    u = tf.matmul(in_x_temp, w_x) + b_x
    u = tf.tanh(u)

    w_u = tf.get_variable(
        'w_self_u_%s' % name,
        [u_size, 1],
        initializer=tf.random_normal_initializer(stddev=0.01))
    # [(batch x max_time) x 1]
    att = tf.matmul(u, w_u)
    # [batch x max_time]
    att = tf.reshape(att, [in_x_shape[0], in_x_shape[1]])
    # [batch x max_time]
    mask = tf.sequence_mask(in_len, maxlen=in_x_shape[1], dtype=tf.float32)
    att = tf.exp(att) * mask
    # [batch x 1]
    att_sum = tf.reduce_sum(att, axis=1, keep_dims=True)
    att_prob = att / att_sum

    topk = 6
    # [batch x topk]
    v, idx = tf.nn.top_k(att_prob, k=topk, sorted=False)
    # create full indices
    my_range = tf.range(0, tf.shape(att_prob)[0], dtype=tf.int32)  # [batch]
    my_range = tf.expand_dims(my_range, 1)       # [batch x 1]
    my_range = tf.tile(my_range, [1, topk])      # [batch x topk]
    # [batch x 3 x 1]  and [batch x 3 x 1]  = > [batch x 3 x 2]
    full_idx = tf.concat(
        [tf.expand_dims(my_range, 2), tf.expand_dims(idx, 2)], axis=2)
    full_idx = tf.reshape(full_idx, [-1, 2])     # [(batch x 3) x 2]

    # reorder the full index
    full_idx = tf.SparseTensor(
        values=tf.reshape(v, [-1]),
        indices=tf.cast(full_idx, dtype=tf.int64),
        dense_shape=tf.shape(att_prob, out_type=tf.int64))
    full_idx = tf.sparse_reorder(full_idx, 're_order')

    # topk_att_prob = tf.sparse_to_dense(
    #     full_idx, tf.shape(att_prob), 1.0, default_value=0.0)
    # full_idx must be ordered !!!!!!!!!!!!!!!!

    topk_att_prob = tf.sparse_tensor_to_dense(full_idx)
    return topk_att_prob

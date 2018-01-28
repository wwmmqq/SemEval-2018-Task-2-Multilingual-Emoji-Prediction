# coding=utf-8
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def dense_layer(inputs, n_in, n_out, activation=None, scope='fc'):
    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w', [n_in, n_out], dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [n_out], dtype=tf.float32)
    h = tf.nn.xw_plus_b(inputs, w, b)
    if activation:
        h = activation(h)
    return h


def avg_pool(inputs, in_len):
    """  B: batch T: max_len D: dim
    Args:
        inputs: 3-D tensor, [B, T, D]
        in_len: integer tensor, [B]
    Returns: 2-D tensor, [B, D]
    """
    # batch inputs dim
    x_sum = tf.reduce_sum(inputs, axis=1)  # [B, D]
    in_len = tf.expand_dims(tf.cast(in_len, tf.float32), -1)  # [B, 1]
    avg = x_sum / in_len    # [B, D] / [B, 1]
    return avg


def avg_pooling(inputs, in_len):
    """
    :param inputs: 3D tensor, [batch, time, dim]
    :param in_len: 1D, [batch]
    :return: 2D, [batch, dim]
    """
    max_len_sent = inputs.get_shape().as_list()[1]
    mask = tf.sequence_mask(in_len, max_len_sent, dtype=tf.float32)
    norm = mask / (tf.reduce_sum(mask, -1, keep_dims=True) + 1e-30)
    output = tf.reduce_sum(inputs * tf.expand_dims(norm, -1), axis=1)
    return output


def max_pooling(inputs, inputs_len):
    """
    Max pooling.
    Args:
        inputs: [batch, time, embedding]
        inputs_len: [batch]
    Returns:
        [batch, sent_embedding]
    """
    max_len_sent = inputs.get_shape().as_list()[1]
    mask = tf.sequence_mask(inputs_len, max_len_sent, dtype=tf.float32)
    mask = tf.expand_dims((1 - mask) * -1e30, -1)
    output = tf.reduce_max(inputs + mask, axis=1)
    return output


def max2d_pooling(inputs, max_len_sent):
    """
    :param inputs: 3D tensor [B, T, D]
    :param max_len_sent:
    :return:
    """
    contex = tf.expand_dims(inputs, -1)  # [B, T, D, 1]
    pooled = tf.nn.max_pool(contex,
                            ksize=[1, max_len_sent, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
    pooled = tf.squeeze(pooled)  # [B, D]
    return pooled


def self_attention(inputs, inputs_len, att_size, return_alphas=False):
    """" Hierarchical Attention Networks for Document Classification, 2016
        http://www.aclweb.org/anthology/N16-1174.
    :param inputs: 3D tensor, [batch, time, embedding]
    :param inputs_len: 1D tensor, [batch]
    :param att_size:
    :param return_alphas:
    :return:
    """
    _, maxlen, dim_h = inputs.get_shape().as_list()
    w_x = tf.Variable(tf.random_normal([dim_h, att_size], stddev=0.1))
    b_x = tf.Variable(tf.zeros([att_size]))
    # [B, T, D] dot [D, A] = [B, T, A], where A=att_size
    v = tf.tanh(tf.tensordot(inputs, w_x, axes=1) + b_x)

    w_u = tf.Variable(tf.random_normal([att_size], stddev=0.1))
    u = tf.tensordot(v, w_u, axes=1)  # [B, T]
    alphas = tf.nn.softmax(u)
    mask = tf.sequence_mask(inputs_len, maxlen=maxlen, dtype=tf.float32)
    alphas = alphas * mask

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if return_alphas:
        return output, alphas
    return output


def structured_self_attention(inputs, in_len, att_size, att_num, return_alpha=False, name='str_att'):
    """
    Args:
        inputs: 3-D tensor [batch, max_time, h_size]
        in_len: sequence length
        att_size: int
        att_num: multi size
        return_alpha: True / False
        name:
    Returns:
        2-D tensor [batch x (inputs_shape[2]*att_num)]
        | alpha, a 3-D tensor [batch, max_time, r]

    """
    # Because ce use batch, H is a 3D matrix while w_x and w_u are 2D matrix.
    # M = A * H = softmax(w_u * tanh(w_h * H^T)) * H

    # [batch_size, max_time, h_size]
    in_shape = inputs.get_shape().as_list()
    # [(batch_size x max_time), h_size]
    in_x_temp = tf.reshape(
        inputs,
        [-1, in_shape[2]])
    w_x = tf.get_variable(
        'w_self_hidden_%s' % name,
        [inputs.get_shape().as_list()[-1], att_size],
        initializer=tf.random_normal_initializer(stddev=0.01))
    b_x = tf.get_variable(
        'b_self_hidden_%s' % name,
        [att_size],
        initializer=tf.zeros_initializer)

    # [(batch x max_time) x att_size]
    u = tf.tanh(
        tf.matmul(in_x_temp, w_x) + b_x)

    w_u = tf.get_variable(
        'w_self_u_%s' % name,
        [att_size, att_num],
        initializer=tf.random_normal_initializer(stddev=0.01))
    # [(batch x max_time) x att_num]
    att = tf.matmul(u, w_u)

    # [batch x max_time x att_num]
    att = tf.reshape(att, [-1, in_shape[1], att_num])

    # [batch x max_time]
    mask = tf.sequence_mask(
        in_len, maxlen=in_shape[1], dtype=tf.float32)
    # [batch x max_time x att_num]
    att = tf.exp(att) * tf.expand_dims(mask, axis=2)
    # [batch x max_time x 1]
    att_sum = tf.reduce_sum(att, axis=2, keep_dims=True)

    # [batch x max_time x att_num]
    alphas = att / att_sum
    # alphas = tf.nn.softmax(att, dim=1)

    # [batch, dim, max_time]
    h_t = tf.transpose(inputs, perm=[0, 2, 1], name='h_T')
    # in tensorflow, [B, D, T] dot [B, T, N_A] = [B, D, N_A]
    sent_rep_2d = tf.matmul(h_t, alphas)
    # flatten
    sent_rep = tf.reshape(sent_rep_2d, shape=[-1, in_shape[2]*att_num])
    if return_alpha:
        return sent_rep, alphas
    return alphas


def self_attention_topk(in_x, in_len, u_size, name='att'):
    """
    Args:
        in_x:  3-D tensor [batch, max_time, h_size]
        in_len: 1-D tensor [batch]
        u_size: int
        name: string

    Returns: 2-D tensor [batch, max_time]

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


def cosine_distance(x1, x2):
    """
    :param x1: 2D tensor, [batch, dim]
    :param x2: 2D tensor, [batch, dim]
    :return: 1D tensor,  [batch]
    """
    # cosine = x*y / (|x||y|)
    # |x| = sqrt(x1^2+x2^2+...+xn^2)
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    d = tf.reduce_sum(tf.multiply(x1, x2), axis=1) / tf.multiply(x1_norm, x2_norm)
    return d


def l1_distance(x1, x2):
    """
    :param x1: batch x dim
    :param x2: batch x dim
    :return: batch x 1
    """
    return tf.reduce_sum(x1 - x2, axis=1)


def l2_distance(x1, x2):
    """
    :param x1: batch x dim
    :param x2: batch x dim
    :return: batch x 1
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=1))


def bilinear_score(inputs1, inputs2):
    """ A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations
    Args:
        inputs1: 2D tensor, [B, D]
        inputs2: 2D tensor, [B, D]
    Returns: [B]

    """
    h_size = inputs1.get_shape().as_list()[1]
    w_bilinear = tf.get_variable(
        "w_bilinear", shape=[h_size, h_size], dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01))
    # b_bilinear = tf.get_variable(
    #     "b_bilinear", shape=[h_size], dtype=tf.float32,
    #     initializer=tf.zeros_initializer)
    x1 = tf.matmul(inputs1, w_bilinear)  # [B, D] x [D, D] = [B, D]
    x1 = tf.expand_dims(x1, axis=1)  # [B, 1, D]
    x2 = tf.expand_dims(inputs2, axis=2)
    rst = tf.matmul(x1, x2)  # [B, 1, D] x [B, D, 1] = [B, 1]
    return tf.squeeze(rst)


def centroids_loss(inputs, y, centroids):
    """B: batch , D: dim, C: class_num
    Args:
        inputs: 2D tensor, [B, D]
        y: 1D tensor, [B]
        centroids: 2D tensor [C x D]

    Returns:
    """
    diff_to_center = inputs - tf.gather(centroids, y)
    d_l2 = tf.sqrt(tf.reduce_sum(tf.square(diff_to_center), axis=1))
    return tf.reduce_mean(d_l2)


def centroids_logits(inputs, centroids):
    """B: batch , D: dim, C: class_num
    Args:
        inputs: 2D tensor, [B, D]
        centroids: 2D tensor, [C x D]

    Returns:
    """
    inputs = tf.expand_dims(inputs, axis=1)  # [B, 1, D]
    c = tf.expand_dims(centroids, axis=0)   # [1, C, D]
    distance = inputs - c   # [B, C, D]
    logits = tf.sqrt(tf.reduce_sum(tf.square(distance), axis=2))  # [B, C]
    return logits


def centroids_update(inputs, y, centroids):
    """
    Args:
        inputs: 2D tensor, [B, D]
        y: 1D tensor, [B]
        centroids: 2D tensor, [C, D]
    Returns:

    """
    centers_diff = []
    c_n = centroids.get_shape().as_list()[0]
    cs = tf.unstack(centroids, num=c_n, axis=0)
    for c_id, c in enumerate(cs):
        idx = tf.where(tf.equal(y, c_id))
        c_point = tf.gather(inputs, idx)
        new_c = tf.transpose(tf.reduce_mean(c_point, axis=0))
        result = tf.cond(tf.cast(idx.get_shape()[0].value is None, tf.bool),
                         lambda: tf.zeros_like(c, dtype=tf.float32),
                         lambda: new_c - c)
        centers_diff.append(result)
    return centers_diff

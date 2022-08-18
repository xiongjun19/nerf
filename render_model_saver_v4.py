# coding=utf8

import os
import argparse
import numpy as np
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_path', type=str, help='the path of the weights')
    parser.add_argument('-d', '--dim', type=int, help='the batch size of the model')
    parser.add_argument('-o', '--output', type=str, help='the output path of the graph')
    args = parser.parse_args()
    return args


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w, b, name=None):
    if name is None:
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID') + b
    return tf.add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID'),
                  b,
                  name=name)


def save_alpha_model(model_path, dim, pt_dim):
    """ build graph """
    pts_emb = tf.placeholder(tf.float32, shape=[None, dim, dim, pt_dim])
    hid_dim = 256
    # first linear [63, 256]
    w1 = weight_variable([1, 1, pt_dim, hid_dim])
    b1 = weight_variable([hid_dim])
    h = conv2d(pts_emb, w1, b1)

    lin_arr = []
    layer_nums = 7
    concat_idx = 4
    for i in range(layer_nums):
        _shp = [1, 1, hid_dim, hid_dim]
        if i == concat_idx:
            # _shp = [319, 256]
            _shp = [1, 1, pt_dim + hid_dim, hid_dim]
            h = tf.concat([pts_emb, h], -1)
        tmp_w = weight_variable(_shp)
        tmp_b = weight_variable([hid_dim])
        lin_arr.append(tmp_w)
        lin_arr.append(tmp_b)
        h = tf.nn.relu(conv2d(h, tmp_w, tmp_b))

    # get alpha_out
    alpha_w = weight_variable([1, 1, hid_dim, 64])
    alpha_b = weight_variable([64])
    alpha_out = conv2d(h, alpha_w, alpha_b, name='alpha')
    """ step2 parse and set weight """
    init_op = tf.global_variables_initializer()
    # weight_path = args.ckp_path
    # weight_arr = np.load(weight_path, allow_pickle=True)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        """ step3 save the graph """
        tf.saved_model.simple_save(sess, model_path,
                inputs={'pts': pts_emb},
                outputs={'alpha': alpha_out, 'hid': h})


def save_rgb_model(model_path, dim, view_dim):
    hid_dim = 256
    dir_emb = tf.placeholder(tf.float32, shape=[None, dim, dim, view_dim])
    h = tf.placeholder(tf.float32, shape=[None, dim, dim, hid_dim])
    feat_w = weight_variable([1, 1, hid_dim, hid_dim])
    feat_b = weight_variable([hid_dim])
    feat_out = conv2d(h, feat_w, feat_b) # [b, dim, hid_dim]
    input_viewdirs = tf.concat([feat_out, dir_emb], -1) # [b, dim, hid_dim + 28]
    view_w = weight_variable([1, 1, hid_dim+view_dim, 128])
    view_b = weight_variable([128])
    view_out = tf.nn.relu(conv2d(input_viewdirs, view_w, view_b))
    rgb_w = weight_variable([1, 1, 128, 64])
    rgb_b = weight_variable([64])
    rgb_out = conv2d(view_out, rgb_w, rgb_b, name='rgb')
    """ step2 parse and set weight """
    init_op = tf.global_variables_initializer()
    # weight_path = args.ckp_path
    # weight_arr = np.load(weight_path, allow_pickle=True)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        """ step3 save the graph """
        tf.saved_model.simple_save(sess, model_path,
                inputs={'viewdir': dir_emb, 'hid': h},
                outputs={'rgb': rgb_out})


def main(args):
    """ step1 build graph """
    dim = args.dim
    pt_dim = 64 #  实际上应该是63
    dir_dim = 64  # 实际上应该是27
    export_dir = args.output
    alpha_path = os.path.join(export_dir, 'alpha_model')
    save_alpha_model(alpha_path, dim, pt_dim)
    rgb_path = os.path.join(export_dir, 'rgb_model') 
    save_rgb_model(rgb_path, dim, dir_dim)


if __name__ == '__main__':
    t_args = get_args()
    main(t_args)

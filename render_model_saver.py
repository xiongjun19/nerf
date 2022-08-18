# coding=utf8

import argparse
import numpy as np
import tensorflow as tf


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


def main(args):
    """ step1 build graph """
    dim = args.dim
    pts_emb = tf.placeholder(tf.float32, shape=[None, 1, dim, 63])
    dir_emb = tf.placeholder(tf.float32, shape=[None, 1, dim, 27])
    # first linear [63, 256]
    w1 = weight_variable([63, 256])
    b1 = weight_variable([256])
    test_out = tf.add(tf.matmul(pts_emb, w1),  b1, name='test_out')
    # h = tf.matmul(pts_emb, w1) + b1

    # lin_arr = []
    # layer_nums = 7
    # concat_idx = 4
    # for i in range(layer_nums):
    #     _shp = [256, 256]
    #     if i == concat_idx:
    #         _shp = [319, 256]
    #         h = tf.concat([pts_emb, h], -1)
    #     tmp_w = weight_variable(_shp)
    #     tmp_b = weight_variable([256])
    #     lin_arr.append(tmp_w)
    #     lin_arr.append(tmp_b)
    #     h = tf.nn.relu(tf.matmul(h, tmp_w) + tmp_b)

    # feat_w = weight_variable([256, 256])
    # feat_b = weight_variable([256])
    # feat_out = tf.matmul(h, feat_w) + feat_b # [b, dim, 256]
    # input_viewdirs = tf.concat([feat_out, dir_emb], -1) # [b, dim, 256 + 28]
    # view_w = weight_variable([283, 128])
    # view_b = weight_variable([128])
    # view_out = tf.nn.relu(tf.matmul(input_viewdirs, view_w) + view_b)
    # rgb_w = weight_variable([128, 64])
    # rgb_b = weight_variable([64])
    # rgb_out = tf.add(matmul(view_out, rgb_w), rgb_b, name='rgb')

    # # get alpha_out
    # alpha_w = weight_variable([256, 64])
    # alpha_b = weight_variable([64])
    # alpha_out = tf.add(tf.matmul(h, alpha_w), alpha_b, name='alpha')

    """ step2 parse and set weight """
    init_op = tf.global_variables_initializer()
    weight_path = args.ckp_path
    weight_arr = np.load(weight_path, allow_pickle=True)
    with tf.Session() as sess:
        sess.run(init_op)
        """ step3 save the graph """
        export_dir = args.output
        tf.saved_model.simple_save(sess, export_dir,
                # inputs={'pts': pts_emb, 'viewdirs': dir_emb},
                inputs={'pts': pts_emb},
                # outputs={'alpha': alpha_out, 'rgb': rgb_out})
                outputs={'test_out': test_out})


if __name__ == '__main__':
    t_args = get_args()
    main(t_args)

# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio

import load_data
import model as model

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('validation_num', 3, """...""")

train_dir = './data/train/'
test_dir = './data/test/'
logs_train_dir = './tmp/logs'+ str(FLAGS.validation_num) +'/train/'
logs_val_dir = './tmp/logs'+ str(FLAGS.validation_num) +'/val/'
model_dir = './tmp/model'+ str(FLAGS.validation_num) +'/'


def eval():
    with tf.Graph().as_default():
        data, sequence_length, label = load_data.load_train_data()
        data = np.reshape(data, [-1, 107, 20])
        sequence_length = np.reshape(sequence_length, [-1])
        label = np.reshape(label, [-1])

        train_kfold_indices, valid_kfold_indices = load_data.K_Fold_Split(data, label)
        valid_k_indices = valid_kfold_indices[FLAGS.validation_num]
        valid_data = data[valid_k_indices]
        valid_sequence_length = sequence_length[valid_k_indices]
        valid_label = label[valid_k_indices]


        test_data, test_sequence_length, test_label = load_data.load_test_data()
        test_data = np.reshape(test_data, [-1, 107, 20])
        test_sequence_length = np.reshape(test_sequence_length, [-1])
        test_label = np.reshape(test_label, [-1])

        data_feed = tf.placeholder(dtype=tf.float32, shape=[None, 107, 20])
        label_feed = tf.placeholder(dtype=tf.int32, shape=[None])
        sequence_length_feed = tf.placeholder(dtype=tf.int32, shape=[None])
        keep_prob = tf.placeholder(tf.float32)

        logits = model.inference(data_feed, sequence_length_feed, keep_prob)
        probability = tf.nn.softmax(logits=logits)
        loss = model.loss(logits, label_feed)
        train_op = model.training(loss)
        accuracy_op, TP_op, TN_op = model.evaluation(logits, label_feed)
        global_step_op = tf.train.get_or_create_global_step()

        fine_tuned_matrix = tf.global_variables(scope='conv_pssm')[0]

        saver = tf.train.Saver()

        session_config = tf.ConfigProto()
        session_config.log_device_placement = False
        session_config.allow_soft_placement = True
        session_config.gpu_options.allow_growth = True
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        with tf.Session(config=session_config) as sess:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())

            acc_valid, probability_valid, TP_valid, TN_valid = sess.run([accuracy_op, probability, TP_op, TN_op],
                                        feed_dict={data_feed: valid_data, sequence_length_feed: valid_sequence_length,
                                                   label_feed: valid_label, keep_prob: 1.0})
            P = np.sum(valid_label)
            N = len(valid_label) - P


            acc_test, probability_test, TP_test, TN_test = sess.run([accuracy_op, probability, TP_op, TN_op],
                                        feed_dict={data_feed: test_data, sequence_length_feed: test_sequence_length,
                                                   label_feed: test_label, keep_prob: 1.0})
            prediction_probability_test = probability_test[:,1]
            sio.savemat('pssm_real.mat', {'pred': prediction_probability_test, 'label':test_label})

            # print('eval accuracy = %.2f%%' % (acc * 100.0))

            # M = sess.run(fine_tuned_matrix)
            # M = np.reshape(M, [20, 20])
            # sio.savemat('./pssm_result_' + str(FLAGS.validation_num) + '.mat', {'pssm': M})

            # f = open('./tmp/result_eval.txt', 'a+')
            # f.write(str(FLAGS.validation_num) + ',')
            # f.write(str(acc_valid) + ',')
            # f.write(str(TP_valid) + ',')
            # f.write(str(TN_valid) + ',')
            # f.write(str(P) + ',')
            # f.write(str(N) + ',')
            # f.write(str(acc_test) + ',')
            # f.write(str(TP_test) + ',')
            # f.write(str(TN_test) + ',')
            # f.write('\n')

            # f.close()

if __name__ == '__main__':
    eval()

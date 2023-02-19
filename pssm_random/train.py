# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import tensorflow as tf

import load_data
import model as model

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('validation_num', 0, """...""")

train_dir = './data/train/'
test_dir = './data/test/'
logs_train_dir = './tmp/logs'+ str(FLAGS.validation_num) +'/train/'
logs_val_dir = './tmp/logs'+ str(FLAGS.validation_num) +'/val/'
model_dir = './tmp/model'+ str(FLAGS.validation_num) +'/'


def train():
    with tf.Graph().as_default():
        data, sequence_length, label = load_data.load_train_data()
        data = np.reshape(data, [-1, 107, 20])
        sequence_length = np.reshape(sequence_length, [-1])
        label = np.reshape(label, [-1])

        train_kfold_indices, valid_kfold_indices = load_data.K_Fold_Split(data, label)
        train_k_indices = train_kfold_indices[FLAGS.validation_num]
        valid_k_indices = valid_kfold_indices[FLAGS.validation_num]
        train_data = data[train_k_indices]
        train_sequence_length = sequence_length[train_k_indices]
        train_label = label[train_k_indices]
        valid_data = data[valid_k_indices]
        valid_sequence_length = sequence_length[valid_k_indices]
        valid_label = label[valid_k_indices]


        data_feed = tf.placeholder(dtype=tf.float32, shape=[None, 107, 20])
        label_feed = tf.placeholder(dtype=tf.int32, shape=[None])
        sequence_length_feed = tf.placeholder(dtype=tf.int32, shape=[None])
        keep_prob = tf.placeholder(tf.float32)

        logits = model.inference(data_feed, sequence_length_feed, keep_prob)
        loss = model.loss(logits, label_feed)
        train_op = model.training(loss)
        accuracy_op, _, _ = model.evaluation(logits, label_feed)
        global_step_op = tf.train.get_or_create_global_step()

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, tf.get_default_graph())
        val_writer = tf.summary.FileWriter(logs_val_dir, tf.get_default_graph())

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


            temp = 100000
            temp_acc = 0
            for step in np.arange(320):
                _, tra_loss, tra_acc, global_step, summary_str = sess.run([train_op, loss, accuracy_op, global_step_op, summary_op], feed_dict={data_feed: train_data, sequence_length_feed: train_sequence_length, label_feed: train_label, keep_prob: 0.8})
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (global_step, tra_loss, tra_acc * 100.0))
                train_writer.add_summary(summary_str, global_step)

                val_loss, val_acc, summary_str = sess.run([loss, accuracy_op, summary_op], feed_dict={data_feed: valid_data, sequence_length_feed: valid_sequence_length, label_feed: valid_label, keep_prob: 1.0})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (global_step, val_loss, val_acc * 100.0))
                val_writer.add_summary(summary_str, global_step)

                if val_acc > temp_acc:
                    temp_acc = val_acc

                    checkpoint_path = os.path.join(model_dir, str(temp_acc)+'_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            print(temp_acc)
            f = open('./tmp/result_cross_validation.txt', 'a+')
            f.write(str(FLAGS.validation_num)+' ')
            f.write(str(temp_acc)+'\n')
            f.close()

if __name__ == '__main__':
    train()

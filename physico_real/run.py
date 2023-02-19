# -*- coding: utf-8 -*-
import subprocess
import os
import sys
import time

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('work_type', "all", "The work type.")

def train_cross_validation():
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    f = open('./tmp/result_cross_validation.txt', 'w')
    f.write(str(time.time()) + '\n')
    f.close()
    for k in range(5):
        print("Cross Validation " + str(k) + " start...")

        try:
            output = subprocess.call(['python', 'train.py', '--validation_num', str(k)])
        except:
            print("Cross Validation " + str(k) + " false...")
            continue

        print("Cross Validation " + str(k) + " finish...")

    print('finish')

def test_cross_validation():
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    f = open('./tmp/result_eval.txt', 'w')
    f.write(str(time.time()) + '\n')
    f.close()
    for k in range(5):
        print("Cross Validation " + str(k) + " start...")

        try:
            output = subprocess.call(['python', 'eval.py', '--validation_num', str(k)])
        except:
            print("Cross Validation " + str(k) + " false...")
            continue

        print("Cross Validation " + str(k) + " finish...")

    print('finish')


if __name__ == "__main__":
    if FLAGS.work_type == 'all':
        train_cross_validation()
        test_cross_validation()
    if FLAGS.work_type == 'eval':
        test_cross_validation()
    if FLAGS.work_type == 'train':
        train_cross_validation()




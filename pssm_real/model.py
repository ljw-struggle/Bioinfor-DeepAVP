# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io as sio

def inference(data, sequence_length, keep_prob):
    """
    The inference function.
    :param data: [batch_size, 107, 20]
    :param sequence_length: [batch_size]
    :param keep_prob: the parameter for dropout layer.
    :return: the logits.
    """
    tf.set_random_seed(0)
    batch_size_op = tf.shape(data)[0]

    with tf.variable_scope('lstm_variable_sequence') as scope:
        # For the LSTM weight and biases initialization.
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=64, initializer=tf.glorot_normal_initializer(seed=0))
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob, seed=0)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=64, initializer=tf.glorot_normal_initializer(seed=0))
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob, seed=0)
        init_fw = cell_fw.zero_state(batch_size_op, dtype=tf.float32)
        init_bw = cell_bw.zero_state(batch_size_op, dtype=tf.float32)

        bidrnn_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=data, sequence_length = sequence_length, initial_state_fw=init_fw, initial_state_bw=init_bw)

        fw_lstm_outputs = final_states[0][1]
        bw_lstm_outputs = final_states[1][1]
        lstm_outputs = tf.concat((fw_lstm_outputs, bw_lstm_outputs), axis=1) # shape = [batch_size, 128]

    with tf.variable_scope('conv_pssm') as scope:
        matrix = sio.loadmat('./data/pssm.mat')['pssm']
        initializer_filters = tf.reshape(tf.constant(matrix, dtype=tf.float32), [1, 20, 1, 20])
        initializer_biases = tf.constant_initializer(0)
        filters = tf.get_variable('filters', initializer=initializer_filters, dtype=tf.float32)
        biases = tf.get_variable('biases', [20], initializer=initializer_biases, dtype=tf.float32, trainable=False)

        input = tf.reshape(data, [batch_size_op, 107, 20, 1])
        temp = tf.nn.conv2d(input, filters, strides=[1, 1, 20, 1], padding='SAME')
        temp_b = tf.nn.bias_add(temp, biases)
        conv_pssm = temp_b  # shape= [batch_size, 107, 1, 20]

        bandwidth = tf.floor(tf.divide(sequence_length, 4))
        width = tf.cast(tf.multiply(bandwidth, 4), tf.int32)

        Tensor_array = tf.TensorArray(tf.float32, batch_size_op)

        def cond(i, array):
            return i < batch_size_op

        def body(i, array):
            avblock_temp = tf.reshape(conv_pssm[i][0:width[i]], [4, -1, 20])
            avblock = tf.reshape(tf.reduce_mean(avblock_temp, axis=1), [4, 20])
            array = array.write(i, avblock)
            return i + 1, array

        i, array = tf.while_loop(cond, body, (0, Tensor_array))
        outputs = array.stack()


    with tf.variable_scope('conv_feature_extraction') as scope:
        initializer_filters = tf.truncated_normal_initializer(stddev=0.4, seed=0)
        initializer_biases = tf.constant_initializer(0)
        filters = tf.get_variable('filters', [4, 4, 1, 20], initializer=initializer_filters, dtype=tf.float32)
        biases = tf.get_variable('biases', [20], initializer=initializer_biases, dtype=tf.float32)

        input = tf.reshape(outputs, [batch_size_op, 4, 20, 1])
        temp = tf.nn.conv2d(input, filters, strides=[1, 4, 4, 1], padding='SAME')
        temp_b = tf.nn.bias_add(temp, biases)
        conv_feature_extraction = tf.nn.relu(temp_b)  # shape= [batch_size, 1, 5, 20]

    with tf.variable_scope('dropout') as scope:
        dropout = tf.nn.dropout(conv_feature_extraction, keep_prob=keep_prob, seed=0)


    with tf.variable_scope('Merge_features') as scope:
        conv = tf.reshape(dropout, [batch_size_op, 100])
        merge_features = tf.concat([lstm_outputs, conv], axis=1) # shape = [batch_size, 228]

    with tf.variable_scope('fully_connected_1') as scope:
        initializer_weights = tf.truncated_normal_initializer(stddev=0.4, seed=0)
        initializer_biases = tf.constant_initializer(0.1)
        weights = tf.get_variable('weight', [228, 100], initializer=initializer_weights, dtype=tf.float32)
        biases = tf.get_variable('biases', [100], initializer=initializer_biases, dtype=tf.float32)
        f1_l2_loss = tf.multiply(tf.nn.l2_loss(weights), 0.2, name='f1_weight_loss')
        tf.add_to_collection('losses', f1_l2_loss)
        temp = tf.nn.xw_plus_b(merge_features, weights, biases)
        fc1 = tf.nn.relu(temp)

    with tf.variable_scope('fully_connected_2') as scope:
        initializer_weights = tf.truncated_normal_initializer(stddev=0.4, seed=0)
        initializer_biases = tf.constant_initializer(0.1)
        weights = tf.get_variable('weight', [100, 2], initializer=initializer_weights, dtype=tf.float32)
        biases = tf.get_variable('biases', [2], initializer=initializer_biases, dtype=tf.float32)
        f2_l2_loss = tf.multiply(tf.nn.l2_loss(weights), 0.2, name='f2_weight_loss')
        tf.add_to_collection('losses', f2_l2_loss)
        logits = tf.nn.xw_plus_b(fc1, weights, biases)

    return logits


def loss(logits, labels):
    """
    The loss function.
    :param logits: the logits.
    :param labels: the labels.
    :return: return loss_op
    """
    with tf.variable_scope('loss') as scope:
        loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits=logits)
        # l2_loss = tf.add_n(tf.get_collection('losses'))
        # total_loss = tf.add_n([loss_cross_entropy, l2_loss])
        tf.summary.scalar(scope.name + '/loss', loss_op)
        return loss_op


def training(loss):
    """
    The training function.
    :param loss: the loss_op.
    :return: the train_op
    """
    with tf.variable_scope('training') as scope:
        global_step = tf.train.get_or_create_global_step()
        Optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = Optimizer.minimize(loss, global_step=global_step)
        return train_op


def evaluation(logits, labels):
    """
    The evaluation function.
    :param logits: the logits. shape = [batch_size, 2]
    :param labels: the labels. shape = [batch_size]
    :return: the evaluation op.
    """
    with tf.variable_scope('evaluation') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy_op = tf.reduce_mean(correct)
        correct_num = tf.cast(tf.reduce_sum(correct), tf.int32)
        TP = tf.reduce_sum(tf.multiply(tf.cast(correct, tf.int32), labels))
        TN = correct_num - TP
        tf.summary.scalar('accuracy', accuracy_op)
        return accuracy_op, TP, TN


def prediction(logits):
    """
    The prediction function.
    :param logits: the logits.
    :return: the prediction op.
    """
    with tf.variable_scope('prediction') as scope:
        predicted_classes = tf.argmax(logits, 1)
        predictions = {
            "class_id": predicted_classes,
            "probabilities": tf.nn.softmax(logits)
        }
        return predictions
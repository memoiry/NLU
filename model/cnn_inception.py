#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCNN_Inception(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.expand_dims(tf.nn.embedding_lookup(embedding, self.input_x),-1)

        return _inputs

    def cnn(self):
        """cnn模型"""
        embedding_inputs = self.input_embedding()

        #with tf.name_scope("cnn"):
        #    # cnn 与全局最大池化
        #    conv = tf.layers.conv1d(embedding_inputs,
        #        self.config.num_filters,
        #        self.config.kernel_size, name='conv')#

            # global max pooling
        #    gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                conv = tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout_keep_prob)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            #fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            #fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            #fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(self.h_drop, self.config.num_classes,
                name='fc2')
            self.pred_y = tf.nn.softmax(self.logits)

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimize"):
            # 优化器
            optimizer = tf.train.RMSPropOptimizer(
                self.config.learning_rate, 0.9)
            self.optim = optimizer.minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                tf.argmax(self.pred_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

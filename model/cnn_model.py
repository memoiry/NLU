#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tf.add_to_collection("input_x", self.input_x)
        tf.add_to_collection("input_y", self.input_y)
        tf.add_to_collection("keep_prob", self.keep_prob)
        self.cnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            
        return _inputs

    def cnn(self):
        """cnn模型"""
        embedding_inputs = self.input_embedding()

        with tf.name_scope("cnn"):
            # cnn 与全局最大池化
            conv = tf.layers.conv1d(embedding_inputs,
                self.config.num_filters,
                self.config.kernel_size, name='conv')

            # global max pooling
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                name='fc2')
            self.pred_y = tf.nn.softmax(self.logits)
            tf.add_to_collection("activation", self.pred_y)

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

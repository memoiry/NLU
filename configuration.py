#!/usr/bin/python
# -*- coding: utf-8 -*-

class TCNNConfig(object):
    """CNN配置参数"""

    # 模型参数
    embedding_dim = 256      # 词向量维度
    seq_length = 86        # 序列长度
    num_classes = 135        # 类别数
    num_filters = 128       # 卷积核数目
    kernel_size = 3         # 卷积核尺寸
    vocab_size = 2791       # 词汇表达小

    hidden_dim = 256        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 12          # 总迭代轮次

    print_per_batch = 200    # 每多少轮输出一次结果


class TCNN_Inception_Config(object):
    """CNN配置参数"""

    # 模型参数
    embedding_dim = 256      # 词向量维度
    seq_length = 33        # 序列长度
    num_classes = 32        # 类别数
    num_filters = 128       # 卷积核数目
    vocab_size = 2791       # 词汇表达小
    filter_sizes = [3,4,5]
    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 16          # 总迭代轮次

    print_per_batch = 200    # 每多少轮输出一次结果


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 128      # 词向量维度
    seq_length = 86        # 序列长度
    num_classes = 135        # 类别数
    vocab_size = 2791       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 256        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 12          # 总迭代轮次

    print_per_batch = 200    # 每多少轮输出一次结果

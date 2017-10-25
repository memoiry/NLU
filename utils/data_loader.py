#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os
import codecs
import random

def _read_file(filename):
    """读取文件数据"""
    contents = []
    labels = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f.readlines():
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab(filename, vocab_size=2791):
    """根据训练集构建词汇表，存储"""
    data, labels = _read_file(filename)

    all_data = []
    for content in data:
        all_data.extend(content)

    labels = set(labels)

    file_dir = filename.split('/')[:-1]
    file_dir = ''.join(file_dir)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    print('{} words'.format(len(words)))
    print('{} categories'.format(len(labels)))

    #codecs.open('data/cnews/vocab_cnews.txt', 'w',
    #    'utf-8').write('\n'.join(words))
    codecs.open('{}/vocab.txt'.format(file_dir), 'w',
        'utf-8').write('\n'.join(words))
    codecs.open('{}/categories.txt'.format(file_dir), 'w',
        'utf-8').write('\t'.join(labels))

def _read_vocab(filename):
    """读取词汇表"""
    words = list(map(lambda line: line.strip(),
       codecs.open(filename, 'r', 'utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category(filename):
    """读取分类目录，固定"""
    categories = codecs.open(filename, 'r', 'utf-8').readline().split('\t')
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def get_id_to_cat(filename):

    categories = codecs.open(filename, 'r', 'utf-8').readline().split('\t')
    id_to_cat = dict(zip(range(len(categories)), categories))

    return id_to_cat

def _file_to_ids(filename, word_to_id, max_length=600):
    """将文件转换为id表示"""
    categories_name = filename.split('/')[:-1]
    categories_name = ''.join(categories_name) + '/categories.txt'
    _, cat_to_id = read_category(categories_name)
    contents, labels = _read_file(filename)

    data_id = []
    label_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    max_length_ = 0
    for ele in data_id:
        if len(ele) > max_length_:
            max_length_ = len(ele)
            #print(ele)
    print('sequence max length: {}'.format(max_length_))
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def _text_to_ids(text, word_to_id, max_length=600):
    data_id = [[word_to_id[x] for x in text if x in word_to_id]]
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad

def process_text(text, vocab_file, seq_length=86):
    words, word_to_id = _read_vocab(vocab_file)
    x = _text_to_ids(text, word_to_id, seq_length)
    return x


def process_file(train_file, val_file, test_file, vocab_file, seq_length=86):
    """一次性返回所有数据"""
    words, word_to_id = _read_vocab(vocab_file)
    x_train, y_train = _file_to_ids(train_file, word_to_id, seq_length)
    x_test, y_test = _file_to_ids(test_file, word_to_id, seq_length)
    x_val, y_val = _file_to_ids(val_file, word_to_id, seq_length)

    return x_train, y_train, x_test, y_test, x_val, y_val, words
    #return x_train, y_train, words

def batch_iter(data, batch_size=64, num_epochs=5):
    """生成批次数据"""
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]

        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def build_dataset(input_folder, out_name):
    print("Start generate training and testing samples")
    input_dirs = os.listdir(input_folder)
    for dir_ in input_dirs:
        if len(dir_.split('.')) >= 2:
            input_dirs.remove(dir_)
    input_files = []
    for dir_ in input_dirs:
        files = os.listdir(os.path.join(input_folder, dir_))
        #print(files)
        #print(len(files))
        files = list(map(lambda x: os.path.join(input_folder, dir_, x), files))
        for file in files:
            if len(file.split('.')) == 1:
                temp_files = os.listdir(file)
                temp_files = list(map(lambda x: os.path.join(file, x), temp_files))
                files.remove(file)
                files.extend(temp_files)
        #print(dir_)
        #print(files)
        #print('\n')
        input_files.extend(files)
    try:
        for file in input_files:
            if file.split('.')[-1] == 'DS_Store':
                input_files.remove(file)
    except:
        pass
    train_data = codecs.open('{}.train'.format(os.path.join(input_folder, out_name)), 'w', 'utf-8')
    test_data = codecs.open('{}.test'.format(os.path.join(input_folder, out_name)), 'w', 'utf-8')
    val_data = codecs.open('{}.val'.format(os.path.join(input_folder, out_name)), 'w', 'utf-8')
    for input_file in input_files:
        #input_file = os.path.join(input_folder, input_file)
        input_data = codecs.open(input_file, 'r', 'utf-8')
        print(input_file)
        count = 0
        num_line = sum(1 for line in input_data)
        #print(num_line)
        input_data = codecs.open(input_file, 'r', 'utf-8')
        lines = input_data.readlines()
        random.shuffle(lines)
        #print(lines)
        for line in lines:
            count = count + 1
            train_num = round(num_line * 0.85)
            test_num = train_num + round(num_line * 0.10)
            val_num = num_line
            #print("{}, {}, {}".format(train_num, test_num, val_num))
            if count <= train_num:
                word_list = line.strip().split()
                #cutted_text = thu_model.cut(word_list[0], text = True)
                #cutted_text = ' '.join(jieba.cut(word_list[0]))
                label_index = 1
                for i in range(1, len(word_list)):
                    if '_' in list(word_list[i]):
                        label_index = i
                        break

                for i in range(label_index):
                    train_data.write(word_list[label_index] + '\t' + word_list[i])
                    #print(cutted_text + ' ' + '__label__' + word_list[1])
                    train_data.write("\n")
            elif count <= test_num:
                word_list = line.strip().split()
                #cutted_text = thu_model.cut(word_list[0], text = True)
                #cutted_text = ' '.join(jieba.cut(word_list[0]))
                label_index = 1
                for i in range(1, len(word_list)):
                    if '_' in list(word_list[i]):
                        label_index = i
                        break
                for i in range(label_index):
                    test_data.write(word_list[label_index] + '\t' + word_list[i])
                    #print(cutted_text + ' ' + '__label__' + word_list[1])
                    test_data.write("\n")
            else:
                word_list = line.strip().split()
                #cutted_text = thu_model.cut(word_list[0], text = True)
                #cutted_text = ' '.join(jieba.cut(word_list[0]))
                label_index = 1
                for i in range(1, len(word_list)):
                    if '_' in list(word_list[i]):
                        label_index = i
                        break

                for i in range(label_index):
                    val_data.write(word_list[label_index] + '\t' + word_list[i])
                    #print(cutted_text + ' ' + '__label__' + word_list[1])
                    val_data.write("\n") 

        input_data.close()
    train_data.close()
    test_data.close()
    val_data.close()


if __name__ == '__main__':
    #if not os.path.exists('raw/vocab.txt'):
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    build_dataset(input_file, output_file)
    build_vocab('raw/nlu.train')

    train_file = 'raw/nlu.train'
    val_file = 'raw/nlu.val'
    test_file = 'raw/nlu.test'
    vocab_file = 'raw/vocab.txt'
    x_train, y_train, x_test, y_test, x_val, y_val,words = preocess_file(train_file, val_file, test_file, vocab_file)
    #x_train, y_train, words= preocess_file()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_val.shape, y_val.shape)

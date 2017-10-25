# encoding=utf8
from __future__ import print_function

import os 
import codecs
import pickle
import itertools
from collections import OrderedDict
import fire
import tensorflow as tf
import numpy as np
from model.model import Model
from utils.loader import load_sentences, update_tag_scheme
from utils.loader import char_mapping, tag_mapping
from utils.loader import augment_with_pretrained, prepare_dataset
from utils.utils import get_logger, make_path, clean, create_model, save_model
from utils.utils import print_config, save_config, load_config, test_ner
from utils.utils import plot_confusion_matrix
from utils.data_utils import make_train_int, make_train_ner
from utils.data_utils import load_word2vec, create_input, input_from_line, BatchManager

from model.rnn_model import *
from model.cnn_model import *
from model.cnn_inception import *
from configuration import *
from utils.data_loader import *

import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

    
import time
from datetime import timedelta

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iob",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   25,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "nlu_ner.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "nlu_ner.val"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "nlu_ner.test"),   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


model_path = 'ckpt'
data_path = 'data'

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            self.input_x = tf.get_collection('input_x')[0]
            self.input_y = tf.get_collection('input_y')[0]
            self.keep_prob = tf.get_collection('keep_prob')[0] 
            self.activation = tf.get_collection('activation')[0]


    def run(self, input_x, keep_prob):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={self.input_x: input_x,
            self.keep_prob: keep_prob})

# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def prepare_dataset(data_type, data_folder, output_name):
    if data_type == 'intent':
        make_train_int(data_folder, output_name)
    elif data_type == 'ner':
        make_train_ner(data_folder, output_name)
    else:
        print("Unsupported data type, you can only choose either intent or ner")
#    make_train_int(data_folder, output_name)


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1



def train_int(model_type, train_file, val_file, test_file, vocab_file):
    # 载入数据
    print('Loading data...')
    start_time = time.time()

    x_train, y_train, x_test, y_test, x_val, y_val, words = process_file(train_file, val_file, test_file, vocab_file)
    #x_train, y_train, words = preocess_file()

    if model_type == 'cnn':
        print('Using CNN model...')
        config = TCNNConfig()
        config.vocab_size = len(words)
        model = TextCNN(config)
        tensorboard_dir = 'tensorboard/textcnn'
    elif model_type == 'rnn':
        print('Using RNN model...')
        config = TRNNConfig()
        config.vocab_size = len(words)
        model = TextRNN(config)
        tensorboard_dir = 'tensorboard/textrnn'
    elif model_type == 'inception_cnn':
        print('Using Inception_CNN model...')
        config = TCNN_Inception_Config()
        config.vocab_size = len(words)
        model = TextCNN_Inception(config)
        tensorboard_dir = 'tensorboard/textcnn'
    else:
        print('Unsupported model type, current option is rnn and cnn')

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)


    print('Constructing TensorFlow Graph...')
    with tf.Session() as session: 
        session.run(tf.global_variables_initializer())

        # 配置 tensorboard
        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("accuracy", model.acc)

        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(session.graph)

        # 生成批次数据
        print('Generating batch...')
        batch_train = batch_iter(list(zip(x_train, y_train)),
            config.batch_size, config.num_epochs)

        def feed_data(batch):
            """准备需要喂入模型的数据"""
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch
            }
            return feed_dict, len(x_batch)

        def evaluate(x_, y_):
            """
            模型评估
            一次运行所有的数据会OOM，所以需要分批和汇总
            """
            batch_eval = batch_iter(list(zip(x_, y_)), 128, 1)

            total_loss = 0.0
            total_acc = 0.0
            cnt = 0
            for batch in batch_eval:
                feed_dict, cur_batch_len = feed_data(batch)
                feed_dict[model.keep_prob] = 1.0
                loss, acc = session.run([model.loss, model.acc],
                    feed_dict=feed_dict)
                total_loss += loss * cur_batch_len
                total_acc += acc * cur_batch_len
                cnt += cur_batch_len

            return total_loss / cnt, total_acc / cnt

        # 训练与验证
        print('Training and evaluating...')
        start_time = time.time()
        print_per_batch = config.print_per_batch
        for i, batch in enumerate(batch_train):
            feed_dict, _ = feed_data(batch)
            feed_dict[model.keep_prob] = config.dropout_keep_prob

            if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)

            if i % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
                loss_train, acc_train = session.run([model.loss, model.acc],
                    feed_dict=feed_dict)
                loss, acc = evaluate(x_val, y_val)
                #loss, acc = evaluate(x_train, y_train)

                # 时间
                end_time = time.time()
                time_dif = end_time - start_time
                time_dif = timedelta(seconds=int(round(time_dif)))

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
                print(msg.format(i + 1, loss_train, acc_train, loss, acc, time_dif))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化

        # 最后在测试集上进行评估
        print('Evaluating on test set...')
        loss_test, acc_test = evaluate(x_test, y_test)


        #loss_test, acc_test = evaluate(x_train, y_train)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))

        # 保存模型
        saver = tf.train.Saver()
        model_path = "{}/model_{}".format(model_path, model_type)
        save_path = saver.save(session, model_path)
        print("Model saved in file: %s" % save_path)

def train_ner():
    clean(FLAGS)
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(25):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line_ner():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)


def evaluate_line_int():
    id_to_cat = get_id_to_cat('{}/categories.txt'.format(data_path))

    print("Loading the model....")
    model_1 = ImportGraph('{}/model_cnn'.format(model_path))
    model_2 = ImportGraph('{}/model_rnn'.format(model_path))
    print("Model loaded..")
    flag = 0
    while True:
        if flag == 0:
            flag = 1
            pass
        else:
            print("")
        text = input("请输入要进行意图识别的句子：\n")
        id_text = process_text(text, '{}/vocab.txt'.format(data_path))
        pred_1 = model_1.run(id_text, 1.0)
        pred_2 = model_2.run(id_text, 1.0)
        pred = pred_1 + pred_2
        res = id_to_cat[int(np.argmax(pred))]
        print(res)

def evaluate_line():
    # 加载意图识别模型

    id_to_cat = get_id_to_cat('{}/categories.txt'.format(data_path))

    print("==========================Loading the Intention Classification model....==========================")
    model_1 = ImportGraph('{}/model_cnn'.format(model_path))
    model_2 = ImportGraph('{}/model_rnn'.format(model_path))
    print("Model loaded..")
    flag = 0

    # 加载命名实体识别模型
    print("==========================Loading the NER model....==========================")
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger) 

        # 循环识别

        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

            # 获取测试句子
            text = input("请输入要进行识别的句子：")

            # intent 识别
            id_text = process_text(text, '{}/vocab.txt'.format(data_path))
            pred_1 = model_1.run(id_text, 1.0)
            pred_2 = model_2.run(id_text, 1.0)
            pred = pred_1 + pred_2
            res = id_to_cat[int(np.argmax(pred))]
            print(res)  

            # NER 识别
            result = model.evaluate_line(sess, input_from_line(text, char_to_id), id_to_tag)
            print(result)

def confusion_int():


    x_train, y_train, x_test, y_test, x_val, y_val, words = process_file('data/nlu_int.train', 'data/nlu_int.val', 
        'data/nlu_int.test', 'data/vocab.txt')
    id_to_cat = get_id_to_cat('{}/categories.txt'.format(data_path))
    categories = []
    for i in range(135):
        categories.append(id_to_cat[i])
    print("Loading the model....")
    model_1 = ImportGraph('{}/model_cnn'.format(model_path))
    model_2 = ImportGraph('{}/model_rnn'.format(model_path))
    print("Model loaded..")
    flag = 0
    labels = np.argmax(y_test, 1)
    pred_1 = model_1.run(x_test, 1.0)
    pred_2 = model_2.run(x_test, 1.0)
    pred_temp = pred_1 + pred_2
    pred = np.argmax(pred_temp, 1)


    tf.InteractiveSession()
    confusion_matrix = tf.contrib.metrics.confusion_matrix(labels, pred, dtype = 'float32').eval()
    plot_confusion_matrix(confusion_matrix, categories)
    class_num = np.sum(confusion_matrix,1)
    for i in range(135):
        confusion_matrix[i, :] = confusion_matrix[i, :] / np.sum(confusion_matrix[i, :])
    acc_ = confusion_matrix[range(135),range(135)]
    below_line = 0.90

    below_index = np.argwhere(acc_ < below_line).flatten()
    print("{} classes below the line".format(len(below_index)))
    below_acc = acc_[below_index]
    below_class_num = class_num[below_index]
    below_categories = [id_to_cat[index_] for index_ in below_index]
    stat_result = dict(zip(below_categories, zip(below_acc, below_class_num)))
    for key, item in stat_result.items():
        if item[1] > 50:
            print(key, item)


    new_labels = labels[np.where([ele in below_index for ele in labels])]
    new_pred = pred[np.where([ele in below_index for ele in labels])]
    total_set = set(new_labels).union(set(new_pred))
    cat_dict = dict(zip(range(135), categories))
    map_dict = dict(zip(total_set,range(len(total_set))))
    inver_dict = {}
    for key, item in map_dict.items():
        inver_dict[item] = key
    new_categories = []
    for i in range(len(inver_dict)):
        new_categories.append(cat_dict[inver_dict[i]])
    new_labels = [map_dict[label_] for label_ in new_labels]
    new_pred = [map_dict[pred_] for pred_ in new_pred]
    new_confusion_matrix = tf.contrib.metrics.confusion_matrix(new_labels, new_pred).eval()
    plot_confusion_matrix(new_confusion_matrix, new_categories)



def test_int():

    model_1 = ImportGraph('{}/model_cnn'.format(model_path))
    model_2 = ImportGraph('{}/model_rnn'.format(model_path))

    x_train, y_train, x_test, y_test, x_val, y_val, words = process_file('{}/nlu_int.train'.format(data_path),
    '{}/nlu_int.val'.format(data_path), '{}/nlu_int.test'.format(data_path), '{}/vocab.txt'.format(data_path))
    #text = input('请输入测试句子')
    #id_text = process_text(text, 'raw/vocab.txt')

    # Test cnn model
    pred_1 = model_1.run(x_test, 1.0)
    acc_1 = np.mean(np.equal(np.argmax(pred_1, 1), np.argmax(y_test, 1)).astype(float))
    print("CNN model accuracy: {} %".format(acc_1 * 100))

    # Test rnn model
    pred_2 = model_2.run(x_test, 1.0)
    acc_2 = np.mean(np.equal(np.argmax(pred_2, 1), np.argmax(y_test, 1)).astype(float))
    print("RNN model accuracy: {} %".format(acc_2 * 100))

    # Test multi model
    acc_3 = np.mean(np.equal(np.argmax(pred_1 + pred_2, 1), np.argmax(y_test, 1)).astype(float))
    print("Multi model accuracy: {} %".format(acc_3 * 100))

    #total_loss += loss * cur_batch_len
    #total_acc += acc * cur_batch_len
    #cnt += cur_batch_len
    #model_2 = ImportGraph('{}/model_rnn'.format(model_path))
    #evaluate

if __name__ == "__main__":
    #tf.app.run(main)
    fire.Fire()




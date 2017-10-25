# encoding = utf8
import re
import math
import codecs
import random
from collections import Counter
import sys
import os
import numpy as np
from tqdm import tqdm
import jieba
jieba.initialize()


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs

del_list = ['COMPOSER','FROM_PROVINCE', 'SOLAR_NAME', 'RANKING', 'IDIOM_AFFECTION',
'NEWS_ATTRIBUTE',
'POETRY_TYPE']

def write_line(file, line):
    word_list = line.strip().split('\t')
    #cutted_text = thu_model.cut(word_list[0], text = True)
    #cutted_text = ' '.join(jieba.cut(word_list[0]))
    label_index = 1
    for i in range(1, len(word_list)):
        if '_' in list(word_list[i]):
            label_index = i
            break
    if label_index == len(word_list):
        return
    entity_list = word_list[(label_index+1):]
    label_list = ['O'] * len(word_list[0]) 
    #print(entity_list)
    for entity in entity_list:
        label_name = entity.split(':')[0]
        if label_name in del_list:
            continue
        label_value = entity.split(':')[1]
        start_index = word_list[0].find(label_value)
        if start_index == -1:
            continue
        end_index = start_index + len(label_value)
        #print(word_list[0])
        #print(start_index)
        #print(end_index)
        #print(input_file)
        for cur_index in range(start_index, end_index):
            if cur_index == start_index:
                label_list[cur_index] = 'B-{}'.format(label_name)
            else:
                label_list[cur_index] = 'I-{}'.format(label_name)

    words = word_list[0]
    for cur_index in range(len(words)):
        file.write(words[cur_index] + '\t' + label_list[cur_index] + '\n')
        #print(cutted_text + ' ' + '__label__' + word_list[1])
    file.write("\n")


def make_train_ner(input_folder, output_file):
    print("Start segment the word and generate training and testing samples")
    input_dirs = os.listdir(input_folder)
    for dir_ in input_dirs:
        #if not os.path.isdir(dir_):
        #    input_dirs.remove(dir_)
        if len(dir_.split('.')) >= 2:
            input_dirs.remove(dir_)
    input_files = []
    print(input_dirs)
    if '.DS_Store' in input_dirs:
        input_dirs.remove('.DS_Store')
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
    train_data = codecs.open('{}/{}_ner.train'.format(input_folder, output_file), 'w', 'utf-8')
    test_data = codecs.open('{}/{}_ner.test'.format(input_folder, output_file), 'w', 'utf-8')
    val_data = codecs.open('{}/{}_ner.val'.format(input_folder, output_file), 'w', 'utf-8')
    all_lines = []
    total_num_lines = 0
    for input_file in input_files:
        #input_file = os.path.join(input_folder, input_file)
        input_data = codecs.open(input_file, 'r', 'utf-8')
        num_line = sum(1 for line in input_data)
        total_num_lines += num_line
        input_data = codecs.open(input_file, 'r', 'utf-8')
        lines = input_data.readlines()
        all_lines.extend(lines)
        input_data.close()
    unique_entity = []
    for line in all_lines:
        temp_entity = get_entity(line)
        unique_entity.extend(temp_entity)
    counter_entity = Counter(unique_entity).most_common()
    print(counter_entity)
    print(len(counter_entity))
    feasible_unique_entity = []
    for (entity, nums) in counter_entity:
        print((entity, nums))
        if nums >= 20:
            feasible_unique_entity.append((entity, nums))
    pbar = tqdm(total=len(feasible_unique_entity) * len(all_lines)) 
    for (entity_name, nums) in feasible_unique_entity:
        train_num = round(nums * 0.85)
        test_num = train_num + round(nums * 0.10)
        count = 0

        for line in all_lines:
            pbar.update(1)
            entity_list_cur = get_entity(line)
            if entity_name in entity_list_cur:
                if count < train_num:
                    write_line(train_data, line)
                elif count < test_num:
                    write_line(test_data, line)
                else:
                    write_line(val_data, line)
                count = count + 1
    pbar.close()
    print(len(feasible_unique_entity))
    train_data.close()
    test_data.close()
    val_data.close()

def get_entity(line):
    word_list = line.strip().split('\t')
    #cutted_text = thu_model.cut(word_list[0], text = True)
    #cutted_text = ' '.join(jieba.cut(word_list[0]))
    label_index = 1
    for i in range(1, len(word_list)):
        if '_' in list(word_list[i]):
            label_index = i
            break
    if label_index == len(word_list):
        return
    entity_list = word_list[(label_index+1):]
    return [entity.split(':')[0] for entity in entity_list]


def make_train_int(input_folder, output_file):
    print("Start segment the word and generate training and testing samples")
    input_dirs = os.listdir(input_folder)
    for dir_ in input_dirs:
        if len(dir_.split('.')) >= 2:
            #print(dir_)
            input_dirs.remove(dir_)
    input_files = []
    #print(input_dirs)
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
    train_data = codecs.open('{}/{}_int.train'.format(input_folder, output_file), 'w', 'utf-8')
    test_data = codecs.open('{}/{}_int.test'.format(input_folder, output_file), 'w', 'utf-8')
    val_data = codecs.open('{}/{}_int.val'.format(input_folder, output_file), 'w', 'utf-8')
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


class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

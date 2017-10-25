#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xuguodong
#
# 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
# __label__ for label prefix

import codecs
import sys
import os
import random
from tqdm import tqdm
from collections import Counter
#import jieba
#import thulac
import numpy as np

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



def character_tagging(input_folder, output_file):
    print("Start segment the word and generate training and testing samples")
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
    train_data = codecs.open('{}.train'.format(output_file), 'w', 'utf-8')
    test_data = codecs.open('{}.test'.format(output_file), 'w', 'utf-8')
    val_data = codecs.open('{}.val'.format(output_file), 'w', 'utf-8')
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


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("pls use: python make_train_data.py input output")
        sys.exit()
    #thu_model = thulac.thulac(seg_only = True)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    character_tagging(input_file, output_file)
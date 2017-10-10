#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xuguodong
#
# 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
# __label__ for label prefix

import codecs
import sys
import os
import jieba
import thulac
import numpy as np

def character_tagging(thu_model, input_folder, output_file, ratio = 0.7):
    print("Start segment the word and generate training and testing samples")
    input_files = os.listdir(input_folder)
    output_data = codecs.open(output_file, 'w', 'utf-8')
    count = 0
    for input_file in input_files:
        input_file = os.path.join(input_folder, input_file)
        input_data = codecs.open(input_file, 'r', 'utf-8')
        for line in input_data.readlines():
            count = count + 1
            word_list = line.strip().split()
            #cutted_text = thu_model.cut(word_list[0], text = True)
            cutted_text = ' '.join(jieba.cut(word_list[0]))
            output_data.write(cutted_text + ' ' + '__label__' + word_list[1])
            #print(cutted_text + ' ' + '__label__' + word_list[1])
            output_data.write("\n")
        input_data.close()
    output_data.close()
    train_num = int(np.round(count * ratio))
    test_num = int(np.round(count * (1 - ratio)))
    os.system('head -n {} {} > {}.train'.format(train_num, output_file, output_file))
    os.system('tail -n {} {} > {}.test'.format(test_num, output_file, output_file))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("pls use: python make_train_data.py input output")
        sys.exit()
    thu_model = thulac.thulac(seg_only = True)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    character_tagging(thu_model, input_file, output_file)
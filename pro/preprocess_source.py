#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:yanqiuxia
import json
import os
import re

from collections import Counter, OrderedDict
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

http_pattern = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
www_pattern = re.compile(r'www.[a-zA-Z0-9.?/&=:]*', re.S)
html_pattern = re.compile(r'<[^>]+>', re.S)


def cut_sent(para):
    para = re.sub(r'([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。！？\?][”’])([。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def remove_html_url(text):
    if text:
        text = html_pattern.sub("", text)
        text = http_pattern.sub("", text)
        text = www_pattern.sub("", text)

        text = re.sub('&nbsp', '', text)
        text = text.strip()
        # text = re.sub('\r\n', '', text)
        # text = re.sub('\n', '', text)

    return text


def load_govmsboxdata(file_in, all_reshapedata):
    fp = open(file_in, 'r', encoding='utf-8')
    all_data = json.load(fp)['RECORDS']
    if all_data:
        for data in all_data:
            id_ = data['id']
            title = data['title']
            content = data['content']
            value = {'title': title, 'content': content}
            if len(content) > 0:
                if all_reshapedata:
                    temp_value = all_reshapedata.get(id_)
                    if not temp_value:
                        all_reshapedata.update({id_: value})
                    else:
                        temp_content = temp_value.get('content')

                        if content != temp_content:
                            all_reshapedata.update({id_: value})
                else:
                    all_reshapedata.update({id_: value})
    fp.close()
    return all_reshapedata


def parser_alldata(dir_in,file_out):
    files = os.listdir(dir_in)  # 得到文件夹下的所有文件名称
    all_data = {}
    op =  open(file_out, 'w', encoding='utf-8')
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):
                # and file!='trs_govmsgbox_guizhougjj.json':
            file_path = os.path.join(dir_in, file)
            print(file_path)
            all_data = load_govmsboxdata(file_path, all_data)
    if all_data:
        for id_, data in all_data.items():
            json.dump(data, op, ensure_ascii=False)
            op.write('\n')
    sum_ = all_data.__len__()
    print('数据总量：%d' %sum_)
    op.close()


def json2txt(file_in,file_out):
    '''

    '''
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding='utf-8')
    lines = fp.readlines()
    for line in lines:
        data = json.loads(line)
        title = data['title'].strip()
        content = data['content']
        content = remove_html_url(content)
        sentences = cut_sent(content)
        op.write(title)
        op.write('\n')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.__len__()>0:
                op.write(sentence)
        op.write('\n\n')
    fp.close()
    op.close()


def train_dev_splits(files_in, files_out, train_set_rate=0.8):
    train_op = open(files_out + '/train.json', 'w', encoding="utf-8")
    dev_op = open(files_out + '/dev.json', 'w', encoding="utf-8")

    train_count, dev_count = 0, 0

    files = os.listdir(files_in)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file) :  # 判断是否是文件夹，不是文件夹才打开
            file_path = os.path.join(files_in,file)
            print(file_path)
            fp = open(file_path, 'r', encoding='utf-8');  # 打开文件

            lines = fp.readlines()
            data_len = len(lines)

            indices = list(range(data_len))
            np.random.shuffle(indices)

            test_size = 1 - train_set_rate
            x_train, x_test, _, _ = train_test_split(indices, indices, test_size=test_size, random_state=0)

            for id_, line in enumerate(lines):
                if id_ % 10000 == 0:
                    print(id_)
                data = json.loads(line)
                if id_ in x_train:
                    train_count += 1
                    json.dump(data, train_op, ensure_ascii=False)
                    train_op.write('\n')
                else:
                    dev_count += 1
                    json.dump(data, dev_op, ensure_ascii=False)
                    dev_op.write('\n')

    print("Split Later, There have {} train cases, {} dev cases".format(train_count, dev_count))


def merger_two_json(file_in1, file_in2, file_out):

    fp1 = open(file_in1,'r',encoding='utf-8')
    fp2 = open(file_in2, 'r', encoding='utf-8')
    op = open(file_out,'w',encoding='utf-8')

    all_data1 = json.load(fp1)
    all_data2 = json.load(fp2)
    datas = {}
    for data in all_data1:
        reply_id = data['reply_id']
        datas.update({reply_id:data})

    for data in all_data2:
        reply_id = data['reply_id']
        if not datas.__contains__(reply_id):
            datas.update({reply_id:data})

    json_list = []
    for k,v in datas.items():
        json_list.append(v)

    json.dump(json_list, op, ensure_ascii=False, indent=1)
    print('reply label data 总数:%d' % len(json_list))

    fp1.close()
    fp2.close()
    op.close()


if __name__ == '__main__':
    '''
    '''
    # dir_in = '../data/source'
    # file_out = '../data/letter.json'
    # parser_alldata(dir_in, file_out)

    # files_in = '../data/'
    # files_out = '../data/v0_0_1'
    # train_dev_splits(files_in, files_out)

    # file_in = '../data/v0_0_1/dev.json'
    # file_out = '../data/v0_0_1/dev.txt'
    # json2txt(file_in, file_out)
    #
    # file_in = '../data/v0_0_1/train.json'
    # file_out = '../data/v0_0_1/train.txt'
    # json2txt(file_in, file_out)





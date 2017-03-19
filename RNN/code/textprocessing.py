# -*- coding: utf-8 -*-
import cPickle
import gzip
import os
import sys
import re
import numpy
import theano
import pandas
from sklearn.feature_extraction.text import CountVectorizer
import pdb
from statsmodels.regression.tests.test_quantile_regression import idx
from sympy.tensor.indexed import Idx

datapath = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), 'data'), 'my_data')

def prepare_data(seqs, labels, maxlen=None):
    #返回句子以及其对应的标志矩阵和标签
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels

def split_sentence(sent):
    #将句子分成词和标点符号
#     return sent.split()
    return re.findall(r"[^\s']+|[.,!?;]", sent)

def check_dir_files(dataset_path):

    # 检查数据集的路径是否存在
    if not os.path.exists(dataset_path):
        print "数据集路径不存在"
        sys.exit(1)
    # check if all files are available
    filenames = ['greatewall_train.txt', 'greatewall_test.txt']
    filepaths = []
    for f in filenames:
        filepath = os.path.join(dataset_path, f)
        if not os.path.isfile(filepath):
            print "%s 不存在" % f
            sys.exit(1)
        filepaths.append(filepath)
    return filepaths

def build_dict(dataset_path):
    #构建所有词和词下标的词典
    dict_path = os.path.join(dataset_path, 'dictionary.pkl')
    filepaths = check_dir_files(dataset_path)
    if os.path.isfile(dict_path):
        dictionary = cPickle.load(dict_path)
        return dictionary

    print "dictionary.pkl 不存在 - 开始生成dictionary..."
    all_sents = []
    for f in filepaths:
        df = pandas.read_csv(f,error_bad_lines=False)
        sentences = list(df.ix[:, 0])
        all_sents = all_sents + sentences
    
    vectorizer = CountVectorizer().fit(all_sents)
    dictionary = vectorizer.vocabulary_
    dictionary_series = pandas.Series(dictionary.values(), index=dictionary.keys()) + 2
    dictionary_series.sort(axis=1, ascending=False)
    dictionary = list(dictionary_series.index)

    
    dictionary_path = os.path.join(dataset_path, 'dictionary.pkl')
    cPickle.dump(dictionary, open(dictionary_path, 'wb'))

    return dictionary

def get_dataset_file(csv_path_dir):
    
    filepaths = check_dir_files(csv_path_dir)
    dictionary = build_dict(csv_path_dir)
    result = []
    for fidx, f in enumerate(filepaths):       
        df = pandas.read_csv(f, sep=',')       
#         sentences = list(df['sentence'])
#         labels = list(df['label'])
        sentences = list(df.ix[:, 0])
        labels = list(df.ix[:, 1])
        for idx, sent in enumerate(sentences):
            sent_vect = []
            for wrd in split_sentence(sent):
                try:
                    sent_vect.append(dictionary.index(wrd))
                except:
                    sent_vect.append(1)
            sentences[idx] = sent_vect
            print idx
            print sentences[idx]
        result.append((sentences, labels))

    output_path = os.path.join(csv_path_dir, 'my_data.pkl')
    output_file = open(output_path, 'wb')
    for r in result:
        cPickle.dump(r, output_file)
    
    return output_path

def load_data(path=os.path.join(datapath, 'my_data.pkl'),
              n_words=1000000, valid_portion=0.1, 
              maxlen=None,sort_by_len=True):
    #加载数据集
    if not os.path.isfile(path):
        # 如果没有my_data.pkl就去生成
        data_dir, data_file = os.path.split(path)
        get_dataset_file(data_dir)
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # 将训练集分出一部分成验证集
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set
    
    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    #按照句子的长度进行排序
    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    return train, valid, test


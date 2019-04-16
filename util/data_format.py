# -*- encoding:utf-8 -*-

import numpy as np
import pickle
import json
import math
import codecs


def load_vec_txt(file_name, vocab, k=100):
    f = codecs.open(file_name, 'r', encoding='utf-8')
    w2v = {}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknown_token = 0
    for line in f:
        if len(line) < k:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-1 * math.sqrt(3 / k), math.sqrt(3 / k), k)
    for word in vocab:
        if not w2v.__contains__(word):
            print('UNK---------------- ', word)
            w2v[word] = w2v["**UNK**"]
            unknown_token += 1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]
    print('Unknown tokens in w2v', unknown_token)
    return k, W


def load_vec_onehot(vocab_w_inx):
    """
    Loads 300x1 word vecs from word2vec
    """
    k = vocab_w_inx.__len__()
    W = np.zeros(shape=(vocab_w_inx.__len__() + 1, k + 1))
    for word in vocab_w_inx:
        W[vocab_w_inx[word], vocab_w_inx[word]] = 1.
    # W[1, 1] = 1.
    return k, W


def make_idx_word_index(file, max_s, max_c, source_vob, target_vob, target_1_vob, source_char, ismulti):
    data_s_all = []
    data_t_all = []
    data_t_1_all = []
    data_c_all = []
    f = codecs.open(file, 'r', encoding='utf-8')
    fr = f.readlines()
    for num, line in enumerate(fr):
        print(num)
        if len(line) <= 1:
            continue
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['title']
        t_sent = sent['label']
        data_s = []
        if len(s_sent) > max_s:
            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:
            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)
        targetvec = np.zeros(len(target_vob))
        targetvec[target_vob[t_sent]] = 1
        data_t_all.append(targetvec)
        if ismulti:
            t_1_sent = sent['label1']
            targetvec1 = np.zeros(len(target_1_vob))
            targetvec1[target_1_vob[t_1_sent]] = 1
            data_t_1_all.append(targetvec1)
        data_w = []
        for ii in range(0, min(max_s, len(s_sent))):
            word = s_sent[ii]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_char.__contains__(word[chr]):
                    data_c.append(source_char["**UNK**"])
                else:
                    data_c.append(source_char[word[chr]])
            num = max_c - word.__len__()
            for i in range(0, max(num, 0)):
                data_c.append(0)
            data_w.append(data_c)
        num = max_s - len(s_sent)
        for inum in range(0, num):
            data_tmp = []
            for i in range(0, max_c):
                data_tmp.append(0)
            data_w.append(data_tmp)
        data_c_all.append(data_w)
    f.close()
    if ismulti:
        return data_s_all, data_t_all, data_t_1_all, data_c_all
    else:
        return data_s_all, data_t_all, data_c_all


def get_Char_index(files):
    source_vob = {}
    sourc_idex_word = {}
    count = 1
    max_s = 0
    dict = {}
    for file in files:

        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['title']
            for word in sourc:
                for i in range(len(word)):
                    if not source_vob.__contains__(word[i]):
                        source_vob[word[i]] = count
                        sourc_idex_word[count] = word[i]
                        count += 1
                if word.__len__() in dict.keys():
                    dict[word.__len__()] = dict[word.__len__()] + 1
                else:
                    dict[word.__len__()] = 1
                if word.__len__() > max_s:
                    max_s = word.__len__()

        f.close()

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1
    return source_vob, sourc_idex_word, max_s


def get_Word_index(files, testfile, ismulti):
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}

    target_1_vob = {}
    tarcount1 = 0
    target_1_idex_word = {}

    max_s = 0
    tarcount = 0
    count = 1
    dict = {}
    for file in files:
        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['title']
            for word in sourc:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1

            if sourc.__len__() in dict.keys():
                dict[sourc.__len__()] = dict[sourc.__len__()] + 1
            else:
                dict[sourc.__len__()] = 1

            if sourc.__len__() > max_s:
                max_s = sourc.__len__()

            target = sent['label']
            if not target_vob.__contains__(target):
                target_vob[target] = tarcount
                target_idex_word[tarcount] = target
                tarcount += 1

            if ismulti:
                target1 = sent['label1']
                if not target_1_vob.__contains__(target1):
                    target_1_vob[target1] = tarcount1
                    target_1_idex_word[tarcount1] = target1
                    tarcount1 += 1
        f.close()
    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    f = codecs.open(testfile, 'r', encoding='utf-8')
    fr = f.readlines()
    for line in fr:
        if line.__len__() <= 1:
            continue
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['title']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

    f.close()
    if ismulti:
        return source_vob, sourc_idex_word, target_vob, target_1_vob, target_idex_word, target_1_idex_word, max_s
    else:
        return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def make_idx_char_index(trainfile, max_s, max_c, source_char):
    data_c_all = []

    f1 = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f1.readlines()
    for num, line in enumerate(lines):
        print(num)
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['title']
        data_w = []
        for word in sourc:
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_char.__contains__(word[chr]):
                    data_c.append(source_char["**UNK**"])
                else:
                    data_c.append(source_char[word[chr]])

            num = max_c - word.__len__()
            for i in range(0, max(num, 0)):
                data_c.append(0)

            data_w.append(data_c)

        num = max_s - len(sourc)
        for inum in range(0, num):
            data_tmp = []
            for i in range(0, max_c):
                data_tmp.append(0)
            data_w.append(data_tmp)

        data_c_all.append(data_w)
    f1.close()
    return data_c_all


def get_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen=50, isMulti=False,
             left=0):
    char_vob, vob_idex_char, max_c = get_Char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)
    max_c = 6
    if not isMulti:
        word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_Word_index({trainfile}, testfile, False)
        print("word_vob vocab size: ", str(len(word_vob)))
        print("max_s: ", max_s)
        print("target vocab size: " + str(target_vob))
        max_s = 20

        word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
        print("source_W  size: " + str(len(word_W)))
        char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
        print('char_W shape:', char_W.shape)

        train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, None,
                                                                    char_vob, False)
        print('train_all size', len(train_all), 'target_all', len(target_all))
        print('train_all_char size', len(train_all_char))

        extra_test_num = int(len(train_all) / 5)

        # left = 0
        right = left + 1
        test = train_all[extra_test_num * left:extra_test_num * right]
        test_label = target_all[extra_test_num * left:extra_test_num * right]
        train = train_all[:extra_test_num * left] + train_all[extra_test_num * right:]
        train_label = target_all[:extra_test_num * left] + target_all[extra_test_num * right:]
        print('extra_test_num', extra_test_num)
        print('train len  ', train.__len__(), len(train_label))
        print('test len  ', test.__len__(), len(test_label))

        test_char = train_all_char[extra_test_num * left:extra_test_num * right]
        train_char = train_all_char[:extra_test_num * left] + train_all_char[extra_test_num * right:]
        print('test_char len  ', test_char.__len__(), )
        print('train_char len  ', train_char.__len__())

        print("dataset created!")
        out = codecs.open(datafile, 'wb')
        pickle.dump([train, train_char, train_label,
                     test, test_char, test_label,
                     word_vob, vob_idex_word, word_W, word_k,
                     target_vob, vob_idex_target,
                     char_vob, vob_idex_char, char_W, char_k,
                     max_s, max_c
                     ], out, 0)
        out.close()

    elif isMulti:
        word_vob, vob_idex_word, target_vob, target_1_vob, vob_idex_target, vob_idex_target_1, max_s = get_Word_index(
            {trainfile}, testfile, True)
        print("word_vob vocab size: ", str(len(word_vob)))
        print("max_s: ", max_s)
        print("target vocab size: " + str(target_vob))
        max_s = 700  # max_s = 800

        word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
        print("source_W  size: " + str(len(word_W)))
        char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
        print('char_W shape:', char_W.shape)

        train_all, target_all, target_1_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob,
                                                                                  target_vob, target_1_vob,
                                                                                  char_vob, True)
        print('train_all size', len(train_all), 'target_all', len(target_all), 'target_1_all', len(target_1_all))
        print('train_all_char size', len(train_all_char))

        extra_test_num = int(len(train_all) / 5)

        right = left + 1
        test = train_all[extra_test_num * left:extra_test_num * right]
        test_label = target_all[extra_test_num * left:extra_test_num * right]
        test_label1 = target_1_all[extra_test_num * left:extra_test_num * right]
        train = train_all[:extra_test_num * left] + train_all[extra_test_num * right:]
        train_label = target_all[:extra_test_num * left] + target_all[extra_test_num * right:]
        train_label1 = target_1_all[:extra_test_num * left] + target_1_all[extra_test_num * right:]
        print('extra_test_num', extra_test_num)
        print('train len  ', train.__len__(), len(train_label))
        print('test len  ', test.__len__(), len(test_label))

        test_char = train_all_char[extra_test_num * left:extra_test_num * right]
        train_char = train_all_char[:extra_test_num * left] + train_all_char[extra_test_num * right:]
        print('test_char len  ', test_char.__len__(), )
        print('train_char len  ', train_char.__len__())

        print("dataset created!")
        out = codecs.open(datafile, 'wb')
        pickle.dump([train, train_char, train_label, train_label1,
                     test, test_char, test_label, test_label1,
                     word_vob, vob_idex_word, word_W, word_k,
                     target_vob, vob_idex_target, target_1_vob, vob_idex_target_1,
                     char_vob, vob_idex_char, char_W, char_k,
                     max_s, max_c
                     ], out, 0)
        out.close()


if __name__ == "__main__":
    maxlen = 50
    trainfile = "../data/train_data.txt"
    testfile = "../data/test_data.txt"
    w2v_file = "../modfile/Word2Vec.mod"
    char2v_file = "../modfile/Char2Vec.mod"
    w2v_k = 100
    c2v_k = 100
    datafile = "../modfile/data.pkl"
    modelfile = "../modfile/model.pkl"

    char_vob, vob_idex_char, max_c = get_Char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)

    word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_Word_index({trainfile}, testfile)
    print("word_vob vocab size: ", str(len(word_vob)))
    print("max_s: ", max_s)
    print("target vocab size: " + str(target_vob))

    word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("source_W  size: " + str(len(word_W)))
    char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
    print('char_W shape:', char_W.shape)

    train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, char_vob)
    print('train_all size', len(train_all), 'target_all', len(target_all))
    print('train_all_char size', len(train_all_char))
    print(train_all_char[0])

    extra_test_num = int(len(train_all) / 6)

    left = 0
    right = 1
    test = train_all[extra_test_num * left:extra_test_num * right]
    test_label = target_all[extra_test_num * left:extra_test_num * right]
    train = train_all[:extra_test_num * left] + train_all[extra_test_num * right:]
    train_label = target_all[:extra_test_num * left] + target_all[extra_test_num * right:]
    print('extra_test_num', extra_test_num)
    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))

    test_char = train_all_char[extra_test_num * left:extra_test_num * right]
    train_char = train_all_char[:extra_test_num * left] + train_all_char[extra_test_num * right:]
    print('test_char len  ', test_char.__len__(), )
    print('train_char len  ', train_char.__len__())
    print(train_char[0])

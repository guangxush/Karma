# -*- encoding:utf-8 -*-

import codecs
import json
import pickle
import numpy as np
import sys
import time
import os
import jieba

sys.path.append("..")
import news_classification


def make_idx_word_index(s_sent, max_s, max_c, word_vob, char_vob):
    data_s = []
    if len(s_sent) > max_s:
        i = 0
        while i < max_s:
            if not word_vob.__contains__(s_sent[i]):
                data_s.append(word_vob["**UNK**"])
            else:
                data_s.append(word_vob[s_sent[i]])
            i += 1
    else:
        i = 0
        while i < len(s_sent):
            if not word_vob.__contains__(s_sent[i]):
                data_s.append(word_vob["**UNK**"])
            else:
                data_s.append(word_vob[s_sent[i]])
            i += 1
        num = max_s - len(s_sent)
        for inum in range(0, num):
            data_s.append(0)

    data_w = []
    for ii in range(0, min(max_s, len(s_sent))):
        word = s_sent[ii]
        data_c = []
        for chr in range(0, min(word.__len__(), max_c)):
            if not char_vob.__contains__(word[chr]):
                data_c.append(char_vob["**UNK**"])
            else:
                data_c.append(char_vob[word[chr]])

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

    return [data_s], [data_w]


def predict_news_classes(model_name, datafile, modelfile, testfile):
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    nn_model = news_classification.select_model(model_name, sourcevocabsize=len(word_vob),
                                                targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
                                                emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)

    acc_best = 0.
    acc = 0.
    if os.path.exists("../modfile/" + modelfile):
        nn_model.load_weights("../modfile/" + modelfile)

        loss, acc = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                      batch_size=batch_size)
        print('\n test_test score:', loss, acc)

    if os.path.exists("../modfile/" + modelfile + ".best_model.h5"):
        nn_model.load_weights("../modfile/" + modelfile + ".best_model.h5")
        nn_model.load_weights("../modfile/" + modelfile + ".best_model.h5")
        loss, acc_best = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                           batch_size=batch_size)
        print('best model...\n test_test score:', loss, acc_best)

    if acc >= acc_best:
        nn_model.load_weights("../modfile/" + modelfile)

    else:
        nn_model.load_weights("../modfile/" + modelfile + ".best_model.h5")
        nn_model.summary()

    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    t = str(int(time.time()))
    fw = codecs.open("../result/classify_result_" + t + ".txt", 'w', encoding='utf-8')
    for num, line in enumerate(lines):
        print(num)
        item = json.loads(line.rstrip('\n'))
        # id = item['title']
        # todo add id
        title = item['title']
        test_words, test_char = make_idx_word_index(title, max_s, max_c, word_vob, char_vob)
        predictions = nn_model.predict([np.array(test_words),
                                        np.array(test_char)], verbose=0)
        for si in range(0, len(predictions)):
            sent = predictions[si]
            item_p = np.argmax(sent)
            label = vob_idex_target[item_p]
            fw.write(str(title) + '\t' + str(label) + '\n')
    fw.close()


def predict_news_classes_online(model_name, datafile, modelfile, title):
    document_cut = jieba.cut(title)
    result = '@+@'.join(document_cut)
    results = result.split('@+@')
    lines = []
    for w in results:
        lines.append(w)
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    nn_model = news_classification.select_model(model_name, sourcevocabsize=len(word_vob),
                                                targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
                                                emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)

    acc_best = 0.
    acc = 0.
    if os.path.exists("../modfile/" + modelfile):
        nn_model.load_weights("../modfile/" + modelfile)

        loss, acc = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                      batch_size=batch_size)
        print('\n test_test score:', loss, acc)

    if os.path.exists("../modfile/" + modelfile + ".best_model.h5"):
        nn_model.load_weights("../modfile/" + modelfile + ".best_model.h5")
        nn_model.load_weights("../modfile/" + modelfile + ".best_model.h5")
        loss, acc_best = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                           batch_size=batch_size)
        print('best model...\n test_test score:', loss, acc_best)

    if acc >= acc_best:
        nn_model.load_weights("../modfile/" + modelfile)

    else:
        nn_model.load_weights("../modfile/" + modelfile + ".best_model.h5")
        nn_model.summary()

    test_words, test_char = make_idx_word_index(lines, max_s, max_c, word_vob, char_vob)
    predictions = nn_model.predict([np.array(test_words),
                                    np.array(test_char)], verbose=0)
    for si in range(0, len(predictions)):
        sent = predictions[si]
        item_p = np.argmax(sent)
        label = vob_idex_target[item_p]
        print(str(lines) + '\t' + str(label) + '\n')


if __name__ == '__main__':
    batch_size = 128
    title = sys.argv[1]

    # predict_news_classes(model_name='mlp',
    #                      datafile="../modfile/news_data.pkl",
    #                      modelfile="mlp_1.pkl",
    #                      testfile='../data/test_data.txt')
    predict_news_classes_online(model_name='mlp',
                                datafile="../modfile/news_data.pkl",
                                modelfile="mlp_1.pkl",
                                title=title)

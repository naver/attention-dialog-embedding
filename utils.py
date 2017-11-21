import re
import random
import ast
import unicodedata
import pickle
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import torch
import torchwordemb
from fields import Dial
from fields import BClass
from torch.autograd import Variable
from settings import *

def make_false_sample_movie(dial_list, input_dial):
    #with open(dial_list_dir, encoding='utf-8') as file:
    #    dial_list = file.read()
    # dial_list = ast.literal_eval(dial_list)
    len_dial = len(dial_list)
    false_var = []
    for i in range(len_dial):
        temp_dial = dial_list[i]
        dial_num_list = list(range(len_dial))
        dial_num_list.remove(i)
        for j in range(1,min(len(temp_dial),20)):
            temp_input_var = temp_dial[:j]
            rand_dial = random.sample(dial_num_list, 1)[0]
            temp_false_sent = random.sample(dial_list[rand_dial], 1)[0]
            temp_pair = (temp_false_sent, [0,0,30])
            temp_pair = variablesFromPair_glove(temp_pair, input_dial)
            temp_input_var.append(temp_false_sent)
            temp_input_var = [variableFromSentence_glove(sent, input_dial) for sent in temp_input_var]
            false_var.append((temp_pair, temp_input_var))            
    return false_var

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_glove(path, input_dial, glove_dim, load_glove):
    if load_glove:
        voc, vec = torchwordemb.load_glove_text(path)
    else:
        if glove_dim == 100:
            voc = pickle.load(open('./data/voc_100', 'rb'))
            vec = pickle.load(open('./data/vec_100', 'rb'))
        elif glove_dim == 50:
            voc = pickle.load(open('./data/voc', 'rb'))
            vec = pickle.load(open('./data/vec', 'rb'))
    input_dial.glove_voc.update(voc)
    input_dial.glove_vec = vec


def mark_unknown(dial_list, input_dial):
    glove_words = list(input_dial.glove_voc.keys())
    unk_dial = []
    for dial in dial_list:
        temp_dial = []
        for sent in dial:
            unk_sent = [x if x in glove_words else "<unknown>" for x in sent.split()]
            temp_dial.append(unk_sent)
        unk_dial.append(temp_dial)

    return unk_dial


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = re.sub(r"([0-9]+)", r" <number> ", s)
    s = re.sub(r"[^a-zA-Z<>]+", r" ", s)
    s = unicodeToAscii(s.lower())
    s = " ".join(s.split())
    return s


def readDialog(dial_list_dir='../dial_list.txt', tgt_list_dir='../tgt_list.txt'):
    print("Reading lines...")

    # Read the file and split into lines
    with open(dial_list_dir, encoding='utf-8') as file:
        dial_list = file.read()
    dial_list = ast.literal_eval(dial_list)

    with open(tgt_list_dir, encoding='utf-8') as file:
        tgt_list = file.read()
    tgt_list = ast.literal_eval(tgt_list)

    pairs = []
    dial_list = [[normalizeString(' '.join(dial)) for dial in dials] for dials in dial_list]
    tgt_list = [[[str(s) for s in score] for score in scores] for scores in tgt_list]
    for i in range(len(dial_list)):
        temp_dial = dial_list[i]
        if len(temp_dial) == 20:
            temp_dial = [temp_dial[j] for j in range(1, 20, 2)]
            temp_tgt = tgt_list[i]
            temp_tgt = [x for x in temp_tgt if x!=['0','0','0']]
        elif len(temp_dial) == 21:
            temp_dial = [temp_dial[j] for j in range(2,21,2)]
            temp_tgt = tgt_list[i]
            temp_tgt = [x for x in temp_tgt if x!=['0','0','0']]
        else:
            print("Error: Dial list has odd element - length it nor 20 or 21")
        temp_pairs = list(zip(temp_dial, temp_tgt))
        pairs.append(temp_pairs)

    input_dial = Dial()
    output_class = BClass(len(tgt_list[0][0]), 30)

    return dial_list, input_dial, output_class, pairs

def readDialog_movie(dial_list_dir='../dial_list.txt', tgt_list_dir='../tgt_list.txt'):
    print("Reading lines...")

    # Read the file and split into lines
    with open(dial_list_dir, encoding='utf-8') as file:
        dial_list = file.read()
    dial_list = ast.literal_eval(dial_list)

    with open(tgt_list_dir, encoding='utf-8') as file:
        tgt_list = file.read()
    tgt_list = ast.literal_eval(tgt_list)

    pairs = []
    dial_list = [[normalizeString(' '.join(dial)) for dial in dials] for dials in dial_list]
    tgt_list = [[[str(s) for s in score] for score in scores] for scores in tgt_list]
    for i in range(len(dial_list)):
        temp_dial = dial_list[i]
        temp_dial = temp_dial[1:]
        temp_tgt = tgt_list[i]
        temp_tgt = [x for x in temp_tgt if x!=['0','0','0']]
        # print("Error: Dial list has odd element - length it nor 20 or 21")
        temp_pairs = list(zip(temp_dial, temp_tgt))
        pairs.append(temp_pairs)

    input_dial = Dial()
    output_class = BClass(len(tgt_list[0][0]), 30)

    return dial_list, input_dial, output_class, pairs

def prepareData(dial_list_dir='../dial_list.txt', tgt_list_dir='../tgt_list.txt', movie=False):
    if not movie:
        dial_list, input_dial, output_class, pairs = readDialog(dial_list_dir, tgt_list_dir)
    if movie:
        dial_list, input_dial, output_class, pairs = readDialog_movie(dial_list_dir, tgt_list_dir)
    print("Read %s sentence-annotation_score pairs" % len([item for sublist in pairs for item in sublist]))
    print("Counting words...")
    _ = [input_dial.addSentence(dial) for dials in dial_list for dial in dials]
    for dials in pairs:
        for dial in dials:
            output_class.addScoreList(dial[1])

    print("Counted words:")
    print(input_dial.n_words)
    print("Number of class:")
    print(output_class.n_class)
    return dial_list, input_dial, output_class, pairs

def indexesFromSentence(dial, sentence):
    if type(sentence) == list:
        return [dial.word2index[word] for word in sentence]
    elif type(sentence) == str:
        return [dial.word2index[word] for word in sentence.split(' ')]
    else:
        return "Error"


def variableFromSentence(dial, sentence):
    indexes = indexesFromSentence(dial, sentence)
    # indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variableFromSentence_glove(sentence, input_dial):
    words = sentence.split()
    indexes = [input_dial.glove_voc[word] for word in words]
    # indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variableFromAnnotation(bclass, scores):
    scores = [bclass.word2index[score] for score in scores]
    result = Variable(torch.LongTensor(scores).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def PvariableFromAnnotation(scores):
    scores = [int(score) for score in scores]
    scores = [score / sum(scores) for score in scores]
    result = Variable(torch.FloatTensor(scores).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair, input_dial):
    input_variable = variableFromSentence(input_dial, pair[0])
    target_variable = PvariableFromAnnotation(pair[1])
    return (input_variable, target_variable)


def variablesFromPair_glove(pair, input_dial):
    input_variable = variableFromSentence_glove(pair[0], input_dial)
    target_variable = PvariableFromAnnotation(pair[1])
    return (input_variable, target_variable)


def make_context(encoder_outputs, input_length):
    new_list = encoder_outputs[:input_length, ]
    context_tensor = torch.mean(new_list, 0)
    return context_tensor

def variableFromAnnotation(bclass, scores):
    scores = [bclass.word2index[score] for score in scores]
    result = Variable(torch.LongTensor(scores).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def PvariableFromAnnotation(scores):
    scores = [int(score) for score in scores]
    scores = [score / sum(scores) for score in scores]
    result = Variable(torch.FloatTensor(scores).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair, input_dial):
    input_variable = variableFromSentence(input_dial, pair[0])
    target_variable = PvariableFromAnnotation(pair[1])
    return (input_variable, target_variable)


def variablesFromPair_glove(pair, input_dial):
    input_variable = variableFromSentence_glove(pair[0], input_dial)
    target_variable = PvariableFromAnnotation(pair[1])
    return (input_variable, target_variable)


def make_context(encoder_outputs, input_length):
    new_list = encoder_outputs[:input_length, ]
    context_tensor = torch.mean(new_list, 0)
    # new_list columewise average
    return context_tensor


def random_pair_var_list_glove(pairs, dial_list, input_dial):
    rand_pair_num = random.choice(range(len(pairs)))
    rand_dial_num = random.choice(range(10))
    training_pair = pairs[rand_pair_num][rand_dial_num]
    training_pair = variablesFromPair_glove(training_pair, input_dial)
    temp_data = dial_list[rand_pair_num]
    if len(temp_data) == 20:
        input_variable_list = temp_data[:rand_dial_num * 2 + 2]
    elif len(temp_data) == 21:
        input_variable_list = temp_data[:rand_dial_num * 2 + 3]
    else:
        print("ERROR: dial contains nor 20 or 21 sents")
    input_variable_list = [variableFromSentence_glove(var, input_dial) for var in input_variable_list]
    return training_pair, input_variable_list


def random_pair_var_list(pairs, dial_list, input_dial):
    rand_pair_num = random.choice(range(len(pairs)))
    rand_dial_num = random.choice(range(10))
    training_pair = pairs[rand_pair_num][rand_dial_num]
    previous_texts = dial_list[rand_pair_num][:2*rand_dial_num+1]
    text = (previous_texts, training_pair[0])
    training_pair = variablesFromPair(training_pair, input_dial)
    input_variable_list = dial_list[rand_pair_num][:rand_dial_num * 2 + 2]
    input_variable_list = [variableFromSentence(input_dial, var) for var in input_variable_list]
    return text, training_pair, input_variable_list


def split_data(pairs, test_ratio, dial_list, input_dial):
    num_list = list(range(len(pairs)))
    random.shuffle(num_list)
    test_idx = math.floor(len(pairs) * test_ratio)
    test_pair_num = num_list[:test_idx]
    train_pair_num = num_list[test_idx:]
    test_vars = []
    for num in test_pair_num:
        temp_pairs = pairs[num]
        temp_pair = [temp_pairs[i] for i in range(10)]
        temp_pair = [variablesFromPair_glove(pair, input_dial) for pair in temp_pair]
        if len(dial_list[num]) == 20:
            temp_input_var = [dial_list[num][:i * 2 + 2] for i in range(10)]
        elif len(dial_list[num]) == 21:
            temp_input_var = [dial_list[num][:i*2+3] for i in range(10)]
        temp_input_var = [[variableFromSentence_glove(sent, input_dial) for sent in var] for var in temp_input_var]
        test_vars.extend(list(zip(temp_pair, temp_input_var)))
    train_vars = []
    for num in train_pair_num:
        temp_pairs = pairs[num]
        temp_pair = [temp_pairs[i] for i in range(10)]
        temp_pair = [variablesFromPair_glove(pair, input_dial) for pair in temp_pair]
        if len(dial_list[num]) == 20:
            temp_input_var = [dial_list[num][:i * 2 + 2] for i in range(10)]
        elif len(dial_list[num]) == 21:
            temp_input_var = [dial_list[num][:i*2+3] for i in range(10)]
        temp_input_var = [[variableFromSentence_glove(sent, input_dial) for sent in var] for var in temp_input_var]
        train_vars.extend(list(zip(temp_pair, temp_input_var)))
    return train_vars, test_vars

def split_data_features(pairs, test_ratio, dial_list, input_dial, features=None):
    num_list = list(range(len(pairs)))
    random.shuffle(num_list)
    test_idx = math.floor(len(pairs) * test_ratio)
    test_pair_num = num_list[:test_idx]
    train_pair_num = num_list[test_idx:]
    train_features = None
    test_features = None
    if not features is None:
        use_feature = True
        features = [features[i*10:i*10+10] for i in range(415)]
        train_features = []
        test_features = []
    test_vars = []
    for num in test_pair_num:
        temp_pairs = pairs[num]
        temp_pair = [temp_pairs[i] for i in range(10)]
        temp_pair = [variablesFromPair_glove(pair, input_dial) for pair in temp_pair]
        if len(dial_list[num]) == 20:
            temp_input_var = [dial_list[num][:i * 2 + 2] for i in range(10)]
        elif len(dial_list[num]) == 21:
            temp_input_var = [dial_list[num][:i*2+3] for i in range(10)]
        temp_input_var = [[variableFromSentence_glove(sent, input_dial) for sent in var] for var in temp_input_var]
        test_vars.extend(list(zip(temp_pair, temp_input_var)))
        if use_feature:
            test_features.extend(features[num])
    train_vars = []
    for num in train_pair_num:
        temp_pairs = pairs[num]
        temp_pair = [temp_pairs[i] for i in range(10)]
        temp_pair = [variablesFromPair_glove(pair, input_dial) for pair in temp_pair]
        if len(dial_list[num]) == 20:
            temp_input_var = [dial_list[num][:i * 2 + 2] for i in range(10)]
        elif len(dial_list[num]) == 21:
            temp_input_var = [dial_list[num][:i*2+3] for i in range(10)]
        temp_input_var = [[variableFromSentence_glove(sent, input_dial) for sent in var] for var in temp_input_var]
        train_vars.extend(list(zip(temp_pair, temp_input_var)))
        if use_feature:
            train_features.extend(features[num])

    return train_vars, test_vars, train_features, test_features

def split_data_movie(pairs, test_ratio, dial_list, input_dial):
    num_list = list(range(len(pairs)))
    random.shuffle(num_list)
    test_idx = math.floor(len(pairs) * test_ratio)
    train_pair_num = num_list
    train_vars = []
    for num in train_pair_num:
        temp_pairs = pairs[num]
        temp_pair = [temp_pairs[i] for i in range(len(temp_pairs))]
        temp_pair = [variablesFromPair_glove(pair, input_dial) for pair in temp_pair]
        temp_input_var = [dial_list[num][:i+2] for i in range(len(temp_pairs))]
        temp_input_var = [[variableFromSentence_glove(sent, input_dial) for sent in var] for var in temp_input_var]
        train_vars.extend(list(zip(temp_pair, temp_input_var)))
    return train_vars

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def savePlot(points, epoch):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.5)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    timestr = time.strftime("%m%d_%H%M%S")
    fig.savefig("./loss_figures/loss_figure_" + timestr + "_" + str(epoch))
    plt.close(fig)

def readDialog_submit(dial_list, tgt_list):
    pairs = []
    dial_list = [[normalizeString(' '.join(dial)) for dial in dials] for dials in dial_list]
    tgt_list = [[[str(s) for s in score] for score in scores] for scores in tgt_list]
    for i in range(len(dial_list)):
        temp_dial = dial_list[i]
        if len(temp_dial) == 20:
            temp_dial = [temp_dial[j] for j in range(1, 20, 2)]
            temp_tgt = tgt_list[i]
            temp_tgt = [x for x in temp_tgt if x!=['0','0','0']]
        elif len(temp_dial) == 21:
            temp_dial = [temp_dial[j] for j in range(2,21,2)]
            temp_tgt = tgt_list[i]
            temp_tgt = [x for x in temp_tgt if x!=['0','0','0']]
        else:
            print("Error: Dial list has odd element - length it nor 20 or 21")
        temp_pairs = list(zip(temp_dial, temp_tgt))
        pairs.append(temp_pairs)

    input_dial = Dial()
    output_class = BClass(len(tgt_list[0][0]), 30)

    return dial_list, input_dial, output_class, pairs

def load_separate_data_vars(dial_list, pairs, features=None):
    i_dial, i_pairs = dial_list[:100], pairs[:100]
    t_dial, t_pairs = dial_list[100:200], pairs[100:200]
    c_dial, c_pairs = dial_list[200:315], pairs[200:315]
    y_dial, y_pairs = dial_list[315:], pairs[315:]
    train_features, test_features, i_features, t_features, c_features, y_features = None, None, None, None, None, None
    if not features is None:
        i_features, t_features, c_features, y_features = features[:1000], features[1000:2000], features[2000:3150], features[3150:]
    
    # split data to train, test data if option selcted as random
    i_train_vars, i_test_vars, i_train_features, i_test_features= split_data_features(i_pairs, 0.1, i_dial, input_dial, i_features)
    t_train_vars, t_test_vars, t_train_features, t_test_features= split_data_features(t_pairs, 0.1, t_dial, input_dial, t_features)
    c_train_vars, c_test_vars, c_train_features, c_test_features= split_data_features(c_pairs, 0.1, c_dial, input_dial, c_features)
    y_train_vars, y_test_vars, y_train_features, y_test_features= split_data_features(y_pairs, 0.1, y_dial, input_dial, y_features)
    if not features is None:
        train_features = i_train_features+t_train_features+c_train_features+y_train_features
        test_features = i_test_features+t_test_features+c_test_features+y_test_features
    train_vars = i_train_vars+t_train_vars+c_train_vars+y_train_vars
    test_vars = i_test_vars+t_test_vars+c_test_vars+y_test_var
    return (train_vars,train_features), (test_vars, test_features), (i_train_vars, i_test_vars, i_train_features, i_test_features), (t_train_vars, t_test_vars, t_train_features, t_test_features), (c_train_vars, c_test_vars, c_train_features, c_test_features), (y_train_vars, y_test_vars, y_train_features, y_test_features)

def import_separate_data_vars(args):
    i_train_features, i_test_features = None, None
    t_train_features, t_test_features = None, None
    c_train_features, c_test_features = None, None
    y_train_features, y_test_features = None, None
    train_features, test_features = None, None

    i_train_vars = pickle.load(open('./data/i_train_vars_1', 'rb'))
    i_test_vars = pickle.load(open('./data/i_test_vars_1', 'rb'))
    t_train_vars = pickle.load(open('./data/t_train_vars_1', 'rb'))
    t_test_vars = pickle.load(open('./data/t_test_vars_1', 'rb'))
    c_train_vars = pickle.load(open('./data/c_train_vars_1', 'rb'))
    c_test_vars = pickle.load(open('./data/c_test_vars_1', 'rb'))
    y_train_vars = pickle.load(open('./data/y_train_vars_1', 'rb'))
    y_test_vars = pickle.load(open('./data/y_test_vars_1', 'rb'))
    train_vars = i_train_vars+t_train_vars+c_train_vars+y_train_vars
    test_vars = i_test_vars+t_test_vars+c_test_vars+y_test_vars

    if not args.no_feature:
        i_train_features = pickle.load(open('./data/i_train_features_1', 'rb'))
        i_test_features = pickle.load(open('./data/i_test_features_1', 'rb'))
        t_train_features = pickle.load(open('./data/t_train_features_1', 'rb'))
        t_test_features = pickle.load(open('./data/t_test_features_1', 'rb'))
        c_train_features = pickle.load(open('./data/c_train_features_1', 'rb'))
        c_test_features = pickle.load(open('./data/c_test_features_1', 'rb'))
        y_train_features = pickle.load(open('./data/y_train_features_1', 'rb'))
        y_test_features = pickle.load(open('./data/y_test_features_1', 'rb'))
        train_features = i_train_features+t_train_features+c_train_features+y_train_features
        test_features = i_test_features+t_test_features+c_test_features+y_test_features
    return (train_vars,train_features), (test_vars, test_features), (i_train_vars, i_test_vars, i_train_features, i_test_features), (t_train_vars, t_test_vars, t_train_features, t_test_features), (c_train_vars, c_test_vars, c_train_features, c_test_features), (y_train_vars, y_test_vars, y_train_features, y_test_features)
   

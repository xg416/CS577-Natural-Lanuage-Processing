import numpy as np
import string
#import argparse
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

'''Following are some helper functions from https://github.com/lixin4ever/E2E-TBSA/blob/master/utils.py to help parse the Targeted Sentiment Twitter dataset. You are free to delete this code and parse the data yourself, if you wish.

You may also use other parsing functions, but ONLY for parsing and ONLY from that file.
'''

def viterbi_decode(backpoint, pred_last_tag):
    tags = [pred_last_tag]
    idx = pred_last_tag
    backpoint.reverse()
    for p in backpoint:
        idx = p[idx]
        tags.append(idx)
    tags.reverse()
    return tags
        

def find_prefix(unit):
    length = len(unit)
    ini = unit[0]
    if ini == '@' and length > 1:
        return 1
    if ini == '#' and length > 1:
        return 2
    if ini.isupper():
        return 3
    if unit.isdigit():
        return 4
    if length > 4 and unit[0:4] == 'http':
        return 5
    return 0
    
def N_gram(words, N = 1):
    gram_list = []
    l = len(words)
    for i in range(N):
        words.insert(0, 'START_TAG')
        words.append('END_TAG')
    for i in range(N, l + N):
        gram_list.append(words[i-N:i+N+1])
    return gram_list
    
def add_context(dataset, window_size = 1):
    ct_list = []
    for data in dataset:
        gram_list = N_gram(data['words'], N = window_size)
        ct_list.append(gram_list)
    return ct_list


def tag2num(tag):
    assert tag in ['T-POS','T-NEU','T-NEG','O'], "illegal tag"
    if tag == 'T-POS':
        label = 0
    elif tag == 'T-NEU':
        label = 1
    elif tag == 'T-NEG':
        label = 2
    else:
        label = 3
    return label


def num2tag(num):
    assert num in [0,1,2,3], "illegal input label num"
    if num == 0:
        tag = 'T-POS'
    elif num == 1:
        tag = 'T-NEU'
    elif num == 2:
        tag = 'T-NEG'
    else:
        tag = 'O'
    return tag
    

def make_labels(dataset):
    # 4 means start and end tag
    label_list = []
    for data in dataset:
        label = [4] + [tag2num(tag) for tag in data['ts_raw_tags']] + [4]
        label_list.append(label)
    return label_list


def num2onehot(length, num):
    onehot_vec = np.zeros((length,))
    onehot_vec[num] = 1
    return onehot_vec

def encode_prefix(dataset, prefix_num):
    prefix_list = []
    for data in dataset:
        length = len(data['prefix'])
        onehot_vec = np.zeros((length, prefix_num))
        onehot_vec[range(length), data['prefix']] = 1
        prefix_list.append(onehot_vec)
    return prefix_list

def option1_trainset(dataset, window_size = 1):
    data_dic = {}
    voc = []
    sen_num = len(dataset)
    for data in dataset:
        for word in data['words']:
            if word not in voc:
                voc.append(word)
                data_dic[word] = len(voc) - 1
    
    voc.append('START_TAG')
    data_dic['START_TAG'] = len(voc)- 1
    voc.append('END_TAG')
    data_dic['END_TAG'] = len(voc)- 1
    voc.append('OUT_VOC')
    data_dic['OUT_VOC'] = len(voc)- 1
    vocab_size = len(voc)
    
    ct = add_context(dataset, window_size)
    ct_idx = []
    for sen in ct:
        unit_idx = []
        for word_list in sen:
            # note: in testing some words are out of voc
            unit_idx.append([data_dic[word] for word in word_list])
        ct_idx.append(unit_idx)
        
    tags = make_labels(dataset)
    pre_tags = [tag_list[0:-2] for tag_list in tags]
    labels = [tag_list[1:-1] for tag_list in tags]
    fo_tags = [tag_list[2:] for tag_list in tags]
    prefix = encode_prefix(dataset, 6)
    
    data_list = []
    for i in range(sen_num):
        word_list = []
        for j in range(len(ct_idx[i])):
            word_list.append((ct_idx[i][j], pre_tags[i][j], labels[i][j], fo_tags[i][j], prefix[i][j,:]))
        data_list.append(word_list)
    return data_dic, vocab_size, data_list
   

def option1_vt(data_dic, uknown_symb, dataset, window_size = 1):
    sen_num = len(dataset)
    vocab_list = data_dic.keys()
    ct = add_context(dataset, window_size)
    ct_idx = []
    for sen in ct:
        unit_idx = []
        for word_list in sen:
            words_idx = []
            # note: in testing some words are out of voc
            for word in word_list:
                if word in vocab_list:
                    words_idx.append(data_dic[word])
                else:
                    words_idx.append(data_dic[uknown_symb])
            unit_idx.append(words_idx)
        ct_idx.append(unit_idx)
        tags = make_labels(dataset)
    pre_tags = [tag_list[0:-2] for tag_list in tags]
    labels = [tag_list[1:-1] for tag_list in tags]
    fo_tags = [tag_list[2:] for tag_list in tags]
    prefix = encode_prefix(dataset, 6)
    
    data_list = []
    for i in range(sen_num):
        word_list = []
        for j in range(len(ct_idx[i])):
            word_list.append((ct_idx[i][j], pre_tags[i][j], labels[i][j], fo_tags[i][j], prefix[i][j,:]))
        data_list.append(word_list)
    return data_list


def dataform_option23(w2v, dataset, window_size = 1):
    sen_num = len(dataset)
    ct = add_context(dataset, window_size)
    ct_embedding = []
    for sen in ct:
        unit_embedding = []
        for word_list in sen:
            words = []
            for word in word_list:
                if word in w2v.vocab:
                    vec = w2v[word]
                else:
                    if word in ['START_TAG','END_TAG']:
                        vec = np.zeros(300,)
                    else:
                        vec = np.random.normal(0, 0.4, 300)
                    
                words.append(vec)
            # note: in testing some words are out of voc
            unit_embedding.append(np.concatenate(words))
        ct_embedding.append(unit_embedding)
        
    tags = make_labels(dataset)
    pre_tags = [tag_list[0:-2] for tag_list in tags]
    labels = [tag_list[1:-1] for tag_list in tags]
    fo_tags = [tag_list[2:] for tag_list in tags]
    prefix = encode_prefix(dataset, 6)
    
    data_list = []
    for i in range(sen_num):
        word_list = []
        for j in range(len(ct_embedding[i])):
            word_list.append((ct_embedding[i][j], pre_tags[i][j], labels[i][j], fo_tags[i][j], prefix[i][j,:]))
        data_list.append(word_list)
    return data_list


def train_regrouping(train_data):
    train_list = []
    for tp_list in train_data:
        train_list += [tp for tp in tp_list]
    return train_list
    

def batch_generate(train_set, batch_size):
    shuffled_index = np.random.permutation(len(train_set))   
    train_set = [train_set[i] for i in shuffled_index]
    
    total_batch = int(len(train_set)/batch_size)
    for i in range(total_batch):
        ct_data = []
        prefix = []
        pre_tag = []
        label = []
        for tp in train_set[i*batch_size : (i+1)*batch_size]:
            ct_data.append(tp[0])
            pre_tag.append(tp[1])
            label.append(tp[2])
            prefix.append(tp[4])
        yield ct_data, prefix, pre_tag, label

def accuracy(pred, gt):
    assert len(pred) == len(gt)
    true = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            true += 1
    acc = true / len(gt)
    return acc
    

def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {} 
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # tag sequence for opinion target extraction
            ote_tags = []
            # word sequence
            words = []
            # prefix category
            pref = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                pref.append(find_prefix(word))
                if find_prefix(word) == 2:
                    word = word[1:]
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')
                if tag == 'O':
                    ote_tags.append('O')
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ote_tags.append('T')
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['prefix'] = pref.copy()
            record['ote_raw_tags'] = ote_tags.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset


class RandInitEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, tag_embedding_dim, size_prefix, size_tag, context_size):
        super(RandInitEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.size_tag = size_tag
        self.tag_embeddings = nn.Embedding(size_tag + 1, tag_embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim + size_prefix + tag_embedding_dim, 120)
        self.linear2 = nn.Linear(120, size_tag)

    def forward(self, inputs, inputs_pref, prev_tag, batch_size):
        embeds = self.embeddings(inputs).view((batch_size, -1))
        tag_embeds = self.tag_embeddings(prev_tag).view((batch_size, -1))
        inputs_pref = inputs_pref.view((batch_size, -1))
        # print(embeds.size(), embeds.type(), tag_embeds.size(), tag_embeds.type(), inputs_pref.size(),inputs_pref.type())
        new_embeds = torch.cat((inputs_pref, embeds, tag_embeds), -1)
        out = F.relu(self.linear1(new_embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    
    def Viterbi_trans(self, inputs, inputs_pref, batch_size):
        transition = np.zeros((self.size_tag, self.size_tag))
        for pt in range(self.size_tag):
            pre_tag_idx = torch.tensor(pt, dtype=torch.long)
            log_probs = self.forward(inputs, inputs_pref, pre_tag_idx, batch_size)
            transition[pt,:] = log_probs.detach().numpy()
        return transition


class W2VEmbedding(nn.Module):
    def __init__(self, embedding_dim, tag_embedding_dim, size_prefix, size_tag, context_size):
        super(W2VEmbedding, self).__init__()
        self.size_tag = size_tag
        self.tag_embeddings = nn.Embedding(size_tag + 1, tag_embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim + size_prefix + tag_embedding_dim, 120)
        self.linear2 = nn.Linear(120, 30)
        self.linear3 = nn.Linear(30, size_tag)

    def forward(self, inputs, inputs_pref, prev_tag, batch_size):
        embeds = inputs.view((batch_size, -1))
        tag_embeds = self.tag_embeddings(prev_tag).view((batch_size, -1))
        inputs_pref = inputs_pref.view((batch_size, -1))
        new_embeds = torch.cat((inputs_pref, embeds, tag_embeds), -1)
        out = F.relu(self.linear1(new_embeds))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def Viterbi_trans(self, inputs, inputs_pref, batch_size):
        transition = np.zeros((self.size_tag, self.size_tag))
        for pt in range(self.size_tag):
            pre_tag_idx = torch.tensor(pt, dtype=torch.long).view((batch_size, -1))
            #print(inputs.size(), inputs.type(), pre_tag_idx.size(), pre_tag_idx.type(), inputs_pref.size(),inputs_pref.type())
            log_probs = self.forward(inputs, inputs_pref, pre_tag_idx, batch_size)
            transition[pt,:] = log_probs.detach().numpy()
        return transition


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, tag_embedding_dim, size_prefix, size_tag, context_size):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.tag_embeddings = nn.Embedding(size_tag + 1, tag_embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim + size_prefix + tag_embedding_dim, 120)
        self.linear2 = nn.Linear(120, size_tag)

    def forward(self, inputs, inputs_pref, prev_tag, batch_size):
        embeds = self.embeddings(inputs).view((batch_size, -1))
        tag_embeds = self.tag_embeddings(prev_tag).view((batch_size, -1))
        inputs_pref = inputs_pref.view((batch_size, -1))
        new_embeds = torch.cat((inputs_pref, embeds, tag_embeds), -1)
        out = F.relu(self.linear1(new_embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


train_path = r"D:\S\CS577\HW\HW2\data\twitter1_train.txt"
test_path = r"D:\S\CS577\HW\HW2\data\twitter1_test.txt"
train_set = read_data(train_path)
test_set = read_data(test_path)

shuffled_index = np.random.permutation(len(train_set))
train_set = [train_set[i] for i in shuffled_index]
train_ratio = 0.9
train_total = len(train_set)
train_num = int(train_ratio * train_total)
data4train = train_set[0:train_num]
data4validate = train_set[train_num:]
del train_set
window_size = 3

w2v = KeyedVectors.load_word2vec_format(datapath(r"D:\S\CS577\HW\HW2\w2v.bin"), binary=True)
train_data = dataform_option23(w2v, data4train, window_size)
train_data = train_regrouping(train_data)
del data4train
valid_data = dataform_option23(w2v, data4validate, window_size)
del data4validate
test_data = dataform_option23(w2v, test_set, window_size)
del test_set
del w2v
        

embedding_dim = 300
tag_embedding_dim = 10
batch_size = 64
total_epoch = 50
epoch_val = 1

train_losses = []
validation_losses = []
loss_function = nn.NLLLoss()
W2V = W2VEmbedding(embedding_dim, tag_embedding_dim, 
                        size_prefix = 6, size_tag = 4, context_size = (window_size*2+1))
optimizer = optim.Adam(W2V.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.005)

for epoch in range(total_epoch):
    train_total_loss = 0
    validation_total_loss = 0
    train_num = 0
    for data, prefix, pre_tag, label in batch_generate(train_data, batch_size):
        train_num += 1
        context_data = torch.tensor(data, dtype=torch.float)
        torch_prefix = torch.tensor(prefix, dtype=torch.float)
        pre_tag_idx = torch.tensor(pre_tag, dtype=torch.long)
        train_label = torch.tensor(label, dtype=torch.long)
        
        W2V.zero_grad()
        log_probs = W2V(context_data, torch_prefix, pre_tag_idx, batch_size)
        loss = loss_function(log_probs, train_label)
        loss.backward()
        optimizer.step()
        
        train_total_loss += loss.item()
    train_ave_loss = train_total_loss / train_num
    train_losses.append(train_ave_loss)
    print('Epoch:', epoch, 'Training average loss:', train_ave_loss)
    
    if epoch % epoch_val == 0 and len(valid_data) >= 1:
    # when testing, all training data is used to train so no need to validate
        val_num = 0
        val_len = len(valid_data)
        groud_truth = []
        prediction = []
        validation_total_loss = 0
        for v_sen in valid_data:
            gt = []
            backpoint = []
            viterbivars = np.zeros((1,4))
            for i in range(len(v_sen)):
                val_num += 1
                data, pre_tag, label, fo_tag, prefix = v_sen[i]
                gt.append(label)
                context_data = torch.tensor(data, dtype=torch.float)
                torch_prefix = torch.tensor(prefix, dtype=torch.float)
                pre_tag_idx = torch.tensor(pre_tag, dtype=torch.long)
                valid_label = torch.tensor([label], dtype=torch.long)
                if i == 0:
                    init_probs = W2V(context_data, torch_prefix, pre_tag_idx, 1)
                    viterbivars = init_probs.detach().numpy()
                    loss = loss_function(init_probs, valid_label)
                else:
                    gold_probs = W2V(context_data, torch_prefix, pre_tag_idx, 1)
                    loss = loss_function(gold_probs, valid_label)
                    transition = W2V.Viterbi_trans(context_data, torch_prefix, 1)
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning from tag i to next_tag.
                    # since we are using log prob here so we can add
                    next_tag_mat = viterbivars.reshape((4,1)) + np.asarray(transition)
                    viterbivars = np.max(next_tag_mat, axis=0)
                    backpoint.append(np.argmax(next_tag_mat, axis=0))

                validation_total_loss += loss.item()
            pred_last_tag = np.argmax(viterbivars)
            pred_tags = viterbi_decode(backpoint, pred_last_tag)
            prediction += pred_tags
            groud_truth += gt

        acc = accuracy(prediction, groud_truth)
        precision = sklearn.metrics.precision_score(groud_truth, prediction, [0,1,2],average='micro')
        recall = sklearn.metrics.recall_score(groud_truth, prediction, [0,1,2],average='micro')
        f1 = sklearn.metrics.f1_score(groud_truth, prediction, [0,1,2],average='micro')
        
        print('validation:', epoch, acc, precision, recall, f1)
        print('validation loss:',validation_total_loss / val_num)


test_num = 0
groud_truth = []
prediction = []
testing_total_loss = 0
for v_sen in test_data:
    gt = []
    backpoint = []
    viterbivars = np.zeros((1,4))
    for i in range(len(v_sen)):
        test_num += 1
        data, pre_tag, label, fo_tag, prefix = v_sen[i]
        gt.append(label)
        context_data = torch.tensor(data, dtype=torch.float)
        torch_prefix = torch.tensor(prefix, dtype=torch.float)
        pre_tag_idx = torch.tensor(pre_tag, dtype=torch.long)
        test_label = torch.tensor([label], dtype=torch.long)
        if i == 0:
            init_probs = W2V(context_data, torch_prefix, pre_tag_idx, 1)
            viterbivars = init_probs.detach().numpy()
            loss = loss_function(init_probs, test_label)
        else:
            gold_probs = W2V(context_data, torch_prefix, pre_tag_idx, 1)
            loss = loss_function(gold_probs, test_label)
            transition = W2V.Viterbi_trans(context_data, torch_prefix, 1)
            # next_tag_var[i] holds the viterbi variable for tag i at the
            # previous step, plus the score of transitioning from tag i to next_tag.
            # since we are using log prob here so we can add
            next_tag_mat = viterbivars.reshape((4,1)) + np.asarray(transition)
            viterbivars = np.max(next_tag_mat, axis=0)
            backpoint.append(np.argmax(next_tag_mat, axis=0))
            
        testing_total_loss += loss.item()
    pred_last_tag = np.argmax(viterbivars)
    pred_tags = viterbi_decode(backpoint, pred_last_tag)
    prediction += pred_tags
    groud_truth += gt

acc = accuracy(prediction, groud_truth)
precision = sklearn.metrics.precision_score(groud_truth, prediction, [0,1,2],average='micro')
recall = sklearn.metrics.recall_score(groud_truth, prediction, [0,1,2],average='micro')
f1 = sklearn.metrics.f1_score(groud_truth, prediction, [0,1,2],average='micro')


print("")
print('Testing Loss:', testing_total_loss / test_num)
print("Testing Accuracy: ", acc)
print("Testing Precision: ", precision)
print("Testing Recall: ", recall)
print("Testing F1: ", f1)
import csv
import spacy
import numpy as np
import copy
from sklearn.metrics import f1_score

'''
The code uses word2vec in spacy, please load the module by:
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
'''

def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r", encoding="utf8")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label

    #print("Read", row_count, "data points...")
    return tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label


def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):

    with open(output_csv_file, mode='w', encoding="utf8") as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])

def TopicSplitter(phrase):
    phrase_len = len(phrase)
    if phrase[0] == '@' and phrase_len > 1:
        return [phrase[0], phrase[1:]]
    
    result = [phrase[0]]
    i = 1
    while i < phrase_len:
        if phrase[i].isdigit():
            if not phrase[i-1].isdigit():
                result.append(phrase[i])
            else:
                result[-1] += phrase[i]
                
        elif phrase[i].isupper():
            if not phrase[i-1].isupper():
                result.append(phrase[i])
            elif i < phrase_len - 1 and phrase[i+1].islower():
                result.append(phrase[i])
            else:
                result[-1] += phrase[i]
                
        else:
            result[-1] += phrase[i]

        i += 1
    return result


def Textprocessing(dict_text, issue, author, md = 0):
    splitter = spacy.load('en_core_web_sm')
    embedding = spacy.load('en_core_web_lg')
    
    id2vector = {} 
    id2wordlist = {}
    max_dim = 2
    for i in dict_text.keys():
        element_splitter = []
        words = [issue[i], author[i]]
        sen = splitter(dict_text[i])
        for token in sen:
            element_splitter.append(token.text)
        for j in range(len(element_splitter)):
            if element_splitter[j][0] == '@' and len(element_splitter[j]) > 1:
                words.append('@')
                words.append(element_splitter[j][1:])
            elif j >= 1 and element_splitter[j-1] == '#':
                phrase = TopicSplitter(element_splitter[j])
                for w in phrase:
                    words.append(w)
            elif element_splitter[j].isalnum():
                words.append(element_splitter[j])
        if max_dim < len(words):
            max_dim = len(words)
        id2wordlist[i] = words
        
    for i in dict_text.keys():
        if md == 0:
            text_vector = np.zeros((max_dim, 300))
            dim = 0
            for word in id2wordlist[i]:
                text_vector[dim,:] = embedding.vocab[word].vector
                dim += 1
            id2vector[i] = text_vector.copy()
        else:
            text_vector = np.zeros((md, 300))
            dim = 0
            for word in id2wordlist[i]:
                if dim < md:
                    text_vector[dim,:] = embedding.vocab[word].vector
                    dim += 1
            id2vector[i] = text_vector.copy()           
    if md == 0:
        return id2vector, max_dim
    else:
        return id2vector


def relu(x):
    return x * (x > 0)
    

def softmax(x):
    e = np.exp(x)
    yhat = e.T / np.sum(e, axis=1)
    return yhat.T


def negative_log_likelihood(Y, prediction):
    nll = -np.sum(Y * np.log(prediction + 1e-6), axis = 1)
    return np.mean(nll)


def Measure(label_vec, yhat, num_class):
    label_vec = label_vec.astype(int)
    yhat = yhat.astype(int)
    confusion_matrix = np.zeros((num_class, num_class))
    accuracy = np.mean(label_vec == yhat)
    for i in range(label_vec.shape[0]):
        confusion_matrix[yhat[i], label_vec[i]] += 1
    # if some class is missing in the batch or not predicted, assign a 1 positive true to it
    cm_sum_0 = np.sum(confusion_matrix, axis = 0)
    zero_idx = np.where(cm_sum_0 == 0)
    confusion_matrix[zero_idx, zero_idx] = 1
    cm_sum_1 = np.sum(confusion_matrix, axis = 1)
    zero_idx = np.where(cm_sum_1 == 0)
    confusion_matrix[zero_idx, zero_idx] = 1
    
    precision_vec = confusion_matrix.diagonal() / np.sum(confusion_matrix, axis = 0)
    recall_vec = confusion_matrix.diagonal() / np.sum(confusion_matrix, axis = 1)
    F1_vec = 2 * precision_vec * recall_vec / (precision_vec + recall_vec)
    
    precision = np.mean(precision_vec)
    recall = np.mean(recall_vec)  
    F1 = np.mean(F1_vec)
    
    print(F1, f1_score(label_vec, yhat, average='macro'),'pre:', precision, 'rec:', recall)
    return accuracy, f1_score(label_vec, yhat, average='macro'), confusion_matrix


def Sampling(data_num, fold_num):
    assert fold_num > 1, "fold_num must be larger than 1"
    data_per_fold = int(data_num/fold_num)
    for i in range(fold_num - 1):
        train_list = []
        valid_list = []
        for j in range(data_num):
            valid_list.append(j) if i*data_per_fold <= j < (i+1)*data_per_fold else train_list.append(j)
        yield i+1, train_list, valid_list
    yield fold_num, [_ for _ in range(data_per_fold*(fold_num-1))], [_ for _ in range(data_per_fold*(fold_num-1), data_num)]


class LogisticReg():
    def __init__(self, input_dim, num_class):
        self.input_dim = input_dim
        self.num_class = num_class
        self.W = np.zeros((self.num_class, self.input_dim))
        self.b = np.zeros((1, self.num_class))

    def param_init(self):
        self.W = np.zeros((self.num_class, self.input_dim))
        self.b = np.zeros((1, self.num_class))
        
    def compute_gradient(self, X, Y, yhat):
        res = Y - yhat
        W_grad = np.dot(res.T, X)
        b_grad = np.mean(res, axis=0)
        return W_grad, b_grad
        
    def train_step(self, data, labels, lr = 0.001, L2_reg = 0):
        pred = softmax(np.dot(data, self.W.T) + self.b)
        # print('L2: 2', L2_reg)
        W_grad, b_grad = self.compute_gradient(data, labels, pred)
        self.W += lr * W_grad - lr * L2_reg * self.W / data.shape[0]
        self.b += lr * b_grad
        return negative_log_likelihood(labels, pred)
    
    def train(self, train_data, train_labels, val_data, val_labels, lr = 0.001, 
              L2_reg = 0, steps = 100, verbose = 10):
        data_num = train_data.shape[0]
        for step in range(steps):
            shuffled_index = np.random.permutation(data_num)
            train_data = train_data[shuffled_index]
            train_labels = train_labels[shuffled_index]
            nll = self.train_step(train_data, train_labels, lr = lr, L2_reg = L2_reg)
            loss = nll + L2_reg * np.linalg.norm(self.W)
            if step % verbose == 0:
                nll_val = self.validate(val_data, val_labels)
                loss_val = nll_val + L2_reg * np.linalg.norm(self.W)
                print('step:', step, 'training loss:', nll, loss, 'validation loss:', nll_val, loss_val)
        nll_val = self.validate(val_data, val_labels)
        loss_val = nll_val + L2_reg * np.linalg.norm(self.W)
        return loss, loss_val
    
    def predict(self, x_v):
        pred_onehot = softmax(np.dot(x_v, self.W.T) + self.b)
        pred_vec = np.argmax(pred_onehot, axis=1)
        return pred_vec
        
    def validate(self, data, labels):
        pred = softmax(np.dot(data, self.W.T) + self.b)
        nll = negative_log_likelihood(labels, pred)
        return nll


class NeuralNet():
    def __init__(self, input_dim, hidden_dim, num_class):
        assert len(hidden_dim) == 2
        assert len(input_dim) == 2
        self.word_num = input_dim[0]
        self.word_dim = input_dim[1]
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.W = []
        self.b = []
        self.W.append(np.random.normal(0.1, 0.1, size=(self.hidden_dim[0], self.word_dim)))
        self.b.append(np.random.normal(0.1, 0.1, size=(1, self.hidden_dim[0])))
        self.W.append(np.random.normal(0.1, 0.1, size=(self.hidden_dim[1], self.hidden_dim[0]*self.word_num)))
        self.b.append(np.random.normal(0.1, 0.1, size=(1, self.hidden_dim[1])))
        self.W.append(np.random.normal(0.1, 0.1, size=(self.num_class, self.hidden_dim[1])))
        self.b.append(np.random.normal(0.1, 0.1, size=(1, self.num_class)))
        self.W_grad = [np.zeros(i.shape) for i in self.W]
        self.b_grad = [np.zeros(i.shape) for i in self.b]
        
    def param_init(self):
        self.W[0] = np.random.normal(0, 0.1, size=self.W[0].shape)
        self.W[1] = np.random.normal(0, 0.1, size=self.W[1].shape)
        self.W[2] = np.random.normal(0, 0.1, size=self.W[2].shape)
        self.b[0] = np.random.normal(0, 0.1, size=self.b[0].shape)
        self.b[1] = np.random.normal(0, 0.1, size=self.b[1].shape)
        self.b[2] = np.random.normal(0, 0.1, size=self.b[2].shape)

    def backpropogate(self, X, Y, yhat, hidden_1, hidden_2):
        res = Y - yhat
        self.W_grad[2] = np.dot(res.T, hidden_2)
        self.b_grad[2] = np.mean(res, axis = 0)
        grad_H_2 = np.dot(res, self.W[2]) * (hidden_2 > 0)
        self.W_grad[1] = np.dot(grad_H_2.T, hidden_1.reshape((X.shape[0], -1)))
        self.b_grad[1] = np.mean(grad_H_2, axis = 0)
        #print('hidden shape:',hidden_1.shape, hidden_2.shape)
        grad_H_1 = np.dot(grad_H_2, self.W[1]).reshape(hidden_1.shape) * (hidden_1 > 0)
        self.W_grad[0] = np.dot(grad_H_1.T, X.reshape((-1, X.shape[2])))
        self.b_grad[0] = np.mean(grad_H_1, axis = 0)

    def forward(self, X):
        word_em = relu(np.dot(X, self.W[0].T) + self.b[0])
        word_em_bp = np.reshape(word_em, (-1, word_em.shape[2]), order='C')
        word_em_fw = np.reshape(word_em, (word_em.shape[0], -1), order='C')
        sen_em = relu(np.dot(word_em_fw, self.W[1].T) + self.b[1])
        pred_onehot = softmax(np.dot(sen_em, self.W[2].T) + self.b[2])
        return pred_onehot, word_em_bp, sen_em
        
    def predict(self, x_v):
        word_em = relu(np.dot(x_v, self.W[0].T) + self.b[0])
        word_em_fw = np.reshape(word_em, (word_em.shape[0], -1), order='C')
        sen_em = relu(np.dot(word_em_fw, self.W[1].T) + self.b[1])
        pred_onehot = softmax(np.dot(sen_em, self.W[2].T) + self.b[2])
        pred_vec = np.argmax(pred_onehot, axis=1)
        return pred_vec

    def validate(self, data, labels):
        word_em = relu(np.dot(data, self.W[0].T) + self.b[0])
        word_em_fw = np.reshape(word_em, (word_em.shape[0], -1), order='C')
        sen_em = relu(np.dot(word_em_fw, self.W[1].T) + self.b[1])
        pred_onehot = softmax(np.dot(sen_em, self.W[2].T) + self.b[2])
        nll = negative_log_likelihood(labels, pred_onehot)
        return nll
    
    def train_step(self, data, labels, lr = 0.001, L2_reg = 0):
        pred_onehot, hidden_1, hidden_2 = self.forward(data)
        self.backpropogate(data, labels, pred_onehot, hidden_1, hidden_2)
        for i in range(len(self.W)):
            #print("i=",i, self.W[i].shape, self.W_grad[i].shape)
            self.W[i] += lr * self.W_grad[i] - lr * L2_reg * self.W[i] / data.shape[0]
            self.b[i] += lr * self.b_grad[i]
        nll = negative_log_likelihood(labels, pred_onehot)
        return nll
    
    def train(self, train_data, train_labels, val_data, val_labels, batch_size = 64,
              lr = 0.001, L2_reg = 0, steps = 100, verbose = 10):
        data_num = train_data.shape[0]
        for step in range(steps):
            shuffled_index = np.random.permutation(data_num)
            train_data = train_data[shuffled_index]
            train_labels = train_labels[shuffled_index]

            for k in range(0, train_data.shape[0], batch_size):
                batch_data = train_data[k:k+batch_size]
                batch_label = train_labels[k:k+batch_size]
                nll = self.train_step(batch_data, batch_label, lr = lr, L2_reg = L2_reg)
                
            if step % verbose == 0:
                nll = self.validate(train_data, train_labels)
                loss = nll + L2_reg * sum([np.linalg.norm(w) for w in self.W])
                nll_val = self.validate(val_data, val_labels)
                print('step:', step, 'training nll:', nll, 'training loss:', loss,'validation nll:', nll_val)
        nll_val = self.validate(val_data, val_labels)
        return nll, nll_val
    

def LR():

    # Read training data
    id2text, id2issue, id2author_label, id2label = ReadFile('train.csv')
    # prepare the dataset
    id2vector, max_dim = Textprocessing(id2text, id2issue, id2author_label)
    num_class = 17
    
    data_num = len(id2text)
    text_vec = np.zeros((data_num, max_dim * 300))
    label_vec = np.zeros((data_num,))
    label_onehot = np.zeros((data_num, num_class))
    idx_vec = np.zeros((data_num,))
    
    for (i, idx) in zip(range(data_num), id2vector.keys()):
        text_vec[i,:] = id2vector[idx].flatten(order='C')
        label_vec[i] = int(id2label[idx]) - 1
        idx_vec[i] = idx
    
    shuffled_index = np.random.permutation(data_num)
    text_vec = text_vec[shuffled_index, :]
    label_vec = label_vec[shuffled_index]
    idx_vec = idx_vec[shuffled_index]
    
    label_onehot[np.arange(data_num), label_vec.astype(int)] = 1

    training_steps = 1500
    verbose = 300
    '''
    Implement your Logistic Regression classifier here
    Following code implemented cross validation
    
    fold_num = 5
    lr_list = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
    L2_list = [0.1, 0.5, 2, 10, 50]   
    
    F1_mat = np.zeros((len(lr_list), len(L2_list)))
    max_F1 = 0
    best_lr = 0
    best_L2 = 0
    
    for (i, learning_rate) in zip([_ for _ in range(len(lr_list))], lr_list):
        for (j, L2_reg) in zip([_ for _ in range(len(L2_list))], L2_list):
            LR = LogisticReg(input_dim = text_vec.shape[1], num_class = num_class)
            accu_list = []
            F1_list = []
            for fold_idx, train_list, val_list in Sampling(data_num, fold_num):
                print('fold', fold_idx, ':')
                train_data = text_vec[train_list]
                train_label = label_onehot[train_list]
                train_label_vec = label_vec[train_list]
                val_data = text_vec[val_list]
                val_label = label_onehot[val_list]
                val_label_vec = label_vec[val_list]
                LR.param_init()
                loss_train, loss_val = LR.train(train_data, train_label, val_data, val_label,
                                                lr = learning_rate, L2_reg = L2_reg, steps = training_steps, verbose = verbose)
                
                train_yhat_vec = LR.predict(train_data)
                val_yhat_vec = LR.predict(val_data)
                train_acc, train_F1, train_cm = Measure(train_label_vec, train_yhat_vec, num_class)
                val_acc, val_F1, val_cm = Measure(val_label_vec, val_yhat_vec, num_class)
                accu_list.append(val_acc)
                F1_list.append(val_F1)
                print("Result:")
                print('training_accuracy:', train_acc, 'validation accuracy:', val_acc)
                print('training F1 score:', train_F1, 'validation F1 score', val_F1)
            avg_acc = sum(accu_list) / fold_num
            avg_F1 = sum(F1_list) / fold_num
            print('lr =', learning_rate, 'regularization:', L2_reg, 'Macro F1:', avg_F1, 'Accuracy:', avg_acc)
            F1_mat[i, j] = avg_F1
            if avg_F1 > max_F1:
                max_F1 = val_F1
                best_lr = learning_rate
                best_L2 = L2_reg
                best_LR = copy.deepcopy(LR)
    '''
    # when we find the best parameter, retrain the module with all data including validation data
    best_lr = 1e-4
    best_L2 = 0.5
    best_LR = LogisticReg(input_dim = text_vec.shape[1], num_class = num_class)

    loss_train, loss_val = best_LR.train(text_vec, label_onehot, text_vec, label_onehot,
                                    lr = best_lr, L2_reg = best_L2, steps = training_steps, verbose = verbose)
    
    train_yhat_vec = best_LR.predict(text_vec)
    train_acc, train_F1, train_cm = Measure(label_vec, train_yhat_vec, num_class)

    print("Result:")
    print('training_accuracy:', train_acc)
    print('training F1 score:', train_F1)   

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    test_id2vector = Textprocessing(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, max_dim)
    
    test_data_num = len(test_tweet_id2text)
    test_text_vec = np.zeros((test_data_num, max_dim * 300))
    test_idx_vec = np.zeros((test_data_num,))
    
    for (i, idx) in zip(range(test_data_num), test_id2vector.keys()):
        test_text_vec[i,:] = test_id2vector[idx].flatten(order='C')
        test_idx_vec[i] = idx
    # Predict test data by learned model

    '''
    Replace the following random predictor by your prediction function.
    '''
    pred_label_test = best_LR.predict(test_text_vec) + 1
    for i in range(test_data_num):
        # Predict the label
        tweet_id = test_idx_vec[i]

        # Store it in the dictionary
        test_tweet_id2label[tweet_id] = str(pred_label_test[i])
        print(tweet_id, test_tweet_id2label[tweet_id])
    # Save predicted labels in 'test_lr.csv'
    return (test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label)
    

def NN():

    # Read training data
    id2text, id2issue, id2author_label, id2label = ReadFile('train.csv')
    id2vector, max_dim = Textprocessing(id2text, id2issue, id2author_label)
    num_class = 17
    
    data_num = len(id2text)
    text_vec = np.zeros((data_num, max_dim, 300))
    label_vec = np.zeros((data_num,))
    label_onehot = np.zeros((data_num, num_class))
    idx_vec = np.zeros((data_num,))

    
    for (i, idx) in zip(range(data_num), id2vector.keys()):
        text_vec[i] = id2vector[idx]
        label_vec[i] = int(id2label[idx]) - 1
        idx_vec[i] = idx
    
    shuffled_index = np.random.permutation(data_num)
    text_vec = text_vec[shuffled_index]
    label_vec = label_vec[shuffled_index]
    idx_vec = idx_vec[shuffled_index]
    
    label_onehot[np.arange(data_num), label_vec.astype(int)] = 1
    
    learning_rate = 0.00002
    L2_reg = 0.1
    training_steps = 1000
    verbose = 50
    batch_size = 32
    
    '''
    Implement your Neural Network classifier here

    dim_list = [20, 40, 60, 80, 160, 200]
    fold_num = 5
    L2_reg = 0.1
    F1_list = []
    best_F1 = 0
    best_hd = 0
    
    for d in dim_list:
        NN = NeuralNet(input_dim = (text_vec.shape[1], text_vec.shape[2]), hidden_dim=(30, d), num_class = num_class)
        F1_fold = []
        for fold_idx, train_list, val_list in Sampling(data_num, fold_num):
            print('fold', fold_idx, ':')
            train_data = text_vec[train_list]
            train_label = label_onehot[train_list]
            train_label_vec = label_vec[train_list]
            val_data = text_vec[val_list]
            val_label = label_onehot[val_list]
            val_label_vec = label_vec[val_list]
            NN.param_init()
            loss_train, loss_val = NN.train(train_data, train_label, val_data, val_label, batch_size = batch_size,
                                            lr = learning_rate, L2_reg = L2_reg, steps = training_steps, verbose = verbose)
            
            train_yhat_vec = NN.predict(train_data)
            val_yhat_vec = NN.predict(val_data)
            train_acc, train_F1, train_cm = Measure(train_label_vec, train_yhat_vec, num_class)
            val_acc, val_F1, val_cm = Measure(val_label_vec, val_yhat_vec, num_class)
            
            print("Result for hidden_dim =:", d)
            print('training_accuracy:', train_acc, 'validation accuracy:', val_acc)
            print('training F1 score:', train_F1, 'validation F1 score', val_F1)
            F1_fold.append(val_F1)
        F1_avg = sum(F1_fold)/fold_num
        F1_list.append(F1_avg)
        if F1_avg > best_F1:
            best_F1 = F1_avg
            best_NN = copy.deepcopy(NN)
            best_hd = d
    '''
    best_hd = 60
    best_NN = NeuralNet(input_dim = (text_vec.shape[1], text_vec.shape[2]), hidden_dim=(30, best_hd), num_class = num_class)
    
    loss_train, loss_val = best_NN.train(text_vec, label_onehot, text_vec, label_onehot, batch_size = batch_size,
                                    lr = learning_rate, L2_reg = L2_reg, steps = training_steps, verbose = verbose)


    train_yhat_vec = best_NN.predict(text_vec)
    train_acc, train_F1, train_cm = Measure(label_vec, train_yhat_vec, num_class)

    print("Result:")
    print('training_accuracy:', train_acc)
    print('training F1 score:', train_F1)   

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    test_id2vector = Textprocessing(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, max_dim)
    
    test_data_num = len(test_tweet_id2text)
    test_text_vec = np.zeros((test_data_num, max_dim, 300))
    test_idx_vec = np.zeros((test_data_num,))
    
    for (i, idx) in zip(range(test_data_num), test_id2vector.keys()):
        test_text_vec[i] = test_id2vector[idx]
        test_idx_vec[i] = idx
        
    # Predict test data by learned model
    # Replace the following random predictor by your prediction function
    
    pred_label_test = best_NN.predict(test_text_vec) + 1
    for i in range(test_data_num):
        # Predict the label
        tweet_id = test_idx_vec[i]

        # Store it in the dictionary
        test_tweet_id2label[tweet_id] = str(pred_label_test[i])
        print(tweet_id, test_tweet_id2label[tweet_id])

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_nn.csv')

if __name__ == '__main__':
    #(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label) = LR()
    #SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_lr.csv')
    NN()
    
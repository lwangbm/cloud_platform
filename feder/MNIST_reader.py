import collections
import os
import pickle
import struct

import nltk
import numpy as np

nltk.download('punkt')
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    print(fname_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


    # Reshape and normalize

    img = np.reshape(img, [img.shape[0], img.shape[1]*img.shape[2]])*1.0/255.0

    return img, lbl


def get_data(d):
    # load the data
    x_train, y_train = read('training', d + '/MNIST_original')
    x_test, y_test = read('testing', d + '/MNIST_original')

    # create validation set
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # create test_set
    x_train = x_train[:50000].astype(float)
    y_train = y_train[:50000].astype(float)

    # sort test set (to make federated learning non i.i.d.)
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train])
    sorted_y_train = list(y_train[indices_train])

    # create a test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test


class Data:
    def __init__(self, save_dir, n):
        raw_directory = save_dir + '/DATA'
        self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)

# Word embedding
def word_embedding(words):
    vocabulary = collections.Counter(words).most_common()
    vocabulary_dictionary = dict()
    for word, _ in vocabulary:
        # Assign a numerical unique value to each word inside vocabulary
        vocabulary_dictionary[word] = len(vocabulary_dictionary)
    rev_vocabulary_dictionary = dict(zip(vocabulary_dictionary.values(), vocabulary_dictionary.keys()))
    return vocabulary_dictionary, rev_vocabulary_dictionary


# Build Training data. For example if X = ['long', 'ago', ','] then Y = ['the']
def sampling(words, vocabulary_dictionary, window):
    X = []
    Y = []
    sample = []
    for index in range(0, len(words) - window):
        for i in range(0, window):
            sample.append(vocabulary_dictionary[words[index + i]])
            if (i + 1) % window == 0:
                X.append(sample)
                Y.append(vocabulary_dictionary[words[index + i + 1]])
                sample = []
    return X,Y

class Data_word:
    def __init__(self, save_dir, nc, window,b):
        raw_directory = save_dir + '/Data_word/'
        # self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        # self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)

        with open(raw_directory+"data.txt") as f:
            content = f.read()
        words = nltk.tokenize.word_tokenize(content.decode('utf-8'))
        vocabulary_dictionary, reverse_vocabulary_dictionary = word_embedding(words)
        training_data, label = sampling(words, vocabulary_dictionary, window)
        NN=label.__len__()
        nn=int(NN/3)

        self.sorted_x_train=training_data[:nn]
        self.sorted_y_train=label[:nn]

        self.x_vali = training_data[nn:2*nn]
        self.y_vali = label[nn:2*nn]

        self.x_test = training_data[2*nn:]
        self.y_test = label[2 * nn:]
        self.vocabulary_dictionary=vocabulary_dictionary
        self.reverse_vocabulary_dictionary=reverse_vocabulary_dictionary
        self.client_set=[]

        mc=int (nn/nc)
        mc=int(mc/b)*b
        for i in range(nc):
            self.client_set.append(range(mc*i,mc*(i+1)))



        aa=1;

class Data_word_2:
    def __init__(self, save_dir, nc, window,b):
        raw_directory = save_dir+"/.."
        # self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        # self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)

        with open("/home/luping/work/fede_imp/zebra/feder/DATAWORD_10TO100.txt") as f:
            content = f.read()
        words = nltk.tokenize.word_tokenize(content.decode('utf-8'))
        vocabulary_dictionary, reverse_vocabulary_dictionary = word_embedding(words)
        training_data, label = sampling(words, vocabulary_dictionary, window)
        NN=label.__len__()
        nn=int(NN/3)

        self.sorted_x_train=[]
        self.sorted_y_train=[]

        self.x_vali = training_data[nn:2*nn]
        self.y_vali = label[nn:2*nn]

        self.x_test = training_data[2*nn:]
        self.y_test = label[2 * nn:]
        self.vocabulary_dictionary=vocabulary_dictionary
        self.reverse_vocabulary_dictionary=reverse_vocabulary_dictionary
        self.client_set=[]

        mc=int (nn/nc)
        mc=int(mc/b)*b
        index_mc=0
        for i in range(nc):
            fileN = '/home/luping/work/fede_imp/zebra'+'/feder/DATA_W_10TO100' + '/' + str(i) + '.txt'
            with open(fileN) as f:
                content = f.read()
            words = nltk.tokenize.word_tokenize(content.decode('utf-8'))
            vocabulary_dictionary=self.vocabulary_dictionary
            reverse_vocabulary_dictionary = self.reverse_vocabulary_dictionary
            training_data, label = sampling(words, vocabulary_dictionary, window)
            mc = label.__len__()
            mc = int(mc / b) * b

            self.client_set.append(range(index_mc,index_mc+mc))
            index_mc=index_mc+mc
            self.sorted_x_train=self.sorted_x_train+training_data
            self.sorted_y_train=self.sorted_y_train+label
            zzz=1
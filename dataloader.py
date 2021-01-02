import json
import os
import numpy as np
import unicodedata
import string
import re
import torch
from torch.utils import data
from torch.utils.data import DataLoader

def readJson(root, mode):
    path = os.path.join(root, mode + '.json')
    with open(path, 'r') as reader:
        data_ = json.load(reader)

    data = []
    labels = []
    
    for i in range(len(data_)):
        for inputs in data_[i]['input']:
            data.append(inputs)
            labels.append(data_[i]['target'])

    return data, labels

class JsonDataloader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.data, self.labels = readJson(self.root, self.mode)
        self.chardict = CharDict()

    # Return the length of the word
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.chardict.StringToLongtensor(self.data[index]), self.chardict.StringToLongtensor(self.labels[index])

class CharDict:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0
        
        for i in range(26):
            self.addWord(chr(ord('a') + i))

        tokens = ["SOS", "EOS"]
        for t in tokens:
            self.addWord(t)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def StringToLongtensor(self, s):
        s = ["SOS"] + list(s) + ["EOS"]
        return torch.LongTensor([self.word2index[char] for char in s])

    def LongtensorToString(self, l, show_token=False, check_end=True):
        s = ""
        for i in l:
            ch = self.index2word[i.item()]
            if len(ch) > 1:
                if show_token:
                    __ch = "<{}>".format(ch)
                else:
                    __ch = ""
            else:
                __ch = ch
            s += __ch
            if check_end and ch == "EOS":
                break
        return s
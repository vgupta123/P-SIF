import os, pdb
import pandas as pd
import nltk.data
import pickle
import logging
import numpy as np
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import time
from sklearn.preprocessing import normalize
import sys
import csv
stop_words_list = open("stop_words.txt","r").read().split("\n\n")

folders_sport =  [folder for folder in os.listdir("classic")]


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
data_list = []
for each_file_path in folders_sport:
    each_file = os.path.join("classic",each_file_path)
    data_list.append([open(each_file,"r").read(),each_file_path.split(".")[0]])
#pdb.set_trace()

all_sen = []
data_list_process = []
for each_data in data_list:
    data_list_process.append([" ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True)),each_data[1]])
    all_sen.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True)))


pickle.dump(set(" ".join(all_sen).split()),open("word_set.pkl","w"))

pickle.dump(data_list_process,open("data_all.pkl","w"))
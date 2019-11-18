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

#folders_sport =  [folder for folder in os.listdir("bbcsport") if os.path.isdir(os.path.join("bbcsport",folder))]


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

text_file = open("all_twitter_by_line.txt","r").readlines()
data_list = []
for each_doc in text_file:
        class_doc = each_doc.split("\t")[0]
        doc = " ".join(each_doc.split("\t")[1:])
        data_list.append([doc[:-1], class_doc])


amazon_file = open("twitter_text.txt","w")
for each_data in data_list:
    text = " ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True))
    amazon_file.write(text+"\n")
pdb.set_trace()
all_sen = []
data_list_process = []
for each_data in data_list:
    data_list_process.append([" ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True)),each_data[1]])
    all_sen.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True)))


pickle.dump(set(" ".join(all_sen).split()),open("word_set.pkl","w"))

pickle.dump(data_list_process,open("data_all.pkl","w"))


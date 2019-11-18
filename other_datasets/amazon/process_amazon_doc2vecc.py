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
#stop_words_list = open("stop_words.txt","r").read().split("\n\n")



folders_sport =  [folder for folder in os.listdir("sorted_data_acl") if os.path.isdir(os.path.join("sorted_data_acl",folder))]

def proess_amazon(each_file, class_name,data_list):
    docs = open(each_file, "r").read().split("<review>")
    for each_doc in docs:
        if(len(each_doc)>10):
            doc_text = each_doc[(each_doc.find("<review_text>")+len("<review_text>\n")):each_doc.find("</review_text>")]
            data_list.append([doc_text,class_name])
    return data_list

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
data_list = []
for each_folder in folders_sport:
    file_path = os.path.join("sorted_data_acl", each_folder)
    for each_file_path in os.listdir(file_path):
        if("unlabeled" not in each_file_path):
            each_file = os.path.join(file_path,each_file_path)
            data_list = proess_amazon(each_file, each_folder,data_list)


amazon_file = open("amazon_text.txt","w")
for each_data in data_list:
    text = " ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True))
    amazon_file.write(text+"\n")
#pdb.set_trace()
print len(data_list)
all_sen = []
data_list_process = []
for each_data in data_list:
    data_list_process.append([" ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True)),each_data[1]])
    all_sen.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True)))



pickle.dump(set(" ".join(all_sen).split()),open("word_set.pkl","w"))
pickle.dump(data_list_process,open("data_all.pkl","w"))
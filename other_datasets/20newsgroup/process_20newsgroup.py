import pickle
import gensim, pdb
from KaggleWord2VecUtility import KaggleWord2VecUtility
import numpy as np
#words_set = pickle.load(open("word_set.pkl","r"))

words_set = []
both_files = ["20ng-train-no-stop.txt","20ng-test-no-stop.txt"]
for each_file in both_files:
    f = open(each_file,"r").readlines()
    for line in f:
        each_class, doc = line.split("\t")
        words_set.extend(doc[:-1].split())

words_set = set(words_set)
train_data, test_data, Y_train, Y_test = [],[],[],[]

fil = open("20ng-train-no-stop.txt", "r").readlines()
for line in fil:
    each_class, doc = line.split("\t")
    Y_train.append(each_class)
    train_data.append(doc[:-1])

fil = open("20ng-test-no-stop.txt", "r").readlines()
for line in fil:
    each_class, doc = line.split("\t")
    Y_test.append(each_class)
    test_data.append(doc[:-1])

amazon_file = open("20newsgroup_text.txt","w")
for each_data in train_data+test_data:
    #pdb.set_trace()
    #text = " ".join(KaggleWord2VecUtility.review_to_wordlist(each_data[0], True))
    amazon_file.write(each_data+"\n")
#model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

#pdb.set_trace()

# model_part = {}
# for word in words_set:
#     if(word in model):
#         model_part[word] = model[word]
#
#
# print len(model_part)
#
# pickle.dump(model_part,open("word_embedding_dict.pkl","w"))
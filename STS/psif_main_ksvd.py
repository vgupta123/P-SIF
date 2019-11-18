import pickle, sys
import os,pdb
import numpy as np
import pandas as pd
#sys.path.append('../src')
import data_io, sim_algo, eval, params, time
from joblib import Parallel, delayed

np.set_printoptions(threshold=np.inf)
params = params.params()

wordfiles = ['data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
    'data/glove.6B.300d.txt'  # need to download it first
    ]

parameters = pd.read_csv("parameters_ksvd.csv",delimiter=",")
#cluster_list = [22,25,27,29]


filename = "Scdv_ksvd_output.txt"
num_features = 300
cluster_list = list(range(2, 4))

files_list = []
for index_count in range(12, 17):
    for i in range(index_count, index_count + 1):
        folder1 = "SentEval/data/senteval_data/STS/STS" + str(i) + "-en-test"
        # print(folder1)
        for filei in os.listdir(folder1):
            if ("input" in filei):
                input_file_split = filei.split(".")
                filej = ".".join([input_file_split[0], "gs"] + input_file_split[2:])
                if (filej in os.listdir(folder1)):
                    files_list.append([os.path.join(folder1, filei), os.path.join(folder1, filej)])

#pdb.set_trace()
def file_processing(files_list):
  word_list = []
  for file_index in files_list:
    f = open(file_index[0], 'r')
    # print(f)
    line = f.readlines()
    lines = [lin for lin in line]
    f = open(file_index[1], 'r')
    # print(f)
    score_line = f.readlines()
    score_lines = [score for score in score_line]
    golds = []
    seq1 = []
    seq2 = []
    word_list_ind = []
    for index in range(len(lines)):
        i = lines[index]
        j = score_lines[index]
        i = i.split("\t")
        # print(i)
        p1 = i[0].lower();
        p2 = i[1].lower();
        # print(j)
        try:
            score = float(j)
            word_list.extend(p1.split() + p2.split())
            word_list_ind.extend(p1.split() + p2.split())
        except:
            pass
  list_words_set = list((set(word_list)))
  return list_words_set


list_words_set = file_processing(files_list)

for index,row in parameters.iterrows():
    file_task = row['task']
    weightpara = row['weight_parameter']
    rmpc = row['principal component removed']
    transform_n_nonzero_coefs = int(row['non zero coefs'])
    n_clusteri = int(row['ksvd clusters'])
    time1 = time.time()
    textfile = 'data/paragram_sl999_small.txt'
    folder_name = "data_files"
    weightfile = 'data/enwiki_vocab_min200.txt'
    prob_wordvecs_name = "prob_wordvecs_ksvd_transform_n_nonzero_coefs_" + str(transform_n_nonzero_coefs) + \
                             textfile.split("/")[-1] + "_all_words_" + str(n_clusteri) + ".pkl"
    prob_wordvecs = pickle.load(open(os.path.join(folder_name,prob_wordvecs_name),"rb"))
    wordfile = prob_wordvecs
    (words, We) = data_io.getWordmap_ksvd(wordfile, list_words_set)
    all_s= []
    word2weight = data_io.getWordWeight(weightfile, weightpara)
    weight4ind = data_io.getWeight(words, word2weight)
    params.rmpc = rmpc
    for file_index in files_list:
        if(file_index[0]==file_task):
            each_file = file_index
    print_line = prob_wordvecs_name+" "+weightfile+" "+str(weightpara)+" "+str(rmpc)
    simi = eval.sim_evaluate_one1(print_line,each_file,We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
    print simi


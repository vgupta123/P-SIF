import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility
import operator

def weight_file_processing(all):
	traindata = []
	for i in range(0, len(all["text"])):
		traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["text"][i], True)))
	all_wordlist = " ".join(traindata).split()
	counts = dict()
	for i in all_wordlist:
		counts[i] = counts.get(i, 0) + 1
	sorted_x = sorted(counts.items(), key=operator.itemgetter(1),reverse=True)
	weight_file_build = open("data/reuters_vocab.txt","ab")
	for each in sorted_x:
		weight_file_build.write(each[0]+" "+str(each[1])+" "+"\n")


all = pd.read_pickle('all.pkl')
weight_file_processing(all)
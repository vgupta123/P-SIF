import pickle
import gensim,pdb
from KaggleWord2VecUtility import KaggleWord2VecUtility
#words_set = pickle.load(open("word_set.pkl","r"))

words_set = []
both_files = ["r8-train-no-stop.txt","r8-test-no-stop.txt"]
amazon_file = open("reuters_text.txt","w")
for each_file in both_files:
    f = open(each_file,"r").readlines()
    for line in f:
        each_class, doc = line.split("\t")
        words_set.extend(doc[:-1].split())
        text = " ".join(KaggleWord2VecUtility.review_to_wordlist(doc[:-1], True))
        amazon_file.write(text + "\n")



words_set = set(words_set)



model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

#pdb.set_trace()

model_part = {}
for word in words_set:
    if(word in model):
        model_part[word] = model[word]


print len(model_part)

pickle.dump(model_part,open("word_embedding_dict.pkl","w"))
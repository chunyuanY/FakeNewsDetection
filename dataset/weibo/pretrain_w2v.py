import json
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import jieba
import pickle
import time

start = time.time()

jieba.set_dictionary('dict.txt.big')

path = "./Weibo/"
files= os.listdir(path)
output_path = "./alltext.txt"

filename = []
with open(output_path, 'w', encoding='utf-8') as output:
	for file in files:
		with open(path + file, 'r', encoding='utf-8') as input:
			s = "".join(input.readlines())
			s = json.loads(s)
			for item in s:
				if not item['text'].startswith("转发微博") and not item['text'].startswith("轉發微博"):
					words = list(jieba.cut(item['text']))
					if len(words) > 1:
						output.write(" ".join(words) + "\n")


output_w2v = "./weibo16_w2v.bin"
vocab_path = "./vocab.pkl"
sentences = LineSentence(output_path)
model = Word2Vec(sentences, size=300, window=10, min_count=5, sg=1, workers=multiprocessing.cpu_count())
model.wv.save_word2vec_format(fname=output_w2v, binary=True)
pickle.dump(model.wv.vocab, file=open(vocab_path, 'wb'))
print("vocab size:", len(model.wv.vocab))
os.remove(output_path)
print("use time: ", (time.time()-start)/60 , "min")



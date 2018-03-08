import pandas as pd
import re
from nltk import ngrams
from gensim.models import Word2Vec
import logging

def transform_row(TXT):
    TXT = TXT.replace(",", " ").replace(".", " ").replace(";", " ").replace(":", " ").replace("[", " ").replace("]", " ").replace('"', " ").replace("'", " ").replace("!", " ").replace("?", " ")
    TXT = TXT.strip()
    return TXT 

def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]

# Doc du lieu vao dataFrames
df = pd.read_csv("TXT.csv").dropna()

# Lam sach du lieu
df["TXT"] = df.TXT.apply(transform_row)

# Chia du lieu thanh n-grams
df["1gram"] = df.TXT.apply(lambda t: kieu_ngram(t, 1))
df["2gram"] = df.TXT.apply(lambda t: kieu_ngram(t, 2))

# Add n-grams vao list
df["context"] = df["1gram"] + df["2gram"]
train_data = df.context.tolist()

# Tinh ty le xuat hien cac tu
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
model = Word2Vec(train_data, size=100, window=10, min_count=3, workers=4, sg=1)


print(model.wv.similar_by_word("good contributor"))




import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Lam sach du lieu text
def clean_text(df, col):
    return df[col].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                  .apply(lambda x: re.sub('\s+', ' ', x).strip())


# Loai bo Stopwords
def normalize(tokenized_words):
    stop_words = stopwords.words('english')
    return [[w.lower() for w in sent
             if (w.lower() not in stop_words)]
            for sent in tokenized_words]

# Tach tu
def split_on_word(text):
    if type(text) is list:
        return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*") for sentence in text]
    else:
        return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")

# Dem so pattern xuat hien trong cot TXT
def count_pattern(df, col, pattern):
    df = df.copy()
    return df[col].str.count(pattern)

#
def flatten_words(list1d, get_unique=False):
    qa = [s.split() for s in list1d]
    if get_unique:
        return sorted(list(set([w for sent in qa for w in sent])))
    else:
        return [w for sent in qa for w in sent]


# Load dataset
data = pd.read_csv('test_dataset.csv')
data['TXT_clean'] = clean_text(data, 'TXT')

#  FEATURE EXTECTION FROM TEXT: Dem tan so xuat hien cua tu

# 1. TF-IDF: DO DO TUONG TU GIUA CAC VAN BAN (Term-frquency)
# Dac diem: gia tri cac thanh phan cua vector duoc tinh bang: term frequency * (1 / document fraquency)

# Gan cac gia tri cot TXT trong "TXT_train.csv" va "TXT_test.csv" vao list
all_text = data['TXT_clean'].values.tolist()


#chuyen hoa van ban thanh vector
vocab = flatten_words(all_text, get_unique=True)
tf_idf = TfidfVectorizer(analyzer='word',min_df=1, ngram_range=(1,1),stop_words='english', vocabulary=vocab)
X = tf_idf.fit_transform(data['TXT_clean'].values)

# Split the data data
train, test = cross_validation.train_test_split(data, test_size = 0.1, random_state = 150)
num_train = len(train) 

X_train = X[:num_train]
X_test = X[num_train:]

y_train = data["Class"].values[:num_train]
y_test = data["Class"].values[num_train:]

# train classifier
svm = SVC(probability=True, kernel='rbf')
svm.fit(X_train, y_train)

# predict and evaluate predictions
predictions = svm.predict(X_test)
print accuracy_score(y_test, predictions)
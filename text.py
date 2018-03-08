import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class NLP(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe

    #Loai bo dau cau ".", "," ...
    def clean_text(self, dataframe, col):
        return dataframe[col].fillna('').apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                    .apply(lambda x: re.sub('\s+', ' ', x).strip())

    # Loai bo Stop word trong tieng anh: nhung tu hay xuat hien nhung khong co
    # nhieu y nghia nhu: of, a, an, the...
    def remove_stopwords(self, tokenized_words):
        self.stop_words = stopwords.words('english')
        return [[w.lower() for w in sent
                if (w.lower() not in stop_words)]
                for sent in tokenized_words]

    # Dem so lan xuat hien cua pattern
    def count_pattern(self, dataframe, col, pattern):
        dataframe = dataframe.copy()
        return dataframe[col].str.count(pattern)

    def split_on_word(self, text):
        if type(text) is list:
            return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*") for sentence in text]
        else:
            return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")

    def flatten_words(self, list1d, get_unique=False):
        qa = [s.split() for s in list1d]
        if get_unique:
            return sorted(list(set([w for sent in qa for w in sent])))
        else:
            return [w for sent in qa for w in sent]

    
dataframe = pd.read_csv("TXT.csv")
nlp = NLP(dataframe)

# Lam sach cot "TXT"
text = nlp.clean_text(dataframe, 'TXT')

# Add text vao list
text_list = text.values.tolist()

# Vector hoa
vocab = nlp.flatten_words(text_list, get_unique=True)
feature_extraction = TfidfVectorizer(analyzer='word',min_df=1, ngram_range=(1,2),stop_words='english', vocabulary=vocab)
X = feature_extraction.fit_transform(dataframe["TXT"].values)

# Split the training data
X_train, X_test = cross_validation.train_test_split(X, test_size = 0.1, random_state = 50)
print X_test

# Training with SVM
# y_train = dataframe["Class"].values[7]
# y_test = dataframe["Class"].values[7]

# clf = SVC(probability=True, kernel='rbf')
# clf.fit(X_train, y_train)

# # predict and evaluate predictions
# predictions = clf.predict_proba(X_test)
# print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))























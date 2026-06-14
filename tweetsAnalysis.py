import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

dataset = pd.read_csv('sentiment_analysis.csv')
X = dataset.iloc[:,4].values
y_raw = dataset.iloc[:, 5].to_list()

diccionario_sentimientos = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

y = np.array([diccionario_sentimientos[label] for label in y_raw], dtype=int)

nltk.download('stopwords')
all_stopwords = stopwords.words('english')
#all_stopwords_set = set(all_stopwords)
negaciones = {'not', 'no', 'nor', 'neither', 'never', 'but'}
all_stopwords_set = set(all_stopwords) - negaciones
ps = PorterStemmer()
corpus = []
for i in range(0,len(X)):
    comment = re.sub('[^a-z-A-Z]', ' ', X[i])
    comment = comment.lower().split()
    comment = [ps.stem(word) for word in comment if not word in all_stopwords_set]
    comment = " ".join(comment)
    corpus.append(comment)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
x = tfidf.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'linear', random_state = 0)
classifierSVM.fit(X_train,y_train)
y_pred = classifierSVM.predict(X_test)
print(classifierSVM.predict(X_test))
import pickle

with open('modelo_svm_3clases.pkl', 'wb') as f:
    pickle.dump(classifierSVM, f)


with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
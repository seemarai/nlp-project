# Natural Language Processing

# Importing the libraries
import numpy as np #array  ma kaam mgarna lai
import matplotlib.pyplot as plt #graph plotting garna lai
import pandas as pd #dataset import garne proper format ma halne

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re #regular expressions
import nltk#natural language toolkit
nltk.download('stopwords') #nachaine words like [is, are the, this,etc]
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer#root word nikalne. eg. loving, loved =>love
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])#letter bahek sabai hataune
    review = review.lower()#lower ma lane.
    review = review.split()#sentence lai array ma split garne
    ps = PorterStemmer()#port stemmer class instantiate gareko
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]#stopword hataune, root rakhne
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #review ra words haru ko mapping
y = dataset.iloc[:, 1].values#sabai row 1 column ko (liked wala lcoulm)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB#naive bayes vaneko classifier, main algorithm
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix#accurary haru check garna lai
cm = confusion_matrix(y_test, y_pred)

#showing results in graph
plt.hist([y_pred,y_test],label=['Predicted data','Test data'], bins=2)
plt.legend(loc='upper right')
plt.show()
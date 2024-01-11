# -*- coding: utf-8 -*-
"""Amazon_Product_Review_Sentiment_Analysis_NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nqMjnv0yBVJZgl-R4N4ke1RxONCcXm2R
"""

## 1000 rows taken from 300000+ rows in "Review.csv" file.

"""## Amazon Product Review Sentiment Analysis

## About Dataset

### Context
This is a small subset of dataset of Product reviews from Amazon Store category.

### Columns
- ProductId
- UserId
- ProfileName
- HelpfulnessNumerator
- HelpfulnessDenominator
- Score
- Time
- Summary
- Text

### Data source:-
 Data is taken from kaggle website.

### Inspiration
- Sentiment analysis on reviews.
- Understanding how people rate usefulness of a review/ What factors influence helpfulness of a review.
- or similarity between products based on reviews alone .

#### Best Practises
1. EDA & Data Processing
2. Train Test Split
3. Using NLP Techniques BOW,TFIDF,Word2vec
4. Train ML algorithms
"""

# Load the dataset
import pandas as pd
data = pd.read_csv('Reviews.csv', encoding='latin1',skiprows=[45837,73048,77030], error_bad_lines=False,nrows=1000)
# data = pd.read_csv('Reviews.csv', skiprows=[12406], error_bad_lines=False)
data.head(10)

data.info()

# Only fetching the required columns
df=data[['Score','Summary','Text']]
df.head(5)

df.isnull().sum()

df.head(5)

# Replacing null values
import pandas as pd

# Assuming df is your DataFrame
df['Summary'] = df['Summary'].fillna('NA')
# df['Text'] = df['Text'].fillna('NA')
# df['Score']=df['Score'].fillna(3)

df.isnull().sum()

df['Score'].unique()

df.head(5)

# Individual count of all the rows
df['Score'].value_counts()

# Function to convert Score categories from [1,2,3,4,5] to [0,1]...indicating 1 as positive review & 0 as negative review
df['Score']=df['Score'].apply(lambda x:0 if x<3 else 1)

df['Score'].value_counts()

df.head(5)

df['Text']=df['Text'].str.lower()

df.head(5)

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from bs4 import BeautifulSoup

## Removing special characters
df['Text']=df['Text'].apply(lambda x:re.sub('[^a-z A-z 0-9-]+', '',x))
## Remove the stopswords
df['Text']=df['Text'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))
## Remove url
df['Text']=df['Text'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
## Remove html tags
df['Text']=df['Text'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
## Remove any additional spaces
df['Text']=df['Text'].apply(lambda x: " ".join(x.split()))

df.head(5)

# Using technique WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

import nltk
nltk.download('wordnet')

df['Text']=df['Text'].apply(lambda x:lemmatize_words(x))

df.head(5)

## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df['Text'],df['Score'],
                                              test_size=0.25)

# Using Bag of words
from sklearn.feature_extraction.text import CountVectorizer
bow=CountVectorizer()
X_train_bow=bow.fit_transform(X_train).toarray()
X_test_bow=bow.transform(X_test).toarray()

# Using TF-IDF technique
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X_train_tfidf=tfidf.fit_transform(X_train).toarray()
X_test_tfidf=tfidf.transform(X_test).toarray()

X_train_bow

# The Gaussian Naive Bayes classifier is a type of Naive Bayes classifier specifically designed
# for data with continuous features, assuming that each feature follows a Gaussian (normal) distribution.
# and use it to train a model on your data.

from sklearn.naive_bayes import GaussianNB
nb_model_bow=GaussianNB().fit(X_train_bow,y_train)
nb_model_tfidf=GaussianNB().fit(X_train_tfidf,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

y_pred_bow=nb_model_bow.predict(X_test_bow)

y_pred_tfidf=nb_model_bow.predict(X_test_tfidf)

confusion_matrix(y_test,y_pred_bow)

print("BOW accuracy: ",accuracy_score(y_test,y_pred_bow))

confusion_matrix(y_test,y_pred_tfidf)

print("TFIDF accuracy: ",accuracy_score(y_test,y_pred_tfidf))
























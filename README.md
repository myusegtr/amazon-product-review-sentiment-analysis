## Amazon Product Review Sentiment Analysis

### Aim of the Project
The main goal of this project is to perform sentiment analysis on Amazon product reviews. Specifically, it aims to predict whether a review is positive or negative based on the text content. The project involves exploring natural language processing (NLP) techniques, including Bag of Words (BOW), TF-IDF, and a Naive Bayes classifier.

### Steps Involved

#### Data Loading and Exploration:
Load the dataset from the 'Reviews.csv' file, which is a small subset of product reviews from the Amazon store category.
Explore the dataset's structure, including columns such as ProductId, UserId, Score, Summary, and Text.

#### Data Preprocessing:

Select relevant columns (Score, Summary, Text).
Handle missing values by replacing them with appropriate defaults.
Convert the original 5-point rating scale to a binary sentiment classification (positive or negative).
Clean and preprocess the text data by converting to lowercase, removing special characters, stopwords, URLs, and HTML tags.
Lemmatize words to reduce them to their base form.

#### Train-Test Split:

Split the dataset into training and testing sets.

#### Feature Extraction:

Utilize Bag of Words (BOW) and TF-IDF techniques to convert the text data into numerical features.
For BOW, use CountVectorizer from scikit-learn.
For TF-IDF, use TfidfVectorizer from scikit-learn.

#### Model Training:

Train a Naive Bayes classifier using the Gaussian Naive Bayes algorithm.
Fit the model on both BOW and TF-IDF representations of the text.

#### Evaluation:

Evaluate the performance of the models using metrics such as accuracy, confusion matrix, and classification report.

### Tools/Technologies Required
Python (programming language)
Jupyter Notebook (or any Python IDE)
pandas (for data manipulation)
scikit-learn (for machine learning)
nltk (Natural Language Toolkit)
BeautifulSoup (for HTML parsing)

### Conclusion
The project successfully demonstrates sentiment analysis on Amazon product reviews using NLP techniques and a Gaussian Naive Bayes classifier. By converting text data into numerical features using BOW and TF-IDF, the models can predict the sentiment of reviews.

### What Extra Can Be Done in This Project
#### Hyperparameter Tuning:
Fine-tune the parameters of the Naive Bayes classifier to optimize performance.

#### Ensemble Methods:
Explore ensemble methods such as Random Forest or Gradient Boosting to improve predictive accuracy.

#### Deep Learning Approaches:
Experiment with more advanced techniques like recurrent neural networks (RNNs) or transformer-based models for sentiment analysis.

#### Feature Engineering:
Explore additional feature engineering techniques or use pre-trained word embeddings for better representation of text data.

#### Interactive Visualization:
Create interactive visualizations to showcase the distribution of sentiments and key insights from the dataset.

#### Deployment:
Deploy the trained model as a web application or API for real-time sentiment analysis.



### Kaggle link to download the dataset:- https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews?resource=download

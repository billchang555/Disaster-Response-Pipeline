import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import datetime
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath):
    # load data from database and extract X, Y data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    engine.dispose()

    X = df["message"]
    Y = df.iloc[:, 4:]
    categories = Y.columns.tolist()
    return X, Y, categories

def tokenize(text):
    # tokenize text
    text = re.sub(r'[^a-zA-Z0-9]', " ", text.lower())
    tokens = [word for word in word_tokenize(text) if word not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(word).strip() for word in tokens]
    return tokens


def build_model():
    # buile pipeline for model, return the ML model after GridSearchCV
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])
    # select parameters for CV
    parameters = {
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__learning_rate': [0.1, 0.5, 1]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # evaluate model
    pred_y = model.predict(X_test)
    for i in range(len(categories)):
        category = categories[i]
        print(category)
        print(classification_report(Y_test[category], pred_y[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

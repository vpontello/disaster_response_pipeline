import sys
import pandas as pd
import numpy as np
import re
import ast

import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sqlite3
import sqlalchemy as sqldb

import pickle
import json



def load_data(database_filepath):
    '''
    This function reads the data from the "messages_dataset" given a defined SQLite DB.
    It also splits the data between X an y for the following supervised learning
    '''
    # connect with the disaster SQLite DB
    connection = sqlite3.connect('../../data/02_trusted/'+database_filepath)
    # Extracting all the data from the main table 'messages_dataset'
    df = pd.read_sql("SELECT * FROM messages_dataset", con=connection, index_col='index')
    # Splitting dataset into X and y
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    return X, y


def tokenize(text):
    '''
    Function that normalizes, tokenizes, takes out the stop 
    words and lemmatizes a given text.
    '''
    # cleaning and normalizing
    text = re.sub("\W",' ',text.lower())
    # tokenizing
    words = word_tokenize(text)
    # removing stop words
    words = np.array(words)
    words = words[~np.isin(words,stopwords.words('english'))]
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def build_pipelines(classifiers):
    '''
    This function builds the pipelines from the given classifiers.
    '''
    # instantiating an empty dictionaru for the pipelines
    pipelines = {}
    # iterating through the classifiers dictionary
    for algorithm,classifier in classifiers.items():
        # creation of the pipeline
        pipeline = Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer(smooth_idf=False)),
            ('clf',MultiOutputClassifier(classifier)) # adding defined classifier into the pipeline
        ])
        # storing created pipeline into the pipelines dictionary
        pipelines[algorithm] = pipeline

    return pipelines


def build_models(classifiers,parameters_dict):
    '''
    This function creates the pipelines and a grid search instance for each created pipeline.
    The output is a dictionary with the configured grid search instances
    '''
    # building pipelines
    pipelines = build_pipelines(classifiers)
    # instantiation of the grid search pipeline
    cvs = {}
    # iterating through the parameters to create the grid search instances for each specific algorithm
    for algorithm,parameters in parameters_dict.items():
        # create grid search object
        cv = GridSearchCV(pipelines[algorithm], param_grid=parameters)
        # storing the grid seach instance into a dictionary
        cvs[algorithm] = cv
    
    return cvs


def evaluate_model(cv,y_test,y_pred,algorithm):
    '''
    this function receives the model pipeline, the test and predicted data as long as the
    base algorithm of the model. the classification reports are printed and the algorithm, it's best
    parameters and the classification reports are exported to a json file as log data of the actual model.
    '''
    # extracting the feature names from the y_test dataset
    columns = y_test.columns
    # instantiating an empty list for the classification reports
    classification_reports = []
    # instantiating a dict for the performance of each feature (f1-score)
    d_f1_score = {}
    # iterating through the y features
    for i, col in enumerate(columns):
        print(col)
        # getting the classification report for each column (feature) an printing them
        classif_report = classification_report(y_test[col], y_pred[:,i])
        print(classif_report)
        # appending the classification report to the list of classification reports
        classification_reports.append(classif_report)
        # storing the f1-score for each feature into a f1_score dictionary
        d_f1_score[col] = f1_score(y_test[col], y_pred[:,i],average=None)
        print('___________________________________')
    # printing the models best parameters
    print("\nBest Parameters:", cv.best_params_)
    print('##################################')

    # preparing the json output by creating a performance dictionary
    performance = {
        'algorithm':algorithm,
        'best_params':cv.best_params_,
        'class':classification_reports
    }
    # definig a patch filename with the base algorithm from the model
    patch = '../../data/03_models/out/'
    filename = patch + algorithm
    # saving the model`s performance
    with open(filename+'_results.json', 'w') as fp:
        json.dump(performance,fp) 
    return d_f1_score

def comparing_models(performances_f1):
    '''
    this function gets the performances (f1-scores) for each model and each feature,
    uses just the values for the class "True" and gets the mean value of all features
    in order to make easier to compare which model has the best overall performance
    '''
    # transforming the dictionary into a pandas.DataFrame
    df_performance = pd.DataFrame(performances_f1)
    # iterating through the features
    for col in df_performance.columns:
        # selecting just the class "True" of each feature
        df_performance[col] = df_performance[col].apply(lambda x : x[-1])
    # Defining the patch where the results are going to be stored
    print('____________________________________________________________________________')
    print(f'it seams the model {df_performance.mean().sort_values().index[-1]} has the best overall performance')
    print(df_performance.mean())
    print('____________________________________________________________________________')
    # saving output to a csv file
    patch = '../../data/03_models/out/'
    df_performance.mean().to_csv(patch+'mean_f1-scores.csv') 



def save_model(model, algorithm):
    '''
    This funciton saves the model into the desired patch folder and with the name of the algorithm
    '''
    # define file path name with the base algorithm from the model
    patch = '../../data/03_models/out/'
    filename = patch + algorithm

    print(f'saving model into {filename}')
    # saving model in a picke format
    pickle.dump(model.best_estimator_, open(filename+'.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, params_filename = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data('../../data/02_trusted/'+database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # reading the training parameters for grid search
        with open(f'../../data/03_models/in/{params_filename}') as json_file:
            parameters = json.load(json_file)
            for key_1, value_1 in parameters.items():
                for key_2, value_2 in value_1.items():
                    parameters[key_1][key_2] = ast.literal_eval(value_2)

        # defining which classifiers are going to be used
        classifiers = {
            'LGBMClassifier':LGBMClassifier(),
            'XGBClassifier':XGBClassifier()
        }

        
        print(f'Building models pipelines: {classifiers.keys()}')
        models = build_models(classifiers,parameters)

        # instantiation of a dictionary to compare the models performances (f1_score)
        performances_f1 = {}
        for algorithm, model in models.items():
            print(f'Training model {algorithm}')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print('Saving model...\n    MODEL: {}'.format(algorithm))
            save_model(model, algorithm)
            print('Trained model saved!')

            print(f'Evaluating model {algorithm}')            
            performances_f1[algorithm] = evaluate_model(model, y_test, y_pred, algorithm)
            
        comparing_models(performances_f1)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of grid search parameters '\
              'JSON file to train the models as second argument. \n\nExample: python '\
              'train_classifier.py DisasterResponse.db params.json')


if __name__ == '__main__':
    main()
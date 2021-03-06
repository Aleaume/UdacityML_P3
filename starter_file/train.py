from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, roc_curve, auc, roc_auc_score, accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset


#Requirements: azureml-sdk (pip install azureml-sdk)

def clean_data(data):
    
    # Split x train, y test were "points" is dropped
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop("vintage",axis=1) # no relevance for prediction was just needed to match weather with history
    x_df['points']- 83 #rearrange rankings


    ### NEED TO use OneHotEncoder  (one-of-K) algo to trnasform String to integer
    #x_df["variety"].describe()

    unique_varieties = x_df.variety.unique()
    varieties = x_df['variety']
    #'Sangiovese', 'Cabernet Franc', 'Red Blend', 'Sangiovese Grosso','Chardonnay', 'Vernaccia', 'Syrah', 'White Blend', 'Vermentino','Pinot Bianco', 'Viognier', 'Merlot', 'Rosato','Cabernet Sauvignon', 'Petit Verdot', 'Pinot Nero', 'Tempranillo','Aleatico', 'Rosé'

    values_varieties= np.array(varieties)
    label_encoder =LabelEncoder()
    int_encoded = label_encoder.fit_transform(values_varieties)
    x_df['variety'] = int_encoded
    

    wineries = x_df['winery']

    values_wineries= np.array(wineries)
    label_encoder =LabelEncoder()
    int_encoded = label_encoder.fit_transform(values_wineries)
    x_df['winery'] = int_encoded

#SOURCE : https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/


    y_df = x_df.pop("points")


    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=10000, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    #More argument could be passed
    

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    #classification task
    model = LogisticRegression(C=args.C, max_iter=args.max_iter,class_weight='balanced').fit(x_train, y_train)


    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    #accuracy_score = model.score(y_test,y_train)
    #run.log("Accuracy Score", np.float(accuracy_score))


    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'./outputs/model.joblib')

#  Create TabularDataset 

url_data="https://docs.google.com/spreadsheets/d/1X9M3eNuBDv0ZKsOdidkdaBx9W1NubNDz3kxXuULsXmo/export?format=csv"

dataset = Dataset.Tabular.from_delimited_files(path=url_data)


x, y = clean_data(dataset)

# TODO: Split data into train and test sets.


#output x_train, x_test , y_train, y_test needed for the main

x_train, x_test , y_train, y_test = train_test_split(x, y,test_size=0.33,random_state=64)

#more parameter to define the way the split is done could be given here. At this stage, not sure if needed.
#SOURCE/HELP: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html 
#https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/

run = Run.get_context()



if __name__ == '__main__':
    main()

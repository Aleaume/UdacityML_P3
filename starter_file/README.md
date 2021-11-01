# Udacity - Capstone Project - Azure Machine Learning Engineer

In this Project, I inted to build by own Azure ML Model based on a Dataset of my choice (see Dataset section for details)
This is splitted in 2 parts:
- On one side I try to train a model using AutoML
- And on the other we us a python training script and tune Hyperparameters with Hyperdrive

After that we can compare the model performances, deploy the Best Model and Test its Endpoint.

![image](https://user-images.githubusercontent.com/32632731/139536624-13d4d29a-7de5-4da1-9b90-da15c20a2922.png)


## Project Set Up and Installation

This project is intended & has been ran from a local Visual Studio Code using the Azure Machine Learning extension.

Once the Udacity Lab is up & running, you will first need to create a compute instance.

After that from Visual Studio Code simply connect to Compute Instance to Jupyter Server.

More info here: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-vs-code-remote?tabs=extension![image](https://user-images.githubusercontent.com/32632731/139536868-aadd8214-7d7c-4e9f-9d26-d661807db0d4.png)


## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The Dataset is composed of 2 combined dataset:
- A Dataset made of a wine list with their review score found in filtered on county="Tuscany"
Available on data.world here: https://data.world/markpowell/global-wine-points/workspace/query?filename=Wines.xlsx&newQueryType=SQL&selectedTable=wines&tempId=1634310537112

- An Extract of weather data for the Tuscan regions found here 
Available here: https://www.historique-meteo.net/amerique-du-nord/californie/


#### Once retrieved those 2 Dataset are merged into one in order to have the the final dataset used:
-vintage : Year of harvest of the wine (int)

-points : score given to the wine out of 100 (int)

-variety : grape variety of the wine (str)

-winery: winery name of the wine (str)

-avg winter temp: average winter temperature for the vintage year (eg. vintage 2010, winter: Dec2009,Jan2010,Feb2010) (float)

-avg spring temp: average spring temperature for the vintage year (float)

-avg summer temp: average summer temperature for the vintage year (float)
-avg fall temp: average fall temperature for the vintage year (float)
-avg winter sun hour: average winter sun hours for the vintage year (float)
-avg spring sun hour: average spring sun hours for the vintage year (float)
-avg summer sun hour: average summer sun hours for the vintage year (float)
-avg fall sun hour: average fall sun hours for the vintage year (float)
-avg daily precip winter: average daily winter precipitation in mm for the vintage year (float)
-avg daily precip spring: average daily spring precipitation in mm for the vintage year (float)
-avg daily precip summer: average daily summer precipitation in mm for the vintage year (float)
-avg daily precip fall: average daily fall precipitation in mm for the vintage year (float)

This Dataset can then be accessed here: https://docs.google.com/spreadsheets/d/1X9M3eNuBDv0ZKsOdidkdaBx9W1NubNDz3kxXuULsXmo/edit#gid=0 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

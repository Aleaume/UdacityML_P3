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

The main objective of is to be able to predict the review score of a wine based on either the winery, variety or all weather factors available.


### Access
The data is accessed via the publicly available url in the shared google sheets (see above), and then a Dataset is created using the register_pandas_dataframe function.

```python
#Google Drive path to dl csv
url_data = "https://docs.google.com/spreadsheets/d/1X9M3eNuBDv0ZKsOdidkdaBx9W1NubNDz3kxXuULsXmo/export?format=csv"

#Loads into Dataframe
df = pd.read_csv(url_data)

#Cleaning / Transform
df.drop("vintage") # nor relevance for prediction was just needed to match weather with history

#Save into Datastore AND Register as dataset
dataset = Dataset.Tabular.register_pandas_dataframe(df,datastore,'Dataset_Wine',description="A Dataset composed of Tuscan wines with ratings, vintage, grapes and corresponding season avg weather")


datasetWineTuscan = dataset.register(workspace=ws, name='Dataset_Wine', description="A Dataset composed of Tuscan wines with ratings, vintage, grapes and corresponding season avg weather")

```

```
```


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
The AutoML settings go as follow:
- A timeout after 45 minutes
- Classification algorithm task
- Looking for accuracy as primary metric
- Splitting the data with 80% train, 20% test
- Column to look for will be "points"
- With Cross-Validation in 3 subsets


```python

automl_config = AutoMLConfig(
    experiment_timeout_minutes=45,
    task='classification',
    primary_metric='accuracy',
    training_data= dataset,
    validation_size = 0.20,
    label_column_name='points',
    compute_target = cluster,
    n_cross_validations=3
    )

```

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

Once Ran we can see the details with:

```python
RunDetails(runAutoML).show()
```

The best model gives us a accuracy of XX%.

Also we can look at the details of the results via the Azure ML Studio:



Finally, There is definitely an imbalanced data issue with the dataset as there only 2 values with 83 points and 2 with 97.

![image](https://user-images.githubusercontent.com/32632731/139694728-8c65852e-2640-49d2-bfec-a88ff77b8de3.png)


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

For the part using Hyperparameter tuning, the setup is different. We use a train.py that contains the experiment logic and then pass it to the Hyperdrive to improve and fine tune parameters.

#### In the train.py

Here we first go ahead and apply different data cleaning / preparing activities, such as removing uneccasry column, splitting, rearannging ranges, and also converting string inputs to integers.

```python

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

```

Once done we pass 2 arguments to log on the prediction model Logistic Regression:
- Regularization strength
- Maximum itarations number

#### In the Hyperdrive settings

We then simply specify an Early termination policy:

```python
early_termination_policy = BanditPolicy(evaluation_interval=1, slack_factor=0.1, delay_evaluation=5)
#Ends runs when the primary metric isn't above (1/(1+0,1) ~90%. Start evaluating after 5 intervals.

```

And then detail the parameters to fine tune on the model

```python

param_sampling = RandomParameterSampling({
        "--C": uniform(0.001, 0.5),
        "--max_iter": choice(5,10,25,100,10000)
})
#Random / Grid  https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?view=azure-ml-py
#Bayesian sampling does not support early termination

```

Finally we create an estimator and define the remaining parameters in the HyperdriveConfig:

```python
stimator = SKLearn(
    source_directory=os.path.join('./'),
    compute_target=cluster,
    entry_script="./Users/odl_user_162459/train.py"
)



hyperdrive_run_config = HyperDriveConfig(hyperparameter_sampling=param_sampling,
                                        policy = early_termination_policy,
                                        primary_metric_name="Accuracy",
                                        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                        max_duration_minutes = 30,
                                        max_concurrent_runs= 5,
                                        estimator= estimator,
                                        max_total_runs = 50
                                        )

```

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

Using the RunDetails widget we can the results of the run:

We can conclude that ... influence ....

One possible way to improve the model would be expand the greatly the data volume and not being limited to the few years of analysis.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

After the best model is registered & saved we proceed to deploy it.

#### First as a local service (for testing)

```python

#To create a local deployment configuration
deployment_config = LocalWebservice.deploy_configuration(port=9064)

#Create Environment

env = Environment(name="AzureML-AutoML")
#myenv=env.clone("myenv")
#myenv.python.conda_dependencies.add_pip_package("joblib==1.1.0")

#Uploaded manually entry scripts in the below defined "entry_script" path.

my_inference_config = InferenceConfig(
    environment=env,
    source_directory= './',
    entry_script="./Users/odl_user_162459/score.py"
    #entry_script="./score.py"
)



# Deploy the service locally

service = model.deploy(ws, "local-service", [model], my_inference_config, deployment_config)
service.reload()
print(service.get_logs())

print(service.scoring_uri)

service.wait_for_deployment(show_output=True)

```

We can then call the service for testing

```python

#Call model to test 

#service.update(enable_app_insights=True)


uri = service.scoring_uri
requests.get("http://localhost:9064")
headers = {"Content-Type": "application/json"}
data = {
            "vintage": 2016,
            "variety": "Sangiovese",
            "winery": "Casa Raia",
            "avg winter temp": 4.2,
            "avg spring temp": 15,
            "avg summer temp": 27,
			"avg fall temp":16,
			"avg winter sun hour":7,
			"avg spring hour":12,
			"avg summer sun hour":16,
			"avg fall sun hour":9,
			"avg daily precip winter":3,
			"avg daily precip spring":3,
			"avg daily precip summer":1,
			"avg daily precip fall":2
}
data = json.dumps(data)
response = requests.post(uri, data=data, headers=headers)
print(response.json())

```

#### Finally as a WebService using ACI

```python
#Deploy to ACI

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1, auth_enabled=True
)

service = model.deploy(
    ws,
    "mywebservice",
    [model],
    my_inference_config,
    deployment_config,
    overwrite=True,
)
service.wait_for_deployment(show_output=True)

print(service.get_logs())


```

Once service is checked and in a Healthy status we test if by sending a request:

```

import requests
import json

primary, secondary = service.get_keys()

# URL for the web service
scoring_uri = service.scoring_uri
# If the service is authenticated, set the key or token
key = primary

# Two sets of data to score, so we get two results back
data = {
			"vintage": 2011,
			"variety": "Sangiovese",
			"winery": "Casa Raia",
			"avg winter temp": 4.2,
			"avg spring temp": 15,
			"avg summer temp": 27,
			"avg fall temp":16,
			"avg winter sun hour":7,
			"avg spring hour":12,
			"avg summer sun hour":16,
			"avg fall sun hour":9,
			"avg daily precip winter":3,
			"avg daily precip spring":3,
			"avg daily precip summer":1,
			"avg daily precip fall":2
	
}
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)


```



## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions

-Convert Model to ONNX format and export

-Fine tune json requests when testing deployments

-Automate upload of entry scripts + train script via az copy or else

-Retrieve userfolder in Azure instance directly from the notebook


### On the Model itself:

A lot improvement has to be done on that model to be successful. Accuracy is poor.
For instead I believe a better & larger volume of data would help.


# Udacity - Capstone Project - Azure Machine Learning Engineer

In this Project, I inted to build by own Azure ML Model based on a Dataset of my choice (see Dataset section for details)
This is splitted in 2 parts:
- On one side I try to train a model using AutoML
- And on the other we us a python training script and tune Hyperparameters with Hyperdrive

After that we can compare the model performances, deploy the Best Model and Test its Endpoint.

![image](https://user-images.githubusercontent.com/32632731/139536624-13d4d29a-7de5-4da1-9b90-da15c20a2922.png)
>Diagram of the setps of the Project

## Project Set Up and Installation

This project is intended & has been ran from a local Visual Studio Code using the Azure Machine Learning extension.

Once the Udacity Lab is up & running, you will first need to create a compute instance.

After that from Visual Studio Code simply connect to Compute Instance to Jupyter Server.

More info here: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-vs-code-remote?tabs=extension![image](https://user-images.githubusercontent.com/32632731/139536868-aadd8214-7d7c-4e9f-9d26-d661807db0d4.png)


## Dataset

### Overview

The Dataset is composed of 2 combined dataset:
- A Dataset made of a wine list with their review score found in filtered on county="Tuscany"
Available on data.world here: https://data.world/markpowell/global-wine-points/workspace/query?filename=Wines.xlsx&newQueryType=SQL&selectedTable=wines&tempId=1634310537112

- An Extract of weather data for the Tuscan regions found here 
Available here: https://www.historique-meteo.net/amerique-du-nord/californie/


#### Once retrieved those 2 Dataset are merged into one in order to have the the final dataset used:
- vintage : Year of harvest of the wine (int)

- points : score given to the wine out of 100 (int)

- variety : grape variety of the wine (str)

- winery: winery name of the wine (str)

- avg winter temp: average winter temperature for the vintage year (eg. vintage 2010, winter: Dec2009,Jan2010,Feb2010) (float)

- avg spring temp: average spring temperature for the vintage year (float)

- avg summer temp: average summer temperature for the vintage year (float)

- avg fall temp: average fall temperature for the vintage year (float)

- avg winter sun hour: average winter sun hours for the vintage year (float)

- avg spring sun hour: average spring sun hours for the vintage year (float)

- avg summer sun hour: average summer sun hours for the vintage year (float)

- avg fall sun hour: average fall sun hours for the vintage year (float)

- avg daily precip winter: average daily winter precipitation in mm for the vintage year (float)

- avg daily precip spring: average daily spring precipitation in mm for the vintage year (float)

- avg daily precip summer: average daily summer precipitation in mm for the vintage year (float)

- avg daily precip fall: average daily fall precipitation in mm for the vintage year (float)


This Dataset can then be accessed here: https://docs.google.com/spreadsheets/d/1X9M3eNuBDv0ZKsOdidkdaBx9W1NubNDz3kxXuULsXmo/edit#gid=0 

### Task


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

automl_settings = {
    "n_cross_validations":3,
    "validation_size":0.20,
    "label_column_name":'points',
    "primary_metric":'accuracy',
    "experiment_timeout_minutes":45  
}

automl_config = AutoMLConfig(
    task='classification',
    training_data= dataset,
    compute_target = cluster,
    **automl_settings
    )


```

![image](https://user-images.githubusercontent.com/32632731/139710169-4c9952fe-6da4-452f-8ca4-6ec284dacb7b.png)
>Output of the AutoML experiment running

### Results

Once Ran we can see the details with:

```python
RunDetails(runAutoML).show()
```

![image](https://user-images.githubusercontent.com/32632731/139710314-2db77038-40da-4cd7-9e8c-cab03422490d.png)
>Output of the RunDetails command on the AutoML experiment

The best model gives us a accuracy of 24.6%.

Also we can look at the details of the results via the Azure ML Studio:

![image](https://user-images.githubusercontent.com/32632731/139710471-9aabf06e-0460-4c05-92c8-eea4c4fd23f2.png)
>Screenshot of the Top 6 Child runs of the AutoML Experiment

![image](https://user-images.githubusercontent.com/32632731/139710390-db38eabe-45ef-4297-b257-108415e93a17.png)
>Screenshot of the Cummulative Gains Curve & the Confusion Matrix of the Best AutomL run


![image](https://user-images.githubusercontent.com/32632731/139711131-9375a003-2be2-43c6-889c-761dd4c5422c.png)
>Screenshot of the Top 5 features of the Best AutomL run


Finally, There is definitely an imbalanced data issue with the dataset as there only 2 values with 83 points and 2 with 97.

![image](https://user-images.githubusercontent.com/32632731/139694728-8c65852e-2640-49d2-bfec-a88ff77b8de3.png)
>Output showing the balancing issue in detail

#### Best AutoML run

Once done we retrieve the best run:

```python

best_run, fitted_model = runAutoML.get_output()

runAutoML.wait_for_completion()
print(best_run)

RunDetails(best_run).show()

```

![image](https://user-images.githubusercontent.com/32632731/139948839-3fb25654-82ab-4463-9222-4977d9e12bcf.png)
>Screenshot of the best AutoML run with id and experiment name followed by a Precision-Recall chart

![image](https://user-images.githubusercontent.com/32632731/139948962-fdbd6c39-9b12-4aec-b0f8-42b18a025771.png)
>ROC curve of the best AutoML run

![image](https://user-images.githubusercontent.com/32632731/139949057-179fafd0-b291-4787-ba53-e57dce2e8306.png)
>Confusion Matrix of the Best AutoML Run

![image](https://user-images.githubusercontent.com/32632731/139949176-62ea4ba4-7189-40cd-ab19-57eb429bc6fb.png)
>Top 10 Feature importance of the Best AutoML Run

To finish we simply need to register this best run as model:

```python

We can also have a look at the fitted model:

![image](https://user-images.githubusercontent.com/32632731/140036966-d75ac38f-b729-4295-8d04-caf397df951f.png)
>Sceenshot of the output of the fitted Model

# Register run as Model

model_name = "AleaumeModelAutoML"
description = "Best AutoML Model"
model_path ="outputs/modelAutoML.pkl"

model = runAutoML.register_model(model_name = model_name, description = description)


```

![image](https://user-images.githubusercontent.com/32632731/139951651-f223ad40-89cc-43b8-9ae7-b4a7fd0323da.png)
>Details of the just registered Model with its RunID (Best run)

## Hyperparameter Tuning

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


![image](https://user-images.githubusercontent.com/32632731/139721182-d94219a8-53fc-4e20-a3f1-81c2769ca49a.png)
>Output of the train.py script being ran

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

We can see in this piece of code that the hyperparamter search is focused on 2 parameters:

- C, for Inverse of Regularization Strength. We are looking here at intending to reduce overfitting. Regularization is applying a penalty to increase the impact of parameter values. 
By looking at values from 0.001 (higher penalty) to 0.5 (lower penalty applied) we expect to find what penalty would be best for this experiment.
The selecting method used here is uniform, meaning we apply a unifrom distribution between the 2 values.
This was thought to be relevant in oder to balance the weather metrics & the other values such as vintage.

- Max_iter, for the maximum number of iteration. We are looking here at the number of iteration to be used in each run
The selecting method used here gives the possibility to specify ourselves the exhaustive list of values to be used (5,10, 25, 100 or even 10000)
This was used in order to test and assess if increasing greatly the number of iterations or simpy using low "simple" values such as 25 or 10 would have any influence in the result of the model.

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

Using the RunDetails widget we can the results of the run:

![image](https://user-images.githubusercontent.com/32632731/139716905-0c9d2eaf-608a-4e23-a467-c19cf44ea6af.png)
>Screenshot of the Hpyerdrive RunDetails with the top 6 runs based on accuracy

![image](https://user-images.githubusercontent.com/32632731/139716957-67ce1c1c-85bf-4b4e-8ad4-5d8a13759830.png)
>Screenshot of the Hpyerdrive RunDetails with the distribution of accuracy among the runs on top, and at the bottom a chart plotting the 2 parameters against (-C & max_iter)

Then we retrieve the best run of the experiment and display its details
![image](https://user-images.githubusercontent.com/32632731/139717045-1a298f13-d3e9-4812-9781-4183791367c3.png)
>Screenshot of the Hpyerdrive RunDetails of the best run

We can conclude that both the Regularization strenght and the maximum number of iteration of little to no   influence in improving the primary metric (Accuracy).

![image](https://user-images.githubusercontent.com/32632731/139717115-5f85f788-38bc-4c1c-b660-f5e848a4054e.png)
>Screenshot of All the child runs and their metrics on the left, and on the right the dispersion of the 2 focused parameters and the Accuracy of the runs.

One possible way to improve the model would be expand the greatly the data volume and not being limited to the few years of analysis.

## Overview of the 2 Models

Before we go ahead and deploy one Model let's review the main perks & fails of the 2 models:

The AutoML best model performed overall better looking at the accuracy metric we were mainly looking for.
We reached a **accuracy of ~25%** using a VotingEnsemble algorithm. We can see here in Detail the different step that the model goes through:

![image](https://user-images.githubusercontent.com/32632731/139952130-df756062-3014-4e68-acc2-e6013c8b6a19.png)
>Data transformation steps of the Best AutoML run

For the Hyperdrive model, things look a bit more complicated. We did achieve a **best accuracy of about ~14%** using a LogisticRegression algorithm.
Here is a little diagram detailing the steps of the model:

![image](https://user-images.githubusercontent.com/32632731/139954606-0997a50d-e331-4084-85fa-afd728f40c5b.png)
>Diagram of the main steps of the Hyperdrive Experiment

 ##### Following those conclusions, I decided to go ahead and deploy the model with the best accuracy, the AutoML one.

## Model Deployment


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

![image](https://user-images.githubusercontent.com/32632731/139713523-d7adcc89-722e-4fdf-91f8-5a030b4ec1bf.png)
> Screenshot of the Deployed Web Service and its Status

![image](https://user-images.githubusercontent.com/32632731/139713726-fe6c40eb-9d0e-4e8b-8156-6b29566644ff.png)
> Screenshot of the Deployed Web Service logs


Once service is checked and in a Healthy status we test if by sending a request:

```python

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

A Screenast can be found here: https://youtu.be/cXrj7JMoKlo

## Standout Suggestions To be implemented for improvement

- Convert Model to ONNX format and export.
So far unsuccessful with a few try-outs. I couldn't find relevant documentation where converting AutoML model to ONNX was clearly detailed. Udacity course does mention coreml, not sure yet how relevant ?

- Fine tune json requests when testing deployments (at the moment unsucessful). One idea to look at is the auto-generated entry scripts when registering the best run as a model.

- Automate upload of entry scripts + train script via az copy or else. Due to the development context I used for this project, I couldnt manage this time to have every single steps of the project to be processed from within the notebooks. One of the step where this is visible is when making use of the entry scripts & the train.py script. Here at the moment I do upload those files manually.

- Retrieve userfolder in Azure instance directly from the notebook. Not sure if this possible, user folder name is linked to udacity lab session name. a scrapper of the page or log of authentication at the loading of the workspace followed by a parsing function to get the odl_user_**XXXXX** id.
#### Udpate: solved via extracting the last 6 characters of the workspace name
```python

lab_id = ws.name[-6:]

```
![image](https://user-images.githubusercontent.com/32632731/140037386-5cf785cc-4936-4310-af92-9e65f248dc09.png)
>Output of the retrieved lab id

Can then be used for specifying path in the deployment config:

entry_script_path = "Users/odl_user_"+lab_id+"/score.py"

```python
my_inference_config = InferenceConfig(
    environment=env,
    source_directory= './',
    entry_script= entry_script_path #"./Users/odl_user_162653/score.py"
    #entry_script="./score.py"
)

```

### On the Model itself:

- A lot of improvement has to be done on that model to be successful. Accuracy is poor.
For instead I believe a better & larger volume of data would help.
The current dataset is composed of only 646 entires with a low range of both points (very relevant for the accuracy) and small interval of time period (2010-2015).

For the purpose of this course I could have definitely taken an "easier" dataset and achieve better results, such as the typical "heart attack" datasets.

- Also, on the Hyperdrive part I think I could have tested maybe more parameters.
The current parameters could first be tested with different selection method such as normal / quniform / qloguniform


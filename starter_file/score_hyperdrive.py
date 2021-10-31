#ENTRY SCRIPT
#The entry script receives data submitted to a deployed web service and passes it to the model. 
# It then returns the model's response to the client. The script is specific to your model. 
# The entry script must understand the data that the model expects and returns.

#make sure file "score.py" is placed in the directory : "./source_dir"

import json
import joblib
import pickle
import numpy as np


import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.core.model import Model




def init():
    global model
    print("This is init")

    model_path = Model.get_model_path("AleaumeModelHyperdrive")
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'modelAutoML.pkl')
    print("Model Path is:",model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)
        result = model.predict(data['data'])
        return {'data' : result.tolist() , 'message' : 'Successfully predicted'}
    except Exception as e:
      error = str(e)
      return {'data' : error , 'message' : 'Failed to predict '}
    
#SOURCE / HELP :https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python

#https://medium.com/daily-programming-tips/how-to-deploy-a-local-ml-model-as-a-web-service-on-azure-machine-learning-studio-5eb788a2884c
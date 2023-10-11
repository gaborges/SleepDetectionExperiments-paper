#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from multiprocessing import Process

import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
import sklearn
import time

import numpy as np
import pandas as pd
from pandas import DataFrame

#!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Make TensorFlow log less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# In[22]:


# input folder
#inputFolders = "../02-transformed-data-new-testes/dados2019/"
inputFolderPath = "../data_2019_processed/"

# General configuration
NUMBER_OF_ITERATIONS = 120
NUM_EPOCHS = 1
BATCH_SIZE = 32
VERBOSE = 0

# output folder
outputFolder = "result-unbalanced_test"

# train file name modifier
fileSufixTrain = "" # _smote for smote

fl.common.logger.configure(identifier="myFlowerExperiment", filename="log"+outputFolder+".txt")


# In[23]:


# selected features
inputFeatures = ["activity","location","day_of_week","light","phone_lock","proximity","sound","time_to_next_alarm", "minutes_day"]
outputClasses = ["awake","asleep"]
#outputClasses = ["class"]


# In[24]:


# client datasets used on the training process (75% of data)
trainFolders =  ['0Jf4TH9Zzse0Z1Jjh7SnTOe2MMzeSnFi7feTnkG6vgs',
                '0tdmm6rwW3KquQ73ATYYJ5JkpMtvbppJ0VzA2GExdA', 
                '2cyV53lVyUtlMj0BRwilEWtYJwUiviYoL48cZBPBq0', 
                '2J22RukYnEbKTk7t+iUVDBkorcyL5NKN6TrLe89ys', 
                #['5FLZBTVAPwdq9QezHE2sVCJIs7p+r6mCemA2gp9jATk'], #does not have the file
                '7EYF5I04EVqisUJCVNHlqn77UAuOmwL2Dahxd3cA', 
                'a9Qgj8ENWrHvl9QqlXcIPKmyGMKgbfHk9Dbqon1HQP4', 
                'ae4JJBZDycEcY8McJF+3BxyvZ1619y03BNdCxzpZTc', 
                'Ch3u5Oaz96VSrQbf0z31X6jEIbeIekkC0mwPzCdeJ1U', 
                'CH8f0yZkZL13zWuE9ks1CkVJRVrr+jsGdUXHrZ6YeA', 
                'DHO1K4jgiwZJOfQTrxvKE2vn7hkjamigroGD5IaeRc', 
                #'DHPqzSqSttiba1L3BD1cptNJPjSxZ8rXxF9mY3za6WA', # does not have asleep data
                'dQEFscjqnIlug8Tgq97JohhSQPG2DEOWJqS86wCrcY', 
                'HFvs2CohmhHte+AaCzFasjzegGzxZKPhkrX23iI6Xo', 
                'jgB9E8v3Z6PKdTRTCMAijBllA9YEMtrmHbe4qsbmJWw', 
                'JkY++R7E8myldLN3on6iQ78Ee78zCbrLuggfwGju3I', 
                'K4SLohf+TN1Ak8Dn8iE3Lme7rEMPISfppB2sXfHX8', 
                'oGaWetJJJEWHuvYdWYo826SQxfhCExVVQ2da8LE1Y7Q', 
                'pyt24oiDAHsmgWMvkFKz2fn2pwcHiXchd6KchLM', 
                #'PZCf1nfvhR+6fk+7+sPNMYOgb8BAMmtQtfoRS83Suc', # does not have asleep data
                'QUNCATForxzK0HHw46LrGOMWh0eVA8Y5XWEiUXX+cQ', 
                #'rIl2UK9+bQ+tzpFdbJAdbBxEa5GbgrgC030yEaENLw', 
                #'RoBW3cDOO9wWRMPO2twQff83MPc+OXn6gJ+a1DafreI', 
                'SH3kQeyd5volraxw8vOyhlowNqWBPr1IJ9URNXUL4']
                #'VVpwFNMrEglveh6MDN8lrRzTy5OwzglD4FURfM4A2is', 
                #'Wa1mcNmbh66S7VS6GIzyfCFMD3SGhbtDQyFP1ywJEsw', 
                #'XCKRE0BWRHxfP1kZIihgtT+jUjSp2GE8v5ZlhcIhVmA', 
                #'YI5Y79K6GXqAUoGP6PNyII8WKlAoel4urDxWSVVOvBw', 
                #'ypklj+8GJ15rOIH1lpKQtFJOuK+VdvyCuBPqhY3aoM', 
                #'ZSsAZ0Pq+MCqFrnjsRFn5Ua09pMCVaOV9c8ZuYb7XQY']
            
# client datasets used on the training process (25% of data)
testFolders =  [#'0Jf4TH9Zzse0Z1Jjh7SnTOe2MMzeSnFi7feTnkG6vgs',
                #'0tdmm6rwW3KquQ73ATYYJ5JkpMtvbppJ0VzA2GExdA', 
                #'2cyV53lVyUtlMj0BRwilEWtYJwUiviYoL48cZBPBq0', 
                #'2J22RukYnEbKTk7t+iUVDBkorcyL5NKN6TrLe89ys', 
                #['5FLZBTVAPwdq9QezHE2sVCJIs7p+r6mCemA2gp9jATk'], #does not have the file
                #'7EYF5I04EVqisUJCVNHlqn77UAuOmwL2Dahxd3cA', 
                #'a9Qgj8ENWrHvl9QqlXcIPKmyGMKgbfHk9Dbqon1HQP4', 
                #'ae4JJBZDycEcY8McJF+3BxyvZ1619y03BNdCxzpZTc', 
                #'Ch3u5Oaz96VSrQbf0z31X6jEIbeIekkC0mwPzCdeJ1U', 
                #'CH8f0yZkZL13zWuE9ks1CkVJRVrr+jsGdUXHrZ6YeA', 
                #'DHO1K4jgiwZJOfQTrxvKE2vn7hkjamigroGD5IaeRc', 
                #'DHPqzSqSttiba1L3BD1cptNJPjSxZ8rXxF9mY3za6WA', # does not have asleep data
                #'dQEFscjqnIlug8Tgq97JohhSQPG2DEOWJqS86wCrcY', 
                #'HFvs2CohmhHte+AaCzFasjzegGzxZKPhkrX23iI6Xo', 
                #'jgB9E8v3Z6PKdTRTCMAijBllA9YEMtrmHbe4qsbmJWw', 
                #'JkY++R7E8myldLN3on6iQ78Ee78zCbrLuggfwGju3I', 
                #'K4SLohf+TN1Ak8Dn8iE3Lme7rEMPISfppB2sXfHX8', 
                #'oGaWetJJJEWHuvYdWYo826SQxfhCExVVQ2da8LE1Y7Q', 
                #'pyt24oiDAHsmgWMvkFKz2fn2pwcHiXchd6KchLM', 
                #'PZCf1nfvhR+6fk+7+sPNMYOgb8BAMmtQtfoRS83Suc', # does not have asleep data
                #'QUNCATForxzK0HHw46LrGOMWh0eVA8Y5XWEiUXX+cQ', 
                'rIl2UK9+bQ+tzpFdbJAdbBxEa5GbgrgC030yEaENLw', 
                'RoBW3cDOO9wWRMPO2twQff83MPc+OXn6gJ+a1DafreI', 
                #'SH3kQeyd5volraxw8vOyhlowNqWBPr1IJ9URNXUL4'] 
                'VVpwFNMrEglveh6MDN8lrRzTy5OwzglD4FURfM4A2is', 
                'Wa1mcNmbh66S7VS6GIzyfCFMD3SGhbtDQyFP1ywJEsw', 
                'XCKRE0BWRHxfP1kZIihgtT+jUjSp2GE8v5ZlhcIhVmA', 
                'YI5Y79K6GXqAUoGP6PNyII8WKlAoel4urDxWSVVOvBw', 
                'ypklj+8GJ15rOIH1lpKQtFJOuK+VdvyCuBPqhY3aoM', 
                'ZSsAZ0Pq+MCqFrnjsRFn5Ua09pMCVaOV9c8ZuYb7XQY']


# In[25]:


def generateMetrics(y_test,yhat_probs):
    # predict crisp classes for test set deprecated
    #yhat_classes = model.predict_classes(X_test, verbose=0)
    #yhat_classes = np.argmax(yhat_probs,axis=1)
    yhat_classes = yhat_probs.round()
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    #print(matrix)
    
    array = []
    results = dict()
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['cohen_kappa_score'] = kappa
    results['roc_auc_score'] = auc
    results['matrix'] = ("[[ " +str(matrix[0][0]) + " " +str(matrix[0][1]) +"][ " +str(matrix[1][0]) + " " + str(matrix[1][1]) +"]]") # array.append(np.array(matrix,dtype=object))
    results['TP'] = matrix[0][0]
    results['FP'] = matrix[0][1]
    results['FN'] = matrix[1][0]
    results['TN'] = matrix[1][1]
    
    array.append(accuracy)
    array.append(precision)
    array.append(recall)
    array.append(f1)
    array.append(kappa)
    array.append(auc)
    array.append("[[ " +str(matrix[0][0]) + " " +str(matrix[0][1]) +"][ " +str(matrix[1][0]) + " " + str(matrix[1][1]) +"]]") # array.append(np.array(matrix,dtype=object))
    array.append(matrix[0][0]) # TP
    array.append(matrix[0][1]) # FP
    array.append(matrix[1][0]) # FN
    array.append(matrix[1][1]) # TN
    
    return results, array

# y_test     = Array with real values
# yhat_probs = Array with predicted values
def printMetrics(y_test,yhat_probs):
    # generate metrics
    results, array= generateMetrics(y_test,yhat_probs)

    # accuracy: (tp + tn) / (p + n)
    accuracy = results['accuracy']
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = results['precision']
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = results['recall'] 
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = results['f1_score']
    print('F1 score: %f' % f1)
    # kappa
    kappa = results['cohen_kappa_score']
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = results['roc_auc_score']
    print('ROC AUC: %f' % auc)
    # confusion matrix
    print("\Confusion Matrix")
    matrix = results['matrix']
    print(matrix)
    
    return results, array

def generateGlobalMetrics(metrics):
    accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score = 0,0,0,0,0,0
    for metric in metrics:
        accuracy = accuracy + metric['accuracy']
        precision = precision + metric['precision']
        recall = recall + metric['recall']
        f1_score = f1_score + metric['f1_score']
        cohen_kappa_score = cohen_kappa_score + metric['cohen_kappa_score']
        roc_auc_score = roc_auc_score + metric['roc_auc_score']
        
    # mean
    size = len(metrics)
    print(size)
    accuracy = accuracy / size
    precision = precision / size
    recall = recall / size
    f1_score = f1_score / size
    cohen_kappa_score = cohen_kappa_score / size
    roc_auc_score = roc_auc_score / size
    
    return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score]

def showGlobalMetrics(metrics):
    res = generateGlobalMetrics(metrics)
    
    accuracy = res[0]
    precision = res[1]
    recall = res[2]
    f1_score = res[3]
    cohen_kappa_score = res[4]
    roc_auc_score = res[5]
    
    #show:\
    print("accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1_score: ",f1_score)
    print("cohen_kappa_score: ",cohen_kappa_score)
    print("roc_auc_score: ",roc_auc_score)
    
    return res


# In[26]:


# take the list of directories and concat them
def loadDataFromFolders(foldersToLoad,inputFolders,fileType = ""):
    print(len(foldersToLoad), "datasets")
    for i in range(0,len(foldersToLoad)):
        currentFolder = foldersToLoad[i]
        print(i , "-", currentFolder,inputFolders+"student_"+currentFolder+"_transformed"+fileType+".csv")
        #print(trainingDataSet[i])
        if(i == 0):
            temp_data = pd.read_csv(inputFolders+"student_"+currentFolder+"_transformed"+fileType+".csv")
        else:
            dataset = pd.read_csv(inputFolders+"student_"+currentFolder+"_transformed"+fileType+".csv")
            temp_data = pd.concat([temp_data, dataset])
    # return the dataset        
    return temp_data

# take the list of directories and concat them
def loadDataFromFoldersOnList(foldersToLoad,inputFolders,fileType = ""):
    clientList = []
    print(len(foldersToLoad), "datasets")
    for i in range(0,len(foldersToLoad)):
        currentFolder = foldersToLoad[i]
        print(i , "-", currentFolder,inputFolders+"student_"+currentFolder+"_transformed"+fileType+".csv")
        #print(trainingDataSet[i])
        temp_data = pd.read_csv(inputFolders+"student_"+currentFolder+"_transformed"+fileType+".csv")
        print("Adding to the list: ", temp_data.shape)
        clientList.append(temp_data)
    # return the dataset        
    return clientList


# In[27]:


print("Preparing test data")
 
# test data comprising 25% of the data. It must be fixed to all models being evaluated
#X_test  = pd.read_csv(inputFolders+"test/allData-classification-numeric-normalized.csv")
X_test = loadDataFromFolders(testFolders,inputFolderPath,"")

print()
# undestand the dataset by looking on their infos
print(X_test.info())

X_test


# In[28]:


print("Preparing X_train data")
# load cliend data
clientList = loadDataFromFoldersOnList(trainFolders,inputFolderPath,fileSufixTrain)
        
NUMBER_OF_CLIENTS = len(clientList)
print("Total",(len(clientList)))


# In[29]:


# one-hot encoding function
def transform_output_nominal_class_into_one_hot_encoding(dataset):
    # create two classes based on the single class
    one_hot_encoded_data = pd.get_dummies(dataset['class'])
    #print(one_hot_encoded_data)
    dataset['awake'] = one_hot_encoded_data['awake']
    dataset['asleep'] = one_hot_encoded_data['asleep']
    
    return dataset

# one-hot encoding function
def transform_output_numerical_class_into_one_hot_encoding(dataset):
    # create two classes based on the single class
    one_hot_encoded_data = pd.get_dummies(dataset['class'])
    #print(one_hot_encoded_data)
    dataset['awake'] = one_hot_encoded_data[0]
    dataset['asleep'] = one_hot_encoded_data[1]
    
    return dataset

# transform output to one_hot_encoding for the testing dataset
X_test = transform_output_nominal_class_into_one_hot_encoding(X_test)

# transform output to one_hot_encoding for the input dataset
for i in range(0,len(clientList)):
    clientList[i] = transform_output_nominal_class_into_one_hot_encoding(clientList[i])
    #print (clientList[i])
    

X_test.info()


# In[30]:


def transform_data_type(dataframe):
    
    # transform inputs
    for column in inputFeatures:
        dataframe[column] = dataframe[column].astype('float32')
    
    # transform outputs
    for column in outputClasses:
        dataframe[column] = dataframe[column].astype('float32')
    
    return dataframe

# transforms the data
X_test = transform_data_type(X_test)

X_test.info()


# In[31]:


print("Prepering the test dataset")
# selects the data to train and test
X_test_data = X_test[inputFeatures]
y_test_label = X_test[outputClasses]

# transtorm data to tensor slices
client_test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data.values, y_test_label.values))

#client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE, drop_remainder=True)
client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)

print(client_test_dataset.element_spec)
client_test_dataset


# In[32]:


print("preparing the training datasets")
federated_training_data = []
# transform the data
for i in range(0,len(clientList)):
    # selects the data to train and test
    data   = clientList[i][inputFeatures]
    labels = clientList[i][outputClasses]
    # transform the data to tensor slices
    client_train_dataset = tf.data.Dataset.from_tensor_slices((data.values, labels.values))
    # apply the configs
    client_train_dataset = client_train_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)
    # transform the data to
    federated_training_data.append(client_train_dataset)


# In[33]:


print("creating model")

def create_keras_model():
    return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(9,)),
      #tf.keras.layers.Dense(9, activation=tf.keras.activations.relu), 
      tf.keras.layers.Dense(16, activation=tf.keras.activations.relu),
      tf.keras.layers.Dense(8, activation=tf.keras.activations.relu),
      tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
      #tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])

keras_model = create_keras_model()
#keras_model.summary()
keras_model.summary()


# In[34]:


# Load model and data (MobileNetV2, CIFAR-10)
model = keras_model
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# In[35]:


print("Declarating the function to generate the metrics")

def evaluate_and_save_results(keras_model,X_test_data, y_test_label, current_round_index, 
                              clientId, prefix_string = "Results", lossValue = -1):
     # predict values
    yhat_probs = keras_model.predict(X_test_data)
    
    # as we deal with a classification problem with one hot encoding, we must round the values to 0 and 1.
    yhat_probs_rounded = yhat_probs.round()
    
    # create a dataframe with the predicted data
    y_predicted_df = pd.DataFrame(data=yhat_probs_rounded,columns=['awake','asleep']) 
    #y_test_label_label = pd.DataFrame(data=y_test_label,columns=['awake','asleep']) 
    
    roundData = []

    columns = ['client','round','loss','class','accuracy','precision','recall', 
               'f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
               'TP','FP','FN','TN']
    
    # Instantiate the list that will contain the results
    listOfMetrics = list()
    
    #print('awake')    
    #res,resA = printMetrics(y_test_label['awake'],y_predicted_df['awake'])
    res,resA = generateMetrics(y_test_label['awake'],y_predicted_df['awake'])
    listOfMetrics.append(res)
    
    classData = np.concatenate(([clientId,lossValue,current_round_index,'awake'], resA))
    roundData.append(classData)
    
    #print('')
    #print('asleep')
    #res,resA = printMetrics(y_test_label['asleep'],y_predicted_df['asleep'])
    res,resA = generateMetrics(y_test_label['asleep'],y_predicted_df['asleep'])
    listOfMetrics.append(res)
    # new data
    classData = np.concatenate(([clientId,lossValue,current_round_index,'asleep'], resA))
    roundData.append(classData)
    
    #print('Global')
    #resA = showGlobalMetrics(listOfMetrics) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    resA = generateGlobalMetrics(listOfMetrics) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    # new data
    classData = np.concatenate(([clientId,lossValue,current_round_index,'avg'], resA))
    roundData.append(classData)
    
    dataMetrics = pd.DataFrame(data=roundData,columns=columns) 
    
    # write file
    outputMetricFile = outputFolder+"/"+prefix_string+"_MLP_unbalanced_client_" + str(clientId) + "_round_" + str(current_round_index) + ".csv"
    dataMetrics.to_csv(outputMetricFile, sep=',', encoding='utf-8', index=False)


# In[36]:


print("Declarating server function")
# based on: https://flower.dev/blog/2021-01-14-single-machine-simulation-of-federated-learning-systems/
def start_server(num_rounds: int, num_clients: int):
    """Start the server with FedAvg strategy."""
    print("Starting server with FedAvg to "+str(num_clients)+" for "+str(num_rounds)+" rounds")
    strategy = FedAvg(min_available_clients=num_clients)
    # Exposes the server by default on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8099",
        strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )


# In[47]:


# Define a Flower client
print("Declarating client class")

class FlowerISABELASleepClient(fl.client.NumPyClient):

    def __init__(self, clientId, model, X_train_data, y_train_label):
        self.round_index = 0
        self.clientId = clientId
        self.model = model
        self.X_train_data = X_train_data
        self.y_train_label = y_train_label


    def get_parameters(self, config):
        """Return current weights."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training examples."""
        model.set_weights(parameters)
        model.fit(X_train_data, y_train_label, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,verbose=VERBOSE)

        # Evaluate local model parameters on the local test data
        loss, accuracy = model.evaluate(X_test_data, y_test_label,verbose=VERBOSE)

        # print model results
        evaluate_and_save_results(model,X_test_data, y_test_label, round_index, clientId,"local_model_results",loss)
        return model.get_weights(), len(X_train_data), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        model.set_weights(parameters)

        # Evaluate global model parameters on the local test data
        loss, accuracy = model.evaluate(X_test_data, y_test_label,verbose=VERBOSE)

        print("test metrics Accuracy: "+accuracy+" - Loss: "+loss)

        file_global_model = outputFolder+"/global_model_MLP_"+str(round_index)
        # print global model results
        if(not(os.path.isfile(file_global_model))):
            evaluate_and_save_results(keras_model,X_test_data, y_test_label, round_index, 0 ,"global_model",loss)
        # increment round
        round_index = round_index + 1

        # Return results, including the custom accuracy metric
        num_examples_test = len(X_test_data)
        return loss, num_examples_test, {"accuracy": accuracy}


# In[48]:


print("Declarating client function")

def start_client(X_train_data: DataFrame, y_train_label: DataFrame, 
                 clientId : int
                ) -> None:
    """Start a single client with the provided dataset."""

    print("creating client model")
    # Load and compile a Keras model for CIFAR-10
    model = create_keras_model()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8099", client=FlowerISABELASleepClient(X_train_data,y_train_label, clientId, model))


# In[ ]:





# In[49]:


print("Define simulation script")

# based on
def run_simulation(num_rounds: int, num_clients: int):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(target=start_server, args=(num_rounds, num_clients))
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(3)
    
    for i in range(0,num_clients):
        # selects the data to train and test
        data   = clientList[i][inputFeatures]
        labels = clientList[i][outputClasses]
        
        client_process = Process(target=start_client, args=(data,labels,i))
        client_process.start()
        processes.append(client_process)
        
    start_client(data,labels,i)
    # Block until all processes are finished
    for p in processes:
        p.join()


# In[50]:


print("Start simulation")

if __name__ == "__main__":
    run_simulation(num_rounds=NUMBER_OF_ITERATIONS, num_clients=NUMBER_OF_CLIENTS)
    #run_simulation(num_rounds=10, num_clients=2)


# In[ ]:





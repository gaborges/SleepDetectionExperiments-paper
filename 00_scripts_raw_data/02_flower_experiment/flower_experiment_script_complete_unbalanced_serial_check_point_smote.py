#!/usr/bin/env python
# coding: utf-8

# In[27]:


import glob
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from multiprocessing import Process
import gc

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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Trigger a garbage collection
gc.collect()

# In[30]:


# argumentos
n = len(sys.argv)
print("Total arguments passed:", n)
iteracoes = 0
finalIterations = 0
if(n > 0):
    for value in sys.argv:
        print("arg:", value)
        try:
            iteracoes = int(value)
            break
        except:
            print("no")
print("iteracoes:",0)        


# In[29]:


# input folder
#inputFolders = "../02-transformed-data-new-testes/dados2019/"
inputFolderPath = "../data_2019_processed/"

# General configuration
NUMBER_OF_ITERATIONS_FINAL = 200
    
NUM_EPOCHS = 1
BATCH_SIZE = 32
VERBOSE = 0

# output folder
outputFolder = "result_smote_epoch_"+str(NUM_EPOCHS)+"_rounds_"+str(NUMBER_OF_ITERATIONS_FINAL)
#outputFolder = "test_checkpoint"
checkPointFolder = outputFolder+"/checkpoints"

# train file name modifier
fileSufixTrain = "_smote" # _smote for smote

fl.common.logger.configure(identifier="myFlowerExperiment", filename="log_"+outputFolder+fileSufixTrain+".txt")

# usado para checkpoints
if(iteracoes > 0):
    NUMBER_OF_ITERATIONS_FINAL = iteracoes
    
NUMBER_OF_ITERATIONS = NUMBER_OF_ITERATIONS_FINAL


# In[3]:


print("Checking whether the folder exists or not")
isExist = os.path.exists(outputFolder)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(outputFolder)
    print("The new directory is created!")
else:
    print("The directory exists!")


# In[4]:


print("Checking whether the checkpoint folder exists or not")
isExist = os.path.exists(checkPointFolder)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(checkPointFolder)
    print("The new checkpoint directory is created!")
else:
    print("The checkpoint directory exists!")


# In[5]:


# selected features
inputFeatures = ["activity","location","day_of_week","light","phone_lock","proximity","sound","time_to_next_alarm", "minutes_day"]
outputClasses = ["awake","asleep"]
#outputClasses = ["class"]


# In[6]:


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


# In[ ]:





# In[7]:


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


# In[8]:


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


# In[9]:


print("Preparing test data")
 
# test data comprising 25% of the data. It must be fixed to all models being evaluated
#X_test  = pd.read_csv(inputFolders+"test/allData-classification-numeric-normalized.csv")
X_test = loadDataFromFolders(testFolders,inputFolderPath,"")

print()
# undestand the dataset by looking on their infos
print(X_test.info())

X_test


# In[10]:


print("Preparing X_train data")
# load cliend data
clientList = loadDataFromFoldersOnList(trainFolders,inputFolderPath,fileSufixTrain)
        
NUMBER_OF_CLIENTS = len(clientList)
print("Total",(len(clientList)))


# In[11]:


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


# In[12]:


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


# In[13]:


print("Prepering the test dataset")
# selects the data to train and test
X_test_data = X_test[inputFeatures]
y_test_label = X_test[outputClasses]

# transtorm data to tensor slices
#client_test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data.values, y_test_label.values))

#client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE, drop_remainder=True)
#client_test_dataset = client_test_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)

#print(client_test_dataset.element_spec)
#client_test_dataset


# In[14]:


#print("preparing the training datasets")
#federated_training_data = []
# transform the data
#for i in range(0,len(clientList)):
#    # selects the data to train and test
#    data   = clientList[i][inputFeatures]
#    labels = clientList[i][outputClasses]
#    # transform the data to tensor slices
#    client_train_dataset = tf.data.Dataset.from_tensor_slices((data.values, labels.values))
    # apply the configs
#    client_train_dataset = client_train_dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE)
    # transform the data to
 #   federated_training_data.append(client_train_dataset)


# In[15]:


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


# In[16]:


# Load model and data (MobileNetV2, CIFAR-10)
#model = keras_model
#model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# In[17]:


def evaluate_and_save_results(keras_model,X_test_data, y_test_label, current_round_index, 
                              clientId, prefix_string = "Results", lossValue = -1):
     # predict values
    yhat_probs = keras_model.predict(X_test_data,verbose=VERBOSE)
    
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
    
    classData = np.concatenate(([clientId,current_round_index,lossValue,'awake'], resA))
    roundData.append(classData)
    
    #print('')
    #print('asleep')
    #res,resA = printMetrics(y_test_label['asleep'],y_predicted_df['asleep'])
    res,resA = generateMetrics(y_test_label['asleep'],y_predicted_df['asleep'])
    listOfMetrics.append(res)
    # new data
    classData = np.concatenate(([clientId,current_round_index,lossValue,'asleep'], resA))
    roundData.append(classData)
    
    #print('Global')
    #resA = showGlobalMetrics(listOfMetrics) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    resA = generateGlobalMetrics(listOfMetrics) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    # new data
    classData = np.concatenate(([clientId,current_round_index,lossValue,'avg'], resA))
    roundData.append(classData)
    
    dataMetrics = pd.DataFrame(data=roundData,columns=columns) 
    # write file
    if(clientId >= 0):
        outputMetricFile = outputFolder+"/"+prefix_string+"_MLP_client_" + str(clientId) + "_round_" + str(current_round_index) + ".csv"
    else:
        outputMetricFile = outputFolder+"/global_model_MLP_metrics.csv"
        # print global model results
        if(os.path.isfile(outputMetricFile)):
            dataset = pd.read_csv(outputMetricFile)
            dataMetrics = pd.concat([dataset, dataMetrics], axis=0)
        # Perform garbage collection
        gc.collect()
        
    dataMetrics.to_csv(outputMetricFile, sep=',', encoding='utf-8', index=False)


# In[18]:


print("Loading checkpoint model",checkPointFolder+"/round-*")
list_of_files = [fname for fname in glob.glob(checkPointFolder+"/round-*")]
last_round_checkpoint = -1
latest_round_file = None
model_check_point = None
filename_np = None
filename_h5 = None

if len(list_of_files) > 0:
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    if(len(latest_round_file) > 0):
        # load the name
        last_round = latest_round_file.replace(checkPointFolder+"/round-","")
        last_round = last_round.replace("-weights.npz","")
        last_round = last_round.replace("-weights.h5","")
        print("Last round: ",last_round)
    
        last_round_checkpoint = int(last_round)
        filename_h5 = checkPointFolder+"/round-"+last_round+"-weights.h5"
        filename_np = checkPointFolder+"/round-"+last_round+"-weights.npz"
else:
    print("No checkpoint found")

    #check_point_model = tf.keras.models.load_model(latest_round_file)
    


# In[ ]:





# In[19]:


last_round_checkpoint


# In[20]:


#if latest_round_file is not None:
#    keras_model.load_weights(latest_round_file)


# In[ ]:





# In[21]:


NUMBER_OF_ITERATIONS


# In[22]:


if(last_round_checkpoint > -1):
    NUMBER_OF_ITERATIONS = NUMBER_OF_ITERATIONS_FINAL - (last_round_checkpoint)

    print("Number of iteractions after the checkpoint: ",NUMBER_OF_ITERATIONS)


# In[23]:


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        #print("TEsteeee", aggregated_parameters)
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            fileName = f"{checkPointFolder}/round-{server_round}-weights.npz"
            print(fileName)
            #print(aggregated_parameters)
            print()
            np.savez(fileName, *aggregated_ndarrays)
            #np.savez(fileName+"2", *aggregated_parameters)
            #keras_model = create_keras_model()
            #keras_model.set_weights(aggregated_parameters)
            #keras_model.save_weights(fileName)

        return aggregated_parameters, aggregated_metrics


# In[24]:


print("Declarating client function")

# Define a Flower client
class FlowerISABELASleepClient(fl.client.NumPyClient):

    def __init__(self, clientId, model, X_train_data, y_train_label,round_index=0):
        self.round_index = round_index
        self.clientId = clientId
        self.model = model
        self.X_train_data = X_train_data
        self.y_train_label = y_train_label


    def get_parameters(self, config):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training examples."""
        self.model.set_weights(parameters)
        
        # use the checkpoint if it is not -1
        #if(last_round_checkpoint == self.round_index and last_round_checkpoint != -1):
        #    print("loading checkpoint: ",filename_h5, " to client ",self.clientId)
        #    print("loading", latest_round_file)
        #    self.model = tf.keras.models.load_weights(filename_h5)
            
        self.model.fit(self.X_train_data, self.y_train_label, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,verbose=VERBOSE)

        # Evaluate local model parameters on the local test data
        loss, accuracy = self.model.evaluate(X_test_data, y_test_label,verbose=VERBOSE)

        # print model results
        evaluate_and_save_results(self.model,X_test_data, y_test_label, self.round_index, self.clientId,"global",loss)
        return self.model.get_weights(), len(self.X_train_data), {}
    


# In[25]:


def get_evaluate_fn( model):
    
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        current_round = server_round
        
        if(last_round_checkpoint > -1):
            current_round = server_round + last_round_checkpoint
            
        print("Evaluating global model round",current_round)
        
        model.set_weights(parameters)
        
        # Evaluate local model parameters on the local test data
        loss, accuracy = model.evaluate(X_test_data, y_test_label,verbose=VERBOSE)

        # only saves if the server_round + last_round_checkpoint != last_round_checkpoint to avaid duble metrics
        if(current_round > last_round_checkpoint):
            # print model results
            evaluate_and_save_results(model,X_test_data, y_test_label, current_round, -1,"global_model_MLP_metrics",loss)

            #checkpoint
            fileName = f"{checkPointFolder}/round-{current_round}-weights.h5"
            model.save_weights(fileName)
        else:
            print("Round already evaluated")
        
        return loss, { 'accuracy': accuracy }
    return evaluate


# In[26]:


import ray
import math
# Create an instance of the model and get the parameters

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
#if DEVICE.type == "cuda":

client_resources = {"num_cpus": 1}

#keras_model = create_keras_model()
keras_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])



def client_fn(cid) -> FlowerISABELASleepClient:
    print("starting client: "+str(cid),type(cid))
    #convert client ID to int
    clientId = int(cid)
    print("starting client: ", type(clientId))

    data   = clientList[clientId][inputFeatures]
    labels = clientList[clientId][outputClasses]
    
    print("Creating client model to client: "+str(cid))
    print("Data X: "+str(len(data)))
    print("Data Y: "+str(len(labels)))
    
    file_global_model = outputFolder+"/global_model_MLP_metrics.csv"
    index_round = 0 
    
    # get last
    if(os.path.isfile(file_global_model)):
        dataset = pd.read_csv(file_global_model)
        index_round = dataset["round"].max() + 1
        del dataset
    
    # update the index round in the previous checkpoint
    if(last_round_checkpoint > -1 and (index_round == last_round_checkpoint)):
        index_round = last_round_checkpoint
    
    print("Creating client model to client: "+str(cid),"round",index_round)
    # Load and compile a Keras model for CIFAR-10
    model = create_keras_model()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    
    return FlowerISABELASleepClient(clientId,model,data,labels,index_round)

strategy = SaveModelStrategy(
    min_available_clients=NUMBER_OF_CLIENTS,
    evaluate_fn=get_evaluate_fn(keras_model)
) # (same arguments as FedAvg here)

# load checkpoint
if(filename_h5 is not None):
    
    #npzFile = np.load(filename_np)
    keras_model.load_weights(filename_h5)
    
    initial_parameters = keras_model.get_weights() 
    # Convert the weights (np.ndarray) to parameters (bytes)
    init_param = fl.common.ndarrays_to_parameters(initial_parameters)

    strategy = SaveModelStrategy(
        min_available_clients=NUMBER_OF_CLIENTS,
        evaluate_fn=get_evaluate_fn(keras_model),
        initial_parameters = init_param
    ) # (same arguments as FedAvg here)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUMBER_OF_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUMBER_OF_ITERATIONS),  # Just three rounds
    client_resources=client_resources,
    strategy = strategy
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





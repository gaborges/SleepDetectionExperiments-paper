#!/usr/bin/env python
# coding: utf-8

# In[2]:


# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from tensorflow import keras

import tensorflow as tf

from scipy import stats

import datetime;
import warnings
warnings.filterwarnings("ignore")


# In[6]:


# configs
#EPOCHS = 5
BATCH_SIZE = 32

VERBOSE = 0
baseFolder = "../data_2019_processed/"

fileSufixTrain = "" # unbalanced file sufix is empty
outputFileSufixTrain = "unb" # unbalanced file sufix is umb

# selected features
inputFeatures = ["activity","location","day_of_week",
                 "light","phone_lock","proximity",
                 "sound","time_to_next_alarm", "minutes_day"]
outputClasses = ["awake","asleep"]
#outputClasses = ["class"]

TIME_SERIES_SIZE = -1 # we do not use it
TIME_STEP_SHIFT = -1 # we do not use it

NN_type = 'MPL'
UNITS_NUMBER = "16-8";
EPOCHS_ARRAY_TEST = [5,15,30,50,80,100,120,150,200]
#EPOCHS_ARRAY_TEST = [2,7]

outputMetricFile = "result_trad_"+str(NN_type)+"_"+outputFileSufixTrain+"_batch_size_"+str(BATCH_SIZE)+".csv"


# In[7]:


outputMetricFile


# In[8]:


# y_test     = Array with real values
# yhat_probs = Array with predicted values
def printMetrics(y_test,yhat_probs):
    # predict crisp classes for test set deprecated
    #yhat_classes = model.predict_classes(X_test, verbose=0)
    #yhat_classes = np.argmax(yhat_probs,axis=1)
    yhat_classes = yhat_probs.round()
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    print("\Confusion Matrix")
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    
    array = []
    results = dict()
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['cohen_kappa_score'] = kappa
    results['roc_auc_score'] = auc
    #results['matrix'] = np.array(matrix,dtype=object)
    results['matrix'] = 1
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
    #array.append(np.array(matrix,dtype=object))
    array.append(0)
    array.append(matrix[0][0]) # TP
    array.append(matrix[0][1]) # FP
    array.append(matrix[1][0]) # FN
    array.append(matrix[1][1]) # TN
    
    return results, array

def showGlobalMetrics(metrics):
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
    
    #show:\
    print("accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1_score: ",f1_score)
    print("cohen_kappa_score: ",cohen_kappa_score)
    print("roc_auc_score: ",roc_auc_score)
    
    return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score]
    
def transform_data_type(dataframe):
    
    # transform inputs
    for column in inputFeatures:
        dataframe[column] = dataframe[column].astype('float32')
    
    # transform outputs
    for column in outputClasses:
        dataframe[column] = dataframe[column].astype('float32')
    
    return dataframe

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


def create_dataset_time_series_with_one_output(X, y, window_time_steps=1, shift_step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - window_time_steps, shift_step):
        v = X.iloc[i:(i + window_time_steps)].values
        labels = y.iloc[i: i + window_time_steps]
        Xs.append(v)        
        ys.append(stats.mode(labels)[0][0])
        
    if len(y.columns) == 1:
        return np.array(Xs), np.array(ys).reshape(-1, 1)
    else:
        return np.array(Xs), np.array(ys).reshape(-1, len(y.columns))


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


# In[25]:


print("Preparing test data")
 
# test data comprising 25% of the data. It must be fixed to all models being evaluated
#X_test  = pd.read_csv(inputFolders+"test/allData-classification-numeric-normalized.csv")
X_test = loadDataFromFolders(testFolders,baseFolder,"")

print()
# undestand the dataset by looking on their infos
print(X_test.info())

X_test


# In[26]:


print("Preparing X_train data")

# test data comprising 25% of the data. It must be fixed to all models being evaluated
#X_test  = pd.read_csv(inputFolders+"test/allData-classification-numeric-normalized.csv")
X_train = loadDataFromFolders(trainFolders,baseFolder,fileSufixTrain)

print()
# undestand the dataset by looking on their infos
print(X_train.info())

X_train


# In[27]:


#X_train = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized.csv")
#X_test  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized.csv")
#X_train = pd.read_csv(baseFolder+"train/allData-classification-numeric-normalized_balanced_undersample.csv")
#X_test  = pd.read_csv(baseFolder+"test/allData-classification-numeric-normalized_balanced_oversample.csv")

#AA = pd.read_csv(baseFolder+"allData-classification-numeric-normalized.csv")
#X_train, X_test = train_test_split(AA,test_size=0.25)


# In[28]:


print(X_train.info())
X_train


# In[29]:


# transform output to one_hot_encoding for the testing dataset
X_test = transform_output_nominal_class_into_one_hot_encoding(X_test)

# transform output to one_hot_encoding for the testing dataset
X_train = transform_output_nominal_class_into_one_hot_encoding(X_train)


# transforms the input data to float32
X_test = transform_data_type(X_test)

# transforms the input data to float32
X_train = transform_data_type(X_train)


# In[30]:


# selects the data to train and test
X_train_data = pd.DataFrame(data=X_train,columns=inputFeatures)
y_train_data = pd.DataFrame(data=X_train,columns=outputClasses)
# selec test dataset (fixed to all)
X_test_data = pd.DataFrame(data=X_test,columns=inputFeatures)
y_test_data = pd.DataFrame(data=X_test,columns=outputClasses)


# In[31]:


columnsOutputMetrics = ['NN_type','units','epochs','batch_size','window_size','time_step_shift',
           'start_time','end_time','time_s','time_m',
           'class','accuracy','precision','recall','f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
           'TP','FP','FN','TN']

allMetrics = []


# In[32]:


for TEST_EPOCHS in EPOCHS_ARRAY_TEST:
    # general data from the run
    generalData = [NN_type,UNITS_NUMBER,TEST_EPOCHS,BATCH_SIZE,TIME_SERIES_SIZE,TIME_STEP_SHIFT]
    
    #generate model
    model = keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(len(inputFeatures),)))
    model.add(tf.keras.layers.Dense(len(inputFeatures)))
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(8, activation=tf.keras.activations.relu),)
    model.add(tf.keras.layers.Dense(len(outputClasses), activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])
              #loss='binary_crossentropy',loss='categorical_crossentropy',
              #loss='binary_crossentropy',  sparse_categorical_crossentropy   
    
    print("input_shape=[", X_train_data.shape[0], X_train_data.shape[1],"]")
    print("output shape:",len(outputClasses))
    print("Epochs:",TEST_EPOCHS)
    
    current_time = datetime.datetime.now()
    time_stamp = current_time.timestamp()
    print("Start timestamp:", time_stamp,current_time)
    print()

    verbose, epochs, batch_size = VERBOSE, TEST_EPOCHS, BATCH_SIZE  
    # fit network
    model.fit(X_train_data, y_train_data, epochs=epochs, verbose=verbose, batch_size=batch_size) #, batch_size=batch_size, validation_split=0.1
    # evaluate model
    #accuracy = model.evaluate(test_dataset_series1) # , batch_size=batch_size, verbose=0
    # predict
    yhat_probs = model.predict(X_test_data,verbose=0)
    # predict crisp classes for test set deprecated
    
    # generate time metrics
    current_time2 = datetime.datetime.now()
    time_stamp2 = current_time2.timestamp()
    processing_time_s = (time_stamp2-time_stamp)
    # generate general metrics
    rowData = [current_time,current_time2,processing_time_s,(processing_time_s)/60]

    y_pred_labels = pd.DataFrame(data=yhat_probs,columns=['awake','asleep'])
    y_test_labels = pd.DataFrame(data=y_test_data,columns=['awake','asleep'])

    feature_metrics_gathered = []
    # print('')
    print('awake')    
    res,resA = printMetrics(y_test_labels['awake'],y_pred_labels['awake'])
    feature_metrics_gathered.append(res)
   
    #columns = ['NN_type','units','epochs','batch_size','max_iterations',''Users',
    #            round_iteration','start_time','end_time','round_time_s','round_time_m',
    #           'class','accuracy','precision','recall','f1_score','cohen_kappa_score','roc_auc_score','confusion_matrix',
    #           'TP','FP','FN','TN']
    # new data
    classData = np.concatenate((['awake'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    allMetrics.append(classData)
    
    print('')
    print('asleep')
    res,resA = printMetrics(y_test_labels['asleep'],y_pred_labels['asleep'])
    feature_metrics_gathered.append(res)
    # new data
    classData = np.concatenate((['asleep'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    allMetrics.append(classData)
    print('')
    print('Global')
    resA = showGlobalMetrics(feature_metrics_gathered) #return [accuracy,precision,recall,f1_score,cohen_kappa_score,roc_auc_score
    # new data
    classData = np.concatenate((['avg'], resA))
    classData = np.concatenate((rowData, classData))
    classData = np.concatenate((generalData, classData))
    allMetrics.append(classData)
    print('')
    print("End timestamp:", time_stamp2,current_time2)
    print('')
    print('')


# In[ ]:


dataMetrics = pd.DataFrame(data=allMetrics,columns=columnsOutputMetrics) 

dataMetrics


# In[ ]:


dataMetrics.to_csv(outputMetricFile, sep=',', encoding='utf-8', index=False)


# In[ ]:





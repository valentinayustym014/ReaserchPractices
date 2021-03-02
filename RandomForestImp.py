#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: valentinayusty
"""

# Implementation of Random Forest 


# importing required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
plt.rc("font", size=14)
 
dfRF = pd.read_excel('CleanDB.xlsx')
 
train, test = train_test_split(dfRF, test_size=0.30)

#Variable Selection 
 
selectedVar=['ANTIGUEDAD_RAD', 'PERSONASCARGO_RAD', 'EDAD_RAD', 'CUOTASINF_RAD',
 'INGRESOS', 'ESTADOCIV_RAD', 'GENERO_RAD', 'ESTUDIO_RAD',
 'SUBTIPOCLI_RAD', 'VIVIENDA_RAD', 'PLAZO',"ESTADO1"]

trainingDatax=train[selectedVar]
trainingDatay=train['ESTADO'].to_frame()
 
testingDatax=test[selectedVar]
testingDatay=test['ESTADO'].to_frame()
  
model = RandomForestClassifier()
 
# Fit model 
model.fit(trainingDatax,trainingDatay)
 
# Trees Used
print(model.n_estimators)
 
predictionTrain = model.predict(trainingDatax)

# Accuray Score Training DS
accuracy_train = accuracy_score(trainingDatay,predictionTrain)
print(accuracy_train)
 
# Predict Target 
predictionTest = model.predict(testingDatax)
print(predictionTest) 
 
# Accuracy Score Testing DS
accuracy_test = accuracy_score(testingDatay,predictionTest)
print(accuracy_test)
 
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(testingDatay,predictionTest)
print(confusion_matrix)
 
from sklearn.metrics import classification_report
print(classification_report(testingDatay,predictionTest))
 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Roc Curve
logit_roc_auc = roc_auc_score(testingDatay, model.predict(testingDatax))
fpr, tpr, thresholds = roc_curve(testingDatay, model.predict_proba(testingDatax)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area= %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
 
import matplotlib.pyplot as plt

def plot_CM(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(confusion_matrix, cmap=cmap) # imshow
    plt.colorbar()
    tick_marks = np.arange(len(confusion_matrix.columns))
    plt.xticks(tick_marks, confusion_matrix.columns, rotation=45)
    plt.yticks(tick_marks, confusion_matrix.index)
    plt.ylabel(confusion_matrix.index.name)
    plt.xlabel(confusion_matrix.columns.name)
 
plot_CM(confusion_matrix)
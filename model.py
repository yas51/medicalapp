# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 00:15:08 2019

@author: PC EXPERT
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:14:05 2019

@author: Yasin
"""

#https://towardsdatascience.com/deploying-a-heart-failure-prediction-model-using-flask-and-heroku-55fdf51ee18e
#https://github.com/Ifeoluwa-hub/Heart-Failure-Prediction-and-Deployment-with-Flask-and-Heroku/blob/main/templates/index.html
#

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import pickle



# Importing the dataset

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
##this returns the first five rows of the dataset
#df.head() 

# #Verification des lignes si elles ne sont pas vides
# df.info()

# # verifier la description
# df.describe()

# df.shape ##returns the no. of rows and columns
# df.dtypes
# df.describe
# df.isnull().sum() ##check for null values
# df.duplicated().any() ##check for duplicate values
# df['DEATH_EVENT'].value_counts().plot(kind='bar')
# plt.show()

#on cree les variables X et y
y = df['DEATH_EVENT']
X = df.drop(['DEATH_EVENT'], axis = 1)

#Split en set de données pour test et training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

#Nomalisation des Xtrain et Xtest
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

#Application du modele SVM
from sklearn import svm
svm = svm.SVC()
svm.fit(x_train, y_train)
predictions = svm.predict(x_test)


#mATRICE DE CONFUSION ET ACCURACY
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print("Confusion Matrix : \n\n" , confusion_matrix(predictions,y_test))
print("Classification Report : \n\n" , classification_report(predictions,y_test),"\n")


#CREER LE MODELE PICKLE EN FICHIER 

pickle.dump(svm, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# model = pickle.load(open('model.pkl', 'rb'))
# print(model)









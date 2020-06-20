# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:31:13 2019

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import random

#import the train data
data = pd.read_csv('train.csv')
y = data['Purchase']
data = data.drop(['Purchase'],axis = 1)

#import the test data
test_data = pd.read_csv('test.csv')

#combine the train and test data
total_data = data.append(test_data)

#one hot encoding of the gender column
total_data['Gender'] = pd.Categorical(total_data['Gender'])
encoded_gender = pd.get_dummies(total_data['Gender'], prefix = 'category')
total_data = pd.concat([total_data, encoded_gender], axis=1)
total_data = total_data.drop(['Gender'],axis =1)

total_data = pd.get_dummies(total_data, columns=['City_Category']) 

#creating an array of age with the defined ranges in the dataset
age_counts = total_data['Age'].value_counts()
age_array = []
age_range =[]
for age in total_data['Age']:
    if age == '0-17':
        age_range.append(random.randrange(0, 17))
        age_array.append(age)
    if age == '18-25':
        age_range.append(random.randrange(18, 25))
    if age == '26-35':
        age_range.append(random.randrange(26, 35))
    if age == '36-45':
        age_range.append(random.randrange(36, 45))
    if age == '46-50':
        age_range.append(random.randrange(46, 50))
    if age == '51-55':
        age_range.append(random.randrange(51, 55))
    if age == '55+':
        age_range.append(random.randrange(55, 60))

age = pd.DataFrame (age_range,columns=['age'])
total_data['age'] = age
total_data = total_data.drop(['Age'],axis =1)

#
total_data = total_data.replace(to_replace ="4+", 
                 value ="5") 
Stay_In_Current_City_Years = total_data['Stay_In_Current_City_Years'].astype(int)
total_data['Stay_In_Current_City_Years_'] = Stay_In_Current_City_Years
total_data = total_data.drop(['Stay_In_Current_City_Years'],axis =1)

#handle the object/strings of the product id column
ids = total_data['Product_ID'].value_counts()
total_data['Product_ID'] = total_data['Product_ID'].map(lambda x: x.lstrip('P').rstrip('aAbBcC'))
total_data['Product_ID'] = total_data['Product_ID'].astype(int)
ids_1 = total_data['Product_ID'].value_counts()

#check all nan values and fill them with mean- as of now
#all_nan = total_data.isna()
total_data = total_data.fillna(total_data.mean())

#all data in X with datatype different
X = total_data.iloc[:,:].values

#divide into train and test
train = X[:550068,:]
test = X[550068:,:]

test_df = pd.DataFrame(test)
test_df.to_csv('test_preprocessed.csv')

#output changed to int64- before doing this datatype is pandas Series
y = y.iloc[:].values

#saving models using pickle
import pickle

#fit and train the model-multiple regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train, y)
#predict 
ypred =regressor.predict(test)

#save the model
pickle.dump(regressor, open('regressor.sav','wb'))

#Dt regression
from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor(random_state = 0)
regressorDT.fit(train, y)
ypred_dt = regressorDT.predict(test)

pickle.dump(regressorDT, open('regressorDT.sav','wb'))

#RF regression
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 100, random_state = 0,max_features = "log2",oob_score = True)
regressorRF.fit(train, y)
ypred_rf = regressorRF.predict(test)

#save the RF model
pickle.dump(regressorRF, open('regressorRF.sav','wb'))

from sklearn.ensemble import AdaBoostRegressor
regressorRF_ada = AdaBoostRegressor(n_estimators = 50,random_state = 0)
regressorRF_ada.fit(train,y)
regressorRF_ada.feature_importances_
regressorRF_ada.score(train,y)
y_pred_boosting = regressorRF_ada.predict(test)

#save the model
pickle.dump(regressorRF_ada, open('regressorRF_ada.sav','wb'))


from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(train, y)
y_gb = est.predict(test)

#save the model
pickle.dump(est, open('est.sav','wb'))












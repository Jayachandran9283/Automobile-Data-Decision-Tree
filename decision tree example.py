# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\jayac\Downloads\Dataset\Automobile_data.csv')
data.replace('?',np.nan,inplace=True)
string_col=data.select_dtypes(exclude=np.number).columns.tolist()
num_cols=['normalized-losses','bore','stroke',
 'horsepower','peak-rpm','price']
#covert into numeric data type
for i in num_cols:
    data[i]=pd.to_numeric(data[i],errors='raise')
    
#Categorical conversion
for i in data:
    if is_string_dtype(data[i]):
        data[i]=data[i].astype('category').cat.as_unordered()
        
#catcode conversion

for i in data:
    if (str(data[i].dtype)=='category'):
        data[i]=data[i].cat.codes
        
#imputation
data.fillna(data.median(),inplace=True)

#Modelling
X=data.drop('symboling',axis=1 )
y=data['symboling']

#train test split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=100,test_size=0.2)

# dt=DecisionTreeClassifier()
# dt.fit(x_train,y_train)
# print(dt.score(x_test,y_test))

lr=LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))

#Hyperparameter

NoofEstimator = [5, 10, 15, 20]
MinSampleLeaf = [1, 3, 5, 7]
MaxFeature = np.arange(0.1, 1.1, 0.1)
best_score = []

for i in NoofEstimator:
    for j in MinSampleLeaf:
        for k in MaxFeature:
            result = [i, j, k]
            rfc = RandomForestClassifier(n_estimators = i,
                                         min_samples_leaf = j,
                                         max_features = k,random_state=100)
            rfc.fit(x_train, y_train)
            result.append(rfc.score(x_train, y_train))
            result.append(rfc.score(x_test, y_test))
            if len(best_score) == 0:
                best_score = result
            elif best_score[4] < result[4]:
                best_score = result
                print(best_score)

print('The final best result is:', best_score)

#Grid Search
rf=RandomForestClassifier()
rf_grid=GridSearchCV(estimator=rf, param_grid=
                     dict(n_estimators = NoofEstimator,
                                         min_samples_leaf = MinSampleLeaf,
                                         max_features = MaxFeature))
rf_grid.fit(x_train,y_train)
print(rf_grid.best_estimator_)
print(rf_grid.score(x_test,y_test))

#Randomized Search CV
rf=RandomForestClassifier()
rf_random=RandomizedSearchCV(estimator=rf,
                     param_distributions=  dict(n_estimators = NoofEstimator,
                                         min_samples_leaf = MinSampleLeaf,
                                         max_features = MaxFeature))
rf_random.fit(x_train,y_train)
print(rf_grid.best_estimator_)
print(rf_grid.score(x_test,y_test))

##Checking out of bag score
rf_o=RandomForestClassifier(oob_score=True)
rf_o.fit(x_train,y_train)
rf_o.oob_score_



#Feature importance
imp_features=rf_grid.best_estimator_.feature_importances_
feature_list=list(X.columns)
feature_importance=sorted(zip(imp_features,feature_list),reverse=True)
df=pd.DataFrame(feature_importance,columns=['importance','feature'])

# Set the style
plt.style.use('bmh')
# list of x locations for plotting
x_values = list(range(len(feature_importance)))
importance= list(df['importance'])
feature= list(df['feature'])
# Make a bar chart
plt.figure(figsize=(15,10))
plt.bar(x_values, importance, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')

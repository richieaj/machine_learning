#!/usr/bin/env python
# coding: utf-8

# # LAB 3 - Prediction of Numeric Values

# ## By
#  - Name: JOLINSON RICHI 
#  - Register Number: 21122030
#  - Class: 2MscDs

# # Lab Overview

# # Problem Definition
# 
# **Common Instructions**
#  - Use Pandas to Import the Dataset
#  - Do the necessary Exploratory Data Analysis
#  - Use the train_test_split method available in SCIKIT to split the dataset into Train Dataset and Test Dataset.
#  - Show the Regression Score, Intercept and other parameters etc in the Output
#  - Use visualizations and plots wherever possible
#  - Format the outputs neatly; Do Documentation, Data Set Description, Objectives, Observations, Conclusions etc as you have done in your previous lab
#  
# **Questions**
# 1. What are your observations on the Dataset?
# 2. What are the different Error Measures (Evaluation Metrics) in relation to Linear Regression? How much do you get in the above cases?
# 3. Note down the errors/losses when the train-test ratio is 50:50, 60:40, 70:30, and 80:20
# 4. During LinearRegression() process, what is the impact of giving TRUE/FALSE as the value for Normalize Parameter?
# 
# **Cases
# Try to predict the rent of the below houses -**
# 1. 1 BHK with 2 Baths in Portofino Street
# 2. Fully Furnished 2 BHK in School Street
# 3. Single Room anywhere in Lavasa

# # Objective
#  - Understand the dataset and features.
#  - Analyse the dataset.
#  - Exploring the given insight.

# # Approach
#  - Importing all libraries which we needed.
#  - Perform data preprocessing technique to get balanced structured data.
#  - Perform statistical data analysis and derive valuable inference.
#  - perform exploratory data analysis and derive valuable inference.
#  - Visualizing things with some plot and derive valuable inference.
#  - Train and test through LinearRegression models for better prediction.

# **Common Instructions**
#  - Use Pandas to Import the Dataset
#  - Do the necessary Exploratory Data Analysis
#  - Use the train_test_split method available in SCIKIT to split the dataset into Train Dataset and Test Dataset.
#  - Show the Regression Score, Intercept and other parameters etc in the Output
#  - Use visualizations and plots wherever possible
#  - Format the outputs neatly; Do Documentation, Data Set Description, Objectives, Observations, Conclusions etc as you have done in your previous lab

# ### Importing libraries

# In[45]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[14]:


hbk = pd.read_csv("D:\AJR\machine learning\HousePrices.csv")
hbk


# In[3]:


hbk.describe()


# In[4]:


hbk.info


# In[21]:


hbk.isnull().sum()


# In[23]:


hbk.groupby('Size')['AreaSqFt'].mean()


# In[25]:


hbk.duplicated().sum()


# In[27]:


hbk.skew()


# In[28]:


hbk.corr()


# In[33]:


fig=plt.figure(figsize=(10,6))
hbk.boxplot(column=['AreaSqFt'])
plt.semilogy()


# In[35]:


plt=sns.countplot(x='NoOfBalcony',data=hbk)


# In[41]:


plt.figure(figsize = [26,10])
sns.countplot(x = 'BuildingType', palette = "dark", alpha = 1, data = hbk)
sns.despine()


# In[12]:


plt=sns.countplot(x='Location',data=hbk)
sns.set(rc = {'figure.figsize':(15,10)})


# In[50]:


Pie = hbk['BuildingType'].value_counts().reset_index()
Pie.columns = ['BuildingType','Percent']
Pie['Percent'] /= len(hbk)
fig = px.pie(Pie, names = 'BuildingType', values = 'Percent', title = 'BuildingType', color = "Percent", color_discrete_sequence = px.colors.sequential.RdBu)
fig.show()


# In[47]:


Pie = hbk['Location'].value_counts().reset_index()
Pie.columns = ['Location','Percent']
Pie['Percent'] /= len(hbk)
fig = px.pie(Pie, names = 'Location', values = 'Percent', title = 'Location', color = "Percent", color_discrete_sequence = px.colors.sequential.RdBu)
fig.show()


# In[52]:


sns.set_palette("Spectral")
sns.pairplot(hbk,hue='BuildingType')


# In[53]:


plt.figure(figsize = [18,8])
sns.heatmap(hbk.corr(),annot=True)


# In[57]:


plt.figure(figsize=(40,30))
sns.distplot(hbk['AreaSqFt'], color = '#8F00FF')


# **Use the train_test_split method available in SCIKIT to split the dataset into Train Dataset and Test Dataset.**

# In[15]:


tts(hbk,shuffle=False)


# In[59]:


X=hbk[['BuildingType','Location','Size','AreaSqFt','NoOfBath','NoOfPeople','NoOfBalcony']]
y=hbk["RentPerMonth"]


# In[60]:


hbk["BuildingType"].value_counts()


# In[61]:


hbk["BuildingType"].unique()


# In[62]:


hbk['Size'].unique()


# In[63]:


hbk['Size'].value_counts()


# In[64]:


one_hot_encoded_data = pd.get_dummies(X, columns = ['BuildingType', 'Location'])


# In[65]:


X1=one_hot_encoded_data
X1


# In[68]:


def remove_BHK(x):
    x=int(x[0])
    return x
remove_BHK('1 BHK')


# In[72]:


t_size=[]
norm=[]
reg_score=[]
reg_intercept=[]
test_mae=[]
test_mse=[]
test_r2=[]
test_rmse=[]
train_mae = []
train_r2 = []
train_mse = []
train_rmse = []


# **50:50**

# In[73]:


test_size=1
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=True)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))


y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))

print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(True)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[74]:


df1=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1


# In[75]:


test_size=0.5
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=False)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(False)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[76]:


df2=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df2


# **60:40**

# In[77]:


# Creating a LinearRegressor model with normalization True and finding error,accuracy
test_size=0.4
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=True)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(True)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[78]:


df3=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df3


# In[79]:


# Creating a LinearRegressor model with normalization False and finding error,accuracy
test_size=0.4
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=False)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(False)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[80]:


df4=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df4


# **70:30**

# In[81]:


# Creating a LinearRegressor model with normalization True and finding error,accuracy
test_size=0.3
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=True)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(True)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[82]:


df5=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df5


# In[83]:


# Creating a LinearRegressor model with normalization False and finding error,accuracy
test_size=0.3
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=False)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(False)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[84]:


df6=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df6


# In[85]:


# Creating a LinearRegressor model with normalization False and finding error,accuracy
test_size=0.2
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=False)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))


# Append all these in above empty list
t_size.append(test_size)
norm.append(False)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[86]:


# Create a dataframe and compare between actual and predicted values
df7=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df7


# **80:20**

# In[87]:


# Creating a LinearRegressor model with normalization True and finding error,accuracy
test_size=0.2
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=test_size,random_state=42)
reg=LinearRegression(normalize=True)
reg.fit(X_train, y_train)
print("Regression score: ",reg.score(X_test, y_test))
print("Regression intercept: ",reg.intercept_)
print("coef: ",reg.coef_)
print("param: ",reg.get_params(deep=True))

y_pred=reg.predict(X_test)
train_pred = reg.predict(X_train)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, train_pred))
print("MSE: ",mean_squared_error(y_train, train_pred))
print("r2: ",r2_score(y_train, train_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, train_pred)))


print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# Append all these in above empty list
t_size.append(test_size)
norm.append(True)
reg_score.append(reg.score(X_test, y_test))
reg_intercept.append(reg.intercept_)
test_mae.append(mean_absolute_error(y_test, y_pred))
test_mse.append(mean_squared_error(y_test, y_pred))
test_r2.append(r2_score(y_test, y_pred))
test_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred)))


train_mae.append(mean_absolute_error(y_train, train_pred))
train_mse.append(mean_squared_error(y_train, train_pred))
train_r2.append(r2_score(y_train, train_pred))
train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))


# In[88]:


# Create a dataframe and compare between actual and predicted values
df8=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df8


# In[89]:


# Createing a dataframe with all error and accuracy values of all split case
df9=pd.DataFrame({'test_size':t_size, 'norm':norm, 'reg_score':reg_score, 'reg_intercept':reg_intercept, 'train_mae':train_mae, 'train_mse':train_mse, 'train_r2':train_r2, 'train_rmse':train_rmse, 'test_mae':test_mae, 'test_mse':test_mse, 'test_r2':test_r2, 'test_rmse':test_rmse})
df9


# In[90]:


df9


# In[91]:


df9


# **Questions**
# 1. What are your observations on the Dataset?
# 2. What are the different Error Measures (Evaluation Metrics) in relation to Linear Regression? How much do you get in the above cases?
# 3. Note down the errors/losses when the train-test ratio is 50:50, 60:40, 70:30, and 80:20
# 4. During LinearRegression() process, what is the impact of giving TRUE/FALSE as the value for Normalize Parameter?

# **1. Observation on the dataset**
# - The dataset shows the 
#   1. Building Type - Is it a fully/semi/Un furnished Single Room, Flat, or Villa ?
#   1. Location - Where is the property located?
#   1. Size - Is it 1BHK, 2BHK, 3BHK ?
#   1. AreaSqFt - How much big is the property ? 
#   1. No of Bath - How many bathrooms in the property?
#   1. No of Balcony - How many balconies in the property?
#   1. No of People - How many people stayed in the building in the academic year 2020-21.
#   1. RentPerMonth - Rent to be paid per month which is demanded by the current building owners.

# **2. What are the different Error Measures (Evaluation Metrics) in relation to Linear Regression? How much do you get in the above cases?**
#  - Different Error Measures (Evaluation Metrics) in relation to Linear Regression are MAE, MSE, r2, RMSE

# ### Predicting rent for the given requirements
#  - Cases
#    - Try to predict the rent of the below houses -
#    - 1. 1 BHK with 2 Baths in Portofino Street
#    - 2. Fully Furnished 2 BHK in School Street
#    - 3. Single Room anywhere in Lavasa

#  - From all the above spliting train-test ration we can see that our 80:20 split ratio with normalization True is good model with accuracy 90%. So we can predict these cases from that model.

# In[93]:


req_dict = dict()
for i in X_test.columns:
    print(i,": ")
    req_dict[i] = float(input())


# In[94]:


req_df=pd.DataFrame(req_dict,index=[0])
req_df


# In[96]:


reg.predict(req_df)


# # Conclusion:
#  - In this lab, we have tried to gain the knowledge about data and its varibles, further we did some preprocessing to the data in order to bring it into more analyst friendly mode, laterly we implemented various graphs using various libraries in order to get valuable insights, furthermore, we implemented and evaluated LinearRegression model to get high accuracy in term of predicting rental price to find houses for people who are in search for the one according to their preferences.

# # Reference
#  - https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
#  - https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/
#  - https://pandas.pydata.org/
#  - https://matplotlib.org/
#  - https://seaborn.pydata.org/
#  - https://plotly.com/
#  - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
#  - https://www.kaggle.com/c/house-prices-advanced-regression-techniques

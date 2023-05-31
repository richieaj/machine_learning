#!/usr/bin/env python
# coding: utf-8

# # Lab 7 - PCA & LDA

# # Submitted by
#  - Name: Jolinson Richie
#  - Register Number: 21122030
#  - Class: 2MscDs

# # Lab Overview

# # Problem Definition
#  - Perform PCA and LDA.
#  - Demonstrate the breast cancer dataset.

# # Objective
#   - Demonstrate the PCA and LDA methods.
#   - Illustrate the effect of changing various method parameters of PCA and LDA.
#   - Compare the accuracies, and provide visualizations and interpretations for the evaluation metrices.

# # Approach
#  - Import all necessry libraries.
#  - Works on the dataset.
#  - Doing necessary EDA parts for visualizations.
#  - Using PCA and LDA build the model.
#  - We also compare accuracy in PCA and LDA.

# # Code

# ## Question
#  - Part A. Perform PCA and LDA on Breast Cancer Dataset, write down your obsevations. While loading, use the toy dataset available in SKLearn (load_breast_cancer)
#  - Part B. Illustrate the effect of changing various method parameters of PCA and LDA. Compare the accuracies, and provide visualizations and interpretations for the evaluation metrices.

# ### Importing Liabraries

# In[46]:


# import data science basic liabrary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score


# ### Loading Data

# In[2]:


# import breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
# load data
cancer = load_breast_cancer()
# create dataframe from the data
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)


# In[3]:


df.head()


# ### Exploratory Data Analysis

# In[4]:


# data shape
df.shape


# In[5]:


# print data types
df.dtypes


# In[6]:


# data statistical summary
df.describe()


# In[7]:


# show data info
df.info()


# In[8]:


# Columns unique values
df.nunique()


# In[9]:


# count missing values in decending order
df.isnull().sum().sort_values(ascending=False)


# In[13]:


# target column
df['target'] = pd.Series(cancer.target)
# target_name column
df['target_names'] = pd.Series(cancer.target_names)
# count target names (We don't need this column ('target_names'). I'll drop it later)
df['target_names'].value_counts()


# In[14]:


# target_names replace acording to target
df['target_names'] = df['target'].replace({0: 'malignant', 1: 'benign'})


# In[15]:


# count target names column
df['target_names'].value_counts()


# In[16]:


# count target column
df['target'].value_counts()


# In[18]:


# Separete malignant and benign from target
Malingnant=df[df['target'] == 0]
Benign=df[df['target'] == 1]

# Shape of malignant and benign
print(Benign.shape)
print(Malingnant.shape)


# ### Visualization

# In[19]:


# correaltion matrix
plt.figure(figsize=(18,14))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.show()


# In[20]:


# distribution of all mean columns with malingnant and benign
# Figure size
fig = plt.figure(figsize = (18,14),tight_layout=True)
# Enumerate starting at 0 to 10, add 1 for subplotting
for i,b in enumerate(list(df.columns[0:10])):
    i = i + 1
    ax = fig.add_subplot(4,3,i)
    
    sns.distplot(df[b][df['target'] == 1],label = 'Benign', color = 'teal', bins = 20,hist = True )
    sns.distplot(df[b][df['target'] == 0],label = 'Malingnant', color = "r", bins = 20,hist = True)
    
    ax.set_xlabel('Value')    
    ax.set_title(b)
    plt.legend()
plt.suptitle('Distribution of Mean', y=1.04, size=20)
plt.tight_layout()
plt.show()


# In[21]:


# distribution of all error columns with malingnant and benign
# Figure size
fig = plt.figure(figsize = (18,14),tight_layout=True)
# Enumerate starting at 10 to 20, add 1 for subplotting
for i,b in enumerate(list(df.columns[10:20])):
    i = i + 1
    ax = fig.add_subplot(4,3,i)
    
    sns.distplot(df[b][df['target'] == 1],label = 'Benign', color = 'teal', bins = 20,hist = True )
    sns.distplot(df[b][df['target'] == 0],label = 'Malingnant', color = "r", bins = 20,hist = True)
    
    ax.set_xlabel('Value')    
    ax.set_title(b)
    plt.legend()
plt.suptitle('Distribution of Error', y=1.04, size=20)
plt.tight_layout()
plt.show()


# In[23]:


# distribution of all worst columns with malingnant and benign
# figure size
fig = plt.figure(figsize = (18,14),tight_layout=True)
# Enumerate starting at 0 to 10, add 1 for subplotting
for i,b in enumerate(list(df.columns[20:30])):
    i = i + 1
    ax = fig.add_subplot(4,3,i)
    
    sns.distplot(df[b][df['target'] == 1],label = 'Benign', color = 'teal', bins = 20,hist = True )
    sns.distplot(df[b][df['target'] == 0],label = 'Malingnant', color = "r", bins = 20,hist = True)
    
    ax.set_xlabel('Value')    
    ax.set_title(b)
    plt.legend()
plt.suptitle('Distribution of Worst', y=1.04, size=20)
plt.tight_layout()
plt.show()


# In[26]:


# box plot of all mean columns for cheaking outliers
fig = plt.figure(figsize = (18,14),tight_layout=True)
# Enumerate starting at 0 to 10, add 1 for subplotting
for i,b in enumerate(list(df.columns[0:10])):
    i = i + 1
    ax = fig.add_subplot(4,3,i)
    sns.boxplot(x=df['target'],y=b,data=df)
    ax.set_title(b)
plt.suptitle('Boxplot of Mean', y=1.04, size=20)
plt.show()


# In[27]:


#pairplot only 10 columns with target
sns.pairplot(df, hue='target',vars=list(df.columns[0:10]))
plt.suptitle('Pairplot of Mean', y=1.04, size=30)
plt.show()


# In[28]:


# joint plot with 'mean concavity' and 'mean concave points'
sns.jointplot(x='mean concavity',y='mean concave points',data=df,kind='reg')
plt.show()


# In[29]:


# joint plot with 'mean concave points' and 'worst concave points'
sns.jointplot(x='mean concave points',y='worst concave points',data=df,kind='reg')
plt.show()


# ### Data Preprocessing

# In[40]:


corr_matrix = df.corr().abs() 

mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_df = corr_matrix.mask(mask)

to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.90)]

df = df.drop(to_drop, axis = 1)

print(f"The reduced dataframe has {df.shape[1]} columns.")


# In[41]:


# drop target_names column from dataframe & assign to new variable df1
df1 = df.drop(['target_names'], axis=1)
# Replace column extra space to '_' underscore ( Its line create only avoid error for XGboost tree plot)
df1.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
# drop target column from dataframe
X= df1.drop(['target'], axis=1)
# copy target column from dataframe & assign to y      
y= df1["target"].copy() 


# In[42]:


df1.head()


# In[43]:


X.head()


# In[44]:


y.tail()


# ### PCA

# #### Using Standard Scaler

# In[47]:


scaler = preprocessing.StandardScaler().fit(X)
X_standardised = scaler.transform(X)


# In[48]:


X_standardised


# #### PCA Variable

# In[49]:


principal_component_analysis = PCA(n_components = 3)
results = principal_component_analysis.fit(X_standardised)
results_transformed = results.transform(X_standardised)


# In[50]:


results_transformed


# In[51]:


pca_dataframe = pd.DataFrame(data = results_transformed)


# In[52]:


pca_dataframe


# ### Decision Trees

# In[54]:


from sklearn.tree import DecisionTreeClassifier


# #### Entire Dataset

# In[55]:


model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = tts(X, y)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is : {}".format(accuracy))


# In[56]:


model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = tts(pca_dataframe, y)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is : {}".format(accuracy))


# In[57]:


def performDecisionTree(X, mode, n_components = 3):
    
    principal_component_analysis = PCA(n_components = n_components, random_state = 23)
    results_transformed = results.fit_transform(X_standardised)
    pca_dataframe = pd.DataFrame(data = results_transformed)
    
    if mode == "PCA":
        X = pca_dataframe
    elif mode == "X":
        X = X
    elif mode == "X_Standardised":
        X = X_standardised
    
    model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = tts(X, y, random_state= 49)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


# In[58]:


for m in ["PCA", "X", "X_Standardised"]:
    if m =="PCA":
        for i in range(1, 5):
            print("Mode: {}, PCA Components: {}, Accuracy Score : {}".format(m, i, performDecisionTree(X, m, i)))
    else:
        print("Mode: {}, Accuracy Score : {}".format(m, performDecisionTree(X, m, i)))


# ### LDA

# In[59]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[60]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 2, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[61]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

print('Accuracy -> ' + str(accuracy_score(y_test, y_pred)))


# ## Part C. "PCA could be used in applications such as Image Processing, to reduce the complexity of data and improve performance or to compress images". Justify this statement with your own findings.
# 
#  - Dimensionality reduction refers to techniques for reducing the number of input variables in training data. When dealing with high dimensional data, it is often useful to reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the “essence” of the data.
#  - Dimensionality reduction is the mapping of data from a high dimensional space to a lower dimension space such that the result obtained by analyzing the reduced dataset is a good approximation to the result obtained by analyzing the original data set.
#  - It reduces the time and storage space required. It helps Remove multi-collinearity which improves the interpretation of the parameters of the machine learning model. It becomes easier to visualize the data when reduced to very low dimensions such as 2D or 3D.
#  - Principal Component Analysis (PCA) is very useful to speed up the computation by reducing the dimensionality of the data. Plus, when you have high dimensionality with high correlated variable of one another, the PCA can improve the accuracy of classification model.

# ### Conclusion

#  - LDA tries to reduce the dimensionality by taking into consideration the information that discriminates the output classes. LDA tries to find the decision boundary around each cluster of class.
#  - It projects the data points to new dimension in a way that the clusters are as seperate from each other as possible and individual elements within a class are as close to the centroid as possible.
#  - In other words, the inter-class seperability is increased in LDA. Intra-class seperability is reduced.

# ### References

#  ##### PCA
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://towardsdatascience.com/principal-component-analysis-for-breast-cancer-data-with-r-and-python-b312d28e911f
# https://www.kaggle.com/jahirmorenoa/pca-to-the-breast-cancer-data-set
# https://www.youtube.com/watch?v=e2sM7ccaA9c&ab_channel=DigitalSreeni
# https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
# https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118
# https://www.kaggle.com/mirzarahim/introduction-to-pca-image-compression-example
# https://github.com/gtraskas/breast_cancer_prediction/blob/master/breast_cancer.ipynb
# 
# ##### LDA
# http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
# https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
# https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2
# https://www.mygreatlearning.com/blog/linear-discriminant-analysis-or-lda/
# https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/

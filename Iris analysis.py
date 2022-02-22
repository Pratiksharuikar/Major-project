#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[128]:


df = pd.read_excel('iris2.xlsx')
df


# In[129]:


df.info()


# In[130]:


df.shape


# In[131]:


#to describe stat of data
df.describe(include='all')


# In[132]:


#to display no. of samples on each class
df['Species'].value_counts()


# processing the dataset

# In[133]:


#check for null values
df.isnull()


# EDA

# In[134]:


df['SepalLengthCm'].hist()


# In[135]:


df['SepalWidthCm'].hist()


# In[136]:


df['PetalLengthCm'].hist()


# In[137]:


df['PetalWidthCm'].hist()


# visualisation on target columns

# In[138]:


plt.title('Species Count') # this further tells that our dataset is balanced with equal records for all 3 species
sns.countplot(df['Species']);


# visualizing relations between variables 

# In[139]:


sns.pairplot(df,hue="Species",height=3);


# In[140]:


df.corr()


# In[141]:


fig=plt.figure(figsize=(15,9))
sns.heatmap(df.corr(),cmap='Blues',annot=True);


# In[142]:


#we are going to plot scatter plot for each class , 3 classes are there
colors=['red','blue','yellow']
species=['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[143]:


for i in range (3):
    x=df[df['Species']==species[i]]   #filter the point for each class
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=colors[i],label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[144]:


for i in range (3):
    x=df[df['Species']==species[i]]   #filter the point for each class
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()


# correlation matrix

# In[145]:


df.corr()


# In[146]:


df.corr(method='pearson')


# In[147]:


sns.boxplot(x='SepalWidthCm',data=df,color='orange')


# In[148]:


import sklearn
from sklearn.datasets import load_boston


# In[149]:


Q1=np.percentile(df['SepalWidthCm'], 25,interpolation='midpoint')
Q3=np.percentile(df['SepalWidthCm'], 75,interpolation='midpoint')
IQR=Q3-Q1
print(IQR)
print("old shape:",df.shape)


# In[150]:


Upper=np.where(df['SepalWidthCm']>=(Q3+1.5*IQR))
lower=np.where(df['SepalWidthCm']<=(Q1-1.5*IQR))
print(Upper) 
print(lower)


# In[151]:


df.drop(Upper[0],inplace=True)
df.drop(lower[0],inplace=True)
print("new:",df.shape)
sns.boxplot(x='SepalWidthCm',data=df)


# In[154]:


from sklearn.model_selection import train_test_split
#train-70
#test-30
X=df.drop(columns=['Species'])
Y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.30)


# In[155]:


#logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[156]:


model.fit(x_train,y_train)


# In[159]:


#print metric to get performance
print("Accuracy:",model.score(x_test,y_test)*100)


# In[160]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)


# In[161]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[162]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:





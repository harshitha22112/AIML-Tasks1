#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[3]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[7]:


iris.info()


# In[11]:


iris.isnull().sum()


# In[13]:


iris.describe()


# In[15]:


iris.head()


# In[25]:


iris.duplicated()


# ### Observations
# - There are no null values
# -There are 50 varities in the y-column(classes)
# - There are 150 rows and 5 columns
# - The y-column is categorical and all x-columns are continuos

# ### Transform the y-cloumn to categorical using LabelEncoder()

# In[30]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])


# In[32]:


iris


# In[ ]:





# In[ ]:





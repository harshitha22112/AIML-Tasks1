#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### Clustering- Divide the universities into groups (clusters)

# In[6]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[8]:


Univ.info()


# In[10]:


Univ.isnull().sum()


# In[12]:


Univ.boxplot()


# In[14]:


Univ.describe()


# In[26]:


# Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[28]:


Univ1


# In[34]:


cols = Univ1.columns


# In[36]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





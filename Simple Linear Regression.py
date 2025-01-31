#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[74]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[76]:


# Printing the information
data1.info()


# In[78]:


data1.describe()


# In[ ]:


# Observations
- It is observed that there are two outliers are observed from right-skewed
- There are no missing values 


# In[ ]:


### Observations
- A positive correlation strength is observed between daily and sunday


# In[80]:


data1.isnull().sum()


# ### Correlation

# In[82]:


data1["daily"].corr(data1["sunday"])


# In[84]:


data1[["daily","sunday"]].corr()


# In[86]:


data1.corr(numeric_only=True)


# In[96]:


plt.scatter(data1["daily"], data1["sunday"])
plt.show()


# In[102]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[104]:


plt.show()


# In[106]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





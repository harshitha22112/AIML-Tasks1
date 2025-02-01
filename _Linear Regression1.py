#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries and Data Set

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[17]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# In[19]:


data1.info()


# In[21]:


data1.isnull().sum()


# In[23]:


data1.describe()


# In[25]:


# Boxplot for daily column
plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert= False)
plt.show()


# In[29]:


sns.histplot(data1['daily'], kde = True, stat='density',)
plt.show()


# In[31]:


sns.histplot(data1['sunday'], kde = True, stat='density',)
plt.show()


# In[35]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["sunday"], vert= False)
plt.show()


# ### Observations
# - The are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed

# Scatter plot and Correlation Strength

# In[41]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[43]:


data1["daily"].corr(data1["sunday"])


# In[45]:


data1[["daily","sunday"]].corr()


# In[47]:


data1.corr(numeric_only=True)


# ### Observations on Correlation strength
# - The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong and positive with pearson's correlation coefficient of 0.958154	 

# ### Fit a regression model

# In[53]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[57]:


model1.summary()


# In[ ]:





# In[ ]:





# In[ ]:





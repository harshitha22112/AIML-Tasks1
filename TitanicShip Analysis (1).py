#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Install mLxted library
get_ipython().system('pip install mlxtend')


# In[7]:


# Import libraries
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[12]:


# print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# In[14]:


titanic.info()


# In[16]:


titanic.isnull().sum()


# In[18]:


titanic.describe()


# ### Observations
# - There are no missing values.
# - All datatypes are objective and categorical.
# - All columns are object datatype and categorical.
# - As the columns are categorical, we can adopt one hot encoding

# In[34]:


# Plot a bar chart to visualize the category of people on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[36]:


plt.show()


# ### Observations
# - In the above bar plot,maximum travelers are crew.
# - Children are less compared to adults.
# - Least travelers are 2nd class in the bar chart.

# In[39]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[41]:


plt.show()


# In[43]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[45]:


plt.show()


# In[47]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[49]:


plt.show()


# In[51]:


# Perform one hot encoding on ategorical columns
df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[54]:


df.info()


# ### Apriori Algorithm

# In[58]:


# Apply apriori algorithm to get itemset combinations
frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[60]:


frequent_itemsets.info()


# In[64]:


# Generate association rules with metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[66]:


rules.sort_values(by='lift', ascending=True)


# In[70]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





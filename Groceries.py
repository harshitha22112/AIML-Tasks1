#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ans


# In[23]:


groceries_df = pd.read_csv("Groceries_dataset.csv")
groceries_df


# In[25]:


groceries_df.head()


# In[27]:


groceries_df.tail()


# In[29]:


groceries_df.info()


# In[31]:


groceries_df.describe()


# In[33]:


groceries_df.isnull().sum()


# In[17]:


df = pd.get_dummies(groceries,dtype=int)
df.head()


# In[39]:


groceries_df.duplicated()


# In[45]:


counts = groceries_df['Member_number'].value_counts()
plt.bar(counts.index, counts.values)


# In[47]:


counts = groceries_df['Date'].value_counts()
plt.bar(counts.index, counts.values)


# In[49]:


counts = groceries_df['itemDescription'].value_counts()
plt.bar(counts.index, counts.values)


# In[51]:


print(groceries_df.Member_number.unique())
len(groceries_df.Member_number.unique())


# In[57]:


from mlxtend.frequent_patterns import apriori

# Ensure df is a binary DataFrame
df = df.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True, max_len=None)
print(frequent_itemsets)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





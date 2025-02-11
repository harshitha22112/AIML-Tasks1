#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### Clustering- Divide the universities into groups (clusters)

# In[3]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[4]:


Univ.info()


# In[5]:


Univ.isnull().sum()


# In[6]:


Univ.boxplot()


# In[7]:


Univ.describe()


# ### Standardization of the data

# In[8]:


# Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[9]:


Univ1


# In[10]:


cols = Univ1.columns


# In[11]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df


# In[32]:


# Build 3 clusters using Kmeans cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[36]:


# Print the cluster Labels
clusters_new.labels_


# In[40]:


set(clusters_new.labels_)


# In[42]:


# Assign clusters to the univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[44]:


Univ


# In[46]:


Univ.sort_values(by = "clusterid_new")


# In[48]:


# Use groupby() to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations
# - Cluster 2 appears to be the top rated universities cluster as the cutoff score, Top10,SFRatio parameter mean values are highest.
# - Cluster 1 appears to occupy the middle level rated universities.
# - Cluster 0 comes as the lower level rated universities.

# In[56]:


### Find optimal k value using elbow plot
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel("Number of clusters")
plt.ylabel('WCSS')
plt.show()


# ### Observations
# - From the above graph, we choose k=3 or 4 which indicates the elbow joint that is the rate of change of slope decreases.

# In[ ]:





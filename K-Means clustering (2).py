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

# In[9]:


# Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[10]:


Univ1


# In[11]:


cols = Univ1.columns


# In[12]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df


# In[13]:


# Build 3 clusters using Kmeans cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[14]:


# Print the cluster Labels
clusters_new.labels_


# In[15]:


set(clusters_new.labels_)


# In[16]:


# Assign clusters to the univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[17]:


Univ


# In[18]:


Univ.sort_values(by = "clusterid_new")


# In[19]:


# Use groupby() to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations
# - Cluster 2 appears to be the top rated universities cluster as the cutoff score, Top10,SFRatio parameter mean values are highest.
# - Cluster 1 appears to occupy the middle level rated universities.
# - Cluster 0 comes as the lower level rated universities.

# In[21]:


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

# In[48]:


# Quality of clusters is expressed in terms of silhouette score
from sklearn.metrics import silhouette_score
score = silhouette_score(scaled_Univ_df, clusters_new.labels_, metric='euclidean')
score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





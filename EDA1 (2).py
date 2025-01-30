#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


# Printing the information
data.info()


# In[4]:


# Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[5]:


# Drop duplicate column(Temp c) and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"], axis=1)
data1


# In[6]:


data1.info()


# In[7]:


# Covert the month column data typr to float data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


# Print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[9]:


# Data duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[10]:


# Checking for the duplicated rows in the table 
# Print only the duplicated row(one) only
data1[data1.duplicated()]


# In[11]:


# Change column names - Rename the columns
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[12]:


# Display data1 info()
data1.info()


# In[13]:


# Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[14]:


# Visualize data1 missing values using heat map
cols = data1.columns
colors = ['black', 'silver']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[15]:


# Find the mena and median values of each numeric 
# Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[16]:


# Replace the ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1


# In[17]:


mean_solar = data1["Solar"].mean()
print("Mean of Solar: ", mean_solar)


# In[18]:


data1['Solar'] = data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[19]:


median_solar = data1["Solar"].median()
print("Median of Solar: ", median_solar)


# In[20]:


data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()


# In[21]:


# Find the mode values of categorical column (weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[22]:


# Impute missing values (replace NaN with mode etc) of weather using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[23]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[24]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[25]:


# Create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 3]})

# Plot the boxplot in the first top subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

# Plot the histogram with KDE curve in the second bottom subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")

# Adjust Layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[26]:


# Create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 3]})

# Plot the boxplot in the first top subplot
sns.boxplot(data=data1["Solar"], ax=axes[0], color='brown', width=0.5, orient= 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

# Plot the histogram with KDE curve in the second bottom subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='lightgreen', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")

# Adjust Layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[27]:


# No outliers are observed from the boxplot (solar column)
# Extreme values in the column compared to other (OUTLIERS)
# Identifying the outliers through boxplot and histogram


# In[28]:


# Create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='purple')
# Show the plot
plt.show()


# In[29]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[30]:


# Extract outliers from the boxplot for ozone column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert = False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[31]:


# Using mu +/-3* sigma limits (standard deviation method)
data1["Ozone"].describe()


# In[32]:


# Extraction of outliers
mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[33]:


# Observations
- It is observed that only two outliers are identified
- In box plot method more no of outliers are idetified
- This is because the assumption of normality is not satisfied in this column


# In[34]:


# Quantile-Quantile plot for detection of outliers
import scipy.stats as stats
# Create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoritical Quantiles", fontsize=12)


# In[ ]:


# Observations from Q-Q plot
- The data does not follow normal distributions as the data points are deviating significantly away from the red line 
- The data shows a right-skewed distribution and possible outliers


# In[36]:


# Quantile-Quantile plot for detection of outliers
import scipy.stats as stats
# Create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoritical Quantiles", fontsize=12)


# In[38]:


# Create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='blue')
plt.title("Violin Plot")
# Show the plot
plt.show()


# In[40]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone", color = "orange", palette="Set2", size = 6)


# In[42]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone", color = "orange", palette="Set1", size = 6, jitter = True)


# In[44]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[46]:


# Category wise boxplot for ozone
sns.boxplot(data=data1, x = "Weather", y = "Ozone")


# In[48]:


# Correlation coefficient and pair plots
plt.scatter(data1["Wind"], data1["Temp"])


# In[50]:


# Compute pearson correlation coefficient between wind speed and temperature
data1["Wind"].corr(data1["Temp"])


# In[52]:


# Read all numeric columns into a new table
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[54]:


data1.head()


# In[ ]:


# Observation 
- The correlation between wind and temp is observed to be negatively correlated with mild strength


# In[56]:


# Read all numeric (float) columns into a new table data1_numeric
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[60]:


# Print correlation coefficients foe all the above columns
data1_numeric.corr()


# In[ ]:


# Observations
- The correlation between ozone and wind is (-0.523738)
- The highest correlation strength is observed between ozone and temp  (0.597087)
- The highest correlation strength is observed between wind and temp (-0.441228)
- The least correlation strength is observed between solar and wind (-0.055874)


# In[63]:


# Plot a pair plot all numeric columns  using seaborn
sns.pairplot(data1_numeric)


# In[65]:


# Creating dummy variable for weather column
data2=pd.get_dummies(data1,columns=['Month','Weather'])
data2


# In[67]:


data2=pd.get_dummies(data1,columns=['Weather'])
data2


# In[69]:


data1_numeric.values


# In[71]:


# Normalization of the data
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
array = data1_numeric.values
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array)

# Transformed data
set_printoptions(precision=2)
print(rescaledX[0:10,:])


# In[ ]:





# In[ ]:





# In[ ]:





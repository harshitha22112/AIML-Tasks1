#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


# Rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### Description of columns
# - MPG: Milege of the car(Mile per Gallon) Y-column to be predicted
# - HP: Horse power of the car(X1 column)
# - VOL: Volume of the car(size)(X2 column)
# - SP: Top speed of the car (Miles per Hour)(X3 column)
# - WT: Weight of the car (pounds)(X4 column)

# ### Assumptions in Multilinear Regression
# 1. Linearity: The relationship between the predictors(X) and the response(Y) is linear.
# 2. Independence: Observations are independent of each other.
# 3. Homoscedasticity: The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Errors: The residuals of the model are normally distributes.
# 5. No Multicollinearity: The independent variables should npt be too highly correlated with each other.
# 
# Violatios of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.

# ### EDA

# In[7]:


cars.info()


# In[8]:


# Check for missing values
cars.isna().sum()


# ### Observations
# - There are no missing values
# - There are 81 observations (diff cars data)
# - The data types of the columns are also relevant and valid

# In[10]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
# CReating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
# Addjust layout
plt.tight_layout()
plt.show()


# In[11]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
# CReating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
# Addjust layout
plt.tight_layout()
plt.show()


# In[12]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
# CReating a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
# Addjust layout
plt.tight_layout()
plt.show()


# ### Observations from boxplot and histograms
# - There are some extreme values observed towards the right tail of SP and HP dustributions.
# - In VOL and WT columns, a few outliers are observed in both tails of thier distributions.
# - The extreme values of cars data may have come from the specially designed nature of cars.
# - In multi dimensional data, the outliers that are spatial dimensions may have to be considered while builiding the regression model.

# ### Checking for dulpicated rows

# In[28]:


cars[cars.duplicated()]


# ### Pair plots and correaltion coefficients

# In[17]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[33]:


cars.corr()


# ### Observations from pair plot 
# - The highest correlation is observed between WT and VOL (0.999203)
# - The correlation is observed between SP and HP (0.973848)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





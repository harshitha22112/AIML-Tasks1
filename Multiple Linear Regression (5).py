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

# In[15]:


cars[cars.duplicated()]


# ### Pair plots and correaltion coefficients

# In[17]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[18]:


cars.corr()


# ### Observations from pair plot 
# - The highest correlation is observed between WT and VOL (0.999203)
# - The correlation is observed between SP and HP (0.973848)

# ### Observations
# - Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG.
# - Therefore, the dataset qualifies for building a multiple linear regression model to predict MPG.
# - aMONG X COLUMNS(X1,X2,X3 AND X4), some high correlation strengths are observed between SP vs HP, VOL vs WT.
# - The high correlation among x columns is not desirable as ti might lead to multicollinearity problem.

# In[21]:


#build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()


# In[22]:


model1.summary()


# ### Observations from model summary
# The R-squared and adjusted R-squared values are good and about 75% of variablity in Y is explained by X xolumns
# The probability value with respect to F-statistic is close to zero, indicating that all or someof x columns are significant
# The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# ### Performance metrics and model1
# 

# In[25]:


# Find the performance metrics
# Create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[26]:


# Predict for the given x data columns
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[27]:


# Compute the Mean Squared Error(MSE), RMSE for model1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# ### Checking for multicollinearity among x colums using VIF method

# In[29]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### observations for VIF values:
# - The ideal range of VIF values shall be between 0 to 10.However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOl and Wt, it is clear that they are prone to multicollinearity proble.
# - Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity
# - It is decided to drop WT and retain VOL column in further models.

# In[31]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[32]:


model2=smf.ols('MPG~HP+VOL+SP', data=cars1).fit()
model2.summary()


# ### Performanace metrics for model2

# In[34]:


# find the perfomance metrics
#  create a data frame with actual y nad predicted y columns

df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[35]:


# Predict for the given X data columns
pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[36]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# ### Observations from model2 summary()
# - The adjusted R-squared value improved slightly to 0.76
# - All the p-values for model parameters are less than 5% hence they are significant.
# - Threfore the HP, VOL, SP columns are finalized as the significant predictor for the MPG response variable.
# - There is no improvement in MSE value
# 

# #### Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[40]:


# DEfine variables amd assign values
k = 3 
n = 81
levarage_cutoff = 3*((k + 1)/n)
levarage_cutoff


# In[48]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=0.5)
y=[i for i in range(-2,8)]
x=[levarage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# ### Observations
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influencers.
# - As their H levarage values are higher and size is higher.

# In[51]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[59]:


# Discard the data points which are influencers and reasign the row number
cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[61]:


cars2


# ### Build model3 on cars2 dataset

# In[64]:


# Rebuild the model 
model3 = smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[66]:


model3.summary()


# ### Performance metrics for model3

# In[73]:


df3 = pd.DataFrame()
df3["actual_y3"] = cars2["MPG"]
df3.head()


# In[75]:


# Predict on all x data columns
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[77]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[ ]:


# Obtain q-q plot for model3.residuals
- scatterplot for (y-y') vs y' from homoscedasticity
- normality for model3


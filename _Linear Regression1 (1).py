#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries and Data Set

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# In[4]:


data1.info()


# In[5]:


data1.isnull().sum()


# In[6]:


data1.describe()


# In[7]:


# Boxplot for daily column
plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert= False)
plt.show()


# In[8]:


sns.histplot(data1['daily'], kde = True, stat='density',)
plt.show()


# In[9]:


sns.histplot(data1['sunday'], kde = True, stat='density',)
plt.show()


# In[10]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["sunday"], vert= False)
plt.show()


# ### Observations
# - The are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed

# Scatter plot and Correlation Strength

# In[13]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[14]:


data1["daily"].corr(data1["sunday"])


# In[15]:


data1[["daily","sunday"]].corr()


# In[16]:


data1.corr(numeric_only=True)


# ### Observations on Correlation strength
# - The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong and positive with pearson's correlation coefficient of 0.958154	 

# In[18]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[19]:


model1.summary()


# ### Interpretation

# - R2 = 1 - Perfect fit(all variance explained)
# - R2 = 0 - Model does not explain any variance
# - R2 close to 1 - Good model fit
# - R2 close to 0 - Poor model fit

# In[33]:


# Plot the scatter plot and overlay the fitted straight line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
# Predicted response vector
y_hat = b0 + b1*x
# Plotting the regression line
plt.plot(x, y_hat, color = "g")
# Putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ### Observations for model summary
# - The probability(p-value) for intercept (beta_0) is 0.707>0.05
# - Therefore the intercept coefficient may not be thet much significant in prediction
# - However the p-value for "daily" (beta_1) is 0.00<0.05
# - Therefore the beta_1 coefficient is highly significant and is contributint to prediction

# In[37]:


# Print the fitted coefficients (beta_0 and beta_1)
model1.params


# In[39]:


# Print the model statistics (t and p-values)
print(f'model t-values:\n{model1.tvalues}\n---------------------\nmodel p-values: \n{model1.pvalues}')


# In[52]:


# Print the quality of fitted line(R2 values)
(model1.rsquared,model1.rsquared_adj)


# ### predict for new data point

# In[48]:


# Predict for 200 and 300 daily conversation
newdata=pd.Series([200,300,1500])


# In[50]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[56]:


# Predict on all given training data
pred = model1.predict(data1["daily"])
pred


# In[58]:


# Add predicted values as a column in data1
data1["Y_hat"] = pred
data1


# In[60]:


# Compute the error values (residuals) and add as another column
data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[64]:


# Compute Mean Squared Error for the model 
mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





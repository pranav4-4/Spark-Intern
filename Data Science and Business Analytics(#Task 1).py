#!/usr/bin/env python
# coding: utf-8

# #Task 1:- Prediction using Supervised ML Linear Regression with Python 
# 
# By Pranav
# 

# In[16]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# In[2]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[7]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Preparing Data

# In[8]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# ## Training Algorithm

# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[11]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X,y)
plt.plot(X, line);
plt.show()


# # Making the predictions

# In[12]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test)


# In[13]:


# Comparing
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[14]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[15]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:





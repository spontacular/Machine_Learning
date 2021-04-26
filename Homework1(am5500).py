#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (15,12)
pd.set_option('display.max_rows', None)


# ## Importing Datasets

# In[2]:


x_train = pd.read_csv('hw1-data/X_train.csv', header = None)
y_train = pd.read_csv('hw1-data/y_train.csv', header = None)
x_test = pd.read_csv('hw1-data/X_test.csv', header = None)
y_test = pd.read_csv('hw1-data/y_test.csv', header = None)


# ## Solution

# ### Part 1 (a)

# In[3]:


#Intentional typing error to avoid key-word
lamda = range(5001)


# In[4]:


#Calculating W for Ridge Regression and SVD Matrices
identity = np.identity(x_train.shape[1])
wrr = []
svd = []
for i in lamda:
    temp = np.linalg.inv((i*identity) + x_train.transpose().dot(x_train))
    temp2 = temp.dot(x_train.transpose()).dot(y_train.values)
    wrr.append(temp2)
    u, s, v = np.linalg.svd(x_train)
    temp3 = np.sum((s*s) / ((s*s) + i))
    svd.append(temp3)


# In[5]:


#Printing the W for Ridge Regression for all Lambdas
pd.DataFrame(np.asarray(wrr).reshape(5001,7))


# In[6]:


#Plotting Degrees of Freedom vs Weight
wrr_np = np.asarray(wrr)
svd_np = np.asarray(svd)
plt.figure()
dimensions = ["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Year Made", "Bias"]
for i in range(wrr_np[0].shape[0]):
    plt.plot(svd_np, wrr_np[:,i], linestyle = 'dashed', linewidth = 0.5)
    plt.scatter(svd_np, wrr_np[:,i], s = 10, label = dimensions[i])
plt.axhline(y = 0, color = 'black', linewidth = 0.7)
plt.axvline(x = 0, color = 'black', linewidth = 0.7)
plt.xlabel('df($\lambda$)')
plt.ylabel('weight')
plt.title('Degrees of Freedom')
plt.legend()
plt.show()


# ### Part 1 (b)

# #### The 2 dimensions that stand out are 'Year Made' and 'Weight'. 'Year Made' has a high positive weight and thus indicates that newer cars have higher Miles per Gallon. 'Weight' has a high negative weight and thus indicates that lighter cars have higher Miles per Gallon. 

# ### Part 1 (c)

# In[7]:


#Making Predictions for lambdas till 50
y_pred = []
for w in wrr[:51]:
    y_pred.append(x_test.values.dot(w))


# In[8]:


#Calculating RMSE using formula given in Assignment Sheet
rmse = []
for y in y_pred:
    rmse.append(np.sqrt((np.sum((y - y_test)*(y - y_test)))/42).values.item(0))


# In[9]:


#Plotting RMSE vs Lambda
plt.plot(lamda[:51], rmse, linestyle = 'dashed', linewidth = 0.5)
plt.scatter(lamda[:51], rmse, s = 10)
plt.xlabel('$\lambda$')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error')
plt.show()


# #### Based on this figure, we can see that the optimal Lambda for this problem having the lowest RMSE is 0. We also know that for Lambda = 0, W for Ridge Regression is the same as W for Least Squares. Hence, based on this graph we should choose Least Squares for this problem.

# ### Part 1 (d)

# #### Polynomial Order = 1

# In[10]:


#Making Predictions for lambdas till 100
y_pred = []
for w in wrr[:101]:
    y_pred.append(x_test.values.dot(w))
rmse_1 = []
for y in y_pred:
    rmse_1.append(np.sqrt((np.sum((y - y_test)*(y - y_test)))/42).values.item(0))


# #### Polynomial Order = 2

# In[11]:


#Creating additional dimensions of 2nd order for Training Set and standardising them
temp = (x_train*x_train).values
temp = temp[:,:6]
mean_2 = np.mean(temp, axis = 0)
std_2 = np.std(temp, axis = 0)
temp = temp - mean_2
temp = temp/std_2


# In[12]:


#Creating final dataframe of 2nd order by concatenating with original dataframe
x_train_2 = np.hstack([x_train.values[:,:6], temp, np.ones((350,1))])


# In[13]:


#Checking shape of final dataframe
x_train_2.shape


# In[14]:


#Calculating W for Ridge Regression and SVD Matrices
identity = np.identity(x_train_2.shape[1])
wrr_2 = []
svd_2 = []
for i in lamda:
    temp = np.linalg.inv((i*identity) + x_train_2.transpose().dot(x_train_2))
    temp2 = temp.dot(x_train_2.transpose()).dot(y_train.values)
    wrr_2.append(temp2)
    u, s, v = np.linalg.svd(x_train_2)
    temp3 = np.sum((s*s) / ((s*s) + i))
    svd_2.append(temp3)


# In[15]:


#Creating additional dimensions of 2nd order for Test Set and standardising them
temp = (x_test*x_test).values
temp = temp[:,:6]
temp = temp - mean_2
temp = temp/std_2


# In[16]:


#Creating final dataframe of 2nd order by concatenating with original dataframe
x_test_2 = np.hstack([x_test.values[:,:6], temp, np.ones((42,1))])


# In[17]:


#Making Predictions for lambdas till 100
y_pred = []
for w in wrr_2[:101]:
    y_pred.append(x_test_2.dot(w))
rmse_2 = []
for y in y_pred:
    rmse_2.append(np.sqrt((np.sum((y - y_test)*(y - y_test)))/42).values.item(0))


# #### Polynomial Order = 3

# In[18]:


#Creating additional dimensions of 2nd order for Training Set and standardising them
temp = (x_train*x_train).values
temp = temp[:,:6]
mean_2 = np.mean(temp, axis = 0)
std_2 = np.std(temp, axis = 0)
temp = temp - mean_2
temp = temp/std_2


# In[19]:


#Creating additional dimensions of 3rd order for Training Set and standardising them
temp2 = (x_train*x_train*x_train).values
temp2 = temp2[:,:6]
mean_3 = np.mean(temp2, axis = 0)
std_3 = np.std(temp2, axis = 0)
temp2 = temp2 - mean_3
temp2 = temp2/std_3


# In[20]:


#Creating final dataframe of 2nd order and 3rd order by concatenating with original dataframe
x_train_3 = np.hstack([x_train.values[:,:6], temp, temp2, np.ones((350,1))])


# In[21]:


#Checking shape of final dataframe
x_train_3.shape


# In[22]:


#Calculating W for Ridge Regression and SVD Matrices
identity = np.identity(x_train_3.shape[1])
wrr_3 = []
svd_3 = []
for i in lamda:
    temp = np.linalg.inv((i*identity) + x_train_3.transpose().dot(x_train_3))
    temp2 = temp.dot(x_train_3.transpose()).dot(y_train.values)
    wrr_3.append(temp2)
    u, s, v = np.linalg.svd(x_train_3)
    temp3 = np.sum((s*s) / ((s*s) + i))
    svd_3.append(temp3)


# In[23]:


#Creating additional dimensions of 2nd order for Test Set and standardising them
temp = (x_test*x_test).values
temp = temp[:,:6]
temp = temp - mean_2
temp = temp/std_2


# In[24]:


#Creating additional dimensions of 3rd order for Test Set and standardising them
temp2 = (x_test*x_test*x_test).values
temp2 = temp2[:,:6]
temp2 = temp2 - mean_3
temp2 = temp2/std_3


# In[25]:


#Creating final dataframe of 2nd order and 3rd order by concatenating with original dataframe
x_test_3 = np.hstack([x_test.values[:,:6], temp, temp2, np.ones((42,1))])


# In[26]:


#Making Predictions for lambdas till 100
y_pred = []
for w in wrr_3[:101]:
    y_pred.append(x_test_3.dot(w))
rmse_3 = []
for y in y_pred:
    rmse_3.append(np.sqrt((np.sum((y - y_test)*(y - y_test)))/42).values.item(0))


# In[27]:


#Finding Lambda for smallest RMSE
rmse_3.index(min(rmse_3))


# #### RMSE Plot for different Polynomial Orders

# In[28]:


#Plotting RMSE vs Lambda for different Polynomial Orders
plt.plot(lamda[:101], rmse_1, linestyle = 'dashed', linewidth = 0.5)
plt.scatter(lamda[:101], rmse_1, s = 10, label = 'Polynomial Order 1')
plt.plot(lamda[:101], rmse_2, linestyle = 'dashed', linewidth = 0.5)
plt.scatter(lamda[:101], rmse_2, s = 10, label = 'Polynomial Order 2')
plt.plot(lamda[:101], rmse_3, linestyle = 'dashed', linewidth = 0.5)
plt.scatter(lamda[:101], rmse_3, s = 10, label = 'Polynomial Order 3')
plt.xlabel('$\lambda$')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error')
plt.legend()
plt.show()


# #### Based on this plot, we should choose a Polynomial of Order 3 (p = 3) as it has the least RMSE overall. Our assessment of the ideal value of Lambda also changes to 51 from our initial value of 0 as the lowest RMSE value for p = 2 occurs at this point.

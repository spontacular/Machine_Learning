#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import math
from math import exp, pow, factorial
from scipy.stats import poisson
from scipy.special import expit
import matplotlib.pyplot as plt 
import seaborn as sns


# ## Problem 2

# ### Part a - Naive Bayes

# In[2]:


df_2_x = pd.read_csv('hw2-data/Bayes_classifier/X.csv', header = None)
df_2_y = pd.read_csv('hw2-data/Bayes_classifier/y.csv', header = None)


# In[3]:


df_2_x_np = df_2_x.to_numpy()
df_2_y_np = df_2_y.to_numpy()


# In[4]:


def my_prior(y, i):
    if i == 0:
        return 1 - np.mean(y)
    elif i == 1:
        return np.mean(y)
    else:
        return None


# In[5]:


def my_lamda(x, y, i):
    x_0 = np.take(x, np.where(y.flatten() == 0), 0)
    y_0 = x_0.shape[1]
    x_0 = x_0.reshape(x_0.shape[1], x_0.shape[2]).sum(axis=0).reshape(-1,1)
    x_1 = np.take(x, np.where(y.flatten() == 1), 0)
    y_1 = x_1.shape[1]
    x_1 = x_1.reshape(x_1.shape[1], x_1.shape[2]).sum(axis=0).reshape(-1,1)
    if i == 0:
        return (1 + x_0)/(1 + y_0)
    elif i == 1:
        return (1 + x_1)/(1 + y_1)
    else:
        return None


# In[6]:


def my_poisson(x, lamda):
    return poisson.pmf(x, lamda)


# In[7]:


def my_likelihood(x, lamda):
    temp = x.astype(float).copy()
    for i in range(x.shape[1]):
        temp[:, i] = np.asarray(np.vectorize(my_poisson)(x[:, i], lamda[i]))
    temp = np.multiply.reduce(temp, axis=1)
    return temp


# In[8]:


def my_shuffler(x, y):
    shuffler = np.random.permutation(x.shape[0])
    return x[shuffler], y[shuffler]


# In[9]:


def my_naive_bayes(x, y, splits):
    prediction_all = []
    y_all = []
    x_shuffle, y_shuffle = my_shuffler(x, y)
    lamda_0_average = np.ones((x.shape[1], splits))
    lamda_1_average = np.ones((x.shape[1], splits))
    
    for i in range(splits):
        size = int(x.shape[0]/splits)
        start = int(i*size)
        end = int((i+1)*size)
        x_test = x_shuffle[start:end,:]
        y_test = y_shuffle[start:end,:]
        x_train = np.delete(x_shuffle, slice(start,end), axis=0)
        y_train = np.delete(y_shuffle, slice(start,end), axis=0)
        prior_0 = my_prior(y_train, 0)
        prior_1 = my_prior(y_train, 1)
        lamda_0 = my_lamda(x_train, y_train, 0)
        lamda_1 = my_lamda(x_train, y_train, 1)
        lamda_0_average[:, i] = list(lamda_0)
        lamda_1_average[:, i] = list(lamda_1)
        likelihood_0 = my_likelihood(x_test, lamda_0)
        likelihood_1 = my_likelihood(x_test, lamda_1)
        posterior_0 = prior_0*likelihood_0
        posterior_1 = prior_1*likelihood_1
        prediction = np.reshape((posterior_0 < posterior_1).astype(int), (-1, 1))
        prediction_all.append(prediction)
        y_all.append(y_test)
    
    lamda_0_average = np.mean(lamda_0_average, axis = 1).reshape(-1,1)
    lamda_1_average = np.mean(lamda_1_average, axis = 1).reshape(-1,1)
    prediction_all = pd.Series(np.asarray(prediction_all).reshape(x_shuffle.shape[0], 1).flatten(), name = 'Predicted')
    y_all = pd.Series(np.asarray(y_all).reshape(y_shuffle.shape[0], 1).flatten(), name = 'Actual')
    df_confusion = pd.crosstab(y_all, prediction_all)
    return prediction_all, y_all, df_confusion, lamda_0_average, lamda_1_average


# In[10]:


df_2_x_np_nb = df_2_x_np.copy()
df_2_y_np_nb = df_2_y_np.copy()
prediction_all_nb, y_all_nb, df_confusion_nb, lamda_0_average, lamda_1_average = my_naive_bayes(df_2_x_np_nb, df_2_y_np_nb, 10)


# In[11]:


accuracy = "\033[1m" + 'Prediction Accuracy = ' + str((df_confusion_nb[0][0] + df_confusion_nb[1][1])/df_confusion_nb.sum().sum()) + "\033[0m"
print(accuracy)
df_confusion_nb


# ### Part b - Naive Bayes

# In[12]:


labels = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '0', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', ';', '(', '[', '!', '$', '#']
numbers = list(range(1, 55))


# In[13]:


fig = plt.figure()
fig.set_figheight(12)
fig.set_figwidth(18)
plt.rcParams["figure.figsize"] = (20,10)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Stem plot for \u03BB = 0')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Weight')
ax1.stem(numbers, lamda_0_average)
ax2.set_title('Stem plot for \u03BB = 1')
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Weight')
ax2.stem(numbers, lamda_1_average)
plt.show()


# In[14]:


lamda_0_subset = [lamda_0_average[15].item(), lamda_0_average[51].item()]
lamda_1_subset = [lamda_1_average[15].item(), lamda_1_average[51].item()]
label_subset = [labels[15], labels[51]]


# In[15]:


barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(lamda_0_subset))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, lamda_0_subset, color='#7f6d5f', width=barWidth, edgecolor='white', label='\u03BB = 0')
plt.bar(r2, lamda_1_subset, color='#557f2d', width=barWidth, edgecolor='white', label='\u03BB = 1')
 
# Add xticks on the middle of the group bars
plt.title('Comparing 16th and 52nd Dimensions')
plt.xlabel('Term', fontweight='bold', fontsize=15)
plt.ylabel('Weight', fontweight='bold', fontsize=15)
plt.xticks([r + (barWidth/2) for r in range(len(lamda_0_subset))], label_subset, fontsize=13)
 
# Create legend & Show graphic
plt.legend()
plt.show()


# #### 16th and 52nd Dimension refer to the terms 'free' and '!'. These terms are seen having more weight for the case  $\lambda$ = 1 than $\lambda$ = 0. This is expected as the terms 'free' and '!' are more likely to present in spam emails.

# ### Part c - Logistic Regression

# In[16]:


df_2_x_np_lr = df_2_x_np.copy()
df_2_y_np_lr = df_2_y_np.copy()
df_2_x_np_lr = np.hstack((df_2_x_np_lr, np.ones((df_2_x_np_lr.shape[0], 1))))
df_2_y_np_lr = np.where(df_2_y_np_lr == 0, -1, df_2_y_np_lr)


# In[17]:


def my_sigmoid(x):
    return 1 / (1 + exp(-x))


# In[18]:


def my_logistic_regression_steepest_ascent(x, y, splits, iterations):
    eta = 0.01/4600
    prediction_all = []
    objective_all = np.zeros((iterations, splits))
    y_all = []
    x_shuffle, y_shuffle = my_shuffler(x, y)
    
    for i in range(splits):
        size = int(x.shape[0]/splits)
        start = int(i*size)
        end = int((i+1)*size)
        x_test = x_shuffle[start:end,:]
        y_test = y_shuffle[start:end,:]
        x_train = np.delete(x_shuffle, slice(start,end), axis=0)
        y_train = np.delete(y_shuffle, slice(start,end), axis=0)
        weights = np.zeros((x_train.shape[1], 1))
        
        for j in range(iterations):     
            sigma_y_w = np.vectorize(my_sigmoid)((y_train*np.dot(x_train, weights)))
            objective = np.sum(np.log(sigma_y_w))
            update = np.dot(x_train.T, (y_train*(1 - sigma_y_w)))
            objective_all[j, i] = objective
            weights = weights + eta*update
        
        prediction = np.sign(np.dot(x_test, weights))
        prediction_all.append(prediction)
        y_all.append(y_test)
        
    prediction_all = pd.Series(np.asarray(prediction_all).reshape(x_shuffle.shape[0], 1).flatten(), name = 'Predicted')
    y_all = pd.Series(np.asarray(y_all).reshape(y_shuffle.shape[0], 1).flatten(), name = 'Actual')
    df_confusion = pd.crosstab(y_all, prediction_all)
    return objective_all, prediction_all, y_all, df_confusion


# In[19]:


objective_all_sa, prediction_all_sa, y_all, df_confusion_sa = my_logistic_regression_steepest_ascent(df_2_x_np_lr, df_2_y_np_lr, 10, 1000)


# In[20]:


accuracy = "\033[1m" + 'Prediction Accuracy = ' + str((df_confusion_sa[-1][-1] + df_confusion_sa[1][1])/df_confusion_sa.sum().sum()) + "\033[0m"
print(accuracy)
df_confusion_sa


# In[21]:


objective_all_sa_df = pd.DataFrame(objective_all_sa)
objective_all_sa_df.columns = [1,2,3,4,5,6,7,8,9,10]
ax = plt.gca()
ax.set_title('Objective Training Function vs Number of Iterations by Test Group')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Objective Training Function')
objective_all_sa_df.plot(kind='line', ax = ax)
plt.show()


# ### Part - d Newton's Method

# In[22]:


def my_logistic_regression_newtons_method(x, y, splits, iterations):
    eta = 1
    prediction_all = []
    objective_all = np.zeros((iterations, splits))
    y_all = []
    x_shuffle, y_shuffle = my_shuffler(x, y)
    
    for i in range(splits):
        size = int(x.shape[0]/splits)
        start = int(i*size)
        end = int((i+1)*size)
        x_test = x_shuffle[start:end,:]
        y_test = y_shuffle[start:end,:]
        x_train = np.delete(x_shuffle, slice(start,end), axis=0)
        y_train = np.delete(y_shuffle, slice(start,end), axis=0)
        weights = np.zeros((x_train.shape[1], 1))
        
        for j in range(iterations):     
            sigma_y_w = np.vectorize(my_sigmoid)((y_train*np.dot(x_train, weights)))
            objective = np.sum(np.log(sigma_y_w))
            first_gradient = np.dot(x_train.T, (y_train*(1 - sigma_y_w)))
            second_gradient = -np.dot(((sigma_y_w*(1 - sigma_y_w))*x_train).T, x_train)
            objective_all[j, i] = objective
            weights = weights - eta*(np.dot(np.linalg.pinv(second_gradient), first_gradient))
        
        prediction = np.sign(np.dot(x_test, weights))
        prediction_all.append(prediction)
        y_all.append(y_test)
        
    prediction_all = pd.Series(np.asarray(prediction_all).reshape(x_shuffle.shape[0], 1).flatten(), name = 'Predicted')
    y_all = pd.Series(np.asarray(y_all).reshape(y_shuffle.shape[0], 1).flatten(), name = 'Actual')
    df_confusion = pd.crosstab(y_all, prediction_all)
    return objective_all, prediction_all, y_all, df_confusion


# In[23]:


objective_all_nm, prediction_all_nm, y_all_nm, df_confusion_nm = my_logistic_regression_newtons_method(df_2_x_np_lr, df_2_y_np_lr, 10, 100)


# In[24]:


accuracy = "\033[1m" + 'Prediction Accuracy = ' + str((df_confusion_nm[-1][-1] + df_confusion_nm[1][1])/df_confusion_nm.sum().sum()) + "\033[0m"
print(accuracy)
df_confusion_nm


# In[25]:


objective_all_nm_df = pd.DataFrame(objective_all_nm)
objective_all_nm_df.columns = [1,2,3,4,5,6,7,8,9,10]
ax = plt.gca()
ax.set_title('Objective Training Function vs Number of Iterations by Test Group')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Objective Training Function')
objective_all_nm_df.plot(kind='line', ax = ax)
plt.show()


# ## Question 3

# ### Part - a Gaussian Process

# In[26]:


df_3_x_train = pd.read_csv('hw2-data/Gaussian_process/X_train.csv', header = None)
df_3_x_test = pd.read_csv('hw2-data/Gaussian_process/X_test.csv', header = None)
df_3_y_train = pd.read_csv('hw2-data/Gaussian_process/y_train.csv', header = None)
df_3_y_test = pd.read_csv('hw2-data/Gaussian_process/y_test.csv', header = None)


# In[27]:


df_3_x_train_np = df_3_x_train.to_numpy()
df_3_x_test_np = df_3_x_test.to_numpy()
df_3_y_train_np = df_3_y_train.to_numpy()
df_3_y_test_np = df_3_y_test.to_numpy()


# In[28]:


b = [5, 7, 9, 11, 13, 15]
sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# In[29]:


def my_k(x, b):
    k = np.ones((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            k[i,j] = exp(np.linalg.norm(x[i] - x[j])*(-1/b))
    return k


# In[30]:


def my_gaussian_process(x_train, y_train, x_test, y_test, b, sigma):
    term_1 = my_k(np.concatenate((x_test, x_train)), b)[:x_test.shape[0], x_test.shape[0]:]
    term_2 = np.linalg.inv(sigma*np.identity(x_train.shape[0]) + my_k(x_train, b))
    term_3 = y_train
    mu = np.matmul(term_1, np.matmul(term_2, term_3))
    rmse = np.sqrt((np.sum((mu - y_test)*(mu - y_test)))/mu.shape[0])
    return mu, rmse


# In[31]:


rmse_matrix = pd.DataFrame(columns = b, index = sigma)
index = rmse_matrix.index
index.name = "Sigma^2"
column = rmse_matrix.columns
column.name = "b"
for b_i in b:
    for sigma_i in sigma:
        mu, rmse = my_gaussian_process(df_3_x_train_np, df_3_y_train_np, df_3_x_test_np, df_3_y_test_np, b_i, sigma_i)
        rmse_matrix[b_i][sigma_i] = rmse


# In[32]:


rmse_matrix


# ### Part - b Gaussian Process

# In[33]:


b_min = 0
sigma_min = 0
min = rmse_matrix.to_numpy().min()
for b_i in b:
    for sigma_i in sigma:
        if rmse_matrix[b_i][sigma_i] == min:
            b_min = b_i
            sigma_min = sigma_i
            break
print('Minimum RMSE :', min)
print('Ideal b :', b_min)
print('Ideal \u03C3^2 :', sigma_min)


# #### The minimum RMSE in Assignment 1 was 2.100110972109874 while the minimum RMSE using the Gaussian Process Approach is 1.9303087263142182. As we can see, Gaussian Process has a lower RMSE value and is thus better at making predictions. Some drawbacks of using this approach are - Higher Computation Time, Higher Prediction Variance and Difficulty in Interpreting Feature Importance (Due to Kernel based Approach).

# ### Part - c Gaussian Process

# In[34]:


rmse_matrix_weight = pd.DataFrame(columns = [5], index = [2])
index = rmse_matrix_weight.index
index.name = "Sigma^2"
column = rmse_matrix_weight.columns
column.name = "b"
mu_weight, rmse_weight = my_gaussian_process(df_3_x_train_np[:, (3)], df_3_y_train_np, df_3_x_train_np[:, (3)], df_3_y_train_np, 5, 2)
rmse_matrix_weight[5][2] = rmse_weight


# In[35]:


rmse_matrix_weight


# In[36]:


plt.scatter(df_3_x_train_np[:, (3)], df_3_y_train_np, c = 'green', label = 'Data')
plt.plot(df_3_x_train_np[:, (3)][np.argsort(df_3_x_train_np[:, (3)].reshape(-1,1).flatten())], mu_weight[np.argsort(df_3_x_train_np[:, (3)].reshape(-1,1).flatten())], c = 'red', linewidth = 5, label = 'Predictive Mean')
plt.legend()
plt.title('Data and Predictive Mean vs Standardised Car Weight')
plt.xlabel('Standardised Car Weight')
plt.ylabel('Y')
plt.show()


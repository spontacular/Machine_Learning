#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal
import matplotlib


# ## Question 1

# In[2]:


df_1_x = pd.read_csv('hw3-data/Prob1_X.csv', header = None)
df_1_y = pd.read_csv('hw3-data/Prob1_y.csv', header = None)


# In[3]:


df_1_x_np = df_1_x.to_numpy()
df_1_y_np = df_1_y.to_numpy()


# In[4]:


df_1_w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(df_1_x_np.T, df_1_x_np)), df_1_x_np.T), df_1_y_np)


# In[5]:


base_error = sum(np.sign(np.matmul(df_1_x_np, df_1_w)) != df_1_y_np)/df_1_x_np.shape[0]


# In[6]:


def linear_boosted(df_x, df_y, iterations, bootstrap):
    
    index = [i for i in range(df_x.shape[0])]
    weight = [1/df_x.shape[0] for i in range(df_x.shape[0])]
    w_all = np.ones((iterations, df_x.shape[1]))
    pred_all = np.ones((iterations, df_x.shape[0]))
    ephsilon_all = np.ones((iterations, 1))
    alpha_all = np.ones((iterations, 1))
    weight_all = np.ones((iterations, df_x.shape[0]))
    error_all = np.ones((iterations, 1))
    z_all = np.ones((iterations, 1))
    bound_all = np.ones((iterations, 1))
    output_all = np.zeros((iterations, 1))
    
    for t in range(iterations):
        weight_all[t] = np.asarray(weight).reshape(df_x.shape[0],1).flatten()
        sample = random.choices(index, weights = weight, k = bootstrap)
        x = df_x[sample]
        y = df_y[sample]
        w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.T, x)), x.T), y)
        pred = np.sign(np.matmul(df_x, w))
        ephsilon = np.sum(np.multiply(np.asarray(weight).reshape(df_x.shape[0],1), (pred != df_y).reshape(df_x.shape[0],1)))
        if ephsilon > 0.5:
            w = -1*np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.T, x)), x.T), y)
            pred = np.sign(np.matmul(df_x, w))
            ephsilon = np.sum(np.multiply(np.asarray(weight).reshape(df_x.shape[0],1), (pred != df_y).reshape(df_x.shape[0],1)))
        w_all[t] = w.flatten()
        ephsilon_all[t] = ephsilon
        pred_all[t] = pred.flatten()
        alpha = 0.5*(np.log((1-ephsilon)/ephsilon))
        alpha_all[t] = alpha
        w_hat = np.multiply(np.asarray(weight).reshape(df_x.shape[0],1), np.exp(-1*alpha*np.multiply(df_y, pred)))
        weight = np.divide(w_hat, (np.sum(w_hat)))
        z_all[t] = 2*np.sqrt(ephsilon*(1-ephsilon))
    bound_all = np.cumprod(z_all).reshape((iterations, 1))
    output_all = np.sign(np.cumsum((pred_all*alpha_all), axis = 0))
    error_all = ((np.sum((output_all.T != df_y), axis = 0))/df_x.shape[0]).reshape((iterations, 1))
    
    return w_all, pred_all, ephsilon_all, alpha_all, weight_all, error_all, z_all, bound_all


# In[7]:


w_1, pred_1, ephsilon_1, alpha_1, weight_1, error_1, z_1, bound_1 = linear_boosted(df_1_x_np, df_1_y_np, 2500, 1000)


# ### Question 1(a)

# In[8]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('AdaBoost Algorithm on Least Squares Classifier')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.plot(error_1, label = 'Empirical Training Error')
plt.plot(bound_1, label = 'Training Error Upper Bound')
plt.legend()
plt.show()


# ### Question 1(b)

# In[9]:


numbers = list(range(1, 1001))


# In[10]:


weights = np.mean(weight_1, axis = 0)


# In[11]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Average of Distribution of Data')
plt.xlabel('Weights')
plt.ylabel('Value')
plt.stem(numbers, weights)
plt.xticks(np.arange(1, 1001, 200))
plt.show()


# ### Question 1(c)

# In[12]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Ephsilon by Iteration for AdaBoost Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Ephsilon')
plt.plot(ephsilon_1, label = 'Ephsilon', color = 'green')
plt.xticks(np.arange(1, 2501, 300))
plt.show()


# In[13]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Alpha by Iteration for AdaBoost Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Alpha')
plt.plot(alpha_1, label = 'Ephsilon', color = 'red')
plt.xticks(np.arange(1, 2501, 300))
plt.show()


# ## Question 2

# In[14]:


mean1 = np.asarray([0,0])
covariance1 = np.asarray([1,0,0,1]).reshape((2,2))
mean2 = np.asarray([3,0])
covariance2 = np.asarray([1,0,0,1]).reshape((2,2))
mean3 = np.asarray([0,3])
covariance3 = np.asarray([1,0,0,1]).reshape((2,2))
dist1 = np.random.multivariate_normal(mean1, covariance1, 500)
dist2 = np.random.multivariate_normal(mean2, covariance2, 500)
dist3 = np.random.multivariate_normal(mean3, covariance3, 500)


# In[15]:


index = [1, 2, 3]
weight = [0.2, 0.5, 0.3]
sample = random.choices(index, weights = weight, k = 500)


# In[16]:


dist = np.ones((500, 2))
for i, s in enumerate(sample):
    if s == 1:
        dist[i] = dist1[i]
    elif s == 2:
        dist[i] = dist2[i]
    elif s == 3:
        dist[i] = dist3[i]


# In[17]:


def k_means(x, k, iterations):
    mu_old = x[np.random.choice(x.shape[0], k, replace=False), :]
    mu_new = x[np.random.choice(x.shape[0], k, replace=False), :]
    c_old = np.zeros((x.shape[0], 1))
    c_new = np.zeros((x.shape[0], 1))
    distance = np.zeros((x.shape[0], k))
    objective = np.zeros((iterations, 1))
    
    for i in range(iterations):
        mu_old = mu_new
        c_old = c_new
        for a in range(x.shape[0]):
            for b in range(k):
                distance[a, b] = np.linalg.norm(x[a] - mu_old[b])
        c_new = np.argmin(distance, axis=1).reshape(x.shape[0], 1)
        for c in range(k):
            mu_new[c] = np.mean(x[(c_new == c).flatten()], axis = 0)
        objective[i] = np.sum(np.min(distance, axis = 1))
        #I implemented stopping after convergence but commented it out to have uniform iterations in all graphs
        if (mu_old == mu_new).all():
            if (c_old == c_new).all():
                pass
                #print(i)
                #for d in range(i+1, iterations):
                    #objective[j] = np.sum(np.min(distance, axis = 1))
                #return mu_new, c_new, objective
    return mu_new, c_new, objective


# ### Question 2(a)

# In[18]:


mu2, c2, obj2 = k_means(dist, 2, 20)
mu3, c3, obj3 = k_means(dist, 3, 20)
mu4, c4, obj4 = k_means(dist, 4, 20)
mu5, c5, obj5 = k_means(dist, 5, 20)


# In[19]:


df2 = pd.DataFrame({'K = 2':obj2.flatten(), 'K = 3':obj3.flatten(), 'K = 4':obj4.flatten(), 'K = 5':obj5.flatten()})
df2.index = [i for i in range(1, 21)]


# In[20]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('K-Means Clustering for K = 2, 3, 4, 5')
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.plot(df2['K = 2'], label = 'K = 2')
plt.plot(df2['K = 3'], label = 'K = 3')
plt.plot(df2['K = 4'], label = 'K = 4')
plt.plot(df2['K = 5'], label = 'K = 5')
plt.legend()
plt.show()


# ### Question 2(b)

# In[21]:


mu3, c3, obj3 = k_means(dist, 3, 20)
mu5, c5, obj5 = k_means(dist, 5, 20)


# In[22]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
plt.title('Clusters for K = 3')
plt.xlabel('X [0]')
plt.ylabel('X [1]')
colors = ['red', 'green', 'blue']
plt.scatter(dist[:,0].flatten(), dist[:,1].flatten(), c = c3.flatten(), cmap = matplotlib.colors.ListedColormap(colors))
plt.show()


# In[23]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
plt.title('Clusters for K = 5')
plt.xlabel('X [0]')
plt.ylabel('X [1]')
colors = ['red', 'green', 'blue', 'purple', 'black']
plt.scatter(dist[:,0].flatten(), dist[:,1].flatten(), c = c5.flatten(), cmap = matplotlib.colors.ListedColormap(colors))
plt.show()


# ## Question 3

# In[2]:


df_3_x_train = pd.read_csv('hw3-data/Prob3_Xtrain.csv', header = None)
df_3_x_test = pd.read_csv('hw3-data/Prob3_Xtest.csv', header = None)
df_3_y_train = pd.read_csv('hw3-data/Prob3_ytrain.csv', header = None)
df_3_y_test = pd.read_csv('hw3-data/Prob3_ytest.csv', header = None)


# In[3]:


df_3_x_train_np = df_3_x_train.to_numpy()
df_3_x_test_np = df_3_x_test.to_numpy()
df_3_y_train_np = df_3_y_train.to_numpy()
df_3_y_test_np = df_3_y_test.to_numpy()


# In[4]:


df_3_x_train_np_0 = df_3_x_train_np[(df_3_y_train_np == 0).flatten()]
df_3_x_train_np_1 = df_3_x_train_np[(df_3_y_train_np == 1).flatten()]


# In[5]:


df_3_x_test_np_0 = df_3_x_test_np[(df_3_y_test_np == 0).flatten()]
df_3_x_test_np_1 = df_3_x_test_np[(df_3_y_test_np == 1).flatten()]


# In[6]:


def initialise(x, k):
    mean = np.mean(x, axis = 0)
    covariance = np.cov(x.T)
    centroids = np.random.multivariate_normal(mean, covariance, k)
    covariance = np.broadcast_to(covariance, (k,) + covariance.shape)
    mixing = np.full((1, k), 1/k)
    return centroids, mixing, covariance


# In[7]:


def e_step(x, mixing, centroids, covariance, k):
    phi = np.zeros((x.shape[0], k))
    for i in range(k):
        phi[:, i] = (mixing[:, i]*(multivariate_normal.pdf(x, centroids[i], covariance[i], allow_singular = True)))
    phi = (phi/phi.sum(axis=1, keepdims=True))
    return phi


# In[8]:


def m_step(x, phi, k):
    n = np.sum(phi, axis = 0)
    mixing = (n/(x.shape[0])).reshape((1, k))
    centroids = np.matmul(phi.T, x)/n.reshape((k,1))
    covariance = np.zeros((k, x.shape[1], x.shape[1]))
    for i in range(k):
        temp = (x - centroids[i])
        cov = (np.matmul(temp.T, (temp*(phi[:, i]).reshape((x.shape[0],1)))))
        covariance[i, :, :] = cov
    covariance = covariance/n.reshape((k,1,1))
    return mixing, centroids, covariance


# In[9]:


def get_objective(x, mixing, centroids, covariance, k):
    objective = np.zeros((x.shape[0], k))
    for i in range(k):
        objective[:, i] = (mixing[:, i]*(multivariate_normal.pdf(x, centroids[i], covariance[i], allow_singular = True)))
    objective = np.sum(np.log(np.sum(objective, axis = 1)))
    return objective


# In[10]:


def gmm(x, k, iterations, runs):
    objective = np.zeros((iterations, runs))
    objective_optimal = 0
    run_optimal = 0
    for i in range(runs):
        centroids, mixing, covariance = initialise(x, k)
        for j in range(iterations):
            phi = e_step(x, mixing, centroids, covariance, k)
            mixing, centroids, covariance = m_step(x, phi, k)
            obj = get_objective(x, mixing, centroids, covariance, k)
            objective[j, i] = obj
        if np.max(objective) > objective_optimal:
            centroids_optimal = centroids
            mixing_optimal = mixing
            covariance_optimal = covariance
            phi_optimal = phi
            objective_optimal = np.max(objective)
            run_optimal = i + 1
    return objective, centroids_optimal, mixing_optimal, covariance_optimal, phi_optimal, objective_optimal, run_optimal


# In[11]:


objective_0, centroids_optimal_0, mixing_optimal_0, covariance_optimal_0, phi_optimal_0, objective_optimal_0, run_optimal_0 = gmm(df_3_x_train_np_0, 3, 30, 10)
objective_1, centroids_optimal_1, mixing_optimal_1, covariance_optimal_1, phi_optimal_1, objective_optimal_1, run_optimal_1 = gmm(df_3_x_train_np_1, 3, 30, 10)


# ### Question 3(a)

# In[24]:


df3 = pd.DataFrame({'Run 1':objective_0[:, 0].flatten(), 'Run 2':objective_0[:, 1].flatten(), 'Run 3':objective_0[:, 2].flatten(), 'Run 4':objective_0[:, 3].flatten(), 'Run 5':objective_0[:, 4].flatten(), 'Run 6':objective_0[:, 5].flatten(), 'Run 7':objective_0[:, 6].flatten(), 'Run 8':objective_0[:, 7].flatten(), 'Run 9':objective_0[:, 8].flatten(), 'Run 10':objective_0[:, 9].flatten()})
df3.index = [i for i in range(1, 31)]
df3 = df3.iloc[4:]


# In[27]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Gaussian Mixture Model for Class 0')
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.plot(df3['Run 1'], label = 'Run 1')
plt.plot(df3['Run 2'], label = 'Run 2')
plt.plot(df3['Run 3'], label = 'Run 3')
plt.plot(df3['Run 4'], label = 'Run 4')
plt.plot(df3['Run 5'], label = 'Run 5')
plt.plot(df3['Run 6'], label = 'Run 6')
plt.plot(df3['Run 7'], label = 'Run 7')
plt.plot(df3['Run 8'], label = 'Run 8')
plt.plot(df3['Run 9'], label = 'Run 9')
plt.plot(df3['Run 10'], label = 'Run 10')
plt.xticks(np.arange(5, 31, 5))
plt.legend()
plt.show()


# In[28]:


df4 = pd.DataFrame({'Run 1':objective_1[:, 0].flatten(), 'Run 2':objective_1[:, 1].flatten(), 'Run 3':objective_1[:, 2].flatten(), 'Run 4':objective_1[:, 3].flatten(), 'Run 5':objective_1[:, 4].flatten(), 'Run 6':objective_1[:, 5].flatten(), 'Run 7':objective_1[:, 6].flatten(), 'Run 8':objective_1[:, 7].flatten(), 'Run 9':objective_1[:, 8].flatten(), 'Run 10':objective_1[:, 9].flatten()})
df4.index = [i for i in range(1, 31)]
df4 = df4.iloc[4:]


# In[29]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Gaussian Mixture Model for Class 1')
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.plot(df4['Run 1'], label = 'Run 1')
plt.plot(df4['Run 2'], label = 'Run 2')
plt.plot(df4['Run 3'], label = 'Run 3')
plt.plot(df4['Run 4'], label = 'Run 4')
plt.plot(df4['Run 5'], label = 'Run 5')
plt.plot(df4['Run 6'], label = 'Run 6')
plt.plot(df4['Run 7'], label = 'Run 7')
plt.plot(df4['Run 8'], label = 'Run 8')
plt.plot(df4['Run 9'], label = 'Run 9')
plt.plot(df4['Run 10'], label = 'Run 10')
plt.xticks(np.arange(5, 31, 5))
plt.legend()
plt.show()


# ### Question 3(b)

# In[30]:


def naive_bayes(x, mixing, centroids, covariance, k):
    prediction = np.zeros((x.shape[0], k))
    for i in range(k):
        prediction[:, i] = (mixing[:, i]*(multivariate_normal.pdf(x, centroids[i], covariance[i], allow_singular = True)))
    prediction = (np.sum(prediction, axis = 1)).reshape((x.shape[0], 1))
    return prediction


# In[31]:


def prediction(x_train, y_train, k, x_test, y_test):
    x_train_0 = x_train[(y_train == 0).flatten()]
    x_train_1 = x_train[(y_train == 1).flatten()]
    objective_all_0, centroids_0, mixing_0, covariance_0, phi_0, objective_0, run_0 = gmm(x_train_0, k, 30, 10)
    objective_all_1, centroids_1, mixing_1, covariance_1, phi_1, objective_1, run_1 = gmm(x_train_1, k, 30, 10)
    prediction_0 = naive_bayes(x_test, mixing_0, centroids_0, covariance_0, k)
    prediction_1 = naive_bayes(x_test, mixing_1, centroids_1, covariance_1, k)
    prediction = np.asarray((prediction_0 < prediction_1).astype(int))
    accuracy = ((np.sum(prediction == y_test))/y_test.shape[0])*100
    prediction_series = pd.Series(np.asarray(prediction).flatten(), name = 'Predicted')
    y_series = pd.Series(np.asarray(y_test).flatten(), name = 'Actual')
    confusion_matrix = pd.crosstab(y_series, prediction_series)
    return prediction, accuracy, confusion_matrix


# In[32]:


prediction_1, accuracy_1, confusion_matrix_1 = prediction(df_3_x_train_np, df_3_y_train_np, 1, df_3_x_test_np, df_3_y_test_np)
prediction_2, accuracy_2, confusion_matrix_2 = prediction(df_3_x_train_np, df_3_y_train_np, 2, df_3_x_test_np, df_3_y_test_np)
prediction_3, accuracy_3, confusion_matrix_3 = prediction(df_3_x_train_np, df_3_y_train_np, 3, df_3_x_test_np, df_3_y_test_np)
prediction_4, accuracy_4, confusion_matrix_4 = prediction(df_3_x_train_np, df_3_y_train_np, 4, df_3_x_test_np, df_3_y_test_np)


# In[33]:


print("\033[1m" + '1-Gaussian Mixture Model' + "\033[0m")
print('Prediction Accuracy = ' + str(accuracy_1))
print(confusion_matrix_1.to_markdown() + '\n') 
print("\033[1m" + '2-Gaussian Mixture Model' + "\033[0m")
print('Prediction Accuracy = ' + str(accuracy_2))
print(confusion_matrix_2.to_markdown() + '\n') 
print("\033[1m" + '3-Gaussian Mixture Model' + "\033[0m")
print('Prediction Accuracy = ' + str(accuracy_3))
print(confusion_matrix_3.to_markdown() + '\n') 
print("\033[1m" + '4-Gaussian Mixture Model' + "\033[0m")
print('Prediction Accuracy = ' + str(accuracy_4))
print(confusion_matrix_4.to_markdown() + '\n') 


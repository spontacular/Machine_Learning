#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import texttable
import scipy


# # Question 1

# In[2]:


def get_m(data_file, teams):
    scores = pd.read_csv(data_file, header = None)
    scores[4] = (scores[1] > scores[3]).astype(int)
    scores[5] = (scores[1] < scores[3]).astype(int)
    M = np.zeros((teams, teams))
    for score in np.asarray(scores):
        i = score[0] - 1
        points_i = score[1]
        j = score[2] - 1
        points_j = score[3]
        indicator_i = score[4]
        indicator_j = score[5]
        M[i,i] = M[i,i] + indicator_i + points_i/(points_i + points_j)
        M[j,j] = M[j,j] + indicator_j + points_j/(points_i + points_j)
        M[i,j] = M[i,j] + indicator_j + points_j/(points_i + points_j)
        M[j,i] = M[j,i] + indicator_i + points_i/(points_i + points_j)
    M = M/M.sum(axis = 1, keepdims = 1)
    return M


# In[3]:


M = get_m('hw4-data/CFB2019_scores.csv', 769)


# In[4]:


def get_w_0(teams):
    w_0 = np.full((1, teams), 1/teams)
    return w_0


# In[5]:


w_0 = get_w_0(769)


# In[6]:


def run_iterations(M, w_0, iterations):
    w_all = np.zeros((iterations, w_0.shape[1]))
    for t in range(iterations):
        w_0 = np.matmul(w_0, M)
        w_all[t] = w_0
    return w_all


# In[7]:


w_all = run_iterations(M, w_0, 10000)


# In[8]:


def get_top_teams(w_all, iteration, top_n):
    iteration = iteration - 1
    index = w_all[iteration].argsort()[-top_n:][::-1]
    teams = pd.read_csv('hw4-data/TeamNames.txt', delimiter = '\n', header = None)
    teams = np.asarray(teams)
    top_teams = teams[index]
    weights = w_all[iteration][index]
    df = pd.DataFrame({'Team':top_teams.flatten(), 'Weight':weights})
    df.index = np.arange(1, len(df) + 1)
    return df


# ## Question 1(a)

# In[9]:


get_top_teams(w_all, 10, 25)


# In[10]:


get_top_teams(w_all, 100, 25)


# In[11]:


get_top_teams(w_all, 1000, 25)


# In[12]:


get_top_teams(w_all, 10000, 25)


# ## Question 1(b)

# In[22]:


def get_norm_difference(M):
    eigenvalue, eigenvector = scipy.sparse.linalg.eigs(M.T, k = 1)
    w_inf = (eigenvector.real).T
    w_inf = w_inf/w_inf.sum()
    difference = (w_all - w_inf)
    norm_difference = np.sum(np.abs(difference), axis = -1)
    return norm_difference


# In[23]:


norm_difference = get_norm_difference(M)


# In[24]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Markov Chain')
plt.xlabel('Iteration')
plt.ylabel('L1 Norm')
plt.plot(norm_difference, label = 'Objective', color = 'green')
plt.xticks(np.arange(1, 10001, 1000))
plt.show()


# # Question 2

# In[16]:


def load_data(data_file, vocabulary_file):
    data = pd.read_csv(data_file, delimiter = '\n', header = None)
    vocab = pd.read_csv(vocabulary_file, delimiter = '\n', header = None)
    X = np.zeros((vocab.shape[0], data.shape[0]))
    for line_number, line in enumerate(np.asarray(data[0])):
        for pair in line.split(','):
            key, value = pair.split(':')
            key = int(key) - 1
            X[key, line_number] = value
    return X


# In[17]:


def initialise(X, rank, lower_limit, upper_limit):
    W = np.random.uniform(lower_limit, upper_limit, (X.shape[0], rank))
    H = np.random.uniform(lower_limit, upper_limit, (rank, X.shape[1]))
    return W, H


# In[18]:


def update_h(X, W, H):
    M1 = (W.T)/((W.T).sum(axis = 1, keepdims = 1))
    M2 = np.divide(X, (np.matmul(W, H) + np.power(0.1, 16)))
    result = np.multiply(H, np.matmul(M1, M2))
    return result


# In[19]:


def update_w(X, W, H):
    M1 = np.divide(X, (np.matmul(W, H) + np.power(0.1, 16)))
    M2 = (H.T)/((H.T).sum(axis = 0, keepdims = 1))
    result = np.multiply(W, np.matmul(M1, M2))
    return result


# In[20]:


def get_objective(X, W, H):
    M1 = np.matmul(W, H)
    M2 = np.multiply(X, np.log(M1 + np.power(0.1, 16)))
    result = np.sum(M2 - M1)
    return result*(-1)


# In[21]:


def NMF(data_file, vocabulary_file, iterations, rank, lower_limit, upper_limit):
    objective = np.zeros((iterations, 1))
    X = load_data(data_file, vocabulary_file)
    W, H = initialise(X, rank, lower_limit, upper_limit)
    for i in range(iterations):
        H = update_h(X, W, H)
        W = update_w(X, W, H)
        objective[i,0] = get_objective(X, W, H)
    return objective


# In[22]:


objective = NMF('hw4-data/nyt_data.txt', 'hw4-data/nyt_vocab.dat', 100, 25, 1, 2)


# ## Question 2(a)

# In[23]:


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(12, 8)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Nonnegative Matrix Factorization')
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.plot(objective, label = 'Objective', color = 'green')
plt.xticks(np.arange(1, 101, 10))
plt.show()


# ## Question 2(b)

# In[24]:


def get_w(data_file, vocabulary_file, iterations, rank, lower_limit, upper_limit):
    X = load_data(data_file, vocabulary_file)
    W, H = initialise(X, rank, lower_limit, upper_limit)
    for i in range(iterations):
        H = update_h(X, W, H)
        W = update_w(X, W, H)
    W = W/W.sum(axis = 0, keepdims = 1)
    return W


# In[25]:


W = get_w('hw4-data/nyt_data.txt', 'hw4-data/nyt_vocab.dat', 100, 25, 1, 2)


# In[26]:


def get_weights(W, top_n):
    top_weights = np.zeros((W.shape[1], top_n))
    top_indexes = np.zeros((W.shape[1], top_n))
    for topic, weight in enumerate(W.T):
        index = (-weight).argsort()[:top_n]
        top_indexes[topic] = index
        top_weights[topic] = weight[index]
    return top_indexes, top_weights


# In[27]:


top_indexes, top_weights = get_weights(W, 10)


# In[28]:


def get_words(top_indexes, vocabulary_file):
    vocab = pd.read_csv(vocabulary_file, delimiter = '\n', header = None)
    words = np.asarray(vocab)[top_indexes.astype(int)].reshape(top_indexes.shape)
    return words


# In[29]:


top_words = get_words(top_indexes, 'hw4-data/nyt_vocab.dat')


# In[30]:


def get_table(top_words, top_weights):
    df = pd.DataFrame(np.core.defchararray.add(top_words.astype(str), np.char.add(': ', top_weights.astype(str))))
    df['result'] = df[df.columns[0]].str.cat([df[c] for c in df.columns[1:]], sep='\n')
    table = texttable.Texttable()
    table.set_cols_align(["c", "c", "c", "c", "c"])
    table.set_cols_valign(["m", "m", "m", "m", "m"])
    table.set_cols_width([20 for i in range(5)])
    table.add_rows([[df['result'][0], df['result'][1], df['result'][2], df['result'][3], df['result'][4]],
                    [df['result'][5], df['result'][6], df['result'][7], df['result'][8], df['result'][9]],
                    [df['result'][10], df['result'][11], df['result'][12], df['result'][13], df['result'][14]],
                    [df['result'][15], df['result'][16], df['result'][17], df['result'][18], df['result'][19]],
                    [df['result'][20], df['result'][21], df['result'][22], df['result'][23], df['result'][24]]])
    return table


# In[31]:


table = get_table(top_words, top_weights)


# In[32]:


print(table.draw())


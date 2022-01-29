#!/usr/bin/env python
# coding: utf-8

# ## HW2: Linear Discriminant Analysis
# In hw2, you need to implement Fisher’s linear discriminant by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

# ## Load data

# In[1461]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[1462]:


x_train = pd.read_csv("x_train.csv").values
y_train = pd.read_csv("y_train.csv").values[:,0]
x_test = pd.read_csv("x_test.csv").values
y_test = pd.read_csv("y_test.csv").values[:,0]


# In[1463]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## 1. Compute the mean vectors mi, (i=1,2) of each 2 classes

# In[1464]:


## Your code HERE
class1 = np.zeros((750, 2), dtype = np.float64)
class2 = np.zeros((750, 2), dtype = np.float64)
m1 = np.array([0.0, 0.0], dtype = np.float64)
m2 = np.array([0.0, 0.0], dtype = np.float64)
cnt = [0, 0]
for i in range(len(y_train)):
    if y_train[i] == 0:
        class1[cnt[0]] = x_train[i]
        m1 += x_train[i]
        cnt[0] += 1
    else:
        class2[cnt[1]] = x_train[i]
        m2 += x_train[i]
        cnt[1] += 1
m1 = m1 / cnt[0]
m2 = m2 / cnt[1]  


# In[1465]:


print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")


# ## 2. Compute the Within-class scatter matrix SW

# In[1466]:


## Your code HERE
sw = np.array([[0.0, 0.0], [0.0, 0.0]])
m1 = np.expand_dims(m1, axis=1)
m2 = np.expand_dims(m2, axis=1)
for i in range(cnt[0]):
    cp = class1[i]
    cp = np.expand_dims(cp, axis=1)
    sw += (cp - m1).dot((cp - m1).T)
for i in range(cnt[1]):
    cp = class2[i]
    cp = np.expand_dims(cp, axis=1)
    sw += (cp - m2).dot((cp - m2).T)


# In[1467]:


assert sw.shape == (2,2)
print(f"Within-class scatter matrix SW: {sw}")


# ## 3.  Compute the Between-class scatter matrix SB

# In[1468]:


## Your code HERE
sb = np.array([[0.0, 0.0], [0.0, 0.0]])
sb = (m2 - m1).dot((m2 - m1).T)


# In[1469]:


assert sb.shape == (2,2)
print(f"Between-class scatter matrix SB: {sb}")


# ## 4. Compute the Fisher’s linear discriminant

# In[1470]:


## Your code HERE
w = np.array([[0.0], [0.0]])
w = np.linalg.inv(sw).dot(m2 - m1)


# In[1471]:


assert w.shape == (2,1)
print(f" Fisher’s linear discriminant: {w}")


# In[1472]:


p_set = []
for i in range(x_train.shape[0]):
    p_set.append(w.T.dot(x_train[i]))


# In[1473]:


def search_neighbor(val):
    K = 9
    dist=15000
    ans=0
    all_dist = np.zeros(x_train.shape[0])
    for i in range(x_train.shape[0]):
        all_dist[i] = abs(val - p_set[i])
    sort_idx = all_dist.argsort()
    neighbor = list(y_train[sort_idx][:K])
    ans = max(neighbor, key=neighbor.count)
    return ans


# ## 5. Project the test data by linear discriminant to get the class prediction by nearest-neighbor rule and calculate the accuracy score 
# you can use accuracy_score function from sklearn.metric.accuracy_score

# In[1474]:


y_pred = []
for i in range(x_test.shape[0]):
    y_pred.append(search_neighbor(w.T.dot(x_test[i])))    
acc = accuracy_score(y_test, y_pred)


# In[1475]:


print(f"Accuracy of test-set {acc}")


# ## 6. Plot the 1) projection line 2) Decision boundary and colorize the data with each class
# ### the result should look like this [image](https://i2.kknews.cc/SIG=fe79fb/26q1000on37o7874879n.jpg) (Red line: projection line, Green line: Decision boundary)

# In[1476]:


fig,ax = plt.subplots(figsize=(8,8))
plt.axis("equal")
plt.xlim(-2, 5)
plt.ylim(-2, 5)
for i in range(cnt[0]):
    plt.scatter(class1[i][0], class1[i][1], s=10, c='r', label='class1')
for i in range(cnt[1]):
    plt.scatter(class2[i][0], class2[i][1], s=10, c='b', label='class2')
slope = (w[1][0] / w[0][0])
w *= -1000
w = np.squeeze(w)
for i in range(cnt[0]):
    #print((np.dot(class1[i], w) / np.dot(w, w)))
    proj_point = (np.dot(class1[i], w) / np.dot(w, w)) * w
    plt.scatter(proj_point[0], proj_point[1], s=10, c='r')
    plt.plot([proj_point[0], class1[i][0]], [proj_point[1], class1[i][1]], c='b', alpha=0.03)
for i in range(cnt[1]):
    proj_point = (np.dot(class2[i], w) / np.dot(w, w)) * w
    #print(class2[i],w)
    plt.scatter(proj_point[0], proj_point[1], s=10, c='b')
    plt.plot([proj_point[0], class2[i][0]], [proj_point[1], class2[i][1]], c='b', alpha=0.03)
x = np.linspace(-2, 4, 50)
y = x * slope
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Projection Line: w={slope},b=0')


# In[ ]:





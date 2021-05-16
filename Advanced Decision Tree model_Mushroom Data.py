#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
mushrooms = pd.read_csv(r"C:\Users\HP\Downloads\Python materials\mushrooms data.csv")


# In[2]:


from sklearn.model_selection import train_test_split
# split the data int x(training data) and y (results)
y = mushrooms['class']
x = mushrooms.drop(['class'], axis=1)
x = pd.get_dummies(x)
y = pd.get_dummies(y)
x.info()
y.info()


# In[3]:


x.info()


# In[4]:


y.info()


# In[5]:


x.dtypes


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
mushrooms['habitat_cat'] = mushrooms["habitat"].astype("category").cat.codes
mushrooms['class_cat'] = mushrooms["class"].astype("category").cat.codes
mushrooms.dtypes
#mushrooms['habitat_cat'].unique()
sns.stripplot(x='class', y='habitat_cat', data=mushrooms, jitter=True)


# In[7]:


from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

parameters = {'criterion':('gini', 'entropy'), 
              'min_samples_split':[2,3,4,5], 
              'max_depth':[9,10,11,12],
              'class_weight':('balanced', None),
              'presort':(False,True),
             }


# In[8]:


tr = tree.DecisionTreeClassifier()
gsearch = GridSearchCV(tr, parameters)
gsearch.fit(X_train, y_train)
model = gsearch.best_estimator_
model
# gsearch.cv_results_
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
# scores
# model.fit(X_train, y_train)


# In[9]:


gsearch.cv_results_
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
scores
model.fit(X_train, y_train)


# In[10]:


gsearch.cv_results_
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
scores
model.fit(X_train, y_train)


# In[11]:


scores


# In[12]:


#The scores are really great, so fit the model and predict
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
score


# In[ ]:





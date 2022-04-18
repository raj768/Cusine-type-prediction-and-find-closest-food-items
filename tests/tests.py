#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pytest
import project2
import sys
import os
import numpy as np
from project2 import *


# In[5]:


path = 'yummly.json'
input_ingredients = ['romaine lettuce', 'black olives', 'grape tomatoes','garlic','pepper']
n_top=5


# In[6]:


def read_json_test():
    data = project2.read_json(path)
    return data
    assert data.shape is not None


# In[7]:


df = read_json_test()


# In[8]:


def feature_matrix_test():
    matrix = project2.feature_matrix(df,input_ingredients)
    return matrix
    assert matrix.shape is not None


# In[9]:


mat = feature_matrix_test()


# In[15]:


def prediction_test():
    predicted_cuisine,predicted_cuisine_score = project2.prediction(df,mat)
    return (predicted_cuisine,predicted_cuisine_score)
    assert predicted_cuisine !=0
    assert predicted_cuisine_score !=0


# In[16]:


predict,score = prediction_test() 


# In[19]:


def top_n_similar_test():
    similar_foods = project2.top_n_similar(mat,df,predict,score,5)
    return similar_foods
    assert similar_foods is not None


# In[20]:


top_n_similar_test()


# In[ ]:





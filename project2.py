#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ## Read json file and stored the data into the Dataframe

# In[34]:


def read_json(filepath):
    df = pd.read_json(filepath)
    return df


# ## Featurization using TF-IDF vectorizer

# In[36]:


def feature_matrix(df,input_ingredient_value):
    ingredients = df['ingredients'].values
    ingred = pd.Series(ingredients)
    ingredients_list = []
    for i in ingred:
        u = np.array(i)
        x = list(u)
        s = ' '.join(x)
        ingredients_list.append(s)
    input_list = ' '.join(input_ingredient_value)
    ingredients_list.append(input_list)
    tfidf_vect = TfidfVectorizer()
    matrix = tfidf_vect.fit_transform(ingredients_list)
    return matrix


# ## Predicting the Cuisine type and score

# In[202]:


def prediction(df, matrix):
    cuisine_values = df['cuisine'].values
    cuisine = pd.Series(cuisine_values)
    input_vec= matrix[149]
    mat_shape=matrix.shape[0]-1
    matrix_in = matrix[:mat_shape]
    clf = LogisticRegression(random_state=0,max_iter=10000)
    model=clf.fit(matrix_in,cuisine)
    predicted_cuisine = model.predict(input_vec)
    predicted_cuisine_score = model.predict_proba(input_vec).max()
    return (predicted_cuisine,predicted_cuisine_score)


# ## Top-n Similar Foods with similarity score

# In[209]:


def top_n_similar(matrix,df,predicted_cuisine,predicted_cuisine_score,top_n_value):
    w = list(predicted_cuisine[0])
    k = ''.join(w)
    input_vec= matrix[149]
    mat_shape=matrix.shape[0]-1
    matrix_in = matrix[:mat_shape]
    scores = cosine_similarity(input_vec,matrix_in)
    sorted_scores = scores[0].argsort()[::-1]
    sorted_scores_top = sorted_scores[:top_n_value]
    scores_indices = []
    for ind in sorted_scores_top:
        f = (ind,round(scores[0][ind],3))
        scores_indices.append(f)
    id_list = list(df['id'])
    h = []
    for idx, score in scores_indices:
        l = {"id": str(id_list[idx]), 
             "score" : score}
        h.append(l)
    closest_similar_foods_with_scores = {"cuisine": k,"score": predicted_cuisine_score, "closest": h}                              
    return closest_similar_foods_with_scores


# In[ ]:





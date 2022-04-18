# cs5293sp22-project2

For this project,  I have developed an analyzer which takes the data in the format of json and predict the cuisine type and also the project also provides similar cuisines possible. 

## Libraries used:

Pandas, json, sklearn



## Steps for completing this project:
1. Read the data from json file and save them into a dataframe
2. Ask the user to input all the ingredients that  they might be interested in.
3. Use the embeddings obtained from the existing dataset to train the necessary classifier.
4. Use the machine learning model to predict the type of cuisine and display it to the user.
5. Find the top N closest foods and return the IDs of those dishes to the user.

## Functions:

### read_json(filename):

This method returns a dataframe containing json data read from the supplied file path.

### feature_matrix(df,input_ingredient_value): 

This method takes the dataframe resulted from the previous method and converts each recipe's ingredients list to a string before appending it to a list. Also, I convert the user's input ingredient list to a string and append it to the list as the last item. I have used Tfidfvectorizer to featurize the data to extract the feature matrix.

### prediction(df, matrix):

This method takes the dataframe, the feature matrix generated by tfidfvectorizer, and filters the matrix to remove all feature rows except the last row, as well as the cuisine labels of the corresponding rows from the dataframe, and passes it to the decision tree classifier model to train it. The model is then fed the 150th row(randomly choosen), which is the input feature vector, to predict the cuisine type The cuisine type and as well as is returned once it has been predicted.

### top_n_similar(matrix,df,predicted_cuisine,predicted_cuisine_score,top_n_value):

The feature matrix, dataframe, predicted cuisine type, predicted cuisine score and the top n  value, which is the number of top dishes we want to display, are all passed to this function. To retrieve all the similarity scores between the input feature vector and all the other feature vectors in the recipe files, I used the sklearn cosine similarity technique. Then I reversed the order of the scores array and extracted only the n top indices. Then, using those indices, I scanned the dataframe's list of ids and scores to create a list of dictionaries  comprising the recipe id, score. For the displaying output, I have created another dictionary in the format which is similar to the format mentioned in the project description. 

## Fuctions of Test cases:

The tests.py file contains the test cases designed to test the each function of the project2.

## read_json_test():
This function checks whether it is able to read the json file or not and it is able to store the json data into the dataframe or not.

## feature_matrix_test():
This function checks whether it generates feature matrix or not.

## prediction_test():

This function checks whether it returns any predicted cuisine or not. 

## top_n_similar_test():

This function checks whether it returns any similar foods or not. 











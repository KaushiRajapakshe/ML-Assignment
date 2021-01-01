#!/usr/bin/env python
# coding: utf-8

# In[19]:


# IT16178700 - D.I.K. Rajapakshe
# Heart Diseases using Random Forest Classifier Algorithm
# Importing all required libraries to work with the dataset
import numpy as np # linear algebra
# Matplotlib library all graph types for dataset
import matplotlib.pyplot as pylt 
import sklearn.metrics as skm
# sklearn.metrics includes score functions, performance metrics and pairwise metrics and distance computations
import pandas as pds
# visualization of statistical data
import seaborn as sns
# Python Interface to Graphviz’s Dot language.
import pydotplus
# LabelEncoder use to converting the labels into the numeric form so as to convert it into the machine-readable form
from sklearn.preprocessing import LabelEncoder
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# create charts using pyplot, and color them with
from matplotlib.cm import rainbow
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
# Impliment the Random Forest Classifier Algorithm
from sklearn.ensemble import RandomForestClassifier
# Decision tree generation
# Load libraries
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
# Implementation of the Ada Boost Classifier 
from sklearn.ensemble import AdaBoostClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
# Read the dataset heart.csv from the file location
data_with_dup = pds.read_csv("/Users/kaushirajapakshe/Desktop/ML Assignment/heart.csv")
# Display the data with column
data_dup_top = data_with_dup.head()
# display 
data_dup_top 


# In[20]:


file_name_output = "heart_without_dupes.csv"
# removing duplicates value sets
data_with_dup.drop_duplicates(subset=None, inplace=True)

# Write the results to a different file
data_with_dup.to_csv(file_name_output)
data = pds.read_csv("/Users/kaushirajapakshe/heart_without_dupes.csv")
# Display the data with column
data_top = data.head()
# display 
data_top 


# In[3]:


# Show the detailed information of each and every attributes of the dataset
data_attr = data.describe().transpose()
# display
data_attr


# In[4]:


# Check for existing null values
total_null_values = sum(data.isnull().sum())
print(total_null_values)


# In[5]:


# count the number of zero values in each column
print((data[['age','trestbps','chol','thalach']] == 0).sum())


# In[6]:


data.target.value_counts()


# In[7]:


countNoDisease = len(data[data.target == 0])
countHaveDisease = len(data[data.target == 1])
print("Percentage of Patients have no Heart Disease : {:.2f}%".format((countNoDisease / (len(data.target))*100)))
print("Percentage of Patients have Heart Disease : {:.2f}%".format((countHaveDisease / (len(data.target))*100)))


# In[8]:


import seaborn as sns
sns.countplot(x="target", data=data, palette="icefire")
# Save a figure
plt.savefig('target.png')
plt.show()


# In[9]:


sns.countplot(x='sex', data=data, palette="cubehelix")
plt.xlabel("Sex (0 = female, 1= male)")
# Save a figure
plt.savefig('sex.png')
plt.show()


# In[10]:


# convert strings to indicators (only numerics for correlation)
dummies = pd.get_dummies(data, columns=['sex', 'target']) 
corr = dummies.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            vmin=-1,
            cmap='Set3',
            annot=True,
            mask=np.tri(corr.shape[0], k=0))
# Save a figure
plt.savefig('sex_and_target_heatmap.png')
plt.show()


# In[11]:


pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6),color=['#ffb3b3','#660066' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Disease"])
plt.ylabel('Frequency')
# Save a figure
plt.savefig('sex_and_target_crosstab.png')
plt.show()


# In[12]:


data.groupby('target').mean()


# In[13]:


pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(26,9))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
# Save a figure
plt.savefig('heart_disease_and_ages.png')
plt.show()


# In[14]:


# Create a figure
plt.figure(figsize=(12,8))
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="red")
plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
# Save a figure
plt.savefig('age_target_and_thalach_target_scatter.png')
plt.show()


# In[15]:


pd.crosstab(data.slope,data.target).plot(kind="bar",figsize=(15,6),color=['#ccf5ff','#b3ffb3' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
# Save a figure
plt.savefig('slop_and_target_crosstab.png')
plt.show()


# In[16]:


pd.crosstab(data.fbs,data.target).plot(kind="bar",figsize=(15,6),color=['#ff99e6','#ccccff' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["No Disease", "Disease"])
plt.ylabel('Frequency of Disease or Not')
# Save a figure
plt.savefig('fbs_and_target.png')
plt.show()


# In[17]:


pd.crosstab(data.cp,data.target).plot(kind="bar",figsize=(15,6),color=['#740b63', '#0e878b'])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.legend(["No Disease", "Disease"])
plt.ylabel('Frequency of Disease or Not')
# Save a figure
plt.savefig('sp_and_target.png')
plt.show()


# In[18]:


# create charts using pyplot, and color them with
from matplotlib.cm import rainbow

colors = rainbow(np.linspace(0, 1, len(estimators)))
# Create a figure
plt.figure(figsize=(12,4))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')
# Save a figure
plt.savefig('rfc_scor_over_estimators.png')
plt.show()


# In[ ]:


# Plotted Feature importance generates
get_ipython().run_line_magic('matplotlib', 'inline') 
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
# Create a figure
plt.figure(figsize=(10,4))
plt.bar(x_values, importances, orientation = 'vertical') 

plt.xticks(x_values, data_feature_list, rotation='vertical') 
plt.ylabel('Importance')
plt.xlabel('Variable') 
plt.title('Variable Importances')
# Save a figure
plt.savefig('feature_importances.png')
plt.show()


# In[ ]:


import seaborn as sns; 
sns.set() 
get_ipython().run_line_magic('matplotlib', 'inline') 
 
mat = confusion_matrix(Y_test, Y_pred) 
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Set3',) 
plt.xlabel('True class') 
plt.ylabel('Predicted class')
# Save a figure
plt.savefig('metrix_heatmap.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# LabelEncoder use to converting the labels into the numeric form so as to convert it into the machine-readable form,
# which they contain only values between 0 and no_classes-1. 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Get the copy of original data set as a training data set
data_td = data.copy() 
for i in data.columns:
    data_td[i]=labelencoder.fit_transform(data[i])
# Drop the 'target' column which represents the feature which going to predict 
X = data_td.drop(['target'], axis=1)
# assign 'Outcome' column to Y variable
Y = data_td['target']
data_feature_list = list(X.columns)

# Dataset separated to 80/20 for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
print(X_train.shape,' is the shape of Training Data Features')
print(Y_train.shape,' is the shape of Training Data Lables')
print(X_test.shape,' is the shape of Testing Data Features')
print(Y_test.shape,' is the shape of Testing Data Lables')

# Impliment the Random Forest Classifier Algorithm
from sklearn.ensemble import RandomForestClassifier
rf_scores = []
# Calculate test data scores over 10, 100, 200, 500 and 1000 trees.
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf_classifier.fit(X_train, Y_train)
    rf_scores.append(rf_classifier.score(X_test, Y_test))
# The predictions using test data
Y_pred = rf_classifier.predict(X_test)
Y_pred
# create charts using pyplot, and color them with
from matplotlib.cm import rainbow

colors = rainbow(np.linspace(0, 1, len(estimators)))
# Create a figure
plt.figure(figsize=(12,4))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')
# Save a figure
plt.savefig('rfc_scor_over_estimators.png')
plt.show()

# The predictions using test dataset
Y_pred = rf_classifier.predict(X_test)
Y_pred
# Get the accuracy score for dataset
accuracy_score=sm.accuracy_score(Y_test, Y_pred)
print('Accuracy score given for test data:',str(accuracy_score))
#Generate classification report based on the predicted values 
 
from sklearn import metrics
print("Classification Report : \n\n", metrics.classification_report(Y_pred, Y_test, target_names = ["Heart Disease","No Heart Disease"]))
# Generates the confusion matrix based on the test data values
from sklearn.metrics import confusion_matrix 
import seaborn as sns; 
sns.set() 
get_ipython().run_line_magic('matplotlib', 'inline') 
 
mat = confusion_matrix(Y_test, Y_pred) 
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Set3',) 
plt.xlabel('True class') 
plt.ylabel('Predicted class')
# Save a figure
plt.savefig('metrix_heatmap.png')
plt.show()
# Feature importances as numeric
importances = list(rf_classifier.feature_importances_)

# Tuples list with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data_feature_list, importances)]

# Sort the feature importances  
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print the dataset importances and feature  
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# Plotted Feature importance generates
get_ipython().run_line_magic('matplotlib', 'inline') 
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
# Create a figure
plt.figure(figsize=(10,4))
plt.bar(x_values, importances, orientation = 'vertical') 

plt.xticks(x_values, data_feature_list, rotation='vertical') 
plt.ylabel('Importance')
plt.xlabel('Variable') 
plt.title('Variable Importances')
# Save a figure
plt.savefig('feature_importances.png')
plt.show()

# Decision tree generation
# Load libraries
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
tree = rf_classifier.estimators_[1] 

export_graphviz(tree, out_file=dot_data,feature_names = data_feature_list,rounded = True, precision = 1) 
 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Show graph
Image(graph.create_png())
# Create PDF
graph.write_pdf("decision_tree.pdf")
# Create PNG
graph.write_png("decision_tree.png")


# Implimentation of the Ada Boost Classifier 
from sklearn.ensemble import AdaBoostClassifier
rf_classifier = AdaBoostClassifier(n_estimators=1000)
rf_classifier.fit(X_train, Y_train)
predictions = rf_classifier.predict(X_test)
(Y_test, predictions)
fig = plt.figure(figsize = (12,12))
ax = fig.gca()
data.hist(ax=ax)
# Save a figure
plt.savefig('data_hist.png')
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,12))
ax = fig.gca()
data.hist(ax=ax)
# Save a figure
plt.savefig('data_hist.png')
plt.show()


# In[ ]:



import pandas as pd
file_name_output = "my_file_without_dupes.csv"
data = pd.read_csv("/Users/kaushirajapakshe/Desktop/ML Assignment/heart.csv")

# Notes:
# - the `subset=None` means that every column is used 
#    to determine if two rows are different; to change that specify
#    the columns as an array
# - the `inplace=True` means that the data structure is changed and
#   the duplicate rows are gone  
data.drop_duplicates(subset=None, inplace=True)

# Write the results to a different file
data.to_csv(file_name_output)


# In[ ]:



# Feature importance as numeric
importan = list(rf_classifier.feature_importances_)

# Tuples list with variable and importance
feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(data_feature_list, importan)]

# Sort the feature importance  
feature_importance = sorted(feature_importance, key = lambda x: x[1], reverse = True)

# Print the dataset importance and feature  
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance];
# Plotted Feature importance generates
get_ipython().run_line_magic('matplotlib', 'inline') 
plt.style.use('fivethirtyeight')
x_values = list(range(len(importan)))
# Create a figure
plt.figure(figsize=(10,4))
plt.bar(x_values, importan, orientation = 'vertical') 

plt.xticks(x_values, data_feature_list, rotation='vertical') 
plt.ylabel('Importance')
plt.xlabel('Variable') 
plt.title('Variable Importance')
# Save a figure
plt.savefig('feature_importance.png')
plt.show()


# In[ ]:


# Decision tree generation
# Load libraries
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
tree = rf_classifier.estimators_[1] 

export_graphviz(tree, out_file=dot_data,feature_names = data_feature_list,rounded = True, precision = 1) 
 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Show graph
Image(graph.create_png())
# Create PDF
graph.write_pdf("decision_tree.pdf")
# Create PNG
graph.write_png("decision_tree.png")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=1000)
rnd_clf.fit(X_train, Y_train)
#Do the predicitions using test data
Y_pred = rnd_clf.predict(X_test)
Y_pred


# In[ ]:


sns.set() 
get_ipython().run_line_magic('matplotlib', 'inline') 
 
mat = confusion_matrix(Y_test, Y_pred) 
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Set3',) 
plt.xlabel('True class') 
plt.ylabel('Predicted class')
# Save a figure
plt.savefig('metrix_heatmap.png')
plt.show()


# In[ ]:





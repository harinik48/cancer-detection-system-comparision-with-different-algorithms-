#!/usr/bin/env python
# coding: utf-8

# # logistic regression

# In[24]:


import pandas as pd


# In[25]:


# Read breast cancer csv file to pandas data frame data
data = pd.read_csv('Downloads/wisconsin_breast_cancer.csv')


# In[26]:


# Display the first 5 rows of the csv file
data.head()


# In[27]:


data.shape 


# In[28]:


# There are 699 rows and 11 columns in this CSV file


# In[29]:


data.isnull().sum() #to find out how many cells have missing values
# the field nucleoli has 16 missing values


# In[30]:


data=data.dropna(how='any') # Dropping any rows that has missing values


# In[31]:


x=data[['thickness','size','shape','adhesion','single','nuclei','chromatin','nucleoli','mitosis']]


# In[32]:


x.head() # printing the first 5 rows to see whether we got all the features


# In[33]:


# to extract the 'class' field from 'data' and store it in variable y
# This is the variable that we want to predict 0= no cancer 1 = cancer 
y=data['class']
y.head()


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[35]:


# train logistic regression model 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)


# In[36]:


y_pred_class=logreg.predict(x_test) # make predictions based on x_test and store it to y_pred_class


# In[37]:


from sklearn.metrics import accuracy_score #works
print(accuracy_score(y_test, y_pred_class))


# In[38]:


## 94.7% of the time our model was able to identify breast cancer based on the training data


# In[39]:


y_test.value_counts() # "0" is more prevalent


# In[40]:



# all the time
1-y_test.mean()


# # confusion matrix
# 

# In[41]:


import sklearn.metrics as metrics
print(metrics.confusion_matrix(y_test, y_pred_class))


# In[42]:


#  let us see what this means
#
#                Predicted 0    Predicted 1    Total
#                                
#Actual  0        103              4            107
#Actual  1          5             59             64           
#Total            108             63


# In[43]:


confusion =metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[44]:


# Let us see the sensitivity of our logistic regression model
print(TP / float(TP+FN))


# In[45]:


# Our model's sensitivity is 92.1%


# In[46]:


# Let us calculate specificity
print(TN / float(TN+FP))


# In[47]:


# Calculate false postive rate - predicting cancer when pt does not have cancer
print(FP/ float(TN+FP))


# In[48]:


# precison - when it is predicting cancer how precise is it 
# positive predictive value 
print(TP / float(TP+FP))


# In[49]:


# Negative predictive value
print(TN / float(TN+ FN))


# In[50]:


# to figure out the probaility of cancer from a set of features
# use the predict_proba function
#see the predicted answers
logreg.predict(x_test)[0:10] # predicting cancer  based on the data from first 10 rows of x


# In[51]:


logreg.predict_proba(x_test)[0:10, :]


# In[52]:


# The first colun is the probability of it being favourable. Second column is the probablity of it being cancerous


# In[53]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt 
import random 
get_ipython().run_line_magic('matplotlib', 'inline')
# calculates the probability of predicting "1" (cancer) and store the out put in probab_cancer
proba_cancer=logreg.predict_proba(x_test)[:,1]


# In[54]:


# we need the actual values in the cancer column and the predicted probabilities of postive value "1"
roc_auc_score(y_test, proba_cancer)


# In[55]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, proba_cancer)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[56]:


plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # SDGclassifier

# In[60]:


import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[61]:


data = pd.read_csv("Downloads/wisconsin_breast_cancer.csv")
data.head()


# In[62]:


data.info()


# In[63]:


new_data = data.dropna()
new_data.info()


# In[64]:


y = new_data.iloc[:,10:11]
X = new_data.iloc[:,1:10]
print(y)
print(X)


# In[65]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1502775)
X_train.head()
X_train.info()


# In[66]:


from sklearn.linear_model import SGDClassifier


# In[67]:


def powerset(set):
    if set == []:
        yield set
    else:
        for i in powerset(set[1:]):
            yield i
            yield [set[0]] + i


# In[68]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[69]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

cv_scores = list()
cv_pred_accu = list()
best_set = list()
for i in powerset(list(X_train.columns)):
    if(i!=[]):
        sgd_clf = SGDClassifier(random_state=1502775)
        cross_validate(sgd_clf, X_train[i], y_train.values.ravel(), cv = 10)
        results = cross_val_score(sgd_clf, X_train[i], y_train.values.ravel(), cv = 10)
        cv_scores.append(results.mean())
        cv_pred_accu.append(accuracy_score(y_test, cross_val_predict(sgd_clf, X_test, y_test.values.ravel(), cv = 10)))
        best_set.append(i)


# In[70]:


print(max(cv_scores))
print(cv_pred_accu[cv_scores.index(max(cv_scores))])  
cv_best=best_set[cv_scores.index(max(cv_scores))]
print(cv_best)  


# In[71]:


test_score=list()
subset = list()
for i in powerset(list(X_train.columns)):
    if(i!=[]):
        sgd_clf = SGDClassifier(random_state=1502775)
        sgd_clf.fit(X_train[i],y_train.values.ravel())
        test_score.append(accuracy_score(y_test,sgd_clf.predict(X_test[i])))
        subset.append(i)


# In[72]:


print(max(test_score))
print(subset[test_score.index(max(test_score))])
print()
print("test score for subset with best cv_accuracy:")
print(test_score[subset.index(cv_best)])


# In[73]:


import matplotlib.pyplot as plt
plt.xlabel("cv-accuracy")
plt.ylabel("test-accuracy")
plt.scatter(cv_scores,test_score,alpha=0.4)
plt.show()


# # Random forest classifier

# In[1]:


from sklearn.ensemble import RandomForestClassifier


# In[2]:


all_scoresRF = list()
pred_accuRF = list()
best_setRF = list()


# In[16]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
for i in powerset(list(X_train.columns)):
    if(i!=[]):
        rf_clf = RandomForestClassifier(n_estimators=30,random_state=1502775)
        cross_validate(rf_clf, X_train[i], y_train.values.ravel(), cv = 10)
        results = cross_val_score(rf_clf, X_train[i], y_train.values.ravel(), cv = 10)
        all_scoresRF.append(results.mean())
        pred_accuRF.append(accuracy_score(y_test, cross_val_predict(rf_clf, X_test, y_test.values.ravel(), cv = 10)))
        best_setRF.append(i)


# In[17]:


print(max(all_scoresRF))
print(pred_accuRF[all_scoresRF.index(max(all_scoresRF))])  
cv_bestRF=best_setRF[all_scoresRF.index(max(all_scoresRF))]
print(cv_bestRF)


# In[18]:


test_scoreRF=list()
subsetRF = list()
for i in powerset(list(X_train.columns)):
    if(i!=[]):
        rf_clf = RandomForestClassifier(n_estimators=30,random_state=1502775)
        rf_clf.fit(X_train[i],y_train.values.ravel())
        test_scoreRF.append(accuracy_score(y_test,rf_clf.predict(X_test[i])))
        subsetRF.append(i)


# In[19]:


print("Prediction Accuracy")
print(max(test_scoreRF))
print()
print("Subset with best prediction accuracy")
print(subsetRF[test_scoreRF.index(max(test_scoreRF))])
print()
print("test score for subset with best cv_accuracy:")
print(test_scoreRF[subsetRF.index(cv_bestRF)])


# In[74]:


plt.xlabel("cv-accuracy")
plt.ylabel("test-accuracy")
plt.scatter(all_scoresRF,test_scoreRF,alpha=0.4)
plt.show()


# In[57]:


print("accuracy using logistic regression:")
print(accuracy_score(y_test, y_pred_class))


# In[75]:


print("accuracy using sdg:")
print(max(cv_scores))


# In[22]:


print("Prediction Accuracy using random forest")
print(max(test_scoreRF))
print()


# In[ ]:





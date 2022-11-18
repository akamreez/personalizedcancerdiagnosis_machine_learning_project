#!/usr/bin/env python
# coding: utf-8

# In[241]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[8]:


df=pd.read_csv('ttnew22.csv')


# In[9]:


df


# In[30]:


df


# In[31]:


data_text=df.loc[:,'ID,Text']


# In[32]:


data_text=pd.DataFrame(data_text)


# In[33]:


data_text


# In[36]:


data_text[['ID', 'TEXT']] = data_text["ID,Text"].apply(lambda x: pd.Series(str(x).split("||")))


# In[38]:


data_text.drop('ID,Text',axis='columns',inplace=True)


# In[39]:


data_text


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer


# In[42]:


gene_vectorizer=CountVectorizer()


# In[ ]:





# In[48]:


data=pd.read_csv('training_variants.csv')


# In[49]:


data.head()


# In[59]:


print(data_text.dtypes)
print(50*'*')
print(data.dtypes)


# In[61]:


convert_dict={'ID':int}


# In[62]:


data_text=data_text.astype(convert_dict)
print(data_text.dtypes)


# In[63]:


result=pd.merge(data,data_text,on='ID',how='left')


# In[64]:


result


# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


val_new,test_df=train_test_split(result,test_size=0.2,random_state=42,stratify=result['Class'])


# In[67]:


train_df,val_df=train_test_split(val_new,test_size=0.2,random_state=42,stratify=val_new['Class'])


# In[68]:


a=list(result.columns)


# In[69]:


a.remove('Class')


# In[70]:


input_cols=a


# In[71]:


target_col='Class'


# In[72]:


print(input_cols)
print(target_col)


# In[73]:


train_inputs=train_df[input_cols]
train_target=train_df[target_col]


# In[74]:


val_inputs=val_df[input_cols]
val_target=val_df[target_col]


# In[79]:


test_inputs=test_df[input_cols]
test_target=test_df[target_col]


# In[80]:


print(train_inputs.shape)
print(train_target.shape)

print(200 * '*')

print(val_inputs.shape)
print(val_target.shape)

print(200 * '*')

print(test_inputs.shape)
print(test_inputs.shape)


# In[81]:


gene_vectorizer


# In[82]:


#gene


# In[83]:


train_gene_onehotencoding=gene_vectorizer.fit_transform(train_inputs['Gene'])


# In[84]:


val_gene_onehotencoding=gene_vectorizer.transform(val_inputs['Gene'])


# In[86]:


test_gene_onehotencoding=gene_vectorizer.transform(test_inputs['Gene'])


# In[92]:


len(gene_vectorizer.get_feature_names())


# In[94]:


len(train_inputs['Gene'].unique())


# In[95]:


alpha=[10 ** i for i in range(-5,1)]


# In[103]:


from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


# In[101]:


cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_gene_onehotencoding, train_target)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_gene_onehotencoding, train_target)
    predict_y = sig_clf.predict_proba(val_gene_onehotencoding)
    cv_log_error_array.append(log_loss(val_target, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(val_target, predict_y, labels=clf.classes_, eps=1e-15))


# In[123]:


predict_y=sig_clf.predict_proba(train_gene_onehotencoding)



# In[126]:


a=log_loss(train_target,predict_y)
print(a)


# In[127]:




predict_y=sig_clf.predict_proba(val_gene_onehotencoding)
b=log_loss(val_target,predict_y)
print('validation loss :',b)


# In[129]:


predict_y=sig_clf.predict_proba(test_gene_onehotencoding)
c=log_loss(test_target,predict_y)
print('Test loss : ',c)


# In[104]:


fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')


# In[107]:


fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))


# In[118]:


#variati


# In[116]:


variation_vectorizer=CountVectorizer()


# In[119]:


train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_inputs['Variation'])


# In[120]:


test_variation_feature_onehotCoding = variation_vectorizer.transform(test_inputs['Variation'])
val_variation_feature_onehotCoding = variation_vectorizer.transform(val_inputs['Variation'])


# In[130]:


alpha=[10 ** i for i in range(-5,1)]


# In[131]:


for i in alpha:
    clf=SGDClassifier(alpha=i,penalty='l2',loss='log',random_state=42)
    clf.fit(train_variation_feature_onehotCoding,train_target)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_variation_feature_onehotCoding,train_target)
    predict_y=sig_clf.predict_proba(val_variation_feature_onehotCoding)
    print('for alpha :',i,'the log loss value for validation set is :',log_loss(val_target,predict_y,labels=clf.classes_,eps=1e-15))
    
    


# In[132]:


alpha=0.001


# In[133]:


predict_y=sig_clf.predict_proba(train_variation_feature_onehotCoding)
a=log_loss(train_target,predict_y)
print(a)


# In[134]:


predict_y=sig_clf.predict_proba(val_variation_feature_onehotCoding)
a=log_loss(val_target,predict_y)
print(a)


# In[135]:


predict_y=sig_clf.predict_proba(test_variation_feature_onehotCoding)
a=log_loss(test_target,predict_y)
print(a)


# In[136]:


train_inputs.shape


# In[238]:


from sklearn.metrics import confusion_matrix


# In[239]:


def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
    
    B =(C/C.sum(axis=0))
    
    labels = [1,2,3,4,5,6,7,8,9]
    
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    
    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    #divid each element of the confusion matrix with the sum of elements in that row


# In[137]:


test_inputs.shape


# In[144]:


test_cover=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]


# In[155]:


print(len(test_inputs.Variation.unique()))
print(len(list(set(test_inputs.Variation))))
ab=len(list(set(test_inputs.Variation)))
print(ab)


# In[148]:


test_coverage=test_inputs[test_inputs['Variation'].isin(list(set(train_inputs['Variation'])))].shape[0]


# In[149]:


val_coverage=val_inputs[val_inputs['Variation'].isin(list(set(train_inputs['Variation'])))].shape[0]


# In[156]:


print('Out of',ab,'unique categories in variation in test data,There are',test_coverage,'categories present in the train data')


# In[190]:


text_vectorizer = CountVectorizer()
train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_inputs['TEXT'])
# getting all the feature names (words)


# In[191]:


from sklearn.preprocessing import normalize


# In[194]:


train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)

# we use the same vectorizer that was trained on train data
test_text_feature_onehotCoding = text_vectorizer.transform(test_inputs['TEXT'])
# don't forget to normalize every feature
test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

# we use the same vectorizer that was trained on train data
val_text_feature_onehotCoding = text_vectorizer.transform(val_inputs['TEXT'])
# don't forget to normalize every feature
val_text_feature_onehotCoding = normalize(val_text_feature_onehotCoding, axis=0)

train_text_feature_onehotCoding.shape
# In[195]:


train_text_feature_onehotCoding.shape


# In[196]:


train_text_features= text_vectorizer.get_feature_names()
print("Total number of unique words in train data :", len(train_text_features))


# In[197]:


alpha=[10 ** i for i in range(-5,1)]


# In[198]:


cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_text_feature_onehotCoding, train_target)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_text_feature_onehotCoding, train_target)
    predict_y = sig_clf.predict_proba(val_text_feature_onehotCoding)
    cv_log_error_array.append(log_loss(val_target, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(val_target, predict_y, labels=clf.classes_, eps=1e-15))


# In[199]:


clf = SGDClassifier(alpha=0.0001, penalty='l2', loss='log', random_state=42)
clf.fit(train_text_feature_onehotCoding, train_target)
    
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_text_feature_onehotCoding, train_target)


# In[200]:


predict_y=sig_clf.predict_proba(train_text_feature_onehotCoding)
a=log_loss(train_target,predict_y)
print(a)


# In[201]:


predict_y=sig_clf.predict_proba(val_text_feature_onehotCoding)
a=log_loss(val_target,predict_y)
print(a)


# In[202]:


predict_y=sig_clf.predict_proba(test_text_feature_onehotCoding)
a=log_loss(test_target,predict_y)
print(a)


# In[203]:


import numpy as np


# In[225]:


def predict_and_plot_confusion_matrix(train_inputs, train_target,test_inputs, test_target, clf):
    clf.fit(train_inputs, train_target)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_inputs, train_target)
    pred_y = sig_clf.predict(test_inputs)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :",log_loss(test_target, sig_clf.predict_proba(test_inputs)))
    # calculating the number of data points that are misclassified
    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_target))/test_inputs.shape[0])
    plot_confusion_matrix(test_target, pred_y)


# In[226]:


def report_log_loss(train_inputs, train_target, test_inputs, test_target,  clf):
    clf.fit(train_inputs, train_target)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_inputs, train_target)
    sig_clf_probs = sig_clf.predict_proba(test_inputs)
    return log_loss(test_target, sig_clf_probs, eps=1e-15)


# In[206]:


from scipy.sparse import hstack


# In[207]:


train_df


# In[208]:


train_gene_var_onehotCoding = hstack((train_gene_onehotencoding,train_variation_feature_onehotCoding))
test_gene_var_onehotCoding = hstack((test_gene_onehotencoding,test_variation_feature_onehotCoding))
val_gene_var_onehotCoding = hstack((val_gene_onehotencoding,val_variation_feature_onehotCoding))


# In[209]:


train_inputs_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
train_target = np.array(list(train_df['Class']))

test_inputs_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
test_target = np.array(list(test_df['Class']))

val_inputs_onehotCoding = hstack((val_gene_var_onehotCoding, val_text_feature_onehotCoding)).tocsr()
val_target = np.array(list(val_df['Class']))


# In[210]:


print("One hot encoding features :")
print("(number of data points * number of features) in train data = ", train_inputs_onehotCoding.shape)
print("(number of data points * number of features) in test data = ", test_inputs_onehotCoding.shape)
print("(number of data points * number of features) in cross validation data =", val_inputs_onehotCoding.shape)


# In[217]:


alpha=[10 ** i for i in range(-6,1)]


# In[218]:


cv_log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(train_inputs_onehotCoding, train_target)
    
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_inputs_onehotCoding, train_target)
    predict_y = sig_clf.predict_proba(val_inputs_onehotCoding)
    cv_log_error_array.append(log_loss(val_target, predict_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(val_target, predict_y, labels=clf.classes_, eps=1e-15))


# In[219]:


alpha=0.0001


# In[220]:


clf = SGDClassifier(alpha=0.0001, penalty='l2', loss='log', random_state=42)
clf.fit(train_inputs_onehotCoding, train_target)
    
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_inputs_onehotCoding, train_target)


# In[221]:


predict_y=sig_clf.predict_proba(train_inputs_onehotCoding)
a=log_loss(train_target,predict_y)
print(a)


# In[222]:


predict_y=sig_clf.predict_proba(val_inputs_onehotCoding)
a=log_loss(val_target,predict_y)
print(a)


# In[223]:


predict_y=sig_clf.predict_proba(test_inputs_onehotCoding)
a=log_loss(test_target,predict_y)
print(a)


# In[228]:


alpha=[10 ** i for i in range(-6,1)]


# In[229]:


alpha


# In[233]:


best_alpha_index=np.argmin(cv_log_error_array)


# In[234]:


alpha=alpha[best_alpha_index]


# In[235]:


alpha


# In[243]:


clf = SGDClassifier(class_weight='balanced', alpha=0.0001, penalty='l2', loss='log', random_state=42)
predict_and_plot_confusion_matrix(train_inputs_onehotCoding, train_target, val_inputs_onehotCoding, val_target, clf)


# In[248]:


clf = SGDClassifier(class_weight='balanced', alpha=0.0001, penalty='l2', loss='log', random_state=42)
clf.fit(train_inputs_onehotCoding,train_target)
test_point_index = 1
no_feature = 500
predicted_cls = sig_clf.predict(test_inputs_onehotCoding[test_point_index])
print("Predicted Class :", predicted_cls[0])
print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_inputs_onehotCoding[test_point_index]),4))
print("Actual Class :", test_target[test_point_index])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('/Users/jwalavaishnavikarri/Documents/Project_new/final_data.xlsx')
df.head().style.background_gradient('turbo')
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.columns

df.info()

df.rename(columns={'shortest distance Agent-Pathlab(m)' : 'Distance Agent-Pathlab', ##unit = meters
                   'shortest distance Patient-Pathlab(m)' : 'Distance Patient-Pathlab',  ##unit = meters
                   'shortest distance Patient-Agent(m)' : 'Distance Patient-Agent',  ##unit = meters
                   'Availabilty time (Patient)' : 'Patient Availabilty',  ##range format
                   'Test Booking Date' : 'Booking Date',  
                   'Test Booking Time HH:MM' : 'Booking Time',
                   'Way Of Storage Of Sample' : 'Specimen Storage',
                   ' Time For Sample Collection MM' : 'Specimen collection Time',
                   'Time Agent-Pathlab sec' : 'Agent-Pathlab sec',
                   'Agent Arrival Time (range) HH:MM' : 'Agent Arrival Time',
                   'Exact Arrival Time MM' : 'Exact Arrival Time'   ##output time
                  }, inplace=True)



# In[6]:


print("duplicate values",df.duplicated().sum())
print("Null values",df.isna().sum())


# In[3]:


df.columns


# In[7]:


#Plots:
sns.distplot(df['Exact Arrival Time'])


# In[8]:


ID_columns = df[['Patient ID', 'Agent ID', 'pincode']]
num_columns = df[['Age', 'Distance Agent-Pathlab', 'Distance Patient-Pathlab', 'Distance Patient-Agent', 
                        'Specimen collection Time' , 'Agent-Pathlab sec', 'Exact Arrival Time']]
categ_columns = df[['patient location', 'Diagnostic Centers', 'Time slot', 'Patient Availabilty', 'Gender', 
                          'Booking Date', 'Specimen Storage', 'Sample Collection Date', 'Agent Arrival Time']]
num_columns.info()
categ_columns.head()


# In[9]:


list(categ_columns['Diagnostic Centers'].unique())


# In[10]:


categ_columns['Diagnostic Centers'].value_counts().plot(kind = 'bar')


# In[11]:


def name_change(text):
    if text == 'Medquest Diagnostics Center' or text == 'Medquest Diagnostics':
        return 'Medquest Diagnostics Center'
    elif text == 'Pronto Diagnostics' or text == 'Pronto Diagnostics Center':
        return 'Pronto Diagnostics Center'
    elif text == 'Vijaya Diagonstic Center' or text == 'Vijaya Diagnostic Center':
        return 'Vijaya Diagnostic Center'
    elif text == 'Viva Diagnostic' or text == 'Vivaa Diagnostic Center':
        return 'Vivaa Diagnostic Center'
    else:
        return text

categ_columns['Diagnostic Centers'] = categ_columns['Diagnostic Centers'].apply(name_change)    


# In[12]:


categ_columns['Diagnostic Centers'].value_counts().plot(kind = 'bar')


# In[ ]:


categ_columns['Time slot'].value_counts().plot(kind = 'bar')


# In[ ]:


categ_columns['Specimen Storage'].value_counts().plot(kind = 'bar')


# In[ ]:


len(categ_columns['Patient Availabilty'].unique())


# In[ ]:


categ_columns['Patient Availabilty'].value_counts().plot(kind = 'bar')


# In[ ]:


len(categ_columns['Agent Arrival Time'].unique())


# In[ ]:


categ_columns['Gender'].value_counts().plot(kind = 'bar')


# In[13]:


new_df = pd.concat([ID_columns,
                    categ_columns[['Diagnostic Centers', 'Time slot', 'Patient Availabilty', 'Gender',
                                         'Specimen Storage', 'Agent Arrival Time']],
                    num_columns[['Distance Patient-Agent', 'Specimen collection Time', 'Exact Arrival Time']]
                   ], axis = 1)
new_df.info()


# In[14]:


final = new_df[new_df['Distance Patient-Agent'] != 0]
final.info()


# In[15]:


sns.distplot(np.log(final['Distance Patient-Agent']))


# In[16]:


for col in final.columns[:]:
    print(col, ' : ', len(final[col].unique()), 'Unique Values')


# In[17]:


final.describe()


# In[18]:


final.drop(['Patient ID', 'pincode'], axis = 1, inplace = True)


# In[19]:


final['Distance Patient-Agent'] = np.log(final['Distance Patient-Agent'])


# In[20]:


final = final[final['Patient Availabilty'] != '19:00 to 22:00']


# In[21]:


from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

#from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC 


# In[22]:


final.columns


# In[ ]:


# x = final.drop(['Exact Arrival Time'], axis = 1)
# y = final[['Exact Arrival Time']]


# In[ ]:


# xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3)


# In[ ]:


# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[1, 2, 3, 4, 5, 6])
# ],remainder='passthrough')

# step2 = LogisticRegression(multi_class='ovr')

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(xtrain, ytrain)

# ypred = pipe.predict(xtest)

# print('Accruacy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
# print('Classification Report: \n', classification_report(ytest, ypred))


# In[ ]:


# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[1, 2, 3, 4, 5, 6])
# ],remainder='passthrough')

# step2 = LogisticRegression(multi_class='ovr',
#                            penalty = 'l2',
#                            solver='newton-cg',
#                            C = 16.0,
#                            fit_intercept=True,
#                            class_weight='balanced',
#                            random_state=50
#                           )

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(xtrain, ytrain)

# ypred = pipe.predict(xtest)

# print('Accruacy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
# print('Classification Report: \n', classification_report(ytest, ypred))


# In[23]:


final['Patient Availabilty From'] = final['Patient Availabilty'].apply(lambda x:x.split(':')[0])


# In[24]:


a = final['Patient Availabilty'].apply(lambda x:x.split('to')[1])
final['Patient Availabilty To'] = a.apply(lambda x:x.split(':')[0])


# In[25]:


b = final['Agent Arrival Time'].apply(lambda x:x.split('to')[1])
final['Agent Arrive Before'] = b.apply(lambda x:x.split(':')[0])


# In[26]:


final['Patient Availabilty From'] = final['Patient Availabilty From'].astype('int64')
final['Patient Availabilty To'] = final['Patient Availabilty To'].astype('int64')
final['Agent Arrive Before'] = final['Agent Arrive Before'].astype('int64')


# In[27]:


final1 = final.drop(['Patient Availabilty', 'Agent Arrival Time', 'Diagnostic Centers'], axis = 1)


# In[28]:


variables = final1.drop(['Exact Arrival Time'], axis = 1)
target = final1[['Exact Arrival Time']]


# In[29]:


xtrain, xtest, ytrain, ytest = train_test_split(variables, target, test_size=0.3)


# In[30]:


xtrain.info()


# In[ ]:


# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[1, 2, 3])
# ],remainder='passthrough')

# step2 = LogisticRegression(multi_class='ovr',
#                            penalty = 'none',
#                            solver='newton-cg',
#                            C = 1e-05,
#                            fit_intercept=True,
#                            class_weight='balanced',
#                            random_state=50
#                           )

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(xtrain, ytrain)

# ypred = pipe.predict(xtest)
# ypredtrain = pipe.predict(xtrain)

# print('Train Accruacy score: {:.4f}'.format(accuracy_score(ytrain, ypredtrain))) 
# print('Test Accruacy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
# print('Test Classification Report: \n', classification_report(ytest, ypred))


# In[31]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
xtrain['Time slot']= labelencoder.fit_transform(xtrain['Time slot'])
xtrain['Gender'] = labelencoder.fit_transform(xtrain['Gender'])
xtrain['Specimen Storage'] = labelencoder.fit_transform(xtrain['Specimen Storage'])
xtest['Time slot']= labelencoder.fit_transform(xtest['Time slot'])
xtest['Gender'] = labelencoder.fit_transform(xtest['Gender'])
xtest['Specimen Storage'] = labelencoder.fit_transform(xtest['Specimen Storage'])

# grid search logistic regression model
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


xtest.info()


# In[22]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# space['solver'] = ['newton-cg']
# space['penalty'] = ['none']
# space['C'] = [1e-5]

search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)

result = search.fit(xtrain, ytrain)
print('BEST score: {:.4f}'.format(result.best_score_))
print('Best Hyperparameters: %s' % result.best_params_)


# In[32]:






model = LogisticRegression(multi_class='ovr',
                           penalty = 'none',
                           solver='newton-cg',
                           C = 1e-05,
                           fit_intercept=True,
                           class_weight='balanced',
                           random_state=50
                          )

result = model.fit(xtrain, ytrain)

ypred = result.predict(xtest)
ypredtrain = result.predict(xtrain)

print('Train Accruacy score: {:.4f}'.format(accuracy_score(ytrain, ypredtrain))) 
print('Test Accruacy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
print('Test Classification Report: \n', classification_report(ytest, ypred))


# In[37]:


import pickle


# In[38]:


pickle.dump(model,open('model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





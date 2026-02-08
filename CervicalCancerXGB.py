#!/usr/bin/env python
# coding: utf-8

# # CREVICAL CANCER RISK PREDICTION

# ## Problem statement/Goal
# 
# Cervical cancer remains a leading cause of cancer-related mortality among women. Early detection is critical for successful treatment outcomes. This project aims to develop a predictive model using XGBoost to predict cerical cancer in 858 patients. The dataset was collected at the Hospitpal Universitario de Caracas, Venezuela and contains the 858 paitents: demographic information, habits, and  historical records. Studies have shown that high sexual activity Human papilloma virus (HPV) is one of the key factors that inceease the risk of having cervical cancer. Also the presence of hormones in oral contraceptives, having many children, and smoking also increase the risk for developing cervical cancer, particulary in women with HPV. Also, people with weak immune systems(HIV/AIDS) have a high risk of HPV. Cervical cancer kills around 4,000 women in the U.S. and about 300,000 women worldwide. By leveraging Machine Learning an AI cervical cancer dealth can be reduced with early detection.

# ## Import Library and Dataset 

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install plotly')
import plotly.express as px


# In[2]:


df = pd.read_csv('/Users/marlandhamilton/Downloads/cervical_cancer.csv')


# In[3]:


# import the csv files using pandas 

# (int) Age
# (int) Number of sexual partners
#  (int) First sexual intercourse (age)
# (int) Num of pregnancies
# (bool) Smokes
# (bool) Smokes (years)
# (bool) Smokes (packs/year)
# (bool) Hormonal Contraceptives
# (int) Hormonal Contraceptives (years)
# (bool) IUD ("IUD" stands for "intrauterine device" and used for birth control
# (int) IUD (years)
# (bool) STDs (Sexually transmitted disease)
# (int) STDs (number)
# (bool) STDs:condylomatosis
# (bool) STDs:cervical condylomatosis
# (bool) STDs:vaginal condylomatosis
# (bool) STDs:vulvo-perineal condylomatosis
# (bool) STDs:syphilis
# (bool) STDs:pelvic inflammatory disease
# (bool) STDs:genital herpes
# (bool) STDs:molluscum contagiosum
# (bool) STDs:AIDS
# (bool) STDs:HIV
# (bool) STDs:Hepatitis B
# (bool) STDs:HPV
# (int) STDs: Number of diagnosis
# (int) STDs: Time since first diagnosis
# (int) STDs: Time since last diagnosis
# (bool) Dx:Cancer
# (bool) Dx:CIN
# (bool) Dx:HPV
# (bool) Dx
# (bool) Hinselmann: target variable - A colposcopy is a procedure in which doctors examine the cervix. 
# (bool) Schiller: target variable - Schiller's Iodine test is used for cervical cancer diagnosis
# (bool) Cytology: target variable - Cytology is the exam of a single cell type used for cancer screening.
# (bool) Biopsy: target variable - Biopsy is performed by removing a piece of tissue and examine it under microscope, 


# We will use of the features above to predict the target variables : (Hinelmann, Schiller, Cytology, and Biopsy). Biopsy is the main way doctors diagnose most types of cancer. 

# ## Exploratory Data Analysis

# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# when cleaning the data you see there is missing data (?). lets replace the missing data with the (NAN) 

# In[8]:


# replace ? with NAN
df = df.replace('?', np.nan)
df.head(20)


# In[9]:


# create a heatmap to see null values 
sns.heatmap(df.isnull(), yticklabels=False)


# Above you can see that the STD: time of first daignosis and last diagnosis have a lot of null values. so in this case, we can drop  the columns

# In[10]:


df= df.drop(columns =['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])


# In[11]:


df.head()


# In[12]:


df.dtypes


# In[13]:


# convert my object dtypes to numeric
df = df.apply(pd.to_numeric)
df.dtypes


# In[14]:


df.describe()


# In[15]:


df.mean()


# In[16]:


# lets fill out the null values with mean 
df = df.fillna(df.mean())
df


# In[17]:


# create a heatmap is check for null values 
sns.heatmap(df.isnull(), yticklabels=False)


# # Data Visualization 

# In[18]:


corr_matrix = df.corr()
corr_matrix


# In[19]:


# plot the heatmap for the corr_matrix
plt.figure(figsize = (30,30))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# In[20]:


df.hist(bins=10, figsize=(30,30), color='blue');


# # Preparing The Data Before Training

# In[21]:


target_df = df['Biopsy']
input_df = df.drop(columns=['Biopsy'])


# In[22]:


target_df.shape


# In[23]:


input_df.shape


# In[24]:


X = np.array(input_df).astype('float32')
y = np.array (target_df).astype('float32')


# In[25]:


# reshape the y 
y = y.reshape(-1,1)
y.shape


# In[26]:


#import the machine learning libraries 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# In[27]:


# scale the data before feeding the model
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[28]:


X


# In[29]:


# train and test data 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.25)
X_test, X_val, y_test, y_val = train_test_split(X, y, test_size =0.5)


# In[30]:


get_ipython().system('pip install xgboost')


# In[31]:


import xgboost as xgb

model = xgb.XGBClassifier(learning_rate = .1, max_depth = 5, n_estimators= 10)
#now fit the model with X_train and y_train data 

model.fit(X_train,y_train)


# In[32]:


# evaluate the model on the training set 
result_train = model.score(X_train, y_train)
print('Accuracy : {}'.format(result_train))


# In[33]:


# evaluate the model on the test set 
result_test = model.score(X_test, y_test)
print('Accuracy : {}'.format(result_test))


# In[34]:


# make prediction on testing data
y_predict = model.predict(X_test)


# In[35]:


from sklearn.metrics import confusion_matrix, classification_report


# In[36]:


print(classification_report(y_test, y_predict))


# In[37]:


cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot= True, fmt='.2f')


# In[ ]:





# In[ ]:





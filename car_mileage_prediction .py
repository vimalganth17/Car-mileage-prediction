#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas==1.3.4')
get_ipython().system('pip install scikit-learn==1.0.2')
get_ipython().system('pip install numpy==1.21.6')


# In[ ]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression


# In[ ]:


URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"
df = pd.read_csv(URL)


# In[6]:


df.sample(5)


# In[7]:


df.shape


# In[8]:


df.plot.scatter(x="Hourse power",y="MPG")


# In[9]:


df.plot.scatter(x = "Horsepower", y = "MPG")


# In[10]:


target = df["MPG"]


# In[13]:


features = df[["Horsepower","Weight"]]


# In[17]:


#creating a linear regression model
lr=LinearRegression()


# In[18]:


lr.fit(features,target)


# In[19]:


#highrer the score ,better the model
lr.score(features,target)


# In[21]:


#now gonna predict mileage based on linear regression with hourse power=100,weight=2000
lr.predict([[100,2000]])


# In[22]:


# this is the predicted mileage for the given hourse power and weight


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("/Users/venka/OneDrive/Desktop/Data science course/ML/NewspaperData.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[7]:


df.corr(numeric_only=True) 


# In[9]:


import seaborn as sns
sns.distplot(df['daily']) 


# In[11]:


sns.distplot(df['sunday'])


# # Fitting a Linear Regression Model

# In[13]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = df).fit() #syntax(smf.ols("target~actual, data=ourdataframe").fit())


# In[14]:


sns.regplot(x="daily", y="sunday", data=df);


# In[15]:


#Coefficients
model.params


# In[16]:


#1100(manual calculation)
(1.3*1100)+13.835630 #mx+c m=slope c=constant


# In[23]:


#R squared values
(model.rsquared,model.rsquared_adj)


# # # Predict for new data point

# In[20]:


#Predict for 200 and 300 daily circulation
new=pd.Series([200,300])


# In[21]:


new


# In[18]:


data_pred=pd.DataFrame(newdata,columns=['daily'])


# In[22]:


data_pred


# In[19]:


model.predict(data_pred)


# In[ ]:





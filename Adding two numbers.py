#!/usr/bin/env python
# coding: utf-8

# ## Adding two numbers using ML

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('add.csv')
data.head()


# In[3]:


data.shape 


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.scatter(data['x'],data['sum'])


# In[6]:


plt.scatter(data['y'],data['sum'])


# In[7]:


X = data[['x','y']]
y = data['sum']


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.33, random_state= 42)


# In[9]:


X_train


# In[10]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[11]:


model.score(X_train, y_train)


# In[12]:


model.score(X_test, y_test)


# In[13]:


y_pred  = model.predict(X_test)


# In[14]:


y_pred


# In[15]:


y_test


# In[16]:


df = pd.DataFrame({'actual':y_test,'prediction':y_pred})
df


# In[17]:


model.predict([[16,5]])


# In[18]:


import joblib
joblib.dump(model,'model_add')


# In[19]:


model = joblib.load('model_add')


# In[20]:


model.predict([[12,10]])


# In[21]:


X= data[['x','y']]
y= data['sum']


# In[22]:


model= LinearRegression()
model.fit(X,y)


# In[23]:


joblib.dump(model,'model_add')


# In[24]:


model = joblib.load('model_add')
model.predict([[12,10]])


# In[25]:


from tkinter import *

master= Tk()
master.title("Addition of Two Numbers")
label =Label(master,text="Addition of two numbers using ML", bg='black',fg="white").grid(row=0,columnspan=2)


# In[26]:


def show_entry_fields():
    p1=float(e1.get())
    p2=float(e2.get())
    model=joblib.load('model_joblib')
    result=model.predict([[p1,p2]])
    Label(master,text='sum is = ').grid(row=4)
    Label(master,text= result).grid(row=5)
    print("sum is", result)


# In[ ]:


Label(master,text="Enter First Number").grid(row=1)
Label(master,text="Enter Second Number").grid(row=2)
e1=Entry(master)
e2=Entry(master)
e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
Button(master,text='predict',command=show_entry_fields).grid()
mainloop()


# In[ ]:





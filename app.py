#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle


# In[3]:


app= Flask(__name__)


# In[4]:


trans=pickle.load(open("Transform.pkl","rb"))


# In[5]:


@app.route('/')
def home():
    return render_template('CovidTweet.html')


# In[6]:


def ValuePredictor(to_predict):
    loaded_model = pickle.load(open("CountModel.pkl", "rb")) 
    re= loaded_model.predict(to_predict) 
    return re 


# In[7]:


@app.route('/Result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form["Tweet Message"]
        pred=[to_predict_list]
        tr=trans.transform(pred).toarray()
        result = ValuePredictor(tr)         
        if int(result)== 1: 
            prediction = "Neutral"
        elif int(result)== 2: 
            prediction = "Positive"
        else:
            prediction = "Negative"
        return render_template("Result.html", prediction = prediction)


# In[9]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:





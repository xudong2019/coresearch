#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client.quanLiang    


# In[11]:


l = list(db.tagInfo.find())
tagDict = {}
for x in l:
    tagDict[x['tagName']]={'file':x['tag'], 'off_start':(x['off_start'][0], x['off_start'][1])}


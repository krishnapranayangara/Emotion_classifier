#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import shutil
import urllib.request


# In[2]:

#we need a header with infor regrding the agent and browser configurations
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}


# In[11]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&numberofpeople=one&phrase=happy%20faces&sort=best&page=1',headers=headers).text
soup=BeautifulSoup(source,'lxml')
happy_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')
#append links
for i in range(len(img_links)):
    happy_images.append(img_links[i]['src'])

#path
for i in range(len(happy_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/happy/"+str(i)+".jpg"
    urllib.request.urlretrieve(happy_images[i],name)

length=len(happy_images)


# In[12]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&numberofpeople=one&phrase=happy%20faces&sort=best&page=2',headers=headers).text
soup=BeautifulSoup(source,'lxml')
happy_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')

for i in range(len(img_links)):
    happy_images.append(img_links[i]['src'])


for i in range(len(happy_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/happy/"+str(length+i)+".jpg"
    urllib.request.urlretrieve(happy_images[i],name)

length=length+len(happy_images)


# In[13]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&numberofpeople=one&phrase=happy%20faces&sort=best&page=3',headers=headers).text
soup=BeautifulSoup(source,'lxml')
happy_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')

for i in range(len(img_links)):
    happy_images.append(img_links[i]['src'])


for i in range(len(happy_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/happy/"+str(length+i)+".jpg"
    urllib.request.urlretrieve(happy_images[i],name)

length=length+len(happy_images)


# In[14]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&numberofpeople=one&phrase=happy%20faces&sort=best&page=4',headers=headers).text
soup=BeautifulSoup(source,'lxml')
happy_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')

for i in range(len(img_links)):
    happy_images.append(img_links[i]['src'])


for i in range(len(happy_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/happy/"+str(length+i)+".jpg"
    urllib.request.urlretrieve(happy_images[i],name)

print("final length= ",length+len(happy_images))


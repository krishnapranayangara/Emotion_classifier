#!/usr/bin/env python
# coding: utf-8

# In[82]:


from bs4 import BeautifulSoup
import requests
import shutil
import urllib.request


# In[83]:


#we need a header with infor regrding the agent and browser configurations
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}


# In[84]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&page=1&phrase=sad%20people&sort=best&irgwc=1&cid=IS&utm_medium=affiliate_SP&utm_source=FreeImages&clickid=3oOWwh3VjxyNR7kTdz0yTWjoUkDSD80HuQFnUc0&utm_term=sad%20people&utm_campaign=srp_freephotos_top-view-more&utm_content=270498&irpid=246195',headers=headers).text
soup=BeautifulSoup(source,'lxml')
sad_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')
#append links
for i in range(len(img_links)):
    sad_images.append(img_links[i]['src'])

#path
for i in range(len(sad_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/sad/"+str(i)+".jpg"
    urllib.request.urlretrieve(sad_images[i],name)

length=len(sad_images)


# In[85]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&page=2&phrase=sad%20people&sort=best&irgwc=1&cid=IS&utm_medium=affiliate_SP&utm_source=FreeImages&clickid=3oOWwh3VjxyNR7kTdz0yTWjoUkDSD80HuQFnUc0&utm_term=sad%20people&utm_campaign=srp_freephotos_top-view-more&utm_content=270498&irpid=246195',headers=headers).text
soup=BeautifulSoup(source,'lxml')
sad_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')
#append links
for i in range(len(img_links)):
    sad_images.append(img_links[i]['src'])
#path
for i in range(len(sad_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/sad/"+str(length+i)+".jpg"
    urllib.request.urlretrieve(sad_images[i],name)
    
length=length+len(sad_images)


# In[86]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&page=3&phrase=sad%20people&sort=best&irgwc=1&cid=IS&utm_medium=affiliate_SP&utm_source=FreeImages&clickid=3oOWwh3VjxyNR7kTdz0yTWjoUkDSD80HuQFnUc0&utm_term=sad%20people&utm_campaign=srp_freephotos_top-view-more&utm_content=270498&irpid=246195',headers=headers).text
soup=BeautifulSoup(source,'lxml')
sad_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')
#append links
for i in range(len(img_links)):
    sad_images.append(img_links[i]['src'])
#path
for i in range(len(sad_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/sad/"+str(length+i)+".jpg"
    urllib.request.urlretrieve(sad_images[i],name)

length=length+len(sad_images)


# In[87]:


source=requests.get('https://www.istockphoto.com/search/2/image?mediatype=photography&page=4&phrase=sad%20people&sort=best&irgwc=1&cid=IS&utm_medium=affiliate_SP&utm_source=FreeImages&clickid=3oOWwh3VjxyNR7kTdz0yTWjoUkDSD80HuQFnUc0&utm_term=sad%20people&utm_campaign=srp_freephotos_top-view-more&utm_content=270498&irpid=246195',headers=headers).text
soup=BeautifulSoup(source,'lxml')
sad_images=[]
img_links=soup.select('img[src^="https://media.istockphoto.com/id/"]')
#append links
for i in range(len(img_links)):
    sad_images.append(img_links[i]['src'])
#path
for i in range(len(sad_images)):
    name="C:/Users/prana/Downloads/CAP Homework/Course project/images/sad/"+str(length+i)+".jpg"
    urllib.request.urlretrieve(sad_images[i],name)

print("final length= ",length+len(sad_images))


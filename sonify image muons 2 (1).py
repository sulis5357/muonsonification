#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import PIL 
import os

import skimage                 # form 1, load whole skimage library
import skimage.io              # form 2, load skimage.io module only
from skimage.io import imread  # form 3, load only the imread function
import numpy as np  
from astropy.table import QTable, Table, Column
from astropy import units as u
import numpy as np


import random
from numpy import asarray
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# In[ ]:





# In[3]:


im1 =Image.open("aa-removebg-preview-modified-modified.png")

imgWidth, imgHeight = im1.size
im1 = im1.convert("RGBA")
imgdata = im1.getdata()

x_pos = 0
y_pos = 1

pixel_value = []
x = []
y = []

for item in imgdata:
    if (x_pos) == imgWidth:
        x_pos = 1
        y_pos += 1
    else:
        x_pos += 1

    if item[3] != 0:
        pixel_value.append(item[2])
        x.append(x_pos)
        y.append(y_pos)

pixel_value, x, y = zip(*sorted(zip(pixel_value, x, y), reverse=True))


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    pic = imageio.imread("aa-removebg-preview-modified-modified.png")
    plt.figure(figsize = (10,10))

  


gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
gray = gray(pic)  

plt.figure( figsize = (10,10))
plt.imshow(gray, cmap = plt.get_cmap(name = 'gray'))


#gray[ 2077, 3]

gray_pixels = asarray(gray[0:350,0:350])


# In[4]:


print((gray_pixels))


# In[5]:


#x goes from 0 to 2382
#y goes from 0 to 2078
#gray[ x, y]

image = skimage.io.imread(fname="result.png")
#skimage.io.imshow(image)
h= image.shape[0]
w= image.shape[0]

h1= np.zeros(h)
for i in range (0,h-1):
    h1[i+1] = h1[i] + 1
w1 = np.zeros(w)
for j in range (0,w-1):
    w1[j+1] = w1[j] + 1    


# In[ ]:





# In[6]:


plt.imshow(gray_pixels, cmap = plt.get_cmap(name = 'gray'))


# In[ ]:





# In[7]:


xmin = np.zeros(350)
xmin = np.delete(xmin, -1)
xmax = np.zeros(350)
xmax = np.delete(xmax, -1)
ymin = np.zeros(350)
ymin = np.delete(ymin, -1)
ymax = np.zeros(350) 
ymax = np.delete(ymax, -1)

for j in range (350-1):
    xmin[j] = h1[j]
    xmax[j] = h1[j+1]

for k in range (350-1):
    ymin[k] = w1[k]
    ymax[k]=w1[k+1]
       


# In[8]:


#print((gray_pixels[:,0]))
#print((gray_pixels[0,:]))


#gray_pixels.shape


# In[9]:


N = np.array(((gray_pixels/25.0)), dtype=int)
print(N.shape)


#o =[( print(Nx[k]), print(xmin[i])) for i in range(len(xmin)) for k in range(len(Nx))  if i!=k] 
 
    
imagex = []
for i in range(len(xmin)):
    for k in range(len(xmin)):
        if N[i][k]>1:
            imagex.append( np.random.uniform(low=xmin[i], high=xmax[i], size= N[i][k]) )

#sorted_list = sorted(d, key=len)

f = np.array(imagex,dtype=object)


#print ('xmin is',xmin)
#print ('xmax is',xmax)
#print(j)
#print(f.shape)
#print((f))
#print((f))
print(N)


# In[10]:


imagey= []
for k in range(len(N)):
    for i in range(len(ymin)):
        if N[k][i]>1:
            imagey.append( np.random.uniform(low=ymin[i], high=ymax[i], size=N[k][i]) )

g = np.array(imagey,dtype=object)

print(len(g))


# In[ ]:


plt.figure(figsize=(7,7))


for i in range(len(f)):
    plt.scatter(g[i][:],f[i][:],alpha=1, s=0.01, c='white')
 
plt.title('image reproduced with scatterplot')    
plt.grid()
#plt.xlim(0,150)
#plt.ylim(0,150)
ax = plt.gca()
ax.set_facecolor('xkcd:black')
plt.figure(2)
plt.figure(figsize=(7,7))
plt.title('real image in grayscale')
plt.imshow(gray_pixels, cmap = plt.get_cmap(name = 'gray'),origin='lower')

plt.grid()


# In[11]:



from astronify.series import SoniSeries
from astropy.table import Table
import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astronify import simulator, series
import pandas
import pandas as pd


# In[ ]:





# In[12]:



#for i in range(len(f)):
 #   for j in (N):
  #      data = dict( flux  = (g), time =(f))
   #     df = pandas.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))

#print (type(df))


# In[13]:


#print(df)


# In[14]:


a = np.zeros([len(f),len(max(f,key = lambda x: len(x)))])
for i,j in enumerate(f):
    a[i][0:len(j)] = j

b = np.zeros([len(g),len(max(g,key = lambda x: len(x)))])
for i,j in enumerate(g):
    b[i][0:len(j)] = j
c=a.flatten()
d = b.flatten()
q = list(filter(lambda a: a != 0, c))
w = list(filter(lambda a: a != 0, d))


# In[ ]:


print((w[20000:60000])) #'q is time-x axis'
print(len(w)) #'w is flux- yaxis'


# In[ ]:


plt.figure(figsize=(7,7))


for i in range(len(f)):
    plt.scatter(g[i][:],f[i][:],alpha=0.3, s=0.3, c='white')
 
plt.title('image reproduced with scatterplot')   

x= [42.25692108436029,42.25692108436029]
y= [0, 150]
plt.plot(x,y, c='r')
plt.grid()
ax = plt.gca()
ax.set_facecolor('xkcd:black')


# In[22]:


data_table = Table({"time":q[29000:60000],
                    "flux":w[29000:60000]})


data_soni = SoniSeries(data_table)

data_soni.note_spacing = 0.0007
data_soni.pitch_mapper.pitch_map_args 
{'pitch_range': [100, 10000],
 'center_pitch': 440,
 'zero_point': 175,
 'stretch': 'linear'}
data_soni.sonify()
data_soni.play()


# In[ ]:





# In[ ]:


num = int(len(q))
for i in range(1, num):
  if num % i == 0:
    print(i)
    
num = int(len(w))
for i in range(1, num):
  if num % i == 0:

    print(i)


# In[ ]:


list_numbers= (np.arange(0,len(q)+1,562))

print((list_numbers))
print(len(list_numbers))
print(len(q))


# In[ ]:


for i in range(len(list_numbers)-1):
    c= np.array(q [list_numbers[i]:list_numbers[i+1]])
  
    d = w[list_numbers[i]:list_numbers[i+1]]
    


# In[ ]:



for i in range(len(list_numbers)-1):
    data_table1 = Table({"time":q[list_numbers[i]:list_numbers[i+1]],
                        "flux":w[list_numbers[i]:list_numbers[i+1]]})
    
    data_soni = SoniSeries(data_table1)

    data_soni.note_spacing = 0.001
    data_soni.pitch_mapper.pitch_map_args 
    {'pitch_range': [100, 10000],
     'center_pitch': 440,
     'zero_point': 'median',
     'stretch': 'linear'}
    data_soni.sonify()  
    data_soni.play()

    data_soni.stop()


# In[ ]:





# In[ ]:





# In[ ]:



   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#This code reproduce any image as a scatterplot. You can freely set the accuracy of the reproduction by playing with the values of 
#xmin, xmax, ymin, and ymax. You can also background reduce noisy images by setting N to an higher value.

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

#testing if branching works


# In[100]:


im1 =Image.open("picture.jpg")

img1width, img1height = im1.size
im1 = im1.convert("RGBA")
imgdata = im1.getdata()

x_pos = 0
y_pos = 1

pixel_value = []
x = []
y = []

for item in imgdata:
    if (x_pos) == img1width:
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

    pic = imageio.imread("picture.jpg")
    plt.figure(figsize = (10,10))

  


gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
gray = gray(pic)  

plt.figure( figsize = (10,10))
plt.imshow(gray, cmap = plt.get_cmap(name = 'gray'))


gray[ 2077, 3]

gray_pixels = asarray(gray[:300,:300])
#print((gray_pixels[2077,3]))


# In[101]:


#x goes from 0 to 2382
#y goes from 0 to 2078
#gray[ x, y]

image = skimage.io.imread(fname="picture.jpg")
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





# In[102]:


plt.imshow(gray_pixels, cmap = plt.get_cmap(name = 'gray'))


# In[103]:


xmin = np.zeros(300)
xmin = np.delete(xmin, -1)
xmax = np.zeros(300)
xmax = np.delete(xmax, -1)
ymin = np.zeros(300)
ymin = np.delete(ymin, -1)
ymax = np.zeros(300) 
ymax = np.delete(ymax, -1)

for j in range (300-1):
    xmin[j] = h1[j]
    xmax[j] = h1[j+1]

for k in range (300-1):
    ymin[k] = w1[k]
    ymax[k]=w1[k+1]
       


# In[104]:


#print((gray_pixels[:,0]))
#print((gray_pixels[0,:]))



#gray_pixels.shape


# In[105]:


N = np.array(gray_pixels, dtype=int)
print(N.shape)


#o =[( print(Nx[k]), print(xmin[i])) for i in range(len(xmin)) for k in range(len(Nx))  if i!=k] 
 
    
imagex = []
for i in range(len(xmin)):
    for k in range(len(xmin)):
        if N[i][k]>35:
            imagex.append( np.random.uniform(low=xmin[i], high=xmax[i], size= N[i][k]) )

#sorted_list = sorted(d, key=len)

f = np.array(imagex,dtype=object)


print ('xmin is',xmin)
print ('xmax is',xmax)
#print(j)
#print(f.shape)
#print((f))
#print((f))


# In[106]:


imagey= []
for k in range(len(N)):
    for i in range(len(ymin)):
        if N[k][i]>35:
            imagey.append( np.random.uniform(low=ymin[i], high=ymax[i], size=N[k][i]) )

g = np.array(imagey,dtype=object)
print(len(g))


# In[127]:


plt.figure(figsize=(7,7))


for i in range(len(f)):
    plt.scatter(g[i][:],f[i][:],alpha=0.03, s=0.5, c='white')
 
plt.title('image reproduced with scatterplot')    
plt.grid()
#plt.xlim(0,300)
#plt.ylim(0,300)
ax = plt.gca()
ax.set_facecolor('xkcd:black')
plt.figure(2)
plt.figure(figsize=(7,7))
plt.imshow(gray_pixels, cmap = plt.get_cmap(name = 'gray'),origin='lower')

plt.grid()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





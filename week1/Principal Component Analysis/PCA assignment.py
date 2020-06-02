import gzip
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import math


f = gzip.open('train-images-idx3-ubyte.gz','r')
g = gzip.open('train-labels-idx1-ubyte.gz','r')

image_size = 28
num_images = 60000

f.seek(16)
g.seek(8)

buf = f.read(image_size * image_size * num_images)
images = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
images = images.reshape(num_images,image_size*image_size)

buf2=g.read(num_images)
labels = np.frombuffer(buf2, dtype=np.uint8).astype(np.uint8)

digits={}
for i in range (10):
    filt = []
    for j in labels:  
        if(j==i):
            filt.append(True)
        else:
            filt.append(False)
    digits[i]=images[filt]

for k in digits:
    mean = digits[k].mean(axis=0)
    covmat = np.cov(digits[k], rowvar = False)
    values, vectors = LA.eigh(covmat)
    idx = values.argsort()[::-1]
    values=values[idx]
    vectors=vectors[:,idx]
    print(f'percentage variance accounted by first principal component of {k} is {values[0]/np.sum(values)*100} %')
    x=np.arange(1,785)
    plt.plot(x,values)
    plt.xlabel('number')
    plt.ylabel('eigenvalue')
    plt.title(f'all eigenvalues of {k}')
    plt.grid(True)
    plt.savefig(f'plot {k}.png')
    plt.close()
    plt.bar(x[:30],values[:30])
    plt.xlabel('number')
    plt.ylabel('eigenvalue')
    plt.title(f'30 largest eigenvalues of {k}')
    plt.text(5,500000,f'largest eigenvalue = {values[0]}')
    plt.grid(True)
    plt.savefig(f'bar {k}.png')
    plt.close()
    mean2 = mean.reshape(image_size,image_size)
    plt.imshow(mean2)
    plt.title(f'mean image of {k}')
    plt.savefig(f'mean {k}.png')
    plt.close()
    meanminus = mean - math.sqrt(values[0])*vectors[:,0]
    meanminus2 = meanminus.reshape(image_size,image_size)
    plt.imshow(meanminus2)
    plt.title(f'mean image - the first principal component of {k}')
    plt.savefig(f'mean - 1st PC {k}.png')
    plt.close()
    meanplus = mean + math.sqrt(values[0])*vectors[:,0] ;
    meanplus2 = meanplus.reshape(image_size,image_size)
    plt.imshow(meanplus2)
    plt.title(f'mean image + the first principal component of {k}')
    plt.savefig(f'mean + 1st PC {k}.png')
    plt.close()

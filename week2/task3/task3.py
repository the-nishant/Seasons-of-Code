import argparse
import numpy as np 
from PIL import Image 
from scipy import cluster as clst

parser = argparse.ArgumentParser() 
parser.add_argument('--input')
parser.add_argument('-k', type=int)
parser.add_argument('--output')

args = parser.parse_args()

img = Image.open(args.input) 
img.load()
M = np.asarray(img, dtype = np.float64)
w,h = img.size
img.close()
M = M.reshape(w*h,3)

k = args.k 
centroid,label = clst.vq.kmeans2(M,k,minit='++')
M[np.arange(0,w*h)] = centroid[label[np.arange(0,w*h)]]   

M = M.reshape(h,w,3)
M = M.astype(np.uint8)
img = Image.fromarray(M)
img.save(args.output)
img.close()


    








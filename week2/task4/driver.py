import task4
from matplotlib import pyplot as plt
import numpy as np

clean_sin = task4.generate_sin_wave(2,(-2,8),1000)
x = np.arange(-2,8,0.01)
plt.plot(x,clean_sin)
plt.xlabel('x')
plt.ylabel('y')
plt.title('clean sin wave')
plt.savefig('clean_sin.png')
plt.close()

dirty_sin = task4.noisify(clean_sin,0.05*0.05)
plt.plot(x,dirty_sin)
plt.xlabel('x')
plt.ylabel('y')
plt.title('noisy sin wave')
plt.savefig('dirty_sin.png')
plt.close()

cleaned_sin = task4.mean_filter(dirty_sin,1)
plt.plot(x,cleaned_sin)
plt.xlabel('x')
plt.ylabel('y')
plt.title('filtered noisy sin wave')
plt.savefig('cleaned_sin.png')
plt.close()
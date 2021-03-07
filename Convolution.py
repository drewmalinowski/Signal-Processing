# =============================================================================
# Drew Malinowski
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
plt.rcParams.update({'font.size': 14}) 


# inputs
delta = 1e-2
t = np.arange(0,20+delta,delta)
T = np.arange(0,2*t[len(t)-1], delta)



# basic signal functions
def u(t):
   y = np.zeros(t.shape)
   for i in range(len(t)):
       if t[i] >= 0 :
           y[i] = 1
       else:
           y[i] = 0 
   return y  

def r(t):
   y = np.zeros(t.shape)
   for i in range(len(t)):
       if t[i] >= 0 :
           y[i] = t[i]
       else:
           y[i] = 0 
   return y


def f1(t):
    y = u(t-2)-u(t-9)
    return y

def f2(t):
    y = np.zeros(t.shape)
    y = np.exp(-t)*u(t)
    return y

def f3(t):
    y = r(t-2)*(u(t-2)-u(t-3))+r(4-t)*(u(t-3)-u(t-4))
    return y


# convolution function
def CONVOLV(x,y):
    f1extend = np.append(x, np.zeros ((1, len(y)-1)))
    f2extend = np.append(y, np.zeros ((1, len(x)-1)))
    result = np.zeros(f1extend.shape)
    for i in range(len(x) + len(y) - 2):
        result[i] = 0
        for j in range(len(x)):
            if(i - j +1 > 0):
                try:
                    result[i] = result[i] + f1extend[j]*f2extend[i-j+1]
                except:
                    print(i,j)
    return result

# =============================================================================
# The convolution is performed by using for loops to iteratively add the 
# overlap of each function at each delta t. Using a for loop within a for 
# loop, the total overlap at every time step is iteratively summed. 
# The inner loop sums the total overlap at the given dt, while the outer loop
# sums the results of the inner loop of all dt in the given range.
# =============================================================================

x = f1(t)
y = f2(t)
z = f3(t)

C1 = sig.convolve(x,y)
C2 = sig.convolve(y,z)
C3 = sig.convolve(x,z)


#STEP 5 - Plots

#Figure 1
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(t, f1(t),label='f1(t)')
plt.plot(t, f2(t),label='f2(t)')
plt.plot(t, f3(t),label='f3(t)')
plt.legend()
plt.grid(True)
plt.ylabel("f(t)")
plt.xlabel("t")
plt.title("f1(t), f2(t), f3(t)")
plt.ylim([0,1.2])
plt.xlim([0,5])

#Figure 2
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(T, CONVOLV(x,y))
plt.grid(True)
plt.ylabel("f1(t)*f2(t)")
plt.xlabel("t")
plt.title("Convolution of f1(t) and f2(t)")
plt.ylim([0,110])
plt.xlim([0,20])

#Figure 3
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(T, CONVOLV(x,z))
plt.grid(True)
plt.ylabel("f1(t)*f3(t)")
plt.xlabel("t")
plt.title("Convolution of f1(t) and f3(t)")
plt.ylim([0,105])
plt.xlim([0,20])

#Figure 4
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(T, CONVOLV(y,z))
plt.grid(True)
plt.ylabel("f2(t)*f3(t)")
plt.xlabel("t")
plt.title("Convolution of f2(t) and f3(t)")
plt.ylim([0,60])
plt.xlim([0,20])
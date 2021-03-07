import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) 

delta = 1e-2

t = np.arange(-10,10+delta,delta)
T = np.arange(-20,20+delta/2, delta)

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

def H1(t):
    y = np.exp(2*t)*u(1-t)
    return y

def H2(t):
    y = u(t-2) - u(t-6)
    return y

def H3(t):
    f = 0.25
    w = f*2*3.14159
    y = np.cos(w*t)*u(t)
    return y

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

x = H1(t)
y = H2(t)
z = H3(t)

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(t, H1(t),label='H1(t)')
plt.plot(t, H2(t),label='H2(t)')
plt.plot(t, H3(t),label='H3(t)')
plt.legend()
plt.grid(True)
plt.ylabel("H(t)")
plt.xlabel("t")
plt.title("H1(t), H2(t), H3(t)")
plt.ylim([-2,8])
plt.xlim([-10,10])

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(T,delta*CONVOLV(x,u(t)))
plt.grid(True)
plt.ylabel("H1(t)*u(t)")
plt.xlabel("t")
plt.title("H1(t) step response")
plt.ylim([0,4])
plt.xlim([-20,20])

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(T,delta*CONVOLV(y,u(t)))
plt.grid(True)
plt.ylabel("H2(t)*u(t)")
plt.xlabel("t")
plt.title("H2(t) step response")
plt.ylim([0,5])
plt.xlim([-20,20])

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(T,delta*CONVOLV(z,u(t)))
plt.grid(True)
plt.ylabel("H3(t)*u(t)")
plt.xlabel("t")
plt.title("H3(t) step response")
plt.ylim([-1,1])
plt.xlim([-20,20])
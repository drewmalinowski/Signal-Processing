import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

delta = 1e-3
t = np.arange(0,20+delta,delta)
T = 8
y =0
N = [1, 3, 15, 50, 150, 1500]

def ak(k):
    ak = 0;
    return ak

def bk(k):
    bk = (2/(k*np.pi))*(1-np.cos(k*np.pi))
    return bk

print("a_0 =", ak(0))
print("a_1 =", ak(1))
print("b_1 =", bk(1))
print("b_2 =", bk(2))
print("b_3 =", bk(3))


y = 0        
w = (2*np.pi)/T 
for h in [1,2]:
    for i in ([1+(h-1)*3,2+(h-1)*3,3+(h-1)*3]):
        for k in np.arange(1,N[i-1]+1):
            b = bk(k)
            x = b*np.sin(k*w*t)
            y += x
        plt.figure(h,figsize=(10,8))
        plt.subplot(3,1,(i-1)%3+1)
        plt.plot(t,y)
        plt.grid()
        plt.ylabel('N = %i' %N[i-1])
        if i == 1 or i ==4:
            plt.title('Fourier Series')     
        if i == 3 or i == 6:
           plt.xlabel('t[s]')
           plt.show()
        y = 0
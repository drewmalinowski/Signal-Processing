# =============================================================================
# Drew Malinowski
# =============================================================================


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from math import pi, exp, cos, sin, log, sqrt

NN = 5000
N2 = int(NN/2)
x = np.zeros(NN)
y = np.zeros(NN)

dt = .05
TT = np.linspace(0,dt*(NN-1),NN)
DF = 1/(dt*NN)
FF = np.linspace(0,DF*(NN-1),NN)

w1 = 1
w2 = 2
w3 = 3
w4 = 4

f1 = w1 / 2*np.pi
f2 = w2 / 2*np.pi
f3 = w3 / 2*np.pi
f4 = w4 / 2*np.pi

x = np.sin(w1*TT) + 1*np.sin(w2*TT) + 1*np.sin(w3*TT) + 1*np.sin(w4*TT)


plt.figure(figsize=(12,10))

plt.subplot(321)
plt.plot(TT,x,'k')
plt.title('FIR Filter')
plt.ylabel('a). x[k]')
plt.xlabel('T (sec)')
plt.xlim(0,50)
plt.grid()


X = (1/NN)*np.fft.fft(x)
H = np.zeros(NN)

for n in range(70,130):
    H[n] =  exp(-.5*((n-100)/45)**2)
    
""" Reflect the positive frequencies to the right side """
for n in range(1,N2-1):
    H[NN-n] = H[n] 
    
Y = H*X 

plt.subplot(322)
plt.plot(FF,abs(X),'k',label='X')
plt.plot(FF,H,'k--',label='H')
plt.legend(loc='upper right')
plt.ylabel('b). H(w),X(w)')
plt.xlabel('Freq (Hz)')
plt.axis([0,1,0,1.1])
plt.grid()

h = np.fft.ifft(H)

plt.subplot(323)
plt.plot(h.real,'k')
plt.xlabel('k')
plt.ylabel('c). h[k]')
plt.grid()

M = 20
hh = np.zeros(NN)

""" Move the filter to the left side """
for n in range(M):
    hh[n+M] = h[n].real
    hh[M-n] = hh[M+n]

plt.subplot(324)
plt.plot(hh,'ok')
plt.axis([0 ,2*M,-.015,.04])
plt.xlabel('k')
plt.ylabel('d). hh[k]')
plt.grid()


yy=np.convolve(H,x)
for n in range(NN):
    y[n] = yy[n+M]


plt.subplot(325) 
plt.plot(TT,y,'k')
plt.ylabel('e). y[k]')
plt.xlabel('T (sec)')
plt.xlim(0,50)
plt.ylim(-55,60)
plt.grid()

Y2 = (1/NN)*np.fft.fft(y)

plt.subplot(326) 
plt.plot(FF,abs(Y),'k')
plt.ylabel('f). Y[w]')
plt.xlabel('Freq (Hz)')
plt.axis([0,1,0,0.4])
plt.grid()

plt.show()




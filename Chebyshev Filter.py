# =============================================================================
# Drew Malinowski
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# =============================================================================
# Fourth Order Chebyshev I low-pass filter (cutoff = 1 rad/s)
# =============================================================================


NUM = [0,0,0,0,0.380303]
DEN = [1,1.23888,1.76746,1.07934,0.400319]

system = sig.lti(NUM,DEN)
w, Hmag, Hphase = sig.bode(system)


plt.figure(figsize=(10,7))
plt.semilogx(w,10**(0.05*Hmag),'k')
plt.grid(which='both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('|H|')
plt.xticks([.1,.8,1.0,1.667,2,10])
plt.yticks([0.0, 0.07, 0.95, 1.0 ])
plt.title('Figure 1: Fourth Order Chebyshev I low-pass filter (cutoff = 1 rad/s)')
plt.show()


# =============================================================================
# High-Pass Filter - 
# =============================================================================

NUM2 = [0.380303,0,0,0,0]
DEN2 = [0.400319,1.07934,1.76746,1.23888,1]

system2 = sig.lti(NUM2,DEN2)
w2, Hmag2, Hphase2 = sig.bode(system2)

plt.figure(figsize=(10,7))
plt.semilogx(w2,10**(0.05*Hmag2),'k')
plt.grid(which='both')
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('|H|')
plt.xticks([.1,.8,1.0,1.667,2,10])
plt.yticks([0.0, 0.07, 0.95, 1.0 ])
plt.title('Figure 2: Fourth Order Chebyshev I high-pass filter (cutoff = 1 rad/s)')
plt.show()

# =============================================================================
# Time Domain Simulation
# =============================================================================


dt = 0.001
NN = 50000
TT = np.arange(0,NN*dt,dt)
y = np.zeros(NN)
f = np.zeros(NN)

A,B,C,D = sig.tf2ss(NUM2,DEN2)
x = np.zeros(np.shape(B))

omega = 0.6
for n in range(NN):
    f[n] = np.sin(omega*n*dt)
    
for m in range(NN):
    x = x + dt*A.dot(x) + dt*B*f[m]
    y[m] = C.dot(x) + D*f[m]
    
plt.figure(figsize=(10,7))

plt.plot(TT,f,'k', label='input')
plt.plot(TT,y,'r--', label = 'output')

plt.title('Time Domain Simulation of Filter Response')
plt.axis([0, NN*dt,-1,1])
plt.yticks([-1.3, -.95, -0.07, 0.07, 0.95, 1.3 ])
plt.grid()
plt.xlabel('T (sec)')
plt.legend(loc='upper right')
plt.show()
























# =============================================================================
# Drew Malinowski
# State Space Formulation Program
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Select step size and time interval:
# =============================================================================

dt = 0.01
Interval = 20

steps = int((Interval/dt))
steps_per_second = int((1/dt))
t = np.arange(0,dt*steps,dt)

# =============================================================================
# State Variables and Output: (based on state space formulation)
# =============================================================================

x = np.matrix('0;0;0;0;0;0')
r = 0

w = np.matrix('0;0;0;0;0;0')
r2 = 0

X1 = np.zeros(steps)
X2 = np.zeros(steps)
X3 = np.zeros(steps)
X4 = np.zeros(steps)
X5 = np.zeros(steps)
X6 = np.zeros(steps)

output = np.zeros(steps)

W1 = np.zeros(steps)
W2 = np.zeros(steps)
W3 = np.zeros(steps)
W4 = np.zeros(steps)
W5 = np.zeros(steps)
W6 = np.zeros(steps)

output2 = np.zeros(steps)

# =============================================================================
# Input functions:
# =============================================================================

f = np.zeros(steps)
g = np.zeros(steps)

for n in range(0,int(2.5*steps_per_second)):
    f[n] = 1
    
for n in range(int(3*steps_per_second),int(6*steps_per_second)):
    g[n] = 1    

# =============================================================================
# State Space Formulation:
# =============================================================================

A = np.matrix(' 0 1 0 0 0 0 ; -10 -5 0 0 0 0 ; 0 0 0 1 0 0 ; 0 1 -2 -7 0 0 ; \
              0 0 0 0 0 1 ; -3 0 0 -2 -1 0 ')
B = np.matrix(' 0 0 ; 1 0 ; 0 0 ; 0 0 ; 0 0 ; 0 1 ')
B1 = B[:,0]
B2 = B[:,1]
C = np.matrix(' 1 0 0 0 1 0 ')

N = steps

for i in range(N):

    x = x + dt*A*x + dt*B1*f[i] + dt*B2*g[i]
    r = r + dt*C*x
    
    
    X1[i] = x[0]
    X2[i] = x[1]
    X3[i] = x[2]
    X4[i] = x[3]
    X5[i] = x[4]
    X6[i] = x[5]
    
    output[i] = r[0]
    
    
# =============================================================================
# State Transition Matrix
# =============================================================================

I2 = np.eye(6)
A2 = A*A
A3 = A*A2
A4 = A2*A2
A5 = A*A4
A6 = A4*A2
F =( I2 + A*dt + 0.5*A2*(dt**2)+(1/6)*A3*(dt**3) + (1/24)*A4*(dt**4)
    + (1/120)*A5*(dt**5) + (1/720)*A6*(dt**6) )
Ainv = np.linalg.inv(A)
G = (F-I2)*Ainv*B
G1 = G[:,0]
G2 = G[:,1]

for i in range(N):
    
    w = F*w + G1*f[i] + G2*g[i]
    r2 = r2 + dt*C*w
    
    W1[i] = w[0]
    W2[i] = w[1]
    W3[i] = w[2]
    W4[i] = w[3]
    W5[i] = w[4]
    W6[i] = w[5]   
    
    output2[i] = r2[0]

# =============================================================================
# Figure 1   
# ============================================================================= 
    
Size = (13,17)
plt.rcParams.update({'font.size': 13})
plt.figure(figsize=Size)

plt.subplot(5,1,1)
plt.plot(t, f, 'k', label='f(t)')
plt.plot(t, g, 'k--', label = 'g(t)')
plt.legend(loc='upper right')
plt.grid(True)
plt.ylabel('Inputs')
plt.xlim([0,Interval])

plt.subplot(5,1,2)
plt.plot(t, X1, 'k', label="W(t)")
plt.plot(t, X2, 'k--', label='dW/dt')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0,Interval])

plt.subplot(5,1,3)
plt.plot(t, X3, 'k', label="Y(t)")
plt.plot(t, X4, 'k--', label='dY/dt')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0,Interval])

plt.subplot(5,1,4)
plt.plot(t, X5, 'k', label='Z(t)')
plt.plot(t, X6, 'k--', label='dZ/dt')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0,Interval])

plt.subplot(5,1,5)
plt.plot(t, output, 'k', label='Rout(t)')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Output')
plt.xlim([0,Interval])


# =============================================================================
# Figure 2
# =============================================================================
 
plt.figure(figsize=Size)

plt.subplot(5,1,1)
plt.plot(t, f, 'k', label='f(t)')
plt.plot(t, g, 'k--', label = 'g(t)')
plt.legend(loc='upper right')
plt.grid(True)
plt.ylabel('Inputs')
plt.xlim([0,Interval])

plt.subplot(5,1,2)
plt.plot(t, W1, 'k', label='W(t)')
plt.plot(t, W2, 'k--', label='dW/dt')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0,Interval])

plt.subplot(5,1,3)
plt.plot(t, W3, 'k', label='Y(t)')
plt.plot(t, W4, 'k--', label='dY/dt')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0,Interval])

plt.subplot(5,1,4)
plt.plot(t, W5, 'k', label='Z(t)')
plt.plot(t, W6, 'k--', label='dZ/dt')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim([0,Interval])

plt.subplot(5,1,5)
plt.plot(t, output2, 'k', label='Rout(t)')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Output')
plt.xlim([0,Interval])

# =============================================================================
# # ===========================================================================
# #  Alternate Input Functions (sinusoidal):
# # ===========================================================================
# 
# from math import cos, sin, pi    
#
# f = np.zeros(steps)
# g = np.zeros(steps)
# 
# """Select Frequency Values"""
# 
# frequency1 = 60
# frequency2 = 100
# 
# omega1 = 2*pi*frequency1*dt
# omega2 = 2*pi*frequency2*dt
# 
# for n in range(0,steps):
#     Tf = n*dt
#     f[n] = 10*sin(omega1*Tf)
#     
# for n in range(0,steps):
#     Tg = n*dt
#     g[n] = 3*cos(omega2*Tg)
# =============================================================================
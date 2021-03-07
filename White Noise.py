# =============================================================================
# Drew Malinowski
# ECE476
# HW6
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.fftpack import fft, fftshift
from scipy.optimize import fsolve
Pi = np.pi

# =============================================================================
# Generating White Noise:
# =============================================================================

mean = 0
var = 0.1

dt = 0.001
steps = 1000
t = np.arange(0, dt*steps, dt)

W = np.random.normal(mean, var, steps)

Exponential = np.zeros(steps)
for n in range(steps):
    Exponential[n] = 10*np.exp(-2*t[n]) + W[n]

plt.figure(figsize=(10,7))
plt.plot(t, W)
plt.grid()
plt.title('White Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude [V]')
plt.show()

plt.figure(figsize=(10,7))
plt.plot(t, Exponential)
plt.grid()
plt.title('Exponential with White Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude [V]')
plt.show()


# =============================================================================
# Numerically determining the mean and variance of the noise:
# =============================================================================

dt2 = 10*dt
steps2 = int((dt*steps) / dt2)
t2 = np.arange(0,dt*steps, dt2)

W2 = np.zeros(steps2)

Total_Measured_Noise = 0

for n in range(steps2):
    n2= n*10
    W2[n] = W[n2]
    Total_Measured_Noise = Total_Measured_Noise + W2[n]

Numerical_Mean = Total_Measured_Noise / steps2

x = 0

for n in range(steps2):
    x = x + (W2[n] - Numerical_Mean)**2

numerical_variance = x / steps2

print("Mean: ", Numerical_Mean)
print("  ")
print("Variance:  ", numerical_variance)
print("  ")

# =============================================================================
# Autocorrelation Function and Fourier Transform
# =============================================================================

def autocorrelation(x):
    results = np.convolve(x,x)
    return results

def Fast_Fourier_Transform(x,fs):
    N = len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_phi)):
        if np.abs(X_mag[i])<.05:
            X_phi[i] = 0
    return X_mag, X_phi, freq

R1 = autocorrelation(W)
Mag1, Phase1, Freq1 = Fast_Fourier_Transform(R1,dt)

R2 = autocorrelation(Exponential)
Mag2, Phase2, Freq2 = Fast_Fourier_Transform(Exponential,dt)

plt.figure(figsize=(10,7))
plt.plot(R1)
plt.grid()
plt.title('Autocorrelation of White Noise')
plt.show()

plt.figure(figsize=(10,7))
plt.stem(Freq1, Mag1, use_line_collection=True)
plt.grid(True)
plt.title('Fourier Transform of the White Noise Autocorrelation')
plt.show()

plt.figure(figsize=(10,7))
plt.plot(R1)
plt.grid()
plt.title('Autocorrelation of Exponential')
plt.show()

plt.figure(figsize=(10,7))
plt.stem(Freq1, Mag1, use_line_collection=True)
plt.grid(True)
plt.title('Fourier Transform of the Exponential Autocorrelation')
plt.show()


# =============================================================================
# Filtering
# =============================================================================

f1 = 600
f2 = 1000

def Passband_Selector(Filter_Info):
    
    # Bandwidth and center frequency
    Beta = Filter_Info[0]
    Omega0 = Filter_Info[1]
    
    # Cut-off frequencies
    Omega1 = f1*(2*Pi)
    Omega2 = f2*(2*Pi)
    
    # Cut-off frequency equations:
    Eq = np.zeros(2)
    Eq[0] = -0.5*Beta + np.sqrt((0.5*Beta)**2 + Omega0**2) - Omega1
    Eq[1] = 0.5*Beta + np.sqrt((0.5*Beta)**2 + Omega0**2) - Omega2
    
    return Eq

Filter_Info = fsolve(Passband_Selector, (200*2*Pi, 1900*2*Pi))

Bandwidth = Filter_Info[0] # Rad/s
Omega0 = Filter_Info[1] # Rad/s

C = 500e-9 # Select capacitor value

R = 1/(C*Bandwidth)
L = 1/(C*Omega0**2)

# =============================================================================
# Transfer functions
# =============================================================================

delta = 100
f = np.arange(1,10e6+delta,delta)

def H(f):
    w = f*2*Pi
    num = w/(R*C)
    den = np.sqrt(w**4+((1/(R*C))**2 - (2/(L*C)))*(w**2)+(1/(L*C))**2)
    y = num/den
    return y

def phase(f):
    w = f*2*Pi
    x = ((np.pi)/2) - np.arctan((w/(R*C))/(-w**2+(1/(L*C))))
    for i in range (len(w)):
        if (1/(L*C)-w[i]**2)<0:
            x[i] -= np.pi
    return x

def H5(f):
    x1 = H(f)
    x2 = H(x1)
    x3 = H(x2)
    x4 = H(x3)
    x5 = H(x4)
    return x5

num = [1/(R*C),0]
den = [1, 1/(R*C), 1/(L*C)]
num2,den2 = sig.bilinear(num, den, dt)

filtered_signal = sig.lfilter(num2,den2,W)


plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.semilogx(f, 20*np.log10(H(f)))
plt.grid(True)
plt.title("Frequency Response of Bandpass Filter")
plt.ylabel("Magnitude [dB]")
plt.subplot(2,1,2)
plt.semilogx(f, phase(f)*180/np.pi)
plt.grid(True)
plt.ylabel("Phase [deg]")
plt.xlabel("Frequency [Hz]")
plt.show

plt.figure(figsize=(10,7))
plt.plot(t, filtered_signal)
plt.grid()
plt.title('Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude [V]')
plt.show()
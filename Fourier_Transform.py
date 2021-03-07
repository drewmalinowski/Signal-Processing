import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
plt.rcParams.update({'font.size': 14})

fs = 100
Ts = 1/fs
t = np.arange(0,2,Ts)
x = np.cos(2*np.pi*t)
y = 5*np.sin(2*np.pi*t)
z = 2*np.cos((2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**2

def FFT_dirty(x,fs):
    N = len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return X_mag, X_phi, freq

def FFT(x,fs):
    N = len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0
    return X_mag, X_phi, freq

M, P, F = FFT(x,fs)
Mx,Px, Fx = FFT_dirty(x,fs)

M2, P2, F2 = FFT(y,fs)
M2x, P2x, F2x = FFT_dirty(y,fs)

M3, P3, F3 = FFT(z,fs)
M3x, P3x, F3x = FFT_dirty(z,fs)


plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid(True)
plt.title("Fast Fourier Transform 1")
plt.subplot(3,2,3)
plt.stem(Fx, Mx, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(Fx, Mx, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(Fx, Px, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(Fx, Px, use_line_collection=True)
plt.grid(True)
plt.xlim(-2,2)
plt.show()


plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, y)
plt.grid(True)
plt.title("Fast Fourier Transform 2")
plt.subplot(3,2,3)
plt.stem(F2x, M2x, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(F2x, M2x, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(F2x, P2x, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(F2x, P2x, use_line_collection=True)
plt.grid(True)
plt.xlim(-2,2)
plt.show()


plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, z)
plt.grid(True)
plt.title("Fast Fourier Transform 3")
plt.subplot(3,2,3)
plt.stem(F3x, M3x, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(F3x, M3x, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(F3x, P3x, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(F3x, P3x, use_line_collection=True)
plt.grid(True)
plt.xlim(-15,15)
plt.show()



plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid(True)
plt.title("Fast Fourier Transform 1 (CLEAN)")
plt.subplot(3,2,3)
plt.stem(F, M, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(F, M, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(F, P, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(F, P, use_line_collection=True)
plt.grid(True)
plt.xlim(-2,2)
plt.show()

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, y)
plt.grid(True)
plt.title("Fast Fourier Transform 2 (CLEAN)")
plt.subplot(3,2,3)
plt.stem(F2, M2, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(F2, M2, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(F2, P2, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(F2, P2, use_line_collection=True)
plt.grid(True)
plt.xlim(-2,2)
plt.show()

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, z)
plt.grid(True)
plt.title("Fast Fourier Transform 3 (CLEAN)")
plt.subplot(3,2,3)
plt.stem(F3, M3, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(F3, M3, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(F3, P3, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(F3, P3, use_line_collection=True)
plt.grid(True)
plt.xlim(-15,15)
plt.show()



T = 8
t = np.arange(0,2*T,Ts)
a =0
N = 15
for i in range(1,N+1):
    b = 2/(i*np.pi)*(1-np.cos(i*np.pi))
    x = b*np.sin(i*(2*np.pi/T)*t)
    a = a + x 
x = a
X1_mag, X1_phi, freq1 = FFT(x,fs)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid(True)
plt.title("Fast Fourier Transform 4")
plt.subplot(3,2,3)
plt.stem(freq1, X1_mag, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,4)
plt.stem(freq1, X1_mag, use_line_collection=True)
plt.xlim(-2,2)
plt.grid(True)
plt.subplot(3,2,5)
plt.stem(freq1, X1_phi, use_line_collection=True)
plt.grid(True)
plt.subplot(3,2,6)
plt.stem(freq1, X1_phi, use_line_collection=True)
plt.grid(True)
plt.xlim(-2,2)
plt.show()
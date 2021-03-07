# =============================================================================
# Drew Malinowski
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 10}) 

# =============================================================================
# =============================================================================

pi = np.pi
x = np.arange(-2,2,0.1)

fx = np.sin((pi/2)*x)

taylor1 = (pi/2)*x

taylor3 = (pi/2)*x - ((pi**3)/36)*(x**3)

taylor5 = (pi/2)*x - ((pi**3)/36)*(x**3) + ((pi**5)/1920)*(x**5)

taylor7 = (pi/2)*x - ((pi**3)/36)*(x**3) + ((pi**5)/1920)*(x**5) - ((pi**7)/322560)*(x**7)


plt.figure(figsize=(10,20))

plt.subplot(5,1,1)
plt.plot(x,fx)
plt.grid()
plt.title('f(x)')
plt.xlim([-2,2])

plt.subplot(5,1,2)
plt.plot(x,taylor1)
plt.grid()
plt.title('first degree')
plt.xlim([-2,2])

plt.subplot(5,1,3)
plt.plot(x,taylor3)
plt.grid()
plt.title('third degree')
plt.xlim([-2,2])

plt.subplot(5,1,4)
plt.plot(x,taylor5)
plt.grid()
plt.title('fifth degree')
plt.xlim([-2,2])

plt.subplot(5,1,5)
plt.plot(x,taylor7)
plt.grid()
plt.title('seventh degree')
plt.xlim([-2,2])
plt.show()




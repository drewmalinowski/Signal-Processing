import matplotlib.pyplot as plt
import scipy.signal as sig
plt.rcParams.update({'font.size': 14}) 

def G(s):
    y = 0.20833/(s+4) - 0.35/(s+2) + 0.141667/(s-8)
    return y

def A(s):
    y = 1.5/(s+1) - 0.5/(s+3)
    return y

def B(s):
    y = s**2 + 26*s + 168
    return y

numG = [1, 9]
denG = [1, -2, -40, -64]

[Zg,Pg,Kg] = sig.tf2zpk(numG,denG)

#print(Zg, Pg, Kg)

numA = [1, 4]
denA = [1, 4, 3]

[Za,Pa,Ka] = sig.tf2zpk(numA,denA)

#print(Za, Pa, Ka)

numB = [1, 26, 168]
denB = [1]

[Zb,Pb,Kb] = sig.tf2zpk(numB,denB)

#print(Zb, Pb, Kb)

def OpenLoopTransfer(s):
    y = (s+9)/((s-8)*(s+2)*(s+3)*(s+1))
    return y

numX = [1,9]
denX = sig.convolve(sig.convolve(sig.convolve([1,-8],[1,2]),[1,3]),[1,1])    

[Zx,Px,Kx] = sig.tf2zpk(numX,denX)

tOUT1, yOUT1 = sig.step((numX,denX))

#print(Zx,Px,Kx)

def CloseLoopTransfer(numG,denG,numA,denA,numB):
    y = (numA*numG)/(denA*denG + numG*denA*numB)
    return y

numC = sig.convolve(numA,numG)
denC = sig.convolve(denA,denG) + sig.convolve(sig.convolve(numG,denA),numB)

[Zc,Pc,Kc] = sig.tf2zpk(numC,denC)

print("Zeroes:",Zc)
print("Poles:",Pc)

#print("NUMERATOR:", numC)
#print("DENOMINATOR:", denC)

tOUT2, yOUT2 = sig.step((numC,denC))

plt.figure(figsize=(10,8))
plt.plot(tOUT1, yOUT1)
plt.grid(True)
plt.ylabel("h(t)")
plt.xlabel("t")
plt.title("Open Loop Step Response")

plt.figure(figsize=(10,8))
plt.plot(tOUT2, yOUT2)
plt.grid(True)
plt.ylabel("h(t)")
plt.xlabel("t")
plt.title("Closed Loop Step Response")
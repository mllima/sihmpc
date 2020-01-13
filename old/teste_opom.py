# using the class OPOM
# MLima

from opom import OPOM, TransferFunction
from scipy import signal
import matplotlib.pyplot as plt

#%% system TF
num11 = [0.2**2]
den11 =[1, 2*0.1*0.2, 0.2**2]   
h11 = TransferFunction(num11, den11, delay=1)  # delay in seconds

num12 = [1.5]
den12 = [23*62, 23+62, 1]
h12 = TransferFunction(num12, den12)

num21 = [-1.4]
den21 = [30*90, 30+90, 1]
h21 = TransferFunction(num21, den21, delay=0)

num22 = [2.8]
den22 = [90, 1]
h22 = TransferFunction(num22, den22)

h = [[h11, h12], [h21, h22]]


#%% OPOM
Ts = 0.1
sys_opom = OPOM(h, Ts)

#%% Response
# -> from the OPOM representation
# -> the input is delta u, y=f(du,x)
# y11 = y[0][:,0]
# y12 = y[1][:,0]
# y21 = y[0][:,1]
# y22 = y[1][:,1]
sys = signal.StateSpace(sys_opom.A, sys_opom.B, sys_opom.C, sys_opom.D, dt=Ts)
t, y = signal.dimpulse(sys, n=3000)

# -> from the original representation
g11 = h11.tf
T11, y11 = signal.step(g11)
T11 += h11.delay

g12 = h12.tf
T12, y12 = signal.step(g12)
T12 += h12.delay

g21 = h21.tf
T21, y21 = signal.step(g21)
T21 += h21.delay

g22 = h22.tf
T22, y22 = signal.step(g22)
T22 += h22.delay

#%% Plot original vs OPOM
plt.subplot(221).set_title("y11")
plt.plot(T11, y11)
plt.step(t, y[0][:,0])
plt.subplot(222).set_title("y12")
plt.plot(T12, y12)
plt.step(t, y[1][:,0])
plt.subplot(223).set_title("y21")
plt.plot(T21, y21)
plt.step(t, y[0][:,1])
plt.subplot(224).set_title("y22")
plt.plot(T22, y22)
plt.step(t, y[1][:,1])

#%% show
plt.show()

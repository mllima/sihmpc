# -*- coding: utf-8 -*-

from opom import OPOM, TransferFunction
from sihmpc import IHMPCController

import time
import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# %% Four tanks - dimension (K. H. Johansson,2000)

kc = 0.5  # V/cm  - level sensor
k1 = 3.14 # cm3/(V.s) - electric actuator 1
k2 = 3.29 # cm3/(V.s) - electric actuator 1

A = np.array([28, 32, 28, 32]) #cm2 
a = np.array([0.071, 0.057, 0.071, 0.057]) # cm2
g = 981 # cm/s2

f1max = f2max = 2.5 # l/min  pumps capacity
cf = 1000/60        # factor to convert l/min to cm3/s

gamma = [0.43, 0.34]  # nom minimum-phase

h0 = np.array([12.6, 13.0, 4.8, 4.9]) # cm - inicial level


# %% Modelo OPOM

Ts = 1

T = np.round((A/a)*np.sqrt(2*h0/g))  # time constants
c11 = kc*k1*T[0]/A[0]
c12 = kc*k2*T[0]/A[0]
c21 = kc*k1*T[1]/A[1]
c22 = kc*k2*T[1]/A[1]
c32 = kc*k2*T[2]/A[2]
c41 = kc*k1*T[3]/A[3]


# Transfer functions
num11 = [gamma[0]*c11]
den11 =[T[0], 1]   
h11 = TransferFunction(num11, den11, delay=0) 

num12 = [(1-gamma[1])*c12]
den12 = [T[0]*T[2], T[0]+T[2], 1]
h12 = TransferFunction(num12, den12, delay=0)

num21 = [(1-gamma[0])*c21]
den21 = [T[1]*T[3], T[1]+T[3], 1]
h21 = TransferFunction(num21, den21, delay=0)

num22 = [gamma[1]*c22]
den22 = [T[1], 1]
h22 = TransferFunction(num22, den22, delay=0) 

num32 = [(1-gamma[1])*c32]
den32 =[T[2], 1]   
h32 = TransferFunction(num32, den32, delay=0)  

num41 = [(1-gamma[0])*c41]
den41 = [T[3], 1]
h41 = TransferFunction(num41, den41, delay=0)

# General system
h = [[h11, h12], [h21, h22], [[], h32], [h41, []]]
sys = OPOM(h, Ts)

# test
# sys2 = ctl.StateSpace(sys.A, sys.B, sys.C, sys.D, Ts)
# T = [i for i in range(0,100,1)]
# T, yout = ctl.impulse_response(sys2, T=T, input=0)


# %% Controlador

N = 5  # horizon in steps
c = IHMPCController(sys, N) #, ulb = [0, 0]) #, uub = [42/k1, 42/k2])

# sub-objectives
Q = 1
R = 1

Vy1 = c.subObj(y=[0], Q=Q, sat= N*(0.01*kc)**2)
Vy2 = c.subObj(y=[1], Q=Q, sat= N*(0.01*kc)**2)
Vy3 = c.subObj(y=[2], Q=Q, sat= N*(15*kc)**2)
Vy4 = c.subObj(y=[3], Q=Q, sat= N*(15*kc)**2)

Vy1N = c.subObj(syN=[0], Q=Q, sat= (0.01*kc)**2)
Vy2N = c.subObj(syN=[1], Q=Q, sat= (0.01*kc)**2)
Vy3N = c.subObj(syN=[2], Q=Q, sat= (0.1*kc)**2)
Vy4N = c.subObj(syN=[3], Q=Q, sat= (0.1*kc)**2)

Vi1N = c.subObj(siN=[0], Q=Q, sat= 0.01**2)
Vi2N = c.subObj(siN=[1], Q=Q, sat= 0.01**2)
Vi3N = c.subObj(siN=[2], Q=Q, sat= 0.01**2)
Vi4N = c.subObj(siN=[3], Q=Q, sat= 0.01**2)

Vdu1 = c.subObj(du=[0], Q=R, sat=N*2.5**2)
Vdu2 = c.subObj(du=[1], Q=R, sat=N*2.5**2)

# pesos - inicialização dos pessos (na ordem de criação dos subobjetivos)
pesos = np.array([1/Vy1.gamma, 1/Vy2.gamma,
                  1/Vy3.gamma, 1/Vy4.gamma,
                  1/Vy1N.gamma, 1/Vy2N.gamma,
                  1/Vy3N.gamma, 1/Vy4N.gamma, 
                  1/Vi1N.gamma, 1/Vi2N.gamma,
                  1/Vi3N.gamma, 1/Vi4N.gamma,
                  1/Vdu1.gamma, 1/Vdu2.gamma])

# %% Closed loop
JPlot = []
duPlot = []
yPlot = []
uPlot = []
xPlot = []
pesosPlot = []
vPlot = []

u = [3, 3]  	# controle anterior (V)
x = kc* np.append([0, 5, 0, 5], np.zeros(c.nx-c.nxs)) 	# Estado inicial
tEnd = 1000     	    # Tempo de simulação 

tocMPC = []

ysp = kc* np.array([0, 0, 0, 0])  # V  - kc transforma cm em V

w0 = []
lam_w0 = []
lam_g0 = []
    
for k in np.arange(0, tEnd/Ts):

    t1 = time.time()
    pesosPlot += [pesos]
        
    # to test a change in the set-point    
    # if k > (tEnd/2)/Ts: 
    #     ysp = [12.6, 12.5, 4.8, 4.9]

    sol = c.mpc(x0=x, ySP=ysp, w0=w0, u0=u, pesos=pesos, lam_w0=lam_w0, lam_g0=lam_g0, ViN_ant=[])
    
    t2 = time.time()
    tocMPC += [t2-t1]

    w0 = sol['w_opt'][:]
    lam_w0 = sol['lam_w']
    lam_g0 = sol['lam_g']
    
    du = sol['du_opt'][:, 0].full()
    duPlot += [du]

    J = float(sol['J'])
    JPlot.append(J)

    #sub-objectives values
    v=[]
    for i in range(len(c.V)):
        v.append(float(c.V[i].F(x, u, w0, ysp)))
    vPlot.append(v)


    # ## Simula o sistema ###
    res = c.dynF(x0=x, du0=du, u0=u)
    x = res['xkp1'].full()
    u = res['ukp1'].full()
    y = res['ykp1'].full()
    yPlot.append(y/kc)
    uPlot.append(u)
    xPlot.append(x)

    w0 = c.warmStart(sol, ysp)
    
    du_warm = w0

    # new_pesos = c.satWeights(x, u, du_warm, ysp)
    # alfa = 0.7
    # pesos = alfa*pesos + (1-alfa)*new_pesos

    
print('Tempo de execução do MPC. Média: %2.3f s, Max: %2.3f s' %
                                    (np.mean(tocMPC), np.max(tocMPC)))

# %% Plot

t = np.arange(0, tEnd, Ts)

yPlot = np.hstack(yPlot)
duPlot = np.hstack(duPlot)
uPlot = np.hstack(uPlot)
xPlot = np.hstack(xPlot)
vPlot = np.vstack(vPlot).T
JPlot = np.hstack(JPlot)
pesosPlot = np.array(pesosPlot).T

fig1 = plt.figure(1)
fig1.suptitle("Output and Control Signals")
fig1.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(1, 3, 1)
plt.step(t, yPlot.T)
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend(['y{}'.format(i) for i in range(len(yPlot[0]))])
plt.subplot(1, 3, 2)
plt.step(t, duPlot.T)
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend(['du{}'.format(i) for i in range(np.shape(duPlot)[0])])
plt.subplot(1, 3, 3)
plt.step(t, uPlot.T)
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend(['u{}'.format(i) for i in range(np.shape(uPlot)[0])])

fig2 = plt.figure(2)
fig2.suptitle("OPOM Variables")
fig2.text(0.5, 0.04, 'Time', ha='center', va='center')
y = round(c.nx/4+0.5)
x = round(c.nx/y+0.5)
for i in range(c.nx):
    plt.subplot(x, y, i+1)
    if i<c.nxs:
        label = 'xs'+str(i)
    elif i<c.nxs+c.nxd:
        label = 'xd'+str(i-c.nxs)
    elif i<c.nxs+c.nxd+c.nxi:
        label = 'xi'+str(i-c.nxs-c.nxd)
    else:
        label = 'xz'+str(i-c.nxs-c.nxd-c.nxi)
    plt.step(t, xPlot[i,:], label=label)
    plt.legend(loc=0, fontsize='large')
    plt.grid()
    plt.legend()

fig3 = plt.figure(3)
fig3.suptitle("Weights")
fig3.text(0.5, 0.04, 'Time', ha='center', va='center')
nw = len(pesos)
y = round(nw/4+0.5)
x = round(nw/y+0.5)
for i in range(nw):
    plt.subplot(x, y, i+1)
    label = c.VJ[i].weight.name()
    plt.step(t, pesosPlot[i], label=label) #'w'+str(i+1))
    plt.legend(loc=0, fontsize='large')
    plt.grid()
    plt.legend()


fig4 = plt.figure(4)
fig4.suptitle("Total Cost")
fig4.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.plot(t,JPlot)
#plt.show()

fig5 = plt.figure(5)
fig5.suptitle("Local Costs")
l = len(c.V)
y = round(np.sqrt(l)+0.5)
x = round(l/y+0.5)
for i in range(l):
    plt.subplot(x,y,i+1)
    label = c.V[i].name
    plt.step(t,vPlot[i], label = label)
    plt.legend()

fig6 = plt.figure(6)
fig6.suptitle("Weighted Local Costs")
l = len(c.VJ)
y = round(np.sqrt(l)+0.5)
x = round(l/y+0.5)
for i in range(l):
    plt.subplot(x,y,i+1)
    label = c.VJ[i].weight.name() + '*' + c.VJ[i].name 
    plt.step(t,pesosPlot[i]*vPlot[i], label = label)
    plt.legend()

plt.show()

pass

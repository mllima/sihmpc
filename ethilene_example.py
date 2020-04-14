# -*- coding: utf-8 -*-

from opom import OPOM, TransferFunction
from sihmpc import IHMPCController

import time
import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# %% Modelo OPOM

Ts = 1  #min

# Transfer functions
num11 = [10e-4*95, -10e-4]
den11 =[32.16, 4.65, 1]   
h11 = TransferFunction(num11, den11, delay=1)  # delay

num12 = [-2.3e-3]
den12 = [1, 0]
h12 = TransferFunction(num12, den12, delay=0)

num13 = [3.2e-3, -3.2e-3]
den13 = [64.55, 8.83, 1]
h13 = TransferFunction(num13, den13, delay=2)

num14 = [-7.5e-6]
den14 = [1, 0]
h14 = TransferFunction(num14, den14, delay=0)

num21 = [-1.69e-4]
den21 = [1, 0]
h21 = TransferFunction(num21, den21, delay=3)

num22 = [2.1e-4]
den22 = [1, 0]
h22 = TransferFunction(num22, den22, delay=8)

num23 = [-1.9e-3*1.47, -1.9e-3]
den23 = [9.67, 13.55, 1]
h23 = TransferFunction(num23, den23, delay=0)

num24 = [-1.07e-4]
den24 = [1, 0]
h24 = TransferFunction(num24, den24, delay=0)

num31 = [-8.1e-3*0.02, 8.1e-3]
den31 = [52.45, 11.92, 1]
h31 = TransferFunction(num31, den31, delay=4)

num32 = [-5.5e-5]
den32 = [1, 0]
h32 = TransferFunction(num32, den32, delay=15)

num33 = [9.6e-3, 9.6e-3]
den33 = [54.42, 6.58, 1]
h33 = TransferFunction(num33, den33, delay=2)

num34 = [-2.53e-3]
den34 = [25, 0] #den34 = [1, 0]
h34 = TransferFunction(num34, den34, delay=10)

num41 = [-3.9e-5]
den41 = [1, 0]
h41 = TransferFunction(num41, den41, delay=4)

num42 = [5.7e-5]
den42 = [1, 0]
h42 = TransferFunction(num42, den42, delay=8)

num43 = [-1.4e-3, -1.4e-3]
den43 = [8.67, 14.48, 1]
h43 = TransferFunction(num43, den43, delay=0)

num44 = [7.6e-5]
den44 = [1, 0]
h44 = TransferFunction(num44, den44, delay=6)

# General system
h = [[h11, h12, h13, h14], [h21, h22, h23, h24], [h31, h32, h33, h34], [h41, h42, h43, h44]]
sys = OPOM(h, Ts)

# %% Controlador

N = 8 + sys.theta_max # horizon in steps
umax = [6900, 5700, 100, 95]
umin = [5700, 4500, 0, 25]
dumax = [25, 25, 2, 10]
dumin = [-25, -25, -2, -10]

c = IHMPCController(sys, N, uub=umax, ulb=umin, duub=dumax, dulb=dumin)

# sub-objectives
Q = 1
R = 1

Vy1 = c.subObjComposed(y=[0], Q=Q)
Vy2 = c.subObjComposed(y=[1], Q=Q)
Vy3 = c.subObjComposed(y=[2], Q=Q)
Vy4 = c.subObjComposed(y=[3], Q=Q)

Vi1N = c.subObj(siN=[0], Q=Q)
Vi2N = c.subObj(siN=[1], Q=Q)
Vi3N = c.subObj(siN=[2], Q=Q)
Vi4N = c.subObj(siN=[3], Q=Q)

Vdu1 = c.subObj(du=[0], Q=R)
Vdu2 = c.subObj(du=[1], Q=R)
Vdu3 = c.subObj(du=[2], Q=R)
Vdu4 = c.subObj(du=[3], Q=R)

# limits of the sub-objectives
# Vy1.lim(0, np.inf)

# satisficing limits 
Vy1.satLim(N*0.1**2)
Vy2.satLim(N*1.85**2)
Vy3.satLim(N*2.5**2)
Vy4.satLim(N*0.3**2)

Vi1N.satLim(1**2)
Vi2N.satLim(1**2)
Vi3N.satLim(1**2)
Vi4N.satLim(1**2)

Vdu1.satLim(N*25**2)
Vdu2.satLim(N*25**2)
Vdu3.satLim(N*2**2)
Vdu4.satLim(N*10**2)

# pesos - inicialização dos pessos
pesos = np.array([1/Vy1.gamma,  1/Vy2.gamma,  1/Vy3.gamma,  1/Vy4.gamma,
                  1/Vi1N.gamma, 1/Vi2N.gamma, 1/Vi3N.gamma, 1/Vi4N.gamma,
                  1/Vdu1.gamma, 1/Vdu2.gamma, 1/Vdu3.gamma, 1/Vdu4.gamma])

fator = 0.1/min(pesos)
pesos = fator*pesos

# %% Closed loop
JPlot = []
duPlot = []
yPlot = []
uPlot = []
xPlot = []
pesosPlot = []

vy1Plot = []
vy2Plot = []
vy3Plot = []
vy4Plot = []

vy1NPlot = []
vy2NPlot = []
vy3NPlot = []
vy4NPlot = []

vi1NPlot = []
vi2NPlot = []
vi3NPlot = []
vi4NPlot = []

vdu1Plot = []
vdu2Plot = []
vdu3Plot = []
vdu4Plot = []

vtPlot = []

u = np.array([6357, 5280, 82, 48]) # controle inicial
x = np.append([6.22e+00, 1.57e+01, 2.785e+02, 1.81e+01], np.zeros(c.nx-c.nxs))  # initial state = inicial output

tEnd = 500     	    # Tempo de simulação (min)

tocMPC = []

w0 = []
lam_w0 = []
lam_g0 = []

# set-points
ysp = [5.9, 17.85, 277.5, 18.8]

for k in np.arange(0, tEnd/Ts):

    t1 = time.time()
    pesosPlot += [pesos]
        
    #to test a change in the set-point    
    # if k > (tEnd/6)/Ts: 
    #     ysp[1] = 20

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
    vy1Plot.append(float(Vy1.Vi[0].F(x, u, w0, ysp)))
    vy1NPlot.append(float(Vy1.Vi[1].F(x, u, w0, ysp)))
    vi1NPlot.append(float(Vi1N.F(x, u, w0, ysp)))
    vy2Plot.append(float(Vy2.Vi[0].F(x, u, w0, ysp)))
    vy2NPlot.append(float(Vy2.Vi[1].F(x, u, w0, ysp)))
    vi2NPlot.append(float(Vi2N.F(x, u, w0, ysp)))
    vy3Plot.append(float(Vy3.Vi[0].F(x, u, w0, ysp)))
    vy3NPlot.append(float(Vy3.Vi[1].F(x, u, w0, ysp)))
    vi3NPlot.append(float(Vi3N.F(x, u, w0, ysp)))
    vy4Plot.append(float(Vy4.Vi[0].F(x, u, w0, ysp)))
    vy4NPlot.append(float(Vy4.Vi[1].F(x, u, w0, ysp)))
    vi4NPlot.append(float(Vi4N.F(x, u, w0, ysp)))
    vdu1Plot.append(float(Vdu1.F(x, u, w0, ysp)))
    vdu2Plot.append(float(Vdu2.F(x, u, w0, ysp)))
    vdu3Plot.append(float(Vdu3.F(x, u, w0, ysp)))
    vdu4Plot.append(float(Vdu4.F(x, u, w0, ysp)))

    #terminal cost
    vtPlot.append(float(c.Vt.F(x, u, w0, ysp)))

    # ## Simula o sistema ###
    res = c.dynF(x0=x, du0=du, u0=u)
    x = res['xkp1'].full()
    u = res['ukp1'].full()
    y = res['ykp1'].full()
    yPlot.append(y)
    uPlot.append(u)
    xPlot.append(x)
    
    w0 = c.warmStart(sol, ysp)
    
    du_warm = w0

    # new_pesos = fator * c.satWeights(x, u, du_warm, ysp)
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
JPlot = np.hstack(JPlot)
pesosPlot = np.array(pesosPlot)

fig1 = plt.figure(1)
fig1.suptitle("Output and Control Signals")
fig1.text(0.5, 0.04, 'Time', ha='center', va='center')
y = round(c.nx/4+0.5)
x = round(c.nx/y+0.5)
for i in range(c.nu + c.ny):
    plt.subplot(2, 4, i+1)
    if i<c.nu:
        label = 'du'+str(i)
        plt.step(t, duPlot[i,:], label=label)
    else:
        label = 'y'+str(i-c.nu)
        plt.step(t, yPlot[i-c.nu,:], label=label)
    plt.grid()
    plt.legend()

#plt.show()
# plt.savefig("SIHMPCOutput.png")

#
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

#plt.show()
#plt.savefig("SisoSIHMPCopomVar.png")

fig3 = plt.figure(3)
fig3.suptitle("Weights")
fig3.text(0.5, 0.04, 'Time', ha='center', va='center')
nw = len(pesos)
y = round(nw/4+0.5)
x = round(nw/y+0.5)
for i in range(nw):
    plt.subplot(x, y, i+1)
    label = c.VJ[i].weight.name()
    plt.step(t, pesosPlot[:,i], label=label)
    plt.legend(loc=0, fontsize='large')
    plt.grid()
    plt.legend()
#plt.show()
#plt.savefig("SisoSIHMPWeigthVar.png")

fig4 = plt.figure(4)
fig4.suptitle("Total Cost")
fig4.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.plot(t,JPlot)
#plt.show()

legends = ['Vy1', 'Vy2', 'Vy3', 'Vy4',
           'Vy1N', 'Vy2N', 'Vy3N', 'Vy4N',
           'Vi1N', 'Vi2N', 'Vi3N', 'Vi4N',
           'Vdu1', 'Vdu2', 'Vdu3', 'Vdu4',
           'Vt']
Values = [vy1Plot, vy2Plot, vy3Plot, vy4Plot, 
          vy1NPlot, vy2NPlot, vy3NPlot, vy4NPlot,
          vi1NPlot, vi2NPlot, vi3NPlot, vi4NPlot,
          vdu1Plot, vdu2Plot, vdu3Plot, vdu4Plot,
          vtPlot]           
fig5 = plt.figure(5)
fig5.suptitle("Local Costs")
fig5.text(0.5, 0.04, 'Time', ha='center', va='center')
x = 4
y = 5
for i in range(len(Values)):
    plt.subplot(x,y,i+1)
    plt.step(t,Values[i], label=legends[i])
    plt.legend()

plt.show()

pass
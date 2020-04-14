# -*- coding: utf-8 -*-
# distillation example - Wood & Berry

from opom import OPOM, TransferFunction
from sihmpc import IHMPCController

import time
import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# %% Modelo OPOM

Ts = 1  #min

# Transfer functions

num11 = [12.8]
den11 = [16.7, 1]
h11 = TransferFunction(num11, den11, delay=1)  

num12 = [-18.9]
den12 = [21.0, 1]
h12 = TransferFunction(num12, den12, delay=3)

num21 = [6.6]
den21 = [10.9, 1]
h21 = TransferFunction(num21, den21, delay=7)

num22 = [-19.4]
den22 = [14.4, 1]
h22 = TransferFunction(num22, den22, delay=3)

# General system
h = [[h11, h12], [h21, h22]]
sys = OPOM(h, Ts)

# %% Controlador

N = 10  # horizon in steps
c = IHMPCController(sys, N)

# sub-objectives
Q1 = 1
Q2 = 1
R = 1

Vy1 = c.subObj(y=[0], Q=Q1)
Vy2 = c.subObj(y=[1], Q=Q2)

Vy1N = c.subObj(syN=[0], Q=Q1)
Vy2N = c.subObj(syN=[1], Q=Q2)

Vi1N = c.subObj(siN=[0], Q=Q1)
Vi2N = c.subObj(siN=[1], Q=Q2)

Vdu1 = c.subObj(du=[0], Q=R)
Vdu2 = c.subObj(du=[1], Q=R)

# satisficing limits 
Vy1.satLim(N*0.1**2)
Vy1N.satLim(0.01**2)
Vi1N.satLim(1**2)

Vy2.satLim(N*0.1**2)
Vy2N.satLim(0.01**2)
Vi2N.satLim(1**2)

Vdu1.satLim(N*0.5**2)
Vdu2.satLim(N*0.5**2)

# pesos - inicialização dos pessos (na ordem de criação dos subobjetivos)
pesos = np.array([1/Vy1.gamma, 1/Vy2.gamma, 
                  1/Vy1N.gamma, 1/Vy2N.gamma,
                  1/Vi1N.gamma, 1/Vi2N.gamma,
                  1/Vdu1.gamma, 1/Vdu2.gamma])

# %% Closed loop
JPlot = []
duPlot = []
yPlot = []
uPlot = []
xPlot = []
pesosPlot = []
vy1Plot = []
vy1NPlot = []
vi1NPlot = []
vy2Plot = []
vy2NPlot = []
vi2NPlot = []
vdu1Plot = []
vdu2Plot = []
vtPlot = []

u = np.ones(sys.nu)*0  	# controle anterior
x = np.append([96, 0.5], np.zeros(sys.nx-2)) 	# Estado inicial
tEnd = 50     	    # Tempo de simulação

tocMPC = []

ysp = [96, 0.5]

w0 = []
lam_w0 = []
lam_g0 = []
    
for k in np.arange(0, tEnd/Ts):

    t1 = time.time()
    pesosPlot += [pesos]
        
    #to test a change in the set-point    
    if k > (tEnd/10)/Ts: 
        ysp = [99, 0.1]
    # if k > (tEnd/2)/Ts: 
    #     ysp = [99, 0.1]

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
    vy1Plot.append(float(Vy1.F(x, u, w0, ysp)))
    vy1NPlot.append(float(Vy1N.F(x, u, w0, ysp)))
    vi1NPlot.append(float(Vi1N.F(x, u, w0, ysp)))

    vy2Plot.append(float(Vy2.F(x, u, w0, ysp)))
    vy2NPlot.append(float(Vy2N.F(x, u, w0, ysp)))
    vi2NPlot.append(float(Vi2N.F(x, u, w0, ysp)))

    vdu1Plot.append(float(Vdu1.F(x, u, w0, ysp)))
    vdu2Plot.append(float(Vdu2.F(x, u, w0, ysp)))
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

    new_pesos = c.satWeights(x, u, du_warm, ysp)
    alfa = 0.7
    pesos = alfa*pesos + (1-alfa)*new_pesos

    
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
plt.subplot(4, 1, 1)
plt.step(t, yPlot[0,:])
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend(['y_1'])
plt.subplot(4, 1, 2)
plt.step(t, yPlot[1,:])
plt.grid()
plt.legend(['y_2'])
plt.subplot(4, 1, 3)
plt.step(t, duPlot.T)
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend(['du{}'.format(i) for i in range(np.shape(duPlot)[0])])
plt.subplot(4, 1, 4)
plt.step(t, uPlot.T)
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend(['u{}'.format(i) for i in range(np.shape(uPlot)[0])])

# import pdb
# pdb.set_trace()

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
#seg = [0,2,1,3,4]
for i in range(nw):
    plt.subplot(x, y, i+1)
    label = c.VJ[i].weight.name()
    plt.step(t, pesosPlot[:,i], label=label) #'w'+str(i+1))
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

fig5 = plt.figure(5)
fig5.suptitle("Local Costs")
fig5.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(3,3,1)
plt.step(t,vy1Plot)
plt.legend(['Vy1'])
plt.subplot(3,3,2)
plt.step(t,vy1NPlot)
plt.legend(['Vy1N'])
plt.subplot(3,3,3)
plt.step(t,vi1NPlot)
plt.legend(['Vi1N'])
plt.subplot(3,3,4)
plt.step(t,vy2Plot)
plt.legend(['Vy2'])
plt.subplot(3,3,5)
plt.step(t,vy2NPlot)
plt.legend(['Vy2N'])
plt.subplot(3,3,6)
plt.step(t,vi2NPlot)
plt.legend(['Vi2N'])
plt.subplot(3,3,7)
plt.step(t,vdu1Plot)
plt.legend(['Vdu1'])
plt.subplot(3,3,8)
plt.step(t,vdu2Plot)
plt.legend(['Vdu2'])

plt.show()

pass
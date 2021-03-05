# -*- coding: utf-8 -*-
# distillation example - Wood & Berry

from opom import OPOM, TransferFunction
from sihmpc import IHMPCController

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import control as ctl
import casadi as csd

# %% Modelo OPOM

##### Modelo

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

# Sistema acoplado
h = [[h11, h12], [h21, h22]]

# %%% modelo OPOM

Ts = 1  #min
sys = OPOM(h, Ts)

# %% Controlador

N = 10  # horizon in steps
c = IHMPCController(sys, N, ulb=[0,0])

# sub-objectives
Q = 1
R = 1

Vy1 = c.subObjComposed(y=[0], Q=Q, sat=N*0.5**2)
Vy2 = c.subObjComposed(y=[1], Q=Q, sat=N*2.0**2)

Vdu1 = c.subObj(du=[0], Q=R, sat=N*0.3**2)
Vdu2 = c.subObj(du=[1], Q=R, sat=N*0.3**2)

Vi1N = c.subObj(siN=[0], Q=Q, addJ=False)   # addJ exclui o subobjetivo da função objetivo
Vi2N = c.subObj(siN=[1], Q=Q, addJ=False)

# recalcula os pesos do custo terminal
c.init_Qt()

# pesos - inicialização dos pessos (na ordem de criação dos subobjetivos)
pesos = c.init_pesos()

# %%% Closed loop

u = [1.95, 1.71]  	                            # controle anterior [Reflux, Steam] <lb/min>
x = np.append([96, 0.5], np.zeros(sys.nx-2)) 	# Estado inicial 
ysp = [96, 0.5]                                 # Concentrações de saída [xD, xB] <mol%>

tEnd = 800     	    # Tempo de simulação

w0 = []
lam_w0 = []
lam_g0 = []

JPlot = []
duPlot = []
yPlot = [(sys.C@x).reshape(c.ny,1)]
uPlot = []
xPlot = []
pesosPlot = []
vPlot = []
vJ = []

tocMPC = []   
for k in np.arange(0, tEnd/Ts):

    t1 = time.time()
    pesosPlot += [pesos]
        
    # to test a change in the set-point    
    if k > 50/Ts: 
        ysp = [96, 1]

    if k > (tEnd/2)/Ts: 
        ysp = [95.5, 1]

    sol = c.mpc(x0=x, ySP=ysp, w0=w0, u0=u, pesos=pesos, lam_w0=lam_w0, lam_g0=lam_g0, ViN_ant=[])
    
    t2 = time.time()
    tocMPC += [t2-t1]

    w0 = sol['w_opt'][:]
    lam_w0 = sol['lam_w']
    lam_g0 = sol['lam_g']
    
    du = sol['du_opt'][:, 0].full()

    J = float(sol['J'])
    JPlot.append(J)

    #all sub-objectives values
    v=[]
    for i in range(len(c.V)):
        v.append(float(c.V[i].F(x, u, w0, ysp)))
    vPlot.append(v)

    #sub-objectives that are in the objective function
    v=[]
    for i in range(len(c.VJ)):
        v.append(float(c.VJ[i].F(x, u, w0, ysp)))
    vJ.append(v)

    # ## Simula o sistema ###
    res = c.dynF(x0=x, du0=du, u0=u)
    x = res['xkp1'].full()
    u = res['ukp1'].full()
    y = res['ykp1'].full()
    yPlot.append(y)
    uPlot.append(u)
    xPlot.append(x)
    duPlot += [du]

    w0 = c.warmStart(sol, ysp)

    new_pesos, _, _ = c.satWeights2(x, u, w0, ysp)
    alfa = 0.0
    pesos = alfa*pesos + (1-alfa)*new_pesos
    
print('Tempo de execução do MPC. Média: {:.3f} s, Max: {:.3f} s (index: {:d}/{:d})'.format(np.mean(tocMPC), 
                np.max(tocMPC), tocMPC.index(np.max(tocMPC)), int(k)))

# %% Plot

t = np.arange(Ts, tEnd+Ts, Ts)

yPlot = np.hstack(yPlot)
duPlot = np.hstack(duPlot)
uPlot = np.hstack(uPlot)
xPlot = np.hstack(xPlot)
vPlot = np.vstack(vPlot).T
vJ = np.vstack(vJ).T
JPlot = np.hstack(JPlot)
pesosPlot = np.array(pesosPlot).T

fig1 = plt.figure(1)
fig1.suptitle("Output and Control Signals")
fig1.text(0.5, 0.04, 'Time', ha='center', va='center')
grid = plt.GridSpec(2, 3)  # 2 rows 3 cols

plt.subplot(grid[0,0])
plt.plot(np.append([0],t), yPlot[0], label=r'$x_D$')
plt.legend()

plt.subplot(grid[1,0])
plt.plot(np.append([0],t), yPlot[1], label=r'$x_B$')
plt.legend()

plt.subplot(grid[:,1])
plt.step(t, duPlot.T)
plt.legend()
plt.legend([r'$\Delta R$', r'$\Delta S$'])

plt.subplot(grid[:,2])
plt.step(t, uPlot.T)
plt.legend(loc=0, fontsize='large')
plt.legend(['R', 'S'])

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
y = round(np.sqrt(nw)+0.5)
x = round(nw/y+0.5)
leg = ['$w_{x_D}$','$w_{x_B}$','$w_{R}$', '$w_{S}$', r'$w_{x_D(N)}$', r'$w_{x_B(N)}$']
for i in range(nw):
    plt.subplot(x, y, i+1)
    label = leg[i] #c.VJ[i].weight.name()
    plt.step(t, pesosPlot[i], label=label)
    plt.legend(loc=0, fontsize='large')
    plt.grid()
    plt.legend()

fig4 = plt.figure(4)
fig4.suptitle("Total Cost")
fig4.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.plot(t,JPlot)

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
    plt.step(t,pesosPlot[i]*vJ[i], label = label)
    plt.legend()

fig7 =plt.figure(7)
fig7.suptitle("Normalized Weights")
nw = len(pesos)
y = round(np.sqrt(nw)+0.5)
x = round(nw/y+0.5)

for i in range(nw):
    #plt.subplot(x,y,i+1)
    label = 'n' + c.VJ[i].weight.name() 
    plt.step(t,pesosPlot[i]*c.VJ[i].gamma, label = label)
    plt.legend()

plt.show()

import plotly.graph_objects as go
fig = go.Figure()
for i in range(nw):
    label = 'n' + c.VJ[i].weight.name() 
    fig.add_trace(go.Scatter(x=t, y=pesosPlot[i]*c.VJ[i].gamma,
                    mode='lines',
                    name=label))
fig.show()

pass

import pickle
func = lambda x: np.sqrt(x/N)
parm = [func(c.VJ[i].gamma) for i in range(len(c.VJ))]
file = 'dist_{}_{}.dat'.format(N, parm)
outfile = open(file,'wb')
pickle.dump((yPlot, duPlot, uPlot, xPlot, vPlot, vJ, JPlot, pesosPlot),outfile)
outfile.close()

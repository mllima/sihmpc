from opom import OPOM, TransferFunction
from sihmpc import IHMPCController

import time
import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# %% Modelo OPOM

Ts = 1.0

# Transfer functions
num11 = [2.5]
den11 =[62, 1]   
h11 = TransferFunction(num11, den11, delay=2)  # delay in seconds
num12 = [1.5]
den12 = [23*62, 23+62, 1]
h12 = TransferFunction(num12, den12, delay=0)
num21 = [1.4]
den21 = [30*90, 30+90, 1]
h21 = TransferFunction(num21, den21, delay=0)
num22 = [2.8]
den22 = [90, 1]
h22 = TransferFunction(num22, den22)

# General system

h = [[h11, h12], [h21, h22]]
sys = OPOM(h, Ts)

# %% Controlador

N = 10  # horizon in steps
c = IHMPCController(sys, N)

# sub-objectives
Q1 = 1
Q2 = 2
R = np.eye(2)

Vy1, Vy1N, Vi1N = c.subObj(y=[0], Q=Q1)
Vy2, Vy2N, Vi2N = c.subObj(y=[1], Q=Q2)
Vdu = c.subObj(du=[0,1], Q=R)

# limits of the sub-objectives
# Vy1.lim(0,np.inf)

# ihmpc
# mpc = c.MPC()

# %% Closed loop

JPlot = []
duPlot = []
yPlot = []
uPlot = []
xPlot = []
pesosPlot = []

u = np.ones(sys.nu)*0  	# controle anterior
x = np.ones(sys.nx)*0  	# Estado inicial
tEnd = 500     	    # Tempo de simulação (seg)

tocMPC = []

ysp = [1, 0.5]

# pesos - inicialização dos pessos
pesos = np.array([1, 1, 1, 1, 1, 1, 1])

w0 = []
lam_w0 = []
lam_g0 = []
    
for k in np.arange(0, tEnd/Ts):

    t1 = time.time()
    pesosPlot += [pesos]
        
    # to test a change in the set-point    
    if k > (tEnd/2)/Ts: 
        ysp = [1, 0.25]

    sol = c.mpc(x0=x, ySP=ysp, w0=w0, u0=u, pesos=pesos, lam_w0=lam_w0, lam_g0=lam_g0)
    
    t2 = time.time()
    tocMPC += [t2-t1]

    w0 = sol['w_opt'][:]
    lam_w0 = sol['lam_w']
    lam_g0 = sol['lam_g']
    
    du = sol['du_opt'][:, 0].full()
    duPlot += [du]

    J = sol['J'].full()
    JPlot += [J]
    # sobj = c.sobj(x0=x, u0=u, w0=w0, ysp=ysp)
    
    # Vy = sobj['Vy']
    # Vt = sobj['Vt']
    # Vdu = sobj['Vdu']
    # VyN = sobj['VyN']
    # ViN = sobj['ViN']
    # Vyt = Vy + Vt

    # ViNant = ViN.full()[0][0]

    # ## Simula o sistema ###
    res = c.F(x0=x, du0=du, u0=u)
    x = res['xkp1'].full()
    u = res['ukp1'].full()
    y = res['ykp1'].full()
    yPlot += [y]
    uPlot += [u]
    xPlot += [x]
    
    w0 = c.warmStart(sol, ysp)
    
    du_warm = w0
    
    # new_pesos = c.satWeights(x, du_warm, ysp)
    # alfa = 0.9
    # pesos = alfa*pesos + (1-alfa)*new_pesos
    
print('Tempo de execução do MPC. Média: %2.3f s, Max: %2.3f s' %
                                    (np.mean(tocMPC), np.max(tocMPC)))

# %% Plot

t = np.arange(0, tEnd, Ts)

yPlot = np.hstack(yPlot)
duPlot = np.hstack(duPlot)
uPlot = np.hstack(uPlot)
xPlot = np.hstack(xPlot)
pesosPlot = np.array(pesosPlot)

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
plt.show()

#plt.savefig("SisoSIHMPCOutput.png")
#
fig2 = plt.figure(2)
fig2.suptitle("OPOM Variables")
fig2.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(2, 2, 1)
plt.step(t, xPlot[0,:], label='xs')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(2, 2, 2)
plt.step(t, xPlot[1,:], label='xd1')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(2, 2, 3)
plt.step(t, xPlot[2,:], label='xd2')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(2, 2, 4)
plt.step(t, xPlot[3,:], label='xi')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.show()
#plt.savefig("SisoSIHMPCopomVar.png")
#
fig3 = plt.figure(3)
fig3.suptitle("Weights")
fig3.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(2, 2, 1)
plt.step(t, pesosPlot[:,0], label='wy')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(2, 2, 2)
plt.step(t, pesosPlot[:,1], label='wdu')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(2, 2, 3)
plt.step(t, pesosPlot[:,2], label='wyN')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(2, 2, 4)
plt.step(t, pesosPlot[:,3], label='wiN')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.show()
#plt.savefig("SisoSIHMPWeigthVar.png")


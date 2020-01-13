# -*- coding: utf-8 -*-
# MLima 14/04/2018
# Single-shooting
# Benetti
# MLima 03/02/2019
# Benetti
# MLima 14/02/2019


from casadi import *
import time
import numpy as np
import matplotlib.pyplot as plt
from opom import OPOM, TransferFunctionDelay
import scipy as sp
#from numpy import matlib
from scipy import signal

# %% Modelo OPOM

Ts = 0.1

#h = signal.TransferFunction([1], [1, 3, 2])
#sys = OPOM([h], Ts)

h11 = TransferFunctionDelay([1],[1, 1])
h12 = TransferFunctionDelay([1],[1, 2])
h21 = TransferFunctionDelay([1],[1, 3])
h22 = TransferFunctionDelay([1],[1, 4])

#h11 = signal.TransferFunction([1],[1, 1])
#h12 = signal.TransferFunction([1],[1, 2])
#h21 = signal.TransferFunction([1],[1, 3])
#h22 = signal.TransferFunction([1],[1, 4])
G = [[h11, h12],[h21, h22]]

#sys = OPOM(G, 0)   #discrete
sys = OPOM(G, Ts)   #Continuous

nx = sys.A.shape[0]     # Número de estados
nu = sys.B.shape[1]     # Número de manipuladas
ny = sys.C.shape[0]     # Número de saídas

nxs = sys.ny
nxd = sys.ny*sys.na*sys.nu
nxi = sys.ny

# %% parâmetros do controlador

xlb = np.ones(nx)*-np.inf     # Lower bound nos estados
xub = np.ones(nx)*np.inf      # Upper bound nos estados
ulb = np.ones(nu)*-np.inf     # Lower bound do controle
uub = np.ones(nu)*np.inf      # Upper bound do control
dulb = np.ones(nu)*-np.inf    # Lower bound no incremento de controle
duub = np.ones(nu)*np.inf     # Upper bound no incremento de controle
sulb = np.ones(nu)*-np.inf    # Lower bound da variável de folga do controle
suub = np.ones(nu)*np.inf     # Upper bound da variável de folga do controle
sylb = np.ones(ny)*-np.inf    # Lower bound da variável de folga da saída
syub = np.ones(ny)*np.inf     # Upper bound da variável de folga da saída
silb = np.ones(ny)*-np.inf    # Lower bound da variável de folga do estado integrador
siub = np.ones(ny)*np.inf     # Upper bound da variável de folga do estado integrador
rslb = np.ones(nxs)*0         # Lower bound na restrição de xs
rsub = np.ones(nxs)*0         # Upper bound na restrição de xs
rilb = np.ones(nxi)*0         # Lower bound na restrição de xi
riub = np.ones(nxi)*0         # Upper bound na restrição de xi

ysp = [-20.0, 20.0]  # Set point da saída

# Controlador #
N = 10  # Horizonte do controlador

# A escolha dos pesos é bastante simplificada: em geral se escolhe peso 1
Qy = 1*np.eye(ny)  		# Matriz de peso das saídas
R = 1*np.eye(nu)  		# Matriz de peso dos controles
Sy = 1*np.eye(ny)  		# Matriz de peso das variáveis de folga dos estados est
Si = 1*np.eye(ny)  	    # Matriz de peso das variáveis de folga dos estados int

# %%%% Definição dos parâmetros do satisficing
deal = 1700 #1256.55 #120.566
gamma1 = deal*N*1**2  	# maximo custo do erro
gamma2 = deal*N*50**2  	# maximo custo de controle
gamma3 = deal*1**2   	# syN
gamma4 = deal*1**2  	# siN
gamma5 = deal*1**2     # cT

# %% Definição da dinâmica

dUk = MX.sym('du', nu)

Xk = MX.sym('x', nx)
Uk = MX.sym('u', nu)
Yk = MX.sym('y', ny)

# Formulate discrete time dynamics
Xkp1 = Xk  # creating the variables
Ykp1 = Yk  # creating the variables
Ukp1 = Uk  # creating the variables

Xkp1 = mtimes(sys.A, Xk) + mtimes(sys.B, dUk)
Ykp1 = mtimes(sys.C, Xkp1)
Ukp1 = Uk + dUk

F = Function('F', [Xk, Uk, dUk], [Xkp1, Ykp1, Ukp1],
                  ['x0', 'u0', 'du0'],
                  ['xkp1', 'ykp1', 'ukp1'])


# %%%% Definição do problema de otimização

w = []     # Variáveis de otimização
w0 = []    # Chute inicial para w
lbw = []   # Lower bound de w
ubw = []   # Upper bound de w

g = []     # Restrições não lineares
lbg = []   # Lower bound de g
ubg = []   # Upper bound de g
#vetor_duk = []  # Vetor de armazenamento das variáveis duk para uso nas restrições
J = 0      # Função objetivo
V1 = 0  # Subobjetivo da referencia
V2 = 0  # Subobjetivo do controle
V3 = 0  # Subobjetivo do atendimento à condição terminal do modo estacionário
V4 = 0  # Subobjetivo do atendimento à condição terminal do modo integral
V5 = 0  # Subobjetivo do custo terminal

# # "Lift" initial conditions
# X0 = MX.sym('X_0', nx)
# U0 = MX.sym('U_0', nu)

# set-point
Ysp = MX.sym('Ysp', ny)

du0 = np.ones(nu)*0

# %%%% horizonte de predição

for k in range(0, N):
    dUk = MX.sym('dU_' + str(k), nu)     # Variável para o controle em k

    # Adiciona duk nas variáveis de decisão
    w += [dUk]  	# variável
    lbw += [dulb]  	# bound inferior na variável
    ubw += [duub]  	# bound superior na variável
    w0 += [du0]  	# Chute inicial da variável

    # Salva o novo elemento duk no vetor para uso nas restrições
    #vetor_duk += [dUk]

    # Adiciona a nova dinâmica
    res = F(x0=Xk, u0=Uk, du0=dUk)
    Xkp1 = res['xkp1']
    Ykp1 = res['ykp1']
    Ukp1 = res['ukp1']

    # Bounds em x
    g += [Xkp1]
    lbg += [xlb]
    ubg += [xub]

    # Bounds em u
    g += [Ukp1]
    lbg += [ulb]
    ubg += [uub]

    # Bounds em y (nenhuma)

    # Função objetivo
    V1 += dot((Ykp1 - Ysp)**2, diag(Qy))
    V2 += dot(dUk**2, diag(R))

# %%%% custo terminal

syN = MX.sym('syN', ny)    # Variáveis de folga na saída
siN = MX.sym('siN', ny)    # Variável de folga terminal

sy0 = np.ones(ny)*0
si0 = np.ones(ny)*0

# Adiciona syT nas variáveis de decisão
w += [syN]  	# variável
lbw += [sylb]  	# bound inferior na variável
ubw += [syub]  	# bound superior na variável
w0 += [sy0]  	# Chute inicial da variável

# Adiciona siT nas variáveis de decisão
w += [siN]  	# variável
lbw += [silb]  	# bound inferior na variável
ubw += [siub]  	# bound superior na variável
w0 += [si0]  	# Chute inicial da variável

# Adiciona o custo destas variáveis

V3 = dot(syN**2, diag(Sy))
V4 = dot(siN**2, diag(Si))

# Adição do custo terminal
# Continuous
Q_lyap = sys.F.T.dot(sys.Psi.T).dot(Qy).dot(sys.Psi).dot(sys.F)
# Q_bar = sp.linalg.solve_discrete_lyapunov(sys.F,Q_lyap)
Q_bar = sp.linalg.solve_discrete_lyapunov(sys.F, Q_lyap, method='bilinear')

XdN = Xkp1[nxs:nxs+nxd]

V5 = dot(XdN**2, diag(Q_bar))


# %%%% custo total

J = -log(gamma1-V1)-log(gamma2-V2)-log(gamma3-V3)-log(gamma4-V4)-log(gamma5-V5)

# %%%% Restrição do ihmpc

XsN = Xkp1[0:nxs]
XiN = Xkp1[nxs+nxd:]

# Restrições terminais

res1 = XiN - siN
res2 = XsN - Ysp - syN

# Adicionando as restrições

# Restrição 1
g += [res1]
lbg += [rilb]
ubg += [riub]

# Restrição 2
g += [res2]
lbg += [rslb]
ubg += [rsub]

# restrições do Satisficing (garantia do log)
g += [V1, V2, V3, V4, V5]
lbg += [0, 0, 0, 0, 0]
ubg += [gamma1-0.001, gamma2-0.001, gamma3-0.001, gamma4-0.001, gamma5-0.001]


g = vertcat(*g)
w = vertcat(*w)
p = vertcat(Xk, Ysp, Uk)     # Parâmetros do NLP

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g, 'p': p}  	# f - função objetivo,
											# x = variaveis de decisão
											# g = restrições
											# p = parâmetros inciais (dados do problema: estado inicial, controle anterior)

# # Alternativa 1
# opt = {'expand': False, 'jit': False,
#        'verbose_init': 0, 'print_time': False}
# ipopt = {'print_level': 0, 'print_timing_statistics': 'no',
#          'warm_start_init_point': 'yes'}
# opt['ipopt'] = ipopt
# MPC = nlpsol('MPC', 'ipopt', prob, opt)

# Alternativa 2
opt = {'expand':False, 'jit':False,
      'verbose_init':0, 'print_time':False}
MPC = nlpsol('MPC', 'sqpmethod', prob, opt)
MPC = nlpsol('MPC', 'sqpmethod', prob, {'max_iter':1})

# Verticalização das listas
w0 = vertcat(*w0)
lbw = vertcat(*lbw)
ubw = vertcat(*ubw)
lbg = vertcat(*lbg)
ubg = vertcat(*ubg)


# %% Simplificação do MPC

W0 = MX.sym('W0', w0.shape[0])
LAM_W0 = MX.sym('LW0', w0.shape[0])  # multiplicadores de lagrange - initial guess
LAM_G0 = MX.sym('LG0', g.shape[0])

sol = MPC(x0=W0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p,
          lam_x0=LAM_W0, lam_g0=LAM_G0)

# loop para retornar o resultado em matriz
du_opt = []
index = 0
for kk in range(0, N):
    auxU = sol['x'][index:(index+nu)]
    du_opt = horzcat(du_opt, auxU)
    index = index + nu


MPC2 = Function('MPC2',
    [W0, Xk, Ysp, Uk, LAM_W0, LAM_G0],
    [sol['f'], du_opt, sol['x'], sol['lam_x'], sol['lam_g'], sol['g']],
    ['w0', 'x0', 'ySP', 'u0', 'lam_w0', 'lam_g0'],
    ['J','du_opt', 'w_opt', 'lam_w', 'lam_g', 'g'])


# %% Closed loop

u = np.ones(nu)*0  	# controle anterior
x = np.ones(nx)*0  	# Estado inicial
tEnd = 500*Ts*1.5     	# Tempo de simulação

# Variáveis para plot
xPlot = [x]
yPlot = []
uPlot = [u]
duPlot = []
JPlot = []
V1Plot = []
V2Plot = []
V3Plot = []
V4Plot = []
V5Plot = []

tOdePlot = [0]
tocMPC = []

# Variáveis para warm-start
lam_w0 = np.zeros(w0.shape[0])
lam_g0 = np.zeros(g.shape[0])


for k in np.arange(0, tEnd/Ts+1):

    # ## Controlador ###
    t1 = time.time()

    sol = MPC2(x0=x, ySP=ysp, w0=w0, u0=u)

    t2 = time.time()
    tocMPC += [t2-t1]

    du = sol['du_opt'][:, 0]
    duPlot += [du]

    J = sol['J']
    V1 = sol['g'][-5]
    V2 = sol['g'][-4]
    V3 = sol['g'][-3]
    V4 = sol['g'][-2]
    V5 = sol['g'][-1]

    JPlot += [J]
    V1Plot += [V1]
    V2Plot += [V2]
    V3Plot += [V3]
    V4Plot += [V4]
    V5Plot += [V5]

    w0 = sol['w_opt'][:]
    lam_w0 = sol['lam_w']
    lam_g0 = sol['lam_g']

    # ## Simula o sistema ###
    if k != tEnd/Ts:
        res = F(x0=x, du0=du, u0=u)
        x = res['xkp1']
        u = res['ukp1']
        y = res['ykp1']

        xPlot += [x]
        yPlot += [y]
        uPlot += [u]
        tOdePlot += [(k+1)*Ts]

print('Tempo de execução do MPC. Média: %2.3f s, Max: %2.3f s' %
                                    (np.mean(tocMPC), np.max(tocMPC)))

# %% Plot

t = np.arange(0, tEnd+Ts, Ts)
yspPlot = np.matlib.repmat(np.array(ysp), len(t), 1)
yspPlot = np.tile(np.array(ysp), (len(t), 1))

xPlot = horzcat(*xPlot)
xPlot = xPlot.full()

duPlot = horzcat(*duPlot)
duPlot = duPlot.full()

uPlot = horzcat(*uPlot)
uPlot = uPlot.full()

yPlot = horzcat(*yPlot)
yPlot = yPlot.full()

#fig1 = plt.figure(1)
#fig1.suptitle("Output and Control Signals")
#fig1.text(0.5, 0.04, 'Time', ha='center', va='center')
#plt.subplot(1, 2, 1)
#plt.step(t[1:], yPlot[0, :], label='y')
#plt.legend(loc=0, fontsize='large')
#plt.grid()
#plt.legend()
#plt.subplot(1, 2, 2)
#plt.step(t, duPlot[0, :], label='du')
#plt.legend(loc=0, fontsize='large')
#plt.grid()
#plt.legend()
#plt.show()
#plt.savefig("SisoSIHMPCOutput.png")
#
#fig2 = plt.figure(2)
#fig2.suptitle("OPOM Variables")
#fig2.text(0.5, 0.04, 'Time', ha='center', va='center')
#plt.subplot(2, 2, 1)
#plt.step(t, xPlot[0, :], label='xs')
#plt.legend(loc=0, fontsize='large')
#plt.grid()
#plt.legend()
#plt.subplot(2, 2, 2)
#plt.step(t, xPlot[1, :], label='xd1')
#plt.legend(loc=0, fontsize='large')
#plt.grid()
#plt.legend()
#plt.subplot(2, 2, 3)
#plt.step(t, xPlot[2, :], label='xd2')
#plt.legend(loc=0, fontsize='large')
#plt.grid()
#plt.legend()
#plt.subplot(2, 2, 4)
#plt.step(t, xPlot[3, :], label='xi')
#plt.legend(loc=0, fontsize='large')
#plt.grid()
#plt.legend()
#plt.show()
#plt.savefig("SisoSIHMPCopomVar.png")
#
#plt.figure(3)
#plt.subplot(3, 2, 1)
#plt.plot(JPlot)
#plt.subplot(3, 2, 2)
#plt.plot(V1Plot)
#plt.subplot(3, 2, 3)
#plt.plot(V2Plot)
#plt.subplot(3, 2, 4)
#plt.plot(V3Plot)
#plt.subplot(3, 2, 5)
#plt.plot(V4Plot)
#plt.subplot(3, 2, 6)
#plt.plot(V5Plot)
#plt.show()

#Mimo
fig1 = plt.figure(1)
fig1.suptitle("Output and Control Signals")
fig1.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(1,2,1)
plt.step(t[1:], yPlot[0,:], label='y1')
plt.step(t[1:], yPlot[1,:], label='y2')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()    
plt.subplot(1,2,2)
plt.step(t, duPlot[0, :], label='du1')
plt.step(t, duPlot[1, :], label='du1')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()    
plt.show()
plt.savefig("SIHMPCOutput.png")

fig2 = plt.figure(2)
fig2.suptitle("OPOM Xs Variables")
fig2.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(1,2,1)
plt.step(t, xPlot[0, :], label='Xs1')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()  
plt.subplot(1,2,2)
plt.step(t, xPlot[1, :], label='Xs2')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.show()
plt.savefig("SIHMPCXs.png")

fig3 = plt.figure(3)
fig3.suptitle("OPOM Xd Variables")
fig3.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(1,4,1)
plt.step(t, xPlot[2, :], label='xd1')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()  
plt.subplot(1,4,2)
plt.step(t, xPlot[3, :], label='xd2')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(1,4,3)
plt.step(t, xPlot[3, :], label='xd3')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.subplot(1,4,4)
plt.step(t, xPlot[4, :], label='xd4')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.show()
plt.savefig("SIHMPCXd.png")

fig4 = plt.figure(4)
fig4.suptitle("OPOM Xi Variables")
fig4.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.subplot(1,2,1)
plt.step(t, xPlot[5, :], label='Xi1')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()  
plt.subplot(1,2,2)
plt.step(t, xPlot[6, :], label='Xi2')
plt.legend(loc=0, fontsize='large')
plt.grid()
plt.legend()
plt.show()
plt.savefig("SIHMPCXi.png")

























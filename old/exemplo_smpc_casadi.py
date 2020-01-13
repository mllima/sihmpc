# -*- coding: utf-8 -*-
# MLima 14/04/2018
# Single-shooting

from casadi import *
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from opom import OPOM
from scipy import signal

# %% Modelo OPOM

Ts = 0.1
# h = signal.TransferFunction([1], [1, 3])
h = signal.TransferFunction([1], [1, 3, 2])
sys = OPOM([h], Ts)

nx = sys.A.shape[0]     # Número de estados
nu = sys.B.shape[1]     # Número de manipuladas
ny = sys.C.shape[0]
print(nx, nu, ny)
# %% parãmetros do controlador

xlb = [-np.inf, -np.inf, -np.inf, -np.inf]  # Lower bound nos estados
xub = [np.inf, np.inf, np.inf, np.inf]     # Upper bound nos estados
ulb = [-np.inf]                    # Lower bound do controle
uub = [np.inf]                     # Upper bound do controle
dulb = [-np.inf]                   # Lower bound no incremento de controle
duub = [np.inf]                    # Upper bound no incremento de controle


ysp = [1.8]
# usp = [0]
# xsp = [16.4664, 10.3378, 11.9180, 1.0000]

# Controlador #
N = 50  # Horizonte do controlador
# Q = 1*np.eye(ny)
# R = 0.1*np.eye(nu)

# %% Definição da dinâmica

dU = MX.sym('du', nu)

X = MX.sym('x', nx)
U = MX.sym('u', nu)
Y = MX.sym('y', ny)

# Formulate discrete time dynamics
Xf = X
Yf = Y
Uf = U

Xf = mtimes(sys.A, X) + mtimes(sys.B, dU)
# Yf = mtimes(sys.C,Xf)
Yf = mtimes(sys.C, Xf)
Uf = U + dU

F = Function('F', [X, U, dU], [Xf, Yf, Uf], ['x0', 'u0', 'du0'], ['xf', 'yf', 'uf'])


# %% Definição do problema de otimização

w = []     # Variáveis de otimização
w0 = []    # Chute inicial para w
lbw = []   # Lower bound de w
ubw = []   # Upper bound de w
J = 0      # Função objetivo
J1 = 0
J2 = 0
g = []     # Restrições não lineares
lbg = []   # Lower bound de g
ubg = []   # Upper bound de g

# "Lift" initial conditions
X0 = MX.sym('X_0', nx)
U0 = MX.sym('U_0', nu)

# set-point
Ysp = MX.sym('Ysp', ny)


Xk = X0
Uk = U0
du0 = [0]

# %%% montagem das variáveis no horizonte de predição
for k in range(0, N):
    dUk = MX.sym('dU_' + str(k), nu)     # Variável para o controle em k

    # Bounds na ação de controle
    w += [dUk]
    lbw += [dulb]
    ubw += [duub]
    w0 += [du0]  # Chute inicial
    # w0 = [w0 uub/2]       #Chute inicial
    # w0 = [w0 [2 2]']      #Chute inicial


    # Integra até k + 1
    res = F(x0=Xk, u0=Uk, du0=dUk)
    Xk = res['xf']
    Yk = res['yf']
    Uk = res['uf']

    # Bounds em x
    g += [Xk]
    lbg += [xlb]
    ubg += [xub]

    # Bounds em u
    g += [Uk]
    lbg += [ulb]
    ubg += [uub]

    # Função objetivo
    #    J = J + (Xk - Xsp)*diag(Q)*(Xk - Xsp)
    #    J = J + (Uk - Usp)*diag(R)*(Uk - Usp)

    #    J = J + dot((Yk - Ysp)**2, diag(Q))
    #    J = J + dot(dUk**2, diag(R))

    J1 = J1 + (Yk - Ysp)**2
    J2 = J2 + dUk**2

w0 = [ 7.36470770e-01,  5.95228480e-01,  4.69247975e-01,  3.58378208e-01,
    2.62218383e-01,  1.80140216e-01,  1.11322394e-01,  5.47913656e-02,
    9.46412446e-03, -2.58101031e-02, -5.22113752e-02, -7.09150820e-02,
   -8.30632461e-02, -8.97413427e-02, -9.19605653e-02, -9.06452241e-02,
   -8.66248052e-02, -8.06301194e-02, -7.32929252e-02, -6.51483998e-02,
   -5.66398531e-02, -4.81251176e-02, -3.98841031e-02, -3.21270641e-02,
   -2.50031931e-02, -1.86092197e-02, -1.29977563e-02, -8.18519233e-03,
   -4.15899050e-03, -8.84287607e-04,  1.69025652e-03,  3.62738381e-03,
    4.99692944e-03,  5.87224974e-03,  6.32733895e-03,  6.43457372e-03,
    6.26301549e-03,  5.87719520e-03,  5.33630162e-03,  4.69369423e-03,
    3.99666198e-03,  3.28635157e-03,  2.59779112e-03,  1.95993792e-03,
    1.39568148e-03,  9.21734668e-04,  5.48346950e-04,  2.78772844e-04,
    1.08426449e-04,  2.36479434e-05]

# %%%%% Definição do problema
    
S1 = 5.0  # 10*N  # 1.445*N
S2 = 1.5  # 10*N

J = -S1*log(S1-J1) - S2*log(S2-J2)

g += [J1, J2]
lbg += [0, 0]
ubg += [S1+0.1, S2+0.1]


g = vertcat(*g)
w = vertcat(*w)
p = vertcat(X0, Ysp, U0)     # Parâmetros do NLP

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g, 'p': p}

# Alternativa 1
#opt = {'expand': False, 'jit': False,
#       'verbose_init': 0, 'print_time': False}
#ipopt = {'print_level': 0, 'print_timing_statistics': 'no',
#         'warm_start_init_point': 'yes'}
#opt['ipopt'] = ipopt
#
#MPC = nlpsol('MPC', 'ipopt', prob, opt)

# Alternativa 2
opt = {'expand':False, 'jit':False,
   'verbose_init':0, 'print_time':False}
MPC = nlpsol('MPC', 'sqpmethod', prob, opt)
# MPC = nlpsol('MPC', 'sqpmethod', prob, {'max_iter':1})

# Verticalização das listas
w0 = vertcat(*w0)
lbw = vertcat(*lbw)
ubw = vertcat(*ubw)
lbg = vertcat(*lbg)
ubg = vertcat(*ubg)

# %%%%%% Simplificação do MPC

W0 = MX.sym('W0', w0.shape[0])
LAM_W0 = MX.sym('W0', w0.shape[0])  # multiplicadores de lagrange 
LAM_G0 = MX.sym('W0', g.shape[0])

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
    [W0, X0, Ysp, U0, LAM_W0, LAM_G0],
    [sol['f'], du_opt, sol['x'], sol['lam_x'], sol['lam_g'], sol['g']],
    ['w0', 'x0', 'ySP', 'u0', 'lam_w0', 'lam_g0'],
    ['J', 'du_opt', 'w_opt', 'lam_w', 'lam_g', 'g'])

# Função pra espiar a esparsidade da matriz
# Jg = Function('Jg', [w, X0], [jacobian(g,w)], ['w','x0'], ['jac'])
# plt.spy(Jg(w0,[5, 15, 5, 15]).full())


# %%%%%%%%%%%% Closed loop

u = [0]
x = [1, 0, 0, 0]  # [0, 0, 0, 0]    # Estado inicial
tEnd = 250*Ts       # Tempo de simulação

# Variáveis para plot
xPlot = [x]
yPlot = []
uPlot = [u]
duPlot = []
JPlot = []
J1Plot = []
J2Plot = []

tOdePlot = [0]
tocMPC = []

# Variáveis para warm-start
lam_w0 = np.zeros(w0.shape[0])
lam_g0 = np.zeros(g.shape[0])


for k in np.arange(0, tEnd/Ts+1):

    # ### Controlador ###
    t1 = time.time()
    # Se eu não declarar 'w0', ele assume como zero
#    sol = MPC2(x0=x, xSP=xsp, uSP=usp)                   #Sem warm-start
    sol = MPC2(x0=x, ySP=ysp, w0=w0, u0=u)
#    sol = MPC2(x0=x, xSP=xsp, uSP=usp, w0=w0, lam_w0=lam_w0, lam_g0=lam_g0)

    t2 = time.time()
    tocMPC += [t2-t1]

    du = sol['du_opt'][:, 0]
    duPlot += [du]

    J = sol['J']
    J1 = sol['g'][-2]
    J2 = sol['g'][-1]

    JPlot += [J]
    J1Plot += [J1]
    J2Plot += [J2]

    w0 = sol['w_opt'][1:].full()
    w0 = np.append(w0,[0])
    lam_w0 = sol['lam_w']
    lam_g0 = sol['lam_g']
    
    # ### Simula o sistema ###
    if k != tEnd/Ts:
        res = F(x0=x, du0=du, u0=u)
        x = res['xf']
        u = res['uf']
        y = res['yf']

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

plt.figure(1)
plt.subplot(1, 2, 1)
plt.step(t, duPlot[0, :])
plt.subplot(1, 2, 2)
plt.step(t[1:], yPlot[0, :])
plt.grid()
plt.show()

plt.figure(2)
plt.subplot(1, 3, 1)
plt.step(t, xPlot[0, :], label='xs')
plt.grid()
plt.subplot(1, 3, 2)
plt.step(t, xPlot[1, :], label='xd')
plt.subplot(1, 3, 3)
plt.step(t, xPlot[2, :], label='xi')
plt.show()


plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(JPlot)
plt.subplot(3, 1, 2)
plt.plot(J1Plot)
plt.subplot(3, 1, 3)
plt.plot(J2Plot)
plt.show()
























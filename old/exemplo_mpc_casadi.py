# -*- coding: utf-8 -*-
#MLima 14/04/2018
#Single-shooting
from casadi import * 
import time
import numpy as np
import matplotlib.pyplot as plt
from opom import OPOM
from scipy import signal

# %% Modelo OPOM

Ts = 0.1
h = signal.TransferFunction([1],[1, 3])
sys = OPOM([h],Ts)

nx = sys.A.shape[0]     # Número de estados
nu = sys.B.shape[1]     # Número de manipuladas
ny = sys.C.shape[0]

# %% parãmetros do controlador

xlb = [-np.inf, -np.inf, -np.inf]  #Lower bound nos estados
xub = [np.inf, np.inf, np.inf]     #Upper bound nos estados
ulb = [-np.inf]                    #Lower bound do controle 
uub = [np.inf]                     #Upper bound do control
dulb = [-0.1]                      #Lower bound no incremento de controle 
duub = [0.1]                       #Upper bound no incremento de controle



ysp = [1]
#usp = [0]
#xsp = [16.4664, 10.3378, 11.9180, 1.0000]       

# Controlador #
N = 10 # Horizonte do controlador
Q = 1*np.eye(ny)
R = 0.1*np.eye(nu)

# %% Definição da dinâmica

dU = MX.sym('du', nu)

X = MX.sym('x', nx)
U = MX.sym('u', nu)
Y = MX.sym('y', ny)

# Formulate discrete time dynamics
Xf = X
Yf = Y
Uf = U

Xf = mtimes(sys.A,X) + mtimes(sys.B,dU) 
# Yf = mtimes(sys.C,Xf)
Yf = mtimes(sys.C,Xf)
Uf = Uf + dU
        
F = Function('F', [X, U, dU], [Xf, Yf, Uf], ['x0','u0', 'du0'], ['xf','yf','uf'])


# %% Definição do problema de otimização

w = []     #Variáveis de otimização
w0 = []    #Chute inicial para w
lbw = []   #Lower bound de w
ubw = []   #Upper bound de w
J = 0      #Função objetivo
g=[]       #Restrições não lineares
lbg = []   #Lower bound de g
ubg = []   #Upper bound de g

# "Lift" initial conditions
X0 = MX.sym('X_0', nx)
U0 = MX.sym('U_0', nu)

# set-point
Ysp = MX.sym('Ysp', ny)


Xk = X0
Uk = U0
du0 = [0]
for k in range(0,N):
    dUk = MX.sym('dU_' + str(k), nu)     #Variável para o controle em k
    
    # Bounds em na ação de controle
    w += [dUk]
    lbw += [dulb]
    ubw += [duub]
    w0 += [du0]  #Chute inicial
#     w0 = [w0 uub/2]       #Chute inicial
#     w0 = [w0 [2 2]']      #Chute inicial
    
    # Integra até k + 1
    res = F(x0=Xk, u0 = Uk, du0=dUk)                         
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
    J = J + dot((Yk - Ysp)**2, diag(Q))
    J = J + dot(dUk**2, diag(R))

    


g = vertcat(*g)
w = vertcat(*w)
p = vertcat(X0, Ysp, U0)     #Parâmetros do NLP

# Create an NLP solver
prob = {'f':J, 'x':w, 'g':g, 'p':p}

# Alternativa 1
opt = {'expand':False, 'jit':False,
       'verbose_init':0, 'print_time':False}
ipopt = {'print_level':0, 'print_timing_statistics':'no',
         'warm_start_init_point':'yes'}
opt['ipopt'] = ipopt
MPC = nlpsol('MPC', 'ipopt', prob, opt)

# Alternativa 2
#opt = {'expand':False, 'jit':False,
#       'verbose_init':0, 'print_time':False}
#MPC = nlpsol('MPC', 'sqpmethod', prob, opt)
#MPC = nlpsol('MPC', 'sqpmethod', prob, {'max_iter':1})

#Verticalização das listas
w0 = vertcat(*w0)
lbw = vertcat(*lbw)
ubw = vertcat(*ubw)
lbg = vertcat(*lbg)
ubg = vertcat(*ubg)

# %% Simplificação do MPC

W0 = MX.sym('W0', w0.shape[0])
LAM_W0 = MX.sym('W0', w0.shape[0]) # multiplicadores de lagrange - initial guess
LAM_G0 = MX.sym('W0', g.shape[0])

sol = MPC(x0=W0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p,
          lam_x0=LAM_W0, lam_g0=LAM_G0)

# loop para retornar o resultado em matriz
du_opt = []
index = 0
for kk in range(0,N):
    auxU = sol['x'][index:(index+nu)]
    du_opt = horzcat(du_opt,auxU)
    index = index + nu


MPC2 = Function('MPC2',
    [W0, X0, Ysp, U0, LAM_W0, LAM_G0],
    [sol['f'], du_opt, sol['x'], sol['lam_x'], sol['lam_g']],
    ['w0', 'x0', 'ySP', 'u0', 'lam_w0', 'lam_g0'],
    ['J','du_opt', 'w_opt', 'lam_w', 'lam_g'])

# Função pra espiar a esparsidade da matriz
#Jg = Function('Jg', [w, X0], [jacobian(g,w)], ['w','x0'], ['jac'])
#plt.spy(Jg(w0,[5, 15, 5, 15]).full())

# %% Closed loop

u=[0]
x = [0, 0, 0]     #Estado inicial    
tEnd = 50*Ts     #Tempo de simulação

#Variáveis para plot
xPlot = [x]
yPlot = []
uPlot = [u]
duPlot = []

tOdePlot = [0]
tocMPC = []

# Variáveis para warm-start
lam_w0 = np.zeros(w0.shape[0])
lam_g0 = np.zeros(g.shape[0])


for k in np.arange(0,tEnd/Ts+1):
    
    ### Controlador ###
    t1 = time.time()
    # Se eu não declarar 'w0', ele assume como zero
#    sol = MPC2(x0=x, xSP=xsp, uSP=usp)                   #Sem warm-start
    sol = MPC2(x0=x, ySP=ysp, w0=w0, u0=u)
#    sol = MPC2(x0=x, xSP=xsp, uSP=usp, w0=w0, lam_w0=lam_w0, lam_g0=lam_g0)
    
    t2 = time.time()
    tocMPC += [t2-t1]
    
    du = sol['du_opt'][:,0]
    duPlot += [du]
    
    w0 = sol['w_opt'][:]
    lam_w0 = sol['lam_w']
    lam_g0 = sol['lam_g']
    
    ### Simula o sistema ###
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

t = np.arange(0,tEnd+Ts,Ts)
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
plt.subplot(1,2,1)
plt.step(t,duPlot[0,:])
plt.subplot(1,2,2)
plt.step(t[1:],yPlot[0,:])
plt.grid()
plt.show()

plt.figure(2)
plt.subplot(1,3,1)
plt.step(t,xPlot[0,:], label='xs')
plt.grid()
plt.subplot(1,3,2)
plt.step(t,xPlot[1,:], label='xd')
plt.subplot(1,3,3)
plt.step(t,xPlot[2,:], label='xi')
plt.show()



























# -*- coding: utf-8 -*-
"""
@author: Marcelo Lima
"""
import casadi as csd
import numpy as np
from opom import OPOM
import scipy as sp
from scipy.linalg import solve_discrete_lyapunov

class IHMPCController(object):
    def __init__(self, sys, N, **kwargs):

        # assert that sys is of type OPOM
        assert issubclass(type(sys), OPOM), 'the system (sys) must be of type OPOM'
        self.sys = sys
        self.Ts = sys.Ts
        self.N = int(N)  # horizonte de controle

        # assert that the control horizon is greater than dead time
        assert (self.N > sys.theta_max),\
            'the control horizon should be greater than the maximal system dead time: {}'\
            .format(sys.theta_max)

        # numbers
        self.nx = sys.nx     # Número de estados
        self.nu = sys.nu     # Número de manipuladas
        self.ny = sys.ny     # Número de saídas

        self.nxs = sys.ny    # Número de estados estacionários
        self.nxd = sys.nd    # Número de estados dinâmicos
        self.nxi = sys.ny    # Número de estados integradores
        self.nxz = sys.nz    # Número de estados devido ao tempo morto
        assert(self.nx == self.nxs+self.nxd+self.nxi+self.nxz)

        #bounds        
        self.xlb = kwargs.get('xlb', np.ones(sys.nx)*-np.inf)      # Lower bound nos estados
        self.xub = kwargs.get('xub', np.ones(sys.nx)*np.inf)       # Upper bound nos estados
        self.ulb = kwargs.get('ulb', np.ones(sys.nu)*-np.inf)      # Lower bound do controle
        self.uub = kwargs.get('uub', np.ones(sys.nu)*np.inf)       # Upper bound do controle
        self.dulb = kwargs.get('dulb', np.ones(sys.nu)*-np.inf)     # Lower bound no incremento de controle
        self.duub = kwargs.get('duub', np.ones(sys.nu)*np.inf)      # Upper bound no incremento de controle
        self.sulb = kwargs.get('sulb', np.ones(sys.nu)*-np.inf)     # Lower bound da variável de folga do controle
        self.suub = kwargs.get('suub', np.ones(sys.nu)*np.inf)      # Upper bound da variável de folga do controle
        self.sylb = kwargs.get('sylb', np.ones(sys.ny)*-np.inf)     # Lower bound da variável de folga da saída
        self.syub = kwargs.get('syub', np.ones(sys.ny)*np.inf)      # Upper bound da variável de folga da saída
        self.silb = kwargs.get('silb', np.ones(sys.ny)*-np.inf)     # Lower bound da variável de folga do estado integrador
        self.siub = kwargs.get('siub', np.ones(sys.ny)*np.inf)      # Upper bound da variável de folga do estado integrador
        self.rslb = kwargs.get('rslb', np.zeros(sys.ny))            # Lower bound na restrição terminal de xs
        self.rsub = kwargs.get('rsub', np.zeros(sys.ny))            # Upper bound na restrição terminal de xs
        self.rilb = kwargs.get('rilb', np.zeros(sys.ny))            # Lower bound na restrição terminal de xi
        self.riub = kwargs.get('riub', np.zeros(sys.ny))             # Upper bound na restrição terminal de xi
        
        # # A escolha dos pesos é bastante simplificada: em geral se escolhe peso 1
        self.Qy = kwargs.get('Q', np.eye(sys.ny))       # Matriz de peso dos estados
        # self.R = kwargs.get('R', np.eye(sys.nu))        # Matriz de peso dos controles
        # self.Sy = kwargs.get('Sy', np.eye(sys.ny))      # Matriz de peso das variáveis de folga dos estados est
        # self.Si = kwargs.get('Si', np.eye(sys.ny))       # Matriz de peso das variáveis de folga dos estados int

        self.Q_bar = []
       
        # symbolic variables
        self.X = csd.MX.sym('x', self.nx)
        self.U = csd.MX.sym('u', self.nu)
        self.Y = csd.MX.sym('y', self.ny)
        self.Ysp = csd.MX.sym('Ysp', self.ny)    # set-point
        self.syN = csd.MX.sym('syN', self.ny)    # Variáveis de folga na saída
        self.siN = csd.MX.sym('siN', self.ny)    # Variável de folga terminal
                
        self.F = self._DynamicF()
        self.X_pred, self.Y_pred, self.U_pred, self.dU_pred = self.prediction()
        
        # Terminal cost: standard sub-objectives
        self.Vt = self.fObj(self._terminalObj(),1, 
                            self.X, self.U, 
                            np.append(self.dU_pred,[self.syN, self.siN]), 
                            self.Ysp)

        # lists
        self.V = [self.Vt]                       # list of sub-objetives
        self.F_ViN = []                          # list of ViN's functions
        self.Pesos = []
        self.ViN_ant = []
        self.ViNant = []

        # total cost
        self.J = 0


    def _DynamicF(self):
        #return the casadi function that represent the dynamic system
        
        sys = self.sys
        X = csd.MX.sym('x', self.nx)
        U = csd.MX.sym('u', self.nu)
        dU = csd.MX.sym('du', self.nu)

        A = sys.A
        B = sys.B
        C = sys.C
        D = sys.D
        
        Xkp1 = csd.mtimes(A, X) + csd.mtimes(B, dU)
        Ykp1 = csd.mtimes(C, Xkp1) + csd.mtimes(D, dU)
        Ukp1 = U + dU

        F = csd.Function('F', [X, U, dU], [Xkp1, Ykp1, Ukp1],
                     ['x0', 'u0', 'du0'],
                     ['xkp1', 'ykp1', 'ukp1'])
        return F


    class fObj:
        def __init__(self,V,w, X, U, var, Ysp):
            self.V = V
            self.min = 0
            self.max = np.inf
            self.weight = w
            var = csd.vertcat(*var)     # var: decision variables
            self.F = csd.Function('F', [X, U, var, Ysp], [V],
                    ['x0', 'u0', 'var','ysp'],
                    ['Value'])

        def lim(self, min, max):
            self.min = min
            self.max = max


    def subObj(self,**kwargs):
        N = self.N
        Ts = self.Ts
        X = self.X
        U = self.U
        Ysp = self.Ysp
        syN = self.syN
        siN = self.siN
        Y_pred = self.Y_pred
        dU_pred = self.dU_pred

        if 'Q' in kwargs:
            Q = kwargs['Q']
            if isinstance(Q, int): Q = [Q]
        else:
            Q = np.eye(self.sys.ny)

        if 'y' in kwargs:
            Vy = 0
            inds = kwargs['y']
            j = -1
            for ind in inds:
                j += 1
                for k in range(0, N):
                    # sub-objetivo em y
                    Vy += (Y_pred[k][ind] - Ysp[ind] - syN[ind] - (k-N)*Ts*siN[ind])**2 *np.diag(Q)[j]
            l = len(self.V)
            
            weight = csd.MX.sym('w_' + str(l))  # peso do sub_objetivo
            Vy = self.fObj(Vy, weight, X, U, np.append(dU_pred,[syN, siN]), Ysp)
            self.Pesos.append(weight)
            self.J += weight * (Vy.V  + self.Vt.V)
            self.V.append(Vy)

            # associated sub-objectives
            VyN = self.subObj(syN=kwargs['y'], Q=kwargs['Q'])
            ViN = self.subObj(siN=kwargs['y'], Q=kwargs['Q'])
            self.F_ViN.append(ViN.F)

            # ViN must be contractive
            ViN_ant = csd.MX.sym('ViN_ant_' + str(l+2))
            ViN.lim(0,ViN_ant)
            self.ViN_ant.append(ViN_ant)
            self.ViNant.append(np.inf)
            return Vy, VyN, ViN

        if 'du' in kwargs:
            Vdu = 0
            inds = kwargs['du']
            j = -1
            for ind in inds:
                j += 1
                for k in range(0, N):
                    # sub-objetivo em du
                    Vdu += dU_pred[k][ind]**2 * np.diag(Q)[j]
            l = len(self.V)
            weight = csd.MX.sym('w_' + str(l))  # peso do sub_objetivo
            Vdu = self.fObj(Vdu, weight, X, U, np.append(dU_pred,[syN, siN]), Ysp)
            self.Pesos.append(weight)
            self.J += weight * Vdu.V
            self.V.append(Vdu)
            return Vdu
            
        # custo das variáveis de folga
        if 'syN' in kwargs:
            VyN = 0
            inds = kwargs['syN']
            j = -1
            for ind in inds:
                j += 1
                VyN += syN[ind]**2 * np.diag(Q)[j]
            l = len(self.V)
            weight = csd.MX.sym('w_' + str(l))  # peso do sub_objetivo
            VyN = self.fObj(VyN, weight, X, U, np.append(dU_pred,[syN, siN]), Ysp)
            self.Pesos.append(weight)
            self.J += weight * VyN.V
            self.V.append(VyN)
            return VyN

        if 'siN' in kwargs:
            ViN = 0
            inds = kwargs['siN']
            j = -1
            for ind in inds:
                j += 1
                ViN += siN[ind]**2 * np.diag(Q)[j]            
            l = len(self.V)
            weight = csd.MX.sym('w_' + str(l))  # peso do sub_objetivo
            ViN = self.fObj(ViN, weight, X, U, np.append(dU_pred,[syN, siN]), Ysp)
            self.Pesos.append(weight)
            self.J += weight * ViN.V
            self.V.append(ViN)
            return ViN
    

    def _terminalObj(self):
        
        # estado terminal
        XN = self.X_pred[-1]  # terminal state
        XdN = XN[self.nxs:self.nxs+self.nxd]
        Vt = 0
 
        # Adição do custo terminal
        # Q terminal
        Q_lyap = self.sys.F.T@self.sys.Psi.T@self.Qy@self.sys.Psi@self.sys.F
        Q_bar = solve_discrete_lyapunov(self.sys.F, Q_lyap, method='bilinear')
        Vt = csd.dot(XdN**2, csd.diag(Q_bar))
        self.Q_bar = Q_bar
        return Vt
                    
        
    def prediction(self):
        N = self.N
        F = self.F
        X = self.X
        U = self.U        
        Xkp1 = X
        Ukp1 = U        
        X_pred = []
        Y_pred = []
        U_pred = []
        dU_pred = []

        for k in range(0, N):
            dU_k = csd.MX.sym('dU_' + str(k), self.nu)  # Variável para o controle em k
            #self.dUk.append(dU_k)
        
            # Adiciona a nova dinâmica
            res = F(x0=Xkp1, u0=Ukp1, du0=dU_k)
            Xkp1 = res['xkp1']
            Ykp1 = res['ykp1']
            Ukp1 = res['ukp1']
            
            X_pred += [Xkp1]
            U_pred += [Ukp1]
            Y_pred += [Ykp1]
            dU_pred += [dU_k]

        return X_pred, Y_pred, U_pred, dU_pred


    def _OptimProbl(self):
        # the symbolic optimization problem

        w = []     # Variáveis de otimização
        lbw = []   # Lower bound de w
        ubw = []   # Upper bound de w
        g = []     # Restrições não lineares
        lbg = []   # Lower bound de g
        ubg = []   # Upper bound de g
       
        N = self.N  
        
        Ysp = self.Ysp    # set-point
        syN = self.syN    # Variáveis de folga na saída
        siN = self.siN    # Variável de folga terminal
        X = self.X
        U = self.U              
        X_pred = self.X_pred
        Y_pred = self.Y_pred
        U_pred = self.U_pred
        dU_pred = self.dU_pred
        J = self.J
        Pesos = self.Pesos
        ViN_ant = self.ViN_ant
        
        w += dU_pred
        g += X_pred
        g += U_pred

        for _ in range(0, N):
            
            # Adiciona duk nas variáveis de decisão
            lbw += [self.dulb]  	# bound inferior na variável
            ubw += [self.duub]  	# bound superior na variável
                       
            # Bounds em x
            lbg += [self.xlb]
            ubg += [self.xub]
        
            # Bounds em u
            lbg += [self.ulb]
            ubg += [self.uub]
        
            # Bounds em y (nenhuma)
        
        # estado terminal
        XN = self.X_pred[-1]  # terminal state
        XsN = XN[0:self.nxs]
        XiN = XN[self.nxs+self.nxd:self.nxs+self.nxd+self.nxi]
        
        # Restrições terminais do ihmpc
        res1 = XiN - siN
        res2 = XsN - Ysp - syN
        
        # Restrição 1
        g += [res1]
        lbg += [self.rilb]
        ubg += [self.riub]
        
        # Restrição 2
        g += [res2]
        lbg += [self.rslb]
        ubg += [self.rsub]
        
        # restrições nos sub-objetivos
        l = len(self.V)
        for i in range(l-1):
            g += [self.V[i+1].V]
            lbg += [self.V[i+1].min]
            ubg += [self.V[i+1].max]
       
        # variáveis de folga
        w += [syN]
        w += [siN]

        lbw += [self.sylb]  	   # bound inferior na variável
        ubw += [self.syub]  	   # bound superior na variável

        lbw += [self.silb]  	# bound inferior na variável
        ubw += [self.siub]  	# bound superior na variável

        g = csd.vertcat(*g)
        w = csd.vertcat(*w)
        p = csd.vertcat(X, Ysp, U, *Pesos, *ViN_ant)
        lbw = csd.vertcat(*lbw)
        ubw = csd.vertcat(*ubw)
        lbg = csd.vertcat(*lbg)
        ubg = csd.vertcat(*ubg)
        
        X_pred = csd.vertcat(*X_pred)
        Y_pred = csd.vertcat(*Y_pred)
        U_pred = csd.vertcat(*U_pred)
        pred = csd.Function('pred', [X, U, csd.vertcat(*dU_pred)], 
                            [X_pred, Y_pred, U_pred],
                            ['x0', 'u0', 'du_opt'],
                            ['x_pred', 'y_pred', 'u_pred'])

        prob = {'f': J, 'x': w, 'g': g, 'p': p} 
        bounds = {'lbw': lbw, 'ubw': ubw, 'lbg': lbg, 'ubg': ubg}

        return prob, bounds, pred
       
    
    def _MPC(self):
        
        opt = {'expand': False, 'jit': False,
                'verbose_init': 0, 'print_time': False}
        ipopt = {'print_level': 0, 'print_timing_statistics': 'no',
                  'warm_start_init_point': 'yes'}
        opt['ipopt'] = ipopt
        
        # optimization problem
        prob, bounds, pred = self._OptimProbl()
        p = prob['p']
        lbw = bounds['lbw']
        ubw = bounds['ubw']
        lbg = bounds['lbg']
        ubg = bounds['ubg']
        
        dim_w = prob['x'].shape[0]
        dim_g = prob['g'].shape[0]
        
        W0 = csd.MX.sym('W0', dim_w )        
        LAM_W0 = csd.MX.sym('LW0', dim_w)  # multiplicadores de lagrange - initial guess
        LAM_G0 = csd.MX.sym('LG0', dim_g)

        # nlp solver
        solver = csd.nlpsol('solver', 'ipopt', prob, opt)
        sol = solver(x0=W0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p,
                  lam_x0=LAM_W0, lam_g0=LAM_G0)

        # loop para retornar o resultado em matriz
        du_opt = []
        index = 0
        for _ in range(0, self.N):
            auxU = sol['x'][index:(index+self.nu)]
            du_opt = csd.horzcat(du_opt, auxU)
            index = index + self.nu

        # MPC function
        Ysp = self.Ysp
        X = self.X
        U = self.U  
        Pesos = csd.vertcat(*self.Pesos)
        ViN_ant = csd.vertcat(*self.ViN_ant)     
        x_pred, y_pred, u_pred = pred(X,U,sol['x'][:-2*self.ny])   # predicted values
        MPC = csd.Function('MPC',
            [W0, X, Ysp, U, LAM_W0, LAM_G0, Pesos, ViN_ant],
            [sol['f'], du_opt, sol['x'], sol['lam_x'], sol['lam_g'], sol['g'], x_pred, y_pred, u_pred],
            ['w0', 'x0', 'ySP', 'u0', 'lam_w0', 'lam_g0', 'pesos', 'ViN_ant'],
            ['J','du_opt', 'w_opt', 'lam_w', 'lam_g', 'g', 'x_pred', 'y_pred','u_pred'])

        return MPC
    

    def mpc(self, x0, ySP, w0, u0, pesos, lam_w0, lam_g0, ViN_ant=[]):
        MPC = self._MPC()
        if ViN_ant == []:
            ViN_ant = csd.vertcat(*self.ViNant)
        sol = MPC(x0=x0, ySP=ySP, w0=w0, u0=u0, pesos=pesos, lam_w0=lam_w0, lam_g0=lam_g0, ViN_ant=ViN_ant)
        # falta atualizar self.ViNant
        l = len(self.F_ViN)
        for i in range(l):
            self.ViNant[i] = self.F_ViN[i]([], [], sol['w_opt'], [])     
        return sol


    def warmStart(self, sol, ysp):
        w_start = []
        w0 = sol['w_opt'][:]
        
        dustart = w0[0:-2*self.ny].full() # retira syN e siN
        dustart = dustart.reshape((self.nu, self.N))[:,1:] # remove firts du
        dustart = np.hstack((dustart, np.zeros((self.nu,1)))) # add 0 at the end
        dustart = dustart.flatten()
        
        xi = sol['x_pred'][-self.nx:]

        # import pdb
        # pdb.set_trace()

        ui = 0
        res = self.F(x0=xi, u0=ui, du0=0)
        Xknext = res['xkp1'].full()
        
        xsNp2 = Xknext[0:self.nxs]
        xiNp2 = Xknext[self.nxs+self.nxd:self.nxs+self.nxd+self.nxi]   
        
        syNnext = xsNp2 - np.array(ysp).reshape(self.ny,1)
        siNnext = xiNp2
        
        w_start = np.append(dustart, syNnext)
        w_start = np.append(w_start, siNnext)
        return w_start


    # def satWeights(self, x, w_start, ysp):
    #     # custos seguintes estimados
    #     pesos = []
    #     #_, _, sobj, pred = self._OptimProbl()
    #     sobj = self.sobj
        
    #     u = np.zeros(self.nu)   # because u is irrelevant
        
    #     # sub-objectives
    #     Vynext, Vdunext, VyNnext, ViNnext, Vtnext = sobj(x, u, w_start, ysp)
    #     Vytnext = Vynext + Vtnext
        
    #     # py =  1/(self.gamma_e - np.clip(Vytnext, 0, 0.99*self.gamma_e))
    #     # pdu = 1/(self.gamma_du - np.clip(Vdunext, 0, 0.99*self.gamma_du))
    #     # pyN = 1/(self.gamma_syN - np.clip(VyNnext, 0, 0.99*self.gamma_syN))
    #     # piN = 1/(self.gamma_siN - np.clip(ViNnext, 0, 0.99*self.gamma_siN))
        
    #     pesos = np.append(pesos,(py, pdu, pyN, piN))
    #     return pesos


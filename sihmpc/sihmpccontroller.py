# -*- coding: utf-8 -*-
"""
Created on Nov 2017
@author: Marcelo Lima
"""
import casadi as csd
import numpy as np
from opom import OPOM
import scipy as sp


class SIHMPCController(object):
    def __init__(self, sys, N, 
                 gamma_e=10,
                 gamma_du=10,
                 gamma_syN=0.0001,
                 gamma_siN=0.0001,
                 **kwargs
                 ):

        assert issubclass(type(sys), OPOM), 'the system (sys) must be of type OPOM'
        self.sys = sys
        self.Ts = sys.Ts
        self.N = int(N)  # int(np.round(m/sys.Ts))  # horizonte de controle

        assert (self.N > sys.theta_max),\
            'the control horizon should be greater than the maximal system dead time: {}'\
            .format(sys.theta_max)

        self.nx = sys.nx     # Número de estados
        self.nu = sys.nu     # Número de manipuladas
        self.ny = sys.ny     # Número de saídas

        self.nxs = sys.ny    # Número de estados estacionários
        self.nxd = sys.nd    # Número de estados dinâmicos
        self.nxi = sys.ny    # Número de estados integradores
        self.nxz = sys.nz    # Número de estados devido ao tempo morto
        assert(self.nx == self.nxs+self.nxd+self.nxi+self.nxz)
                
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

        # A escolha dos pesos é bastante simplificada: em geral se escolhe peso 1
        self.Qy = 1*np.eye(sys.ny)       # Matriz de peso dos estados
        self.R = 1*np.eye(sys.nu)        # Matriz de peso dos controles
        self.Sy = 1*np.eye(sys.ny)       # Matriz de peso das variáveis de folga dos estados est
        self.Si = 1*np.eye(sys.ny)       # Matriz de peso das variáveis de folga dos estados int

        # Parâmetros do satisficing

        self.gamma_e = gamma_e   # maximo custo do erro
        self.gamma_du = gamma_du   # maximo custo de controle
        self.gamma_syN = gamma_syN   # syN
        self.gamma_siN = gamma_siN   # siN

        self.Q_bar = []
       
        # symbolic variables
        self.dUk = csd.MX.sym('du', self.nu)
        self.Xk = csd.MX.sym('x', self.nx)
        self.Uk = csd.MX.sym('u', self.nu)
        self.Yk = csd.MX.sym('y', self.ny)
        self.Ysp = csd.MX.sym('Ysp', self.ny)    # set-point
        self.syN = csd.MX.sym('syN', self.ny)    # Variáveis de folga na saída
        self.siN = csd.MX.sym('siN', self.ny)    # Variável de folga terminal
        self.Pesos = csd.MX.sym('Pesos', 4)
        self.ViN_ant = csd.MX.sym('ViN_ant')

        self.F = self._DynamicF()
        
        self.prob, self.bounds, self.sobj, self.pred = self._OptimProbl()

    def _DynamicF(self):
        #return the casadi function that represent the dynamic system
        
        dUk = self.dUk  
        Xk = self.Xk  
        Uk = self.Uk  

        A = self.sys.A
        B = self.sys.B
        C = self.sys.C
        D = self.sys.D
        
        Xkp1 = csd.mtimes(A, Xk) + csd.mtimes(B, dUk)
        Ykp1 = csd.mtimes(C, Xkp1) + csd.mtimes(D, dUk)
        Ukp1 = Uk + dUk

        F = csd.Function('F', [Xk, Uk, dUk], [Xkp1, Ykp1, Ukp1],
                     ['x0', 'u0', 'du0'],
                     ['xkp1', 'ykp1', 'ukp1'])
        return F
    
    def _OptimProbl(self):
        # the symbolic optimization problem

        w = []     # Variáveis de otimização
        lbw = []   # Lower bound de w
        ubw = []   # Upper bound de w
        g = []     # Restrições não lineares
        lbg = []   # Lower bound de g
        ubg = []   # Upper bound de g

        Vy = 0   # Subobjetivo da referencia
        Vdu = 0  # Subobjetivo do controle
        VyN = 0  # Subobjetivo do atendimento à condição terminal do modo estacionário
        ViN = 0  # Subobjetivo do atendimento à condição terminal do modo integral
        Vt = 0   # Subobjetivo do custo terminal
        J = 0    # Função objetivo
        
        N = self.N
        Ts = self.Ts
        Qy = self.Qy
        R = self.R
        Sy = self.Sy
        Si = self.Si
        
        
        Ysp = self.Ysp    # set-point
        syN = self.syN    # Variáveis de folga na saída
        siN = self.siN    # Variável de folga terminal
        Xk = self.Xk
        Uk = self.Uk        
        Xkp1 = Xk
        Ukp1 = Uk
        
        X_pred = []
        Y_pred = []
        U_pred = []
        dU_pred = []
                
        F = self.F
        
        for k in range(0, N):
            dU_k = csd.MX.sym('dU_' + str(k), self.nu)  # Variável para o controle em k
            
            # Adiciona duk nas variáveis de decisão
            w += [dU_k]  	        # variável
            lbw += [self.dulb]  	# bound inferior na variável
            ubw += [self.duub]  	# bound superior na variável
        
            # Adiciona a nova dinâmica
            res = F(x0=Xkp1, u0=Ukp1, du0=dU_k)
            Xkp1 = res['xkp1']
            Ykp1 = res['ykp1']
            Ukp1 = res['ukp1']
            
            X_pred += [Xkp1]
            U_pred += [Ukp1]
            Y_pred += [Ykp1]
            dU_pred += [dU_k]
                
            # Bounds em x
            g += [Xkp1]
            lbg += [self.xlb]
            ubg += [self.xub]
        
            # Bounds em u
            g += [Ukp1]
            lbg += [self.ulb]
            ubg += [self.uub]
        
            # Bounds em y (nenhuma)
        
            # Função objetivo
            Vy += csd.dot((Ykp1 - Ysp - syN - (k-N)*Ts*siN)**2, csd.diag(Qy))
            Vdu += csd.dot(dU_k**2, csd.diag(R))
        
        # custo das variáveis de folga
        VyN = csd.dot(syN**2, csd.diag(Sy))
        ViN = csd.dot(siN**2, csd.diag(Si))
        
        # estado terminal
        XN = Xkp1  # terminal state
        XsN = XN[0:self.nxs]
        XiN = XN[self.nxs+self.nxd:self.nxs+self.nxd+self.nxi]
        XdN = XN[self.nxs:self.nxs+self.nxd]
 
        # Adição do custo terminal
        # Q terminal
        #Q_lyap = self.sys.F.T.dot(self.sys.Psi.T).dot(self.Qy).dot(self.sys.Psi).dot(self.sys.F)
        Q_lyap = self.sys.F.T@self.sys.Psi.T@self.Qy@self.sys.Psi@self.sys.F
        #Q_lyap = np.eye(self.nxd)
        Q_bar = sp.linalg.solve_discrete_lyapunov(self.sys.F, Q_lyap, method='bilinear')
        Vt = csd.dot(XdN**2, csd.diag(Q_bar))
        self.Q_bar = Q_bar
        
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
        
        # restrições do Satisficing
        g += [Vy, Vt, Vdu, VyN, ViN]
        lbg += [0, 0, 0, 0, 0]
        ViN_ant = self.ViN_ant
        ubg += [np.inf, np.inf, np.inf, np.inf, ViN_ant]
        
        # variáveis de folga
        
        w += [syN]  	           # variávelJ, Vy, Vdu, VyN, ViN
        lbw += [self.sylb]  	   # bound inferior na variável
        ubw += [self.syub]  	   # bound superior na variável
        
        w += [siN]  	# variável
        lbw += [self.silb]  	# bound inferior na variável
        ubw += [self.siub]  	# bound superior na variável
          
        # custo total
        Pesos = self.Pesos
        J = Pesos[0]*(Vy + Vt) + Pesos[1]*Vdu + Pesos[2]*VyN + Pesos[3]*ViN

        g = csd.vertcat(*g)
        w = csd.vertcat(*w)
        p = csd.vertcat(Xk, Ysp, Uk, Pesos, ViN_ant)
        lbw = csd.vertcat(*lbw)
        ubw = csd.vertcat(*ubw)
        lbg = csd.vertcat(*lbg)
        ubg = csd.vertcat(*ubg)
        

        X_pred = csd.vertcat(*X_pred)
        Y_pred = csd.vertcat(*Y_pred)
        U_pred = csd.vertcat(*U_pred)
        pred = csd.Function('pred', [Xk, Uk, csd.vertcat(*dU_pred)], 
                            [X_pred, Y_pred, U_pred],
                            ['x0', 'u0', 'du_opt'],
                            ['x_pred', 'y_pred', 'u_pred'])

        sobj = csd.Function('sobj',[Xk, Uk, w, Ysp],
                                   [Vy, Vdu, VyN, ViN, Vt],
                                   ['x0', 'u0', 'w0', 'ysp'],
                                   ['Vy', 'Vdu', 'VyN', 'ViN', 'Vt'])

        prob = {'f': J, 'x': w, 'g': g, 'p': p} 
        bounds = {'lbw': lbw, 'ubw': ubw, 'lbg': lbg, 'ubg': ubg}

        return prob, bounds, sobj, pred
    
    
    def MPC(self):
        
        opt = {'expand': False, 'jit': False,
                'verbose_init': 0, 'print_time': False}
        ipopt = {'print_level': 0, 'print_timing_statistics': 'no',
                  'warm_start_init_point': 'yes'}
        opt['ipopt'] = ipopt
        
        # nlp solver
        #prob, bounds, _, _ = self._OptimProbl()
        prob = self.prob
        bounds = self.bounds
        solver = csd.nlpsol('solver', 'ipopt', prob, opt)
        
        p = prob['p']
        lbw = bounds['lbw']
        ubw = bounds['ubw']
        lbg = bounds['lbg']
        ubg = bounds['ubg']
        
        dim_w = prob['x'].shape[0]
        dim_g = prob['g'].shape[0]
        
        W0 = csd.MX.sym('W0', dim_w )
        # multiplicadores de lagrange - initial guess
        LAM_W0 = csd.MX.sym('LW0', dim_w)  
        LAM_G0 = csd.MX.sym('LG0', dim_g)
        
        sol = solver(x0=W0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p,
                  lam_x0=LAM_W0, lam_g0=LAM_G0)
        
        # loop para retornar o resultado em matriz
        du_opt = []
        index = 0
        for kk in range(0, self.N):
            auxU = sol['x'][index:(index+self.nu)]
            du_opt = csd.horzcat(du_opt, auxU)
            index = index + self.nu

        Ysp = self.Ysp
        Xk = self.Xk
        Uk = self.Uk  
        
        Pesos = self.Pesos
        ViN_ant = self.ViN_ant
        
        indx = (self.N-1)*(self.nx+self.nu)
        MPC = csd.Function('MPC',
            [W0, Xk, Ysp, Uk, LAM_W0, LAM_G0, Pesos, ViN_ant],
            [sol['f'], du_opt, sol['x'], sol['lam_x'], sol['lam_g'], sol['g'], sol['g'][indx:indx+self.nx]],
            ['w0', 'x0', 'ySP', 'u0', 'lam_w0', 'lam_g0', 'pesos', 'Vant'],
            ['J','du_opt', 'w_opt', 'lam_w', 'lam_g', 'g', 'xN'])

        return MPC
    
    
    def warmStart(self, sol, ysp):
        w_start = []
        w0 = sol['w_opt'][:]
        
        dustart = w0[0:-2*self.ny].full() # retira syN e siN
        dustart = dustart.reshape((self.nu, self.N))[:,1:] # remove firts du
        dustart = np.hstack((dustart, np.zeros((self.nu,1)))) # add 0 at the end
        dustart = dustart.flatten()
        
        xi = sol['xN']
        ui = 0
        res = self.F(x0=xi, u0=ui, du0=0)
        Xknext = res['xkp1'].full()
        
        xsNp2 = Xknext[0:self.nxs]
        xiNp2 = Xknext[self.nxs+self.nxd:self.nxs+self.nxd+self.nxi]   
        
        syNnext = xsNp2 - ysp
        siNnext = xiNp2
        
        w_start = np.append(dustart, [[syNnext],[siNnext]])
        return w_start


    def satWeights(self, x, w_start, ysp):
        # custos seguintes estimados
        pesos = []
        #_, _, sobj, pred = self._OptimProbl()
        sobj = self.sobj
        
        u = np.zeros(self.nu)   # because u is irrelevant
        
        # sub-objectives
        Vynext, Vdunext, VyNnext, ViNnext, Vtnext = sobj(x, u, w_start, ysp)
        Vytnext = Vynext + Vtnext
        
        py =  1/(self.gamma_e - np.clip(Vytnext, 0, 0.99*self.gamma_e))
        pdu = 1/(self.gamma_du - np.clip(Vdunext, 0, 0.99*self.gamma_du))
        pyN = 1/(self.gamma_syN - np.clip(VyNnext, 0, 0.99*self.gamma_syN))
        piN = 1/(self.gamma_siN - np.clip(ViNnext, 0, 0.99*self.gamma_siN))
        
        pesos = np.append(pesos,(py, pdu, pyN, piN))
        return pesos










    
#    def satWeights(self, x, w_start, ysp):
#        # custos seguintes estimados
#        Vynext = 0
#        Vytnext = 0
#        Vdunext = 0
#        ViNnext = 0
#        VyNnext = 0
#        
#        u = np.zeros(self.nu)   # because u is not relevant
#        du_opt = w_start[:-2*self.ny] # removing syN and siN
#        x_pred, y_pred, _ = self.pred(x, u, du_opt)
#    
#        x_pred = x_pred.reshape((self.nx,self.N)).full()
#        XN = x_pred[:,-1]
#        
#        xsN = XN[0:self.nxs]
#        xdN = XN[self.nxs:self.nxs+self.nxd]
#        xiN = XN[self.nxs+self.nxd:self.nxs+self.nxd+self.nxi]   
#        
#        syNnext = xsN - ysp
#        siNnext = xiN
#        
#        VyNnext = np.dot(syNnext**2, np.diag(self.Sy))
#        ViNnext = np.dot(siNnext**2, np.diag(self.Si))
#        Vtnext = (xdN**2).T@np.diag(self.Q_bar)
#        
#        y_pred = y_pred.reshape((self.ny,self.N)).full()
#        du_opt = du_opt.reshape((self.nu, self.N))
#        
#        for j in range(0, self.N):
#            Vynext += np.dot((y_pred[:,j] - ysp - syNnext - \
#                              (j-self.N)*self.Ts*siNnext)**2, np.diag(self.Qy))
#            Vdunext += np.dot(du_opt[:,j]**2, np.diag(self.R))
#            
#        Vytnext = Vynext + Vtnext
#        
#        py =  1/(self.gamma_e - np.clip(Vytnext, 0, 0.999*self.gamma_e))
#        pdu = 1/(self.gamma_du - np.clip(Vdunext, 0, 0.999*self.gamma_du))
#        pyN = 1/(self.gamma_syN - np.clip(VyNnext, 0, 0.999*self.gamma_syN))
#        piN = 1/(self.gamma_siN - np.clip(ViNnext, 0, 0.999*self.gamma_siN))
#        
#        pesos = [py, pdu, pyN, piN]
#        return pesos
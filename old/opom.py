
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017
@author: Igor Yamamoto and Marcelo Lima
Changed 07/04/2018 - Marcelo Lima
Changed 28/10/2019 - Marcelo Lima (deal with complex numbers)
"""
import numpy as np
from scipy import signal
from scipy.linalg import block_diag
import math

class TransferFunction(signal.TransferFunction):
    def __init__(self, num, den, delay=0):
        super().__init__(num,den)
        self.delay = delay
        self.tf = signal.TransferFunction(num,den)

    def __repr__(self):
        return "tf: {}\ndelay: {}".format(self.tf, self.delay)


class OPOM(object):
    def __init__(self, H, Ts):
        if type(H) != list or np.shape(H[0]) == ():
            H = [H]  
        self.H = np.array(H)
        if self.H.size == 1:
            self.ny = 1
            self.nu = 1
        else:
            self.ny = self.H.shape[0]
            self.nu = self.H.shape[1]
        self.Ts = Ts
        self.na = self._order()  # order of Gij
        self.nd = self.na.sum()
        self.delay_matrix = self._delay_matrix()
        self.theta_max = self.delay_matrix.max()
        self.nz = self.theta_max*self.nu
        self.nx = 2*self.ny+self.nd+self.nz
        
        self.X = np.zeros(self.nx)
        self.R_r, self.R_i, self.D0, self.Di, self.Dd_r, self.Dd_i = \
                                                        self._coeff_matrices()
        self.Bd_ref, self.Psi, self.F, self.N = self._Aux()
        
        self.A, self.B, self.C, self.D = self._create_state_space()      


    def __repr__(self):
        return "A=\n%s\n\nB=\n%s\n\nC=\n%s\n\nD=\n%s" % (self.A.__repr__(),
                                                         self.B.__repr__(),
                                                         self.C.__repr__(),
                                                         self.D.__repr__())

    def _delay_matrix(self):
        # the delay is converted in number of Ts
        return np.apply_along_axis(
                lambda row: list(map(lambda tf: round(tf.delay/self.Ts), row)),
                0,
                self.H)

    def _order(self):
        na = np.zeros((self.ny,self.nu), dtype='int')
        for i in range(self.H.shape[0]):
            for j in range(self.H.shape[1]):
                b = self.H[i,j].num
                a = self.H[i,j].den
                r, p, k = signal.residue(b,a)
                na[i,j] = (p != 0).sum()
        return na
    
    
    def _max_order(self):
        na = 0
        for h in self.H.flatten():
            na_h = len(h.den) - 1
            na = max(na, na_h)
        return na


    def _get_coeff(self, b, a):
        # multiply by 1/s (step)
        a = np.append(a, 0)
        
        # do partial fraction expansion
        # r: Residues
        # p: Poles
        # k: Coefficients of the direct polynomial term
        r, p, k = signal.residue(b, a)

        d_s = np.array([])
        d_d_real = np.array([])
        d_d_imag = np.array([])
        d_i = np.array([])
        
        poles_real = np.array([])
        poles_imag = np.array([])
        integrador = 0
        i = 0
        for i in range(np.size(p)):
            if (p[i] == 0):
                if (integrador):
                    d_i = np.append(d_i, r[i].real)
                else:
                    d_s = np.append(d_s, r[i].real)
                    integrador += 1
            else:
                d_d_real = np.append(d_d_real, r[i].real)
                d_d_imag = np.append(d_d_imag, r[i].imag)
                poles_real = np.append(poles_real, p[i].real)
                poles_imag = np.append(poles_imag, p[i].imag)
        if (d_i.size == 0):
            d_i = np.append(d_i, 0)
        return d_s, d_d_real, d_d_imag, d_i, poles_real, poles_imag


    def _coeff_matrices(self):
        n_order = self._max_order()
        D0 = np.zeros((self.ny, self.nu))
        Dd_r = np.zeros((self.ny, self.nu, n_order))
        Dd_i = np.zeros((self.ny, self.nu, n_order))
        Di = np.zeros((self.ny, self.nu))
        R_r = np.zeros((self.ny, self.nu, n_order))
        R_i = np.zeros((self.ny, self.nu, n_order))
        for i in range(self.ny):  # output
            for j in range(self.nu):  # input 
                if self.H.size > 1:
                    b = self.H[i][j].num
                    a = self.H[i][j].den
                else:
                    b = self.H[i].num
                    a = self.H[i].den
                d0, dd_r, dd_i, di, r_r, r_i = self._get_coeff(b, a)
                D0[i,j] = d0
                Di[i,j] = di
                for k in range(self.na[i,j]):  # state
                    Dd_r[i,j,k] = dd_r[k]
                    Dd_i[i,j,k] = dd_i[k]
                    R_r[i,j,k] = r_r[k]
                    R_i[i,j,k] = r_i[k]
        return R_r, R_i, D0, Di, Dd_r, Dd_i
    
    
    def _Aux(self):
        
        F = np.zeros((self.nd, self.nd))
        Dd_ref = np.zeros((self.nd, self.nd))
        psi = np.zeros((self.ny, self.nd))
        N = np.array([])
        
        Dd_r = self.Dd_r
        Dd_i = self.Dd_i
        
        na_max = self._max_order()
        l = 0
        for i in range(self.ny):
            J = np.zeros((self.nu*na_max, self.nu))
            m = 0
            for j in range(self.nu):
                k=0
                while (k < self.na[i,j]):
                    if self.R_i[i,j,k] == 0:
                        F[l,l] = np.exp(self.Ts * self.R_r[i,j,k])
                        Dd_ref[l,l] = Dd_r[i,j,k]
                        psi[i,l] = 1
                        J[m,j] = 1
                        k += 1
                        l += 1
                        m += 1
                    else:
                        F[l,l] = np.exp(self.Ts * self.R_r[i,j,k]) * \
                                    math.cos(-self.Ts*self.R_i[i,j,k])
                        F[l,l+1] = - np.exp(self.Ts * self.R_r[i,j,k]) * \
                                    math.sin(-self.Ts*self.R_i[i,j,k])
                        F[l+1,l] = np.exp(self.Ts * self.R_r[i,j,k]) * \
                                    math.sin(-self.Ts*self.R_i[i,j,k])
                        F[l+1,l+1] = np.exp(self.Ts * self.R_r[i,j,k]) * \
                                    math.cos(-self.Ts*self.R_i[i,j,k])
                        Dd_ref[l,l] = Dd_r[i,j,k] - Dd_i[i,j,k]
                        Dd_ref[l,l+1] = Dd_r[i,j,k] + Dd_i[i,j,k]
                        Dd_ref[l+1,l] = -Dd_r[i,j,k] - Dd_i[i,j,k]
                        Dd_ref[l+1,l+1] = Dd_r[i,j,k] - Dd_i[i,j,k]
                        psi[i,l] = 1
                        psi[i,l+1] = 0
                        J[m,j] = 1
                        J[m+1,j] = 1
                        k += 2
                        l += 2
                        m += 2
            J = J[0:m,:]
            if i == 0:
                N = J
            else:
                N = np.vstack((N,J)) 
        
        Bd_ref = Dd_ref@F@N
        
        return Bd_ref, psi, F, N
            
    def _Bs(self, l):
        return np.where(self.delay_matrix==l,                    
                        self.D0 + self.Ts*self.Di,
                        0)

    def _Bi(self, l):
        return np.where(self.delay_matrix==l,                    
                        self.Di,
                        0)

        
    def _Bd(self, l):
        Bd = self.Bd_ref
        delay_matrix = self.delay_matrix
        
        Jl = np.zeros(Bd.shape)
        ini = 0
        for i in range(self.ny):
            for j in range(self.nu):
                end = ini + self.na[i,j]
                Jl[ini:end,j] = delay_matrix[i,j] 
                ini = end
        
        return np.where(Jl == l,
                        Bd,
                        0)
        

    def _create_Az(self):
        z1_row = np.zeros((self.nu, self.nx))

        if self.theta_max == 1:
            return z1_row
        else:
            zero_block = np.zeros(((self.theta_max-1)*self.nu, self.nx-self.nz))
            eye_diag = block_diag(*([np.eye(self.nu).tolist()]*(self.theta_max-1)))
            zero_column = np.zeros(((self.theta_max-1)*self.nu, self.nu))
            z2_to_ztheta_max_row = np.hstack((zero_block, eye_diag, zero_column))

            return np.vstack((z1_row, z2_to_ztheta_max_row))

    def _Istar(self):
        Istar = []
        for i in range(self.ny):
            Istar.append(int(any(self.Di[i,:])))
        Istar = np.diag(Istar)
        return Istar
    
    def _create_state_space(self):
        a1 = np.hstack((np.eye(self.ny),
                        np.zeros((self.ny, self.nd)),
                        self.Ts*self._Istar()))
        a2 = np.hstack((np.zeros((self.nd, self.ny)),
                        self.F,
                        np.zeros((self.nd, self.ny))))
        a3 = np.hstack((np.zeros((self.ny, self.ny)),
                        np.zeros((self.ny, self.nd)),
                        self._Istar()))
        for i in range(self.theta_max):
            a1 = np.hstack((a1, self._Bs(i+1)))
            a2 = np.hstack((a2, self._Bd(i+1)))
            a3 = np.hstack((a3, self._Bi(i+1)))
        Ax = np.vstack((a1, a2, a3))
        if self.theta_max == 0:
            A = Ax
            B = np.vstack((self._Bs(0),
                           self._Bd(0),
                           self._Bi(0)))
        elif self.theta_max == 1:
            Az = self._create_Az()
            A = np.vstack((Ax, Az))
            B = np.vstack((self._Bs(0),
                           self._Bd(0),
                           self._Bi(0),
                           np.eye(self.nu)))
        else:
            Az = self._create_Az()
            A = np.vstack((Ax, Az))
            B = np.vstack((self._Bs(0),
                           self._Bd(0),
                           self._Bi(0),
                           np.eye(self.nu),
                           np.zeros(((self.theta_max-1)*self.nu, self.nu))))

        def C():
            if self.theta_max == 0:
                return np.hstack((np.eye(self.ny), self.Psi, np.zeros((self.ny, self.ny))))
            else:
                up = np.hstack((np.eye(self.ny),
                                self.Psi,
                                np.zeros((self.ny, self.ny)),
                                np.zeros((self.ny, self.nz))))
                return up
                
                # in case we want to include the past dU, z_l = du(k-l)
                #eye_diag = block_diag(*([np.eye(self.nu).tolist()]*(self.theta_max)))
                #bottom = np.hstack((np.zeros((self.nz, self.nx-self.nz)),
                #                    eye_diag))
                #return np.vstack((up, bottom))
                

        D = np.zeros((self.ny, self.nu))

        return A, B, C(), D  

    def output(self, dU, samples=1):
        try:
            shape = dU.shape[1]
            print(shape)
        except IndexError:
            dU = np.reshape(dU,(1,self.nu))
        X = np.zeros((samples+1, self.nx))
        X[0] = self.X
        Y = np.zeros((samples+1, self.ny+self.nz))
        Y[0] = self.C.dot(X[0])
        for k in range(samples):
            X[k+1] = self.A.dot(X[k]) + self.B.dot(dU[k])
            Y[k+1] = self.C.dot(X[k+1])

        self.X = X[samples]
        return X[samples], Y[samples]


if __name__ == '__main__':
    h11 = TransferFunction([-0.19], [1, 0], delay=2)
    h12 = TransferFunction([-1.7], [19.5, 1])
    h21 = TransferFunction([-0.763], [31.8, 1])
    h22 = TransferFunction([0.235], [1, 0])
    H = [[h11, h12], [h21, h22]]
    Ts = 1
    model = OPOM(H, Ts)

    g11 = TransferFunction([2.6], [62, 1], delay=1)
    g12 = TransferFunction([1.5], [1426, 85, 1], delay=2)  # g12 = 1.5/(1+23s)(1+62s)
    g21 = TransferFunction([1.4], [2700, 120, 1], delay=3)  # g21 = 1.4/(1+30s)(1+90s)
    g22 = TransferFunction([2.8], [90, 1], delay=4)
    G = [[g11, g12], [g21, g22]]
    sys = OPOM(G, Ts)

from opom import OPOM, TransferFunction
from sihmpc import IHMPCController

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import control as ctl
import plotly.graph_objects as go
import casadi as csd

# def sysF(sys):
#     #return the casadi function that represent the dynamic system
    
#     X = csd.MX.sym('x', sys.nx)
#     U = csd.MX.sym('u', sys.nu)
#     dU = csd.MX.sym('du', sys.nu)

#     A = sys.A
#     B = sys.B
#     C = sys.C
#     D = sys.D
    
#     Xkp1 = csd.mtimes(A, X) + csd.mtimes(B, dU)
#     Ykp1 = csd.mtimes(C, Xkp1) + csd.mtimes(D, dU)
#     Ukp1 = U + dU

#     F = csd.Function('F', [X, U, dU], [Xkp1, Ykp1, Ukp1],
#                     ['x0', 'u0', 'du0'],
#                     ['xkp1', 'ykp1', 'ukp1'])
#     return F


def v(x,u,w,sp,N,sys):
    yp=[]
    xp=[]
    up=[]
    du = w[:-4].reshape(-1,2).T
    sp = np.array(sp).reshape((2,1))
    #F = sysF(sys)
    for i in range(N):
        # ## Simula o sistema ###
        res = sys(x0=x, du0=du[:,i], u0=u)
        x = res['xkp1'].full()
        u = res['ukp1'].full()
        y = res['ykp1'].full()
        yp.append(y)
        up.append(u)
        xp.append(x)
        
    yp = np.hstack(yp)
    up = np.hstack(up)
    xp = np.hstack(xp)

    p = {'yp':yp, 'up':up, 'xp':xp}

    e = yp - sp
    s = e[:,-1].reshape((2,1))
    jy = np.sum((e - s)**2,axis=1).reshape((2,1))
    js = s**2
    v = jy + N*js
    return v, p, jy, js



# w1 = np.array([1.62605966e-03, -1.31611132e-02, -5.61592581e-03,  6.07320516e-04,
#        -5.83453931e-02,  8.43446503e-03, -2.49796722e-02,  1.35654420e-02,
#        -2.96143842e-03,  1.45279249e-02,  1.04401450e-02,  6.87823538e-03,
#         2.05114726e-02,  7.71783255e-18,  3.35920395e-02,  5.14522170e-18,
#         2.11595650e-18,  2.57261085e-18,  0.00000000e+00,  0.00000000e+00,
#         1.47136765e-02, -6.17437291e-01,  0.00000000e+00,  0.00000000e+00])

# w0 = np.array([0.00905694, -0.0429244, 0.00162606, -0.0131611, 
#         -0.00561593, 0.000607321, -0.0583454, 0.00843447, 
#         -0.0249797, 0.0135654, -0.00296144, 0.0145279, 
#         0.0104401, 0.00687824, 0.0205115, 7.71783e-18, 
#         0.033592, 5.14522e-18, 2.11596e-18, 2.57261e-18, 
#         0.0147137, -0.232358, 1.6821e-26, -6.6839e-27])

# x = np.array([[ 9.60000000e+01],
#        [ 5.00000000e-01],
#        [-1.34466190e-14],
#        [ 4.07008723e-14],
#        [-2.89785671e-14],
#        [ 3.74583906e-14],
#        [ 0.00000000e+00],
#        [ 0.00000000e+00],
#        [ 9.05694364e-03],
#        [-4.29244186e-02],
#        [ 1.92704055e-15],
#        [-2.02686803e-15],
#        [ 1.32356825e-15],
#        [-1.81923199e-15],
#        [ 1.36390114e-15],
#        [-1.88637519e-15],
#        [-4.83137576e-15],
#        [ 2.84607076e-15],
#        [-2.96874578e-15],
#        [ 1.80860762e-15],
#        [-7.70737366e-16],
#        [ 3.38381293e-16]])


# u = np.array([[1.95905694],
#        [1.66707558]])


# yPlot=[]
# uPlot=[]
# xPlot=[]
# duPlot=[]


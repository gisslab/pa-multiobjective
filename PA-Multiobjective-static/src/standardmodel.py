
##########################################################################################
######                       Module Perfect Information - Standard form             ######
# this file compiles the methods of the symmetric (perfect) information case 
# of the principal-agent model approximation with reservation utilities (standard method)
##########################################################################################


from random import Random
from time import time
import inspyred
import math
from datetime import datetime
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
#roots
from numpy import *
from scipy.optimize import fsolve
import uuid #unique id

################# Params and Variables ###############

YL = 2.0        # Producto bajo.
YH = 4.0        # Producto alto.
h = 0.10        # Coeficiente de aversión al riesgo.
H = h/(1-h)
# precision order 1/n, number of points of the approximation
n = 10000

################# Methods #######################

def set_crra(crra):
    h = 0.10        
    H = h/(1-h)

def set_precision(p):
    n = p

def AccionCM(v):
    a = ((YH-YL) - (v * (1-h))**H ) / ( (YH-YL) + h * (v * (1-h))**(H-1));
    return a

def Salary(a,v):
    W = ((v + a)*(1 - h))**( 1 / (1-h) );
    return W

def PrincipalUtility(W,a):   
    U = math.exp(- a) * YL + (1 - math.exp(- a)) * YH - W ;
    return U

def AgentUtility(W,a): 
    V = (W**(1 - h)) / (1 - h) - a;
    return V 


def function_roots(z,crra):
    v = z[0]
    H = crra/(1-crra)
    F = empty((1))
    F[0] = math.exp(- ((YH-YL) - (v * (1-crra))**H ) / ( (YH-YL) + h * (v * (1-crra))**(H-1) )) * YL + (1 - math.exp(- ((YH-YL) - (v * (1-crra))**H ) / ( (YH-YL) + h * (v * (1-h))**(H-1) ))) * YH - ((v + ((YH-YL) - (v * (1-h))**H ) / ( (YH-YL) + h * (v * (1-h))**(H-1) ))*(1 - h))**( 1 / (1-h) )
    return F

def classical_model_solve(crra,thresh = 0.0):
    """ This method numerically solves the PA problem for the perfect information case with the standard method. 

    Args:
        crra (float): This is the coefficient of relative risk aversion
        thresh (float, optional): This is the upper bound of reservation utilities. Defaults to calculate roots for well-defined values..

    Returns:
        list, list, list, list, list: reservation utilities, efforts, wages, principal utilities, agent utilities
    """
    global h, H
    h = crra
    H = h/(1-h)
    v_0 = []

    #to do: calculate threshold for defined values
    if thresh == 0:
        zGuess = array([20])
        z_root=fsolve(function_roots, zGuess,h)
        print(" root bound : ",z_root[0], " crra : ", crra)
    else:
        z_root = thresh

    # vector of well-defined reservation utilities 
    v_0= [a for a in np.arange(z_root/n, z_root,z_root/n)]

    va_0 = []
    vW_0 = []
    vU_0 = []
    vV_0 = []
    for x in v_0:
        a = AccionCM(x)
        va_0.append(a) 
        W = Salary(a,x)
        vW_0.append(W)
        U = PrincipalUtility(W,a)
        vU_0.append(U)
        V = AgentUtility(W,a)
        vV_0.append(V)
        
    return v_0,va_0,vW_0,vU_0,vV_0

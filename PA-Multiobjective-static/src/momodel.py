
##########################################################################################
######                       Module Perfect Information                           ######
# this file compiles the methods of the symmetric (perfect) information case 
# of the principal-agent model with a multiobjective (Pareto weight) approximation 
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

YL = 2.0      
YH = 4.0        
h = 0.10        
H = h/(1-h)
# precision order 1/n, number of points of the approximation
n = 1000

################# Methods #######################

def set_crra(crra):
    h = 0.10        
    H = h/(1-h)


def set_precision(p):
    n = p

# This function is used in the Euler method, of scipy, fsolve to solve the values with positive utilities for the principal
def upper_bound(approx):
    z = approx[0]
    
    F1 = empty((1))
    F1[0] = math.exp(- math.log((z/(1 - z)) * (YH - YL))) * YL + (1 - math.exp(- math.log((z/(1 - z)) * (YH - YL)))) * YH - ((1 - z)/z)**(1 / h)
    return F1

# This function is used in the Euler method, of scipy, fsolve to solve the values with positive utilities for the agent
def lower_bound(approx):
    z = approx[0]
    F2 = empty((1))
    F2[0] = (((1 - z)/z)**(1 / h)**(1 - h))/ (1 - h) - math.log((z/(1 - z)) * (YH - YL))
    return F2


def AccionMoM(z):
    aa =  math.log((z/(1 - z)) * (YH - YL));
    return aa

def SalaryMoM(z):
    WW = ((1 - z)/z)**(1 / h);
    return WW
    
def PrincipalUtilityMoM(WW,aa):   
    UMoM = math.exp(- aa) * YL + (1 - math.exp(- aa)) * YH - WW ;
    return UMoM

def AgentUtilityMoM(WW,aa): 
    VMoM = (WW**(1 - h))/ (1 - h) - aa;
    return VMoM


def mo_model_solve(crra, thresh_1 = -1, thresh_2 = -1):
    """This method solves the multiobjective version of the principal agent model for a given crra.

    Args:
        crra (float): This is the coefficient of relative risk aversion
        thresh_1 ([type]): Lower bound for bargaining powers
        thresh_2 ([type]): Upper bound for bargaining powers

    Returns:
    list, list, list, list, list: bargaining powers, efforts, wages, principal utilities, agent utilities
    """
    global h, H
    h = crra
    H = h/(1-h)
    
    z = []

    # If the bounds are not explicit, calculate bounds where utilities are positive
    if thresh_1 == -1:
        thresh_1=fsolve(lower_bound, 0.1)
        print(" lower bound ",thresh_1," h ", h)
    if thresh_2 == -1:
        thresh_2=fsolve(upper_bound, 0.1)
        print(" upper bound ",thresh_2," h ", h)

    # lower bound must be greater than 1/3
    thresh_1 = max( 1/3, thresh_1)
    z_final= [z for z in np.arange(thresh_1, thresh_2, (thresh_2-thresh_1)/n)]

    vaa = []
    vWW = []
    vUMoM = []
    vVMoM = []

    for x in z_final:
        aa = AccionMoM(x)
        vaa.append(aa) 
        WW = SalaryMoM(x)
        vWW.append(WW)
        UMoM = PrincipalUtilityMoM(WW,aa)
        vUMoM.append(UMoM)
        VMoM = AgentUtilityMoM(WW,aa)
        vVMoM.append(VMoM)

    return  z_final, vaa, vWW, vUMoM, vVMoM
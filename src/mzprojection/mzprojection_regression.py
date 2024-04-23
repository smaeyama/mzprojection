#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time as timer


def mzprojection_regression(g, f, regression_type="linear", flag_terms=False, flag_debug=False):
    r'''
    Evaluate discrete-time M-Z projection of f(t_n) on g(t_n),
      f(t_n) = Omega(g(t_n)) + \sum_{l=1}^{n-1} Gamma(t_l)[g(t_{n-l})] + r(t_n),
    where f and r are the vectors with the length nf (e.g., f = f_i for i=0,1,...,nf-1),
    while g is the vector with the length ng (e.g, g = g_j for j=0,1,...,ng-1).
    The Markov relation Omega is a nonlinear vector function of g. 
    The memory kernel Gamma(t_l) is a time-dependent nonlinear vector function of g.
    The uncorrelated term r(t_n) is orthogonal to the projection, P[r(t_n)]=0 for any t_n.

    Just for numerical convenience, we define Gamma(0) \equiv Omega. Then, Eq. is
      f(t_n) = \sum_{l=0}^{n-1} Gamma(t_l)[g(t_{n-l})] + r(t_n),
        
    Parameters
    ----------
    g[nsample,nperiod,ng] : Numpy array (float64 or complex128)
        Explanatory variable g(t_n).
        nsample is the number of samples.
        nperiod is the number of time steps of a short-time data.
        ng is the number of independent explanatory variable (g_j for j=0,1,...,ng-1).
    f[nsample,nperiod,nf] : Numpy array (float64 or complex128)
        Response variable f(t).
        nf is the number of independent response variables (f_i for i=0,1,...,nf-1).
        If 2D array f[nsample,nperiod] is passed, it is treated as 3D array with nf=1.
    regression_type : Str
        Choose the regression method.
            regression_type = "linear" # Linear regression
            regression_type = "ridge"  # Ridge regression by using sklearn
        Default: regression_type = "linear"
    flag_terms : bool
        Control flag to output memory and uncorrelated terms. 
        Default: flag_term = False
    flag_debug : bool
        For debug.
        Default: flag_debug = False
    
    Returns
    -------
    omega : An object of regression function
        The Markov retlation Omega.
        Use the "fit" method to evaluate the Markov term, Omega(g(t_n)).
            ### Example ###
            isample = 0; iperiod = 10
            markov_term = omega.fit(g[isample,iperiod,:])
            ###############
    gamma : List of the objects of regression function
        The memory kernel Gamma(t_l).
        The list length corresponds to the temporal index of the memory kernel function, 
        namely, gamma = [Omega, Gamma(t_1), Gamma(t_2), ..., Gamma(t_{n-1})].
        Use the "fit" method to evaluate the memory term, \sum_{l=1}^{n-1} Gamma(t_l)[g(t_{n-l})].
            ### Example ###
            isample = 0; iperiod = len(gamma)//2
            markov_term = gamma[0].fit(g[isample,iperiod,:])
            memory_term = 0.0
            for i in range(1,iperiod+1):
                memory_term = memory_term + gamma[i].fit(g[isample,iperiod-i,:])
            ###############
    
    # if flag_terms==True:
    s[nsample,nperiod,nf] : Numpy array (float64 or complex128)
        The memory term s(t_n) = \sum_{l=1}^{n-1} Gamma(t_l)[g(t_{n-l})].
        Note that the Markov term Omega(g(t_n)) = Gamma(0)[g(t_n)] is not included in the summation.
    r[nsample,nperiod,nf] : Numpy array (float64 or complex128)
        The uncorrelated term r(t_n) (also called orthogonal term or noise term).
    '''
    nsample = g.shape[0]
    nperiod = g.shape[1]
    if g.ndim == 2:
        ng = 1
        g = g.reshape(nsample,nperiod,ng)
    else:
        ng = g.shape[2]
    if f.ndim == 2:
        nf = 1
        f = f.reshape(nsample,nperiod,nf)
    else:
        nf = f.shape[2]
    if flag_debug:
        print("nsample=",nsample,", nperiod=",nperiod,", ng=",ng,", nf=",nf)
        t1=timer()

    # Evaluate Markov relation, Omega(g(0)) = P[f(0) | g(0)]
    omega = regression(g[:,0,:], f[:,0,:], regression_type)
    if flag_debug:
        t2=timer();print("# Elapsed time to calc. Markov relation [sec]:",t2-t1);t1=timer()

    # Evaluate Memory kernel, Gamma(t_n)[g(0)]=P[f(t_n)-Omega(g(t_n))-\sum_{l=1}^{n-1} Gamma(t_l)[g(t_{n-l})] | g(0)]
    gamma = [omega]
    for i in range(1,nperiod):
        wf = f[:,i,:]
        for j in range(i):
            wf = wf - gamma[j].fit(g[:,i-j,:])
        gamma.append(regression(g[:,0,:], wf, regression_type))
    if flag_debug:
        t2=timer();print("# Elapsed time to calc. memory kernel [sec]:",t2-t1);t1=timer()
        
    if flag_terms == False:
        return omega, gamma
    else:
        # Evaluate the memory term s(t_n) and the uncorrelated term r(t_n)
        s, r = calc_residual_regression(g, f, omega, gamma)
        if flag_debug:
            t2=timer();print("# Elapsed time to calc. residual r [sec]:",t2-t1);t1=timer()
            print("s[nsample,nperiod,nf].shape=",s.shape,s.dtype)
            print("r[nsample,nperiod,nf].shape=",r.shape,r.dtype)
        return omega, gamma, s, r


def calc_residual_regression(g, f, omega, gamma):
    r'''
    Parameters
    ----------
    g[nsample,nperiod,ng] : Numpy array
    f[nsample,nperiod,nf] : Numpy array
    omega                 : Object
    gamma[nperiod]        : List of objects  
    
    Returns
    -------
    s[nsample,nperiod,nf] : Numpy array
    r[nsample,nperiod,nf] : Numpy array
    '''
    nperiod = g.shape[1]

    markov_term = omega.fit(g)

    s_all = [markov_term[:,0,:]] # s_all = markov_term + memory term s
    for i in range(1,nperiod):
        ws = markov_term[:,i,:]
        for j in range(1,i):
            ws = ws + gamma[j].fit(g[:,i-j,:])
        s_all.append(ws)
    s_all = np.array(s_all)                  # s[nperiod,nsample,nf]
    s_all = np.transpose(s_all,axes=(1,0,2)) # s[nsample,nperiod,nf]

    s = s_all - markov_term
    r = f - s_all
    return s, r


def regression(g, f, regression_type):
    r"""
    Interface for the regression functions
    """
    if regression_type == "linear_wo_origin":
        return linear_function_wo_origin(g=g,f=f)
    if regression_type == "linear":
        return linear_function(g=g,f=f)
        # return linear_function_sklearn(g=g,f=f)
    if regression_type == "ridge":
        return ridge_sklearn(g=g,f=f)
    else:
        print("Unavailable regression type:", regression_type)
        return




### Followings are the presently implemented classes of the regression functions ###

class linear_function():
    r"""
    Linear regression
        f = matA.(g-g_ave) + f_ave
        
    Parameters for constructor __init__
    -----------------------------------
    g[nsample,nu]
    f[nsample,nf]
    
    Parameters for self.fit
    ------------------------
    g[nu]
    
    Return for self.fit
    --------------------
    f[nf] : = matA.(g-g_ave) + f_ave    
    """
    def __init__(self, *args, **kwargs):
        g = kwargs.get("g")
        f = kwargs.get("f")
        # Calculate coefficients
        g_ave = np.average(g,axis=0)
        f_ave = np.average(g,axis=0)
        wg = g[:,:] - g_ave.reshape(1,g_ave.shape[0])
        wf = f[:,:] - f_ave.reshape(1,f_ave.shape[0])
        gg = np.tensordot(wg,np.conjugate(wg),axes=(0,0))
        gg_inv = np.linalg.inv(gg)
        fg = np.tensordot(wf,np.conjugate(wg),axes=(0,0))
        matA = np.dot(fg,gg_inv)
        self.__g_ave = g_ave
        self.__f_ave = f_ave
        self.__matA = matA
        return
    def fit(self,g):
        return np.dot(g-self.__g_ave,self.__matA.T) + self.__f_ave
    def matA(self):
        return self.__matA


class linear_function_wo_origin():
    r"""
    Linear regression
        f = matA.(g)
    No care on the origin (g_ave,f_ave), which equivalent to the function
    mzprojection_multivariate_discrete_time.
        
    Parameters for constructor __init__
    -----------------------------------
    g[nsample,nu]
    f[nsample,nf]
    
    Parameters for self.fit
    ------------------------
    g[nu]
    
    Return for self.fit
    --------------------
    f[nf] : = matA.(g-g_ave) + f_ave    
    """
    def __init__(self, *args, **kwargs):
        g = kwargs.get("g")
        f = kwargs.get("f")
        # Calculate coefficients
        wg = g[:,:]
        wf = f[:,:]
        gg = np.tensordot(wg,np.conjugate(wg),axes=(0,0))
        gg_inv = np.linalg.inv(gg)
        fg = np.tensordot(wf,np.conjugate(wg),axes=(0,0))
        matA = np.dot(fg,gg_inv)
        self.__matA = matA
        return
    def fit(self,g):
        return np.dot(g,self.__matA.T)
    def matA(self):
        return self.__matA

    
class linear_function_sklearn():
    def __init__(self, *args, **kwargs):
        from sklearn.linear_model import LinearRegression
        g = kwargs.get("g")
        f = kwargs.get("f")
        self.__reg = LinearRegression()
        self.__reg.fit(g,f)
        return
    def fit(self,g):
        return self.__reg.predict(g)
    def matA(self):
        return self.__reg.coef_

    
class ridge_sklearn():
    r'''
    Parameters
    ----------
    delta_t
    u[nsample,nperiod,nu]
    f[nsample,nperiod,nf]
    omega[nf,nu]
    memoryf[nperiod,nf,nu]  
    
    Returns
    -------
    s[nsample,nperiod,nf]
    r[nsample,nperiod,nf]
    '''
    def __init__(self, *args, **kwargs):
        from sklearn.linear_model import Ridge
        g = kwargs.get("g")
        f = kwargs.get("f")
        self.__reg = Ridge()
        self.__reg.fit(g,f)
        return
    def fit(self,g):
        return self.__reg.predict(g)

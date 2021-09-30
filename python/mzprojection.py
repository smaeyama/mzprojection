#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time as timer


def split_long_time_series(u_raw, ista=0, nperiod=None, nshift=10):
    """
    Split a long-time-series data into samples of short-time data.
    
    Parameters
    ----------
    u_raw[nrec,nu] : Numpy array
        nu is the number of independent variables.
        nrec is the length of a long-time-series data.
        1D u_raw[nrec], or multi-dimensional array u_raw[nrec,nu1,nu2,...,nuN] are also available.
    ista : int
        Start time step number for sampling.
        Default: ista = 0
    nperiod : int
        Time step length of a short-time data.
        Default: nperiod = int(nrec/100) 
    nshift : int
        Length of time shift while sampling.
        Default: nshift = 10

    Returns
    -------
    u[nsample,nperiod,nu] : Numpy array
        nsample is the number of samples.
    """
    nrec = u_raw.shape[0]
    if nperiod is None:
        nperiod = int(nrec/100)
    nsample = int((nrec-ista-nperiod)/nshift) + 1
        
    u = []
    for isample in range(nsample):
        tsta = ista + nshift*isample
        tend = tsta + nperiod
        u.append(u_raw[tsta:tend])
    u = np.array(u)
    return u


def calc_correlation(a,b):
    '''
    Parameters
    ----------
    a[nsample,nperiod,na]
    b[nsample,nb]
    
    Returns
    -------
    corr[nperiod,na,nb]
    '''
    coef = 1/a.shape[0] # =1/nsample
    corr = np.tensordot(a,np.conjugate(b),axes=(0,0)) * coef
    return corr


def solve_memory_integration(delta_t, f, u0, dudt0, uu0_inv, ududt, G, wG0_inv):
    '''
    Parameters
    ----------
    delta_t
    f[nsample,nperiod,nf]
    u0[nsample,nu]
    dudt0[nsample,nu]
    uu0_inv[nu,nu]
    ududt[nperiod,nu,nu]
    G[nperiod,nu,nu]
    wG0_inv[nu,nu]
    
    Returns
    -------
    omega[nf,nu]
    memoryf[nperiod,nf,nu]  
    '''
    #= Evaluate Markov coefficient Omega =
    fu0 = calc_correlation(f[:,0,:],u0)
    omega = np.dot(fu0,uu0_inv)
        
    #= Evaluate memory function Gamma(t) =
    fdudt = calc_correlation(f,dudt0)
    F = np.dot(fdudt,uu0_inv) - np.moveaxis(np.dot(omega,np.dot(ududt,uu0_inv)),1,0)
    memoryf = np.zeros_like(F)
    memoryf[0] = F[0]
    memoryf[1] = np.dot(F[1] + 0.5*delta_t*np.dot(memoryf[0],G[1]),wG0_inv)
    for iperiod in range(2,memoryf.shape[0]): # 2nd-order trapezoid integration in time
        memoryf[iperiod] = np.dot(F[iperiod] + 0.5*delta_t*np.dot(memoryf[0],G[iperiod])
                                  + delta_t*np.tensordot(memoryf[1:iperiod,:,:],G[iperiod-1:0:-1,:,:],axes=([0,-1],[0,-2])),wG0_inv)
    return omega, memoryf # omega[nu], memoryf[nperiod,nu]


# def calc_omega_and_memoryf_1f(f, u0, dudt0, uu0_inv, ududt, G, wG0_inv):
#     '''
#     Parameters
#     ----------
#     f[nsample,nperiod]  (for a given f_i)
#     u0[nsample,nu]
#     dudt0[nsample,nu]
#     uu0_inv[nu,nu]
#     ududt[nperiod,nu,nu]
#     G[nperiod,nu,nu]
#     wG0_inv[nu,nu]
#    
#     Returns
#     -------
#     omega[nu]           (for a given f_i)
#     memoryf[nperiod,nu] (for a given f_i)    
#     '''
#     #= Evaluate Markov coefficient Omega =
#     fu0 = calc_correlation(f[:,0],u0)
#     omega = np.dot(fu0,uu0_inv)
#    
#     #= Evaluate memory function Gamma(t) =
#     fdudt = calc_correlation(f,dudt0)
#     F = np.dot(fdudt,uu0_inv) - np.dot(omega,np.dot(ududt,uu0_inv))
#     memoryf = np.zeros_like(F)
#     memoryf[0] = F[0]
#     memoryf[1] = np.dot(F[1] + 0.5*delta_t*np.dot(memoryf[0],G[1]),wG0_inv)
#     for iperiod in range(2,nperiod): # 2nd-order trapezoid integration in time
#         memoryf[iperiod] = np.dot(F[iperiod] + 0.5*delta_t*np.dot(memoryf[0],G[iperiod])
#                                   + delta_t*np.tensordot(memoryf[1:iperiod,:],G[iperiod-1:0:-1,:,:],axes=([0,1],[0,1])),wG0_inv)
#     return omega, memoryf # omega[nu], memoryf[nperiod,nu]
     
    
def mzprojection_multivariate(delta_t, u, dudt0, f, flag_terms=False, flag_debug=False):
    '''
    Evaluate projection of f(t) on u(t),
      f_i(t) = Omega_ij*u_j(t) - int_0^t Gamma_ij(s)*u_j(t-s) ds + r_i(t)
    taking summation over the repeated index j.
    
    Parameters
    ----------
    delta_t : float
        Time step size
    u[nsample,nperiod,nu] : Numpy array (float64 or complex128)
        Explanatory variable u_j(t).
        nsample is the number of samples.
        nperiod is the time step length of a short-time data.
        nu is the number of independent explanatory variable (j=0,1,...,nu-1).
    dudt[nsample,nperiod,nu] : Numpy array (float64 or complex128)
        = du/dt
    f[nsample,nperiod,nf] : Numpy array (float64 or complex128)
        Response variable f_i(t).
        nf is the number of independent response variables (i=0,1,...,nf-1).
        If 2D array f[nsample,nperiod] is passed, it is treated as 3D array with nf=1.
    flag_terms : bool
        Control flag to output memory and uncorrelated terms. 
        Default: flag_term = False
    flag_debug : bool
        For debug.
        Default: flag_debug = False
    
    Returns
    -------
    omega[nf,nu] : Numpy array (float64 or complex128)
        Markov coefficient matrix Omega_ij.
    memoryf[nperiod,nf,nu] : Numpy array (float64 or complex128)
        Memory function matrix Gamma_ij(t).
    
    # if flag_terms==True:
    s[nsample,nperiod,nf] : Numpy array (float64 or complex128)
        Memory term s_i(t) = - int_0^t Gamma_ij(s)*u_j(t-s) ds
    r[nsample,nperiod,nf] : Numpy array (float64 or complex128)
        Uncorrelated term r_i(t) (also called orthogonal term or noise term).
    '''
    nsample = u.shape[0]
    nperiod = u.shape[1]
    if u.ndim == 2:
        nu = 1
        u = u.reshape(nsample,nperiod,nu)
        dudt0 = dudt0.reshape(nsample,nu)
    else:
        nu = u.shape[2]
    if f.ndim == 2:
        nf = 1
        f = f.reshape(nsample,nperiod,nf)
    else:
        nf = f.shape[2]
    if flag_debug:
        print("nsample=",nsample,", nperiod=",nperiod,", nu=",nu,", nf=",nf)
        t1=timer()
    
    #= Evaluate correlations of explanatory variable u =
    u0 = u[:,0,:]
    uu0 = calc_correlation(u0,u0)
    ududt = calc_correlation(u,dudt0)
    uu0_inv = np.linalg.inv(uu0)
    print(u.shape,dudt0.shape,uu0.shape,ududt.shape,uu0_inv.shape)
    G = np.dot(ududt,uu0_inv)
    wG0_inv = np.linalg.inv(np.identity(nu)-0.5*delta_t*G[0,:,:])
    if flag_debug:
        t2=timer();print("# Prepare correlations [sec]:",t2-t1);t1=timer()
        print("      uu0_inv[nu,nu].shape=",uu0_inv.shape,uu0_inv.dtype)
        print("ududt[nperiod,nu,nu].shape=",ududt.shape,ududt.dtype)
        print("    G[nperiod,nu,nu].shape=",G.shape,G.dtype)
        print("      wG0_inv[nu,nu].shape=",wG0_inv.shape,wG0_inv.dtype)

    
    #= Evaluate Markov coefficient Omega nad memory function Gamma =
    omega, memoryf = solve_memory_integration(delta_t, f, u0, dudt0, uu0_inv, ududt, G, wG0_inv)
    if flag_debug:
        t2=timer();print("# Calc. omega & memoryf [sec]:",t2-t1);t1=timer()
        print("          omega[nf,nu].shape=",omega.shape,omega.dtype)
        print("memoryf[nperiod,nf,nu].shape=",memoryf.shape,memoryf.dtype)
        
    if flag_terms == False:
        
        return omega, memoryf
    
    else:
        
        #= Evaluate uncorrelated term r(t) as a residual =
        #- 2nd-order trapezoid integration for memory term s(t) -
        wkmemf = np.zeros([nperiod,nperiod,nf,nu], dtype=memoryf.dtype)
        wkmemf[1,0] = 0.5*memoryf[1]
        wkmemf[1,1] = 0.5*memoryf[0]
        for iperiod in range(2,nperiod):
            wkmemf[iperiod,0        ] = 0.5*memoryf[iperiod]
            wkmemf[iperiod,1:iperiod] = memoryf[iperiod-1:0:-1]
            wkmemf[iperiod,iperiod  ] = 0.5*memoryf[0]
        if flag_debug:
            t2=timer();print("# Prepare to calc. s & r [sec]:",t2-t1);t1=timer()
        s = - delta_t*np.tensordot(u,wkmemf,axes=([1,2],[1,-1]))
        if flag_debug:
            t2=timer();print("# Calc. memory term s [sec]:",t2-t1);t1=timer()
        r = f - np.dot(u,omega.T) - s
        if flag_debug:
            t2=timer();print("# Calc. residual r [sec]:",t2-t1);t1=timer()
            print("s[nsample,nperiod,nf].shape=",s.shape,s.dtype)
            print("r[nsample,nperiod,nf].shape=",r.shape,r.dtype)
        #%%% NOTE %%%
        #% Since the above s, r is defined from u[isample,:,:] and f[isample,:,:],
        #%     s[isample,0,:] = 0
        #% is always imposed. Then, <s(t)s>=0, because s(0)=0.
        #% However, after a long time, s and r is also expected to be in a statistically steady state, 
        #%     <s(t)s> = <s(t+x)s(x)> /= 0.
        #%%%%%%%%%%%%%%%
        
        return omega, memoryf, s, r


### Bellows are functions for 1-to-1 projection, which will be merged to mzprojection_multivariate



def mzprojection_ensemble_of_time_series(nsample, nperiod, delta_t, u, dudt, f):
    '''

    Evaluate projection of f(t) on u(t)

      f(t) = Omega*u(t) - int_0^t Gamma(s)*u(t-s) ds + r(t)

    * User prepares ensembles of time-series data u(t)^i, du/dt(t)^i, f(t)^i

    != INPUT =
    integer          :: nsample              ! # of samples for ensemble average
    integer          :: nperiod              ! Length of a sample
    numpy.float64    :: delta_t              ! Time step size
    numpy.complex128 ::    u[nperiod,nsample]! Variable u(t)
    numpy.complex128 :: dudt[nperiod,nsample]! = du/dt
    numpy.complex128 ::    f[nperiod,nsample]! Analyzed f(t)

    != OUTPUT =
    numpy.complex128 :: omega                ! Markov coefficient Omega
    numpy.complex128 :: memoryf[nperiod]     ! Memory function Gamma(t)
    numpy.complex128 :: s[nperiod,nsample]   ! Memory term
    numpy.complex128 :: r[nperiod,nsample]   ! Uncorrelated term r(t)

    != OUTPUT for check =
    numpy.complex128 ::    uu[nperiod]       ! Correlation <u(t)u>
    numpy.complex128 :: ududt[nperiod]       ! Correlation <u(t)du/dt>
    numpy.complex128 :: fdudt[nperiod]       ! Correlation <f(t)du/dt>
    numpy.complex128 ::    rr[nperiod]       ! Correlation <r(t)r>
    numpy.complex128 :: rdudt[nperiod]       ! Correlation <r(t)du/dt>
    numpy.complex128 ::    ru[nperiod]       ! Correlation <r(t)u>
    numpy.complex128 ::    fu[nperiod]       ! Correlation <f(t)u>
    numpy.complex128 ::    ff[nperiod]       ! Correlation <f(t)f>

    '''
    import numpy as np

    #= Evaluate Markov coefficient Omega =
    omega = sum(f[0,0:nsample]*np.conjugate(u[0,0:nsample])) / sum(u[0,0:nsample]*np.conjugate(u[0,0:nsample])) 

    #= Evaluate memory function Gamma(t) =
    uu    = np.dot(u[:,0:nsample],np.conjugate(   u[0,0:nsample])) / nsample
    ududt = np.dot(u[:,0:nsample],np.conjugate(dudt[0,0:nsample])) / nsample
    fdudt = np.dot(f[:,0:nsample],np.conjugate(dudt[0,0:nsample])) / nsample
    #- 2nd-order solver for memory equation -
    memoryf = np.zeros(nperiod, dtype=np.complex128)
    memoryf[0] = (fdudt[0] - omega*ududt[0]) / uu[0]
    memoryf[1] = (fdudt[1] - omega*ududt[1] + memoryf[0]*ududt[1]*0.5*delta_t) / uu[0]
    memoryf[1] = memoryf[1] / (1.0 - ududt[0]*0.5*delta_t/uu[0])
    for iperiod in range(2,nperiod):
        memoryf[iperiod] = (fdudt[iperiod] - omega*ududt[iperiod] + memoryf[0]*ududt[iperiod]*0.5*delta_t + sum(memoryf[1:iperiod]*ududt[iperiod-1:0:-1])*delta_t) / uu[0]
        memoryf[iperiod] = memoryf[iperiod] / (1.0 - ududt[0]*0.5*delta_t/uu[0])
    #=

    #= Evaluate uncorrelated term r(t) as a residual =
    #- 2nd-order integration for memory term s(t) -
    wkmemf = np.zeros([nperiod,nperiod], dtype=np.complex128)
    wkmemf[1,0] = 0.5*memoryf[1]
    wkmemf[1,1] = 0.5*memoryf[0]
    for iperiod in range(2,nperiod):
        wkmemf[iperiod,0        ] = 0.5*memoryf[iperiod]
        wkmemf[iperiod,1:iperiod] = memoryf[iperiod-1:0:-1]
        wkmemf[iperiod,iperiod  ] = 0.5*memoryf[0]
    s = - np.dot(wkmemf[:,0:nperiod],u[0:nperiod,:]) * delta_t
    r = f[:,:] - omega*u[:,:] - s[:,:]
    #=

    #= For check memory function and <r(t)u>=0 =
    rr    = np.dot(r[:,0:nsample],np.conjugate(   r[0,0:nsample])) / nsample
    rdudt = np.dot(r[:,0:nsample],np.conjugate(dudt[0,0:nsample])) / nsample
    ru    = np.dot(r[:,0:nsample],np.conjugate(   u[0,0:nsample])) / nsample
    fu    = np.dot(f[:,0:nsample],np.conjugate(   u[0,0:nsample])) / nsample
    ff    = np.dot(f[:,0:nsample],np.conjugate(   f[0,0:nsample])) / nsample
    #%%% NOTE %%%
    #% Since the above s, r is defined from u[isample,:] and f[isample,:],
    #%     s[isample,0] = 0
    #% is always imposed. Then, <s(t)s>=0, because s(0)=0.
    #% However, after a long time, s and r is also expected to be in a statistically
    #% steady state, <s(t)s> = <s(t+x)s(x)> /= 0.
    #%%%%%%%%%%%%%%%

    return omega, memoryf, s, r, uu, ududt, fdudt, rr, rdudt, ru, fu, ff



def mzprojection_long_time_series(nrec, ista, nperiod, nshift, delta_t, u_raw, dudt_raw, f_raw):
    '''

    Evaluate projection of f(t) on u(t)

      f(t) = Omega*u(t) - int_0^t Gamma(s)*u(t-s) ds + r(t)

    * User prepares long time-series data u(t), dudt(t), f(t)

    != INPUT =
    integer          :: nrec             ! Total record number
    integer          :: ista             ! Start record number for sampling
    integer          :: nperiod          ! Length of a sample
    integer          :: nshift           ! Length of time shift while sampling
    numpy.float64    :: delta_t          ! Time step size
    numpy.complex128 ::    u[nrec]       ! Variable u(t)
    numpy.complex128 :: dudt[nrec]       ! = du/dt
    numpy.complex128 ::    f[nrec]       ! Analyzed f(t)

    != OUTPUT =
    numpy.complex128 :: omega            ! Markov coefficient Omega
    numpy.complex128 :: memoryf[nperiod] ! Memory function Gamma(t)
    numpy.complex128 :: s[nrec]          ! Memory term
    numpy.complex128 :: r[nrec]          ! Uncorrelated term r(t)

    != OUTPUT for check =
    numpy.complex128 ::    uu[nperiod]   ! Correlation <u(t)u>
    numpy.complex128 :: ududt[nperiod]   ! Correlation <u(t)du/dt>
    numpy.complex128 :: fdudt[nperiod]   ! Correlation <f(t)du/dt>
    numpy.complex128 ::    rr[nperiod]   ! Correlation <r(t)r>
    numpy.complex128 :: rdudt[nperiod]   ! Correlation <r(t)du/dt>
    numpy.complex128 ::    ru[nperiod]   ! Correlation <r(t)u>
    numpy.complex128 ::    fu[nperiod]   ! Correlation <f(t)u>
    numpy.complex128 ::    ff[nperiod]   ! Correlation <f(t)f>

    '''
    import numpy as np
    import time

    nsample_raw = int((nrec-ista-nperiod)/nshift) + 1
    u = np.zeros([nperiod,nsample_raw], dtype=np.complex128)
    dudt = np.zeros([nperiod,nsample_raw], dtype=np.complex128)
    f = np.zeros([nperiod,nsample_raw], dtype=np.complex128)
    s_raw = np.zeros(nrec, dtype=np.complex128)

    #= Split raw data into ensembles =
    #for isample in range(nsample_raw):
    #    for iperiod in range(nperiod):
    #        irec = ista + nshift * isample + iperiod
    #        u[   iperiod,isample] =    u_raw[irec]
    #        dudt[iperiod,isample] = dudt_raw[irec]
    #        f[   iperiod,isample] =    f_raw[irec]
    #
    for iperiod in range(nperiod):
        u[   iperiod,:] =    u_raw[ista+iperiod:ista+iperiod+nshift*(nsample_raw-1)+1:nshift]
        dudt[iperiod,:] = dudt_raw[ista+iperiod:ista+iperiod+nshift*(nsample_raw-1)+1:nshift]
        f[   iperiod,:] =    f_raw[ista+iperiod:ista+iperiod+nshift*(nsample_raw-1)+1:nshift]
    omega, memoryf, s, r, uu, ududt, fdudt, rr, rdudt, ru, fu, ff = mzprojection_ensemble_of_time_series(nsample_raw, nperiod, delta_t, u, dudt, f)

    #= Evaluate uncorrelated term r(t) as a residual =
    #- 2nd-order integration for memory term s(t) -
    s_raw[0:ista] = 0.0
    wu = np.zeros(nrec-ista+nperiod-2, dtype=np.complex128)
    wu[-nrec+ista:] = np.array(u_raw[ista:nrec])
    wkmemf = np.array(memoryf[0:nperiod])
    wkmemf[0] = 0.5*memoryf[0]
    wkmemf[nperiod-1] = 0.5*memoryf[nperiod-1]
    s_raw[ista+1:nrec] = - np.convolve(wu,wkmemf,mode="valid") * delta_t
    s_raw[ista+1:ista+nperiod] = s_raw[ista+1:ista+nperiod] - memoryf[1:nperiod]*u_raw[ista]*0.5*delta_t
    r_raw = f_raw[:] - omega*u_raw[:] - s_raw[:]
    r_raw[0:ista] = 0.0
    #-

    return omega, memoryf, s_raw, r_raw, uu, ududt, fdudt, rr, rdudt, ru, fu, ff



def memoryf_get_fitting_coef(delta_t, memoryf, order=0, t_range=None):
    '''
    Fitting memory function Gamma(t) = \sum_l=0^order c_l/l!*(t/tau)**l*exp(-t/tau)
    where c_l and tau are complex numbers.
    
    Parameters
    ----------
    delta_t : float
        Time step size
    memoryf : Numpy array
        Memory function Gamma(t)
    order : int, optional
        Order of fitting. Default order=0.
    t_range : int, optional
        Fitting for Gamma[0:trange]. Default t_range=len(memoryf).
        
    Returns
    -------
    tau : complex
        Fitting memory time scale
    cl(order+1) : Numpy array, dtype=np.complex128
        Fitting coefficient
    '''
    import numpy as np
    from scipy import optimize

    # Fitting function
    def memoryf_function(param, x):
        tauinv = param[0]+1j*param[1]
        cl = param[2:].view(np.complex128)
        for i in range(len(cl)):
            if i == 0:
                wf = np.exp(-x*tauinv)
                func = cl[0] * wf
            else:
                wf = wf * (x*tauinv) / i
                func = func + cl[i] * wf
        return func
    
    # Error
    def memoryf_error(param, x, y):
        return np.abs(y - memoryf_function(param, x))

    # Initial guess
    #param = np.array([1+0j for i in range(order+2)]).view(np.float64)
    param = np.ones(1+order+1, dtype=np.complex128)
    param[0] = 4/(t_range*delta_t)
    param[1] = 1.0*memoryf[0]
    param = param.view(np.float64)
    
    # Nonlinear regression by Levenberg-Marquardt algorithm
    if t_range is None:
        t_range = len(memoryf)
    else:
        t_range = min(t_range,len(memoryf))
    t_cor = delta_t*np.arange(t_range)
    param, cov = optimize.leastsq(memoryf_error, param, args=(t_cor,memoryf[0:t_range]))
    tau=1/(param[0]+1j*param[1])
    cl=param[2:].view(np.complex128)
    
    return tau, cl


def rr_get_fitting_coef(delta_t, rr, t_range=None):
    '''
    Fitting auto-correlation of the uncorrelated term
    <r(t)r(0)^*> = Re[simga**2/theta]*exp(-t/theta)
    where sigma is real and theta is complex number.
    
    Parameters
    ----------
    delta_t : float
        Time step size
    rr : Numpy array
        <r(t)r(0)^*>
    t_range : int, optional
        Fitting for rr[0:trange]. Default t_range=len(rr).
        
    Returns
    -------
    theta : complex
        Fitting <r(t)r(0)^*> time scale
    sigma : float
        Fitting <r(t)r(0)^*> amplitude
    '''
    import numpy as np
    from scipy import optimize

    # Analyzed time range
    if t_range is None:
        t_range = len(rr)
    else:
        t_range = min(t_range,len(rr))
    t_cor = delta_t*np.arange(t_range)
    
    # Fitting function
    def rr_function(param, x):
        sigma2 = param[0]
        thetainv = param[1]+1j*param[2]
        func = (sigma2*thetainv).real * np.exp(-x*thetainv)
        return func
    
    # Error
    def rr_error(param, x, y):
        return np.abs(y - rr_function(param, x))

    # Initial guess
    #param = np.array([1,1,1])
    param = np.array([np.abs(rr[0])*(t_range*delta_t)/4, 4/(t_range*delta_t), 0])

    # Nonlinear regression by Levenberg-Marquardt algorithm
    param, cov = optimize.leastsq(rr_error, param, args=(t_cor,rr[0:t_range]))
    sigma=np.sqrt(param[0])
    theta = 1/(param[1]+1j*param[2])
    
    return theta, sigma


def memoryf_fitted(tau, cl, x):
    '''
    Gamma(t) = \sum_l=0^order c_l/l!*(t/tau)**l*exp(-t/tau)
    '''
    import numpy as np
    for i in range(len(cl)):
        if i == 0:
            wf = np.exp(-x/tau)
            func = cl[0] * wf
        else:
            wf = wf * (x/tau) / i
            func = func + cl[i] * wf
        return func

def rr_fitted(theta, sigma, x):
    '''
    <r(t)r(0)^*> = Re[simga**2/theta]*exp(-t/theta)
    '''
    import numpy as np
    func = (sigma**2/theta).real * np.exp(-x/theta)
    return func

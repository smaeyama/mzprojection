#!/usr/bin/env python

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
    wu = np.array(u_raw[ista-nperiod+2:nrec])
    wu[0:nperiod-1] = 0.0
    wkmemf = memoryf[0:nperiod]
    wkmemf[0] = 0.5*memoryf[0]
    wkmemf[nperiod-1] = 0.5*memoryf[nperiod-1]
    s_raw[ista+1:nrec] = - np.convolve(wu,wkmemf,mode="valid") * delta_t
    s_raw[ista+1:ista+nperiod] = s_raw[ista+1:ista+nperiod] - memoryf[1:nperiod]*u_raw[ista]*0.5*delta_t
    r_raw = f_raw[:] - omega*u_raw[:] - s_raw[:]
    r_raw[0:ista] = 0.0
    #-

    return omega, memoryf, s_raw, r_raw, uu, ududt, fdudt, rr, rdudt, ru, fu, ff


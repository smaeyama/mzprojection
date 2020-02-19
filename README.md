  mzprojection
=================

Projection operator method for statistical data analysis

### Overview ###

  The Mori-Zwanzig projection operator method splits ensembles of the analyzed time-series data $f(t)^i$ into correlated and uncorrelated parts with regard to the variable of interests $u(t)^i$.


### Contents ###

    fortran/ - Fortran source code. See README.txt in detail  
    python/ - Python source code. See README.txt in detail  
    sample_data/ - A sample of time-series data and its projected results  
    QUICK_START.txt - Simple explanation on how to run this source code  


### How to use ###

  User prepares ensembles of the analyzed time-series data
  f(0:nsample-1,0:nperiod-1), and the variable of interest
  u(0:nsample-1,0:nperiod-1), and its time derivative
  dudt(0:nsample-1,0:nperiod-1).

  *** Fortran ***
    use mzprojection, only : mzprojection_ensemble_of_time_series

    call mzprojection_ensemble_of_time_series(      &
             nsample, nperiod, delta_t,             & ! INPUT
             u, dudt, f,                            & ! INPUT
             omega, memoryf, s, r,                  & ! OUTPUT
             uu, ududt, fdudt, rr, rdudt, ru, fu, ff) ! OUTPUT


### Parameters ###

  != INPUT =
  integer         :: nsample ! # of samples for ensemble average
  integer         :: nperiod ! Length of a sample
  real(kind=8)    :: delta_t ! Time step size
  complex(kind=8) ::    u(0:nsample-1,0:nperiod-1) ! Variable u(t)
  complex(kind=8) :: dudt(0:nsample-1,0:nperiod-1) ! = du/dt
  complex(kind=8) ::    f(0:nsample-1,0:nperiod-1) ! Analyzed f(t)

  != OUTPUT =
  complex(kind=8) :: omega                      ! Markov coefficient Omega
  complex(kind=8) :: memoryf(0:nperiod-1)       ! Memory function Gamma(t)
  complex(kind=8) :: s(0:nsample-1,0:nperiod-1) ! Memory term
  complex(kind=8) :: r(0:nsample-1,0:nperiod-1) ! Uncorrelated term r(t)

  != OUTPUT for check =
  complex(kind=8) ::    uu(0:nperiod-1) ! Correlation <u(t)u>
  complex(kind=8) :: ududt(0:nperiod-1) ! Correlation <u(t)du/dt>
  complex(kind=8) :: fdudt(0:nperiod-1) ! Correlation <f(t)du/dt>
  complex(kind=8) ::    rr(0:nperiod-1) ! Correlation <r(t)r>
  complex(kind=8) :: rdudt(0:nperiod-1) ! Correlation <r(t)du/dt>
  complex(kind=8) ::    ru(0:nperiod-1) ! Correlation <r(t)u>
  complex(kind=8) ::    fu(0:nperiod-1) ! Correlation <f(t)u>
  complex(kind=8) ::    ff(0:nperiod-1) ! Correlation <f(t)f>


### Theoretical description ###

  Mori-Zwanzig projection of the analyzed time-series data f(t) onto the 
  variable of interest u(t) provides a generalized Langevin form,

    f(t) = omega*u(t) + s(t) + r(t)

  where s(t) = - \int_0^t memoryf(t)*u(t-v) dv is the memory term.

  omega, memoryf(t), r(t) are called as the Markov coefficient, memory function,
  uncorrelated term (or so-called noise term), respectively.

  They are defined by

    omega = <f*u^*>.<u*u^*>^-1
    memoryf(t) = <r(t)*du/dt^*>.<u*u^*>^-1
    <r(t)*u^*> = 0

  where ( )^* is the complex conjugate, ( )^-1 is the inverse of matrix, 
  ( ).( ) is the matrix product, < > is the ensemble average.  f, u, dudt 
  are shorthand of initial values, i.e., f=f(0), u=u(0), dudt=dudt(0).

  Therefore, the 1st and 2nd term (omega*u(t) and s(t)) are correlated 
  with u(t), while the 3rd (r(t)) is not.


### Numerical implementation ###

  Let assume ensemble of time-series data f(t)^i and u(t)^i are prepared.

         t =      0, delta_t, 2*delta_t, 3*delta_t, ..., (nperiod-1)*delta_t
    u(t)^i = u(0)^i,  u(1)^i,    u(2)^i,    u(3)^i, ..., u(nperiod-1)^i
    f(t)^i = f(0)^i,  f(1)^i,    f(2)^i,    f(3)^i, ..., f(nperiod-1)^i

  where i = 0, 1, 2, ..., nsample-1.

  Ensemble average is defined as summation over i, e.g.,

    <f*u^*> = (1/nsample) * sum_{i=0}^{nsample-1} f^i*(u^i)^*.

  Then, omega is obtained from

    omega = <f*u^*>.<u*u^*>^-1.

  Multiplying du^*/dt and taking ensemble average, one obtains memory equation,

    memoryf(t) = 1/<u*u^*>*[<f(t)*du/dt^*> - omega*<u(t)*du/dt^*> 
                           + \int_0^t memoryf(v)*<u(t-v)*du/dt^*> dv].

  Since all ensemble averaged quantities are evaluated, one can construct
  memoryf(t) from this equation.

  At t=0, memoryf(0) = 1/<u*u^*>*[<f*du/dt^*> - omega*<u*du/dt^*>].

  At t=delta_t, time-integration is discretized by 2nd-order trapezoid rule,

    \int_0^delta_t memoryf(v)*<u(delta_t-v)*du/dt^*> dv
      = memoryf(0)*<u(delta_t)*du/dt^*> * 0.5*delta_t
      + memoryf(delta_t)*<u(0)*du/dt^*> * 0.5*delta_t

  and then, 

    memoryf(delta_t) = [<f(delta_t)*du/dt^*> - omega*<u(delta_t)*du/dt^*> 
                       + memoryf(0)*<u(delta_t)*du/dt^*> * 0.5*delta_t]
                     /(<u*u^*>)   
                     /(1.0 - <u(0)*du/dt^*> * 0.5*delta_t).

  In the same way, memoryf(t) is calculated from memoryf(t-delta_t).

  The memory term is calculated by the convolution of memoryf(t) and u(t),
  while integration range is reduced from 0<v<t to t-(nperiod-1)*delta_t<v<t,

    s(t) = - \int_{t-(nperiod-1)*delta_t}^t memoryf(t)*u(t-v) dv.

  The uncorrelated term is calculated as a residual,

    r(t) = f(t) - omega*u(t) - s(t).


### License and Copyright ###

  Copyright (c) Shinya Maeyama, Nagoya University, since 2019.

  This is a free software WITHOUT WARRANTY OF ANY KIND. You can use,
  redistribute, and modify it.

  We politely request that you cite the original paper when you use this
  module in publications:

  Reference:
     Shinya Maeyama and Tomo-Hiko Watanabe,
    "Extracting and Modeling the Effects of Small-Scale Fluctuations on
     Large-Scale Fluctuations by Mori-Zwanzig Projection operator method", 
     J. Phys. Soc. Jpn. 89, 024401 (2020).

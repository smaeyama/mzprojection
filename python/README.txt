=================

  mzprojection

=================

### Overview ###

  This is explanation of a Python3 module of Mori-Zwanzig projection, which
  split ensembles of the analyzed time-series data f(t)^i into correlated and 
  uncorrelated parts with regard to the variable of interests u(t)^i.



### How to use ###

  User prepares ensembles of the analyzed time-series data
  f[nperiod,nsample], and the variable of interest
  u[nperiod,nsample], and its time derivative
  dudt[nperiod,nsample].

  *** Python ***
    from mzprojection import mzprojection_ensemble_of_time_series

    omega, memoryf, s, r, uu, ududt, fdudt, rr, rdudt, ru, fu, ff = \
    mzprojection_ensemble_of_time_series(nsample, nperiod, delta_t, u, dudt, f)


### Parameters ###

  != INPUT =
  integer          :: nsample ! # of samples for ensemble average
  integer          :: nperiod ! Length of a sample
  numpy.float64    :: delta_t ! Time step size
  numpy.complex128 ::    u[nperiod,nsample] ! Variable u(t)
  numpy.complex128 :: dudt[nperiod,nsample] ! = du/dt
  numpy.complex128 ::    f[nperiod,nsample] ! Analyzed f(t)

  != OUTPUT =
  numpy.complex128 :: omega              ! Markov coefficient Omega
  numpy.complex128 :: memoryf[nperiod]   ! Memory function Gamma(t)
  numpy.complex128 :: s[nperiod,nsample] ! Memory term
  numpy.complex128 :: r[nperiod,nsample] ! Uncorrelated term r(t)

  != OUTPUT for check =
  numpy.complex128 ::    uu[nperiod] ! Correlation <u(t)u>
  numpy.complex128 :: ududt[nperiod] ! Correlation <u(t)du/dt>
  numpy.complex128 :: fdudt[nperiod] ! Correlation <f(t)du/dt>
  numpy.complex128 ::    rr[nperiod] ! Correlation <r(t)r>
  numpy.complex128 :: rdudt[nperiod] ! Correlation <r(t)du/dt>
  numpy.complex128 ::    ru[nperiod] ! Correlation <r(t)u>
  numpy.complex128 ::    fu[nperiod] ! Correlation <f(t)u>
  numpy.complex128 ::    ff[nperiod] ! Correlation <f(t)f>


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



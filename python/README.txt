=================

  mzprojection

=================

### Overview ###

  This is explanation of a Python3 module of Mori-Zwanzig projection, which is
  a statistical data analysis tool for an ensemble time-series data set.
  One splits the analyzed time-series data f_i(t) into correlated and uncorrelated parts
  with regard to the variable of interests u_j(t).

  In Oct. 2021, the function "mzprojection_multivariate" has been developed 
  for the projection on multiple u_j(t).
  The previous 1-to-1 projection will be obsolete, because it is covered by multivariate one.


### How to use ###

  1.  User prepares an ensemble data set, namely a number of samples of time-series data,
          delta_t               : Time step size
          f[nsample,nperiod,nf] : Response variables
          u[nsample,nperiod,nu] : Explanatory variables
          dudt0[nsample,nu]     : = du/dt at t=0 of each samples
      See also the help on mzprojection_multivariate.

      If one would like to create these samples from a long time-series data,
      the following function could be useful,
      ```
      from mzprojection import split_long_time_series
      u = split_long_time_series(u_raw, ista=100, nperiod=200, nshift=10)
      ```

  2.  Calculate the Markov coefficient matrix omega and memory function memoryf from the data set,
      ```
      from mzprojection import mzprojection_multivariate
      omega, memoryf = mzprojection_multivariate(delta_t, u, dudt0, f)
      ```
      If one would like to evaluate the memory term s_i(t) and the uncorrelated term r_i(t),
      ```
      omega, memoryf, s, r = mzprojection_multivariate(delta_t, u, dudt0, f, flag_terms=True)
      ```

  3.  That's all!

      If one would like to calculate various correlations, just take average over samples.
      Since this would often happens, there is a function (just by numpy.tensordot),
      ```
      from mzprojection import calc_correlation
      u0 = u[:,0,:]
      fu = calc_correlation(f,u0)
      ```
      Then one gets the cross-correlation between f_i(t) and u_j(0) as fu[nperiod,nf,nu].


### Help on mzprojection_multivariate ###

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
          nperiod is the number of time steps of each sample.
          nu is the number of degree of freedom of explanatory variable u_j (j=0,1,...,nu-1).
      dudt0[nsample,nu] : Numpy array (float64 or complex128)
          = du/dt(t=0)
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


### Theoretical description ###

  Mori-Zwanzig projection of the analyzed time-series data f(t) onto the 
  variable of interest u(t) provides a generalized Langevin form,

    f_i(t) = Omega_{ij}u_j(t) + s_i(t) + r_i(t)

  where s_i(t) = - \int_0^t Gamma_{ij}(t)*u_j(t-v) dv is the memory term.
  Take summation over the repeated index j.

  Omega, Gamma(t), r(t) are called as the Markov coefficient matrix,
  memory function matrix, uncorrelated term (or so-called orthogonal 
  dynamics or noise term), respectively.

  They are defined by

    Omega = <f*u^*>.<u*u^*>^-1
    Gamma(t) = <r(t)*du/dt^*>.<u*u^*>^-1
    <r(t)*u^*> = 0

  where ( )^* is the complex conjugate, ( )^-1 is the inverse of matrix, 
  ( ).( ) is the matrix product, < > is the ensemble average.  f, u, dudt 
  are shorthand of initial values, i.e., f=f(0), u=u(0), dudt=dudt(0).

  Therefore, the 1st and 2nd term (Omega.u(t) and s(t)) are correlated 
  with u(t), while the 3rd (r(t)) is not.


### Numerical implementation ###

  Let us assume an ensemble of time-series data f(t)^l and u(t)^l are prepared.

         t =      0, delta_t, 2*delta_t, 3*delta_t, ..., (nperiod-1)*delta_t
    u(t)^l = u(0)^l,  u(1)^l,    u(2)^l,    u(3)^l, ..., u(nperiod-1)^l
    f(t)^l = f(0)^l,  f(1)^l,    f(2)^l,    f(3)^l, ..., f(nperiod-1)^l

  where l = 0, 1, 2, ..., nsample-1.
  Note that u={u_0, u_1, ..., u_nu-1} and f={f_0, f_1, ..., f_nf-1} are vectors.

  Ensemble average is defined as summation over l, e.g.,

    <f_i*u_j^*> = (1/nsample) * sum_{l=0}^{nsample-1} f_i^l*(u_j^l)^*.

  Then, Omega is obtained from

    Omega = <f*u^*>.<u*u^*>^-1.

  Multiplying du^*/dt on the generalized langevin form and taking ensemble average,
  one obtains the memory equation,

    Gamma(t) = F(t) + \int_0^t Gamma(v).G(t-v) dv

  where

    F(t) = [<f(t)*du/dt^*> - Omega.<u(t)*du/dt^*>].<u*u^*>^-1
    G(t) = <u(t)*du/dt^*>.<u*u^*>^-1

  Since all ensemble averaged quantities are evaluated, one can construct Gamma(t).

  At t=0, Gamma(0) = F(0)

  At t=delta_t, time integration is discretized by 2nd-order trapezoid rule,

    \int_0^delta_t Gamma(v).G(delta_t-v) dv
      = Gamma(0).G(delta_t) * 0.5*delta_t
      + Gamma(delta_t).G(0) * 0.5*delta_t

  and then, 

    Gamma(delta_t) = [F(delta_t) + 0.5*delta_t*Gamma(0).G(delta_t)].[I - 0.5*delta_t*G(0)]^-1

  where I is the identity matrix.

  In the same way, Gamma(t) is calculated by using Gamma(0) to Gamma(t-delta_t).

  The memory term is calculated by the convolution of Gamma(t) and u(t),
  while integration range is reduced from 0<v<t to t-(nperiod-1)*delta_t<v<t,

    s(t) = - \int_{t-(nperiod-1)*delta_t}^t Gamma(t).u(t-v) dv.

  The uncorrelated term is calculated as a residual,

    r(t) = f(t) - Omega.u(t) - s(t).


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



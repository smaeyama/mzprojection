  mzprojection
=================

Projection operator method for statistical data analysis (Fortran90 or Python3)  
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/smaeyama/mzprojection.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/smaeyama/mzprojection/context:python)


### Overview ###

The Mori-Zwanzig projection operator method splits ensembles of the analyzed time-series data <img src="https://latex.codecogs.com/gif.latex?f(t)^i" /> into correlated and uncorrelated parts with regard to the variable of interests <img src="https://latex.codecogs.com/gif.latex?u(t)^i" />.


### Contents ###

    fortran/ - Fortran90 source code. See README.txt in detail  
    python/ - Python3 source code. See README.txt in detail  
    sample_data/ - A sample of time-series data and its projected results  
    QUICK_START.txt - Simple explanation on how to run this source code  
    
    
### Usage (Python3) ###
**mzprojection** requires external packages: numpy, scipy.

(i) Input data is ensembles of the analyzed time-series data <img src="https://latex.codecogs.com/gif.latex?f(t)^i" alt="f(t)^i" /> and the variable of interest <img src="https://latex.codecogs.com/gif.latex?u(t)^i" alt="u(t)^i" />, <img src="https://latex.codecogs.com/gif.latex?du(t)^i/dt" alt="du(t)^i/dt" /> in a prescribed format (`nperiod` points in time range, and `nsample` points in ensambles, namely `u[0:nperiod,0:nsample], dudt[0:nperiod,0:nsample], f[0:nperiod,0:nsample]`, and time step size `delta_t`).  

(ii) ***mzprojection_ensemble_of_time_series*** calculates the Mori-Zwanzig projection of <img src="https://latex.codecogs.com/gif.latex?f(t)^i" alt="f(t)^i" /> on <img src="https://latex.codecogs.com/gif.latex?u(t)^i" alt="u(t)^i" /> as,  
  <img src="https://latex.codecogs.com/gif.latex?f(t)=\Omega&space;u(t)+s(t)+r(t)," alt="f(t)=\Omega u(t)+s(t)+r(t)," />  
  <img src="https://latex.codecogs.com/gif.latex?s(t)=-\int_0^t&space;\Gamma(t)&space;u(t-v)dv." alt="s(t)=-\int_0^t \Gamma(t) u(t-v)dv." />  
The Markov coefficient <img src="https://latex.codecogs.com/gif.latex?\Omega" alt="\Omega" />, the memory function <img src="https://latex.codecogs.com/gif.latex?\Gamma(t)" alt="\Gamma(t)" /> and the uncorrelated term <img src="https://latex.codecogs.com/gif.latex?r(t)" alt="r(t)" /> are obtained as outputs.
(Some correlations, e.g., <img src="https://latex.codecogs.com/gif.latex?\langle&space;r(t)u&space;\rangle" alt="<r(t)u>" />, are also obtained to check the result.)
```
from mzprojection import mzprojection_ensemble_of_time_series

omega, memoryf, s, r, uu, ududt, fdudt, rr, rdudt, ru, fu, ff = \
    mzprojection_ensemble_of_time_series(nsample, nperiod, delta_t, u, dudt, f)
```

See also `python/Demo_Jan2021.ipynb`, which clearly shows examples of usage including output figures.


### Reference ###

[Shinya Maeyama and Tomo-Hiko Watanabe, "Extracting and Modeling the Effects of Small-Scale Fluctuations on Large-Scale Fluctuations by Mori-Zwanzig Projection operator method", J. Phys. Soc. Jpn. 89, 024401 (2020).](https://doi.org/10.7566/JPSJ.89.024401)  
[![doi](https://img.shields.io/badge/doi-10.7566/JPSJ.89.024401-5077AB.svg)](https://doi.org/10.7566/JPSJ.89.024401)


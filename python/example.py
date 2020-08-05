#!/usr/bin/env python


if __name__ == '__main__':
    import numpy as np
    from mzprojection import mzprojection_ensemble_of_time_series, mzprojection_long_time_series

    #= Read sample data =
    indata = np.loadtxt('../sample_data/sample_time_series.dat')
    time     = indata[:,0]                      # Time
    u_raw    = indata[:,1] + 1.0j * indata[:,2] # Variable of interest u(t)
    dudt_raw = indata[:,3] + 1.0j * indata[:,4] # = du/dt
    f_raw    = indata[:,5] + 1.0j * indata[:,6] # Analyzed data f(t)

    #= Parameters for ensemble average =
    nrec    = len(time)         # Total record number                
    ista    = 2000              # Start record number for sampling   
    nperiod = 500               # Length of a sample                 
    nshift  = 1                 # Length of time shift while sampling
    delta_t = time[1] - time[0] # Time step size                     

    #= Mori-Zwanzig projection =
    #
    #  f(t) = Omega*u(t) - int_0^t Gamma(s)*u(t-s) ds + r(t)
    #
    omega, memoryf, s_raw, r_raw, uu, ududt, fdudt, rr, rdudt, ru, fu, ff = mzprojection_long_time_series(nrec, ista, nperiod, nshift, delta_t, u_raw, dudt_raw, f_raw)

    ##= Mori-Zwanzig projection =
    ##
    ##  f(t) = Omega*u(t) - int_0^t Gamma(s)*u(t-s) ds + r(t)
    ##
    #nsample = int((nrec-ista-nperiod)/nshift) + 1
    #u    = np.zeros([nperiod,nsample], dtype=np.complex128)
    #dudt = np.zeros([nperiod,nsample], dtype=np.complex128)
    #f    = np.zeros([nperiod,nsample], dtype=np.complex128)
    #for iperiod in range(nperiod):
    #    u[   iperiod,:] =    u_raw[ista+iperiod:ista+iperiod+nshift*(nsample-1)+1:nshift]
    #    dudt[iperiod,:] = dudt_raw[ista+iperiod:ista+iperiod+nshift*(nsample-1)+1:nshift]
    #    f[   iperiod,:] =    f_raw[ista+iperiod:ista+iperiod+nshift*(nsample-1)+1:nshift]
    #omega, memoryf, s, r, uu, ududt, fdudt, rr, rdudt, ru, fu, ff = mzprojection_ensemble_of_time_series(nsample, nperiod, delta_t, u, dudt, f)

    #= Output results =
    outdata = np.real(np.vstack([time, u_raw.real, u_raw.imag, dudt_raw.real, dudt_raw.imag, f_raw.real, f_raw.imag, (omega*u_raw).real, (omega*u_raw).imag, s_raw.real, s_raw.imag, r_raw.real, r_raw.imag]))
    outdata = outdata.transpose()
    np.savetxt('out_timeevolution.dat', outdata, fmt='%17.7e')
    outdata = np.real(np.vstack([time[0:nperiod], uu.real, uu.imag, fu.real, fu.imag, ududt.real, ududt.imag, fdudt.real, fdudt.imag]))
    outdata = outdata.transpose()
    np.savetxt('out_correlation.dat', outdata, fmt='%17.7e')
    outdata = np.real(np.vstack([time[0:nperiod], memoryf.real, memoryf.imag, (rr/uu[0]).real, (rr/uu[0]).imag, (rdudt/uu[0]).real, (rdudt/uu[0]).imag]))
    outdata = outdata.transpose()
    np.savetxt('out_check_memoryfunc.dat', outdata, fmt='%17.7e')
    outdata = np.real(np.vstack([time[0:nperiod], rr.real, rr.imag, (ru/np.sqrt(np.abs(rr[0]*uu[0]))).real, (ru/np.sqrt(np.abs(rr[0]*uu[0]))).imag, (fu/np.sqrt(np.abs(ff[0]*uu[0]))).real, (fu/np.sqrt(np.abs(ff[0]*uu[0]))).imag]))
    outdata = outdata.transpose()
    np.savetxt('out_check_r.dat', outdata, fmt='%17.7e')
    print(omega)


MODULE mzprojection
!-------------------------------------------------------------------------------
!
!  Module of Mori-Zwanzig Projection operator method for data analysis
!
!    f(t) = Omega*u(t) - int_0^t Gamma(s)*u(t-s) ds + r(t)
!
!  Reference: S. Maeyama, T.-H. Watanabe, to be published JPSJ (2020).
!
!-------------------------------------------------------------------------------
  implicit none
  private

  public :: mzprojection_ensemble_of_time_series, &
            mzprojection_long_time_series


 CONTAINS


SUBROUTINE mzprojection_ensemble_of_time_series(  &
               nsample, nperiod, delta_t,         &
               u, dudt, f,                        &
               omega, memoryf, s, r,              &
               uu, ududt, fdudt, rr, rdudt, ru, fu, ff)
!-------------------------------------------------------------------------------
!
!  Evaluate projection of f(t) on u(t)
!
!    *User prepares ensembles of time-series data u(t)^i, du/dt(t)^i, f(t)^i
!
!-------------------------------------------------------------------------------
  implicit none
  != INPUT =
  integer,         intent(in)  :: nsample ! # of samples for ensemble average
  integer,         intent(in)  :: nperiod ! Length of a sample
  real(kind=8),    intent(in)  :: delta_t ! Time step size
  complex(kind=8), intent(in)  ::    u(0:nsample-1,0:nperiod-1) ! Variable u(t)
  complex(kind=8), intent(in)  :: dudt(0:nsample-1,0:nperiod-1) ! = du/dt
  complex(kind=8), intent(in)  ::    f(0:nsample-1,0:nperiod-1) ! Analyzed f(t)
  != OUTPUT =
  complex(kind=8), intent(out) :: omega                      ! Markov coefficient Omega
  complex(kind=8), intent(out) :: memoryf(0:nperiod-1)       ! Memory function Gamma(t)
  complex(kind=8), intent(out) :: s(0:nsample-1,0:nperiod-1) ! Memory term
  complex(kind=8), intent(out) :: r(0:nsample-1,0:nperiod-1) ! Uncorrelated term r(t)
  != OUTPUT for check =
  complex(kind=8), intent(out) ::    uu(0:nperiod-1) ! Correlation <u(t)u>
  complex(kind=8), intent(out) :: ududt(0:nperiod-1) ! Correlation <u(t)du/dt>
  complex(kind=8), intent(out) :: fdudt(0:nperiod-1) ! Correlation <f(t)du/dt>
  complex(kind=8), intent(out) ::    rr(0:nperiod-1) ! Correlation <r(t)r>
  complex(kind=8), intent(out) :: rdudt(0:nperiod-1) ! Correlation <r(t)du/dt>
  complex(kind=8), intent(out) ::    ru(0:nperiod-1) ! Correlation <r(t)u>
  complex(kind=8), intent(out) ::    fu(0:nperiod-1) ! Correlation <f(t)u>
  complex(kind=8), intent(out) ::    ff(0:nperiod-1) ! Correlation <f(t)f>

  integer :: iperiod, isample

    != Evaluate Markov coefficient Omega =
    omega = sum(f(0:nsample-1,0)*conjg(u(0:nsample-1,0)))  &
          / sum(u(0:nsample-1,0)*conjg(u(0:nsample-1,0))) 

    != Evaluate memory function Gamma(t) =
    do iperiod = 0, nperiod-1
         uu(iperiod) = sum(u(0:nsample-1,iperiod) * conjg(   u(0:nsample-1,0))) / nsample
      ududt(iperiod) = sum(u(0:nsample-1,iperiod) * conjg(dudt(0:nsample-1,0))) / nsample
      fdudt(iperiod) = sum(f(0:nsample-1,iperiod) * conjg(dudt(0:nsample-1,0))) / nsample
    end do
    !- 2nd-order solver for memory equation -
    memoryf(0) = (fdudt(0) - omega*ududt(0)) / uu(0)
    memoryf(1) = (fdudt(1) - omega*ududt(1) + memoryf(0)*ududt(1)*0.5d0*delta_t) / uu(0)
    memoryf(1) = memoryf(1) / (1.d0 - ududt(0)*0.5d0*delta_t/uu(0))
    do iperiod = 2, nperiod-1
      memoryf(iperiod) = (fdudt(iperiod) - omega*ududt(iperiod)    &
                       + memoryf(0)*ududt(iperiod)*0.5d0*delta_t  &
                       + sum(memoryf(1:iperiod-1)*ududt(iperiod-1:1:-1))*delta_t) / uu(0)
      memoryf(iperiod) = memoryf(iperiod) / (1.d0 - ududt(0)*0.5d0*delta_t/uu(0))
    end do
    !-

    != Evaluate uncorrelated term r(t) as a residual =
    !- 2nd-order integration for memory term s(t) -
    s(:,0) = 0.d0
    r(:,0) = f(:,0) - omega*u(:,0)
    do iperiod = 1, nperiod-1
      do isample = 0, nsample-1
        s(isample,iperiod) = - (sum(memoryf(0:iperiod)*u(isample,iperiod:0:-1))*delta_t  &
                             - memoryf(0)*u(isample,iperiod) *0.5d0*delta_t              &
                             - memoryf(iperiod)*u(isample,0) *0.5d0*delta_t)
        r(isample,iperiod) = f(isample,iperiod) - omega*u(isample,iperiod) - s(isample,iperiod)
      end do
    end do
    !-

    != For check memory function and <r(t)u>=0 =
    do iperiod = 0, nperiod-1
         rr(iperiod) = sum(r(0:nsample-1,iperiod) * conjg(   r(0:nsample-1,0))) / nsample
      rdudt(iperiod) = sum(r(0:nsample-1,iperiod) * conjg(dudt(0:nsample-1,0))) / nsample
         ru(iperiod) = sum(r(0:nsample-1,iperiod) * conjg(   u(0:nsample-1,0))) / nsample
         fu(iperiod) = sum(f(0:nsample-1,iperiod) * conjg(   u(0:nsample-1,0))) / nsample
         ff(iperiod) = sum(f(0:nsample-1,iperiod) * conjg(   f(0:nsample-1,0))) / nsample
    end do

    !%%% NOTE %%%
    !% Since the above s, r is defined from u(isample,:) and f(isample,:),
    !%     s(isample,0) = 0
    !% is always imposed. Then, <s(t)s>=0, because s(0)=0.
    !% However, after a long time, s and r is also expected to be in a statistically
    !% steady state, <s(t)s> = <s(t+x)s(x)> /= 0.
    !%%%%%%%%%%%%%%%

END SUBROUTINE mzprojection_ensemble_of_time_series


SUBROUTINE mzprojection_long_time_series(             &
               nrec, ista, nperiod, nshift, delta_t,  &
               u_raw, dudt_raw, f_raw,                &
               omega, memoryf, s_raw, r_raw,          &
               uu, ududt, fdudt, rr, rdudt, ru, fu, ff)
!-------------------------------------------------------------------------------
!
!  Evaluate projection of f(t) on u(t)
!
!    *User prepares long time-series data u(t), du/dt(t), f(t)
!
!-------------------------------------------------------------------------------
  implicit none
  != INPUT =
  integer,         intent(in)  :: nrec      ! Total record number
  integer,         intent(in)  :: ista      ! Start record number for sampling
  integer,         intent(in)  :: nperiod   ! Length of a sample
  integer,         intent(in)  :: nshift    ! Length of time shift while sampling
  real(kind=8),    intent(in)  :: delta_t   ! Time step size
  complex(kind=8), intent(in)  ::    u_raw(0:nrec-1) ! Variable of interest u(t)
  complex(kind=8), intent(in)  :: dudt_raw(0:nrec-1) ! = du/dt
  complex(kind=8), intent(in)  ::    f_raw(0:nrec-1) ! Analyzed data f(t)
  != OUTPUT =
  complex(kind=8), intent(out) :: omega                ! Markov coefficient Omega
  complex(kind=8), intent(out) :: memoryf(0:nperiod-1) ! Memory function Gamma(t)
  complex(kind=8), intent(out) :: s_raw(0:nrec-1)      ! Memory term
  complex(kind=8), intent(out) :: r_raw(0:nrec-1)      ! Uncorrelated term r(t)
  != OUTPUT for check =
  complex(kind=8), intent(out) ::    uu(0:nperiod-1) ! Correlation <u(t)u>
  complex(kind=8), intent(out) :: ududt(0:nperiod-1) ! Correlation <u(t)du/dt>
  complex(kind=8), intent(out) :: fdudt(0:nperiod-1) ! Correlation <f(t)du/dt>
  complex(kind=8), intent(out) ::    rr(0:nperiod-1) ! Correlation <r(t)r>
  complex(kind=8), intent(out) :: rdudt(0:nperiod-1) ! Correlation <r(t)du/dt>
  complex(kind=8), intent(out) ::    ru(0:nperiod-1) ! Correlation <r(t)u>
  complex(kind=8), intent(out) ::    fu(0:nperiod-1) ! Correlation <f(t)u>
  complex(kind=8), intent(out) ::    ff(0:nperiod-1) ! Correlation <f(t)f>

  integer :: nsample_raw ! Number of samples for ensemble average
  complex(kind=8), dimension(0:(nrec-ista-nperiod)/nshift) ::  &
              u0, dudt0, f0, s0, r0, &! Initial values of sample u(0)
              u,  dudt,  f,  s,  r    ! Temporal valudes of samples u(t)
  integer :: irec, iperiod, isample

    nsample_raw = (nrec-ista-nperiod)/nshift + 1

    != Split raw data into ensembles =
    !!!!nsample_raw = (nrec-ista-nperiod)/nshift
    !!!!do isample = 0, nsample_raw-1
    !!!!  do iperiod = 0, nperiod-1
    !!!!    irec = ista + nshift * isample + iperiod
    !!!!    u(isample,iperiod) = u_raw(irec)
    !!!!    dudt(isample,iperiod) = dudt_raw(irec)
    !!!!    f(isample,iperiod) = f_raw(irec)
    !!!!  end do
    !!!!end do
    iperiod = 0
      do isample = 0, nsample_raw-1
        irec = ista + nshift * isample + iperiod
           u0(isample) = u_raw(irec)
        dudt0(isample) = dudt_raw(irec)
           f0(isample) = f_raw(irec)
      end do

    != Evaluate Markov correlation Omega =
    omega = sum(f0(0:nsample_raw-1)*conjg(u0(0:nsample_raw-1)))  &
          / sum(u0(0:nsample_raw-1)*conjg(u0(0:nsample_raw-1))) 

    != Evaluate memory function Gamma(t) =
    do iperiod = 0, nperiod-1
      do isample = 0, nsample_raw-1
        irec = ista + nshift * isample + iperiod
           u(isample) = u_raw(irec)
        dudt(isample) = dudt_raw(irec)
           f(isample) = f_raw(irec)
      end do
         uu(iperiod) = sum(u(0:nsample_raw-1) * conjg(   u0(0:nsample_raw-1))) / nsample_raw
      ududt(iperiod) = sum(u(0:nsample_raw-1) * conjg(dudt0(0:nsample_raw-1))) / nsample_raw
      fdudt(iperiod) = sum(f(0:nsample_raw-1) * conjg(dudt0(0:nsample_raw-1))) / nsample_raw
      != For check <r(t)u>=0 =
         fu(iperiod) = sum(f(0:nsample_raw-1) * conjg(   u0(0:nsample_raw-1))) / nsample_raw
         ff(iperiod) = sum(f(0:nsample_raw-1) * conjg(   f0(0:nsample_raw-1))) / nsample_raw
    end do
    !- 2nd-order solver for memory equation -
    memoryf(0) = (fdudt(0) - omega*ududt(0)) / uu(0)
    memoryf(1) = (fdudt(1) - omega*ududt(1) + memoryf(0)*ududt(1)*0.5d0*delta_t) / uu(0)
    memoryf(1) = memoryf(1) / (1.d0 - ududt(0)*0.5d0*delta_t/uu(0))
    do iperiod = 2, nperiod-1
      memoryf(iperiod) = (fdudt(iperiod) - omega*ududt(iperiod)    &
                       + memoryf(0)*ududt(iperiod)*0.5d0*delta_t  &
                       + sum(memoryf(1:iperiod-1)*ududt(iperiod-1:1:-1))*delta_t) / uu(0)
      memoryf(iperiod) = memoryf(iperiod) / (1.d0 - ududt(0)*0.5d0*delta_t/uu(0))
    end do
    !-

    != Evaluate uncorrelated term r(t) as a residual =
    s_raw(0:ista-1) = 0.d0
    r_raw(0:ista-1) = 0.d0
    !- 2nd-order integration for memory term s(t) -
    s_raw(ista) = 0.d0
    r_raw(ista) = f_raw(ista) - omega*u_raw(ista)
    do irec = 1, nperiod-1
      s_raw(ista+irec) = - (sum(memoryf(0:irec)*u_raw(ista+irec:ista:-1)) *delta_t  &
                         - memoryf(0)*u_raw(ista+irec) *0.5d0*delta_t            &
                         - memoryf(irec)*u_raw(ista) *0.5d0*delta_t)
      r_raw(ista+irec) = f_raw(ista+irec) - omega*u_raw(ista+irec) - s_raw(ista+irec)
    end do
    do irec = ista+nperiod, nrec-1
      s_raw(irec) = - (sum(memoryf(0:nperiod-1)*u_raw(irec:irec-nperiod+1:-1)) *delta_t  &
                         - memoryf(0)*u_raw(irec) *0.5d0*delta_t                         &
                         - memoryf(nperiod-1)*u_raw(irec-nperiod+1) *0.5d0*delta_t)
      r_raw(irec) = f_raw(irec) - omega*u_raw(irec) - s_raw(irec)
    end do
    !-

    != For check memory function =
    !- 2nd-order integration for memory term s -
    iperiod = 0
      do isample = 0, nsample_raw-1
        irec = ista + nshift * isample + iperiod
        s0(isample) = (0.d0, 0.d0)
        r0(isample) = f_raw(irec) - omega*u_raw(irec)
      end do
         rr(iperiod) = sum(r0(0:nsample_raw-1) * conjg(   r0(0:nsample_raw-1))) / nsample_raw
      rdudt(iperiod) = sum(r0(0:nsample_raw-1) * conjg(dudt0(0:nsample_raw-1))) / nsample_raw
         ru(iperiod) = sum(r0(0:nsample_raw-1) * conjg(   u0(0:nsample_raw-1))) / nsample_raw
    do iperiod = 1, nperiod-1
      do isample = 0, nsample_raw-1
        irec = ista + nshift * isample + iperiod
        s(isample) = - (sum(memoryf(0:iperiod)*u_raw(irec:irec-iperiod:-1))*delta_t  &
                     - memoryf(0)*u_raw(irec) *0.5d0*delta_t              &
                     - memoryf(iperiod)*u_raw(irec-iperiod) *0.5d0*delta_t)
        r(isample) = f_raw(irec) - omega*u_raw(irec) - s(isample)
      end do
         rr(iperiod) = sum(r(0:nsample_raw-1) * conjg(   r0(0:nsample_raw-1))) / nsample_raw
      rdudt(iperiod) = sum(r(0:nsample_raw-1) * conjg(dudt0(0:nsample_raw-1))) / nsample_raw
         ru(iperiod) = sum(r(0:nsample_raw-1) * conjg(   u0(0:nsample_raw-1))) / nsample_raw
    end do
    !-

END SUBROUTINE mzprojection_long_time_series


END MODULE mzprojection

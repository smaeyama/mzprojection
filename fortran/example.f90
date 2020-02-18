PROGRAM test
  use mzprojection, only : mzprojection_ensemble_of_time_series, &
                           mzprojection_long_time_series
  implicit none
  integer, parameter ::    nrec = 10001 ! Total record number
  integer, parameter ::    ista = 2000  ! Start record number for sampling
  integer, parameter :: nperiod = 500   ! Length of a sample
  integer, parameter ::  nshift = 1     ! Length of time shift while sampling
  real(kind=8) :: delta_t               ! Time step size

  real(kind=8) :: time(0:nrec-1)
  complex(kind=8), dimension(0:nrec-1) :: u_raw, dudt_raw, f_raw, s_raw, r_raw
  complex(kind=8) :: omega, memoryf(0:nperiod-1)
  complex(kind=8), dimension(0:nperiod-1) :: uu, ududt, fdudt, rr, rdudt, ru, fu, ff
  real(kind=8) :: temp(1:7)
  complex(kind=8), parameter :: ci = (0.d0, 1.d0)
  integer :: irec, iperiod
  !integer, parameter :: nsample = (nrec-ista-nperiod)/nshift + 1
  !complex(kind=8), dimension(0:nsample-1,0:nperiod-1) :: u, dudt, f, s, r
  !integer :: isample

    != Read sample data =
    open(10,file="../sample_data/sample_time_series.dat",status="old",action="read")
      read(10,*)
      read(10,*)
      read(10,*)
      read(10,*)
      read(10,*)
      do irec = 0, nrec-1
        read(10,"(99g17.7e3)") temp(1:7)
            time(irec) = temp(1)
           u_raw(irec) = temp(2) + ci * temp(3)
        dudt_raw(irec) = temp(4) + ci * temp(5)
           f_raw(irec) = temp(6) + ci * temp(7)
      end do
    close(10)

    != Evaluate Mori-Zwanzig projection of f(t) on u(t) = 
    delta_t = time(1) - time(0)
    call mzprojection_long_time_series(             &
             nrec, ista, nperiod, nshift, delta_t,  &
             u_raw, dudt_raw, f_raw,                &
             omega, memoryf, s_raw, r_raw,          &
             uu, ududt, fdudt, rr, rdudt, ru, fu, ff)

    !!= Evaluate Mori-Zwanzig projection of f(t) on u(t) = 
    !delta_t = time(1) - time(0)
    !do iperiod = 0, nperiod-1
    !  do isample = 0, nsample-1
    !    irec = ista + nshift * isample + iperiod
    !    u(isample,iperiod) = u_raw(irec)
    !    dudt(isample,iperiod) = dudt_raw(irec)
    !    f(isample,iperiod) = f_raw(irec)
    !  end do
    !end do
    !call mzprojection_ensemble_of_time_series(      &
    !         nsample, nperiod, delta_t,             &
    !         u, dudt, f,                            &
    !         omega, memoryf, s, r,                  &
    !         uu, ududt, fdudt, rr, rdudt, ru, fu, ff)

    != Output results =
    open(10,file="./out_timeevolution.dat")
      do irec = 0, nrec-1
        write(10,"(99g17.7e3)") time(irec),         &
             dble(u_raw(irec)), aimag(u_raw(irec)), & ! u(t)
             dble(dudt_raw(irec)), aimag(dudt_raw(irec)), & ! du/dt(t)
             dble(f_raw(irec)), aimag(f_raw(irec)), & ! f = Omega*u(t) + s(t) + r(t)
             dble(omega*u_raw(irec)), aimag(omega*u_raw(irec)), & ! omega*u(t)
             dble(s_raw(irec)), aimag(s_raw(irec)), & ! Memory term s(t) = - int Gamma(s)*u(t-s) ds
             dble(r_raw(irec)), aimag(r_raw(irec))    ! Noise term r(t)
      end do
    close(10)
    open(10,file="./out_correlation.dat")
      do iperiod = 0, nperiod-1
        write(10,"(99g17.7e3)") time(iperiod),      &
             dble(uu(iperiod)), aimag(uu(iperiod)), & ! <u(t)u*>
             dble(fu(iperiod)), aimag(fu(iperiod)), & ! <f(t)u*>
             dble(ududt(iperiod)), aimag(ududt(iperiod)), & ! <u(t)du*dt>
             dble(fdudt(iperiod)), aimag(fdudt(iperiod))    ! <f(t)du*dt>
      end do
    close(10)
    open(10,file="./out_check_memoryfunc.dat")
      do iperiod = 0, nperiod-1
        write(10,"(99g17.7e3)") time(iperiod),      &
             dble(memoryf(iperiod)), aimag(memoryf(iperiod)),   & ! Memory function Gamma(t)
             dble(rr(iperiod)/uu(0)), aimag(rr(iperiod)/uu(0)), & ! <r(t)r*>/<uu*>
             dble(rdudt(iperiod)/uu(0)), aimag(rdudt(iperiod)/uu(0)) ! <r(t)dudt*>/<uu*>
      end do
    close(10)
    open(10,file="./out_check_r.dat")
      do iperiod = 0, nperiod-1
        write(10,"(99g17.7e3)") time(iperiod),      &
             dble(rr(iperiod)), aimag(rr(iperiod)), & ! <r(t)r*>
             dble(ru(iperiod)/sqrt(rr(0)*uu(0))), aimag(ru(iperiod)/sqrt(rr(0)*uu(0))), & ! <r(t)u*> normalized
             dble(fu(iperiod)/sqrt(ff(0)*uu(0))), aimag(fu(iperiod)/sqrt(ff(0)*uu(0)))    ! <f(t)u*> normalized
      end do
    close(10)

    write(*,*) omega

END PROGRAM test

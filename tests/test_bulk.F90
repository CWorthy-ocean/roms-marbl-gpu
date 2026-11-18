program test_bulk
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use case_io
  use frc_io, only: write_toy_bulk
  use ieee_arithmetic, only: ieee_is_nan
  use geom, only: setup_metrics
  implicit none
  integer(ip) :: nfail, unit
  integer :: llm, mmm, nz
  real(rp) :: t0, t1, smax, qmax

  nfail = 0
  llm = 8; mmm = 6; nz = 4
  call write_toy_case('build', llm, mmm, nz)
  t0 = 0.0_rp
  t1 = 2.0_rp
  call write_toy_bulk('build/toy_bulk.nc', llm + 2, mmm + 2, t0, t1)

  open(newunit=unit, file='build/toy_bulk.nml', status='replace')
  write(unit,'(a)') '&param_settings'
  write(unit,'(a,i0)') '    llm = ', llm
  write(unit,'(a,i0)') '    mmm = ', mmm
  write(unit,'(a,i0)') '    nz = ', nz
  write(unit,'(a)')    '    np_xi = 1'
  write(unit,'(a)')    '    np_eta = 1'
  write(unit,'(a)')    '/'
  write(unit,'(a)') '&grid_settings'
  write(unit,'(a)') "    grdname = 'toy_grid.nc'"
  write(unit,'(a)') '/'
  write(unit,'(a)') '&initial_conditions'
  write(unit,'(a)') "    inifile = 'toy_init.nc'"
  write(unit,'(a)') '/'
  write(unit,'(a)') '&s_coord'
  write(unit,'(a)') '    hc = 250.0'
  write(unit,'(a)') '    theta_b = 6.0'
  write(unit,'(a)') '    theta_s = 6.0'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&time_stepping'
  write(unit,'(a)') '    dt = 900.0'
  write(unit,'(a)') '    ndtfast = 20'
  write(unit,'(a)') '    ntimes = 1'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&forcing_files'
  write(unit,'(a)') "    frcfiles = 'toy_bulk.nc'"
  write(unit,'(a)') '/'
  write(unit,'(a)') '&rho0_settings'
  write(unit,'(a)') '    rho0 = 1025.0'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&imex_igw_settings imex_igw = .false. /'   ! required in every namelist
  close(unit)

  call load_case('build/toy_bulk.nml')
  rmask = 1.0_rp
  call setup_metrics()
  call set_forces_again()
  if (.not. have_bulk) nfail = nfail + 1
  smax = maxval(abs(sustr(1:decomp%nx+1, 1:decomp%ny)))
  qmax = maxval(abs(stflx(1:decomp%nx, 1:decomp%ny, itemp)))
  if (smax < 1.0e-6_rp .or. ieee_is_nan(smax)) nfail = nfail + 1
  if (ieee_is_nan(qmax) .or. qmax < 1.0e-8_rp) nfail = nfail + 1
  if (maxval(srflx) <= 0.0_rp) nfail = nfail + 1
  if (nfail /= 0) then
    write(*,*) 'test_bulk FAIL', nfail, smax, qmax
    error stop
  endif
  write(*,*) ' test_bulk OK'
contains
  subroutine set_forces_again()
    use frc_io, only: set_forces
    tdays = 0.0_rp
    time = 0.0_rp
    call set_forces()
  end subroutine set_forces_again
end program test_bulk

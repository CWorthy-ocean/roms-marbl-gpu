program test_tides
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use case_io
  use tide_io, only: write_toy_tides
  use ieee_arithmetic, only: ieee_is_nan
  implicit none
  integer(ip) :: nfail, unit
  integer :: llm, mmm, nz

  nfail = 0
  llm = 8; mmm = 6; nz = 4
  call write_toy_case('build', llm, mmm, nz)
  call write_toy_tides('build/toy_tides.nc', llm + 2, mmm + 2, 1)

  open(newunit=unit, file='build/toy_tides.nml', status='replace')
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
  write(unit,'(a)') "    frcfiles = 'toy_tides.nc'"
  write(unit,'(a)') '/'
  write(unit,'(a)') '&tides_settings'
  write(unit,'(a)') '    ntides = 1'
  write(unit,'(a)') '    pot_tides = .true.'
  write(unit,'(a)') '    bry_tides = .true.'
  write(unit,'(a)') '    ana_tides = .false.'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&imex_igw_settings imex_igw = .false. /'   ! required in every namelist
  close(unit)

  call load_case('build/toy_tides.nml')
  if (.not. pot_tides .or. .not. bry_tides) nfail = nfail + 1
  if (.not. obc_west) nfail = nfail + 1
  ! ocean_time = 86400, omega = 2*pi/86400 -> cos(omT) ~ 1
  if (ieee_is_nan(zeta_west(3)) .or. abs(zeta_west(3) - 0.1_rp) > 0.02_rp) nfail = nfail + 1
  if (ieee_is_nan(maxval(ptide)) .or. maxval(abs(ptide)) < 0.005_rp) nfail = nfail + 1
  if (nfail /= 0) then
    write(*,*) 'test_tides FAIL', nfail, zeta_west(3), maxval(abs(ptide))
    error stop
  endif
  write(*,*) ' test_tides OK'
end program test_tides

program test_obc
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use case_io
  use frc_io, only: write_toy_flux
  use bry_io, only: write_toy_bry
  use ieee_arithmetic, only: ieee_is_nan
  use step_driver, only: run_baroclinic_step
  use geom, only: setup_metrics
  implicit none
  integer(ip) :: nfail, unit
  integer :: llm, mmm, nz
  real(rp) :: t0, t1

  nfail = 0
  llm = 8; mmm = 6; nz = 4
  call write_toy_case('build', llm, mmm, nz)
  t0 = 0.0_rp
  t1 = 2.0_rp
  call write_toy_flux('build/toy_flux.nc', llm + 2, mmm + 2, t0, t1)
  call write_toy_bry('build/toy_bry.nc', llm + 2, mmm + 2, nz, t0, t1)

  open(newunit=unit, file='build/toy_obc.nml', status='replace')
  write(unit,'(a)') '&param_settings'
  write(unit,'(a,i0)') '    llm = ', llm
  write(unit,'(a,i0)') '    mmm = ', mmm
  write(unit,'(a,i0)') '    nz = ', nz
  write(unit,'(a)')    '    np_xi = 1'
  write(unit,'(a)')    '    np_eta = 1'
  write(unit,'(a)')    '    nt_passive = 0'
  write(unit,'(a)')    '    nt_bgc = 0'
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
  write(unit,'(a)') '    ninfo = 1'
  write(unit,'(a)') '    ntimes = 1'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&forcing_files'
  write(unit,'(a)') "    frcfiles = 'toy_flux.nc', 'toy_bry.nc'"
  write(unit,'(a)') '/'
  write(unit,'(a)') '&surf_frc_settings'
  write(unit,'(a)') '    interp_flux_frc = .false.'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&rho0_settings'
  write(unit,'(a)') '    rho0 = 1025.0'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&ubind_settings'
  write(unit,'(a)') '    ubind = 0.1'
  write(unit,'(a)') '/'
  write(unit,'(a)') '&imex_igw_settings imex_igw = .false. /'   ! required in every namelist
  close(unit)

  call load_case('build/toy_obc.nml')
  rmask = 1.0_rp
  call setup_metrics()
  if (.not. have_flux) then
    write(*,*) 'flux not detected'
    nfail = nfail + 1
  endif
  if (.not. have_bry .or. .not. obc_west) then
    write(*,*) 'bry/OBC not detected', have_bry, obc_west
    nfail = nfail + 1
  endif
  if (abs(sustr(1, 1) - 0.1_rp / rho0_case) > 1.0e-12_rp) then
    write(*,*) 'sustr kinematic', sustr(1, 1), 0.1_rp / rho0_case
    nfail = nfail + 1
  endif
  if (abs(ubar_west(1) - 0.05_rp) > 1.0e-12_rp) then
    write(*,*) 'ubar_west', ubar_west(1)
    nfail = nfail + 1
  endif
  if (abs(t_west(1, 1, 1) - 15.1_rp) > 1.0e-12_rp) then
    write(*,*) 'temp_west', t_west(1, 1, 1)
    nfail = nfail + 1
  endif

  start_time = 86400.0_rp   ! 1 day, between t0=0 and t1=2 days if ocean_time is seconds
  ! toy init ocean_time is 86400 s = 1 day. bry times are 0 and 2 days.
  tdays = start_time * sec2day
  iic = 1
  call run_baroclinic_step()

  if (ieee_is_nan(ubar(1, 3, knew)) .or. abs(ubar(1, 3, knew)) < 1.0e-8_rp) then
    write(*,*) 'west ubar', ubar(1, 3, knew)
    nfail = nfail + 1
  endif
  if (ieee_is_nan(maxval(t(1:decomp%nx, 1:decomp%ny, :, nnew, 1)))) then
    write(*,*) 'NaN in temperature after step'
    nfail = nfail + 1
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_obc FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_obc OK'
end program test_obc

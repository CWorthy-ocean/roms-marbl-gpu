program test_his
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use case_io
  use his_io, only: maybe_wrt_output, wrt_file_his, wrt_file_rst
  use step_driver, only: run_baroclinic_step
  use nc_io
  use ieee_arithmetic, only: ieee_is_nan
  implicit none
  integer(ip) :: nfail, unit
  integer :: llm, mmm, nz, ncid, nrec
  real(rp) :: tbuf(1), z_ic, z_step, z_rst, t_step, t_rst
  real(rp), allocatable :: z2(:,:)

  nfail = 0
  llm = 8; mmm = 6; nz = 4
  call write_toy_case('build', llm, mmm, nz)
  call write_nml('build/toy_his.nml', 'toy_init.nc', .true., .true.)

  call load_case('build/toy_his.nml')
  if (.not. wrt_file_his .or. .not. wrt_file_rst) nfail = nfail + 1
  z_ic = zeta(3, 3, 1)

  call nc_open_read('build/toy_out_his.nc', ncid)
  nrec = nc_var_nrecs(ncid, 'ocean_time')
  if (nrec < 1) nfail = nfail + 1
  call nc_get_1d(ncid, 'ocean_time', tbuf, 1)
  if (ieee_is_nan(tbuf(1)) .or. abs(tbuf(1) - 86400.0_rp) > 1.0e-6_rp) nfail = nfail + 1
  allocate(z2(llm + 2, mmm + 2))
  call nc_read_2d(ncid, 'zeta', z2, llm + 2, mmm + 2, 1)
  if (ieee_is_nan(z2(4, 4)) .or. abs(z2(4, 4) - z_ic) > 1.0e-12_rp) nfail = nfail + 1
  if (.not. nc_has_var(ncid, 'temp') .or. .not. nc_has_var(ncid, 'Akv')) nfail = nfail + 1
  call nc_close(ncid)

  iic = max(ntstart, 1)
  call run_baroclinic_step()
  call maybe_wrt_output()
  z_step = zeta(3, 3, knew)
  t_step = t(3, 3, 1, nnew, 1)
  if (ieee_is_nan(z_step) .or. ieee_is_nan(t_step)) nfail = nfail + 1

  call nc_open_read('build/toy_out_his.nc', ncid)
  nrec = nc_var_nrecs(ncid, 'ocean_time')
  if (nrec < 2) nfail = nfail + 1
  call nc_read_2d(ncid, 'zeta', z2, llm + 2, mmm + 2, 2)
  if (ieee_is_nan(z2(4, 4)) .or. abs(z2(4, 4) - z_step) > 1.0e-12_rp) nfail = nfail + 1
  call nc_close(ncid)
  deallocate(z2)

  call free_state()
  call write_nml('build/toy_rst.nml', 'toy_out_rst.nc', .false., .false.)
  call load_case('build/toy_rst.nml')
  z_rst = zeta(3, 3, 1)
  t_rst = t(3, 3, 1, 1, 1)
  if (ieee_is_nan(z_rst) .or. abs(z_rst - z_step) > 1.0e-12_rp) nfail = nfail + 1
  if (ieee_is_nan(t_rst) .or. abs(t_rst - t_step) > 1.0e-12_rp) nfail = nfail + 1
  call free_state()

  if (nfail /= 0) then
    write(*,*) 'test_his FAIL', nfail, z_ic, z_step, z_rst
    error stop
  endif
  write(*,*) ' test_his OK'
contains
  subroutine write_nml(path, inifile, his, rst)
    character(len=*), intent(in) :: path, inifile
    logical,          intent(in) :: his, rst
    open(newunit=unit, file=trim(path), status='replace')
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
    write(unit,'(a,a,a)') "    inifile = '", trim(inifile), "'"
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
    write(unit,'(a)') '&simulation_name_settings'
    write(unit,'(a)') "    output_root_name = 'toy_out'"
    write(unit,'(a)') "    title = 'toy_his'"
    write(unit,'(a)') '/'
    write(unit,'(a)') '&basic_output_settings'
    if (his) then
      write(unit,'(a)') '    wrt_file_his = .true.'
    else
      write(unit,'(a)') '    wrt_file_his = .false.'
    endif
    if (rst) then
      write(unit,'(a)') '    wrt_file_rst = .true.'
    else
      write(unit,'(a)') '    wrt_file_rst = .false.'
    endif
    write(unit,'(a)') '    output_period_his = 900.0'
    write(unit,'(a)') '    output_period_rst = 900.0'
    write(unit,'(a)') '    nrpf_his = 20'
    write(unit,'(a)') '    wrt_Z = .true.'
    write(unit,'(a)') '    wrt_Ub = .true.'
    write(unit,'(a)') '    wrt_Vb = .true.'
    write(unit,'(a)') '    wrt_U = .true.'
    write(unit,'(a)') '    wrt_V = .true.'
    write(unit,'(a)') '    wrt_R = .true.'
    write(unit,'(a)') '    wrt_O = .true.'
    write(unit,'(a)') '    wrt_W = .false.'
    write(unit,'(a)') '    wrt_Akv = .true.'
    write(unit,'(a)') '    wrt_Akt = .true.'
    write(unit,'(a)') '    wrt_Aks = .true.'
    write(unit,'(a)') '    wrt_Hbls = .true.'
    write(unit,'(a)') '    wrt_Hbbl = .true.'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&ts_output_settings'
    write(unit,'(a)') '    wrt_temp = .true.'
    write(unit,'(a)') '    wrt_salt = .true.'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&imex_igw_settings imex_igw = .false. /'   ! required in every namelist
    close(unit)
  end subroutine write_nml
end program test_his

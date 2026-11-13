#include "mem_attr.inc"
module marbl_gpu
  ! MARBL inside the GPU model: configuration, tracer/state setup, the
  ! per-call driver (surface fluxes, carbonate chemistry, interior
  ! tendencies) and the restart/history hooks. Mirrors the CPU driver
  ! marbl_driver.F90: MARBL runs every marbl_timestep_ratio steps at the
  ! end of the tracer step on the nnew slot, surface flux first (applied to
  ! the top layer), then the interior tendencies, with ifrac = 0 and
  ! atmospheric pressure 1 atm.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  use nc_io
  use frc_io, only: init_bgc_forcing, set_bgc_forces, uwnd, vwnd, bgc_frc_on
  use run_timers, only: rt_dev_begin, rt_dev_end
  use passive_mod, only: ntp, n_marbl, tp, set_passive_marbl, passive_marbl, passive_name, &
       passive_hold, hold_itr, hold_value
  use marbl_host
  use marbl_sflux_mod
  use marbl_interior_mod
  use io_server, only: io_stage_2d, io_stage_3d
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
#include "cuf_launch.inc"
  private

  logical, public :: marbl_on = .false.
  character(len=512), public :: marbl_config_file = 'marbl_in'
  real(rp), public :: marbl_timestep = 3600.0_rp
  integer(ip), public :: marbl_timestep_ratio = 1
  real(rp), public :: xco2_default = 284.7_rp       ! ppm when no xco2 forcing (CPU default)
  ! Constant iron (CPU marbl_driver): with use_constant_Fe, MARBL's Fe is held
  ! at constant_Fe_value (mmol/m3) at every wet point (passive_hold): after
  ! the initial state is loaded, before each MARBL call and after each
  ! step's lateral mixing.
  logical, public :: use_constant_Fe = .false.
  real(rp), public :: constant_Fe_value = 3.0e-3_rp
  integer(ip) :: ncalls = 0
  ! The saturation depths (zsatcalc/zsatarag) are read only by the history
  ! staging. A run that writes history stages its initial record before the
  ! first MARBL call (init_output), so his_staged tells whether any record
  ! can ever need them; without history they are not computed.
  logical :: his_staged = .false.
  ! The CDR output (cdr_out) averages them too: computed at every MARBL call.
  logical, public :: marbl_need_zsat = .false.
  ! marbl_tracers_to_write as read: the CPU writes (and averages) the named
  ! tracers, or every tracer when the list is empty (avg_out)
  character(len=32), public :: marbl_trc_list(64) = ''
  real(rp), parameter :: Cp = 3985.0_rp

  ! history diagnostics (2-D, surface flux module slots)
  integer, parameter :: NHIS = 7
  character(len=16), parameter :: his_name(NHIS) = [character(len=16) :: &
       'PH', 'pCO2SURF', 'DpCO2', 'FG_CO2', 'FG_ALT_CO2', 'PH_ALT_CO2', 'pCO2SURF_ALT_CO2']
  integer, parameter :: his_slot(NHIS) = [D_PH, D_PCO2, D_DPCO2, D_FGCO2, D_FGCO2_ALT, D_PH_ALT, D_PCO2_ALT]
  ! history diagnostics from the interior carbonate kernel: 2-D saturation
  ! depths and 3-D carbonate fields (CPU names, mmol/m^3 and m)
  integer, parameter :: NHIS2I = 2, NHIS3 = 4
  character(len=16), parameter :: his2i_name(NHIS2I) = [character(len=16) :: 'zsatcalc', 'zsatarag']
  character(len=16), parameter :: his3_name(NHIS3) = [character(len=16) :: &
       'CO3', 'CO3_ALT_CO2', 'co3_sat_calc', 'co3_sat_arag']

  public :: read_marbl_namelist
  public :: marbl_gpu_setup_tracers
  public :: marbl_gpu_init_forcing
  public :: marbl_gpu_load_state
  public :: marbl_gpu_step
  public :: marbl_gpu_init_diagnostics
  public :: marbl_gpu_nhis
  public :: marbl_gpu_nhis3
  public :: marbl_gpu_define_his
  public :: marbl_gpu_stage_his
  public :: marbl_gpu_define_rst
  public :: marbl_gpu_write_rst
  public :: marbl_gpu_stage_rst8

contains

  subroutine read_marbl_namelist(unit, nml_dir, nt_bgc)
    ! &marbl_biogeochemistry_settings as the CPU model; MARBL is on when
    ! nt_bgc > 0 in param_settings.
    use comm_mod, only: comm_rank
    integer, intent(in) :: unit
    character(len=*), intent(in) :: nml_dir
    integer(ip), intent(in) :: nt_bgc
    integer :: ios
    character(len=512) :: cfg
    real(rp) :: tstep
    character(len=32) :: marbl_tracers_to_write(64)
    character(len=400) :: marbl_diagnostics_to_write(64)
    ! marbl_balance (UCLA ROMS after 0.14.0, #430) ships MARBL columns between CPU
    ! ranks and changes no result; here every GPU computes the wet columns
    ! of its tile (&tile_settings balances them), so it is read and ignored.
    logical :: marbl_balance
    namelist /marbl_biogeochemistry_settings/ marbl_config_file, marbl_timestep, &
         marbl_tracers_to_write, marbl_diagnostics_to_write, use_constant_Fe, constant_Fe_value, &
         marbl_balance
    marbl_on = (nt_bgc > 0)
    if (.not. marbl_on) return
    marbl_config_file = 'marbl_in'
    marbl_timestep = 3600.0_rp
    marbl_tracers_to_write = ''
    marbl_diagnostics_to_write = ''
    use_constant_Fe = .false.
    constant_Fe_value = 3.0e-3_rp
    marbl_balance = .false.
    rewind(unit)
    read(unit, nml=marbl_biogeochemistry_settings, iostat=ios)
    if (ios > 0) error stop 'marbl_biogeochemistry_settings: read error'
    if (marbl_balance .and. comm_rank() == 0) write(*,'(a)') 'MARBL: marbl_balance = .true. is ignored '// &
         'on the GPU (each GPU computes the wet columns of its tile; &tile_settings balances them)'
    marbl_trc_list = marbl_tracers_to_write
    if (use_constant_Fe) then
      if (constant_Fe_value < 0.0_rp) error stop 'use_constant_Fe: constant_Fe_value must be non-negative'
      write(*,'(a,es12.5,a)') 'MARBL Fe held constant at ', constant_Fe_value, ' (use_constant_Fe)'
    endif
    cfg = marbl_config_file
    if (cfg(1:1) /= '/') marbl_config_file = trim(nml_dir)//'/'//trim(cfg)
    tstep = marbl_timestep
    write(*,'(a,i0,a,a,a,f8.1)') 'MARBL on: nt_bgc = ', nt_bgc, '  config ', trim(marbl_config_file), &
         '  requested timestep ', tstep
  end subroutine read_marbl_namelist

  ! Initialise MARBL (unchanged settings/init code), publish the tracer
  ! names to the passive module and allocate the device arrays. Called
  ! after read_namelist (decomp, dt known), before alloc_passive.
  subroutine marbl_gpu_setup_tracers()
    integer(ip) :: m
    if (.not. marbl_on) return
    call marbl_host_init(decomp%nz, trim(marbl_config_file), .true.)
    call set_passive_marbl(marbl_tracer_names)
    if (use_constant_Fe) then
      do m = 1, ntp
        if (trim(passive_name(m)) == 'Fe') hold_itr = m
      enddo
      if (hold_itr == 0) error stop 'use_constant_Fe: MARBL has no tracer named Fe'
      hold_value = constant_Fe_value
    endif
    call set_marbl_par()
    call alloc_marbl_sflux(ntp)
    call alloc_marbl_interior(decomp%nz, ntp, .false.)
    marbl_timestep_ratio = max(1_ip, int(marbl_timestep / dt, ip))
    marbl_timestep = dt * real(marbl_timestep_ratio, rp)
    sf_xco2 = xco2_default
    sf_xco2_alt = xco2_default
    write(*,'(a,i0,a,f8.1,a)') 'MARBL timestep ratio ', marbl_timestep_ratio, '  (', marbl_timestep, ' s)'
  end subroutine marbl_gpu_setup_tracers

  subroutine marbl_gpu_init_forcing()
    if (.not. marbl_on) return
    call init_bgc_forcing()
  end subroutine marbl_gpu_init_forcing

  ! Saved state from the initial/restart file (CPU names). Missing
  ! variables leave the pH at 0 = "no previous solution".
  subroutine marbl_gpu_load_state(ncid, rec, i0, j0, nxi, neta, fi0, fj0)
    use par_read, only: pr_store_2d, pr_store_3d, pr_taking
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, fi0, fj0
    if (.not. marbl_on) return
    if (nc_has_var(ncid, 'MARBL_PH_SURF')) then
      call pr_store_2d(ncid, 'MARBL_PH_SURF', sf_ph(decomp%is, decomp%js), i0, j0, nxi, neta, rec, fi0, fj0)
    endif
    if (nc_has_var(ncid, 'MARBL_PH_SURF_ALT_CO2')) then
      call pr_store_2d(ncid, 'MARBL_PH_SURF_ALT_CO2', sf_ph_alt(decomp%is, decomp%js), i0, j0, nxi, neta, rec, fi0, fj0)
    endif
    if (nc_has_var(ncid, 'MARBL_PH_3D')) then
      call pr_store_3d(ncid, 'MARBL_PH_3D', it_ph(decomp%is, decomp%js, 1), i0, j0, nxi, neta, decomp%nz, rec, fi0, fj0)
    endif
    if (nc_has_var(ncid, 'MARBL_PH_3D_ALT_CO2')) then
      call pr_store_3d(ncid, 'MARBL_PH_3D_ALT_CO2', it_pha(decomp%is, decomp%js, 1), i0, j0, nxi, neta, decomp%nz, rec, fi0, fj0)
    endif
    if (pr_taking()) write(*,'(a)') 'MARBL saved state (surface and 3-D pH) loaded where present'
  end subroutine marbl_gpu_load_state

  ! One MARBL call: at the end of the tracer step (after the passive
  ! corrector and its open boundaries), as the CPU driver.
  subroutine marbl_gpu_step()
    integer(ip) :: i, j, m, nx, ny, nz, nn, islot, c, nsf, nit, nmb
    real(rp) :: dtm, rc, tfrc
    if (.not. marbl_on) return
    if (mod(iic, marbl_timestep_ratio) /= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nn = nnew
    dtm = marbl_timestep
    rc = rho0_case * Cp
    ! forcing at the corrector's set_forces time (CPU: '1/2 fwd', n+dt/2);
    ! by now the step driver has moved frc_slot on to the n+1 boundary time.
    islot = frc_slot
    frc_slot = 1
    tfrc = frc_modtime()
    frc_slot = islot
    call set_bgc_forces(tfrc, sf_dust, sf_iron, sf_nox, sf_nhy, sf_xco2, sf_xco2_alt)
    ! use_constant_Fe (no-op otherwise): MARBL computes with the constant iron;
    ! the step's changes to Fe are undone after lateral mixing (passive_t3dmix)
    call passive_hold(nn)
    call rt_dev_begin()     ! MARBL proper, as the CPU's marbldrv_column_physics timer
    ! wind speed squared (bulk winds on the device) and shortwave in W/m2
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        sf_u10sqr(i, j) = (uwnd(i, j)**2) + (vwnd(i, j)**2)
        it_qsw(i, j) = srflx(i, j) * rc
      enddo
    enddo
    ! surface fluxes, applied to the top layer of slot nnew; the MARBL
    ! tracers are tp 1 .. n_marbl (generic and CDR_LITE tracers follow; MARBL
    ! does not see them). The kernels take ntp as tp's leading dimension.
    nmb = n_marbl
    call marbl_sflux_compute(msf, ntp, nz, nz, nn, tp, nt, t, rmask)
    ! wet columns c of the interior are tp columns c (gpu_state, tpcol)
    CUF_DO2
    do m = 1, nmb
      do c = 1, nwet
        i = wet_i(c); j = wet_j(c)
        tp(m, c, nz, nn) = real(real(tp(m, c, nz, nn), rp) + &
             sf_flux(i, j, m) * dtm / Hz(i, j, nz), pp)
      enddo
    enddo
    ! interior: carbonate chemistry, then the column ecosystem/sinking sweep
    call marbl_carbonate_compute(ntp, nz, nn, 3_ip, tp, nt, t, z_r, rmask)
    if (his_staged .or. marbl_need_zsat) call marbl_zsat_compute(nz, z_r, z_w, rmask)
    call marbl_interior_compute(ntp, nz, nn, 3_ip, tp, nt, t, z_r, z_w, Hz, rmask, sf_dust, dtm, .false.)
    call rt_dev_end()
    ncalls = ncalls + 1
    ! Carbonate solver failures (no pH bracket, or no convergence) are
    ! MARBL errors, and the CPU model aborts on them; so does this one,
    ! after every call. The counts are a device reduction (its result waits
    ! for the kernels before it), so the managed sf_err/it_err stay on the
    ! device. MiniPac MARBL steps took the same time with and without it.
    nsf = 0; nit = 0
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        if (sf_err(i, j) /= 0) nsf = nsf + 1
        nit = nit + it_err(i, j)
      enddo
    enddo
    if (nsf + nit > 0) call solver_abort(nsf, nit, nn)
    if (ncalls <= 3 .or. mod(ncalls, 24) == 0) then
      write(*,'(a,i0,a,i0,a)') 'MARBL call ', ncalls, ' at iic ', iic, '  carbonate solver: no failures'
    endif
  end subroutine marbl_gpu_step

  ! The CPU's marbldrv_compute_init_diagnostics, which its first wrt_bgc
  ! calls at start-up (before the first MARBL step): the surface-flux and
  ! carbonate diagnostics of the initial state (slot nstp) with the
  ! start-time forcing (load_case's set_forces, frc_slot 0). The tracers
  ! and the saved state stay as loaded: the CPU does not copy MARBL's
  ! saved state back here, so its first MARBL step starts from the loaded
  ! pH, and its pH output (saved state) stays as loaded until then. Until
  ! the first MARBL call the history and the CDR output hold these values,
  ! as the CPU's.
  subroutine marbl_gpu_init_diagnostics()
    integer(ip) :: i, j, k, nx, ny, nz, n0, is, ie, js, je
    real(rp) :: tfrc
    real(rp), WORK_ATTR allocatable :: s1(:,:), s2(:,:), p1(:,:,:), p2(:,:,:)
    if (.not. marbl_on) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    n0 = nstp
    tfrc = frc_modtime()
    call set_bgc_forces(tfrc, sf_dust, sf_iron, sf_nox, sf_nhy, sf_xco2, sf_xco2_alt)
    allocate(s1(is:ie, js:je), s2(is:ie, js:je), p1(is:ie, js:je, nz), p2(is:ie, js:je, nz))
    CUF_DO2
    do j = js, je
      do i = is, ie
        s1(i, j) = sf_ph(i, j); s2(i, j) = sf_ph_alt(i, j)
        do k = 1, nz
          p1(i, j, k) = it_ph(i, j, k); p2(i, j, k) = it_pha(i, j, k)
        enddo
      enddo
    enddo
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        sf_u10sqr(i, j) = (uwnd(i, j)**2) + (vwnd(i, j)**2)
      enddo
    enddo
    call marbl_sflux_compute(msf, ntp, nz, nz, n0, tp, nt, t, rmask)
    call marbl_carbonate_compute(ntp, nz, n0, 3_ip, tp, nt, t, z_r, rmask)
    call marbl_zsat_compute(nz, z_r, z_w, rmask)
    CUF_DO2
    do j = js, je
      do i = is, ie
        sf_ph(i, j) = s1(i, j); sf_ph_alt(i, j) = s2(i, j)
        do k = 1, nz
          it_ph(i, j, k) = p1(i, j, k); it_pha(i, j, k) = p2(i, j, k)
        enddo
      enddo
    enddo
    call sync_arch()
    deallocate(s1, s2, p1, p2)
    write(*,'(a)') 'MARBL initial diagnostics (surface flux, carbonate) from the initial state'
  end subroutine marbl_gpu_init_diagnostics

  subroutine solver_abort(nsf, nit, nn)
    ! MARBL's carbonate solver failed on this tile: list the first columns
    ! (global i, j, their failures, surface inputs and the column's lowest
    ! salinity, as the CPU's MARBL error lists dic, ta, pt, sit, temp and
    ! salt) and stop every rank.
    use comm_mod, only: comm_rank, comm_abort
    integer(ip), intent(in) :: nsf, nit, nn
    integer(ip) :: i, j, k, m, c, n, nz
    integer(ip) :: idic, ialk, ipo4, isio3
    real(rp) :: smin, v(4)
#ifdef USE_CUDA
    integer :: istat
    istat = cudaDeviceSynchronize()    ! the managed arrays are read on the host
#endif
    nz = decomp%nz
    idic = 0; ialk = 0; ipo4 = 0; isio3 = 0
    do m = 1, ntp
      select case (trim(passive_name(m)))
      case ('DIC'); idic = m
      case ('ALK'); ialk = m
      case ('PO4'); ipo4 = m
      case ('SiO3'); isio3 = m
      end select
    enddo
    write(*,'(a,i0,a,i0,a,i0,a,i0,a)') 'MARBL ERROR: carbonate solver failed on rank ', comm_rank(), &
         ' at iic ', iic, ' (', nsf, ' surface columns, ', nit, ' interior points; no pH bracket or no convergence)'
    write(*,'(a)') '   global i     j  surf  intr      temp      salt   min salt       DIC       ALK       PO4      SiO3'
    n = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (sf_err(i, j) == 0 .and. it_err(i, j) == 0) cycle
        n = n + 1
        if (n > 10) exit
        smin = huge(smin)
        do k = 1, nz
          smin = min(smin, t(i, j, k, nn, isalt))
        enddo
        v = 0.0_rp
        c = tpcol_h(i, j)
        if (c > 0) then
          if (idic > 0) v(1) = real(tp(idic, c, nz, nn), rp)
          if (ialk > 0) v(2) = real(tp(ialk, c, nz, nn), rp)
          if (ipo4 > 0) v(3) = real(tp(ipo4, c, nz, nn), rp)
          if (isio3 > 0) v(4) = real(tp(isio3, c, nz, nn), rp)
        endif
        write(*,'(3x,2i6,2i6,7es10.2)') decomp%i0 + i - 1, decomp%j0 + j - 1, sf_err(i, j), it_err(i, j), &
             t(i, j, nz, nn, itemp), t(i, j, nz, nn, isalt), smin, v
      enddo
      if (n > 10) exit
    enddo
    call comm_abort(1)
  end subroutine solver_abort

  ! Debug: NaN census of the surface-flux inputs/outputs after the first
  ! call, plus one sample point.

  integer function marbl_gpu_nhis()
    ! number of 2-D MARBL history fields (staging capacity)
    marbl_gpu_nhis = 0
    if (marbl_on) marbl_gpu_nhis = NHIS + NHIS2I
  end function marbl_gpu_nhis

  integer function marbl_gpu_nhis3()
    ! number of 3-D (nz-level) MARBL history fields (staging capacity)
    marbl_gpu_nhis3 = 0
    if (marbl_on) marbl_gpu_nhis3 = NHIS3
  end function marbl_gpu_nhis3

  subroutine marbl_gpu_define_his(ncid, d_xi, d_eta, d_s, d_t)
    integer, intent(in) :: ncid, d_xi, d_eta, d_s, d_t
    integer :: n, id
    if (.not. marbl_on) return
    do n = 1, NHIS
      id = nc_def_var_f(ncid, trim(his_name(n)), (/d_xi, d_eta, d_t/))
    enddo
    do n = 1, NHIS2I
      id = nc_def_var_f(ncid, trim(his2i_name(n)), (/d_xi, d_eta, d_t/))
    enddo
    do n = 1, NHIS3
      id = nc_def_var_f(ncid, trim(his3_name(n)), (/d_xi, d_eta, d_s, d_t/))
    enddo
  end subroutine marbl_gpu_define_his

  subroutine marbl_gpu_stage_his(i0, i1, j0, j1, di, dj)
    integer, intent(in) :: i0, i1, j0, j1, di, dj
    integer :: n, nz
    if (.not. marbl_on) return
    his_staged = .true.
    nz = decomp%nz
    do n = 1, NHIS
      call io_stage_2d(trim(his_name(n)), sf_diag(decomp%is, decomp%js, his_slot(n)), i0, i1, j0, j1, di, dj)
    enddo
    call io_stage_2d(trim(his2i_name(1)), it_zsatc(decomp%is, decomp%js), i0, i1, j0, j1, di, dj)
    call io_stage_2d(trim(his2i_name(2)), it_zsata(decomp%is, decomp%js), i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(1)), it_co3(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(2)), it_co3a(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(3)), it_co3sat(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(4)), it_co3sata(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
  end subroutine marbl_gpu_stage_his

  subroutine marbl_gpu_define_rst(ncid, d_xi, d_eta, d_s, d_t)
    integer, intent(in) :: ncid, d_xi, d_eta, d_s, d_t
    integer :: id
    if (.not. marbl_on) return
    id = nc_def_var(ncid, 'MARBL_PH_SURF', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'MARBL_PH_SURF_ALT_CO2', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'MARBL_PH_3D', (/d_xi, d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'MARBL_PH_3D_ALT_CO2', (/d_xi, d_eta, d_s, d_t/))
  end subroutine marbl_gpu_define_rst

  subroutine marbl_gpu_stage_rst8(i0, i1, j0, j1, di, dj)
    ! The saved pH state into a double-precision staged (server) record.
    use io_server, only: io_stage8_2d, io_stage8_3d
    integer, intent(in) :: i0, i1, j0, j1, di, dj
    integer :: nz
    if (.not. marbl_on) return
    nz = int(decomp%nz)
    call io_stage8_2d('MARBL_PH_SURF', sf_ph(decomp%is, decomp%js), i0, i1, j0, j1, di, dj)
    call io_stage8_2d('MARBL_PH_SURF_ALT_CO2', sf_ph_alt(decomp%is, decomp%js), i0, i1, j0, j1, di, dj)
    call io_stage8_3d('MARBL_PH_3D', it_ph(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage8_3d('MARBL_PH_3D_ALT_CO2', it_pha(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
  end subroutine marbl_gpu_stage_rst8

  subroutine marbl_gpu_write_rst(ncid, rec, i0, i1, j0, j1, fi0, fj0)
    ! fi0, fj0: 1-based global file offsets of this tile's first interior point
    integer, intent(in) :: ncid, rec, i0, i1, j0, j1, fi0, fj0
    real(rp), allocatable :: t2(:,:), t3(:,:,:)
    integer :: nxi, neta
    if (.not. marbl_on) return
    nxi = i1 - i0 + 1; neta = j1 - j0 + 1
    call sync_arch()
    allocate(t2(nxi, neta), t3(nxi, neta, decomp%nz))
    t2 = sf_ph(i0:i1, j0:j1);     call nc_put_2d_rec(ncid, 'MARBL_PH_SURF', t2, rec, fi0, fj0)
    t2 = sf_ph_alt(i0:i1, j0:j1); call nc_put_2d_rec(ncid, 'MARBL_PH_SURF_ALT_CO2', t2, rec, fi0, fj0)
    t3 = it_ph(i0:i1, j0:j1, :);  call nc_put_3d_rec(ncid, 'MARBL_PH_3D', t3, rec, fi0, fj0)
    t3 = it_pha(i0:i1, j0:j1, :); call nc_put_3d_rec(ncid, 'MARBL_PH_3D_ALT_CO2', t3, rec, fi0, fj0)
    deallocate(t2, t3)
  end subroutine marbl_gpu_write_rst

end module marbl_gpu

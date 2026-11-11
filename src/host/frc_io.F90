#include "mem_attr.inc"
module frc_io
  ! Flux-file or COARE 3.0 bulk surface forcing; two-level time
  ! interpolation. Records are read on the host into managed buffers;
  ! the interpolation onto the grid (also from 2x-coarse inputs, as the
  ! CPU's interp_*_frc) and the bulk fluxes run as device kernels.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use bulk_coare, only: calc_bulk_coare, calc_cflx_corr
  use passive_mod, only: ntp, tp, n_marbl, passive_name
  use cdr_lite_mod, only: cdrl_psf_full, cdrl_psf, psf_cflx
  use arch_mod, only: sync_arch
  use comm_mod, only: comm_rank, comm_size, comm_compute
  use run_timers, only: wall_time
  use netcdf, only: nf90_inquire, nf90_inq_varid, nf90_inquire_variable, nf90_inquire_dimension, nf90_get_var, &
                    nf90_double, nf90_float, nf90_max_var_dims, nf90_open, nf90_nowrite
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private

  integer, parameter :: max_frc = 64
  integer, parameter :: LAY_GRID = 1, LAY_COARSE = 2
  character(len=512), public :: case_frcfiles(max_frc)
  integer,            public :: case_nfrc = 0
  integer, private :: frc_cache(max_frc) = -1   ! read handles kept open during initialization
  public :: frc_file_ncid, close_frc_cache
  public :: prefetch_time_axes, cached_times
  ! Time axes of the forcing files, read once in parallel (file i by rank
  ! mod(i-1, nranks)) and broadcast: every record of a classic-format file
  ! costs one seek, so a month of hourly times is thousands of reads.
  integer, parameter :: MAXTC = 1024
  ! Time axes the binders request (frc_io bind_series, bry_io 'bry_time').
  character(len=16), parameter :: TAXES(15) = [character(len=16) :: 'time', 'sms_time', 'shf_time', &
       'swf_time', 'srf_time', 'sss_time', 'dust_time', 'iron_time', 'nox_time', 'nhy_time', &
       'xco2_time', 'bry_time', 'sst_time', 'sDIC_time', 'sALK_time']
  integer, parameter :: AXIS_BLOCK = 8192   ! netCDF block size for the axis reads, bytes
  type :: tc_t
    integer :: ifile = 0, n = 0
    character(len=64) :: name = ''
    real(rp), allocatable :: t(:)
  end type tc_t
  type(tc_t), save :: tcache(MAXTC)
  integer :: ntc = 0
  logical :: first_bind = .true.
  logical,            public :: interp_flux_frc = .false.

  real(rp), parameter :: Cp = 3985.0_rp
  real(rp), parameter :: cmday2ms = 0.01_rp / 86400.0_rp

  type :: fb_t
    real(rp), allocatable :: b(:)
  end type fb_t

  type :: series2d
    character(len=64)  :: vname = ''
    character(len=64)  :: tname = ''
    character(len=1)   :: gtype = 'r'
    integer :: ntot = 0, it1 = 1, it2 = 2, n1 = 0, n2 = 0, time_axis = 3
    integer :: fi0 = 1, fj0 = 1   ! file start of this tile's window (on-grid fields)
    ! Layout (bind_series, from the file shape): lay = LAY_GRID, a field on
    ! the model grid, read as this tile's window buf(1:n1, 1:n2), which is
    ! dest(li0:li0+n1-1, lj0:lj0+n2-1); lay = LAY_COARSE, a 2x-coarse field
    ! (whole file in buf).
    integer :: lay = 0, li0 = 1, lj0 = 1
    integer :: fn1 = 0, fn2 = 0   ! file shape (xi, eta), the same in every file
    logical :: ready = .false.
    ! climatology: cycle_length attribute of the time axis (<= 0: none);
    ! t1e/t2e are the shifted record times of the current bracket
    real(rp) :: cycle = -1.0_rp, t1e = 0.0_rp, t2e = 0.0_rp
    integer, allocatable :: ifile(:), irec(:)
    character(len=512), allocatable :: paths(:)
    real(rp), allocatable :: times(:)
    real(rp), WORK_ATTR allocatable :: buf(:,:,:)   ! spatial (n1,n2,2)
    ! read-ahead (frc_prefetch): record pf_idx (0: none) arriving in pf_buf
    integer :: pf_idx = 0, pf_rreq = 0
    logical :: pf_own = .false.
    real(rp), allocatable :: pf_buf(:)
    integer, allocatable :: pf_sreq(:)
    type(fb_t), allocatable :: pf_sbuf(:)
  end type series2d

  type(series2d), target :: s_sustr, s_svstr, s_shflx, s_swflx, s_swrad
  type(series2d), target :: s_uwnd, s_vwnd, s_tair, s_qair, s_prate, s_lwrad, s_bswrad
  type(series2d), target :: s_sss              ! SFLX_CORR: sea-surface salinity restoring data
  type(series2d), target :: s_sst              ! QCORRECTION: sea-surface temperature restoring data
  type(series2d), target :: s_sdic, s_salk     ! CFLX_CORR: sea-surface DIC and ALK restoring data
  ! MARBL surface forcing (dust/iron/nox/nhy climatologies, atmospheric xCO2)
  type(series2d), target :: s_dust, s_iron, s_nox, s_nhy, s_xco2, s_xco2a
  ! CDR_LITE surface forcing: carbonate sensitivities and the DIC tracers'
  ! <name>_flx (cdr_lite_io), cataloged in the CPU's file order
  type(series2d), target :: s_ddco2, s_ddalk
  type(series2d), allocatable, target :: s_cflx(:)

  ! Record reads while stepping (ROMS_GPU_FRC_BATCH bitmask, default 3):
  ! bit 1: a series that moves on by one record copies its upper bracket
  !   into the lower slot instead of reading that record again;
  ! bit 2: set_forces / set_bgc_forces first advance every series with the
  !   reads deferred (frc_batch_begin), then frc_batch_flush reads each new
  !   record whole, once, on one rank (round-robin) and sends every tile its
  !   window: one contiguous read per record instead of one strided window
  !   read per tile. The bytes that land in a tile are the window read's.
  ! bit 4 (with bit 2): frc_prefetch, called after step3d_t while the
  !   passive corrector kernels run, reads the next record of
  !   every series ahead (at most one record per rank per call, sent to
  !   the tiles with non-blocking messages); the flush that needs it only
  !   waits for the messages. A record that turns out not to be the one
  !   needed is dropped and read as before. Default 7 (all three).
  ! ROMS_GPU_FRC_TIMING=1 prints the time of every flush and prefetch.
  integer, parameter :: MAXPEND = 64
  type :: pend_t
    type(series2d), pointer :: s => null()
    integer :: idx = 0, slot = 0
  end type pend_t
  type(pend_t), save :: pend(MAXPEND)
  integer :: npend = 0
  logical :: batching = .false.
  integer :: frc_mode = -1                ! ROMS_GPU_FRC_BATCH, read once
  logical :: frc_timing = .false.
  integer :: run_ncid(max_frc) = -1       ! read handles kept open while stepping
  integer, parameter :: FRC_TAG0 = 6000, FRC_PF_TAG0 = 6100
  public :: frc_prefetch, frc_prefetch_drain, frc_run_ncid_path
  logical, public :: bgc_frc_on = .false.
  real(rp), WORK_ATTR allocatable :: uwnd(:,:), vwnd(:,:), qair(:,:), prate(:,:), lwrad_in(:,:)
  real(rp), WORK_ATTR allocatable :: sss(:,:), sst(:,:), sdic(:,:), salk(:,:)
  ! Surface restoring, bulk forcing only (UCLA ROMS bulk_frc.F90; there cpp
  ! keys, here GPU-only &surf_frc_settings switches, see NAMELIST_COMPAT):
  ! SFLX_CORR (sflx_corr): stflx(isalt) = -dSSSdt (S_surface - sss),
  !   &sss_correction dsssdt [cm/day];
  ! QCORRECTION (qcorrection): stflx(itemp) -= dSSTdt (T_surface - sst),
  !   before the SEA_ICE_NOFLUX cap, &sst_correction dsstdt [cm/day];
  ! CFLX_CORR (cflx_corr, MARBL): the surface fluxes of DIC and DIC_ALT_CO2
  !   are -dCdt (DIC_ALT_CO2_surface - sDIC), those of ALK and ALK_ALT_CO2
  !   -dCdt (ALK_ALT_CO2_surface - sALK) (the CPU's expressions: both
  !   tracers of a pair restore with the _ALT_CO2 tracer's value), applied
  !   by the passive corrector from the first step on (CPU b49a3e63,
  !   claude-docs#116), &dic_alk_correction dcdt [cm/day].
  logical,  public :: sflx_corr = .false., qcorrection = .false., cflx_corr = .false.
  real(rp), public :: dsssdt = 7.777_rp, dsstdt = 7.777_rp, dcdt = 7.777_rp
  real(rp)         :: dsssdt_ms = 0.0_rp, dsstdt_ms = 0.0_rp, dcdt_ms = 0.0_rp   ! piston velocities, m/s
  integer(ip)      :: ic_dic = 0, ic_dica = 0, ic_alk = 0, ic_alka = 0          ! CFLX_CORR passive indices

  public :: read_forcing_namelist
  public :: init_forcing
  public :: set_forces
  public :: frc_stress_af
  public :: init_bgc_forcing
  public :: set_bgc_forces
  public :: init_cdr_lite_forcing, set_cdr_lite_forces
  public :: init_cflx_corr
  public :: cyc_bracket
  public :: uwnd, vwnd
  public :: write_toy_flux
  public :: write_toy_bulk

contains

  subroutine read_forcing_namelist(unit, nml_dir)
    integer,          intent(in) :: unit
    character(len=*), intent(in) :: nml_dir
    integer :: ios, i
    character(len=512) :: msg
    character(len=512) :: frcfiles(max_frc), frcfile(max_frc)
    logical :: interp_bulk_frc, check_bulk_frc_units
    namelist /forcing_files/ frcfiles, frcfile
    namelist /surf_frc_settings/ interp_flux_frc, interp_bulk_frc, check_bulk_frc_units, sflx_corr, &
                                 qcorrection, cflx_corr
    namelist /sss_correction/ dsssdt
    namelist /sst_correction/ dsstdt
    namelist /dic_alk_correction/ dcdt
    namelist /flux_frc_settings/ interp_flux_frc
    namelist /bulk_frc_settings/ interp_bulk_frc, check_bulk_frc_units
    real(rp) :: gamma2
    namelist /gamma2_settings/ gamma2

    frcfiles = ''; frcfile = ''
    interp_bulk_frc = .false.; check_bulk_frc_units = .false.
    interp_flux_frc = .false.
    rewind(unit)
    read(unit, nml=forcing_files, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'FORCING_FILES: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    rewind(unit)
    read(unit, nml=surf_frc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'SURF_FRC_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    rewind(unit)
    read(unit, nml=flux_frc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'FLUX_FRC_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    rewind(unit)
    read(unit, nml=bulk_frc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'BULK_FRC_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    rewind(unit)
    read(unit, nml=sss_correction, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'SSS_CORRECTION: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    rewind(unit)
    read(unit, nml=sst_correction, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'SST_CORRECTION: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    rewind(unit)
    read(unit, nml=dic_alk_correction, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'DIC_ALK_CORRECTION: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    ! In single precision, as the CPU (surf_flux.F90: default-real dSSSdt, read from the
    ! namelist, then dSSSdt / (100.*86400.)): about -3.6e-8 relative to the double value.
    ! The same for dSSTdt and dCdt.
    dsssdt_ms = real(real(dsssdt, 4) / (100.0_4 * 86400.0_4), rp)
    dsstdt_ms = real(real(dsstdt, 4) / (100.0_4 * 86400.0_4), rp)
    dcdt_ms = real(real(dcdt, 4) / (100.0_4 * 86400.0_4), rp)
    ! Closed walls: the GPU has free slip only (obc.F90 gamma2 = 1, as the CPU default).
    gamma2 = 1.0_rp
    rewind(unit)
    read(unit, nml=gamma2_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'GAMMA2_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_forcing_namelist'
    endif
    if (gamma2 /= 1.0_rp) error stop 'gamma2_settings: only gamma2 = 1 (free slip) is supported'

    case_nfrc = 0
    do i = 1, max_frc
      if (len_trim(frcfiles(i)) > 0) then
        case_nfrc = case_nfrc + 1
        case_frcfiles(case_nfrc) = resolve_one(nml_dir, frcfiles(i))
      elseif (len_trim(frcfile(i)) > 0) then
        case_nfrc = case_nfrc + 1
        case_frcfiles(case_nfrc) = resolve_one(nml_dir, frcfile(i))
      endif
    enddo
  end subroutine read_forcing_namelist

  subroutine init_forcing()
    integer :: nmiss
    have_flux = .false.
    have_bulk = .false.
    if (case_nfrc < 1) return
    call prefetch_time_axes()
    call bind_series(s_uwnd, 'uwnd', 'time', 'r')
    call bind_series(s_vwnd, 'vwnd', 'time', 'r')
    call bind_series(s_tair, 'Tair', 'time', 'r')
    call bind_series(s_qair, 'qair', 'time', 'r')
    call bind_series(s_prate, 'rain', 'time', 'r')
    call bind_series(s_lwrad, 'lwrad', 'time', 'r')
    call bind_series(s_bswrad, 'swrad', 'time', 'r')
    have_bulk = s_uwnd%ready .and. s_vwnd%ready .and. s_tair%ready .and. s_qair%ready
    if (have_bulk) then
      if (.not. allocated(uwnd)) then
        allocate(uwnd(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(vwnd(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(qair(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(prate(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(lwrad_in(decomp%is:decomp%ie, decomp%js:decomp%je))
      endif
      uwnd = 0.0_rp; vwnd = 0.0_rp; tair = 0.0_rp
      qair = 0.0_rp; prate = 0.0_rp; lwrad_in = 0.0_rp
      write(*,'(a,i0,a,2es14.6)') 'bulk/COARE forcing, Tair records = ', s_tair%ntot, &
           '  days ', s_tair%times(1), s_tair%times(s_tair%ntot)
      ! SFLX_CORR: sea-surface salinity restoring data (bulk forcing only)
      if (.not. allocated(sss)) allocate(sss(decomp%is:decomp%ie, decomp%js:decomp%je))
      sss = 0.0_rp
      if (sflx_corr) then
        call bind_series(s_sss, 'sss', 'sss_time', 'r')
        if (.not. s_sss%ready) error stop 'sflx_corr: sss / sss_time not found in the forcing files'
        write(*,'(a,f8.3,a,es12.4,a)') 'SSS restoring on: dSSSdt =', dsssdt, ' cm/day (', dsssdt_ms, ' m/s)'
      endif
      ! QCORRECTION: sea-surface temperature restoring data
      if (.not. allocated(sst)) allocate(sst(decomp%is:decomp%ie, decomp%js:decomp%je))
      sst = 0.0_rp
      if (qcorrection) then
        call bind_series(s_sst, 'sst', 'sst_time', 'r')
        if (.not. s_sst%ready) error stop 'qcorrection: sst / sst_time not found in the forcing files'
        write(*,'(a,f8.3,a,es12.4,a)') 'SST restoring on: dSSTdt =', dsstdt, ' cm/day (', dsstdt_ms, ' m/s)'
      endif
      ! CFLX_CORR: sea-surface DIC and ALK restoring data (the tracers: init_cflx_corr)
      if (.not. allocated(sdic)) allocate(sdic(decomp%is:decomp%ie, decomp%js:decomp%je), &
                                          salk(decomp%is:decomp%ie, decomp%js:decomp%je))
      sdic = 0.0_rp; salk = 0.0_rp
      if (cflx_corr) then
        call bind_series(s_sdic, 'sDIC', 'sDIC_time', 'r')
        call bind_series(s_salk, 'sALK', 'sALK_time', 'r')
        if (.not. (s_sdic%ready .and. s_salk%ready)) &
          error stop 'cflx_corr: sDIC / sDIC_time or sALK / sALK_time not found in the forcing files'
        write(*,'(a,f8.3,a,es12.4,a)') 'DIC/ALK restoring on: dCdt =', dcdt, ' cm/day (', dcdt_ms, ' m/s)'
      endif
      call warn_unused_restoring()
      return
    endif
    if (sflx_corr) error stop 'sflx_corr needs bulk forcing (as roms_bugfix SFLX_CORR with BULK_FRC)'
    if (qcorrection) error stop 'qcorrection needs bulk forcing (as roms_bugfix QCORRECTION with BULK_FRC)'
    if (cflx_corr) error stop 'cflx_corr needs bulk forcing (as roms_bugfix CFLX_CORR with BULK_FRC)'
    call warn_unused_restoring()
    nmiss = 0
    if (s_uwnd%ready .or. s_vwnd%ready .or. s_tair%ready .or. s_qair%ready) then
      if (.not. s_uwnd%ready) nmiss = nmiss + 1
      if (.not. s_vwnd%ready) nmiss = nmiss + 1
      if (.not. s_tair%ready) nmiss = nmiss + 1
      if (.not. s_qair%ready) nmiss = nmiss + 1
      write(*,'(a,i0,a)') 'ERROR: incomplete bulk forcing (', nmiss, &
           ' of uwnd/vwnd/Tair/qair missing); flux fallback is disabled'
      error stop 'init_forcing: bulk is the default; will not use flux forcing'
    endif
    call bind_series(s_sustr, 'sustr', 'sms_time', 'u')
    call bind_series(s_svstr, 'svstr', 'sms_time', 'v')
    call bind_series(s_shflx, 'shflux', 'shf_time', 'r')
    call bind_series(s_swflx, 'swflux', 'swf_time', 'r')
    call bind_series(s_swrad, 'swrad', 'srf_time', 'r')
    have_flux = s_sustr%ready .and. s_svstr%ready .and. s_shflx%ready
    if (have_flux) write(*,'(a,i0,a)') 'flux forcing (no bulk vars): ', case_nfrc, ' files'
  end subroutine init_forcing

  subroutine set_forces()
    real(rp) :: tneed
    logical, save :: logged_bulk = .false.
    if (have_bulk) then
      tneed = frc_modtime()
      call frc_batch_begin()
      call advance_series(s_uwnd, tneed)
      call advance_series(s_vwnd, tneed)
      call advance_series(s_tair, tneed)
      call advance_series(s_qair, tneed)
      call advance_series(s_prate, tneed)
      call advance_series(s_lwrad, tneed)
      call advance_series(s_bswrad, tneed)
      if (sflx_corr) call advance_series(s_sss, tneed)
      if (qcorrection) call advance_series(s_sst, tneed)
      if (cflx_corr) then
        call advance_series(s_sdic, tneed)
        call advance_series(s_salk, tneed)
      endif
      call frc_batch_flush()
      call fill_r(s_uwnd, tneed, uwnd)
      call fill_r(s_vwnd, tneed, vwnd)
      call fill_r(s_tair, tneed, tair)
      call fill_r(s_qair, tneed, qair)
      if (s_prate%ready) call fill_r(s_prate, tneed, prate)
      if (s_lwrad%ready) call fill_r(s_lwrad, tneed, lwrad_in)
      if (s_bswrad%ready) then
        call fill_r(s_bswrad, tneed, srflx)
        call divide_2d(srflx, rho0_case * Cp)
      else
        call set_2d(srflx, 0.0_rp)
      endif
      if (.not. logged_bulk) then
        call sync_arch()
        write(*,'(a,es14.6,a,2es14.6,a,2i8)') 'bulk interp tdays=', tneed, &
             '  Tair times=', s_tair%times(s_tair%it1), s_tair%times(s_tair%it2), &
             '  recs=', s_tair%it1, s_tair%it2
        write(*,'(a,2es14.6,a,2es14.6)') 'bulk Tair range=', &
             minval(tair), maxval(tair), '  uwnd range=', minval(uwnd), maxval(uwnd)
        logged_bulk = .true.
      endif
      if (sflx_corr) call fill_r(s_sss, tneed, sss)
      if (qcorrection) call fill_r(s_sst, tneed, sst)
      call calc_bulk_coare(uwnd, vwnd, tair, qair, prate, lwrad_in, sflx_corr, dsssdt_ms, sss, &
                           qcorrection, dsstdt_ms, sst)
      if (cflx_corr) then
        ! the passive corrector applies them (the one surface-flux array
        ! with CDR_LITE's, cdrl_psf); from the predicted state at n+1/2,
        ! as the CPU's set_forces before step3d_t
        call fill_r(s_sdic, tneed, sdic)
        call fill_r(s_salk, tneed, salk)
        call calc_cflx_corr(tp, ntp, nrhs, cdrl_psf, sdic, salk, dcdt_ms, ic_dic, ic_dica, ic_alk, ic_alka)
      endif
      return
    endif
    if (.not. have_flux) return
    tneed = frc_modtime()
    call frc_batch_begin()
    call advance_series(s_sustr, tneed)
    call advance_series(s_svstr, tneed)
    call advance_series(s_shflx, tneed)
    call advance_series(s_swflx, tneed)
    call advance_series(s_swrad, tneed)
    call frc_batch_flush()
    ! The unit conversions are kernels, stream-ordered after the fill
    ! kernels and before the step's later readers of these fields (host
    ! statements on the managed arrays here raced with both). Each is the
    ! same single IEEE operation as the former host statement.
    call fill_u(s_sustr, tneed, sustr)
    call fill_v(s_svstr, tneed, svstr)
    call fill_r(s_shflx, tneed, stflx(:,:,itemp))
    call divide_2d(sustr, rho0_case)
    call divide_2d(svstr, rho0_case)
    call divide_2d(stflx(:,:,itemp), rho0_case * Cp)
    call set_2d(stflx(:,:,isalt), 0.0_rp)
    if (s_swflx%ready) then
      call fill_r(s_swflx, tneed, swflx)
      call scale_2d(swflx, -cmday2ms)        ! = -swflx * cmday2ms, bitwise
    endif
    if (s_swrad%ready) then
      call fill_r(s_swrad, tneed, srflx)
      call divide_2d(srflx, rho0_case * Cp)
    endif
  end subroutine set_forces

  subroutine frc_stress_af(alf)
    ! Implicit free surface, fs_stress_time = 'af' (CPU 19383afc/f77dbc19 fs_stress_af, FS_SPEC 3.1, 9b-F):
    ! the surface stress in rufrc, rvfrc (step3d_uv1: dm dn sustr at n+1/2) moved to t' = t_{n+alpha_f}
    ! for the 2-D step: rufrc += dm_u dn_u (sustr(t') - sustr(n+1/2)) on the owned u-faces (v likewise).
    ! The stress data come from the forcing slots already on the device (no read, no advance; past
    ! the second slot a linear extrapolation, as the CPU's frc_data_at):
    !  - bulk forcing (bulk_stress_shift): per rho point the step's wind part k W of sustr_r, svstr_r
    !    (the current-feedback term removed with the same formula; u, v(nrhs) are still the n+1/2
    !    predictor), k = (wx uwnd + wy vwnd)/|W|^2, and the stress at the wind W' of t' with the drag
    !    coefficient and air density held, gustiness 0.5 m/s: dsr = k (sqrt((|W'|^2 + 0.25)/(|W|^2
    !    + 0.25)) W' - W); dsu = 0.5 (dsr(i-1) + dsr(i)) umask. (roms_gpu has no TAU_CORRECTION.)
    !  - flux forcing (flux_stress_shift): dsu = sustr_data(t')/rho0 - sustr.
    ! rufrc is not read after the barotropic step; the 3-D step keeps the n+1/2 stress.
    real(rp), intent(in) :: alf
    real(rp), parameter :: sh_slope = -0.0029_rp, sh_offset = 0.008_rp, sh_wmin = 3.0_rp, sh_sref = -0.0027_rp
    real(rp), parameter :: sh_gus2 = 0.25_rp
    real(rp), WORK_ATTR allocatable, save :: w1(:,:), w2(:,:), dsr(:,:), dsrv(:,:)
    real(rp) :: tsh, r0, cf, wx, wy, ww, k, rat, uw, vw
    integer(ip) :: i, j, nx, ny, nzz, n_rhs, istrU, jstrV
    if (.not. (have_bulk .or. have_flux)) return
    if (.not. allocated(w1)) then
      allocate(w1(decomp%is:decomp%ie, decomp%js:decomp%je), w2(decomp%is:decomp%ie, decomp%js:decomp%je), &
               dsr(decomp%is:decomp%ie, decomp%js:decomp%je), dsrv(decomp%is:decomp%ie, decomp%js:decomp%je))
      w1 = 0.0_rp; w2 = 0.0_rp; dsr = 0.0_rp; dsrv = 0.0_rp
    endif
    tsh = (time + alf * dt) * sec2day        ! time is t_n during the step
    nx = decomp%nx; ny = decomp%ny; nzz = decomp%nz; n_rhs = nrhs; r0 = rho0_case
    istrU = merge(2, 1, decomp%edge_w); jstrV = merge(2, 1, decomp%edge_s)
    if (have_bulk) then
      call fill_at(s_uwnd, tsh, w1, 'r')
      call fill_at(s_vwnd, tsh, w2, 'r')
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 0, ny
        do i = 0, nx
          uw = uwnd(i, j); vw = vwnd(i, j)
          ww = sqrt(uw * uw + vw * vw)
          if (ww > sh_wmin) then
            cf = sh_slope * ww + sh_offset
          else
            cf = sh_sref
          endif
          wx = sustr_r(i, j) - cf * 0.5_rp * (u(i, j, nzz, n_rhs) + u(i + 1, j, nzz, n_rhs)) / r0
          wy = svstr_r(i, j) - cf * 0.5_rp * (v(i, j, nzz, n_rhs) + v(i, j + 1, nzz, n_rhs)) / r0
          ww = uw**2 + vw**2
          k = 0.0_rp
          if (ww > 1.e-8_rp) k = (wx * uw + wy * vw) / ww
          rat = sqrt((w1(i, j)**2 + w2(i, j)**2 + sh_gus2) / (ww + sh_gus2))
          dsr(i, j) = k * (rat * w1(i, j) - uw)
          dsrv(i, j) = k * (rat * w2(i, j) - vw)
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = istrU, nx
          rufrc(i, j) = rufrc(i, j) + dm_u(i, j) * dn_u(i, j) * (0.5_rp * (dsr(i - 1, j) + dsr(i, j)) * umask(i, j))
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = jstrV, ny
        do i = 1, nx
          rvfrc(i, j) = rvfrc(i, j) + dm_v(i, j) * dn_v(i, j) * (0.5_rp * (dsrv(i, j - 1) + dsrv(i, j)) * vmask(i, j))
        enddo
      enddo
    else
      call fill_at(s_sustr, tsh, w1, 'u')
      call fill_at(s_svstr, tsh, w2, 'v')
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = istrU, nx
          rufrc(i, j) = rufrc(i, j) + dm_u(i, j) * dn_u(i, j) * (w1(i, j) / r0 - sustr(i, j))
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = jstrV, ny
        do i = 1, nx
          rvfrc(i, j) = rvfrc(i, j) + dm_v(i, j) * dn_v(i, j) * (w2(i, j) / r0 - svstr(i, j))
        enddo
      enddo
    endif
  end subroutine frc_stress_af

  subroutine fill_at(s, t, dest, gtype)
    ! The field of the two slots of s interpolated to time t [days] with set_forces' weights
    ! (linear; outside the bracket a linear extrapolation), without advancing the series
    ! (the CPU's frc_data_at). NaN -> 0. The layout as fill_any (bind_series): a field on the
    ! grid over this tile's window, or a 2x-coarse field interpolated from the global index.
    type(series2d), intent(inout), target :: s
    real(rp),       intent(in)    :: t
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    character(len=1), intent(in)  :: gtype
    real(rp) :: c1, c2
    if (.not. s%ready) return
    call weights(s, t, c1, c2)
    if (s%lay == LAY_GRID) then
      call interp_on_grid(s%buf, s%n1, s%n2, c1, c2, dest, s%li0, s%lj0)
    else
      call interp_coarse(s%buf, s%n1, s%n2, c1, c2, dest, gtype, .false.)
    endif
  end subroutine fill_at

  subroutine prefetch_time_axes()
    ! Collective over the compute ranks. The 1-D real variables of every
    ! forcing file that are time axes the binders ask for (TAXES; any other
    ! axis is read by its binder, so the list only affects speed) land in
    ! tcache on every rank. They are read through a handle opened with a
    ! small block size: in a classic file every value of a record variable
    ! sits in its own record, and the default 4 MB block per value made
    ! a 744-record axis cost ~3 GB of reads.
    integer :: i, nc, me, owner, ncid, nv, v, ndims, xtype, dimids(nf90_max_var_dims), n, k, cnt, k0, csz
    integer :: nfirst(max_frc), nlast(max_frc), owner_of(max_frc)
    integer(8) :: fsize(max_frc)
    integer(8), allocatable :: load(:)
    logical :: done(max_frc)
    real(8) :: c0, c1, c2
    character(len=64) :: vname
    integer :: ierr
    nc = comm_size(); me = comm_rank()
    allocate(load(0:nc - 1))
    c0 = wall_time()
    ! Owners by file size (largest first to the least-loaded rank): the
    ! read cost scales with the record count, i.e. with the file size.
    do i = 1, case_nfrc
      inquire(file=trim(case_frcfiles(i)), size=fsize(i))
      if (fsize(i) < 0) fsize(i) = 0
    enddo
    load = 0
    done = .false.
    do k = 1, case_nfrc
      i = maxloc(fsize(1:case_nfrc), 1, mask=.not. done(1:case_nfrc))
      done(i) = .true.
      owner_of(i) = minloc(load(0:nc - 1), 1) - 1
      load(owner_of(i)) = load(owner_of(i)) + fsize(i) + 1
    enddo
    ! local reads first (all files of this rank), then the broadcasts
    do i = 1, case_nfrc
      owner = owner_of(i)
      if (me == owner) then
        nfirst(i) = ntc + 1
        csz = AXIS_BLOCK
        call nc_check(nf90_open(trim(case_frcfiles(i)), nf90_nowrite, ncid, chunksize=csz), &
                      'open '//trim(case_frcfiles(i)))
        call nc_check(nf90_inquire(ncid, nvariables=nv), 'inquire forcing file')
        do v = 1, nv
          call nc_check(nf90_inquire_variable(ncid, v, name=vname, xtype=xtype, ndims=ndims, dimids=dimids), 'inquire var')
          if (ndims /= 1) cycle
          if (xtype /= nf90_double .and. xtype /= nf90_float) cycle
          if (.not. any(TAXES == vname)) cycle
          call nc_check(nf90_inquire_dimension(ncid, dimids(1), len=n), 'dim len')
          if (n < 1) cycle
          if (ntc >= MAXTC) error stop 'prefetch_time_axes: raise MAXTC'
          ntc = ntc + 1
          tcache(ntc)%ifile = i; tcache(ntc)%name = vname; tcache(ntc)%n = n
          allocate(tcache(ntc)%t(n))
          call nc_check(nf90_get_var(ncid, v, tcache(ntc)%t), 'get '//trim(vname))
        enddo
        call nc_close(ncid)
        nlast(i) = ntc
      endif
    enddo
    c1 = wall_time()
#ifdef USE_MPI
    do i = 1, case_nfrc
      owner = owner_of(i)
      if (me == owner) then
        cnt = nlast(i) - nfirst(i) + 1
        k0 = nfirst(i) - 1
      else
        cnt = 0
        k0 = ntc
      endif
      call MPI_Bcast(cnt, 1, MPI_INTEGER, owner, comm_compute(), ierr)
      do k = 1, cnt
        if (me /= owner) then
          if (ntc >= MAXTC) error stop 'prefetch_time_axes: raise MAXTC'
          ntc = ntc + 1
          tcache(ntc)%ifile = i
        endif
        call MPI_Bcast(tcache(k0 + k)%name, 64, MPI_CHARACTER, owner, comm_compute(), ierr)
        call MPI_Bcast(tcache(k0 + k)%n, 1, MPI_INTEGER, owner, comm_compute(), ierr)
        if (me /= owner) allocate(tcache(k0 + k)%t(tcache(k0 + k)%n))
        call MPI_Bcast(tcache(k0 + k)%t, tcache(k0 + k)%n, MPI_REAL8, owner, comm_compute(), ierr)
      enddo
    enddo
#endif
    c2 = wall_time()
    write(*,'(a,i0,a,i0,a,i0,a,f7.2,a,f7.2,a)') 'forcing: rank ', me, ' prefetched ', ntc, ' time axes from ', case_nfrc, &
         ' files: reads ', (c1 - c0), ' s  broadcasts ', (c2 - c1), ' s'
  end subroutine prefetch_time_axes

  logical function cached_times(i, tname, tloc) result(found)
    ! The prefetched axis tname of file i into tloc (sizes must agree).
    integer, intent(in) :: i
    character(len=*), intent(in) :: tname
    real(rp), intent(out) :: tloc(:)
    integer :: k
    found = .false.
    do k = 1, ntc
      if (tcache(k)%ifile == i .and. trim(tcache(k)%name) == trim(tname)) then
        if (tcache(k)%n /= size(tloc)) return
        tloc = tcache(k)%t
        found = .true.
        return
      endif
    enddo
  end function cached_times

  function frc_file_ncid(i) result(ncid)
    ! Read-only handle for forcing file i, opened once and kept until
    ! close_frc_cache: the series binders scan every file per variable.
    integer, intent(in) :: i
    integer :: ncid
    if (frc_cache(i) < 0) call nc_open_read(case_frcfiles(i), frc_cache(i))
    ncid = frc_cache(i)
  end function frc_file_ncid

  subroutine close_frc_cache()
    integer :: i
    do i = 1, case_nfrc
      if (frc_cache(i) >= 0) call nc_close(frc_cache(i))
      frc_cache(i) = -1
    enddo
  end subroutine close_frc_cache

  subroutine bind_series(s, vname, tname, gtype, cpu_order)
    ! cpu_order: the records in the order the CPU's find_new_record meets
    ! them (files in list order, each file's records in order), keeping
    ! only records later than the last one kept, and ending at the first
    ! file that has the variable but not its time axis (where the CPU's
    ! search would stop). Otherwise every file's records, sorted by time.
    type(series2d),   intent(inout) :: s
    character(len=*), intent(in)    :: vname, tname, gtype
    logical, optional, intent(in)   :: cpu_order
    integer :: i, ncid, nxi, neta, nrec, tax, k, ntot
    real(rp), allocatable :: tloc(:)
    real(rp) :: cyc
    logical :: lcyc, cord
    cord = .false.
    if (present(cpu_order)) cord = cpu_order
    s%vname = trim(vname)
    s%tname = trim(tname)
    s%gtype = gtype
    s%ready = .false.
    s%ntot = 0
    ntot = 0
    do i = 1, case_nfrc
      ncid = frc_file_ncid(i)   ! cached handle, closed by close_frc_cache
      if (nc_has_var(ncid, vname)) then
        call nc_forcing_layout(ncid, vname, tname, nxi, neta, nrec, tax)
        ntot = ntot + max(nrec, 1)
        if (s%n1 == 0) then
          s%time_axis = tax
          call classify_layout(s, gtype, nxi, neta, case_frcfiles(i))
        elseif (nxi /= s%fn1 .or. neta /= s%fn2) then
          write(*,'(3a,2(i0,a),2a,2(i0,a))') 'bind_series: ', trim(vname), ' is ', nxi, ' x ', neta, &
               ' (xi x eta) in ', trim(case_frcfiles(i)), ' but ', s%fn1, ' x ', s%fn2, ' in an earlier file'
          error stop 'bind_series: forcing variable with different shapes in different files'
        endif
      endif
    enddo
    if (first_bind) call wall_stamp('bind pass1 '//trim(vname))
    if (ntot < 1) return
    allocate(s%times(ntot), s%ifile(ntot), s%irec(ntot), s%paths(ntot))
    s%ntot = 0
    do i = 1, case_nfrc
      ncid = frc_file_ncid(i)   ! cached handle, closed by close_frc_cache
      if (nc_has_var(ncid, vname)) then
        if (cord .and. .not. nc_has_var(ncid, tname)) then
          if (s%ntot < 1) then
            write(*,'(5a)') 'bind_series: ', trim(vname), ' first found in ', trim(case_frcfiles(i)), &
                 ' without its time axis '//trim(tname)
            error stop 'bind_series: forcing variable without its time axis'
          endif
          exit
        endif
        call nc_forcing_layout(ncid, vname, tname, nxi, neta, nrec, tax)
        nrec = max(nrec, 1)
        allocate(tloc(nrec))
        if (nc_has_var(ncid, tname)) then
          if (.not. cached_times(i, tname, tloc)) call nc_get_1d(ncid, tname, tloc)
        else
          tloc = 0.0_rp
        endif
        do k = 1, nrec
          if (cord .and. s%ntot >= 1) then
            if (.not. (tloc(k) > s%times(s%ntot))) cycle
          endif
          s%ntot = s%ntot + 1
          s%times(s%ntot) = tloc(k)
          s%ifile(s%ntot) = i
          s%irec(s%ntot) = k
          s%paths(s%ntot) = case_frcfiles(i)
        enddo
        deallocate(tloc)
      endif
    enddo
    if (first_bind) call wall_stamp('bind pass2 '//trim(vname))
    if (s%ntot < 1) return
    if (.not. cord) call sort_catalog(s)
    ! climatology? cycle_length on the time axis of the first file with the variable
    s%cycle = -1.0_rp
    do i = 1, case_nfrc
      ncid = frc_file_ncid(i)   ! cached handle, closed by close_frc_cache
      if (nc_has_var(ncid, vname) .and. nc_has_var(ncid, tname)) then
        call nc_get_att_real(ncid, tname, 'cycle_length', cyc, lcyc)
        if (lcyc .and. cyc > 0.0_rp) s%cycle = cyc
        exit
      endif
    enddo
    if (s%cycle > 0.0_rp) write(*,'(3a,f10.3)') 'forcing ', trim(vname), ' is cyclic, cycle_length =', s%cycle
    if (first_bind) call wall_stamp('bind pass3 '//trim(vname))
    s%ready = .true.
    allocate(s%buf(s%n1, s%n2, 2))
    s%buf = 0.0_rp
    if (s%ntot >= 2) then
      ! No bracket yet: the first advance_series (the first set_forces /
      ! set_bgc_forces) reads the two records it needs.
      s%it1 = 0
      s%it2 = 0
    else
      s%it1 = 1
      s%it2 = 1
      call read_catalog_rec(s, s%it1, 1)
      call read_catalog_rec(s, s%it2, 2)
    endif
    if (first_bind) call wall_stamp('bound '//trim(vname))
    first_bind = .false.
  end subroutine bind_series

  subroutine classify_layout(s, gtype, nxi, neta, path)
    ! The layout of a forcing field from its file shape (xi x eta), as the
    ! CPU reads it (roms_read_write fill_frc_slice_surf):
    !   on the model grid, read window by window:
    !     r: xi_rho x eta_rho = (gnx+2) x (gny+2), or the interior gnx x gny
    !     u: xi_u x eta_rho   = (gnx+1) x (gny+2), or (gnx+1) x gny
    !     v: xi_rho x eta_v   = (gnx+2) x (gny+1), or gnx x (gny+1)
    !   2x coarse (the CPU's interp_*_frc inputs; gxc = gnx/2, gyc = gny/2):
    !     r: (gxc+2) x (gyc+2), u: (gxc+1) x (gyc+2), v: (gxc+2) x (gyc+1)
    ! Any other shape stops the run.
    type(series2d),   intent(inout) :: s
    character(len=*), intent(in)    :: gtype, path
    integer,          intent(in)    :: nxi, neta
    integer :: gx, gy, gxc, gyc
    gx = int(decomp%gnx); gy = int(decomp%gny)
    gxc = gx / 2; gyc = gy / 2
    s%fn1 = nxi; s%fn2 = neta
    s%lay = LAY_GRID
    s%fi0 = int(decomp%i0); s%fj0 = int(decomp%j0)
    if (gtype == 'r' .and. nxi == gx + 2 .and. neta == gy + 2) then
      s%li0 = 0; s%lj0 = 0
    elseif (gtype == 'r' .and. nxi == gx .and. neta == gy) then
      s%li0 = 1; s%lj0 = 1
    elseif (gtype == 'u' .and. nxi == gx + 1 .and. neta == gy + 2) then
      s%li0 = 1; s%lj0 = 0
    elseif (gtype == 'u' .and. nxi == gx + 1 .and. neta == gy) then
      s%li0 = 1; s%lj0 = 1
    elseif (gtype == 'v' .and. nxi == gx + 2 .and. neta == gy + 1) then
      s%li0 = 0; s%lj0 = 1
    elseif (gtype == 'v' .and. nxi == gx .and. neta == gy + 1) then
      s%li0 = 1; s%lj0 = 1
    elseif ((gtype == 'r' .and. nxi == gxc + 2 .and. neta == gyc + 2) .or. &
            (gtype == 'u' .and. nxi == gxc + 1 .and. neta == gyc + 2) .or. &
            (gtype == 'v' .and. nxi == gxc + 2 .and. neta == gyc + 1)) then
      s%lay = LAY_COARSE
      s%fi0 = 1; s%fj0 = 1
      s%n1 = nxi; s%n2 = neta
      if (comm_rank() == 0) write(*,'(3a,2(i0,a))') 'forcing ', trim(s%vname), ': 2x-coarse input (', nxi, ' x ', &
           neta, '), interpolated to the model grid'
      return
    else
      write(*,'(5a,2(i0,a))') 'bind_series: ', trim(s%vname), ' (', gtype, '-point) has the shape ', nxi, ' x ', neta, &
           ' (xi x eta) in '//trim(path)
      write(*,'(a,2(i0,a))') '  the model grid is LLm x MMm = ', gx, ' x ', gy, '; accepted shapes:'
      write(*,'(a,6(i0,a))') '  rho: ', gx + 2, ' x ', gy + 2, ' or ', gx, ' x ', gy, ', 2x coarse ', gxc + 2, ' x ', gyc + 2
      write(*,'(a,6(i0,a))') '  u:   ', gx + 1, ' x ', gy + 2, ' or ', gx + 1, ' x ', gy, ', 2x coarse ', gxc + 1, ' x ', gyc + 2
      write(*,'(a,6(i0,a))') '  v:   ', gx + 2, ' x ', gy + 1, ' or ', gx, ' x ', gy + 1, ', 2x coarse ', gxc + 2, ' x ', gyc + 1
      error stop 'bind_series: forcing field shape matches neither the model grid nor the 2x-coarse grid'
    endif
    s%n1 = nxi - gx + int(decomp%nx)
    s%n2 = neta - gy + int(decomp%ny)
  end subroutine classify_layout

  subroutine sort_catalog(s)
    type(series2d), intent(inout) :: s
    integer :: i, j, ifile, irec
    real(rp) :: tswap
    character(len=512) :: pswap
    do i = 2, s%ntot
      tswap = s%times(i)
      ifile = s%ifile(i)
      irec = s%irec(i)
      pswap = s%paths(i)
      j = i - 1
      do while (j >= 1)
        if (s%times(j) <= tswap) exit
        s%times(j + 1) = s%times(j)
        s%ifile(j + 1) = s%ifile(j)
        s%irec(j + 1) = s%irec(j)
        s%paths(j + 1) = s%paths(j)
        j = j - 1
      enddo
      s%times(j + 1) = tswap
      s%ifile(j + 1) = ifile
      s%irec(j + 1) = irec
      s%paths(j + 1) = pswap
    enddo
  end subroutine sort_catalog

  subroutine read_catalog_rec(s, idx, slot)
    type(series2d), intent(inout), target :: s
    integer,        intent(in)    :: idx, slot
    integer :: ncid
    real(rp), allocatable :: tmp(:,:)
    if (idx < 1 .or. idx > s%ntot) return
    if (batching) then
      if (npend >= MAXPEND) error stop 'read_catalog_rec: too many deferred reads (raise MAXPEND)'
      npend = npend + 1
      pend(npend)%s => s
      pend(npend)%idx = idx
      pend(npend)%slot = slot
      return
    endif
    ! Read into host memory, then one bulk copy into the managed buffer:
    ! netCDF writing straight into managed memory faults page by page.
    allocate(tmp(s%n1, s%n2))
    if (s%ifile(idx) >= 1 .and. s%ifile(idx) <= max_frc .and. frc_mode > 0) then
      ncid = frc_run_ncid(s%ifile(idx))
      call nc_read_forcing_2d(ncid, s%vname, tmp, s%n1, s%n2, s%irec(idx), s%time_axis, s%fi0, s%fj0)
    else
      call nc_open_read(s%paths(idx), ncid)
      call nc_read_forcing_2d(ncid, s%vname, tmp, s%n1, s%n2, s%irec(idx), s%time_axis, s%fi0, s%fj0)
      call nc_close(ncid)
    endif
    call put_slot(s, slot, tmp)
    deallocate(tmp)
  end subroutine read_catalog_rec

  subroutine put_slot(s, slot, tmp)
    ! One bulk host-to-device copy of a record into buffer slot `slot`.
    type(series2d), intent(inout) :: s
    integer,        intent(in)    :: slot
    real(rp),       intent(in)    :: tmp(s%n1, s%n2)
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaMemcpy(s%buf(:,:,slot), tmp, s%n1 * s%n2, cudaMemcpyHostToDevice)
    if (ierr /= 0) error stop 'read_catalog_rec: cudaMemcpy'
#else
    s%buf(:,:,slot) = tmp
#endif
  end subroutine put_slot

  subroutine copy_upper_to_lower(s)
    ! The new lower bracket is the old upper one: slot 1 = slot 2 on the
    ! device (stream-ordered before any later copy into slot 2).
    type(series2d), intent(inout) :: s
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaMemcpy(s%buf(:,:,1), s%buf(:,:,2), s%n1 * s%n2, cudaMemcpyDeviceToDevice)
    if (ierr /= 0) error stop 'copy_upper_to_lower: cudaMemcpy'
#else
    s%buf(:,:,1) = s%buf(:,:,2)
#endif
  end subroutine copy_upper_to_lower

  integer function frc_run_ncid_path(path) result(ncid)
    ! Kept-open read handle for the forcing file with this path (bry_io).
    character(len=*), intent(in) :: path
    integer :: i
    do i = 1, case_nfrc
      if (trim(case_frcfiles(i)) == trim(path)) then
        ncid = frc_run_ncid(i)
        return
      endif
    enddo
    ncid = -1
  end function frc_run_ncid_path

  integer function frc_run_ncid(i) result(ncid)
    ! Read handle for forcing file i, opened on first use and kept open.
    integer, intent(in) :: i
    if (run_ncid(i) < 0) call nc_open_read(case_frcfiles(i), run_ncid(i))
    ncid = run_ncid(i)
  end function frc_run_ncid

  subroutine frc_check_mode()
    character(len=16) :: val
    integer :: stat
    if (frc_mode >= 0) return
    frc_mode = 7
    call get_environment_variable('ROMS_GPU_FRC_BATCH', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) frc_mode
      if (stat /= 0) frc_mode = 7
    endif
    call get_environment_variable('ROMS_GPU_FRC_TIMING', val, status=stat)
    frc_timing = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '1')
    if (comm_rank() == 0) write(*,'(a,i0)') 'forcing: record-read mode (ROMS_GPU_FRC_BATCH) = ', frc_mode
  end subroutine frc_check_mode

  subroutine frc_batch_begin()
    call frc_check_mode()
    npend = 0
    batching = (iand(frc_mode, 2) /= 0)
  end subroutine frc_batch_begin

  subroutine frc_batch_flush()
    ! Collective over the compute ranks (every rank defers the same reads in
    ! the same order: the record choice depends only on the model time).
    ! A read whose record was prefetched takes the arrived window; each
    ! remaining read r is read whole by rank mod(r-1, nranks) and its
    ! windows sent to every rank; time_axis 1 files keep the window reads.
    integer :: k, nc, me, n, nrem, r
    integer :: rem(MAXPEND)
    logical :: whole
    real(8) :: c0, c1
    type(series2d), pointer :: sp
#ifdef USE_MPI
    integer :: comm, ierr, t, gx, gy
    integer, allocatable :: mine(:,:), win(:,:,:), rreq(:), sreq(:,:)
    type(fb_t), allocatable :: rbuf(:), sbuf(:,:)
    real(rp), allocatable :: fbuf(:,:)
#endif
    batching = .false.
    if (npend == 0) return
    c0 = wall_time()
    nc = comm_size(); me = comm_rank()
    nrem = 0
    do k = 1, npend
      sp => pend(k)%s
      if (sp%pf_idx > 0) then
        call pf_complete(sp)
        if (sp%pf_idx == pend(k)%idx) then
          call put_slot(sp, pend(k)%slot, sp%pf_buf)
          deallocate(sp%pf_buf)
          sp%pf_idx = 0
          cycle
        endif
        deallocate(sp%pf_buf)
        sp%pf_idx = 0
      endif
      nrem = nrem + 1
      rem(nrem) = k
    enddo
    whole = (nc > 1)
    do r = 1, nrem
      if (pend(rem(r))%s%time_axis == 1) whole = .false.
    enddo
#ifdef USE_MPI
    if (whole .and. nrem > 0) then
      comm = comm_compute()
      allocate(mine(4, nrem), win(4, nrem, 0:nc - 1))
      do r = 1, nrem
        sp => pend(rem(r))%s
        mine(:, r) = (/sp%n1, sp%n2, sp%fi0, sp%fj0/)
      enddo
      call MPI_Allgather(mine, 4 * nrem, MPI_INTEGER, win, 4 * nrem, MPI_INTEGER, comm, ierr)
      allocate(rbuf(nrem), rreq(nrem), sbuf(0:nc - 1, nrem), sreq(0:nc - 1, nrem))
      sreq = MPI_REQUEST_NULL
      do r = 1, nrem
        sp => pend(rem(r))%s
        n = sp%n1 * sp%n2
        allocate(rbuf(r)%b(n))
        call MPI_Irecv(rbuf(r)%b, n, MPI_REAL8, mod(r - 1, nc), FRC_TAG0 + r, comm, rreq(r), ierr)
      enddo
      do r = 1, nrem
        if (mod(r - 1, nc) /= me) cycle
        sp => pend(rem(r))%s
        call read_whole_record(sp, pend(rem(r))%idx, fbuf, gx, gy)
        do t = 0, nc - 1
          n = win(1, r, t) * win(2, r, t)
          allocate(sbuf(t, r)%b(n))
          call pack_frc_window(fbuf, gx, gy, win(3, r, t), win(4, r, t), win(1, r, t), win(2, r, t), sbuf(t, r)%b)
          call MPI_Isend(sbuf(t, r)%b, n, MPI_REAL8, t, FRC_TAG0 + r, comm, sreq(t, r), ierr)
        enddo
        deallocate(fbuf)
      enddo
      call MPI_Waitall(nrem, rreq, MPI_STATUSES_IGNORE, ierr)
      call MPI_Waitall(nc * nrem, sreq, MPI_STATUSES_IGNORE, ierr)
      do r = 1, nrem
        call put_slot(pend(rem(r))%s, pend(rem(r))%slot, rbuf(r)%b)
        deallocate(rbuf(r)%b)
        do t = 0, nc - 1
          if (allocated(sbuf(t, r)%b)) deallocate(sbuf(t, r)%b)
        enddo
      enddo
      deallocate(mine, win, rbuf, rreq, sbuf, sreq)
    endif
#else
    whole = .false.
#endif
    if (.not. whole) then
      do r = 1, nrem
        call read_catalog_rec(pend(rem(r))%s, pend(rem(r))%idx, pend(rem(r))%slot)
      enddo
    endif
    c1 = wall_time()
    if (frc_timing) write(*,'(a,i0,a,i0,a,i0,a,l1,a,f8.4,a)') 'forcing: rank ', me, '  records ', npend, &
         '  prefetched ', npend - nrem, '  whole ', whole, '  ', (c1 - c0), ' s'
    do k = 1, npend
      pend(k)%s => null()
    enddo
    npend = 0
  end subroutine frc_batch_flush

  subroutine read_whole_record(s, idx, fbuf, gx, gy)
    ! Record idx of series s over the whole file extent (gx x gy), through
    ! the kept-open handle; the same bytes as any window read of it.
    type(series2d), intent(in) :: s
    integer, intent(in) :: idx
    real(rp), allocatable, intent(out) :: fbuf(:,:)
    integer, intent(out) :: gx, gy
    integer :: ncid, varid, ndims, dimids(nf90_max_var_dims)
    ncid = frc_run_ncid(s%ifile(idx))
    call nc_check(nf90_inq_varid(ncid, trim(s%vname), varid), 'var '//trim(s%vname))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids), 'ndims '//trim(s%vname))
    call nc_check(nf90_inquire_dimension(ncid, dimids(1), len=gx), 'dim '//trim(s%vname))
    gy = 1
    if (ndims >= 2) call nc_check(nf90_inquire_dimension(ncid, dimids(2), len=gy), 'dim '//trim(s%vname))
    allocate(fbuf(gx, gy))
    if (ndims >= 3) then
      call nc_check(nf90_get_var(ncid, varid, fbuf, start=(/1, 1, s%irec(idx)/), count=(/gx, gy, 1/)), &
                    'get frc '//trim(s%vname))
    else
      call nc_check(nf90_get_var(ncid, varid, fbuf, start=(/1, 1/), count=(/gx, gy/)), 'get frc '//trim(s%vname))
    endif
  end subroutine read_whole_record

  subroutine pf_complete(s)
    ! Wait for the prefetch messages of s (receive; the owner's sends).
    type(series2d), intent(inout) :: s
    integer :: t
#ifdef USE_MPI
    integer :: ierr
    if (comm_size() > 1) then
      call MPI_Wait(s%pf_rreq, MPI_STATUS_IGNORE, ierr)
      if (s%pf_own) then
        call MPI_Waitall(size(s%pf_sreq), s%pf_sreq, MPI_STATUSES_IGNORE, ierr)
        do t = lbound(s%pf_sbuf, 1), ubound(s%pf_sbuf, 1)
          if (allocated(s%pf_sbuf(t)%b)) deallocate(s%pf_sbuf(t)%b)
        enddo
        deallocate(s%pf_sbuf, s%pf_sreq)
      endif
    endif
#endif
    s%pf_own = .false.
  end subroutine pf_complete

  subroutine pf_add(s, id, cand, n)
    ! Candidate for read-ahead: the record after the upper bracket. id is
    ! the series' fixed number (message tag; one prefetch per series).
    type(series2d), intent(inout), target :: s
    integer,        intent(in)    :: id
    type(pend_t),   intent(inout) :: cand(:)
    integer,        intent(inout) :: n
    integer :: nxt
    if (.not. s%ready .or. s%ntot < 2 .or. s%time_axis == 1 .or. s%pf_idx /= 0) return
    if (s%it2 < 1) return                      ! not advanced yet
    if (s%cycle > 0.0_rp) then
      nxt = mod(s%it2, s%ntot) + 1
    else
      nxt = s%it2 + 1
      if (nxt > s%ntot) return
    endif
    if (n >= size(cand)) return
    n = n + 1
    cand(n)%s => s
    cand(n)%idx = nxt
    cand(n)%slot = id
  end subroutine pf_add

  subroutine frc_prefetch()
    ! Collective over the compute ranks (the candidates depend only on the
    ! model time). Candidate k (k <= nranks) is read by rank k-1.
    type(pend_t) :: cand(32)
    integer :: n, k, nc, me, nb
    real(8) :: c0, c1
    type(series2d), pointer :: sp
    real(rp), allocatable :: fbuf(:,:)
    integer :: gx, gy
#ifdef USE_MPI
    integer :: comm, ierr, t, m
    integer, allocatable :: mine(:,:), win(:,:,:)
#endif
    call frc_check_mode()
    if (iand(frc_mode, 6) /= 6) return
    n = 0
    call pf_add(s_uwnd, 1, cand, n)
    call pf_add(s_vwnd, 2, cand, n)
    call pf_add(s_tair, 3, cand, n)
    call pf_add(s_qair, 4, cand, n)
    call pf_add(s_prate, 5, cand, n)
    call pf_add(s_lwrad, 6, cand, n)
    call pf_add(s_bswrad, 7, cand, n)
    call pf_add(s_sss, 8, cand, n)
    call pf_add(s_sst, 20, cand, n)
    call pf_add(s_sdic, 21, cand, n)
    call pf_add(s_salk, 22, cand, n)
    call pf_add(s_sustr, 9, cand, n)
    call pf_add(s_svstr, 10, cand, n)
    call pf_add(s_shflx, 11, cand, n)
    call pf_add(s_swflx, 12, cand, n)
    call pf_add(s_swrad, 13, cand, n)
    call pf_add(s_dust, 14, cand, n)
    call pf_add(s_iron, 15, cand, n)
    call pf_add(s_nox, 16, cand, n)
    call pf_add(s_nhy, 17, cand, n)
    call pf_add(s_xco2, 18, cand, n)
    call pf_add(s_xco2a, 19, cand, n)
    if (n == 0) return
    c0 = wall_time()
    nc = comm_size(); me = comm_rank()
    nb = min(n, int(nc))
#ifdef USE_MPI
    if (nc > 1) then
      comm = comm_compute()
      allocate(mine(4, nb), win(4, nb, 0:nc - 1))
      do k = 1, nb
        sp => cand(k)%s
        mine(:, k) = (/sp%n1, sp%n2, sp%fi0, sp%fj0/)
      enddo
      call MPI_Allgather(mine, 4 * nb, MPI_INTEGER, win, 4 * nb, MPI_INTEGER, comm, ierr)
      do k = 1, nb
        sp => cand(k)%s
        allocate(sp%pf_buf(sp%n1 * sp%n2))
        call MPI_Irecv(sp%pf_buf, sp%n1 * sp%n2, MPI_REAL8, k - 1, FRC_PF_TAG0 + cand(k)%slot, comm, &
                       sp%pf_rreq, ierr)
        sp%pf_idx = cand(k)%idx
      enddo
      if (me < nb) then
        k = me + 1
        sp => cand(k)%s
        call read_whole_record(sp, cand(k)%idx, fbuf, gx, gy)
        allocate(sp%pf_sbuf(0:nc - 1), sp%pf_sreq(0:nc - 1))
        do t = 0, nc - 1
          m = win(1, k, t) * win(2, k, t)
          if (win(3, k, t) + win(1, k, t) - 1 > gx .or. win(4, k, t) + win(2, k, t) - 1 > gy) &
               error stop 'frc_prefetch: window outside the file extent'
          allocate(sp%pf_sbuf(t)%b(m))
          call pack_frc_window(fbuf, gx, gy, win(3, k, t), win(4, k, t), win(1, k, t), win(2, k, t), sp%pf_sbuf(t)%b)
          call MPI_Isend(sp%pf_sbuf(t)%b, m, MPI_REAL8, t, FRC_PF_TAG0 + cand(k)%slot, comm, sp%pf_sreq(t), ierr)
        enddo
        sp%pf_own = .true.
        deallocate(fbuf)
      endif
      deallocate(mine, win)
    else
#endif
      do k = 1, nb
        sp => cand(k)%s
        call read_whole_record(sp, cand(k)%idx, fbuf, gx, gy)
        allocate(sp%pf_buf(sp%n1 * sp%n2))
        call pack_frc_window(fbuf, gx, gy, sp%fi0, sp%fj0, sp%n1, sp%n2, sp%pf_buf)
        deallocate(fbuf)
        sp%pf_idx = cand(k)%idx
      enddo
#ifdef USE_MPI
    endif
#endif
    c1 = wall_time()
    if (frc_timing) write(*,'(a,i0,a,i0,a,f8.4,a)') 'forcing: rank ', me, '  prefetch ', nb, &
         ' records  ', (c1 - c0), ' s'
  end subroutine frc_prefetch

  subroutine frc_prefetch_drain()
    ! End of run: complete and drop every prefetch in flight.
    call pf_drain(s_uwnd);  call pf_drain(s_vwnd);  call pf_drain(s_tair)
    call pf_drain(s_qair);  call pf_drain(s_prate); call pf_drain(s_lwrad)
    call pf_drain(s_bswrad); call pf_drain(s_sss);  call pf_drain(s_sustr)
    call pf_drain(s_svstr); call pf_drain(s_shflx); call pf_drain(s_swflx)
    call pf_drain(s_swrad); call pf_drain(s_dust);  call pf_drain(s_iron)
    call pf_drain(s_nox);   call pf_drain(s_nhy);   call pf_drain(s_xco2)
    call pf_drain(s_xco2a); call pf_drain(s_sst);   call pf_drain(s_sdic)
    call pf_drain(s_salk)
  end subroutine frc_prefetch_drain

  subroutine init_cflx_corr()
    ! CFLX_CORR: the passive indices of DIC, DIC_ALT_CO2, ALK and
    ! ALK_ALT_CO2 (MARBL), and the passive surface-flux array at full size.
    ! After init_cdr_lite (which sizes cdrl_psf for CDR_LITE).
    integer(ip) :: m
    if (.not. cflx_corr) return
    ic_dic = 0; ic_dica = 0; ic_alk = 0; ic_alka = 0
    do m = 1, n_marbl
      select case (trim(passive_name(m)))
      case ('DIC'); ic_dic = m
      case ('DIC_ALT_CO2'); ic_dica = m
      case ('ALK'); ic_alk = m
      case ('ALK_ALT_CO2'); ic_alka = m
      end select
    enddo
    if (ic_dic == 0 .or. ic_dica == 0 .or. ic_alk == 0 .or. ic_alka == 0) &
      error stop 'cflx_corr needs the MARBL tracers DIC, DIC_ALT_CO2, ALK and ALK_ALT_CO2 (as roms_bugfix CFLX_CORR with MARBL)'
    call cdrl_psf_full(ntp)
    psf_cflx = .true.
    if (comm_rank() == 0) write(*,'(a,4(1x,i0))') 'cflx_corr: passive DIC, DIC_ALT_CO2, ALK, ALK_ALT_CO2 =', &
         ic_dic, ic_dica, ic_alk, ic_alka
  end subroutine init_cflx_corr

  subroutine warn_unused_restoring()
    ! Restoring data in the forcing files with its switch off: a CPU case
    ! built with the cpp key would restore, the GPU would not (the GPU
    ! cannot see cpp keys). Warn, as the data alone do not prove the key.
    ! The variable names are the CPU's (surf_flux.F90; a flux file's
    ! upper-case SST/SSS are not restoring data).
    if (comm_rank() /= 0) return
    if (.not. sflx_corr .and. frc_has_var('sss')) write(*,'(a)') &
      'WARNING: the forcing files hold sss (SSS restoring data) but &surf_frc_settings sflx_corr is off'
    if (.not. qcorrection .and. frc_has_var('sst')) write(*,'(a)') &
      'WARNING: the forcing files hold sst (SST restoring data) but &surf_frc_settings qcorrection is off'
    if (.not. cflx_corr .and. (frc_has_var('sDIC') .or. frc_has_var('sALK'))) write(*,'(a)') &
      'WARNING: the forcing files hold sDIC/sALK (DIC/ALK restoring data) but &surf_frc_settings cflx_corr is off'
  end subroutine warn_unused_restoring

  logical function frc_has_var(name)
    ! Whether any forcing file has the variable (handles of the init cache).
    character(len=*), intent(in) :: name
    integer :: i
    frc_has_var = .false.
    do i = 1, case_nfrc
      if (nc_has_var(frc_file_ncid(i), name)) then
        frc_has_var = .true.
        return
      endif
    enddo
  end function frc_has_var

  subroutine pf_drain(s)
    type(series2d), intent(inout) :: s
    if (s%pf_idx == 0) return
    call pf_complete(s)
    if (allocated(s%pf_buf)) deallocate(s%pf_buf)
    s%pf_idx = 0
  end subroutine pf_drain

  subroutine pack_frc_window(fbuf, gx, gy, i0f, j0f, n1, n2, out)
    integer,  intent(in)  :: gx, gy, i0f, j0f, n1, n2
    real(rp), intent(in)  :: fbuf(gx, gy)
    real(rp), intent(out) :: out(n1, n2)
    out = fbuf(i0f:i0f + n1 - 1, j0f:j0f + n2 - 1)
  end subroutine pack_frc_window

  ! Climatology bracket, as the CPU model (find_new_record_cycle): the
  ! record times are shifted by whole cycles so that the first shifted time
  ! above tneed is record i2; i1 is the one before it (wrapping to the last
  ! record of the previous cycle).
  subroutine cyc_bracket(times, ntot, cycle, tneed, i1, i2, t1, t2)
    integer,  intent(in)  :: ntot
    real(rp), intent(in)  :: times(ntot), cycle, tneed
    integer,  intent(out) :: i1, i2
    real(rp), intent(out) :: t1, t2
    integer :: icycle, k
    if (ntot < 2) then
      i1 = 1; i2 = 1; t1 = tneed; t2 = tneed
      return
    endif
    icycle = floor((tneed - times(ntot)) / cycle) + 1
    i2 = ntot
    do k = 1, ntot
      if (times(k) + real(icycle, rp) * cycle > tneed) then
        i2 = k
        exit
      endif
    enddo
    t2 = times(i2) + real(icycle, rp) * cycle
    if (i2 == 1) then
      i1 = ntot
      t1 = times(ntot) + real(icycle - 1, rp) * cycle
    else
      i1 = i2 - 1
      t1 = times(i1) + real(icycle, rp) * cycle
    endif
  end subroutine cyc_bracket

  subroutine advance_series(s, tneed)
    type(series2d), intent(inout), target :: s
    real(rp),       intent(in)    :: tneed
    integer :: rec, i, old1, old2, i1, i2
    if (.not. s%ready .or. s%ntot < 2) return
    call frc_check_mode()
    if (s%cycle > 0.0_rp) then
      call cyc_bracket(s%times, s%ntot, s%cycle, tneed, i1, i2, s%t1e, s%t2e)
      if (i1 /= s%it1) then
        old2 = s%it2
        s%it1 = i1
        if (i1 == old2 .and. iand(frc_mode, 1) /= 0) then
          call copy_upper_to_lower(s)
        else
          call read_catalog_rec(s, i1, 1)
        endif
      endif
      if (i2 /= s%it2) then
        s%it2 = i2
        call read_catalog_rec(s, i2, 2)
      endif
      return
    endif
    if (s%it1 >= 1) then
      if (s%times(s%it2) >= tneed .and. s%times(s%it1) <= tneed) return
    endif
    rec = 1
    do i = 1, s%ntot - 1
      if (s%times(i) <= tneed .and. s%times(i + 1) >= tneed) then
        rec = i
        exit
      endif
      if (s%times(i) <= tneed) rec = i
    enddo
    rec = max(1, min(rec, s%ntot - 1))
    if (rec == s%it1 .and. rec + 1 == s%it2) return
    old1 = s%it1; old2 = s%it2
    s%it1 = rec
    s%it2 = rec + 1
    if (s%it1 /= old1) then
      if (s%it1 == old2 .and. iand(frc_mode, 1) /= 0) then
        call copy_upper_to_lower(s)
      else
        call read_catalog_rec(s, s%it1, 1)
      endif
    endif
    if (s%it2 /= old2) call read_catalog_rec(s, s%it2, 2)
  end subroutine advance_series

  subroutine weights(s, tneed, c1, c2)
    type(series2d), intent(in)  :: s
    real(rp),       intent(in)  :: tneed
    real(rp),       intent(out) :: c1, c2
    real(rp) :: dtm
    real(rp) :: t1, t2
    if (s%ntot < 2) then
      c1 = 1.0_rp; c2 = 0.0_rp
      return
    endif
    if (s%cycle > 0.0_rp) then
      t1 = s%t1e; t2 = s%t2e
    else
      t1 = s%times(s%it1); t2 = s%times(s%it2)
    endif
    dtm = t2 - t1
    if (abs(dtm) < 1.0e-12_rp) then
      c1 = 1.0_rp; c2 = 0.0_rp
    else
      c1 = (t2 - tneed) / dtm
      c2 = (tneed - t1) / dtm
    endif
  end subroutine weights

  subroutine fill_r(s, tneed, dest, keep_nan)
    ! keep_nan: NaN (missing) values stay NaN, as the CPU's set_frc_data
    ! leaves them; otherwise they become 0.
    type(series2d), intent(inout), target :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    logical, optional, intent(in) :: keep_nan
    logical :: kn
    kn = .false.
    if (present(keep_nan)) kn = keep_nan
    call fill_any(s, tneed, dest, 'r', kn)
  end subroutine fill_r

  subroutine fill_u(s, tneed, dest)
    type(series2d), intent(inout), target :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    call fill_any(s, tneed, dest, 'u', .false.)
  end subroutine fill_u

  subroutine fill_v(s, tneed, dest)
    type(series2d), intent(inout), target :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    call fill_any(s, tneed, dest, 'v', .false.)
  end subroutine fill_v

  subroutine fill_any(s, tneed, dest, gtype, kn)
    ! The field at tneed on the model grid, by a device kernel: a field on
    ! the grid (any of the layouts of classify_layout) over this tile's
    ! window; a 2x-coarse field interpolated to every point of the tile.
    type(series2d), intent(inout), target :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    character(len=1), intent(in)  :: gtype
    logical,        intent(in)    :: kn
    real(rp) :: c1, c2
    if (.not. s%ready) return
    call advance_series(s, tneed)
    call weights(s, tneed, c1, c2)
    if (s%lay == LAY_GRID) then
      if (kn) then
        call interp_on_grid_nan(s%buf, s%n1, s%n2, c1, c2, dest, s%li0, s%lj0)
      else
        call interp_on_grid(s%buf, s%n1, s%n2, c1, c2, dest, s%li0, s%lj0)
      endif
    else
      call interp_coarse(s%buf, s%n1, s%n2, c1, c2, dest, gtype, kn)
    endif
  end subroutine fill_any

  subroutine interp_coarse(b, n1, n2, c1, c2, dest, gtype, kn)
    ! A 2x-coarse field (two buffered records b(:,:,1:2), the whole file
    ! extent) on this tile, as the CPU (roms_read_write coarse2fine, then
    ! set_frc_data's time interpolation): each record is interpolated
    ! bilinearly to the fine point from its GLOBAL index, then the two are
    ! combined in time. With gi the global fine index (local i + i0 - 1):
    !   rho: coarse index 0.5 gi + 0.25 (file index 1 = coarse rho 0)
    !   u (xi) / v (eta): 0.5 gi + 0.5 (file index 1 = coarse u/v 1)
    ! The weights are exact (multiples of 1/4), so the result does not
    ! depend on the tiling. Coarse points beyond the file extent (the last
    ! fine row of an odd LLm / MMm, and the u/v end faces) count as 0, as
    ! the CPU's zero-initialized coarse buffer; NaN propagates as on the
    ! CPU and becomes 0 at the end unless kn (keep NaN).
    ! Filled: rho i = 0:nx+1, j = 0:ny+1; u i = 1:nx+1, j = 0:ny+1;
    ! v i = 0:nx+1, j = 1:ny+1 (the standard on-grid windows).
    integer,          intent(in) :: n1, n2
    real(rp), WORK_ATTR intent(in) :: b(n1, n2, 2)
    real(rp),         intent(in) :: c1, c2
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    character(len=1), intent(in) :: gtype
    logical,          intent(in) :: kn
    integer :: i, j, ib, jb, ie, je, gi0, gj0, ox, oy, ic, jc
    real(rp) :: xo, yo, xg, yg, xl, yl
    real(rp) :: a00, a10, a01, a11, q00, q10, q01, q11, f1, f2, x
    logical :: in0, in1, jn0, jn1
    ib = 0; jb = 0; xo = 0.25_rp; yo = 0.25_rp; ox = 1; oy = 1
    if (gtype == 'u') then
      ib = 1; xo = 0.5_rp; ox = 0
    elseif (gtype == 'v') then
      jb = 1; yo = 0.5_rp; oy = 0
    endif
    ie = int(decomp%nx) + 1; je = int(decomp%ny) + 1
    gi0 = int(decomp%i0) - 1; gj0 = int(decomp%j0) - 1
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = jb, je
      do i = ib, ie
        xg = 0.5_rp * real(i + gi0, rp) + xo
        yg = 0.5_rp * real(j + gj0, rp) + yo
        ic = floor(xg)
        jc = floor(yg)
        xl = xg - real(ic, rp)
        yl = yg - real(jc, rp)
        ic = ic + ox
        jc = jc + oy
        in0 = (ic >= 1 .and. ic <= n1)
        in1 = (ic + 1 >= 1 .and. ic + 1 <= n1)
        jn0 = (jc >= 1 .and. jc <= n2)
        jn1 = (jc + 1 >= 1 .and. jc + 1 <= n2)
        a00 = 0.0_rp; a10 = 0.0_rp; a01 = 0.0_rp; a11 = 0.0_rp
        q00 = 0.0_rp; q10 = 0.0_rp; q01 = 0.0_rp; q11 = 0.0_rp
        if (in0 .and. jn0) then
          a00 = b(ic, jc, 1); q00 = b(ic, jc, 2)
        endif
        if (in1 .and. jn0) then
          a10 = b(ic + 1, jc, 1); q10 = b(ic + 1, jc, 2)
        endif
        if (in0 .and. jn1) then
          a01 = b(ic, jc + 1, 1); q01 = b(ic, jc + 1, 2)
        endif
        if (in1 .and. jn1) then
          a11 = b(ic + 1, jc + 1, 1); q11 = b(ic + 1, jc + 1, 2)
        endif
        f1 = (1.0_rp - yl) * ((1.0_rp - xl) * a00 + xl * a10) + yl * ((1.0_rp - xl) * a01 + xl * a11)
        f2 = (1.0_rp - yl) * ((1.0_rp - xl) * q00 + xl * q10) + yl * ((1.0_rp - xl) * q01 + xl * q11)
        x = c1 * f1 + c2 * f2
        if (.not. kn .and. x /= x) x = 0.0_rp
        dest(i, j) = x
      enddo
    enddo
  end subroutine interp_coarse

  subroutine interp_on_grid(b, n1, n2, c1, c2, dest, i0, j0)
    ! Device time interpolation of two buffered records onto the grid,
    ! NaN (missing) -> 0.
    integer,  intent(in) :: n1, n2, i0, j0
    real(rp), WORK_ATTR intent(in) :: b(n1, n2, 2)
    real(rp), intent(in) :: c1, c2
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: i, j
    real(rp) :: x
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, n2
      do i = 1, n1
        x = c1 * b(i, j, 1) + c2 * b(i, j, 2)
        if (x /= x) x = 0.0_rp
        dest(i0 + i - 1, j0 + j - 1) = x
      enddo
    enddo
  end subroutine interp_on_grid

  subroutine interp_on_grid_nan(b, n1, n2, c1, c2, dest, i0, j0)
    ! As interp_on_grid, with NaN kept (CPU set_frc_data).
    integer,  intent(in) :: n1, n2, i0, j0
    real(rp), WORK_ATTR intent(in) :: b(n1, n2, 2)
    real(rp), intent(in) :: c1, c2
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: i, j
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, n2
      do i = 1, n1
        dest(i0 + i - 1, j0 + j - 1) = c1 * b(i, j, 1) + c2 * b(i, j, 2)
      enddo
    enddo
  end subroutine interp_on_grid_nan

  ! CDR_LITE surface forcing (CPU surf_flux / tracers): the carbonate
  ! sensitivities ddic_dco2, ddic_dalk (on ddic_dco2_time, ddic_dalk_time)
  ! and the surface flux <name>_flx of each CDR_LITE DIC tracer (on
  ! CDR_time; absent everywhere = 0, as the CPU's allow_missing). Records
  ! are cataloged in the CPU's file order (bind_series cpu_order); NaN in
  ! the sensitivities stays NaN (no gas exchange there, as on the CPU).
  subroutine init_cdr_lite_forcing(nd, names, with_ddic, have_dco2, have_dalk, have_flx)
    ! with_ddic false (online sensitivities): the ddic series are not bound.
    integer,          intent(in)  :: nd
    character(len=*), intent(in)  :: names(:)
    logical,          intent(in)  :: with_ddic
    logical,          intent(out) :: have_dco2, have_dalk, have_flx(:)
    integer :: d
    have_dco2 = .false.; have_dalk = .false.; have_flx = .false.
    if (case_nfrc < 1) return
    if (with_ddic) then
      call bind_series(s_ddco2, 'ddic_dco2', 'ddic_dco2_time', 'r', cpu_order=.true.)
      call bind_series(s_ddalk, 'ddic_dalk', 'ddic_dalk_time', 'r', cpu_order=.true.)
    endif
    have_dco2 = s_ddco2%ready; have_dalk = s_ddalk%ready
    if (allocated(s_cflx)) deallocate(s_cflx)
    allocate(s_cflx(max(nd, 1)))
    do d = 1, nd
      call bind_series(s_cflx(d), trim(names(d))//'_flx', 'CDR_time', 'r', cpu_order=.true.)
      have_flx(d) = s_cflx(d)%ready
    enddo
  end subroutine init_cdr_lite_forcing

  subroutine set_cdr_lite_forces(tneed, nd, ddco2, ddalk, cflx)
    ! The fields at tneed. A record bracket that does not cover tneed stops
    ! the run, as the CPU's search does at the end of its records.
    real(rp), intent(in) :: tneed
    integer,  intent(in) :: nd
    real(rp), WORK_ATTR intent(inout) :: ddco2(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: ddalk(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: cflx(decomp%is:decomp%ie, decomp%js:decomp%je, max(nd, 1))
    integer :: d
    call frc_batch_begin()
    call advance_series(s_ddco2, tneed)
    call advance_series(s_ddalk, tneed)
    do d = 1, nd
      call advance_series(s_cflx(d), tneed)
    enddo
    call frc_batch_flush()
    call cover_check(s_ddco2, tneed)
    call cover_check(s_ddalk, tneed)
    if (s_ddco2%ready) call fill_r(s_ddco2, tneed, ddco2, keep_nan=.true.)
    if (s_ddalk%ready) call fill_r(s_ddalk, tneed, ddalk, keep_nan=.true.)
    if (s_ddco2%ready .and. s_ddalk%ready) call missing_sensitivity(tneed, ddco2, ddalk)
    do d = 1, nd
      call cover_check(s_cflx(d), tneed)
      if (s_cflx(d)%ready) call fill_r(s_cflx(d), tneed, cflx(:, :, d))
    enddo
  end subroutine set_cdr_lite_forces

  subroutine missing_sensitivity(tneed, ddco2, ddalk)
    ! The CPU's surf_flux interp_missing_sensitivity: where exactly one of
    ! the two bracketing beta (ddic_dco2) records is missing (not > 0, NaN
    ! included: the writers put 0 where the carbonate solve failed, and the
    ! gas flux treats beta <= 0 as no exchange), the flux coefficients
    ! 1/beta and eta/beta are interpolated in time with 0 for the missing
    ! record instead of beta and eta themselves: beta = beta_valid/w_valid,
    ! eta = eta_valid (w_valid = the valid record's weight; w_valid = 0:
    ! beta = 0), so the gas flux ramps linearly to zero toward the missing
    ! record (interpolating beta shrank it to w_valid*beta_valid, k/beta up
    ! to 1/w_valid times too large). Cells where both records are valid (or
    ! both missing) keep fill_r's values. eta's records are taken by time;
    ! if its bracket is not beta's (the writers give both the same times)
    ! or its layout differs, eta keeps its interpolated value. A 2x-coarse
    ! field (a GPU-only layout for the sensitivities: the CPU reads them on
    ! the grid) is judged per fine point after the spatial interpolation of
    ! each record, in the order of interp_coarse.
    real(rp), intent(in) :: tneed
    real(rp), WORK_ATTR intent(inout) :: ddco2(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: ddalk(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp) :: c1, c2, t1b, t2b, t1e, t2e
    logical :: es
    if (s_ddco2%ntot < 2) return
    call weights(s_ddco2, tneed, c1, c2)
    call bracket_times(s_ddco2, t1b, t2b)
    call bracket_times(s_ddalk, t1e, t2e)
    es = s_ddalk%ntot >= 2 .and. t1e == t1b .and. t2e == t2b .and. s_ddalk%lay == s_ddco2%lay &
         .and. s_ddalk%n1 == s_ddco2%n1 .and. s_ddalk%n2 == s_ddco2%n2 &
         .and. s_ddalk%li0 == s_ddco2%li0 .and. s_ddalk%lj0 == s_ddco2%lj0
    if (s_ddco2%lay == LAY_GRID) then
      call missing_on_grid(s_ddco2%buf, s_ddalk%buf, s_ddco2%n1, s_ddco2%n2, size(s_ddalk%buf, 1), &
                           size(s_ddalk%buf, 2), c1, c2, es, ddco2, ddalk, s_ddco2%li0, s_ddco2%lj0)
    else
      call missing_coarse(s_ddco2%buf, s_ddalk%buf, s_ddco2%n1, s_ddco2%n2, size(s_ddalk%buf, 1), &
                          size(s_ddalk%buf, 2), c1, c2, es, ddco2, ddalk)
    endif
  end subroutine missing_sensitivity

  subroutine bracket_times(s, t1, t2)
    ! The times of the current record bracket (shifted for a climatology).
    type(series2d), intent(in)  :: s
    real(rp),       intent(out) :: t1, t2
    if (s%cycle > 0.0_rp) then
      t1 = s%t1e; t2 = s%t2e
    else
      t1 = s%times(max(s%it1, 1)); t2 = s%times(max(s%it2, 1))
    endif
  end subroutine bracket_times

  subroutine missing_on_grid(bb, be, n1, n2, m1, m2, c1, c2, es, ddco2, ddalk, i0, j0)
    ! missing_sensitivity on this tile's window of an on-grid field.
    integer,  intent(in) :: n1, n2, m1, m2, i0, j0
    real(rp), WORK_ATTR intent(in) :: bb(n1, n2, 2), be(m1, m2, 2)
    real(rp), intent(in) :: c1, c2
    logical,  intent(in) :: es
    real(rp), WORK_ATTR intent(inout) :: ddco2(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: ddalk(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: i, j, m
    real(rp) :: w
    logical :: v1, v2
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, n2
      do i = 1, n1
        v1 = bb(i, j, 1) > 0.0_rp
        v2 = bb(i, j, 2) > 0.0_rp
        if (v1 .neqv. v2) then
          m = 2
          w = c2
          if (v1) then
            m = 1
            w = c1
          endif
          if (w > 0.0_rp) then
            ddco2(i0 + i - 1, j0 + j - 1) = bb(i, j, m) / w
            if (es) ddalk(i0 + i - 1, j0 + j - 1) = be(i, j, m)
          else
            ddco2(i0 + i - 1, j0 + j - 1) = 0.0_rp
          endif
        endif
      enddo
    enddo
  end subroutine missing_on_grid

  subroutine missing_coarse(bb, be, n1, n2, m1, m2, c1, c2, es, ddco2, ddalk)
    ! missing_sensitivity for 2x-coarse rho fields: each record at the fine
    ! point exactly as interp_coarse (gtype 'r') computes it, then the rule.
    integer,  intent(in) :: n1, n2, m1, m2
    real(rp), WORK_ATTR intent(in) :: bb(n1, n2, 2), be(m1, m2, 2)
    real(rp), intent(in) :: c1, c2
    logical,  intent(in) :: es
    real(rp), WORK_ATTR intent(inout) :: ddco2(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: ddalk(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: i, j, ie, je, gi0, gj0, ic, jc, m
    real(rp) :: xg, yg, xl, yl, w, f1, f2, a00, a10, a01, a11, q00, q10, q01, q11
    logical :: in0, in1, jn0, jn1, v1, v2
    ie = int(decomp%nx) + 1; je = int(decomp%ny) + 1
    gi0 = int(decomp%i0) - 1; gj0 = int(decomp%j0) - 1
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, je
      do i = 0, ie
        xg = 0.5_rp * real(i + gi0, rp) + 0.25_rp
        yg = 0.5_rp * real(j + gj0, rp) + 0.25_rp
        ic = floor(xg)
        jc = floor(yg)
        xl = xg - real(ic, rp)
        yl = yg - real(jc, rp)
        ic = ic + 1
        jc = jc + 1
        in0 = (ic >= 1 .and. ic <= n1)
        in1 = (ic + 1 >= 1 .and. ic + 1 <= n1)
        jn0 = (jc >= 1 .and. jc <= n2)
        jn1 = (jc + 1 >= 1 .and. jc + 1 <= n2)
        a00 = 0.0_rp; a10 = 0.0_rp; a01 = 0.0_rp; a11 = 0.0_rp
        q00 = 0.0_rp; q10 = 0.0_rp; q01 = 0.0_rp; q11 = 0.0_rp
        if (in0 .and. jn0) then
          a00 = bb(ic, jc, 1); q00 = bb(ic, jc, 2)
        endif
        if (in1 .and. jn0) then
          a10 = bb(ic + 1, jc, 1); q10 = bb(ic + 1, jc, 2)
        endif
        if (in0 .and. jn1) then
          a01 = bb(ic, jc + 1, 1); q01 = bb(ic, jc + 1, 2)
        endif
        if (in1 .and. jn1) then
          a11 = bb(ic + 1, jc + 1, 1); q11 = bb(ic + 1, jc + 1, 2)
        endif
        f1 = (1.0_rp - yl) * ((1.0_rp - xl) * a00 + xl * a10) + yl * ((1.0_rp - xl) * a01 + xl * a11)
        f2 = (1.0_rp - yl) * ((1.0_rp - xl) * q00 + xl * q10) + yl * ((1.0_rp - xl) * q01 + xl * q11)
        v1 = f1 > 0.0_rp
        v2 = f2 > 0.0_rp
        if (v1 .neqv. v2) then
          m = 2                    ! the valid record: index, weight, value (f2)
          w = c2
          if (v1) then
            m = 1
            w = c1
            f2 = f1
          endif
          if (w > 0.0_rp) then
            ddco2(i, j) = f2 / w
            if (es) then           ! eta's record m at the fine point
              a00 = 0.0_rp; a10 = 0.0_rp; a01 = 0.0_rp; a11 = 0.0_rp
              if (in0 .and. jn0) a00 = be(ic, jc, m)
              if (in1 .and. jn0) a10 = be(ic + 1, jc, m)
              if (in0 .and. jn1) a01 = be(ic, jc + 1, m)
              if (in1 .and. jn1) a11 = be(ic + 1, jc + 1, m)
              ddalk(i, j) = (1.0_rp - yl) * ((1.0_rp - xl) * a00 + xl * a10) + yl * ((1.0_rp - xl) * a01 + xl * a11)
            endif
          else
            ddco2(i, j) = 0.0_rp
          endif
        endif
      enddo
    enddo
  end subroutine missing_coarse

  subroutine cover_check(s, tneed)
    type(series2d), intent(in) :: s
    real(rp),       intent(in) :: tneed
    if (.not. s%ready .or. s%ntot < 2 .or. s%cycle > 0.0_rp) return
    if (tneed < s%times(1) .or. tneed > s%times(s%ntot)) then
      write(*,'(3a,3es16.8)') 'forcing ', trim(s%vname), ': records do not cover the model time', &
           s%times(1), s%times(s%ntot), tneed
      error stop 'set_cdr_lite_forces: no forcing record for the model time'
    endif
  end subroutine cover_check

  ! MARBL surface forcing series. Names/units as the CPU model
  ! (bgc_shared_vars / marbl_driver): dust kg/m2/s, iron nmol/cm2/s ->
  ! mmol/m2/s (x0.01), nox/nhy kg N/m2/s -> mmol/m2/s (x71394.200220751),
  ! xco2_air / xco2_air_alt ppm. Missing series leave the field as is.
  subroutine init_bgc_forcing()
    bgc_frc_on = .false.
    if (case_nfrc < 1) return
    call bind_series(s_dust, 'dust', 'dust_time', 'r')
    call bind_series(s_iron, 'iron', 'iron_time', 'r')
    call bind_series(s_nox, 'nox', 'nox_time', 'r')
    call bind_series(s_nhy, 'nhy', 'nhy_time', 'r')
    call bind_series(s_xco2, 'xco2_air', 'xco2_time', 'r')
    call bind_series(s_xco2a, 'xco2_air_alt', 'xco2_time', 'r')
    bgc_frc_on = s_dust%ready .or. s_iron%ready .or. s_nox%ready .or. s_nhy%ready .or. s_xco2%ready
    write(*,'(a,6l2)') 'MARBL forcing series dust iron nox nhy xco2 xco2_alt ready =', s_dust%ready, &
         s_iron%ready, s_nox%ready, s_nhy%ready, s_xco2%ready, s_xco2a%ready
  end subroutine init_bgc_forcing

  subroutine set_bgc_forces(tneed, dust, iron, nox, nhy, xco2, xco2a)
    real(rp), intent(in) :: tneed
    real(rp), WORK_ATTR intent(inout) :: dust(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: iron(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: nox(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: nhy(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: xco2(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: xco2a(decomp%is:decomp%ie, decomp%js:decomp%je)
    if (.not. bgc_frc_on) return
    call frc_batch_begin()
    call advance_series(s_dust, tneed)
    call advance_series(s_iron, tneed)
    call advance_series(s_nox, tneed)
    call advance_series(s_nhy, tneed)
    call advance_series(s_xco2, tneed)
    call advance_series(s_xco2a, tneed)
    call frc_batch_flush()
    if (s_dust%ready) call fill_r(s_dust, tneed, dust)
    if (s_iron%ready) then
      call fill_r(s_iron, tneed, iron)
      call scale_2d(iron, 0.01_rp)
    endif
    if (s_nox%ready) then
      call fill_r(s_nox, tneed, nox)
      call scale_2d(nox, 71394.200220751_rp)
    endif
    if (s_nhy%ready) then
      call fill_r(s_nhy, tneed, nhy)
      call scale_2d(nhy, 71394.200220751_rp)
    endif
    if (s_xco2%ready) call fill_r(s_xco2, tneed, xco2)
    if (s_xco2a%ready) call fill_r(s_xco2a, tneed, xco2a)
  end subroutine set_bgc_forces

  subroutine scale_2d(a, f)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: f
    integer :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        a(i, j) = a(i, j) * f
      enddo
    enddo
  end subroutine scale_2d

  subroutine set_2d(a, val)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: val
    integer :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        a(i, j) = val
      enddo
    enddo
  end subroutine set_2d

  subroutine divide_2d(a, d)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: d
    integer :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        a(i, j) = a(i, j) / d
      enddo
    enddo
  end subroutine divide_2d

  function find_var_file(vname, tname) result(path)
    character(len=*), intent(in) :: vname
    character(len=*), intent(in), optional :: tname
    character(len=512) :: path, fallback
    integer :: i, ncid
    path = ''
    fallback = ''
    do i = 1, case_nfrc
      ncid = frc_file_ncid(i)   ! cached handle, closed by close_frc_cache
      if (nc_has_var(ncid, vname)) then
        if (present(tname)) then
          if (nc_has_var(ncid, tname)) then
            path = case_frcfiles(i)
            return
          elseif (len_trim(fallback) == 0) then
            fallback = case_frcfiles(i)
          endif
        else
          path = case_frcfiles(i)
          return
        endif
      endif
    enddo
    if (len_trim(path) == 0) path = fallback
  end function find_var_file

  function resolve_one(dir, path) result(out)
    character(len=*), intent(in) :: dir, path
    character(len=512) :: out
    if (len_trim(path) == 0) then
      out = ''
    elseif (path(1:1) == '/') then
      out = trim(path)
    else
      out = trim(dir)//'/'//trim(path)
    endif
  end function resolve_one

  subroutine write_toy_flux(path, nxi, neta, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta
    real(rp),         intent(in) :: t0, t1
    call write_toy_flux_recs(path, nxi, neta, t0, t1)
  end subroutine write_toy_flux

  subroutine write_toy_flux_recs(path, nxi, neta, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta
    real(rp),         intent(in) :: t0, t1
    integer :: ncid, d_xi, d_eta, d_xiu, d_etav, d_t, id
    real(rp), allocatable :: au(:,:,:), av(:,:,:), ar(:,:,:)
    allocate(au(nxi - 1, neta, 2), av(nxi, neta - 1, 2), ar(nxi, neta, 2))
    au = 0.1_rp
    av = 0.0_rp
    ar = 0.0_rp
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'sms_time', 2)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id = nc_def_var(ncid, 'sms_time', (/d_t/))
    id = nc_def_var(ncid, 'shf_time', (/d_t/))
    id = nc_def_var(ncid, 'swf_time', (/d_t/))
    id = nc_def_var(ncid, 'srf_time', (/d_t/))
    id = nc_def_var(ncid, 'sustr', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'svstr', (/d_xi, d_etav, d_t/))
    id = nc_def_var(ncid, 'shflux', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'swflux', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'swrad', (/d_xi, d_eta, d_t/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'sms_time', (/ t0, t1 /))
    call nc_put_1d(ncid, 'shf_time', (/ t0, t1 /))
    call nc_put_1d(ncid, 'swf_time', (/ t0, t1 /))
    call nc_put_1d(ncid, 'srf_time', (/ t0, t1 /))
    call nc_put_3d(ncid, 'sustr', au)
    call nc_put_3d(ncid, 'svstr', av)
    call nc_put_3d(ncid, 'shflux', ar)
    call nc_put_3d(ncid, 'swflux', ar)
    call nc_put_3d(ncid, 'swrad', ar)
    call nc_close(ncid)
    deallocate(au, av, ar)
  end subroutine write_toy_flux_recs

  subroutine write_toy_bulk(path, nxi, neta, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta
    real(rp),         intent(in) :: t0, t1
    integer :: ncid, d_xi, d_eta, d_t, id
    real(rp), allocatable :: ar(:,:,:)
    allocate(ar(nxi, neta, 2))
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 2)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    id = nc_def_var(ncid, 'time', (/d_t/))
    id = nc_def_var(ncid, 'uwnd', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'vwnd', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'Tair', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'qair', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'rain', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'lwrad', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'swrad', (/d_xi, d_eta, d_t/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'time', (/ t0, t1 /))
    ar = 5.0_rp;   call nc_put_3d(ncid, 'uwnd', ar)
    ar = 0.0_rp;   call nc_put_3d(ncid, 'vwnd', ar)
    ar = 15.0_rp;  call nc_put_3d(ncid, 'Tair', ar)
    ar = 0.01_rp;  call nc_put_3d(ncid, 'qair', ar)
    ar = 0.0_rp;   call nc_put_3d(ncid, 'rain', ar)
    ar = 350.0_rp; call nc_put_3d(ncid, 'lwrad', ar)
    ar = 200.0_rp; call nc_put_3d(ncid, 'swrad', ar)
    call nc_close(ncid)
    deallocate(ar)
  end subroutine write_toy_bulk

end module frc_io

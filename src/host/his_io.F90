#include "mem_attr.inc"
module his_io
  ! Serial history / restart writer. Namelist keys match UCLA
  ! BASIC_OUTPUT_SETTINGS / TS_OUTPUT_SETTINGS. Averages are in avg_out;
  ! no monthly calendar.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use io_server
  use io_server, only: io_init8, io_stage8_begin, io_stage8_2d, io_stage8_3d, io_stage8_3d_pt, io_stage8_end
  use comm_mod, only: have_io_server
  use comm_mod, only: comm_rank, comm_size, compute_barrier
  use run_timers, only: rt_tic, rt_toc, rt_mark_rst, RT_HIS, RT_RST
  use passive_mod, only: ntp, tp, wrt_passive, passive_name
  use rivers_mod, only: river_on
  use pflx_mod, only: calc_pflx, u_slow, v_slow, p_slow
  use sponge_io, only: def_rst_ub, stage_rst_ub, put_rst_ub
  use wvlcty_mod, only: wvlcty_compute
  use marbl_gpu, only: marbl_on, marbl_gpu_nhis, marbl_gpu_nhis3, marbl_gpu_define_his, marbl_gpu_stage_his, &
                       marbl_gpu_define_rst, marbl_gpu_write_rst, marbl_gpu_stage_rst8
  use fs_data, only: fs_on, udu, udv, gru, grv, qd, xx, mg_on, fs_mg_lag, mg_se0, mg_scu, mg_scv, fs_nsc
  use implicit_fs_gpu, only: fs_rst_prepare, fs_rst_scal_vals, fs_global_atts
  implicit none
  private

  logical, public :: wrt_file_his = .false.
  logical, public :: wrt_file_rst = .false.
  real(rp) :: output_period_his = 100.0_rp
  real(rp) :: output_period_rst = 2000.0_rp
  integer  :: nrpf_his = 20
  integer  :: nrpf_rst = 2
  logical  :: monthly_restarts = .false.
  logical  :: wrt_Z = .true., wrt_Ub = .true., wrt_Vb = .true.
  logical  :: wrt_U = .true., wrt_V = .true., wrt_R = .true.
  logical  :: wrt_O = .true., wrt_W = .false.
  logical  :: wrt_Akv = .true., wrt_Akt = .true., wrt_Aks = .true.
  logical  :: wrt_Hbls = .true., wrt_Hbbl = .true.
  logical  :: wrt_temp = .true., wrt_salt = .true.

  character(len=512) :: output_root = './output'
  character(len=256) :: output_title = 'roms_gpu'
  character(len=512) :: fname_his = '', fname_rst = ''
  integer :: rec_his = 0, rec_rst = 0
  integer :: nfile_his = 0, nfile_rst = 0
  real(rp) :: output_time_his = 0.0_rp
  real(rp) :: output_time_rst = 0.0_rp
  logical  :: output_ready = .false.
  real(rp), WORK_ATTR allocatable :: wvl(:,:,:)     ! true vertical velocity (wrt_W)

  ! Calendar, as the CPU's roms_read_write: model time is in seconds since
  ! &reference_date_settings reference_date (default 2000/1/1). Output files
  ! are named <root>_<stream>.<YYYYMMDDHHMMSS>.nc, the model date at which
  ! the CPU creates the file (output_fname), and ocean_time carries
  ! long_name = refdatestr.
  integer :: reference_date(3) = (/2000, 1, 1/)
  integer(8) :: ref_offset = 946684800_8           ! seconds from 1970/1/1 to reference_date
  character(len=21), public :: refdatestr = 'Time since 2000/01/01'

  integer :: ir0, ir1, jr0, jr1, iu0, iu1, jv0, jv1
  integer :: nxi, neta, nxi_u, neta_v           ! this tile's written extents
  integer :: gxi, geta, gxi_u, geta_v           ! global file dimensions

  public :: read_output_namelist
  public :: read_refdate_namelist
  public :: output_fname
  public :: output_period_rst   ! sponge_out: the CPU's file-length check
  public :: set_output_root
  public :: init_output
  public :: wrt_his
  public :: wrt_rst
  public :: maybe_wrt_output

contains

  subroutine read_output_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=512) :: msg
    real(rp) :: output_period_his, output_period_rst, output_period_avg
    integer :: nrpf_his, nrpf_avg
    logical :: wrt_file_his, wrt_file_rst, wrt_file_avg, monthly_restarts
    logical :: wrt_Z, wrt_Ub, wrt_Vb, wrt_U, wrt_V, wrt_R, wrt_O, wrt_W
    logical :: wrt_Akv, wrt_Akt, wrt_Aks, wrt_Hbls, wrt_Hbbl
    logical :: wrt_avg_Z, wrt_avg_Ub, wrt_avg_Vb, wrt_avg_U, wrt_avg_V
    logical :: wrt_avg_R, wrt_avg_O, wrt_avg_W, wrt_avg_Akv, wrt_avg_Akt
    logical :: wrt_avg_Aks, wrt_avg_Hbls, wrt_avg_Hbbl
    logical :: wrt_temp, wrt_salt, wrt_temp_dia, wrt_salt_dia
    namelist /basic_output_settings/ output_period_rst, output_period_his, &
         nrpf_his, output_period_avg, nrpf_avg, wrt_file_rst, wrt_file_avg, &
         wrt_file_his, monthly_restarts, wrt_Z, wrt_Ub, wrt_Vb, wrt_U, wrt_V, &
         wrt_R, wrt_O, wrt_W, wrt_Akv, wrt_Akt, wrt_Aks, wrt_Hbls, wrt_Hbbl, &
         wrt_avg_Z, wrt_avg_Ub, wrt_avg_Vb, wrt_avg_U, wrt_avg_V, wrt_avg_R, &
         wrt_avg_O, wrt_avg_W, wrt_avg_Akv, wrt_avg_Akt, wrt_avg_Aks, &
         wrt_avg_Hbls, wrt_avg_Hbbl
    namelist /ts_output_settings/ wrt_temp, wrt_salt, wrt_temp_dia, wrt_salt_dia

    output_period_his = 100.0_rp
    output_period_rst = 2000.0_rp
    output_period_avg = 100.0_rp
    nrpf_his = 20
    nrpf_avg = 20
    wrt_file_his = .false.
    wrt_file_rst = .false.
    wrt_file_avg = .false.
    monthly_restarts = .false.
    wrt_Z = .true.; wrt_Ub = .true.; wrt_Vb = .true.
    wrt_U = .true.; wrt_V = .true.; wrt_R = .true.
    wrt_O = .true.; wrt_W = .false.
    wrt_Akv = .true.; wrt_Akt = .true.; wrt_Aks = .true.
    wrt_Hbls = .true.; wrt_Hbbl = .true.
    wrt_avg_Z = .true.; wrt_avg_Ub = .true.; wrt_avg_Vb = .true.
    wrt_avg_U = .true.; wrt_avg_V = .true.; wrt_avg_R = .true.
    wrt_avg_O = .true.; wrt_avg_W = .true.; wrt_avg_Akv = .true.
    wrt_avg_Akt = .true.; wrt_avg_Aks = .true.
    wrt_avg_Hbls = .true.; wrt_avg_Hbbl = .true.
    wrt_temp = .true.; wrt_salt = .true.
    wrt_temp_dia = .false.; wrt_salt_dia = .false.

    rewind(unit)
    read(unit, nml=basic_output_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'BASIC_OUTPUT_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_output_namelist'
    endif
    rewind(unit)
    read(unit, nml=ts_output_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'TS_OUTPUT_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_output_namelist'
    endif
    call apply_output_flags(output_period_his, output_period_rst, nrpf_his, &
         wrt_file_his, wrt_file_rst, monthly_restarts, wrt_Z, wrt_Ub, wrt_Vb, &
         wrt_U, wrt_V, wrt_R, wrt_O, wrt_W, wrt_Akv, wrt_Akt, wrt_Aks, &
         wrt_Hbls, wrt_Hbbl, wrt_temp, wrt_salt)
  end subroutine read_output_namelist

  subroutine apply_output_flags(p_his, p_rst, nrec, his, rst, monthly, &
       z, ub, vb, u, v, r, o, w, akv, akt, aks, hbls, hbbl, temp, salt)
    real(rp), intent(in) :: p_his, p_rst
    integer,  intent(in) :: nrec
    logical,  intent(in) :: his, rst, monthly
    logical,  intent(in) :: z, ub, vb, u, v, r, o, w
    logical,  intent(in) :: akv, akt, aks, hbls, hbbl, temp, salt
    output_period_his = p_his
    output_period_rst = p_rst
    if (nrec > 0) nrpf_his = nrec
    wrt_file_his = his
    wrt_file_rst = rst
    monthly_restarts = monthly
    wrt_Z = z; wrt_Ub = ub; wrt_Vb = vb
    wrt_U = u; wrt_V = v; wrt_R = r
    wrt_O = o; wrt_W = w
    wrt_Akv = akv; wrt_Akt = akt; wrt_Aks = aks
    wrt_Hbls = hbls; wrt_Hbbl = hbbl
    wrt_temp = temp; wrt_salt = salt
  end subroutine apply_output_flags

  subroutine set_output_root(root, title)
    character(len=*), intent(in) :: root, title
    output_root = trim(root)
    if (len_trim(output_root) == 0) output_root = './output'
    output_title = trim(title)
  end subroutine set_output_root

  subroutine init_output()
    output_ready = .false.
    rec_his = 0; rec_rst = 0
    nfile_his = 0; nfile_rst = 0
    output_time_his = 0.0_rp
    output_time_rst = 0.0_rp
    fname_his = ''; fname_rst = ''
    call set_write_bounds()
    call init_staging()
    output_ready = .true.
    if (monthly_restarts) write(*,*) 'monthly_restarts is not implemented; using output_period_rst'
    if (wrt_file_his) then
      write(*,'(a,es12.4,a)') 'history: initial write, period = ', output_period_his, ' s'
      call wrt_his(.true.)
    endif
  end subroutine init_output

  subroutine maybe_wrt_output()
    if (.not. output_ready) return
    call io_poll()
    if (wrt_file_his) then
      output_time_his = output_time_his + dt
      if (output_time_his + 1.0e-8_rp >= output_period_his) then
        call rt_tic(RT_HIS)
        call wrt_his(.false.)
        call rt_toc(RT_HIS)
        output_time_his = 0.0_rp
      endif
    endif
    if (wrt_file_rst) then
      ! Exact restart (CPU EXACT_RESTART): write the step before the restart
      ! time as well as the restart step, two consecutive records in one
      ! file (nrpf_rst = 2), so a restarted run takes the exact first step.
      output_time_rst = output_time_rst + dt
      if (output_time_rst + 1.0e-8_rp >= output_period_rst .or. &
          output_time_rst + dt + 1.0e-8_rp >= output_period_rst) then
        call rt_mark_rst()
        call rt_tic(RT_RST)
        call wrt_rst()
        call rt_toc(RT_RST)
        if (output_time_rst + 1.0e-8_rp >= output_period_rst) output_time_rst = 0.0_rp
      endif
    endif
  end subroutine maybe_wrt_output

  subroutine wrt_his(initial)
    logical, intent(in), optional :: initial
    logical :: is_init
    integer :: n2, n3, nz, is, js, itp
    real(rp) :: twrite
    character(len=512) :: open_path
    if (.not. wrt_file_his) return
    if (.not. output_ready) call init_output()
    is_init = .false.
    if (present(initial)) is_init = initial
    if (is_init) then
      twrite = start_time
      n2 = max(knew, 1)
      n3 = 1
    else
      twrite = start_time + dt * real(iic - ntstart + 1, rp)
      n2 = knew
      n3 = nnew
    endif
    ! Fields are gathered on the GPU and copied asynchronously into
    ! pinned host buffers (io_server); the file is written later by the
    ! I/O rank, or by this rank once the copies land. Nothing here reads
    ! managed memory from the host.
    open_path = ''
    if (rec_his == 0 .or. rec_his >= nrpf_his) then
      nfile_his = nfile_his + 1
      rec_his = 0
      call output_fname(output_root, 'his', twrite, fname_his)
      if (comm_rank() == 0) call create_his_file(fname_his)
      call compute_barrier()       ! the file exists before any tile announces it
      open_path = fname_his
    endif
    rec_his = rec_his + 1
    nz = decomp%nz
    is = decomp%is
    js = decomp%js
    call io_stage_begin(IO_KIND_HIS, rec_his, nfile_his, twrite, open_path)
    if (wrt_Z)  call io_stage_2d('zeta', zeta(is, js, n2), ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_Ub) call io_stage_2d('ubar', ubar(is, js, n2), iu0, iu1, jr0, jr1, 0, jr0)
    if (wrt_Vb) call io_stage_2d('vbar', vbar(is, js, n2), ir0, ir1, jv0, jv1, ir0, 0)
    if (wrt_U)  call io_stage_3d('u', u(is, js, 1, n3), 1, nz, iu0, iu1, jr0, jr1, 0, jr0)
    if (wrt_V)  call io_stage_3d('v', v(is, js, 1, n3), 1, nz, ir0, ir1, jv0, jv1, ir0, 0)
    if (wrt_temp) call io_stage_3d('temp', t(is, js, 1, n3, itemp), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_salt) call io_stage_3d('salt', t(is, js, 1, n3, isalt), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_passive) then
      do itp = 1, ntp
        call io_stage_3d_pt(trim(passive_name(itp)), tp(1, 0, 1, n3), ntp, itp, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      enddo
    endif
    if (wrt_R)    call io_stage_3d('rho', rho1(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_O)    call io_stage_omega('omega', ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_W) then                ! true vertical velocity (CPU wvlcty, slot nstp)
      if (.not. allocated(wvl)) allocate(wvl(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz))
      call wvlcty_compute(wvl)
      call io_stage_3d('w', wvl(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    endif
    if (wrt_Akv)  call io_stage_3d('Akv', Akv(is, js, 0), 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_Akt)  call io_stage_3d('Akt', Akt(is, js, 0, itemp), 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_Aks)  call io_stage_3d('Aks', Akt(is, js, 0, isalt), 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_Hbls) call io_stage_2d('hbls', hbls(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
    if (wrt_Hbbl) call io_stage_2d('hbbl', hbbl(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
    if (marbl_on) call marbl_gpu_stage_his(ir0, ir1, jr0, jr1, ir0, jr0)
    call io_stage_end()
    write(*,'(a,es14.6,a,i0,a,i0)') 'staged history t = ', twrite, &
         '  rec = ', rec_his, '  file ', nfile_his
  end subroutine wrt_his

  subroutine wrt_rst()
    integer :: ncid, n2, n3, itp, ir, nz, is, js
    real(8) :: fsc(fs_nsc)
    real(rp) :: twrite
    character(len=512) :: open_rst
    open_rst = ''
    if (.not. wrt_file_rst) return
    if (.not. output_ready) call init_output()
    twrite = start_time + dt * real(max(iic - ntstart + 1, 0), rp)
    if (iic < ntstart) twrite = start_time
    n2 = knew
    n3 = nnew
    if (n3 < 1) n3 = 1
    if (n2 < 1) n2 = 1
    if (rec_rst == 0 .or. rec_rst >= nrpf_rst) then
      nfile_rst = nfile_rst + 1
      rec_rst = 0
      ! the CPU names the file at the time of its second record (EXACT_RESTART: time + dt)
      call output_fname(output_root, 'rst', twrite + dt, fname_rst)
      if (comm_rank() == 0) call create_rst_file(fname_rst)
      call compute_barrier()
      open_rst = fname_rst
    endif
    rec_rst = rec_rst + 1
    call wall_stamp('restart write begin')
    if (have_io_server()) then
      ! staged double-precision record: the server assembles the tiles and
      ! writes each variable once; the compute ranks continue.
      nz = decomp%nz; is = decomp%is; js = decomp%js
      call io_stage8_begin(IO_KIND_RST, rec_rst, nfile_rst, twrite, open_rst, tstep=iic)
      call io_stage8_2d('zeta', zeta(is, js, n2), ir0, ir1, jr0, jr1, ir0, jr0)
      call io_stage8_2d('ubar', ubar(is, js, n2), iu0, iu1, jr0, jr1, 0, jr0)
      call io_stage8_2d('vbar', vbar(is, js, n2), ir0, ir1, jv0, jv1, ir0, 0)
      call io_stage8_3d('u', u(is, js, 1, n3), 1, nz, iu0, iu1, jr0, jr1, 0, jr0)
      call io_stage8_3d('v', v(is, js, 1, n3), 1, nz, ir0, ir1, jv0, jv1, ir0, 0)
      call io_stage8_3d('temp', t(is, js, 1, n3, itemp), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      call io_stage8_3d('salt', t(is, js, 1, n3, isalt), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      do itp = 1, ntp
        call io_stage8_3d_pt(trim(passive_name(itp)), tp(1, 0, 1, n3), ntp, itp, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      enddo
      call io_stage8_2d('hbls', hbls(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
      call io_stage8_2d('hbbl', hbbl(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
      if (marbl_on) call marbl_gpu_stage_rst8(ir0, ir1, jr0, jr1, ir0, jr0)
      call io_stage8_2d('DU_avg1', DU_avg1(is, js), iu0, iu1, jr0, jr1, 0, jr0)
      call io_stage8_2d('DV_avg1', DV_avg1(is, js), ir0, ir1, jv0, jv1, ir0, 0)
      call io_stage8_2d('DU_avg2', DU_avg2(is, js), iu0, iu1, jr0, jr1, 0, jr0)
      call io_stage8_2d('DV_avg2', DV_avg2(is, js), ir0, ir1, jv0, jv1, ir0, 0)
      call io_stage8_2d('DU_avg_bak', DU_avg_bak(is, js), iu0, iu1, jr0, jr1, 0, jr0)
      call io_stage8_2d('DV_avg_bak', DV_avg_bak(is, js), ir0, ir1, jv0, jv1, ir0, 0)
      if (fs_on) then   ! implicit free surface: generalized-alpha rates, delta (CPU a0e9e2be fs_rst_wrt)
        call fs_rst_prepare()
        call io_stage8_2d('fs_udu', udu(is, js), iu0, iu1, jr0, jr1, 0, jr0)
        call io_stage8_2d('fs_gru', gru(is, js), iu0, iu1, jr0, jr1, 0, jr0)
        call io_stage8_2d('fs_udv', udv(is, js), ir0, ir1, jv0, jv1, ir0, 0)
        call io_stage8_2d('fs_grv', grv(is, js), ir0, ir1, jv0, jv1, ir0, 0)
        call io_stage8_2d('fs_qd', qd(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
        call io_stage8_2d('fs_delta', xx(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
        call fs_rst_scal_vals(fsc)
        call io_stage8_vec('fs_rst_scal', fsc)
        if (mg_on .and. fs_mg_lag > 1) then
          call io_stage8_2d('fs_mge0', mg_se0(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
          call io_stage8_2d('fs_mgcu', mg_scu(is, js), iu0, iu1, jr0, jr1, 0, jr0)
          call io_stage8_2d('fs_mgcv', mg_scv(is, js), ir0, ir1, jv0, jv1, ir0, 0)
        endif
      endif
      call io_stage8_2d('rufrc', rufrc_bak(is, js, max(nstp, 1)), iu0, iu1, jr0, jr1, 0, jr0)
      call io_stage8_2d('rvfrc', rvfrc_bak(is, js, max(nstp, 1)), ir0, ir1, jv0, jv1, ir0, 0)
      if (river_on) then   ! river faces for the next run's initial masks (CPU wrt_rst)
        call io_stage8_2d('riv_umask', riv_umask(is, js), iu0, iu1, jr0, jr1, 0, jr0)
        call io_stage8_2d('riv_vmask', riv_vmask(is, js), ir0, ir1, jv0, jv1, ir0, 0)
      endif
      if (calc_pflx) then  ! the CPU's wrt_rst_diag_slow
        call io_stage8_3d('u_slow', u_slow(is, js, 1), 1, nz, iu0, iu1, jr0, jr1, 0, jr0)
        call io_stage8_3d('v_slow', v_slow(is, js, 1), 1, nz, ir0, ir1, jv0, jv1, ir0, 0)
        call io_stage8_3d('p_slow', p_slow(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      endif
      call stage_rst_ub()   ! ub_tune: ub_west, ub_east, ub_south, ub_north
      call io_stage8_end()
      call wall_stamp('restart staged')
      if (comm_rank() == 0) write(*,'(a,es14.6,a,i0,a,i0)') 'staged restart t = ', twrite, &
           '  rec = ', rec_rst, '  file ', nfile_rst
      return
    endif
    ! one tile at a time (a single NetCDF writer), each slab at its global offsets
    do ir = 0, int(comm_size()) - 1
    if (ir == comm_rank()) then
    call nc_open_write(fname_rst, ncid)
    call nc_put_1d_rec(ncid, 'ocean_time', twrite, rec_rst)
    call nc_put_ints_rec(ncid, 'time_step', (/iic, rec_rst, rec_rst, 0, 0, 0/), rec_rst)
    call put_r2(ncid, 'zeta', zeta(:,:,n2), rec_rst)
    call put_u2(ncid, 'ubar', ubar(:,:,n2), rec_rst)
    call put_v2(ncid, 'vbar', vbar(:,:,n2), rec_rst)
    call put_u3(ncid, 'u', u(:,:,:,n3), rec_rst)
    call put_v3(ncid, 'v', v(:,:,:,n3), rec_rst)
    call put_r3(ncid, 'temp', t(:,:,:,n3,itemp), rec_rst)
    call put_r3(ncid, 'salt', t(:,:,:,n3,isalt), rec_rst)
    do itp = 1, ntp
      call put_r3_p(ncid, trim(passive_name(itp)), tp(itp,:,:,n3), rec_rst)
    enddo
    call put_r2(ncid, 'hbls', hbls, rec_rst)
    call put_r2(ncid, 'hbbl', hbbl, rec_rst)
    if (marbl_on) call marbl_gpu_write_rst(ncid, rec_rst, ir0, ir1, jr0, jr1, int(decomp%i0) + ir0, int(decomp%j0) + jr0)
    call put_u2(ncid, 'DU_avg1', DU_avg1, rec_rst)
    call put_v2(ncid, 'DV_avg1', DV_avg1, rec_rst)
    call put_u2(ncid, 'DU_avg2', DU_avg2, rec_rst)
    call put_v2(ncid, 'DV_avg2', DV_avg2, rec_rst)
    call put_u2(ncid, 'DU_avg_bak', DU_avg_bak, rec_rst)
    call put_v2(ncid, 'DV_avg_bak', DV_avg_bak, rec_rst)
    if (fs_on) then
      call fs_rst_prepare()
      call put_u2(ncid, 'fs_udu', udu, rec_rst)
      call put_u2(ncid, 'fs_gru', gru, rec_rst)
      call put_v2(ncid, 'fs_udv', udv, rec_rst)
      call put_v2(ncid, 'fs_grv', grv, rec_rst)
      call put_r2(ncid, 'fs_qd', qd, rec_rst)
      call put_r2(ncid, 'fs_delta', xx, rec_rst)
      call fs_rst_scal_vals(fsc)
      call nc_put_vec_rec_d(ncid, 'fs_rst_scal', fsc, rec_rst)
      if (mg_on .and. fs_mg_lag > 1) then
        call put_r2(ncid, 'fs_mge0', mg_se0, rec_rst)
        call put_u2(ncid, 'fs_mgcu', mg_scu, rec_rst)
        call put_v2(ncid, 'fs_mgcv', mg_scv, rec_rst)
      endif
    endif
    call put_u2(ncid, 'rufrc', rufrc_bak(:,:,max(nstp,1)), rec_rst)
    call put_v2(ncid, 'rvfrc', rvfrc_bak(:,:,max(nstp,1)), rec_rst)
    if (river_on) then
      call put_u2(ncid, 'riv_umask', riv_umask, rec_rst)
      call put_v2(ncid, 'riv_vmask', riv_vmask, rec_rst)
    endif
    if (calc_pflx) then
      call put_u3(ncid, 'u_slow', u_slow, rec_rst)
      call put_v3(ncid, 'v_slow', v_slow, rec_rst)
      call put_r3(ncid, 'p_slow', p_slow, rec_rst)
    endif
    call put_rst_ub(ncid, rec_rst)
    call nc_close(ncid)
    endif
    call compute_barrier()
    enddo
    call wall_stamp('restart write end')
    if (comm_rank() == 0) write(*,'(a,es14.6,a,i0,a,i0)') 'wrote restart t = ', twrite, &
         '  rec = ', rec_rst, '  file ', nfile_rst
  end subroutine wrt_rst

  subroutine create_his_file(path)
    character(len=*), intent(in) :: path
    integer :: itp, ncid, d_t, d_xi, d_eta, d_xiu, d_etav, d_s, d_w, id
    call ensure_parent(path)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    d_w = nc_def_dim(ncid, 's_w', decomp%nz + 1)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    if (wrt_Z)  id = nc_def_var_f(ncid, 'zeta', (/d_xi, d_eta, d_t/))
    if (wrt_Ub) id = nc_def_var_f(ncid, 'ubar', (/d_xiu, d_eta, d_t/))
    if (wrt_Vb) id = nc_def_var_f(ncid, 'vbar', (/d_xi, d_etav, d_t/))
    if (wrt_U)  id = nc_def_var_f(ncid, 'u', (/d_xiu, d_eta, d_s, d_t/))
    if (wrt_V)  id = nc_def_var_f(ncid, 'v', (/d_xi, d_etav, d_s, d_t/))
    if (wrt_temp) id = nc_def_var_f(ncid, 'temp', (/d_xi, d_eta, d_s, d_t/))
    if (wrt_salt) id = nc_def_var_f(ncid, 'salt', (/d_xi, d_eta, d_s, d_t/))
    if (wrt_passive) then
      do itp = 1, ntp
        id = nc_def_var_f(ncid, trim(passive_name(itp)), (/d_xi, d_eta, d_s, d_t/))
      enddo
    endif
    if (wrt_R)    id = nc_def_var_f(ncid, 'rho', (/d_xi, d_eta, d_s, d_t/))
    if (wrt_O)    id = nc_def_var_f(ncid, 'omega', (/d_xi, d_eta, d_w, d_t/))
    if (wrt_W)    id = nc_def_var_f(ncid, 'w', (/d_xi, d_eta, d_s, d_t/))
    if (wrt_Akv)  id = nc_def_var_f(ncid, 'Akv', (/d_xi, d_eta, d_w, d_t/))
    if (wrt_Akt)  id = nc_def_var_f(ncid, 'Akt', (/d_xi, d_eta, d_w, d_t/))
    if (wrt_Aks)  id = nc_def_var_f(ncid, 'Aks', (/d_xi, d_eta, d_w, d_t/))
    if (wrt_Hbls) id = nc_def_var_f(ncid, 'hbls', (/d_xi, d_eta, d_t/))
    if (wrt_Hbbl) id = nc_def_var_f(ncid, 'hbbl', (/d_xi, d_eta, d_t/))
    if (marbl_on) call marbl_gpu_define_his(ncid, d_xi, d_eta, d_s, d_t)
    call nc_put_att_text(ncid, '', 'title', output_title)
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_put_att_text(ncid, '', 'history_precision', 'float32 (staged on the GPU)')
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created history file: ', trim(path)
  end subroutine create_his_file

  subroutine create_rst_file(path)
    character(len=*), intent(in) :: path
    integer :: itp, ncid, d_t, d_xi, d_eta, d_xiu, d_etav, d_s, d_aux, id, d_sc
    call ensure_parent(path)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    d_aux = nc_def_dim(ncid, 'auxil', 6)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    id = nc_def_var_i(ncid, 'time_step', (/d_aux, d_t/))   ! as the CPU model: (iic, rec, nrec, 0, 0, 0)
    id = nc_def_var(ncid, 'zeta', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'ubar', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'vbar', (/d_xi, d_etav, d_t/))
    id = nc_def_var(ncid, 'u', (/d_xiu, d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'v', (/d_xi, d_etav, d_s, d_t/))
    id = nc_def_var(ncid, 'temp', (/d_xi, d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'salt', (/d_xi, d_eta, d_s, d_t/))
    do itp = 1, ntp
      if (pp == rp) then
        id = nc_def_var(ncid, trim(passive_name(itp)), (/d_xi, d_eta, d_s, d_t/))
      else
        id = nc_def_var_f(ncid, trim(passive_name(itp)), (/d_xi, d_eta, d_s, d_t/))
      endif
    enddo
    id = nc_def_var(ncid, 'hbls', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'hbbl', (/d_xi, d_eta, d_t/))
    if (marbl_on) call marbl_gpu_define_rst(ncid, d_xi, d_eta, d_s, d_t)
    id = nc_def_var(ncid, 'DU_avg1', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'DV_avg1', (/d_xi, d_etav, d_t/))
    id = nc_def_var(ncid, 'DU_avg2', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'DV_avg2', (/d_xi, d_etav, d_t/))
    id = nc_def_var(ncid, 'DU_avg_bak', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'DV_avg_bak', (/d_xi, d_etav, d_t/))
    if (fs_on) then
      id = nc_def_var(ncid, 'fs_udu', (/d_xiu, d_eta, d_t/))
      id = nc_def_var(ncid, 'fs_udv', (/d_xi, d_etav, d_t/))
      id = nc_def_var(ncid, 'fs_gru', (/d_xiu, d_eta, d_t/))
      id = nc_def_var(ncid, 'fs_grv', (/d_xi, d_etav, d_t/))
      id = nc_def_var(ncid, 'fs_qd', (/d_xi, d_eta, d_t/))
      id = nc_def_var(ncid, 'fs_delta', (/d_xi, d_eta, d_t/))
      ! restart scalars and the fs_mg_lag > 1 fine operator (CPU 22c05bc1 fs_rst_def)
      d_sc = nc_def_dim(ncid, 'fs_nscal', fs_nsc)
      id = nc_def_var(ncid, 'fs_rst_scal', (/d_sc, d_t/))
      if (mg_on .and. fs_mg_lag > 1) then
        id = nc_def_var(ncid, 'fs_mge0', (/d_xi, d_eta, d_t/))
        id = nc_def_var(ncid, 'fs_mgcu', (/d_xiu, d_eta, d_t/))
        id = nc_def_var(ncid, 'fs_mgcv', (/d_xi, d_etav, d_t/))
      endif
    endif
    id = nc_def_var(ncid, 'rufrc', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'rvfrc', (/d_xi, d_etav, d_t/))
    if (river_on) then
      id = nc_def_var(ncid, 'riv_umask', (/d_xiu, d_eta, d_t/))
      id = nc_def_var(ncid, 'riv_vmask', (/d_xi, d_etav, d_t/))
    endif
    if (calc_pflx) then   ! the CPU's names and attributes (wrt_rst_diag_slow)
      id = nc_def_var(ncid, 'u_slow', (/d_xiu, d_eta, d_s, d_t/))
      call nc_put_att_text(ncid, 'u_slow', 'long_name', 'time filtered u')
      call nc_put_att_text(ncid, 'u_slow', 'units', 'm/s')
      id = nc_def_var(ncid, 'v_slow', (/d_xi, d_etav, d_s, d_t/))
      call nc_put_att_text(ncid, 'v_slow', 'long_name', 'time filtered v')
      call nc_put_att_text(ncid, 'v_slow', 'units', 'm/s')
      id = nc_def_var(ncid, 'p_slow', (/d_xi, d_eta, d_s, d_t/))
      call nc_put_att_text(ncid, 'p_slow', 'long_name', 'time filtered pressure')
      call nc_put_att_text(ncid, 'p_slow', 'units', 'Pa??')
    endif
    call def_rst_ub(ncid, d_xi, d_eta, d_t)
    call nc_put_att_text(ncid, '', 'title', output_title)
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created restart file: ', trim(path)
  end subroutine create_rst_file

  subroutine init_staging()
    ! Pinned staging capacity: one history record of float32 fields.
    integer :: n2d, n3d, n3w, nh, nhw, nxy, cap_rec, cap_field
    nxy = max(nxi, nxi_u) * max(neta, neta_v)
    nh = decomp%nz
    nhw = decomp%nz + 1
    n2d = count((/wrt_Z, wrt_Ub, wrt_Vb, wrt_Hbls, wrt_Hbbl/)) + marbl_gpu_nhis()
    n3d = count((/wrt_U, wrt_V, wrt_temp, wrt_salt, wrt_R, wrt_W/))
    if (wrt_passive) n3d = n3d + ntp
    n3d = n3d + marbl_gpu_nhis3()
    n3w = count((/wrt_O, wrt_Akv, wrt_Akt, wrt_Aks/))
    cap_rec = n2d * nxy + n3d * nxy * nh + n3w * nxy * nhw
    cap_field = nxy * nhw
    if (wrt_file_his .or. wrt_file_rst) call io_init(cap_rec, cap_field)
    ! double-precision capacity for one restart record of this tile
    ! (+ the implicit free surface's rates, scalars and fs_mg_lag fields, when on: 8 2-D fields and 64 values)
    if (wrt_file_rst) call io_init8((15 + 2 + 8) * nxy + (4 + ntp + 2 + merge(3, 0, calc_pflx)) * nxy * nh &
         + 4 * (max(nxi, neta) + 2) + 64, nxy * nh)
  end subroutine init_staging

  subroutine set_write_bounds()
    ! Each tile writes the points it owns: its interior, plus the boundary
    ! row/column at a physical edge. A seam face (u nx+1 / v ny+1) belongs to
    ! the east/north tile as its point 1. Halo points are never written, so
    ! arrays that are not exchanged (fast averages, MARBL state and
    ! diagnostics) stay bit-identical to the single-tile output.
    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    iu0 = 1;                          iu1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    jv0 = 1;                          jv1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    nxi = ir1 - ir0 + 1
    neta = jr1 - jr0 + 1
    nxi_u = iu1 - iu0 + 1
    neta_v = jv1 - jv0 + 1
    ! file dimensions span the global grid; each tile writes its slab at
    ! (decomp%i0, decomp%j0), the same offsets for every stagger
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    gxi_u = merge(int(decomp%gnx), int(decomp%gnx) + 1, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    geta_v = merge(int(decomp%gny), int(decomp%gny) + 1, decomp%periodic_y)
  end subroutine set_write_bounds

  subroutine put_r2(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:)
    real(rp), allocatable :: tmp(:,:)
    allocate(tmp(nxi, neta))
    tmp = src(ir0:ir1, jr0:jr1)
    call nc_put_2d_rec(ncid, name, tmp, rec, int(decomp%i0) + ir0, int(decomp%j0) + jr0)
    deallocate(tmp)
  end subroutine put_r2

  subroutine put_u2(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:)
    real(rp), allocatable :: tmp(:,:)
    allocate(tmp(nxi_u, neta))
    tmp = src(iu0:iu1, jr0:jr1)
    call nc_put_2d_rec(ncid, name, tmp, rec, int(decomp%i0), int(decomp%j0) + jr0)
    deallocate(tmp)
  end subroutine put_u2

  subroutine put_v2(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:)
    real(rp), allocatable :: tmp(:,:)
    allocate(tmp(nxi, neta_v))
    tmp = src(ir0:ir1, jv0:jv1)
    call nc_put_2d_rec(ncid, name, tmp, rec, int(decomp%i0) + ir0, int(decomp%j0))
    deallocate(tmp)
  end subroutine put_v2

  subroutine put_r3(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:, :)
    real(rp), allocatable :: tmp(:,:,:)
    allocate(tmp(nxi, neta, decomp%nz))
    tmp = src(ir0:ir1, jr0:jr1, 1:decomp%nz)
    call nc_put_3d_rec(ncid, name, tmp, rec, int(decomp%i0) + ir0, int(decomp%j0) + jr0)
    deallocate(tmp)
  end subroutine put_r3

  subroutine put_r3_p(ncid, name, src, rec)
    ! Passive tracer (real(pp) storage) into the restart: float32 for the
    ! float32 build, double when the passives are FP64 (MARBL runs).
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    ! src(0:ntpcol, nz): one tracer of the compressed passive layout,
    ! gathered to the (i, j) window through tpcol_h (unstored points: 0).
    real(pp),         intent(in) :: src(0:, :)
    real(4), allocatable :: tmp(:,:,:)
    real(rp), allocatable :: tmp8(:,:,:)
    real(pp), allocatable :: full(:,:,:)
    integer(ip) :: i, j
    allocate(full(ir0:ir1, jr0:jr1, decomp%nz))
    do j = jr0, jr1
      do i = ir0, ir1
        full(i, j, :) = src(tpcol_h(i, j), 1:decomp%nz)
      enddo
    enddo
    if (pp == rp) then
      allocate(tmp8(nxi, neta, decomp%nz))
      tmp8 = real(full(ir0:ir1, jr0:jr1, 1:decomp%nz), rp)
      call nc_put_3d_rec(ncid, name, tmp8, rec, int(decomp%i0) + ir0, int(decomp%j0) + jr0)
      deallocate(tmp8)
    else
      allocate(tmp(nxi, neta, decomp%nz))
      tmp = real(full(ir0:ir1, jr0:jr1, 1:decomp%nz), 4)
      call nc_put_3d_rec_f(ncid, name, tmp, nxi, neta, decomp%nz, rec, int(decomp%i0) + ir0, int(decomp%j0) + jr0)
      deallocate(tmp)
    endif
    deallocate(full)
  end subroutine put_r3_p

  subroutine put_u3(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:, :)
    real(rp), allocatable :: tmp(:,:,:)
    allocate(tmp(nxi_u, neta, decomp%nz))
    tmp = src(iu0:iu1, jr0:jr1, 1:decomp%nz)
    call nc_put_3d_rec(ncid, name, tmp, rec, int(decomp%i0), int(decomp%j0) + jr0)
    deallocate(tmp)
  end subroutine put_u3

  subroutine put_v3(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:, :)
    real(rp), allocatable :: tmp(:,:,:)
    allocate(tmp(nxi, neta_v, decomp%nz))
    tmp = src(ir0:ir1, jv0:jv1, 1:decomp%nz)
    call nc_put_3d_rec(ncid, name, tmp, rec, int(decomp%i0) + ir0, int(decomp%j0))
    deallocate(tmp)
  end subroutine put_v3

  subroutine put_rw(ncid, name, src, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: src(decomp%is:, decomp%js:, 0:)
    real(rp), allocatable :: tmp(:,:,:)
    integer :: nz
    nz = decomp%nz
    allocate(tmp(nxi, neta, nz + 1))
    tmp = src(ir0:ir1, jr0:jr1, 0:nz)
    call nc_put_3d_rec(ncid, name, tmp, rec, int(decomp%i0) + ir0, int(decomp%j0) + jr0)
    deallocate(tmp)
  end subroutine put_rw

  subroutine put_omega(ncid, rec)
    integer, intent(in) :: ncid, rec
    real(rp), allocatable :: tmp(:,:,:)
    integer :: i, j, k, nz
    nz = decomp%nz
    allocate(tmp(nxi, neta, nz + 1))
    do k = 0, nz
      do j = jr0, jr1
        do i = ir0, ir1
          tmp(i - ir0 + 1, j - jr0 + 1, k + 1) = &
               pm(i, j) * pn(i, j) * (We(i, j, k) + Wi(i, j, k))
        enddo
      enddo
    enddo
    call nc_put_3d_rec(ncid, 'omega', tmp, rec)
    deallocate(tmp)
  end subroutine put_omega

  subroutine read_refdate_namelist(unit)
    ! &reference_date_settings reference_date = y, m, d (the CPU's
    ! roms_read_write; absent: 2000/1/1). Names and attributes only: model
    ! time comes from the initial file's ocean_time, as on the CPU.
    integer, intent(in) :: unit
    integer :: ios
    character(len=512) :: msg
    namelist /reference_date_settings/ reference_date
    rewind(unit)
    read(unit, nml=reference_date_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'REFERENCE_DATE_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_refdate_namelist'
    endif
    ! the CPU's init_refdate
    ref_offset = 86400_8 * int(julian_day(reference_date(1), reference_date(2), reference_date(3)) - &
                               julian_day(1970, 1, 1), 8)
    write(refdatestr, '(A,I4,A,I0.2,A,I0.2)') 'Time since ', reference_date(1), '/', &
         reference_date(2), '/', reference_date(3)
  end subroutine read_refdate_namelist

  integer function julian_day(y, m, d)
    ! the CPU's init_refdate (integer arithmetic, as written there)
    integer, intent(in) :: y, m, d
    julian_day = d - 32075 + 1461*(y + 4800 + (m - 14)/12)/4 + &
                 367*(m - 2 - ((m - 14)/12)*12)/12 - &
                 3*((y + 4900 + (m - 14)/12)/100)/4
  end function julian_day

  subroutine sec2date(t, date)
    ! The CPU's sec2date: model time t (seconds since reference_date) to
    ! year, month, day, hour, minute, second. Correct for dates from 1970
    ! to 2099 (checked against Python's datetime); earlier dates come out
    ! wrong, on the CPU too. The CPU keeps the seconds since 1970 in a
    ! 4-byte integer, which overflows in 2038; here 8 bytes.
    real(rp), intent(in) :: t
    integer, intent(out) :: date(6)
    integer(8) :: utime, mjday, nsecs
    real(8) :: day
    utime = int(real(t, 8) + real(ref_offset, 8), 8)
    mjday   = int(utime/86400 + 40587, 8)
    date(1) = 1858 + int((mjday + 321.51_8) / 365.25_8)
    day     = aint(mod(mjday + 262.25_8, 365.25_8)) + 0.5_8
    date(2) = 1 + int(mod(day / 30.6_8 + 2.0_8, 12.0_8))
    date(3) = 1 + int(mod(day, 30.6_8))
    nsecs   = mod(utime, 86400_8)
    date(6) = int(mod(nsecs, 60_8))
    nsecs   = nsecs / 60
    date(5) = int(mod(nsecs, 60_8))
    date(4) = int(nsecs / 60)
  end subroutine sec2date

  subroutine output_fname(root, stream, t, path)
    ! The CPU's create_file + append_date_node for joined files (PARALLEL_IO,
    ! dt_format = 0): <root>_<stream>.<YYYYMMDDHHMMSS>.nc, the date of
    ! model time t, which is the CPU's time when it creates the file.
    character(len=*), intent(in)  :: root, stream
    real(rp),         intent(in)  :: t
    character(len=*), intent(out) :: path
    integer :: date(6)
    call sec2date(t, date)
    write(path, '(A,"_",A,".",I4,5I0.2,".nc")') trim(root), trim(stream), date
  end subroutine output_fname

  subroutine ensure_parent(path)
    character(len=*), intent(in) :: path
    integer :: n
    character(len=512) :: dir
    n = index(path, '/', back=.true.)
    if (n <= 1) return
    dir = path(1:n-1)
    if (len_trim(dir) == 0 .or. trim(dir) == '.') return
    call execute_command_line('mkdir -p "'//trim(dir)//'"', wait=.true.)
  end subroutine ensure_parent

end module his_io

#include "mem_attr.inc"
module avg_out
  ! Averaged history (CPU basic_output: wrt_file_avg, calc_avg_ocean_vars,
  ! wrt_avg_ocean_vars) and averaged BGC tracers (CPU bgc_io: wrt_bgc_avg,
  ! calc_avg_bgc), in double as the CPU's files.
  !
  ! Both are running means updated after every step, avg*(1 - coef) +
  ! x*coef with coef = 1/n, n the steps since the last record:
  !  - zeta, ubar, vbar at knew (time n+1); u, v and the tracers at slot
  !    nstp (time n, the CPU's choice); rho = rho1 - qp1*z_r; omega from
  !    We + Wi (times pm*pn at the write); the true vertical velocity w
  !    (wvlcty); Akv, Akt, AKs, hbls, hbbl;
  !  - <root>_avg.<date>.nc (his_io output_fname, the date of the first
  !    write): a record every output_period_avg, nrpf_avg per file,
  !    stamped with the running mean of the model time;
  !  - <root>_bgc_avg.<date>.nc: the MARBL tracers, a record every
  !    output_period_bgc_avg, stamped with the time at the write.
  ! The tracers averaged are those the CPU flags (wrt_t_avg): with MARBL,
  ! the tracers named in marbl_tracers_to_write, or every tracer when the
  ! list is empty; temperature, salinity and the CDR_LITE tracers go to the
  ! ocean file, the MARBL tracers to the BGC file. Without MARBL, none.
  !
  ! CPU behaviours kept: the BGC average's start-up call (when wrt_bgc_his
  ! is off) already adds the initial state and advances the output clock by
  ! one step. Not kept: the CPU starts a new _bgc_avg file for every record
  ! (its record counter is reset at every call); here nrpf_bgc_avg records
  ! go in each file.
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use implicit_fs_gpu, only: fs_global_atts
  use io_server, only: IO_KIND_AVG, IO_KIND_BGCAVG, io_init, io_init8, io_stage8_begin, io_stage8_2d, &
                       io_stage8_3d, io_stage8_3d_pt, io_stage8_end
  use comm_mod, only: comm_rank, compute_barrier
  use passive_mod, only: ntp, n_marbl, tp, passive_name, ncdr, cdr0, n_oae
  use marbl_gpu, only: marbl_on, marbl_trc_list
  use marbl_host, only: marbl_tracer_units, marbl_tracer_lnames
  use wvlcty_mod, only: wvlcty_compute
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: read_avg_namelist, set_avg_root, init_avg_output, wrt_avg, step_bgc_avg

  ! &basic_output_settings (the averaging part)
  logical  :: wrt_file_avg = .false.
  real(rp) :: output_period_avg = 100.0_rp
  integer  :: nrpf_avg = 20
  logical  :: wz = .true., wub = .true., wvb = .true., wu = .true., wv = .true., wr = .true., wo = .true., &
              ww = .true., wakv = .true., wakt = .true., waks = .true., whbl = .true., whbbl = .true.
  ! &bgc_settings (the averaging part, and wrt_bgc_his for the start-up call)
  logical  :: wrt_bgc_avg = .false., wrt_bgc_his = .false.
  real(rp) :: output_period_bgc_avg = 100.0_rp
  integer  :: nrpf_bgc_avg = 20

  character(len=512) :: root = './output', fname = '', bname = ''
  logical :: on = .false., bon = .false.
  integer :: rec = 0, nfile = 0, navg = 0, brec = 0, bnfile = 0, nbavg = 0
  real(rp) :: otime = 0.0_rp, botime = 0.0_rp, tavg = 0.0_rp
  integer :: ir0, ir1, jr0, jr1, iu0, iu1, jv0, jv1, gxi, geta, gxi_u, geta_v
  ! ocean averages
  real(rp), WORK_ATTR allocatable :: za(:,:), uba(:,:), vba(:,:), hba(:,:), hbba(:,:)
  real(rp), WORK_ATTR allocatable :: ua(:,:,:), va(:,:,:), ra(:,:,:), wva(:,:,:), wtmp(:,:,:)
  real(rp), WORK_ATTR allocatable :: oa(:,:,:), akva(:,:,:), akta(:,:,:), aksa(:,:,:), otmp(:,:,:)
  ! tracers of the ocean file: 1 temp, 2 salt, then CDR_LITE (ta(:,:,:,m))
  integer :: nta = 0
  integer, allocatable :: ta_src(:)                  ! -1 temp, -2 salt, > 0 passive index
  character(len=32), allocatable :: ta_name(:)
  character(len=64), allocatable :: ta_lname(:), ta_units(:)
  real(rp), WORK_ATTR allocatable :: ta(:,:,:,:)
  ! BGC averages in the passive layout ba(m, column, k), m over the selected
  ! MARBL tracers bsrc(m)
  integer :: nba = 0
  integer(ip), allocatable :: bsrc(:)
  integer(ip), WORK_ATTR allocatable :: bsrc_d(:)
  real(pp), WORK_ATTR allocatable :: ba(:,:,:)

contains

  subroutine read_avg_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    real(rp) :: output_period_rst, output_period_his
    integer :: nrpf_his, nbgc_flx, nrpf_bgc_his, nrpf_bgc_his_dia, nrpf_bgc_avg_dia
    logical :: wrt_file_rst, wrt_file_his, monthly_restarts, wrt_Z, wrt_Ub, wrt_Vb, wrt_U, wrt_V, &
               wrt_R, wrt_O, wrt_W, wrt_Akv, wrt_Akt, wrt_aks, wrt_Hbls, wrt_Hbbl
    logical :: wrt_avg_Z, wrt_avg_Ub, wrt_avg_Vb, wrt_avg_U, wrt_avg_V, wrt_avg_R, wrt_avg_O, wrt_avg_W, &
               wrt_avg_Akv, wrt_avg_Akt, wrt_avg_Aks, wrt_avg_Hbls, wrt_avg_Hbbl
    logical :: wrt_bgc_dia_his, wrt_bgc_dia_avg, interp_bgc_frc
    real(rp) :: output_period_bgc_his, output_period_bgc_his_dia, output_period_bgc_avg_dia, xco2air_default
    namelist /basic_output_settings/ output_period_rst, output_period_his, nrpf_his, &
         output_period_avg, nrpf_avg, wrt_file_rst, wrt_file_avg, wrt_file_his, monthly_restarts, &
         wrt_Z, wrt_Ub, wrt_Vb, wrt_U, wrt_V, wrt_R, wrt_O, wrt_W, wrt_Akv, wrt_Akt, wrt_aks, wrt_Hbls, &
         wrt_Hbbl, wrt_avg_Z, wrt_avg_Ub, wrt_avg_Vb, wrt_avg_U, wrt_avg_V, wrt_avg_R, wrt_avg_O, &
         wrt_avg_W, wrt_avg_Akv, wrt_avg_Akt, wrt_avg_Aks, wrt_avg_Hbls, wrt_avg_Hbbl
    namelist /bgc_settings/ wrt_bgc_his, wrt_bgc_avg, wrt_bgc_dia_his, wrt_bgc_dia_avg, interp_bgc_frc, &
         output_period_bgc_his, nrpf_bgc_his, output_period_bgc_avg, nrpf_bgc_avg, &
         output_period_bgc_his_dia, nrpf_bgc_his_dia, output_period_bgc_avg_dia, nrpf_bgc_avg_dia, &
         nbgc_flx, xco2air_default
    wrt_avg_Z = .true.; wrt_avg_Ub = .true.; wrt_avg_Vb = .true.; wrt_avg_U = .true.; wrt_avg_V = .true.
    wrt_avg_R = .true.; wrt_avg_O = .true.; wrt_avg_W = .true.; wrt_avg_Akv = .true.; wrt_avg_Akt = .true.
    wrt_avg_Aks = .true.; wrt_avg_Hbls = .true.; wrt_avg_Hbbl = .true.
    rewind(unit)
    read(unit, nml=basic_output_settings, iostat=ios)
    if (ios > 0) error stop 'read_avg_namelist: basic_output_settings'
    wz = wrt_avg_Z; wub = wrt_avg_Ub; wvb = wrt_avg_Vb; wu = wrt_avg_U; wv = wrt_avg_V; wr = wrt_avg_R
    wo = wrt_avg_O; ww = wrt_avg_W; wakv = wrt_avg_Akv; wakt = wrt_avg_Akt; waks = wrt_avg_Aks
    whbl = wrt_avg_Hbls; whbbl = wrt_avg_Hbbl
    rewind(unit)
    read(unit, nml=bgc_settings, iostat=ios)
    if (ios > 0) error stop 'read_avg_namelist: bgc_settings'
    if (ios < 0) then
      wrt_bgc_avg = .false.; wrt_bgc_his = .false.
    endif
  end subroutine read_avg_namelist

  subroutine set_avg_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_avg_root

  subroutine select_tracers(sel_ts, sel_cdr, sel_marbl)
    ! The CPU's wrt_t_avg: the names in marbl_tracers_to_write (up to the
    ! first blank entry) among all tracers, or all tracers when the list
    ! is empty; nothing without MARBL. sel_cdr covers the tracers after
    ! MARBL in tp (generic passives, then CDR_LITE: the CPU's order),
    ! tp indices n_marbl+1 .. ntp.
    logical, intent(out) :: sel_ts(2), sel_cdr(max(int(ntp - n_marbl), 1)), sel_marbl(max(int(n_marbl), 1))
    integer :: l, m
    sel_ts = .false.; sel_cdr = .false.; sel_marbl = .false.
    if (.not. marbl_on) return
    if (len_trim(marbl_trc_list(1)) == 0) then
      sel_ts = .true.; sel_cdr = .true.; sel_marbl = .true.
      return
    endif
    do l = 1, size(marbl_trc_list)
      if (len_trim(marbl_trc_list(l)) == 0) exit
      if (trim(marbl_trc_list(l)) == 'temp') sel_ts(1) = .true.
      if (trim(marbl_trc_list(l)) == 'salt') sel_ts(2) = .true.
      do m = 1, int(ntp - n_marbl)
        if (trim(marbl_trc_list(l)) == trim(passive_name(n_marbl + m))) sel_cdr(m) = .true.
      enddo
      do m = 1, int(n_marbl)
        if (trim(marbl_trc_list(l)) == trim(passive_name(m))) sel_marbl(m) = .true.
      enddo
    enddo
  end subroutine select_tracers

  function cdr_lname(mm) result(ln)
    ! the CPU's t_lname of tp tracer n_marbl+mm (generic passive or CDR_LITE)
    integer, intent(in) :: mm
    character(len=64) :: ln
    character(len=8) :: num
    integer :: m
    m = mm - int(cdr0 - n_marbl)          ! index within the CDR_LITE block
    if (m <= 0) then
      write(num, '(i0)') mm
      ln = 'passive tracer'//trim(num)
    elseif (m <= 2 * int(n_oae)) then
      write(num, '(i0)') (m + 1) / 2
      if (mod(m, 2) == 1) then
        ln = 'CDR OAE ALK tracer'//trim(num)
      else
        ln = 'CDR OAE DIC tracer'//trim(num)
      endif
    else
      write(num, '(i0)') m - 2 * int(n_oae)
      ln = 'CDR DOR DIC tracer'//trim(num)
    endif
  end function cdr_lname

  subroutine init_avg_output()
    ! After init_output (the history), before the outputs that write a
    ! double record at start-up.
    logical :: sel_ts(2), sel_cdr(max(int(ntp - n_marbl), 1)), sel_marbl(max(int(n_marbl), 1))
    integer :: is, ie, js, je, nz, m, nxy, n2, n3, nw
    on = wrt_file_avg
    bon = wrt_bgc_avg .and. marbl_on
    if (wrt_bgc_avg .and. .not. marbl_on) error stop 'init_avg_output: wrt_bgc_avg needs MARBL'
    if (.not. (on .or. bon)) return
    if (on .and. abs(output_period_avg / dt - anint(output_period_avg / dt)) > 1.0e-9_rp) &
      error stop 'init_avg_output: dt is not a factor of output_period_avg'
    if (bon .and. abs(output_period_bgc_avg / dt - anint(output_period_bgc_avg / dt)) > 1.0e-9_rp) &
      error stop 'init_avg_output: dt is not a factor of output_period_bgc_avg'
    if (nrpf_avg < 1 .or. nrpf_bgc_avg < 1) error stop 'init_avg_output: nrpf_avg and nrpf_bgc_avg must be >= 1'
    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    iu0 = 1; iu1 = ir1; jv0 = 1; jv1 = jr1
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    gxi_u = merge(int(decomp%gnx), int(decomp%gnx) + 1, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    geta_v = merge(int(decomp%gny), int(decomp%gny) + 1, decomp%periodic_y)
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    call select_tracers(sel_ts, sel_cdr, sel_marbl)
    n2 = 0; n3 = 0; nw = 0
    if (on) then
      if (wz) allocate(za(is:ie, js:je))
      if (wub) allocate(uba(is:ie, js:je))
      if (wvb) allocate(vba(is:ie, js:je))
      if (whbl) allocate(hba(is:ie, js:je))
      if (whbbl) allocate(hbba(is:ie, js:je))
      if (wu) allocate(ua(is:ie, js:je, nz))
      if (wv) allocate(va(is:ie, js:je, nz))
      if (wr) allocate(ra(is:ie, js:je, nz))
      if (ww) allocate(wva(is:ie, js:je, nz), wtmp(is:ie, js:je, nz))
      if (wo) allocate(oa(is:ie, js:je, 0:nz), otmp(is:ie, js:je, 0:nz))
      if (wakv) allocate(akva(is:ie, js:je, 0:nz))
      if (wakt) allocate(akta(is:ie, js:je, 0:nz))
      if (waks) allocate(aksa(is:ie, js:je, 0:nz))
      nta = count(sel_ts) + count(sel_cdr(1:max(int(ntp - n_marbl), 0)))
      allocate(ta_src(max(nta, 1)), ta_name(max(nta, 1)), ta_lname(max(nta, 1)), ta_units(max(nta, 1)))
      nta = 0
      if (sel_ts(1)) then
        nta = nta + 1; ta_src(nta) = -1; ta_name(nta) = 'temp'
        ta_lname(nta) = 'potential temperature'; ta_units(nta) = 'Celsius'
      endif
      if (sel_ts(2)) then
        nta = nta + 1; ta_src(nta) = -2; ta_name(nta) = 'salt'
        ta_lname(nta) = 'salinity'; ta_units(nta) = 'PSU'
      endif
      do m = 1, int(ntp - n_marbl)
        if (.not. sel_cdr(m)) cycle
        nta = nta + 1; ta_src(nta) = int(n_marbl) + m; ta_name(nta) = passive_name(n_marbl + m)
        ta_lname(nta) = cdr_lname(m); ta_units(nta) = 'mMol m-3'
      enddo
      allocate(ta(is:ie, js:je, nz, max(nta, 1)))
      n2 = count((/wz, wub, wvb, whbl, whbbl/))
      n3 = count((/wu, wv, wr, ww/)) + nta
      nw = count((/wo, wakv, wakt, waks/))
      rec = nrpf_avg; nfile = 0; navg = 0; otime = 0.0_rp; tavg = 0.0_rp
    endif
    if (bon) then
      nba = count(sel_marbl(1:int(n_marbl)))
      allocate(bsrc(max(nba, 1)), bsrc_d(max(nba, 1)))
      nba = 0
      do m = 1, int(n_marbl)
        if (sel_marbl(m)) then
          nba = nba + 1; bsrc(nba) = m
        endif
      enddo
      bsrc_d = bsrc
      allocate(ba(max(nba, 1), 0:ntpcol, nz))
      brec = nrpf_bgc_avg; bnfile = 0; nbavg = 0; botime = 0.0_rp
    endif
    call zero_all()
    nxy = (ir1 - ir0 + 1) * (jr1 - jr0 + 1)
    call io_init(1, 1)
    call io_init8(max((n2 + n3 * nz + nw * (nz + 1)) * nxy, nba * nz * nxy, 1), nxy * (nz + 1))
    if (comm_rank() == 0) then
      if (on) write(*,'(a,es12.4,a,i0,a,i0,a)') 'averages: period ', output_period_avg, ' s, ', nrpf_avg, &
           ' records per file, ', nta, ' tracers'
      if (bon) write(*,'(a,es12.4,a,i0,a,i0,a)') 'BGC averages: period ', output_period_bgc_avg, ' s, ', &
           nrpf_bgc_avg, ' records per file, ', nba, ' tracers'
    endif
    ! the CPU's start-up wrt_bgc: without BGC history output it already
    ! averages the initial state (slot nstp = 1) and advances the clock
    if (bon .and. .not. wrt_bgc_his) call bgc_step(start_time)
  end subroutine init_avg_output

  subroutine zero_all()
    ! the accumulators to 0 on the device (the first update multiplies the
    ! old value by 0: no NaN may be there)
    integer :: m, k, c, nz, nb
    nz = decomp%nz
    if (on) then
      if (wz) call zero2(za)
      if (wub) call zero2(uba)
      if (wvb) call zero2(vba)
      if (whbl) call zero2(hba)
      if (whbbl) call zero2(hbba)
      if (wu) call zero3(ua, 1, nz)
      if (wv) call zero3(va, 1, nz)
      if (wr) call zero3(ra, 1, nz)
      if (ww) call zero3(wva, 1, nz)
      if (ww) call zero3(wtmp, 1, nz)
      if (wo) call zero3(oa, 0, nz)
      if (wo) call zero3(otmp, 0, nz)
      if (wakv) call zero3(akva, 0, nz)
      if (wakt) call zero3(akta, 0, nz)
      if (waks) call zero3(aksa, 0, nz)
      do m = 1, size(ta, 4)
        call zero3(ta(decomp%is, decomp%js, 1, m), 1, nz)
      enddo
    endif
    if (bon) then
      nb = size(ba, 1)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do c = 0, ntpcol
          do m = 1, nb
            ba(m, c, k) = 0.0_pp
          enddo
        enddo
      enddo
    endif
  end subroutine zero_all

  subroutine zero2(a)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: i, j
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        a(i, j) = 0.0_rp
      enddo
    enddo
  end subroutine zero2

  subroutine zero3(a, k0, k1)
    integer, intent(in) :: k0, k1
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: i, j, k
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = k0, k1
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          a(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine zero3

  subroutine acc2(a, x, omc, coef)
    ! a = a*(1 - coef) + x*coef
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: omc, coef
    integer :: i, j
    real(rp) :: o, c
    o = omc; c = coef
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        a(i, j) = a(i, j) * o + x(i, j) * c
      enddo
    enddo
  end subroutine acc2

  subroutine acc3(a, x, k0, k1, omc, coef)
    integer, intent(in) :: k0, k1
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    real(rp), intent(in) :: omc, coef
    integer :: i, j, k
    real(rp) :: o, c
    o = omc; c = coef
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = k0, k1
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          a(i, j, k) = a(i, j, k) * o + x(i, j, k) * c
        enddo
      enddo
    enddo
  end subroutine acc3

  subroutine wrt_avg()
    ! After every step (the CPU's calc_avg_ocean_vars, then wrt_avg_ocean_vars).
    real(rp) :: tnow
    if (.not. on) return
    tnow = start_time + dt * real(iic - ntstart + 1, rp)
    call calc_avg(tnow)
    otime = otime + dt
    if (otime >= output_period_avg) then
      if (rec == nrpf_avg) then
        nfile = nfile + 1
        rec = 0
        call output_fname(root, 'avg', tnow, fname)
        if (comm_rank() == 0) call create_avg_file(fname)
        call compute_barrier()
        call stage_avg(fname)
      else
        call stage_avg('')
      endif
      navg = 0
      otime = 0.0_rp
    endif
  end subroutine wrt_avg

  subroutine calc_avg(tnow)
    ! The CPU's calc_avg_ocean_vars, field by field.
    real(rp), intent(in) :: tnow
    real(rp) :: coef, omc
    integer :: i, j, k, m, is, js, ie, je, nz, kn, ns, src, it
    navg = navg + 1
    coef = 1.0_rp / real(navg, rp)
    omc = 1.0_rp - coef
    tavg = tavg * omc + tnow * coef
    is = decomp%is; js = decomp%js; ie = decomp%ie; je = decomp%je; nz = decomp%nz
    kn = knew; ns = nstp
    if (wz) call acc2(za, zeta(is, js, kn), omc, coef)
    if (wub) call acc2(uba, ubar(is, js, kn), omc, coef)
    if (wvb) call acc2(vba, vbar(is, js, kn), omc, coef)
    if (wu) call acc3(ua, u(is, js, 1, ns), 1, nz, omc, coef)
    if (wv) call acc3(va, v(is, js, 1, ns), 1, nz, omc, coef)
    do m = 1, nta
      src = ta_src(m)
      if (src < 0) then
        it = merge(int(itemp), int(isalt), src == -1)
        call acc3(ta(is, js, 1, m), t(is, js, 1, ns, it), 1, nz, omc, coef)
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
        do k = 1, nz
          do j = js, je
            do i = is, ie
              ta(i, j, k, m) = ta(i, j, k, m) * omc + real(tp(src, tpcol(i, j), k, ns), rp) * coef
            enddo
          enddo
        enddo
      endif
    enddo
    if (wr) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = js, je
          do i = is, ie
            ra(i, j, k) = ra(i, j, k) * omc + (rho1(i, j, k) - qp1(i, j, k) * z_r(i, j, k)) * coef
          enddo
        enddo
      enddo
    endif
    if (wo) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 0, nz
        do j = js, je
          do i = is, ie
            oa(i, j, k) = oa(i, j, k) * omc + (We(i, j, k) + Wi(i, j, k)) * coef
          enddo
        enddo
      enddo
    endif
    if (ww) then
      call wvlcty_compute(wtmp)
      call acc3(wva, wtmp, 1, nz, omc, coef)
    endif
    if (wakv) call acc3(akva, Akv(is, js, 0), 0, nz, omc, coef)
    if (wakt) call acc3(akta, Akt(is, js, 0, itemp), 0, nz, omc, coef)
    if (waks) call acc3(aksa, Akt(is, js, 0, isalt), 0, nz, omc, coef)
    if (whbl) call acc2(hba, hbls, omc, coef)
    if (whbbl) call acc2(hbba, hbbl, omc, coef)
  end subroutine calc_avg

  subroutine stage_avg(open_path)
    character(len=*), intent(in) :: open_path
    integer :: i, j, k, m, is, js, nz, ie, je
    is = decomp%is; js = decomp%js; ie = decomp%ie; je = decomp%je; nz = decomp%nz
    rec = rec + 1
    if (wo) then
      ! the CPU converts w_avg to velocity (times pm*pn) before the write
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 0, nz
        do j = js, je
          do i = is, ie
            otmp(i, j, k) = oa(i, j, k) * pm(i, j) * pn(i, j)
          enddo
        enddo
      enddo
    endif
    call io_stage8_begin(IO_KIND_AVG, rec, nfile, tavg, open_path)
    if (wz) call io_stage8_2d('zeta', za, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wub) call io_stage8_2d('ubar', uba, iu0, iu1, jr0, jr1, 0, jr0)
    if (wvb) call io_stage8_2d('vbar', vba, ir0, ir1, jv0, jv1, ir0, 0)
    if (wu) call io_stage8_3d('u', ua, 1, nz, iu0, iu1, jr0, jr1, 0, jr0)
    if (wv) call io_stage8_3d('v', va, 1, nz, ir0, ir1, jv0, jv1, ir0, 0)
    do m = 1, nta
      call io_stage8_3d(trim(ta_name(m)), ta(is, js, 1, m), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    enddo
    if (wr) call io_stage8_3d('rho', ra, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wo) call io_stage8_3d('omega', otmp, 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (ww) call io_stage8_3d('w', wva, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wakv) call io_stage8_3d('Akv', akva, 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (wakt) call io_stage8_3d('Akt', akta, 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (waks) call io_stage8_3d('AKs', aksa, 0, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    if (whbl) call io_stage8_2d('hbls', hba, ir0, ir1, jr0, jr1, ir0, jr0)
    if (whbbl) call io_stage8_2d('hbbl', hbba, ir0, ir1, jr0, jr1, ir0, jr0)
    call io_stage8_end()
    if (comm_rank() == 0) write(*,'(a,es14.6,a,i0,a,i0)') 'staged averages t = ', tavg, '  rec = ', rec, &
         '  file ', nfile
  end subroutine stage_avg

  subroutine step_bgc_avg()
    ! After every step (the CPU's wrt_bgc: calc_avg_bgc, then the write).
    if (.not. bon) return
    call bgc_step(start_time + dt * real(iic - ntstart + 1, rp))
  end subroutine step_bgc_avg

  subroutine bgc_step(tnow)
    real(rp), intent(in) :: tnow
    real(rp) :: coef, omc
    integer :: m, c, k, nz, nb, ns
    nbavg = nbavg + 1
    coef = 1.0_rp / real(nbavg, rp)
    omc = 1.0_rp - coef
    nz = decomp%nz; nb = nba; ns = nstp
    if (nb > 0) then
      ! one thread per (tracer, column, level), tracers fastest: the tp and
      ! ba accesses of a warp are contiguous
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do c = 0, ntpcol
          do m = 1, nb
            ba(m, c, k) = real(real(ba(m, c, k), rp) * omc + real(tp(bsrc_d(m), c, k, ns), rp) * coef, pp)
          enddo
        enddo
      enddo
    endif
    botime = botime + dt
    if (botime >= output_period_bgc_avg) then
      if (brec == nrpf_bgc_avg) then
        bnfile = bnfile + 1
        brec = 0
        call output_fname(root, 'bgc_avg', tnow, bname)
        if (comm_rank() == 0) call create_bgc_avg_file(bname)
        call compute_barrier()
        call stage_bgc(bname, tnow)
      else
        call stage_bgc('', tnow)
      endif
      botime = 0.0_rp
      nbavg = 0
    endif
  end subroutine bgc_step

  subroutine stage_bgc(open_path, tnow)
    character(len=*), intent(in) :: open_path
    real(rp), intent(in) :: tnow
    integer :: m, nz
    nz = decomp%nz
    brec = brec + 1
    call io_stage8_begin(IO_KIND_BGCAVG, brec, bnfile, tnow, open_path)
    do m = 1, nba
      call io_stage8_3d_pt(trim(passive_name(bsrc(m))), ba(1, 0, 1), nba, m, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    enddo
    call io_stage8_end()
    if (comm_rank() == 0) write(*,'(a,es14.6,a,i0,a,i0)') 'staged BGC averages t = ', tnow, '  rec = ', brec, &
         '  file ', bnfile
  end subroutine stage_bgc

  subroutine open_file(path, ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, d_w)
    character(len=*), intent(in) :: path
    integer, intent(out) :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, d_w
    integer :: n, id
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    d_w = nc_def_dim(ncid, 's_w', decomp%nz + 1)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
  end subroutine open_file

  subroutine create_avg_file(path)
    ! The CPU's variables (double), long names ('averaged ...'), units and order.
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, d_w, m
    character(len=40) :: ts
    call open_file(path, ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, d_w)
    if (wz) call def('zeta', (/d_xi, d_eta, d_t/), 'free-surface elevation', 'meter')
    if (wub) call def('ubar', (/d_xiu, d_eta, d_t/), 'vertically averaged u-momentum component', 'meter second-1')
    if (wvb) call def('vbar', (/d_xi, d_etav, d_t/), 'vertically averaged v-momentum component', 'meter second-1')
    if (wu) call def('u', (/d_xiu, d_eta, d_s, d_t/), 'u-momentum component', 'meter second-1')
    if (wv) call def('v', (/d_xi, d_etav, d_s, d_t/), 'v-momentum component', 'meter second-1')
    do m = 1, nta
      call def(trim(ta_name(m)), (/d_xi, d_eta, d_s, d_t/), trim(ta_lname(m)), trim(ta_units(m)))
    enddo
    if (wr) call def('rho', (/d_xi, d_eta, d_s, d_t/), 'density anomaly', 'kilogram meter-3')
    if (wo) call def('omega', (/d_xi, d_eta, d_w, d_t/), 'S-coordinate vertical velocity', 'meter second-1')
    if (ww) call def('w', (/d_xi, d_eta, d_s, d_t/), 'vertical velocity', 'meter second-1')
    if (wakv) call def('Akv', (/d_xi, d_eta, d_w, d_t/), 'vertical viscosity coefficient', 'meter2 second-1')
    if (wakt) call def('Akt', (/d_xi, d_eta, d_w, d_t/), 'vertical thermal conductivity coefficient', &
                       'meter2 second-1')
    if (waks) call def('AKs', (/d_xi, d_eta, d_w, d_t/), 'salinity vertical diffusion coefficient', &
                       'meter2 second-1')
    if (whbl) call def('hbls', (/d_xi, d_eta, d_t/), 'Thickness of KPP surface boundary layer', 'meter')
    if (whbbl) call def('hbbl', (/d_xi, d_eta, d_t/), 'Thickness of KPP bottom boundary layer', 'meter')
    write(ts, '(f12.1)') output_period_avg
    call nc_put_att_text(ncid, '', 'type', 'ROMS averages file')
    call nc_put_att_text(ncid, '', 'averaging_timescale', trim(adjustl(ts))//' seconds')
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu averages')
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created averages file: ', trim(path)
  contains
    subroutine def(name, dims, ln, un)
      character(len=*), intent(in) :: name, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      vid = nc_def_var(ncid, name, dims)
      call nc_put_att_text(ncid, name, 'long_name', 'averaged '//ln)
      call nc_put_att_text(ncid, name, 'units', un)
    end subroutine def
  end subroutine create_avg_file

  subroutine create_bgc_avg_file(path)
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, d_w, m, vid
    call open_file(path, ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, d_w)
    do m = 1, nba
      vid = nc_def_var(ncid, trim(passive_name(bsrc(m))), (/d_xi, d_eta, d_s, d_t/))
      call nc_put_att_text(ncid, trim(passive_name(bsrc(m))), 'long_name', 'avg_'//trim(marbl_tracer_lnames(bsrc(m))))
      call nc_put_att_text(ncid, trim(passive_name(bsrc(m))), 'units', trim(marbl_tracer_units(bsrc(m))))
    enddo
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu BGC averages')
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created BGC averages file: ', trim(path)
  end subroutine create_bgc_avg_file

end module avg_out

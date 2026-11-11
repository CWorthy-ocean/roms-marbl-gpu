#include "mem_attr.inc"
module passive_mod
  ! Passive (anomaly) tracers stored in real(pp) (float32 by default, see
  ! kinds.F90) with all arithmetic in real(rp). They are stepped exactly as
  ! temperature/salinity (LF-AM3 predictor with C4 advection, UP3/U3
  ! corrector, implicit vertical diffusion with the salinity diffusivity,
  ! harmonic lateral diffusion, Orlanski-type open boundaries), with no
  ! surface forcing and a zero external boundary value, as befits CDR
  ! anomaly tracers. The implicit solve factorizes each column once and
  ! back-substitutes every passive right-hand side.
  !
  ! Validation hook: passive_init = 'mirror_salt' initializes tracer 1 to
  ! salinity and drives it with salinity's surface flux, KPP nonlocal
  ! term and boundary data, so its difference from salinity after a run
  ! is the storage-precision error alone.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use passive_column_mod, only: passive_column_run, passive_horiz_run, passive_t3dmix_run
  use halo_mod, only: halo_fill_3d_tp, halo_tp_start, halo_tp_mid, halo_tp_finish, tp_overlap_on
  use pflx_mod, only: ub_tune, ub_w, ub_e, ub_s, ub_n
  use rivers_mod, only: river_on, rivers_tp, RIV_PRE, RIV_COR
  use cdr_src_mod, only: cdr_on, cdr_apply_tp
  use cdr_lite_mod, only: cdrl_on, cdrl_psf, psf_cflx
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
#include "cuf_launch.inc"
  private

  integer(ip), public :: ntp = 0
  ! (ntp, 0:ntpcol, nz, 3): stored columns only, see gpu_state (tpcol).
  real(pp), WORK_ATTR public, allocatable :: tp(:, :, :, :)
  character(len=16), public :: passive_init = 'zero'           ! zero | blob | mirror_salt
  logical, public :: passive_mirror = .false.
  logical, public :: wrt_passive = .true.
  ! MARBL mode: the passives start with the MARBL tracers (names from
  ! MARBL, read from the initial file by name, external boundary values from
  ! the boundary data below instead of 0), followed by the generic tracers.
  ! Set by marbl_gpu before alloc_passive.
  logical, public :: passive_marbl = .false.
  character(len=32), allocatable, public :: passive_names(:)
  ! CDR_LITE tracers (param_settings nt_cdr_oae ALK/DIC pairs, nt_cdr_dor
  ! DIC): linear ALK/DIC anomalies that MARBL does not see. They follow the
  ! MARBL and generic tracers in tp, at tp indices cdr0+1 .. cdr0+ncdr, in
  ! the CPU's order within the block (CDR_OAE_ALK1, CDR_OAE_DIC1, ...,
  ! CDR_DOR_DIC1, ...).
  ! tp layout: MARBL (1 .. n_marbl), generic passives (n_marbl+1 ..
  ! ntp_base, named passive_tracer<n> as the CPU), CDR_LITE (cdr0+1 .. ntp).
  ! The CPU orders all tracers T, S, generic passives, CDR_LITE, MARBL; tp
  ! keeps MARBL first so the MARBL kernels address their tracers as
  ! 1 .. n_marbl. Inputs indexed by the CPU's order (the river and CDR
  ! release tracer axes) map through cpu_tracer_index.
  integer(ip), public :: n_oae = 0, n_dor = 0, ncdr = 0, cdr0 = 0
  integer(ip), public :: ntp_base = 0      ! MARBL + generic tracers, before the CDR_LITE block
  integer(ip), public :: n_marbl = 0       ! MARBL tracers (tp 1 .. n_marbl; 0 without MARBL)
  integer(ip), public :: n_generic = 0     ! generic passives (tp n_marbl+1 .. ntp_base)
  ! external boundary values from the boundary data (MARBL and CDR_LITE
  ! tracers; missing ones are 0), else 0 (generic tracers)
  logical, public :: passive_bry = .false.
  real(rp), WORK_ATTR public, allocatable :: tb_west(:,:,:), tb_east(:,:,:)    ! (js:je, nz, ntp)
  real(rp), WORK_ATTR public, allocatable :: tb_south(:,:,:), tb_north(:,:,:)  ! (is:ie, nz, ntp)
  ! One passive tracer held at a constant value (marbl_gpu: use_constant_Fe):
  ! hold_itr is its index (0: none), hold_value its value at wet points.
  integer(ip), public :: hold_itr = 0
  ! Background horizontal diffusivity (TS_DIF2 tnu2) of each tracer, set by
  ! case_io from the namelist (0 by default); passive_t3dmix adds the sponge.
  real(rp), WORK_ATTR public, allocatable :: tnu2_tp(:)
  real(rp), public :: hold_value = 0.0_rp

  public :: read_passive_namelist
  public :: passive_exchange_mid, passive_exchange_finish
  public :: alloc_passive
  public :: init_passive
  public :: passive_predictor
  public :: passive_corrector
  public :: passive_t3dmix
  public :: passive_t3dbc
  public :: passive_name
  public :: set_passive_marbl
  public :: set_cdr_lite, passive_finish_list, cpu_tracer_index
  public :: passive_hold

contains

  subroutine read_passive_namelist(unit, n_from_param)
    ! Group passive_tracers is optional; nt_passive from param_settings
    ! is the count unless overridden here.
    integer, intent(in) :: unit
    integer(ip), intent(in) :: n_from_param
    integer :: ios
    integer(ip) :: n_passive
    character(len=16) :: init_mode
    logical :: write_passive
    namelist /passive_tracers/ n_passive, init_mode, write_passive
    n_passive = n_from_param
    init_mode = 'zero'
    write_passive = .true.
    rewind(unit)
    read(unit, nml=passive_tracers, iostat=ios)
    if (ios > 0) error stop 'passive_tracers namelist: read error'
    ntp = max(n_passive, 0_ip)
    passive_init = init_mode
    wrt_passive = write_passive
    passive_mirror = (trim(passive_init) == 'mirror_salt')
    if (passive_mirror .and. ntp < 1) ntp = 1
    if (ntp > 0) write(*,'(a,i0,a,a,a,i0,a)') 'passive tracers: ', ntp, '  init = ', &
         trim(passive_init), '  (', int(storage_size(1.0_pp)), '-bit storage)'
  end subroutine read_passive_namelist

  function passive_name(itp) result(name)
    integer(ip), intent(in) :: itp
    character(len=32) :: name
    character(len=8) :: num
    if (allocated(passive_names)) then
      name = trim(passive_names(itp))
    else
      write(num, '(i0)') itp
      name = 'passive_tracer'//trim(num)
    endif
  end function passive_name

  subroutine set_cdr_lite(noae, ndor)
    ! CDR_LITE tracer counts from param_settings, before the tracer list is
    ! built (set_passive_marbl or passive_finish_list).
    integer(ip), intent(in) :: noae, ndor
    if (noae < 0 .or. ndor < 0) error stop 'param_settings: nt_cdr_oae and nt_cdr_dor must be >= 0'
    n_oae = noae; n_dor = ndor
    ncdr = 2 * n_oae + n_dor
  end subroutine set_cdr_lite

  subroutine set_passive_marbl(names)
    ! MARBL tracers first, then the nt_passive generic tracers (ntp from
    ! read_passive_namelist; zero initial and external boundary values unless
    ! the files carry them), then the CDR_LITE block.
    character(len=*), intent(in) :: names(:)
    character(len=32), allocatable :: base(:)
    integer(ip) :: m
    character(len=8) :: num
    if (passive_mirror) error stop 'passive_init = mirror_salt cannot be combined with MARBL'
    n_marbl = size(names)
    n_generic = ntp
    allocate(base(n_marbl + n_generic))
    do m = 1, n_marbl
      base(m) = trim(names(m))
    enddo
    do m = 1, n_generic
      write(num, '(i0)') m
      base(n_marbl + m) = 'passive_tracer'//trim(num)
    enddo
    passive_marbl = .true.
    passive_init = 'zero'
    call append_cdr_lite(base)
    write(*,'(a,i0,a,i0,a,i0,a,i0,a)') 'passive tracers: MARBL mode, ', n_marbl, ' MARBL + ', n_generic, &
         ' generic + ', ncdr, ' CDR_LITE tracers (', int(storage_size(1.0_pp)), '-bit storage)'
  end subroutine set_passive_marbl

  subroutine passive_finish_list()
    ! Without MARBL: the generic passives (named passive_tracer<n>, as the
    ! CPU), then the CDR_LITE tracers. A no-op in MARBL mode
    ! (set_passive_marbl did it).
    character(len=32), allocatable :: base(:)
    integer(ip) :: m
    character(len=8) :: num
    if (passive_marbl) return
    n_marbl = 0
    n_generic = ntp
    if (passive_mirror .and. ncdr > 0) error stop 'passive_init = mirror_salt cannot be combined with CDR_LITE tracers'
    allocate(base(n_generic))
    do m = 1, n_generic
      write(num, '(i0)') m
      base(m) = 'passive_tracer'//trim(num)
    enddo
    call append_cdr_lite(base)
    if (ncdr > 0) write(*,'(a,i0,a,i0,a,i0,a)') 'passive tracers: ', n_generic, ' generic + ', ncdr, &
         ' CDR_LITE tracers (', int(storage_size(1.0_pp)), '-bit storage)'
  end subroutine passive_finish_list

  subroutine append_cdr_lite(base)
    ! tp = base tracers (MARBL and/or generic), then the CDR_LITE block.
    character(len=32), intent(in) :: base(:)
    integer(ip) :: m, n
    character(len=8) :: num
    ntp_base = size(base)
    cdr0 = ntp_base
    ntp = ntp_base + ncdr
    if (allocated(passive_names)) deallocate(passive_names)
    allocate(passive_names(ntp))
    do m = 1, ntp_base
      passive_names(m) = base(m)
    enddo
    do n = 1, n_oae
      write(num, '(i0)') n
      passive_names(cdr0 + 2 * n - 1) = 'CDR_OAE_ALK' // trim(num)
      passive_names(cdr0 + 2 * n) = 'CDR_OAE_DIC' // trim(num)
    enddo
    do n = 1, n_dor
      write(num, '(i0)') n
      passive_names(cdr0 + 2 * n_oae + n) = 'CDR_DOR_DIC' // trim(num)
    enddo
    passive_bry = passive_marbl .or. ncdr > 0
  end subroutine append_cdr_lite

  integer(ip) function cpu_tracer_index(itp, with_cdr) result(k)
    ! The CPU's tracer index (T = 1, S = 2, generic passives, CDR_LITE, MARBL)
    ! of tp tracer itp. with_cdr = .false.: the index in an axis that omits
    ! the CDR_LITE block (as a river file may); 0 for a CDR_LITE tracer then.
    integer(ip), intent(in) :: itp
    logical, intent(in) :: with_cdr
    integer(ip) :: nc
    nc = merge(ncdr, 0_ip, with_cdr)
    if (itp > cdr0) then
      k = merge(2 + n_generic + (itp - cdr0), 0_ip, with_cdr)     ! CDR_LITE
    elseif (itp > n_marbl) then
      k = 2 + (itp - n_marbl)                                     ! generic
    else
      k = 2 + n_generic + nc + itp                                ! MARBL
    endif
  end function cpu_tracer_index

  subroutine alloc_passive()
    integer(ip) :: is, ie, js, je, nz
    if (ntp <= 0) return
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    if (allocated(tp)) deallocate(tp)
    if (allocated(tnu2_tp)) deallocate(tnu2_tp)
    if (.not. allocated(tpcol_h)) error stop 'alloc_passive: build_tp_columns first'
    allocate(tp(ntp, 0:ntpcol, nz, 3), tnu2_tp(ntp))
    tp = 0.0_pp
    tnu2_tp = 0.0_rp
    if (passive_bry) then
      if (allocated(tb_west)) deallocate(tb_west, tb_east, tb_south, tb_north)
      allocate(tb_west(js:je, nz, ntp), tb_east(js:je, nz, ntp))
      allocate(tb_south(is:ie, nz, ntp), tb_north(is:ie, nz, ntp))
      tb_west = 0.0_rp; tb_east = 0.0_rp; tb_south = 0.0_rp; tb_north = 0.0_rp
    endif
  end subroutine alloc_passive

  subroutine init_passive()
    ! Fill all three time slots (the first LF-AM3 step then behaves as a
    ! forward step for the passives).
    integer(ip) :: i, j, k, n, itp, nx, ny, nz
    real(rp) :: ic, jc, rad, val
    if (ntp <= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    select case (trim(passive_init))
    case ('blob')
      ic = 0.5_rp * real(nx, rp); jc = 0.5_rp * real(ny, rp)
      rad = 0.1_rp * real(min(nx, ny), rp)
      do itp = 1, ntp
        do k = 1, nz
          do j = decomp%js, decomp%je
            do i = decomp%is, decomp%ie
              val = exp(-((real(i, rp) - ic)**2 + (real(j, rp) - jc)**2) / rad**2) * &
                    exp(-real(nz - k, rp) / 20.0_rp) * rmask(i, j)
              if (tpcol_h(i, j) == 0) cycle
              do n = 1, 3
                tp(itp, tpcol_h(i, j), k, n) = real(val, pp)
              enddo
            enddo
          enddo
        enddo
      enddo
    case ('mirror_salt')
      do k = 1, nz
        do j = decomp%js, decomp%je
          do i = decomp%is, decomp%ie
            if (tpcol_h(i, j) == 0) cycle
            do n = 1, 3
              tp(1, tpcol_h(i, j), k, n) = real(t(i, j, k, n, isalt), pp)
            enddo
          enddo
        enddo
      enddo
    case default
      tp = 0.0_pp
    end select
    write(*,'(a,a)') 'passive tracers initialized: ', trim(passive_init)
  end subroutine init_passive

  ! --------------------------------------------------------- predictor
  subroutine passive_predictor(dtau, cf_stp, cf_bak, n_stp, n_new, indx)
    ! Called from pre_step3d after the T/S predictor: LF-AM3 interpolant
    ! with C4 horizontal fluxes from slot nrhs (= nstp), the Hz-weighted
    ! old field saved into slot indx (one thread per tracer and cell), then
    ! the vertical advection + implicit solve (one thread per tracer and
    ! column); kernels in passive_column_mod. The open boundaries of the
    ! predictor slot are set afterwards by passive_t3dbc.
    real(rp), intent(in) :: dtau, cf_stp, cf_bak
    integer(ip), intent(in) :: n_stp, n_new, indx
    if (ntp <= 0) return
    ! Slot nrhs (= nstp) is the previous step's nnew: its exchange was
    ! started after that step's t3dmix (or done at load) and is completed
    ! here, just before the first stencil that reads it. Slot n_new is
    ! exchanged by pre_step3d after passive_t3dbc, and nothing reads its
    ! halo before that (t3dbc reads slot nstp).
    call halo_tp_finish(tp, ntp)
    call passive_horiz_run(tp, ntp, 1_ip, nrhs, n_stp, n_new, indx, cf_stp, cf_bak, dtau)
    if (river_on) call rivers_tp(tp, ntp, RIV_PRE, n_new, dtau)   ! river faces, before the column pass
    ! no surface fluxes in the predictor (the CPU's pre_step3d has none)
    call passive_column_run(tp, ntp, nrhs, n_new, dtau, 1_ip, .true., passive_mirror, .false., cdrl_psf)
  end subroutine passive_predictor

  ! --------------------------------------------------------- corrector
  subroutine passive_corrector(dt_in)
    ! Called from step3d_t after the T/S corrector: UP3 horizontal from
    ! slot nrhs (= 3), U3 vertical + (mirror: salinity forcing; CDR_LITE:
    ! surface fluxes) + implicit solve, open boundaries.
    real(rp), intent(in) :: dt_in
    if (ntp <= 0) return
    ! slot nrhs (= 3): exchange started by pre_step3d after the predictor's
    ! boundary update, completed here
    call halo_tp_finish(tp, ntp)
    call passive_horiz_run(tp, ntp, 2_ip, nrhs, nnew, nnew, nnew, 0.0_rp, 0.0_rp, dt_in)
    if (river_on) call rivers_tp(tp, ntp, RIV_COR, nnew, dt_in)
    ! CDR releases: the CPU adds them in the corrector after the horizontal
    ! advection (and the river flux it contains), before the vertical pass
    if (cdr_on) call cdr_apply_tp(tp, ntp, nnew, dt_in)
    ! CDR_LITE DIC tracers and the CFLX_CORR restoring of DIC/ALK: surface
    ! flux (cdr_lite_io and frc_io, set at n+1/2) on the top layer after
    ! the vertical advection
    call passive_column_run(tp, ntp, nrhs, nnew, dt_in, 0_ip, .false., passive_mirror, cdrl_on .or. psf_cflx, &
                            cdrl_psf)
    call passive_t3dbc()
    ! No exchange of slot nnew here: MARBL and t3dmix read it at the point
    ! only (t3dmix's stencil is slot nrhs) and passive_t3dmix exchanges it.
  end subroutine passive_corrector

  ! ------------------------------------------------ lateral diffusion
  subroutine passive_t3dmix()
    ! Harmonic diffusion along s-surfaces with each tracer's diffusivity,
    ! tnu2_tp(itp) plus the sponge ramp (t3dmix for tp; one thread per
    ! tracer and cell).
    if (ntp <= 0) return
    call passive_t3dmix_run(tp, ntp, nrhs, nnew, dt, tnu2_tp)
    ! A held tracer (use_constant_Fe) is reset after its last update of the
    ! step, in every slot, before the exchange below sends it.
    call passive_hold(0_ip)
    ! nnew is the next step's nstp: post the exchange now, the next step
    ! advances it (passive_exchange_mid) and the predictor completes it.
    if (tp_overlap_on(2)) then
      call halo_tp_start(tp, ntp, nnew)
    else
      call halo_fill_3d_tp(tp, ntp, nnew)
    endif
  end subroutine passive_t3dmix

  subroutine passive_hold(slot)
    ! Set tracer hold_itr to hold_value at every wet point (land 0, as the
    ! CPU model's masked update leaves it) of every stored column: interior,
    ! halo and physical-boundary columns, so no exchange is needed. Time
    ! slot `slot`, or all three when slot is 0. No-op when hold_itr is 0.
    integer(ip), intent(in) :: slot
    integer(ip) :: i, j, k, c, s, s0, s1, itr, nz, is, ie, js, je
    real(rp) :: v
    if (hold_itr <= 0) return
    itr = hold_itr; v = hold_value; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    s0 = slot; s1 = slot
    if (slot == 0) then
      s0 = 1; s1 = 3
    endif
    do s = s0, s1
      CUF_DO2
      do j = js, je
        do i = is, ie
          c = tpcol(i, j)
          if (c > 0) then
            do k = 1, nz
              tp(itr, c, k, s) = real(v * rmask(i, j), pp)
            enddo
          endif
        enddo
      enddo
    enddo
  end subroutine passive_hold

  subroutine passive_exchange_mid()
    ! Advance a pending all-tracer exchange (east-west done, north-south
    ! posted); placed where unrelated kernels hide the transfer.
    if (ntp <= 0) return
    call halo_tp_mid(tp, ntp)
  end subroutine passive_exchange_mid

  subroutine passive_exchange_finish()
    ! Complete a pending all-tracer exchange (end of run, before finalize).
    if (ntp <= 0) return
    call halo_tp_finish(tp, ntp)
  end subroutine passive_exchange_finish

  ! ------------------------------------------------------ open boundaries
  subroutine passive_t3dbc()
    ! t3dbc for the passives: upstream radiation with tangential upstream
    ! advection, nudged on inflow toward the external value (0, or the
    ! salinity boundary data for the mirror tracer), then corners.
    integer(ip) :: i, j, k, nx, ny, nz, ns, nn, nr, itr, ntr
    real(rp) :: cx, cy, cext, dtw, cff, text
    real(rp) :: g0c, g0p, g1c, g1p, tn
    logical :: sw, se, nw, ne, mir, mbry, ubt
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    ns = nstp; nn = nnew; nr = nrhs
    mir = passive_mirror
    mbry = passive_bry
    ubt = ub_tune
    if (nnew == 3) then
      dtw = 0.5_rp * dt
    else
      dtw = dt
    endif

    if (decomp%edge_w) then
      if (obc_west) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(0, j), k, ns), rp) - real(tp(itr, tpcol(0, j - 1), k, ns), rp)) * vmask(0, j)
              g0p = (real(tp(itr, tpcol(0, j + 1), k, ns), rp) - real(tp(itr, tpcol(0, j), k, ns), rp)) * vmask(0, j + 1)
              g1c = (real(tp(itr, tpcol(1, j), k, ns), rp) - real(tp(itr, tpcol(1, j - 1), k, ns), rp)) * vmask(1, j)
              g1p = (real(tp(itr, tpcol(1, j + 1), k, ns), rp) - real(tp(itr, tpcol(1, j), k, ns), rp)) * vmask(1, j + 1)
              cx = -dtw * u(1, j, k, nr) * pm(0, j)
              cy = 0.5_rp * dtw * (v(0, j, k, nr) + v(0, j + 1, k, nr)) * pn(0, j)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(itr, tpcol(0, j), k, ns), rp) &
                   - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                   + cx * (real(tp(itr, tpcol(1, j), k, ns), rp) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_west(j, k, isalt)
              if (mbry) text = tb_west(j, k, itr)
              if (ubt) cext = max(cext, min(ub_w(j), 1.0_rp))   ! SPONGE_TUNE (CPU t3dbc ub_tune)
              tp(itr, tpcol(0, j), k, nn) = real(((1.0_rp - cext) * tn + cext * text) * rmask(0, j), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        ! Closed (CPU t3dbc_tile, j = jstr..jend): gradient, masked.
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(0, j), k, nn) = real(real(tp(itr, tpcol(1, j), k, nn), rp) * rmask(0, j), pp)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_e) then
      if (obc_east) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(nx, j), k, ns), rp) - real(tp(itr, tpcol(nx, j - 1), k, ns), rp)) * vmask(nx, j)
              g0p = (real(tp(itr, tpcol(nx, j + 1), k, ns), rp) - real(tp(itr, tpcol(nx, j), k, ns), rp)) * vmask(nx, j + 1)
              g1c = (real(tp(itr, tpcol(nx + 1, j), k, ns), rp) - real(tp(itr, tpcol(nx + 1, j - 1), k, ns), rp)) * vmask(nx + 1, j)
              g1p = (real(tp(itr, tpcol(nx + 1, j + 1), k, ns), rp) - real(tp(itr, tpcol(nx + 1, j), k, ns), rp)) * vmask(nx + 1, j + 1)
              cx = dtw * u(nx + 1, j, k, nr) * pm(nx + 1, j)
              cy = 0.5_rp * dtw * (v(nx + 1, j, k, nr) + v(nx + 1, j + 1, k, nr)) * pn(nx + 1, j)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(itr, tpcol(nx + 1, j), k, ns), rp) &
                   - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                   + cx * (real(tp(itr, tpcol(nx, j), k, ns), rp) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_east(j, k, isalt)
              if (mbry) text = tb_east(j, k, itr)
              if (ubt) cext = max(cext, min(ub_e(j), 1.0_rp))   ! SPONGE_TUNE (CPU t3dbc ub_tune)
              tp(itr, tpcol(nx + 1, j), k, nn) = real(((1.0_rp - cext) * tn + cext * text) * rmask(nx + 1, j), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(nx + 1, j), k, nn) = real(real(tp(itr, tpcol(nx, j), k, nn), rp) * rmask(nx + 1, j), pp)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_s) then
      if (obc_south) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(i, 0), k, ns), rp) - real(tp(itr, tpcol(i - 1, 0), k, ns), rp)) * umask(i, 0)
              g0p = (real(tp(itr, tpcol(i + 1, 0), k, ns), rp) - real(tp(itr, tpcol(i, 0), k, ns), rp)) * umask(i + 1, 0)
              g1c = (real(tp(itr, tpcol(i, 1), k, ns), rp) - real(tp(itr, tpcol(i - 1, 1), k, ns), rp)) * umask(i, 1)
              g1p = (real(tp(itr, tpcol(i + 1, 1), k, ns), rp) - real(tp(itr, tpcol(i, 1), k, ns), rp)) * umask(i + 1, 1)
              cx = -dtw * v(i, 1, k, nr) * pn(i, 0)
              cy = 0.5_rp * dtw * (u(i, 0, k, nr) + u(i + 1, 0, k, nr)) * pm(i, 0)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(itr, tpcol(i, 0), k, ns), rp) &
                   - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                   + cx * (real(tp(itr, tpcol(i, 1), k, ns), rp) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_south(i, k, isalt)
              if (mbry) text = tb_south(i, k, itr)
              if (ubt) cext = max(cext, min(ub_s(i), 1.0_rp))   ! SPONGE_TUNE (CPU t3dbc ub_tune)
              tp(itr, tpcol(i, 0), k, nn) = real(((1.0_rp - cext) * tn + cext * text) * rmask(i, 0), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        ! Closed (CPU, i = istr..iend); the corners are set below.
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(i, 0), k, nn) = real(real(tp(itr, tpcol(i, 1), k, nn), rp) * rmask(i, 0), pp)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_n) then
      if (obc_north) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(i, ny), k, ns), rp) - real(tp(itr, tpcol(i - 1, ny), k, ns), rp)) * umask(i, ny)
              g0p = (real(tp(itr, tpcol(i + 1, ny), k, ns), rp) - real(tp(itr, tpcol(i, ny), k, ns), rp)) * umask(i + 1, ny)
              g1c = (real(tp(itr, tpcol(i, ny + 1), k, ns), rp) - real(tp(itr, tpcol(i - 1, ny + 1), k, ns), rp)) * umask(i, ny + 1)
              g1p = (real(tp(itr, tpcol(i + 1, ny + 1), k, ns), rp) - real(tp(itr, tpcol(i, ny + 1), k, ns), rp)) * umask(i + 1, ny + 1)
              cx = dtw * v(i, ny + 1, k, nr) * pn(i, ny + 1)
              cy = 0.5_rp * dtw * (u(i, ny + 1, k, nr) + u(i + 1, ny + 1, k, nr)) * pm(i, ny + 1)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(itr, tpcol(i, ny + 1), k, ns), rp) &
                   - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                   + cx * (real(tp(itr, tpcol(i, ny), k, ns), rp) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_north(i, k, isalt)
              if (mbry) text = tb_north(i, k, itr)
              if (ubt) cext = max(cext, min(ub_n(i), 1.0_rp))   ! SPONGE_TUNE (CPU t3dbc ub_tune)
              tp(itr, tpcol(i, ny + 1), k, nn) = real(((1.0_rp - cext) * tn + cext * text) * rmask(i, ny + 1), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(i, ny + 1), k, nn) = real(real(tp(itr, tpcol(i, ny), k, nn), rp) * rmask(i, ny + 1), pp)
            enddo
          enddo
        enddo
      endif
    endif

    sw = decomp%edge_s .and. decomp%edge_w; se = decomp%edge_s .and. decomp%edge_e
    nw = decomp%edge_n .and. decomp%edge_w; ne = decomp%edge_n .and. decomp%edge_e
    if (.not. (sw .or. se .or. nw .or. ne)) return
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do k = 1, nz
      do itr = 1, ntr
        if (sw) then
          cff = rmask(1, 0) + rmask(0, 1)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(itr, tpcol(0, 0), k, nn) = real(cff * (rmask(1, 0) * real(tp(itr, tpcol(1, 0), k, nn), rp) + &
                 rmask(0, 1) * real(tp(itr, tpcol(0, 1), k, nn), rp)), pp)
          else
            tp(itr, tpcol(0, 0), k, nn) = 0.0_pp
          endif
        endif
        if (se) then
          cff = rmask(nx, 0) + rmask(nx + 1, 1)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(itr, tpcol(nx + 1, 0), k, nn) = real(cff * (rmask(nx, 0) * real(tp(itr, tpcol(nx, 0), k, nn), rp) + &
                 rmask(nx + 1, 1) * real(tp(itr, tpcol(nx + 1, 1), k, nn), rp)), pp)
          else
            tp(itr, tpcol(nx + 1, 0), k, nn) = 0.0_pp
          endif
        endif
        if (nw) then
          cff = rmask(1, ny + 1) + rmask(0, ny)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(itr, tpcol(0, ny + 1), k, nn) = real(cff * (rmask(1, ny + 1) * real(tp(itr, tpcol(1, ny + 1), k, nn), rp) + &
                 rmask(0, ny) * real(tp(itr, tpcol(0, ny), k, nn), rp)), pp)
          else
            tp(itr, tpcol(0, ny + 1), k, nn) = 0.0_pp
          endif
        endif
        if (ne) then
          cff = rmask(nx, ny + 1) + rmask(nx + 1, ny)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(itr, tpcol(nx + 1, ny + 1), k, nn) = real(cff * (rmask(nx, ny + 1) * real(tp(itr, tpcol(nx, ny + 1), k, nn), rp) + &
                 rmask(nx + 1, ny) * real(tp(itr, tpcol(nx + 1, ny), k, nn), rp)), pp)
          else
            tp(itr, tpcol(nx + 1, ny + 1), k, nn) = 0.0_pp
          endif
        endif
      enddo
    enddo
  end subroutine passive_t3dbc

end module passive_mod

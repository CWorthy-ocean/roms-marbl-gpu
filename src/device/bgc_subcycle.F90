#include "mem_attr.inc"
#include "cuf_launch.inc"
module bgc_subcycle_mod
  ! BGC tracer subcycling (UCLA ROMS bgc_subcycle_mod, after 0.14.0, #428
  ! and #430): keys bgc_physics_subcycle, bgc_physics_timestep of the
  ! namelist group BGC_SETTINGS (read in avg_out), and the testing keys
  ! bgc_subcycle_pos and bgc_subcycle_diag.
  !
  ! The tracers beyond T and S (tp: MARBL, generic passives, CDR_LITE; the
  ! CPU's itrc > iTandS) are to be transported, mixed and forced once per
  ! window of k = bgc_physics_timestep/dt steps, from the transports summed
  ! every step; MARBL runs at the window ends that are MARBL steps. T, S and
  ! the dynamics are unchanged.
  !
  ! At a window end (bgcsub_step, after step3d_t, in place of
  ! marbl_gpu_step), as the CPU's bgcsub_window_tile: the window's Courant
  ! numbers and per-level sub-pass counts, a Lin-Rood swept-volume pass
  ! horizontally (third-order partial-cell swept flux, pseudo-density
  ! inner operators, the #415 upwind rule on inflow faces of open sides,
  ! TS_DIF2 diffusion of the window-start state over the window), a 1-D
  ! swept flux vertically in the pseudo-volumes left by the horizontal
  ! pass, the sources (surface fluxes: CFLX_CORR restoring of DIC/ALK and
  ! the CDR_LITE fluxes; CDR releases in every mode), implicit vertical
  ! mixing with the window-integrated Akt, the window boundary conditions,
  ! MARBL at MARBL steps, the exchange and the copy to the other time
  ! slots. River faces carry the window's river tracer transport in the
  ! horizontal pass, on every sub-pass. Kernels in bgc_window_mod.
  !
  ! GPU port, stage 4 (GPU_BGC_SUBCYCLE_SCOPE.md): a level whose Courant
  ! numbers need it is split into mk(k) sub-passes, as on the CPU; rivers,
  ! CDR releases (parameterized, volume, depth profiles, 3D), CDR_LITE and
  ! passive tracers with or without MARBL are in; bgc_subcycle_pos and
  ! bgc_subcycle_diag >= 2 are refused at start-up.
  !
  ! Between window ends the BGC state is held in time slots 1 and 2 of tp
  ! and at the surface of slot 3 (the only part of slot 3 read between
  ! windows: CFLX_CORR restoring and the CDR_LITE gas flux read slot nrhs at
  ! the surface); with bgc_subcycle_diag > 0 the rest of slot 3 is filled
  ! with a signalling NaN, as on the CPU.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state, only: Hz, FlxU, FlxV, We, Wi, Akt, pm, pn, rmask, swflx, nwet, ntpcol, build_tp_cross, nxc
  use time_state, only: iic, nnew, nstp, dt, ntstart, ntimes
  use passive_mod, only: ntp, tp, passive_hold, passive_init, tp_subcycled, tnu2_tp, passive_name, &
       passive_bry, tb_west, tb_east, tb_south, tb_north
  use halo_mod, only: halo_fill_3d_tp, halo_fill_3d_tp_k, halo_fill_3d_k
  use marbl_gpu, only: marbl_on, marbl_timestep_ratio, marbl_gpu_step
  use his_io, only: wrt_file_rst, output_period_rst
  use upscale_out, only: do_upscale
  use pflx_mod, only: ub_tune
  use rivers_mod, only: river_on, rivers_win_faces, rivers_win_acc
  use cdr_src_mod, only: cdr_on, cdr_win_nq, cdr_win_cols, cdr_win_acc, cdr_win_tr3d
  use cdr_lite_mod, only: cdrl_on, cdrl_psf, psf_cflx
  use bgc_window_mod, only: WLEV, bgcw_stats, bgcw_hcoef, bgcw_inner, bgcw_outer, bgcw_factor, &
       bgcw_vertical, bgcw_t3dbc, bgcw_reduce, bgcw_vs, bgcw_edge_r, bgcw_subq, bgcw_edge_tp, bgcw_rcol, bgcw_rbits
  use comm_mod, only: comm_rank, comm_allreduce_max, comm_allreduce_min, comm_allreduce_sum, comm_abort
  use arch_mod, only: sync_arch
  use run_timers, only: rt_ev_begin, rt_ev_end, RT_BGCACC, RT_BGCWIN, RT_BGCH0, RT_BGCH, RT_BGCV, RT_BGCBC, &
       RT_BGCHS
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_signaling_nan
  implicit none
  private

  ! Namelist keys (group BGC_SETTINGS, read in avg_out)
  logical, public :: bgc_physics_subcycle = .false.
  real(rp), public :: bgc_physics_timestep = 0.0_rp   ! [s]; 0: marbl_timestep
  logical, public :: bgc_subcycle_pos = .false.       ! testing/expert key
  integer, public :: bgc_subcycle_diag = 0            ! testing/expert key
  integer(ip), public :: bgcsub_k = 1
  public :: bgcsub_init, bgcsub_accumulate, bgcsub_step

  character(len=*), parameter :: module_name = 'bgc_subcycle'
  integer(ip) :: nwin = 0           ! steps accumulated in the current window
  integer(ip) :: nwindows = 0
  integer :: nerr = 0               ! start-up errors raised (bgcsub_init)
  real(rp) :: cmax_run = 0.0_rp, wmax_run = 0.0_rp
  real(rp) :: tnumax = 0.0_rp       ! largest background tnu2 of the BGC tracers
  ! accumulated transports [m^3], vertical transport and diffusivity summed
  ! over the window; window-start thickness [m]
  real(rp), WORK_ATTR allocatable :: VX(:,:,:), VY(:,:,:), WA(:,:,:), KA(:,:,:), HZS(:,:,:)
  ! accumulated surface fluxes [conc m] per tracer and interior wet column
  ! (the CFLX_CORR restoring of DIC/ALK, cdrl_psf); rain volume [m] (shadow)
  real(rp), WORK_ATTR allocatable :: SFA(:,:), RAINA(:,:)
  ! shadow uniform tracer (bgc_subcycle_diag > 0): its window-start value
  ! (1, every stored column), its content / concentration and scratch (tp
  ! layout)
  real(pp), WORK_ATTR allocatable :: QONE(:,:,:), CSH(:,:,:), QSH(:,:,:)
  ! sub-passes per level (mk: device copy of mkh) and the pseudo-thickness
  ! VS of sub-passes 2 and up; the cells of the physical-edge rows
  ! (edge_copy)
  integer(ip), allocatable :: mkh(:)
  integer(ip), WORK_ATTR allocatable :: mk(:)
  real(rp), WORK_ATTR allocatable :: VS(:,:,:)
  integer(ip) :: neb = 0
  integer(ip), WORK_ATTR allocatable :: ebi(:), ebj(:)
  ! window work: 1/(sub-passes) per level; per-level coefficients and the
  ! inner operators of WLEV levels; implicit-diffusion factors; the
  ! Courant-number reductions
  real(rp), WORK_ATTR allocatable :: fmk(:)
  real(rp), WORK_ATTR allocatable :: wvol(:,:,:), wcx(:,:,:), wcy(:,:,:), wdx(:,:,:), wdy(:,:,:)
  real(rp), WORK_ATTR allocatable :: wsx(:), wsy(:)
  real(rp), WORK_ATTR allocatable :: wv1(:,:,:), wfc(:,:,:), wpv(:,:,:), wcf(:,:,:)
  real(rp), WORK_ATTR allocatable :: wrows(:,:,:), wlck(:,:)
  ! diagnostics (bgc_subcycle_diag > 0; one element otherwise): per tracer
  ! and column the content before and after, the surface source, the new
  ! negatives (min, count); the shadow's max |c - 1|; the boundary-face
  ! fluxes of the edge rows; their per-tracer reductions
  real(rp), WORK_ATTR allocatable :: dbef(:,:), daft(:,:), dsrf(:,:), dnmn(:,:), dncn(:,:), dshd(:,:)
  real(rp), WORK_ATTR allocatable :: dew(:,:,:), dee(:,:,:), des(:,:,:), den(:,:,:)
  real(rp), WORK_ATTR allocatable :: rred(:,:)
  ! boundary-data placeholders when the passives have none (passive_bry)
  real(rp), WORK_ATTR allocatable :: tbx0(:,:,:), tby0(:,:,:)
  ! river faces of the window (rivers_mod RIV_WIN; the CPU's hu/hv lists):
  ! location and type (wfi, wfj, wft), window face index per u / v point
  ! (wfu, wfv; 0: none), face bits per cross-region column (rbx) and per
  ! interior wet column (rbo); the tracer transport summed over the window
  ! [conc m^3] (RUA, the CPU's RUA/RVA), the column sum of the summed face
  ! transport (RCL: rcolu/rcolv), the budget's river term per face (DRV)
  integer(ip) :: nwf = 0
  integer(ip), WORK_ATTR allocatable :: wfi(:), wfj(:), wft(:), wfu(:,:), wfv(:,:), rbx(:), rbo(:)
  ! the columns that take the kernels' source variants (bgc_window_mod
  ! *_src): cross-region columns with river faces (rlx), interior wet
  ! columns with river faces (rlo), interior wet columns with point
  ! sources (rlv; hsm = 1)
  integer(ip) :: nrx = 0, nro = 0, nrv = 0
  integer(ip), WORK_ATTR allocatable :: rlx(:), rlo(:), rlv(:), hsm(:)
  real(rp), WORK_ATTR allocatable :: RUA(:,:), RCL(:), DRV(:,:)
  ! CDR releases: profiles of this tile (nqw; cdr_src), their ranges per
  ! interior wet column (wq0..wq1), release contents summed over the
  ! window per tracer, profile and level (CPA [conc m^3/(pm pn)]), the
  ! shadow's volume (CPV, volume releases), the 3D forcing of ALK and DIC
  ! (passive c3a, c3d) summed (C3)
  integer(ip) :: nqw = 0, c3a = 0, c3d = 0
  integer(ip), WORK_ATTR allocatable :: wq0(:), wq1(:)
  real(rp), WORK_ATTR allocatable :: CPA(:,:,:), CPV(:,:,:), C3(:,:,:,:)

contains

  ! ----------------------------------------------------------------------
  subroutine bgcsub_init()
    ! After the initial state and set_depth (load_case): the CPU's checks,
    ! allocation, window-start thickness, BGC state copied to all slots.
    character(len=512) :: info
    integer(ip) :: nrst, is, ie, js, je, nz, nx, ny, l, c, k, m, n, ns, ncol, nd

    if (.not. bgc_physics_subcycle) return
    nerr = 0
    if (ntp < 1) call raise('bgcsub_init', &
         'bgc_physics_subcycle: there are no tracers beyond temperature/salinity')
    ! k = bgc_physics_timestep/dt (0: marbl_timestep as MARBL rounds it, i.e.
    ! marbl_timestep_ratio). MARBL keeps its own interval and runs at the end
    ! of a window, so every MARBL step must end one: k must divide
    ! marbl_timestep/dt.
    if (marbl_on) then
      if (bgc_physics_timestep == 0.0_rp) then
        bgcsub_k = marbl_timestep_ratio
      else
        call bgcsub_window_steps(int(marbl_timestep_ratio, ip))
      endif
    else
      if (bgc_physics_timestep == 0.0_rp) then
        call raise('bgcsub_init', 'bgc_physics_subcycle without MARBL needs bgc_physics_timestep > 0'// &
             ' (the window length in s, a whole multiple of dt = '//secs(dt)//' s)')
      else
        call bgcsub_window_steps(0_ip)
      endif
    endif
    ! Restarts only on window boundaries (the window sums are not in the
    ! file). Windows end on steps iic with modulo(iic, k) == 0 (every MARBL
    ! step among them); restart records are written every output_period_rst
    ! from the start of the run, so the run must also start on a window
    ! boundary. (monthly_restarts is not on the GPU: feature_guard stops it.)
    if (wrt_file_rst .and. bgcsub_k > 1) then
      nrst = nint(output_period_rst / dt, ip)
      if (abs(nrst * dt - output_period_rst) > 1.0e-6_rp * dt .or. &
          modulo(nrst, bgcsub_k) /= 0 .or. modulo(ntstart - 1, bgcsub_k) /= 0) then
        write(info,'(A,I0,A,I0,A,I0,A)') 'bgc_physics_subcycle: restarts must fall on window'// &
             ' boundaries: output_period_rst must be a multiple of bgc_physics_timestep = '// &
             secs(bgcsub_k * dt)//' s (', bgcsub_k, ' x dt), monthly_restarts off, and the run'// &
             ' must start on a window boundary (time step ', ntstart - 1, ' of the initial file'// &
             ' a multiple of ', bgcsub_k, ')'
        call raise('bgcsub_init', info)
      endif
    endif
    ! Not with subcycling (UCLA ROMS: UPSCALING output, SPONGE_TUNE ub_tune;
    ! GPU only: the salinity-mirror validation tracer, and what the GPU does
    ! not have yet)
    if (do_upscale) call raise('bgcsub_init', 'bgc_physics_subcycle does not support UPSCALING output')
    if (ub_tune) call raise('bgcsub_init', 'bgc_physics_subcycle does not support SPONGE_TUNE ub_tune')
    if (trim(passive_init) == 'mirror_salt') call raise('bgcsub_init', &
         "bgc_physics_subcycle does not support &passive_tracers init_mode = 'mirror_salt'")
    if (bgc_subcycle_pos) call raise('bgcsub_init', &
         'bgc_subcycle_pos (the positivity limiter) is not on the GPU yet')
    if (bgc_subcycle_diag >= 2) call raise('bgcsub_init', &
         'bgc_subcycle_diag >= 2 (the positivity test) is not on the GPU yet')
    if (pp /= rp) call raise('bgcsub_init', &
         'bgc_physics_subcycle needs double-precision passive tracers (a build with -DPASSIVE_FP64)')
    if (decomp%nz > COL_KMAX) call raise('bgcsub_init', 'bgc_physics_subcycle: nz exceeds COL_KMAX')
    if (nerr > 0) error stop 'bgc_subcycle: bgc_physics_subcycle settings rejected (see above)'

    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    nx = decomp%nx; ny = decomp%ny
    allocate(VX(is:ie, js:je, nz), VY(is:ie, js:je, nz), HZS(is:ie, js:je, nz))
    allocate(WA(is:ie, js:je, 0:nz), KA(is:ie, js:je, 0:nz))
    allocate(SFA(ntp, max(nwet, 1_ip)), RAINA(is:ie, js:je))
    allocate(fmk(nz), mkh(nz), mk(nz), VS(is:ie, js:je, nz))
    fmk = 1.0_rp; mkh = 1; mk = 1; VS = 0.0_rp
    call build_edge_list()
    allocate(wvol(is:ie, js:je, WLEV), wcx(is:ie, js:je, WLEV), wcy(is:ie, js:je, WLEV))
    allocate(wdx(is:ie, js:je, WLEV), wdy(is:ie, js:je, WLEV))
    allocate(wsx(ntp * (ntpcol + 1) * WLEV), wsy(ntp * (ntpcol + 1) * WLEV))
    allocate(wv1(is:ie, js:je, nz), wfc(is:ie, js:je, nz), wpv(is:ie, js:je, nz), wcf(is:ie, js:je, nz))
    allocate(wrows(7, 0:nx + 1, nz), wlck(7, nz))
    wvol = 0.0_rp; wcx = 0.0_rp; wcy = 0.0_rp; wdx = 0.0_rp; wdy = 0.0_rp
    wsx = 0.0_rp; wsy = 0.0_rp
    wv1 = 1.0_rp; wfc = 0.0_rp; wpv = 1.0_rp; wcf = 0.0_rp
    nd = merge(1_ip, 0_ip, bgc_subcycle_diag > 0)
    if (nd == 1) then
      allocate(QONE(1, 0:ntpcol, nz), CSH(1, 0:ntpcol, nz), QSH(1, 0:ntpcol, nz))
      QONE = 1.0_pp; CSH = 0.0_pp; QSH = 0.0_pp
      allocate(dbef(ntp, max(nwet, 1_ip)), daft(ntp, max(nwet, 1_ip)), dsrf(ntp, max(nwet, 1_ip)))
      allocate(dnmn(ntp, max(nwet, 1_ip)), dncn(ntp, max(nwet, 1_ip)), dshd(1, max(nwet, 1_ip)))
      allocate(dew(ntp, ny, nz), dee(ntp, ny, nz), des(ntp, nx, nz), den(ntp, nx, nz))
    else
      allocate(QONE(1, 1, 1), CSH(1, 1, 1), QSH(1, 1, 1))
      allocate(dbef(1, 1), daft(1, 1), dsrf(1, 1), dnmn(1, 1), dncn(1, 1), dshd(1, 1))
      allocate(dew(1, 1, 1), dee(1, 1, 1), des(1, 1, 1), den(1, 1, 1))
    endif
    dbef = 0.0_rp; daft = 0.0_rp; dsrf = 0.0_rp; dnmn = 0.0_rp; dncn = 0.0_rp; dshd = 0.0_rp
    dew = 0.0_rp; dee = 0.0_rp; des = 0.0_rp; den = 0.0_rp
    allocate(rred(max(ntp, 1_ip), 12))
    rred = 0.0_rp
    if (.not. passive_bry) then
      allocate(tbx0(js:je, nz, 1), tby0(is:ie, nz, 1))
      tbx0 = 0.0_rp; tby0 = 0.0_rp
    endif
    tnumax = maxval(tnu2_tp(1:ntp))
    call build_tp_cross()
    call init_sources(nd)
    call reset_sums()
    nwin = 0

    ! the BGC state into every slot (all stored columns, halos included)
    ns = nstp
    n = ntp
    ncol = ntpcol_n()
    do l = 1, 3
      if (l == ns) cycle
      CUF_DO3
      do k = 1, nz
        do c = 0, ncol
          do m = 1, n
            tp(m, c, k, l) = tp(m, c, k, ns)
          enddo
        enddo
      enddo
    enddo
    tp_subcycled = .true.

    if (comm_rank() == 0) then
      write(*,'(/1x,A,I0,A,I0,A)') 'BGC subcycling: tracers ', 3, '-', 2 + ntp, &
           ', bgc_physics_timestep = '//secs(bgcsub_k * dt)//' s'
      write(*,'(1x,A,I0,A,L1,A,I0)') '  k = ', bgcsub_k, ', positivity limiter ', &
           bgc_subcycle_pos, ', diagnostics ', bgc_subcycle_diag
      if (marbl_on) write(*,'(1x,A,I0,A)') '  MARBL at the end of every ', &
           marbl_timestep_ratio / bgcsub_k, ' window(s)'
      if (modulo(ntstart - 1, bgcsub_k) /= 0) write(*,'(1x,A)') &
           '  the run starts inside a window: the first window is shorter'
      if (modulo(ntstart + ntimes - 1, bgcsub_k) /= 0) write(*,'(1x,A,I0,A)') &
           '  the run ends inside a window: the BGC transport of its last ', &
           modulo(ntstart + ntimes - 1, bgcsub_k), ' steps is not applied'
    endif
  end subroutine bgcsub_init

  subroutine init_sources(nd)
    ! River faces and CDR releases of the window (nd = 1: diagnostics)
    integer(ip), intent(in) :: nd
    integer(ip) :: is, ie, js, je, nx, ny, nz, w
    integer(ip), allocatable :: hi(:), hj(:), ht(:), mu(:,:), mv(:,:)
    call sync_arch()
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nwf = 0
    if (river_on) then
      call rivers_win_faces(nwf, hi, hj, ht)
    else
      allocate(hi(1), hj(1), ht(1))
      hi = 0; hj = 0; ht = 1
    endif
    allocate(mu(is:ie, js:je), mv(is:ie, js:je))
    mu = 0; mv = 0
    do w = 1, nwf
      if (ht(w) == 1) then
        mu(hi(w), hj(w)) = w
      else
        mv(hi(w), hj(w)) = w
      endif
    enddo
    allocate(wfi(max(nwf, 1_ip)), wfj(max(nwf, 1_ip)), wft(max(nwf, 1_ip)))
    allocate(wfu(is:ie, js:je), wfv(is:ie, js:je), rbx(max(nxc, 1_ip)), rbo(max(nwet, 1_ip)))
    wfi = hi(1:max(nwf, 1_ip)); wfj = hj(1:max(nwf, 1_ip)); wft = ht(1:max(nwf, 1_ip))
    wfu = mu; wfv = mv
    rbx = 0; rbo = 0
    deallocate(hi, hj, ht, mu, mv)
    if (nwf > 0) call bgcw_rbits(wfu, wfv, rbx, rbo)
    allocate(RUA(max(ntp, 1_ip), max(nwf, 1_ip)), RCL(max(nwf, 1_ip)))
    RUA = 0.0_rp; RCL = 0.0_rp
    if (nd == 1 .and. nwf > 0) then
      allocate(DRV(max(ntp, 1_ip), nwf))
    else
      allocate(DRV(1, 1))
    endif
    DRV = 0.0_rp
    nqw = cdr_win_nq()
    call cdr_win_tr3d(c3a, c3d)
    allocate(wq0(max(nwet, 1_ip)), wq1(max(nwet, 1_ip)))
    call cdr_win_cols(wq0, wq1)
    allocate(CPA(max(ntp, 1_ip), max(nqw, 1_ip), nz), CPV(1, max(nqw, 1_ip), nz))
    if (c3a > 0) then
      allocate(C3(nx, ny, nz, 2))
    else
      allocate(C3(1, 1, 1, 2))
    endif
    CPA = 0.0_rp; CPV = 0.0_rp; C3 = 0.0_rp
    call sync_arch()
    ! the column lists of the source variants
    allocate(hsm(max(nwet, 1_ip)))
    hsm = 0
    do w = 1, nwet
      if (wq1(w) >= wq0(w) .or. c3a > 0) hsm(w) = 1
    enddo
    nrx = count(rbx(1:max(nxc, 1_ip)) /= 0); nro = count(rbo(1:max(nwet, 1_ip)) /= 0); nrv = count(hsm /= 0)
    if (nxc == 0) nrx = 0
    if (nwet == 0) then
      nro = 0; nrv = 0
    endif
    allocate(rlx(max(nrx, 1_ip)), rlo(max(nro, 1_ip)), rlv(max(nrv, 1_ip)))
    rlx = 1; rlo = 1; rlv = 1
    nrx = 0
    do w = 1, nxc
      if (rbx(w) /= 0) then
        nrx = nrx + 1; rlx(nrx) = w
      endif
    enddo
    nro = 0; nrv = 0
    do w = 1, nwet
      if (rbo(w) /= 0) then
        nro = nro + 1; rlo(nro) = w
      endif
      if (hsm(w) /= 0) then
        nrv = nrv + 1; rlv(nrv) = w
      endif
    enddo
    if (nwf > 0 .or. nrv > 0) write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0)') 'BGC subcycling: rank ', comm_rank(), &
         '  river faces ', nwf, ' (columns: inner ', nrx, ', outer ', nro, ')  point-source columns ', nrv, &
         '  CDR profiles ', nqw
  end subroutine init_sources

  subroutine build_edge_list()
    ! the cells of the physical-edge rows over the whole array (the CPU's
    ! edge_copy: dst(istr-1,:,:) at a western edge, ...), each once
    integer(ip) :: i, j, is, ie, js, je, nx, ny, n
    logical, allocatable :: e(:,:)
    integer(ip), allocatable :: hi(:), hj(:)
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nx = decomp%nx; ny = decomp%ny
    allocate(e(is:ie, js:je))
    e = .false.
    if (decomp%edge_w) e(0, :) = .true.
    if (decomp%edge_e) e(nx + 1, :) = .true.
    if (decomp%edge_s) e(:, 0) = .true.
    if (decomp%edge_n) e(:, ny + 1) = .true.
    neb = int(count(e), ip)
    allocate(hi(max(neb, 1_ip)), hj(max(neb, 1_ip)))
    hi = 0; hj = 0
    n = 0
    do j = js, je
      do i = is, ie
        if (e(i, j)) then
          n = n + 1; hi(n) = i; hj(n) = j
        endif
      enddo
    enddo
    allocate(ebi(max(neb, 1_ip)), ebj(max(neb, 1_ip)))
    if (neb == 0) then
      hi(1) = is; hj(1) = js
    endif
    ebi = hi; ebj = hj
    deallocate(e, hi, hj)
  end subroutine build_edge_list

  integer(ip) function ntpcol_n()
    ! last stored column of tp (tp(:, 0:ntpcol, :, :))
    ntpcol_n = int(ubound(tp, 2), ip)
  end function ntpcol_n

  ! ----------------------------------------------------------------------
  subroutine raise(where, info)
    ! the CPU's error_log%raise_global, printed at once by compute rank 0
    character(len=*), intent(in) :: where, info
    nerr = nerr + 1
    if (comm_rank() == 0) then
      write(*,*) 'ERROR [global] [', module_name//'/'//where, ']:'
      write(*,*) trim(info)
      flush(6)
    endif
  end subroutine raise

  ! ----------------------------------------------------------------------
  subroutine bgcsub_window_steps(nm)
    ! bgcsub_k = bgc_physics_timestep/dt. It must be a whole multiple of dt
    ! (relative tolerance as for output_period_rst), at least dt, and with
    ! MARBL (nm = marbl_timestep/dt > 0) divide marbl_timestep. Otherwise
    ! stop, naming the nearest valid values: never round silently.
    integer(ip), intent(in) :: nm
    character(len=512) :: info
    real(rp) :: x
    integer(ip) :: kl, kh

    x = min(bgc_physics_timestep / dt, 1.0e8_rp)
    bgcsub_k = max(1_ip, nint(x, ip))
    if (x < 1.0_rp - 1.0e-6_rp) then
      info = 'bgc_physics_timestep = '//secs(bgc_physics_timestep)// &
           ' s is shorter than dt = '//secs(dt)//' s'
    else if (abs(bgcsub_k * dt - bgc_physics_timestep) > 1.0e-6_rp * dt) then
      info = 'bgc_physics_timestep = '//secs(bgc_physics_timestep)// &
           ' s is not a whole multiple of dt = '//secs(dt)//' s'
    else if (nm > 0 .and. modulo(nm, bgcsub_k) /= 0) then
      info = 'marbl_timestep = '//secs(nm * dt)//' s (as MARBL rounds it) is not'// &
           ' a whole multiple of bgc_physics_timestep = '//secs(bgc_physics_timestep)//' s'
    else
      return
    endif
    ! nearest valid window lengths below and above (with MARBL: divisors of nm)
    kl = floor(x + 1.0e-6_rp, ip)
    if (nm > 0) kl = min(kl, nm)
    do while (kl >= 1)
      if (nm == 0) exit
      if (modulo(nm, kl) == 0) exit
      kl = kl - 1
    enddo
    kh = max(1_ip, ceiling(x - 1.0e-6_rp, ip))
    do while (nm > 0 .and. kh <= nm)
      if (modulo(nm, kh) == 0) exit
      kh = kh + 1
    enddo
    info = trim(info)//'; nearest valid bgc_physics_timestep:'
    if (kl >= 1) info = trim(info)//' '//secs(kl * dt)//' s'
    if (kl >= 1 .and. (nm == 0 .or. kh <= nm)) info = trim(info)//' or'
    if (nm == 0 .or. kh <= nm) info = trim(info)//' '//secs(kh * dt)//' s'
    if (nm > 0) info = trim(info)//' (0 = marbl_timestep)'
    call raise('bgcsub_window_steps', info)
  end subroutine bgcsub_window_steps

  ! ----------------------------------------------------------------------
  function secs(v) result(str)
    ! A time in s for messages: an integer when it is one.
    real(rp), intent(in) :: v
    character(len=:), allocatable :: str
    character(len=32) :: buf
    if (abs(v) < 1.0e9_rp .and. abs(v - anint(v)) <= 1.0e-9_rp * max(1.0_rp, abs(v))) then
      write(buf,'(I0)') nint(v, kind=8)
    else
      write(buf,'(G0.6)') v
    endif
    str = trim(adjustl(buf))
  end function secs

  ! ----------------------------------------------------------------------
  subroutine bgcsub_accumulate()
    ! Every step, after step3d_uv2 + omega (the corrector transports that
    ! step3d_t uses) and the corrector's set_forces (n+1/2): sum dt*FlxU,
    ! dt*FlxV, dt*(We+Wi) and dt*Akt(salt) over the tile with its halo
    ! (FlxU/FlxV exchanged by step3d_uv2, We/Wi set by omega there), the
    ! surface fluxes of the passives (cdrl_psf: CFLX_CORR and CDR_LITE),
    ! the river-face tracer transports (RUA), the CDR release contents
    ! (CPA, C3; CPV) and, for the shadow tracer, the rain volume dt*swflx.
    ! With IMEX_IGW these are IMEX's corrector transports (step3d_uv2's
    ! wflx_corr fluxes, 'modal' open faces included), as on the CPU
    ! (ucla-roms 809aa24a main.F90: bgcsub_accumulate before step3d_t).
    integer(ip) :: i, j, k, is, ie, js, je, nz, c, m, n, nw
    real(rp) :: dtl
    if (.not. bgc_physics_subcycle) return
    call rt_ev_begin(RT_BGCACC)
    nwin = nwin + 1
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    dtl = dt
    CUF_DO3
    do k = 0, nz
      do j = js, je
        do i = is, ie
          if (k >= 1) then
            VX(i, j, k) = VX(i, j, k) + dtl * FlxU(i, j, k)
            VY(i, j, k) = VY(i, j, k) + dtl * FlxV(i, j, k)
          endif
          WA(i, j, k) = WA(i, j, k) + dtl * (We(i, j, k) + Wi(i, j, k))
          KA(i, j, k) = KA(i, j, k) + dtl * Akt(i, j, k, 2)
        enddo
      enddo
    enddo
    if (river_on) call rivers_win_acc(RUA, ntp, dtl)
    if (cdr_on) call cdr_win_acc(ntp, CPA, merge(1_ip, 0_ip, bgc_subcycle_diag > 0), CPV, C3, dtl)
    if (psf_cflx .or. cdrl_on) then
      n = ntp; nw = nwet
      CUF_DO2
      do c = 1, nw
        do m = 1, n
          SFA(m, c) = SFA(m, c) + dtl * cdrl_psf(m, c)
        enddo
      enddo
    endif
    if (bgc_subcycle_diag > 0) then
      CUF_DO2
      do j = js, je
        do i = is, ie
          RAINA(i, j) = RAINA(i, j) + dtl * swflx(i, j)
        enddo
      enddo
    endif
    call rt_ev_end(RT_BGCACC)
  end subroutine bgcsub_accumulate

  ! ----------------------------------------------------------------------
  subroutine bgcsub_step()
    ! After step3d_t, in place of marbl_gpu_step. At a window end (the CPU's
    ! bgcsub_window_tile): Courant numbers and sub-pass counts, horizontal
    ! pass, vertical pass with sources and implicit mixing, diagnostics,
    ! window boundary conditions, MARBL at MARBL steps, the Fe hold, the
    ! exchange of slot nnew, the copy to the other slots, the reset.
    integer(ip) :: nn, ns, l, c, k, m, n, nz, ncol, nwin_w, k0, k1, msub, idg, kc, isub, klo, khi, kb
    real(rp) :: dtw, cres, v1min, gmax(5), cff
    real(pp) :: snan
    real(8), allocatable :: lb(:), gb(:)
    character(len=512) :: info
    logical :: diag
    if (modulo(iic, bgcsub_k) /= 0) return
    call rt_ev_begin(RT_BGCWIN)
    nwindows = nwindows + 1
    nwin_w = nwin
    nn = nnew; ns = nstp; nz = decomp%nz; n = ntp; ncol = ntpcol_n()
    diag = (bgc_subcycle_diag > 0)
    idg = merge(1_ip, 0_ip, diag)
    dtw = real(nwin, rp) * dt

    ! ---- window Courant numbers, sub-passes, explicit diffusion number ----
    ! cface: |V| / volume of the upstream cell (the partial-cell sweep needs
    ! < 1); c1d: 1-D convergence / volume (the inner pseudo-volumes need < 1);
    ! cout: outflow sum / volume. Level k needs mk(k) = ceiling(crit/0.95)
    ! sub-passes (CPU); vertically any Courant number is fine.
    call rt_ev_begin(RT_BGCH0)
    call bgcw_stats(VX, VY, WA, HZS, tnumax, dtw, wrows, wlck)
    call sync_arch()
    allocate(lb(5 * nz + 2), gb(5 * nz + 2))
    do k = 1, nz
      do m = 1, 5
        lb(5 * (k - 1) + m) = real(wlck(m, k), 8)
      enddo
    enddo
    call comm_allreduce_max(int(5 * nz), lb(1:5 * nz), gb(1:5 * nz))
    lb(1) = real(maxval(wlck(6, 1:nz)), 8); lb(2) = real(maxval(wlck(7, 1:nz)), 8)
    call comm_allreduce_max(2, lb(1:2), gb(5 * nz + 1:5 * nz + 2))
    msub = 1; kc = 1
    gmax = 0.0_rp; v1min = 1.0_rp
    do k = 1, nz
      mkh(k) = max(1_ip, ceiling(real(gb(5 * (k - 1) + 5), rp) / 0.95_rp, ip))
      fmk(k) = 1.0_rp / real(mkh(k), rp)
      mk(k) = mkh(k)
      if (mkh(k) > msub) then
        msub = mkh(k); kc = k
      endif
      gmax(1) = max(gmax(1), real(gb(5 * (k - 1) + 1), rp))
      gmax(2) = max(gmax(2), real(gb(5 * (k - 1) + 2), rp))
      gmax(3) = max(gmax(3), real(gb(5 * (k - 1) + 3), rp))
      if (k == 1) then
        v1min = real(gb(4), rp)
      else
        v1min = max(v1min, real(gb(5 * (k - 1) + 4), rp))
      endif
    enddo
    v1min = -v1min
    gmax(4) = real(gb(5 * nz + 1), rp); gmax(5) = real(gb(5 * nz + 2), rp)
    deallocate(lb, gb)
    cmax_run = max(cmax_run, gmax(1)); wmax_run = max(wmax_run, gmax(5))
    cff = max(gmax(1), gmax(2))
    if (v1min < 0.05_rp .or. gmax(4) > 1.0_rp .or. msub > 50) then
      write(info,'(A,I0,A,F8.3,A,F8.3,A,F7.3,A)') 'BGC window ending at step ', iic, &
           ': horizontal Courant ', cff, ', smallest pseudo-thickness ratio ', v1min, &
           ' (must be >= 0.05) or explicit diffusion number ', gmax(4), ' (must be <= 1):'// &
           ' shorten the window (bgc_physics_timestep, or marbl_timestep if it is 0) or reduce v_sponge/tnu2'
      call window_stop('bgcsub_window', info)
    endif
    call rt_ev_end(RT_BGCH0)
    ! window continuity (the CPU's cont): Hz_end = v1 - d_z W, k < nz
    cres = 0.0_rp
    if (diag) then
      allocate(lb(1), gb(1))
      lb(1) = real(window_cont(), 8)
      call comm_allreduce_max(1, lb, gb)
      cres = real(gb(1), rp)
      deallocate(lb, gb)
    endif

    ! ---- horizontal pass: content per area into slot nnew (shadow: CSH).
    ! Sub-pass 1 from the window-start state (slot nstp) on every level, in
    ! level blocks of WLEV; sub-pass isub > 1 on the levels with
    ! isub <= mk(k) (levels klo..khi hold them all), from the previous
    ! sub-pass's concentrations q = content / VS in slot 3 (free during the
    ! window: every level is rewritten at its end), with VS = HZS - (isub-1)
    ! fm pm pn div V; VS and q exchanged over klo..khi, the edge rows from
    ! HZS and the window-start state (the CPU's edge_copy). Diffusion on
    ! sub-pass 1 only.
    call rt_ev_begin(RT_BGCH)
    ! river faces: the column sums of the summed face transports (rcolu/v)
    if (nwf > 0) call bgcw_rcol(nwf, wfi, wfj, wft, VX, VY, RCL)
    do isub = 1, msub
      if (isub == 1) then
        klo = 1; khi = nz
      else
        if (isub == 2) call rt_ev_begin(RT_BGCHS)
        klo = nz; khi = 1
        do k = 1, nz
          if (isub <= mkh(k)) then
            klo = min(klo, k); khi = max(khi, k)
          endif
        enddo
        call bgcw_vs(isub, klo, khi, mk, fmk, VX, VY, HZS, VS)
        call halo_fill_3d_k(VS, STAGGER_RHO, klo, khi)
        call bgcw_edge_r(klo, khi, neb, ebi, ebj, HZS, VS)
        call bgcw_subq(n, isub, klo, khi, mk, tp(1, 0, 1, nn), VS, tp(1, 0, 1, 3))
        call halo_fill_3d_tp_k(tp, ntp, 3_ip, 3_ip, klo, khi)
        call bgcw_edge_tp(n, klo, khi, neb, ebi, ebj, tp(1, 0, 1, ns), tp(1, 0, 1, 3))
        if (diag) then
          call bgcw_subq(1_ip, isub, klo, khi, mk, CSH, VS, QSH)
          call halo_fill_3d_tp_k(QSH, 1_ip, 1_ip, 1_ip, klo, khi)
          call bgcw_edge_tp(1_ip, klo, khi, neb, ebi, ebj, QONE, QSH)
        endif
      endif
      do kb = klo, khi, WLEV
        ! the block's first and last level with a sub-pass isub (none: skip)
        k0 = kb + WLEV; k1 = kb - 1
        do k = kb, min(kb + WLEV - 1, khi)
          if (isub <= mkh(k)) then
            k0 = min(k0, k); k1 = max(k1, k)
          endif
        enddo
        if (k1 < k0) cycle
        if (isub == 1) then
          call bgcw_hcoef(k0, k1, isub, mk, VX, VY, HZS, fmk, wvol, wcx, wcy, wdx, wdy)
          if (diag) then
            call bgcw_inner(1_ip, k0, k1, isub, mk, QONE, wsx, wsy, VX, VY, wvol, wcx, wcy, wdx, wdy, fmk, &
                            1_ip, rbx, nrx, rlx, wfu, wfv, RUA, RCL)
            call bgcw_outer(1_ip, k0, k1, isub, mk, QONE, wsx, wsy, CSH, VX, VY, HZS, wcx, wcy, fmk, tnu2_tp, &
                            0_ip, dtw, 0_ip, dew, dee, des, den, 1_ip, rbo, nro, rlo, wfu, wfv, RUA, RCL, DRV)
          endif
          call bgcw_inner(n, k0, k1, isub, mk, tp(1, 0, 1, ns), wsx, wsy, VX, VY, wvol, wcx, wcy, wdx, wdy, fmk, &
                          0_ip, rbx, nrx, rlx, wfu, wfv, RUA, RCL)
          call bgcw_outer(n, k0, k1, isub, mk, tp(1, 0, 1, ns), wsx, wsy, tp(1, 0, 1, nn), VX, VY, HZS, wcx, wcy, &
                          fmk, tnu2_tp, 1_ip, dtw, idg, dew, dee, des, den, 0_ip, rbo, nro, rlo, wfu, wfv, RUA, RCL, DRV)
        else
          call bgcw_hcoef(k0, k1, isub, mk, VX, VY, VS, fmk, wvol, wcx, wcy, wdx, wdy)
          if (diag) then
            call bgcw_inner(1_ip, k0, k1, isub, mk, QSH, wsx, wsy, VX, VY, wvol, wcx, wcy, wdx, wdy, fmk, &
                            1_ip, rbx, nrx, rlx, wfu, wfv, RUA, RCL)
            call bgcw_outer(1_ip, k0, k1, isub, mk, QSH, wsx, wsy, CSH, VX, VY, HZS, wcx, wcy, fmk, tnu2_tp, &
                            0_ip, dtw, 0_ip, dew, dee, des, den, 1_ip, rbo, nro, rlo, wfu, wfv, RUA, RCL, DRV)
          endif
          call bgcw_inner(n, k0, k1, isub, mk, tp(1, 0, 1, 3), wsx, wsy, VX, VY, wvol, wcx, wcy, wdx, wdy, fmk, &
                          0_ip, rbx, nrx, rlx, wfu, wfv, RUA, RCL)
          call bgcw_outer(n, k0, k1, isub, mk, tp(1, 0, 1, 3), wsx, wsy, tp(1, 0, 1, nn), VX, VY, HZS, wcx, wcy, &
                          fmk, tnu2_tp, 0_ip, dtw, idg, dew, dee, des, den, 0_ip, rbo, nro, rlo, wfu, wfv, RUA, RCL, DRV)
        endif
      enddo
    enddo
    if (msub > 1) call rt_ev_end(RT_BGCHS)
    call rt_ev_end(RT_BGCH)

    ! ---- vertical swept flux, sources, implicit mixing ----
    call rt_ev_begin(RT_BGCV)
    call bgcw_factor(VX, VY, HZS, KA, wv1, wfc, wpv, wcf)
    ! (slot 3 holds the concentrations here: free during the window)
    call bgcw_vertical(n, tp(1, 0, 1, ns), tp(1, 0, 1, nn), tp(1, 0, 1, 3), WA, HZS, wv1, wfc, wpv, wcf, 0_ip, &
                       SFA, RAINA, idg, dbef, daft, dsrf, dnmn, dncn, max(nqw, 1_ip), nrv, rlv, hsm, wq0, wq1, c3a, c3d, CPA, C3)
    if (diag) call bgcw_vertical(1_ip, QONE, CSH, QSH, WA, HZS, wv1, wfc, wpv, wcf, 1_ip, SFA, RAINA, &
                                 1_ip, dshd, dshd, dshd, dshd, dshd, max(nqw, 1_ip), nrv, rlv, hsm, wq0, wq1, 0_ip, 0_ip, CPV, C3)
    call rt_ev_end(RT_BGCV)

    ! ---- diagnostics ----
    if (diag) call window_diag(nwin_w, msub, gmax, v1min, cres)

    ! ---- boundaries, MARBL, exchange, copy ----
    call rt_ev_begin(RT_BGCBC)
    if (passive_bry) then
      call bgcw_t3dbc(tp, ntp, ns, nn, VX, VY, HZS, .true., ntp, tb_west, tb_east, tb_south, tb_north)
    else
      call bgcw_t3dbc(tp, ntp, ns, nn, VX, VY, HZS, .false., 1_ip, tbx0, tbx0, tby0, tby0)
    endif
    call marbl_gpu_step()               ! returns unless iic is a MARBL step
    call passive_hold(nn)               ! use_constant_Fe (no-op otherwise)
    call halo_fill_3d_tp(tp, ntp, nn)
    ! Copy to the other slots: nnew is 1 or 2 here, and until the next
    ! window end only the other of the two is read in full; slot 3 is read
    ! only at the surface (see the module header).
    l = 3 - nn
    CUF_DO3
    do k = 1, nz
      do c = 0, ncol
        do m = 1, n
          tp(m, c, k, l) = tp(m, c, k, nn)
          if (k == nz) tp(m, c, k, 3) = tp(m, c, k, nn)
        enddo
      enddo
    enddo
    ! Test runs (bgc_subcycle_diag > 0): the rest of slot 3 is a signalling
    ! NaN, so that a reader of it between windows fails loudly.
    if (diag) then
      snan = ieee_value(1.0_pp, ieee_signaling_nan)
      CUF_DO3
      do k = 1, nz - 1
        do c = 0, ncol
          do m = 1, n
            tp(m, c, k, 3) = snan
          enddo
        enddo
      enddo
    endif
    call reset_sums()
    nwin = 0
    call rt_ev_end(RT_BGCBC)
    call rt_ev_end(RT_BGCWIN)
  end subroutine bgcsub_step

  ! ----------------------------------------------------------------------
  subroutine window_stop(where, info)
    ! A window that cannot be transported (the CPU's raise_global +
    ! abort_check in bgcsub_window_tile): the values come from global
    ! reductions, so every compute rank comes here; rank 0 writes the
    ! message, the ranks meet once more, and comm_abort stops all ranks,
    ! writers included (as fs_unconv_stop).
    character(len=*), intent(in) :: where, info
    real(8) :: x(1), y(1)
    call raise(where, info)
    x = 0.0d0
    call comm_allreduce_max(1, x, y)
    call comm_abort(1)
  end subroutine window_stop

  ! ----------------------------------------------------------------------
  subroutine window_diag(nwin_w, msub, gmax, v1min, cres)
    ! bgc_subcycle_diag > 0: the CPU's BGCSUB and BGCSUBNEG lines. Budget
    ! per tracer: content after - before - (boundary + surface + rivers +
    ! point sources), relative to the larger content; shadow: max |c - 1| of the
    ! uniform tracer; new negatives: cells >= 0 at the window start and < 0
    ! after it. Per-tracer reductions in a fixed order, then over the ranks.
    integer(ip), intent(in) :: nwin_w, msub
    real(rp), intent(in) :: gmax(5), v1min, cres
    integer(ip) :: n, nx, ny, nz, ib, lr, lneg
    real(8), allocatable :: bud(:,:), gbud(:,:), x(:), y(:)
    real(rp) :: rr, a, cff, shdev, ncnt
    n = ntp; nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ! rred(:, 1..9): before, after, surface, negative min, negative count,
    ! west, east, south, north boundary fluxes; rred(1, 10): shadow
    rred = 0.0_rp
    call bgcw_reduce(0_ip, n, nwet, dbef, rred(1, 1))
    call bgcw_reduce(0_ip, n, nwet, daft, rred(1, 2))
    call bgcw_reduce(0_ip, n, nwet, dsrf, rred(1, 3))
    call bgcw_reduce(2_ip, n, nwet, dnmn, rred(1, 4))
    call bgcw_reduce(0_ip, n, nwet, dncn, rred(1, 5))
    if (decomp%edge_w) call bgcw_reduce(0_ip, n, ny * nz, dew, rred(1, 6))
    if (decomp%edge_e) call bgcw_reduce(0_ip, n, ny * nz, dee, rred(1, 7))
    if (decomp%edge_s) call bgcw_reduce(0_ip, n, nx * nz, des, rred(1, 8))
    if (decomp%edge_n) call bgcw_reduce(0_ip, n, nx * nz, den, rred(1, 9))
    call bgcw_reduce(1_ip, 1_ip, nwet, dshd, rred(1, 10))
    call source_sums(rred(1, 11), rred(1, 12))
    call sync_arch()
    allocate(bud(6, n), gbud(6, n), x(n), y(n))
    do ib = 1, n
      bud(1, ib) = real(rred(ib, 1), 8)
      bud(2, ib) = real(rred(ib, 2), 8)
      bud(3, ib) = real(rred(ib, 3), 8)
      bud(4, ib) = real(rred(ib, 11), 8)
      bud(5, ib) = real(rred(ib, 6) - rred(ib, 7) + rred(ib, 8) - rred(ib, 9), 8)
      bud(6, ib) = real(rred(ib, 12), 8)
    enddo
    call comm_allreduce_sum(int(6 * n), bud, gbud)
    do ib = 1, n
      x(ib) = real(rred(ib, 4), 8)
    enddo
    call comm_allreduce_min(int(n), x, y)
    rred(1:n, 4) = real(y(1:n), rp)
    do ib = 1, n
      x(ib) = real(rred(ib, 5), 8)
    enddo
    call comm_allreduce_sum(int(n), x, y)
    rred(1:n, 5) = real(y(1:n), rp)
    x(1) = real(rred(1, 10), 8)
    call comm_allreduce_max(1, x(1:1), y(1:1))
    shdev = real(y(1), rp)
    ! budget residual
    rr = 0.0_rp; lr = 1
    do ib = 1, n
      a = real(gbud(2, ib) - gbud(1, ib) - (gbud(5, ib) + gbud(3, ib) + gbud(6, ib) + gbud(4, ib)), rp)
      cff = abs(a) / max(abs(real(gbud(1, ib), rp)), abs(real(gbud(2, ib), rp)), 1.0e-300_rp)
      if (cff > rr) then
        rr = cff; lr = ib
      endif
    enddo
    lneg = minloc(rred(1:n, 4), 1)          ! most negative value made by this window
    ncnt = sum(rred(1:n, 5))
    if (comm_rank() == 0) then
      write(*,'(1x,A,I7,I3,A,I2,A,3F7.3,A,F6.3,A,F7.4,A,F7.4,A,ES9.2,A,ES9.2,A,ES9.2,3A)') 'BGCSUB', &
           iic, nwin_w, ' m ', msub, ' Cf/C1/Co ', gmax(1), gmax(2), gmax(3), ' v1 ', v1min, &
           ' Cz ', gmax(5), ' Sd ', gmax(4), ' cont ', cres, ' shadow ', shdev, ' budget ', rr, &
           ' (', trim(passive_name(lr)), ')'
      write(*,'(1x,A,I7,A,I10,A,ES10.3,3A)') 'BGCSUBNEG', iic, ' new negatives ', nint(ncnt), &
           ' min ', rred(lneg, 4), ' (', trim(passive_name(lneg)), ')'
    endif
    deallocate(bud, gbud, x, y)
  end subroutine window_diag

  ! ----------------------------------------------------------------------
  subroutine source_sums(b4, b6)
    ! diagnostics, per tracer on this tile: the CDR release contents of the
    ! window (b4: sum over the profiles of sum over k of CPA; 3D forcing:
    ! sum of C3 over the interior) and the river inputs into wet interior
    ! cells (b6: sum over the faces of DRV)
    real(rp), WORK_ATTR intent(inout) :: b4(*), b6(*)
    integer(ip) :: itr, q, k, w, n, nq, nz, nd, i, j, nx, ny, ia, id
    real(rp) :: a, b, s3a, s3d
    n = ntp; nq = nqw; nz = decomp%nz
    nd = 0
    if (nwf > 0 .and. size(DRV, 2) == nwf) nd = nwf
    nx = decomp%nx; ny = decomp%ny; ia = c3a; id = c3d
    s3a = 0.0_rp; s3d = 0.0_rp
    if (ia > 0) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            s3a = s3a + C3(i, j, k, 1)
            s3d = s3d + C3(i, j, k, 2)
          enddo
        enddo
      enddo
    endif
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*>>>
#endif
    do itr = 1, n
      a = 0.0_rp
      do q = 1, nq
        b = 0.0_rp
        do k = 1, nz
          b = b + CPA(itr, q, k)
        enddo
        a = a + b
      enddo
      if (itr == ia) a = a + s3a
      if (itr == id) a = a + s3d
      b4(itr) = a
      b = 0.0_rp
      do w = 1, nd
        b = b + DRV(itr, w)
      enddo
      b6(itr) = b
    enddo
  end subroutine source_sums

  ! ----------------------------------------------------------------------
  real(rp) function window_cont() result(cres)
    ! max over interior wet cells, k = 1..nz-1, of |Hz - (v1 - d_z W)|/Hz,
    ! v1 = HZS - pm pn (dVX/dx + dVY/dy), W = WA pm pn with W = 0 at the
    ! bottom (the CPU's cres, same operand order). A device reduction.
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: cm, v1, wk, wkm
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    cm = 0.0_rp
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz - 1
      do j = 1, ny
        do i = 1, nx
          v1 = HZS(i, j, k) - pm(i, j) * pn(i, j) * (VX(i + 1, j, k) - VX(i, j, k) + VY(i, j + 1, k) - VY(i, j, k))
          wk = WA(i, j, k) * pm(i, j) * pn(i, j)
          wkm = 0.0_rp
          if (k > 1) wkm = WA(i, j, k - 1) * pm(i, j) * pn(i, j)
          cm = max(cm, abs(Hz(i, j, k) - (v1 - (wk - wkm))) / Hz(i, j, k) * rmask(i, j))
        enddo
      enddo
    enddo
    cres = cm
  end function window_cont

  ! ----------------------------------------------------------------------
  subroutine reset_sums()
    ! sums to 0, window-start thickness = Hz
    integer(ip) :: i, j, k, is, ie, js, je, nz, c, m, n, nw, nwx
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    CUF_DO3
    do k = 0, nz
      do j = js, je
        do i = is, ie
          if (k >= 1) then
            VX(i, j, k) = 0.0_rp; VY(i, j, k) = 0.0_rp
            HZS(i, j, k) = Hz(i, j, k)
          endif
          WA(i, j, k) = 0.0_rp; KA(i, j, k) = 0.0_rp
          if (k == 0) RAINA(i, j) = 0.0_rp
        enddo
      enddo
    enddo
    n = ntp; nw = max(nwet, 1_ip)
    CUF_DO2
    do c = 1, nw
      do m = 1, n
        SFA(m, c) = 0.0_rp
      enddo
    enddo
    if (nwf > 0) then
      nw = nwf
      CUF_DO2
      do c = 1, nw
        do m = 1, n
          RUA(m, c) = 0.0_rp
        enddo
      enddo
      if (size(DRV, 2) == nwf) then
        CUF_DO2
        do c = 1, nw
          do m = 1, n
            DRV(m, c) = 0.0_rp
          enddo
        enddo
      endif
    endif
    if (nqw > 0) then
      nw = nqw
      CUF_DO3
      do k = 1, nz
        do c = 1, nw
          do m = 1, n
            CPA(m, c, k) = 0.0_rp
            if (m == 1) CPV(1, c, k) = 0.0_rp
          enddo
        enddo
      enddo
    endif
    if (c3a > 0) then
      nw = decomp%nx; nwx = decomp%ny
      CUF_DO3
      do k = 1, nz
        do j = 1, nwx
          do i = 1, nw
            C3(i, j, k, 1) = 0.0_rp; C3(i, j, k, 2) = 0.0_rp
          enddo
        enddo
      enddo
    endif
  end subroutine reset_sums

end module bgc_subcycle_mod

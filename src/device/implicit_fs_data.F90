#include "mem_attr.inc"
#include "cuf_launch.inc"
#ifdef USE_CUDA
#define FSD device,
#else
#define FSD
#endif
module fs_data
  ! Implicit free surface (implicit_fs.F90): the shared state of the device
  ! modules (settings, per-face and per-cell fields, multigrid planes,
  ! statistics), the one-cell ring exchange, the deterministic sums and the
  ! cost timers.
  use kinds
  use decomp_mod, only: decomp
  use comm_mod, only: comm_ring_dev8, comm_ring_h8
  use fs_ipc, only: ipc_on, ipc_ring_fine
  use run_timers, only: wall_time
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  use iso_c_binding
  implicit none
  public

  ! Namelist &implicit_fs (FS_SPEC 6; same keys and defaults as the CPU)
  character(len=8) :: fs_mode = 'split'
  ! Defaults of ucla-roms implicit-fs-defaults (864969bd): two sub-steps at fs_rho_inf 0.5 with fs_pg_rho and
  ! fs_flather_zghost (the previous behaviour: fs_rho_inf = 0, fs_nsub = 1, both keys .false.)
  real(rp) :: fs_rho_inf = 0.5_rp        ! in [0, 1) (0: L-stable; 1, Crank-Nicolson, rejected)
  real(rp) :: fs_theta = 0.5_rp
  real(rp) :: fs_theta_div = 0.5_rp
  real(rp) :: fs_tol = 1.e-8_rp
  integer :: fs_maxit = 1000
  character(len=8) :: fs_precond = 'mg'
  integer :: fs_mg_levels = 0
  integer :: fs_mg_nu = 2
  character(len=8) :: fs_mg_smoother = 'rbgs'
  integer :: fs_mg_lag = 1
  logical :: fs_drag_implicit = .false.
  integer :: fs_warm = 1
  ! Fixed parts of the scheme: AB2 predictor transport, Flather flux
  ! with the generalized-alpha memory, the backward-Euler 'adv' tangential condition (inflow
  ! nudging rate times fs_tang_nudge), the momentum memory without the Coriolis estimate on U^n
  ! (fs_cor_metric: with the curvilinear metric terms of the stock rhs, .false. = a0e9e2be), the
  ! step's own volume sources (Ftil = Fbar = Q_r, Qbar = Q), a memory-less first step
  real(rp) :: fs_tang_nudge = 1.0_rp
  integer :: fs_verbose = 0            ! 1 FS_SOLVE line per step and FS_STAT per solve at the end;
                                       ! 2 + FS_VOL, FS_SRC, FS_WALL lines (FS_SRC also in fs_mode = 'split')
  integer :: fs_selftest = 0
  logical :: fs_cor_metric = .true.
  ! 8edf0a02 (FS_SPEC 9b item 8, 9b-D): time of the tidal potential in implicit mode, 'af' n+alpha_f =
  ! n+1/(1+fs_rho_inf) (default), 'mid' n+1/2 (22c05bc1); pot_phase is read by tide_io's set_tides
  ! (ptide at time + pot_phase dt; 0.5 in the split and without the group)
  character(len=8) :: fs_pot_time = 'af'
  real(rp) :: pot_phase = 0.5_rp
  ! 14233194 (FS_SPEC 5, 9b-E): coarse face factors of the multigrid, 'width' (default: (w_f(fi-1) +
  ! w_f(fi)) / (w_c(I-1) + w_c(I)), the aggregates' widths in fine cells; = 0.5 for full aggregates of
  ! equal cells) or 'half' (0.5, 6fc45b50). The CPU's MPI-layout keys fs_mg_gather, fs_mg_global_on
  ! and fs_exch_thin are read and not used: the GPU's layout is
  ! fs_gpu_gather's (one tile per device, device-held global levels, one-cell ring exchanges)
  character(len=8) :: fs_mg_faces = 'width'
  integer :: fs_mg_gather = 1024
  character(len=8) :: fs_mg_global_on = 'node'
  logical :: fs_exch_thin = .true.
  ! GPU-only keys: fs_gpu_gather > 0 gathers the multigrid hierarchy into
  ! one global grid at the first coarse level with at most fs_gpu_gather
  ! global cells (or earlier by the CPU's rule); 0: the CPU's rule alone
  ! (the CPU's level layout: the strict gate). Default 40000 (session 4
  ! timing: sargasso 2x2 gathers level 1, 192 x 192; Pac 8 x 1 level 3);
  ! fs_gpu_timers 1: FS_COST pieces with a device synchronisation at each
  ! piece boundary (timing runs only).
  ! fs_gpu_coarse 'device': the global levels on every GPU (fs_mgg), the
  ! coarsest's Cholesky solve on the host; 'host': all global levels on
  ! the host (fs_host), as the CPU. Same arithmetic.
  integer :: fs_gpu_gather = 40000
  integer :: fs_gpu_timers = 0
  character(len=8) :: fs_gpu_coarse = 'device'
  integer :: fs_gpu_graph = 0         ! 1: the device global levels' down and up legs as CUDA graphs
  ! fs_gpu_opt (free-surface optimization session): bit mask of the latency levers of the solve,
  ! none of which changes the arithmetic (each path bit-identical to the 0 path):
  !   1 the ring exchanges through mapped pinned host memory (the pack kernel writes the host
  !     send buffers, host MPI, the unpack kernel reads the host receive buffers) instead of
  !     GPU-aware MPI on device buffers
  !   2 the deterministic sums in one kernel (the two fixed-order passes, the second by the last
  !     block), the result written to mapped host memory (no copy)
  !   4 the coarsest level's vector in mapped host memory (no copies around the host solve)
  !   8 the multigrid gathers (vector each V-cycle, coefficients each rebuild) staged through
  !     pinned host memory with host MPI instead of GPU-aware MPI
  !  16 the host Cholesky factor and solve of the coarsest level in column order for the
  !     independent updates (the same operations in the same order for every entry), compiled at
  !     -O2 -Mnovect instead of -O1 (fs_chol, implicit_fs_chol.F90)
  !  32 the device global levels' down and up legs each in one cooperative kernel (fs_mgk)
  !  64 the red-black colour passes of the fine and distributed levels launched over their
  !     colour's cells only (half the threads)
  ! 128 the V-cycle recorded once into CUDA graphs, one per stretch between two host actions
  !     (ring MPI, gather, coarsest solve), and replayed: one graph launch per stretch instead
  !     of one launch per kernel (fs_mg vg_*; not with fs_gpu_timers 1)
  ! 256 the ring exchanges between ranks of one node by CUDA IPC, device-initiated (put and wait
  !     kernels, no host synchronisation or MPI per exchange; fs_ipc), when every rank's
  !     neighbours are on its node, otherwise the MPI rings; and the multigrid vector gather the
  !     same way when all compute ranks (at most 4) share one node
  ! Default 471 = all but 8 and 32, which were slower (sargasso and Pac FS window: 8 +0.2 / +0.15 ms,
  ! 32 +0.2 / +0.3 ms with 128 on: graph-launched kernels beat the cooperative kernels' grid syncs).
  ! fs_gpu_timers 2: the FS window (device synchronised at entry and exit of fs_step) only.
  integer :: fs_gpu_opt = 471
  ! fs_nsub (CPU b6325f06): implicit barotropic sub-steps of dt/fs_nsub per 3-D step (1..4); rufrc is
  ! held over the step, sub-steps after the first add the pressure gradient of their own sea-level
  ! change since n (weight beta1), the generalized-alpha rates are carried from sub-step to sub-step;
  ! DU_avg1 = U of the last sub-step, DU_avg2 = the mean flux of the sub-steps. The explicit forcing
  ! (tidal potential 'af') is taken at n + sft dt, sft = (alpha_f + (fs_nsub-1)/2)/fs_nsub
  ! (= alpha_f for one step), the boundary tidal harmonics at n + fs_bry_phase dt, fs_bry_phase =
  ! (fs_nsub+1)/(2 fs_nsub) (1 for one step). dts = dt/fs_nsub, the (sub-)step length.
  integer :: fs_nsub = 2
  real(rp) :: sft = 1.0_rp, fs_bry_phase = 1.0_rp, dts = 0.0_rp
  ! fs_pg_rho (CPU 7ffc3990): the implicit pressure-gradient operator carries the density weighting of
  ! the depth-integrated 3-D pressure gradient (rhoS, rhoA of rho_eos, as step2d's rubar). roms_gpu
  ! always computes rhoS and rhoA (its step2d always uses them, the CPU's VAR_RHO_2D), so the CPU's
  ! fallback for builds without VAR_RHO_2D (b582aa94) has no GPU counterpart.
  logical :: fs_pg_rho = .true.
  ! fs_flather_zghost (CPU 7be4fafe): open (Flather) faces take the sea level of the condition as the mean
  ! of the first interior and the boundary rho point, as the split's u2dbc (zx); the boundary value follows
  ! the radiation condition of the zeta boundary kernel and is eliminated: a diagonal Robin term with
  ! cb w1, w1 = (1 + 2 cx)/(2 (1 + cx)), and an explicit term w0 zeta_b(ks), w0 = 1/(2 (1 + cx)), cx = dts
  ! pm sqrt(g h) at the first interior point. .false.: the first interior value alone (the boundary-forced
  ! tide later than the split's by about dx/(2 sqrt(g h))). cbu, cbv hold cb; the products cb w1 are formed
  ! where they are used (the same rounded values as the CPU's stored cbu = cb w1).
  logical :: fs_flather_zghost = .true.
  namelist /implicit_fs/ fs_mode, fs_rho_inf, fs_theta, fs_theta_div, fs_tol, fs_maxit, fs_precond, &
       fs_mg_levels, fs_mg_nu, fs_mg_smoother, fs_mg_lag, fs_drag_implicit, &
       fs_warm, fs_verbose, fs_selftest, fs_tang_nudge, &
       fs_cor_metric, fs_pot_time, fs_nsub, fs_pg_rho, fs_flather_zghost, &
       fs_mg_gather, fs_mg_global_on, fs_mg_faces, fs_exch_thin, &
       fs_gpu_gather, fs_gpu_timers, fs_gpu_coarse, fs_gpu_graph, fs_gpu_opt

  logical :: fs_on = .false.               ! fs_mode = 'implicit'
  logical :: talk = .false., ready = .false., mg_on = .false., jac_sm = .false.
  logical :: ga_first = .true., dev_glob = .true.
  logical :: fs_srcdiag = .false.       ! fs_verbose >= 2: FS_SRC line per step (both fs_mode values)
  integer, parameter :: rst_format = 3  ! restart scalar 4: format version of the stored state (rates,
                                        ! delta, operators); a record of another version starts cold
  logical :: fs_capt = .false.          ! a CUDA graph capture is open on work_stream (no syncs)
  real(rp) :: beta1 = 1.0_rp, thp = 0.5_rp, thd = 0.5_rp, gam_a = 0.5_rp, ths = 0.5_rp

  ! per-face and per-cell fields (the CPU's names)
  real(rp), FSD allocatable, dimension(:,:) :: unu, unv, usu, usv, au, av, cu, cv, ftu, ftv, fbu, fbv, &
       u1u, u1v, cbu, cbv, wd, ar, rb, dinv, qs, act, bb, rr, zz, pd, qq, wk2, &
       du1o, dv1o, fdum, xru, xrv, cou, cov, fcr
  ! fs_nsub > 1: sums over the sub-steps of fbu, fbv (fsu, fsv) and of DU_avg2, DV_avg2 (asu, asv)
  real(rp), FSD allocatable, dimension(:,:) :: fsu, fsv, asu, asv
  ! the generalized-alpha rates and the PCG warm start delta: managed, as they are written
  ! to and read from the restart file (a0e9e2be)
  real(rp), WORK_ATTR allocatable, dimension(:,:) :: udu, udv, gru, grv, qd, xx
  integer(ip), FSD allocatable, dimension(:,:) :: rivu, rivv
  ! fs_mg_lag > 1 (restart, FS_SPEC 9b-C): the fine operator of the last coarse-operator build
  ! (mg_se0 = act (ar + rb), mg_scu, mg_scv = cu, cv; managed: in the restart file), the phase
  ! offset of the rebuild counter after a restart (rebuild when mod(n_solve + mg_ph0, fs_mg_lag)
  ! = 0) and mg_rpend: rebuild the coarse operators from them before the first solve
  real(rp), WORK_ATTR allocatable, dimension(:,:) :: mg_se0, mg_scu, mg_scv
  integer :: mg_ph0 = 0, mg_built = 0
  logical :: mg_rpend = .false.
  integer, parameter :: fs_nsc = 8

  ! deterministic sums: one partial per interior cell, two fixed-order passes
  integer, parameter :: FSR_NB = 128, FSR_NT = 256, FSR_MAXS = 8
  real(rp), FSD allocatable :: fs_part(:,:), fs_bpart(:,:)
  ! the sums' result: device view fs_dres, host view fs_hres (fs_gpu_opt 2: one mapped pinned
  ! buffer; otherwise a device array and a pinned copy)
  real(rp), FSD pointer, contiguous :: fs_dres(:) => null()
  real(rp), pointer, contiguous :: fs_hres(:) => null()
  real(rp), FSD allocatable, target :: fs_dres0(:)
#ifdef USE_CUDA
  real(rp), pinned, allocatable, target :: fs_hres0(:)
  integer, device, allocatable :: fsr_cnt(:)
#else
  real(rp), allocatable, target :: fs_hres0(:)
#endif
  logical :: o_ring = .false., o_red = .false., o_cvec = .false., o_gath = .false., o_chol = .false.
  logical :: o_fleg = .false., o_cc = .false., o_vgr = .false., o_ipc = .false.
  ! fs_gpu_opt 128: a V-cycle recording is open (the host actions go through vg_point_p, which
  ! closes the stretch's graph, launches it, does the action and opens the next stretch)
  logical :: vg_rec = .false.
  abstract interface
    subroutine vg_if(kind, p)
      integer, intent(in) :: kind, p(8)
    end subroutine vg_if
  end interface
  procedure(vg_if), pointer :: vg_point_p => null()
  real(8) :: tw_t0 = 0.0d0, tm_win = 0.0d0
  integer :: n_win = 0
  integer :: ncell = 0

  ! one-cell ring exchange buffers (up to 6 fields)
  integer, parameter :: RG_NF = 6
  ! device views (used by the pack and unpack kernels); fs_gpu_opt 1: of mapped pinned host
  ! memory whose host views rgh_* go to host MPI; otherwise of the device block rg_blk
  real(rp), FSD pointer, contiguous :: rg_sw(:) => null(), rg_se(:) => null(), rg_ss(:) => null(), &
       rg_sn(:) => null(), rg_rw(:) => null(), rg_re(:) => null(), rg_rs(:) => null(), rg_rn(:) => null()
  real(rp), pointer, contiguous :: rgh_sw(:) => null(), rgh_se(:) => null(), rgh_ss(:) => null(), &
       rgh_sn(:) => null(), rgh_rw(:) => null(), rgh_re(:) => null(), rgh_rs(:) => null(), rgh_rn(:) => null()
  real(rp), FSD allocatable, target :: rg_blk(:)

  ! multigrid: distributed levels 1..mg_ld in planes (0:mgx+1, 0:mgy+1, l)
  integer, parameter :: mg_maxl = 20
  integer :: mg_ld = 0, mg_ng = 0, mgx = 0, mgy = 0
  integer :: mg_nx(0:mg_maxl) = 0, mg_ny(0:mg_maxl) = 0
  real(rp), FSD allocatable, dimension(:,:,:) :: mx_, mb_, me_, mw_, ms_, md_, mt_
  ! coarse face factors of the distributed levels (fs_mg_faces; (ic, l), ic = 1..mg_nx(l)+1)
  real(rp), FSD allocatable :: mfx(:,:), mfy(:,:)
  real(rp), FSD allocatable :: db_v(:), db_c(:)
#ifdef USE_CUDA
  real(rp), pinned, allocatable :: hb_b(:), hb_x(:), hb_c(:)
#else
  real(rp), allocatable :: hb_b(:), hb_x(:), hb_c(:)
#endif
  real(rp), parameter :: om = 0.8_rp

  ! statistics and cost pieces (FS_STAT, FS_COST)
  integer :: n_solve = 0, n_its = 0, n_unconv = 0, its_last = 0, n_step = 0
  ! fs_nsub > 1: the step's iteration sum and maximum, worst residual and max |delta| of its sub-steps
  integer :: its_sum = 0, its_max = 0
  real(rp) :: res_max = 0.0_rp, dz_max = 0.0_rp
  integer(8) :: n_ring = 0, n_red = 0, n_gath = 0, n_vc = 0
  real(rp) :: res_last = 0.0_rp
  integer, parameter :: ntm = 16
  real(8) :: tm(ntm) = 0.0d0, tm_t0 = 0.0d0
  character(len=12), parameter :: tm_name(ntm) = [character(len=12) :: 'setup', 'exch_coef', &
       'pcg_start', 'pcg_op', 'pcg_pc', 'pcg_exch', 'pcg_allred', 'pcg_vec', 'finish', &
       'exch_fin', 'bc_uv', 'set_depth', 'mg_smooth', 'mg_exch', 'mg_transfer', 'mg_coarse']
  integer, parameter :: T_SETUP = 1, T_EXCO = 2, T_START = 3, T_OP = 4, T_PC = 5, T_PEX = 6, T_RED = 7, &
       T_VEC = 8, T_FIN = 9, T_EXFIN = 10, T_BCUV = 11, T_SETD = 12, T_MSM = 13, T_MEX = 14, &
       T_MTR = 15, T_MCO = 16
  real(8) :: tm_mgsetup = 0.0d0

contains

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function fs_tnew(tnu, vbn, v1, ca, cy, gm, gp, vext) result(fs_tn)
    ! Reference copy of the CPU's tnew (not called: the tangential kernel inlines it, because
    ! host scalars passed by reference from a CUF kernel were read wrongly on the device in
    ! v2a; kept because removing it re-triggers the nvfortran -g fort2 crash in this build).
    ! One tangential boundary face (the 'adv' form): vbn the ghost value at n, v1 the first
    ! interior value at n+1, ca the outward-normal Courant number, cy the along-edge one, gm/gp
    ! the along-edge differences at n, vext the external value (roms_gpu always has the
    ! boundary data), tnu fs_tang_nudge
    real(rp), intent(in) :: tnu, vbn, v1, ca, cy, gm, gp, vext
    real(rp) :: fs_tn
    real(rp) :: c, r
    c = min(1.0_rp, max(-1.0_rp, cy))
    r = vbn - max(c, 0.0_rp) * gm - min(c, 0.0_rp) * gp
    if (ca > 0.0_rp) then
      r = (r + ca * v1) / (1.0_rp + ca)
    else
      c = -ca * tnu
      r = (r + c * vext) / (1.0_rp + c)
    endif
    fs_tn = r
  end function fs_tnew


  subroutine fs_opt_flags()
    o_ring = iand(fs_gpu_opt, 1) /= 0; o_red = iand(fs_gpu_opt, 2) /= 0; o_cvec = iand(fs_gpu_opt, 4) /= 0
    o_gath = iand(fs_gpu_opt, 8) /= 0; o_chol = iand(fs_gpu_opt, 16) /= 0
    o_fleg = iand(fs_gpu_opt, 32) /= 0; o_cc = iand(fs_gpu_opt, 64) /= 0; o_vgr = iand(fs_gpu_opt, 128) /= 0
    o_ipc = iand(fs_gpu_opt, 256) /= 0
    if (fs_gpu_timers == 1) o_vgr = .false.
#ifndef USE_CUDA
    o_ring = .false.; o_red = .false.; o_cvec = .false.; o_gath = .false.; o_fleg = .false.; o_vgr = .false.
    o_ipc = .false.
#endif
  end subroutine fs_opt_flags

  subroutine map_alloc(n, hp, dp)
    ! n doubles of mapped pinned host memory: host view hp, device view dp (zeroed). The caller
    ! passes local pointers and pointer-assigns its module pointers to them: a module device
    ! pointer associated here through the dummy keeps a stale device-side descriptor for the CUF
    ! kernels that name it (illegal address; the first build of this lever)
    integer, intent(in) :: n
    real(rp), pointer, contiguous :: hp(:)
    real(rp), FSD pointer, contiguous :: dp(:)
#ifdef USE_CUDA
    type(c_ptr) :: cp
    type(c_devptr) :: dv
    integer :: ierr
    ierr = cudaHostAlloc(cp, int(max(n, 1), 8) * 8_8, cudaHostAllocMapped)
    if (ierr /= 0) error stop 'implicit_fs: cudaHostAlloc (mapped)'
    call c_f_pointer(cp, hp, [max(n, 1)])
    hp = 0.0_rp
    ierr = cudaHostGetDevicePointer(dv, cp, 0)
    if (ierr /= 0) error stop 'implicit_fs: cudaHostGetDevicePointer'
    call c_f_pointer(dv, dp, [max(n, 1)])
#else
    allocate(hp(max(n, 1))); hp = 0.0_rp; dp => hp
#endif
  end subroutine map_alloc

  subroutine fs_alloc_bufs(m)
    ! the ring buffers (m values per direction and kind) and the sums' result
    integer, intent(in) :: m
    real(rp), FSD pointer, contiguous :: d(:)
    real(rp), pointer, contiguous :: h(:)
    if (o_ring) then
      call map_alloc(8 * m, h, d)
      rgh_sw => h(1:m); rgh_se => h(m + 1:2 * m); rgh_ss => h(2 * m + 1:3 * m); rgh_sn => h(3 * m + 1:4 * m)
      rgh_rw => h(4 * m + 1:5 * m); rgh_re => h(5 * m + 1:6 * m); rgh_rs => h(6 * m + 1:7 * m)
      rgh_rn => h(7 * m + 1:8 * m)
    else
      allocate(rg_blk(8 * m)); rg_blk = 0.0_rp
      d => rg_blk
    endif
    rg_sw => d(1:m); rg_se => d(m + 1:2 * m); rg_ss => d(2 * m + 1:3 * m); rg_sn => d(3 * m + 1:4 * m)
    rg_rw => d(4 * m + 1:5 * m); rg_re => d(5 * m + 1:6 * m); rg_rs => d(6 * m + 1:7 * m); rg_rn => d(7 * m + 1:8 * m)
    if (o_red) then
      call map_alloc(FSR_MAXS, h, d); fs_hres => h; fs_dres => d
    else
      allocate(fs_dres0(FSR_MAXS), fs_hres0(FSR_MAXS)); fs_dres0 = 0; fs_hres0 = 0
      fs_dres => fs_dres0; fs_hres => fs_hres0
    endif
#ifdef USE_CUDA
    allocate(fsr_cnt(1)); fsr_cnt = 0
#endif
  end subroutine fs_alloc_bufs

  subroutine fs_alloc_rates()
    ! the rates and delta (zero); before fs_init when a restart file supplies them
    integer(ip) :: is, ie, js, je
    if (allocated(udu)) return
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    allocate(udu(is:ie, js:je), udv(is:ie, js:je), gru(is:ie, js:je), grv(is:ie, js:je), qd(is:ie, js:je), &
             xx(is:ie, js:je))
    udu = 0; udv = 0; gru = 0; grv = 0; qd = 0; xx = 0
  end subroutine fs_alloc_rates

  subroutine ring(nf, a1, a2, a3, a4, a5, a6, fine)
    ! One-cell halo of nf (<= 6) fine-level fields in one round (four
    ! neighbours at once, GPU-aware MPI; no corners, which the 5-point
    ! stencils and the face loops never read). The east-west strips carry
    ! the physical-edge ghost rows (j = 0 at a south edge, ny+1 at a north
    ! edge), the north-south strips the ghost columns, as halo_mod's
    ! exchange bounds; a physical edge (NBR_NONE) keeps its ghosts; a
    ! periodic self-neighbour wraps.
    integer, intent(in) :: nf
    real(rp), FSD intent(inout) :: a1(decomp%is:decomp%ie, decomp%js:decomp%je), &
         a2(decomp%is:decomp%ie, decomp%js:decomp%je), a3(decomp%is:decomp%ie, decomp%js:decomp%je), &
         a4(decomp%is:decomp%ie, decomp%js:decomp%je), a5(decomp%is:decomp%ie, decomp%js:decomp%je), &
         a6(decomp%is:decomp%ie, decomp%js:decomp%je)
    logical, intent(in) :: fine
    integer(ip) :: nx, ny, j0, j1, i0, i1, ny2, nx2, m, mx, f, w, e, s, n, nb(4), jj, ii, nff
    integer :: ierr
    real(rp) :: v
    nx = decomp%nx; ny = decomp%ny
    nb = [decomp%nbr_w, decomp%nbr_e, decomp%nbr_s, decomp%nbr_n]
    if (all(nb == NBR_NONE)) return
    if (ipc_on) then
      ! fs_gpu_opt 256: device-initiated exchange with the neighbours on this node (fs_ipc)
      call ipc_ring_fine(nf, a1, a2, a3, a4, a5, a6)
      n_ring = n_ring + 1
      return
    endif
    j0 = merge(0, 1, decomp%edge_s); j1 = merge(ny + 1, ny, decomp%edge_n)
    i0 = merge(0, 1, decomp%edge_w); i1 = merge(nx + 1, nx, decomp%edge_e)
    ny2 = j1 - j0 + 1; nx2 = i1 - i0 + 1; mx = max(ny2, nx2); nff = nf
    CUF_DO2
    do f = 1, nff
      do m = 1, mx
        if (m <= ny2) then
          jj = j0 + m - 1
          select case (f)
          case (1); rg_sw(m) = a1(1, jj); rg_se(m) = a1(nx, jj)
          case (2); rg_sw(m + ny2) = a2(1, jj); rg_se(m + ny2) = a2(nx, jj)
          case (3); rg_sw(m + 2 * ny2) = a3(1, jj); rg_se(m + 2 * ny2) = a3(nx, jj)
          case (4); rg_sw(m + 3 * ny2) = a4(1, jj); rg_se(m + 3 * ny2) = a4(nx, jj)
          case (5); rg_sw(m + 4 * ny2) = a5(1, jj); rg_se(m + 4 * ny2) = a5(nx, jj)
          case (6); rg_sw(m + 5 * ny2) = a6(1, jj); rg_se(m + 5 * ny2) = a6(nx, jj)
          end select
        endif
        if (m <= nx2) then
          ii = i0 + m - 1
          select case (f)
          case (1); rg_ss(m) = a1(ii, 1); rg_sn(m) = a1(ii, ny)
          case (2); rg_ss(m + nx2) = a2(ii, 1); rg_sn(m + nx2) = a2(ii, ny)
          case (3); rg_ss(m + 2 * nx2) = a3(ii, 1); rg_sn(m + 2 * nx2) = a3(ii, ny)
          case (4); rg_ss(m + 3 * nx2) = a4(ii, 1); rg_sn(m + 3 * nx2) = a4(ii, ny)
          case (5); rg_ss(m + 4 * nx2) = a5(ii, 1); rg_sn(m + 4 * nx2) = a5(ii, ny)
          case (6); rg_ss(m + 5 * nx2) = a6(ii, 1); rg_sn(m + 5 * nx2) = a6(ii, ny)
          end select
        endif
      enddo
    enddo
    call ring_mpi(nb, int(nf * ny2), int(nf * nx2))
    w = nb(1); e = nb(2); s = nb(3); n = nb(4)
    CUF_DO2
    do f = 1, nff
      do m = 1, mx
        if (m <= ny2) then
          jj = j0 + m - 1
          if (w /= NBR_NONE) then
            if (w == NBR_SELF) then
              v = rg_se(m + (f - 1) * ny2)
            else
              v = rg_rw(m + (f - 1) * ny2)
            endif
            select case (f)
            case (1); a1(0, jj) = v
            case (2); a2(0, jj) = v
            case (3); a3(0, jj) = v
            case (4); a4(0, jj) = v
            case (5); a5(0, jj) = v
            case (6); a6(0, jj) = v
            end select
          endif
          if (e /= NBR_NONE) then
            if (e == NBR_SELF) then
              v = rg_sw(m + (f - 1) * ny2)
            else
              v = rg_re(m + (f - 1) * ny2)
            endif
            select case (f)
            case (1); a1(nx + 1, jj) = v
            case (2); a2(nx + 1, jj) = v
            case (3); a3(nx + 1, jj) = v
            case (4); a4(nx + 1, jj) = v
            case (5); a5(nx + 1, jj) = v
            case (6); a6(nx + 1, jj) = v
            end select
          endif
        endif
        if (m <= nx2) then
          ii = i0 + m - 1
          if (s /= NBR_NONE) then
            if (s == NBR_SELF) then
              v = rg_sn(m + (f - 1) * nx2)
            else
              v = rg_rs(m + (f - 1) * nx2)
            endif
            select case (f)
            case (1); a1(ii, 0) = v
            case (2); a2(ii, 0) = v
            case (3); a3(ii, 0) = v
            case (4); a4(ii, 0) = v
            case (5); a5(ii, 0) = v
            case (6); a6(ii, 0) = v
            end select
          endif
          if (n /= NBR_NONE) then
            if (n == NBR_SELF) then
              v = rg_ss(m + (f - 1) * nx2)
            else
              v = rg_rn(m + (f - 1) * nx2)
            endif
            select case (f)
            case (1); a1(ii, ny + 1) = v
            case (2); a2(ii, ny + 1) = v
            case (3); a3(ii, ny + 1) = v
            case (4); a4(ii, ny + 1) = v
            case (5); a5(ii, ny + 1) = v
            case (6); a6(ii, ny + 1) = v
            end select
          endif
        endif
      enddo
    enddo
  end subroutine ring

  subroutine ring_slot(a, k)
    ! ring of a(:,:,k) of a 4-level 2-D field (zeta(knew))
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip), intent(in) :: k
    call ring(1, a(:, :, k), fdum, fdum, fdum, fdum, fdum, .true.)
  end subroutine ring_slot

  subroutine ring_mpi(nb, nw, ns)
    ! the MPI part of a ring exchange (after the pack kernel)
    integer(ip), intent(in) :: nb(4)
    integer, intent(in) :: nw, ns
    if (.not. any(nb >= 0)) return
    if (vg_rec) then
      call vg_point_p(1, [int(nb(1)), int(nb(2)), int(nb(3)), int(nb(4)), nw, ns, 0, 0])
    else
      call ring_mpi_act(nb, nw, ns)
    endif
  end subroutine ring_mpi

  subroutine ring_mpi_act(nb, nw, ns)
    integer(ip), intent(in) :: nb(4)
    integer, intent(in) :: nw, ns
    integer :: ierr
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'implicit_fs: ring stream sync'
#endif
    if (o_ring) then
      call comm_ring_h8(nb, nw, ns, rgh_sw, rgh_se, rgh_ss, rgh_sn, rgh_rw, rgh_re, rgh_rs, rgh_rn)
    else
      call comm_ring_dev8(nb, nw, ns, rg_sw, rg_se, rg_ss, rg_sn, rg_rw, rg_re, rg_rs, rg_rn)
    endif
    n_ring = n_ring + 1
  end subroutine ring_mpi_act

  subroutine fs_sum(m, out)
    ! out(1:m): the sums of fs_part(:, 1:m) over the tile, fixed order
    integer, intent(in) :: m
    real(rp), intent(out) :: out(:)
    integer :: s, chunk
#ifdef USE_CUDA
    integer :: ierr
    chunk = (ncell + FSR_NB - 1) / FSR_NB
    if (o_red) then
      call fsr_k12<<<FSR_NB, FSR_NT, 0, work_stream>>>(ncell, m, chunk, fs_part, fs_bpart, fs_dres, fsr_cnt)
    else
      call fsr_k1<<<FSR_NB, FSR_NT, 0, work_stream>>>(ncell, m, chunk, fs_part, fs_bpart)
      call fsr_k2<<<1, FSR_NB, 0, work_stream>>>(m, fs_bpart, fs_dres)
      ierr = cudaMemcpyAsync(fs_hres, fs_dres, m, cudaMemcpyDeviceToHost, work_stream)
    endif
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'implicit_fs: fs_sum'
    out(1:m) = fs_hres(1:m)
#else
    do s = 1, m
      out(s) = sum(fs_part(1:ncell, s))
    enddo
#endif
    n_red = n_red + 1
  end subroutine fs_sum

  subroutine zero2(a)
    real(rp), FSD intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j
    CUF_DO2
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        a(i, j) = 0.0_rp
      enddo
    enddo
  end subroutine zero2

  subroutine tq0()
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (fs_gpu_timers /= 1 .or. fs_capt) return
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
#endif
    tm_t0 = wall_time()
  end subroutine tq0

  subroutine tq(id)
    integer, intent(in) :: id
    real(8) :: t
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (fs_gpu_timers /= 1 .or. fs_capt) return
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
#endif
    t = wall_time()
    tm(id) = tm(id) + (t - tm_t0)
    tm_t0 = t
  end subroutine tq

  subroutine tw(e)
    ! fs_gpu_timers 2: the FS window, device synchronised at entry (e = 0) and exit (e = 1) of
    ! fs_step; steps after the first 10 of the run are summed
    integer, intent(in) :: e
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (fs_gpu_timers /= 2) return
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
#endif
    if (e == 0) then
      tw_t0 = wall_time()
    else
      n_win = n_win + 1
      if (n_win > 10) tm_win = tm_win + (wall_time() - tw_t0)
    endif
  end subroutine tw

#ifdef USE_CUDA
  attributes(global) subroutine fsr_k1(n, m, chunk, part, bpart)
    integer, value :: n, m, chunk
    real(8), device, intent(in) :: part(n, FSR_MAXS)
    real(8), device :: bpart(FSR_NB, FSR_MAXS)
    real(8), shared :: sh(FSR_NT)
    integer :: t, b, s, c, c0, c1, w
    real(8) :: a
    t = threadIdx%x; b = blockIdx%x
    c0 = (b - 1) * chunk + 1; c1 = min(n, b * chunk)
    do s = 1, m
      a = 0.0d0
      do c = c0 + t - 1, c1, FSR_NT
        a = a + part(c, s)
      enddo
      sh(t) = a
      call syncthreads()
      w = FSR_NT / 2
      do while (w >= 1)
        if (t <= w) sh(t) = sh(t) + sh(t + w)
        call syncthreads()
        w = w / 2
      enddo
      if (t == 1) bpart(b, s) = sh(1)
      call syncthreads()
    enddo
  end subroutine fsr_k1

  attributes(global) subroutine fsr_k12(n, m, chunk, part, bpart, res, cnt)
    ! fsr_k1, then the last block to finish does fsr_k2 (same partials, same tree, same order)
    integer, value :: n, m, chunk
    real(8), device, intent(in) :: part(n, FSR_MAXS)
    real(8), device, volatile :: bpart(FSR_NB, FSR_MAXS)
    real(8), device :: res(FSR_MAXS)
    integer, device :: cnt(1)
    real(8), shared :: sh(FSR_NT)
    integer, shared :: last
    integer :: t, b, s, c, c0, c1, w, old
    real(8) :: a
    t = threadIdx%x; b = blockIdx%x
    c0 = (b - 1) * chunk + 1; c1 = min(n, b * chunk)
    do s = 1, m
      a = 0.0d0
      do c = c0 + t - 1, c1, FSR_NT
        a = a + part(c, s)
      enddo
      sh(t) = a
      call syncthreads()
      w = FSR_NT / 2
      do while (w >= 1)
        if (t <= w) sh(t) = sh(t) + sh(t + w)
        call syncthreads()
        w = w / 2
      enddo
      if (t == 1) bpart(b, s) = sh(1)
      call syncthreads()
    enddo
    if (t == 1) then
      call threadfence()
      old = atomicadd(cnt(1), 1)
      last = 0
      if (old == FSR_NB - 1) last = 1
    endif
    call syncthreads()
    if (last == 0) return
    call threadfence()
    do s = 1, m
      if (t <= FSR_NB) sh(t) = bpart(t, s)
      call syncthreads()
      w = FSR_NB / 2
      do while (w >= 1)
        if (t <= w) sh(t) = sh(t) + sh(t + w)
        call syncthreads()
        w = w / 2
      enddo
      if (t == 1) res(s) = sh(1)
      call syncthreads()
    enddo
    if (t == 1) cnt(1) = 0
  end subroutine fsr_k12

  attributes(global) subroutine fsr_k2(m, bpart, res)
    integer, value :: m
    real(8), device, intent(in) :: bpart(FSR_NB, FSR_MAXS)
    real(8), device :: res(FSR_MAXS)
    real(8), shared :: sh(FSR_NB)
    integer :: t, s, w
    t = threadIdx%x
    do s = 1, m
      sh(t) = bpart(t, s)
      call syncthreads()
      w = FSR_NB / 2
      do while (w >= 1)
        if (t <= w) sh(t) = sh(t) + sh(t + w)
        call syncthreads()
        w = w / 2
      enddo
      if (t == 1) res(s) = sh(1)
      call syncthreads()
    enddo
  end subroutine fsr_k2
#endif

end module fs_data

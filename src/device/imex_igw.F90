#include "mem_attr.inc"
#include "cuf_launch.inc"
module imex_igw
  ! Implicit-explicit (IMEX) treatment of the internal-wave terms in the
  ! LF-AM3 baroclinic step: the GPU port of UCLA ROMS's IMEX_IGW (branch
  ! imex-modal, commit 75d3b58b, src/imex_igw.F90, with its 'modal' open boundary (imex_obm.inc); design
  ! and history in /glade/work/bachman/imex_internal_waves STATUS.md, IMPLEMENTATION_SPEC.md,
  ! IMEX_KEY_TRIM.md, GPU_IMEX_MODAL_PORT.md). The CPU switches it on with the cpp key
  ! IMEX_IGW; here it is on when the namelist has an &imex_igw_settings
  ! group (same keys and defaults as the CPU). Without the group nothing in
  ! this module runs and the model is the stock one.
  !
  ! What the scheme adds to the step (CPU routine names in brackets):
  ! - PG stores at levels n-1, n and * and the weighted explicit PG
  !   combination at the end of prsgrd [imex_pg_store_combine];
  ! - pre_step3d in two passes: the momentum predictor, its elliptic
  !   correction [imex_solve_stage(pred)] and the predictor tracer
  !   transport Flx_pred [imex_predictor_transport] before the tracer
  !   predictor (pre_step3d_mod's pre_step3d_imex);
  ! - the corrector's elliptic correction after step3d_uv1
  !   [imex_solve_stage(corr)];
  ! - step3d_uv2's tracer transport with the IMEX weights (wflx_corr, u_nm1);
  ! - t3dmix from t^n;
  ! - lateral mixing as a split step from the corrected state (imex_mix_split,
  !   visc3d.F90);
  ! - the end-of-step shift of the stored levels [imex_shift_levels].
  ! Each stage solve: per-step coefficients [imex_step_setup, imex_adj_setup]
  ! from a reference buoyancy b_ref [imex_bref_build] with the 'coef' N2
  ! floor [imex_cz_split] and the taper [imex_taper], the stage rhs
  ! [imex_rhs_transport, imex_stage_increment_coef], the 'eta' system
  ! (N2 + c D_b D_b^T) eta = r0 by Jacobi PCG, and the correction
  ! -theta dt D_b^T eta [imex_correct_fused_coef].
  !
  ! Supported namelist values: the defaults, plus imex_solver_enabled,
  ! imex_allow_explicit (with the weights it allows), the weights, the
  ! tolerances, imex_maxit, imex_verbose 0/1/2, imex_mix_split 0/1,
  ! imex_n2_ref, imex_n2_taper, imex_bref_hours, imex_warm 0/1,
  ! imex_setup_lag 0/1, imex_rst_state 0/1, imex_obc_mode 'fixed'/'modal'/'auto'
  ! with imex_obc_nmodes, imex_obc_q0, imex_obc_tref, imex_obc_cmin,
  ! imex_obc_mprint ('modal' needs the fused solve with imex_gpu_stage and
  ! imex_gpu_sweep on; obm_decide), and the solver paths
  ! imex_pcg_start 0 (the CPU's imex_pcg_warm/imex_pcg) or imex_pcg_start 2/3
  ! with imex_pcg_fused 1 (imex_pcgf8/4, storage imex_solver_prec 8/4;
  ! imex_pf.inc). imex_am3_crv other than 1/6, imex_selftest 1,
  ! imex_stage_fused 0, imex_redu 1 and imex_pcg_start 1 are CPU-only and stop
  ! the run at start-up (check_settings); the keys removed from the CPU on
  ! 2026-11-13 stop it at the read (removed_keys_set).
  !
  ! Defaults (CPU 203a1a7a): imex_setup_lag 1 (the operator - coefficients,
  ! Jacobi diagonal, Hz and its face sums - frozen between b_ref rebuilds,
  ! every imex_bref_hours = 24 h; the rhs projection and the correction use
  ! the current Hz), imex_pcg_start 3 (one-pass start-up, M x0 of the
  ! previous solve reused while the operator is unchanged), imex_pcg_fused 1
  ! (two fused sweeps in the W-scaled form), imex_solver_prec 4 (FP32
  ! storage of the solver's coefficients and p, q, z), imex_tol_corr 1e-3.
  ! GPU-only defaults (session 3): imex_gpu_redu 1 (deterministic sums),
  ! imex_gpu_stage 1, imex_gpu_sweep 1 (same arithmetic, fewer passes, land
  ! skipped).
  !
  ! Arrays are local tile + halo (decomp%is:ie, js:je) like gpu_state;
  ! the CPU's ib0:ib1, jb0:jb1 are 1:nx, 1:ny here, and its w_edge... are
  ! decomp%edge_w....
  use kinds
  use decomp_mod, only: decomp
  use gpu_state, only: u, v, t, Hz, z_r, z_w, FlxU, FlxV, We, Wi, pm, pn, dn_u, dm_v, dm_u, dn_v, &
                       rmask, umask, vmask, rho1, ru, rv, DU_avg1, DU_avg2, DU_avg_bak, &
                       DV_avg1, DV_avg2, DV_avg_bak, h, &
                       t_west, u_west, t_east, u_east, t_south, v_south, t_north, v_north
  use time_state, only: nstp, nnew, nrhs, iic, ntstart, ntimes, dt, is_first_time_step, ninfo_steps
  use formulas, only: g, rho0, AM3_crv, qp2, eos_r00, eos_r01, eos_r02, eos_r03, eos_r04, eos_r05, eos_r10, &
                      eos_r11, eos_r12, eos_r13, eos_r14, eos_rS0, eos_rS1, eos_rS2, eos_r20, eos_K00, eos_K01, &
                      eos_K02, eos_K03, eos_K04, eos_K10, eos_K11, eos_K12, eos_K13, eos_KS0, eos_KS1, eos_KS2, &
                      eos_Ttr, eos_Tsr
  use halo_mod, only: hg_begin, hg_add, hg_end, halo_fill_3d, halo_fill_3d_w, halo_fill_2d, &
                      F_U, F_V, F_FLXU, F_FLXV
  use comm_mod, only: comm_world_rank, comm_allreduce_sum, comm_allreduce_max, comm_allreduce_min, &
                      comm_gpu_direct, comm_ring_dev8
  use arch_mod, only: sync_arch
  use imex_red, only: red_ensure, red_sum, red_part
  use rivers_mod, only: river_on, rivers_pred_flx
#ifdef USE_CUDA
  use arch_mod, only: work_stream
  use cudafor, only: cudaStreamSynchronize, cudaDeviceSynchronize
#endif
  implicit none
  private

  public :: imex_read_namelist, imex_begin_step, imex_pg_store_combine, imex_solve_stage
  public :: imex_predictor_transport, imex_shift_levels, imex_tsmix_active
  ! Exact restart (CPU 22c05bc1 imex_rst_*, FS_SPEC 9b-C): his_io writes and case_io reads the
  ! state below under the CPU's names (imex_pgu ... imex_rst_scal); these routines decide and
  ! finish it, and the arrays are public for the two I/O paths
  public :: imex_rst_prepare, imex_rst_scal_vals, imex_rst_accept, imex_rst_loaded, imex_rst_has, imex_rst_alloc
  public :: imex_stage_on
  ! 'modal' open boundary (imex_obm.inc): the face writes after u3dbc/v3dbc of pre_step3d and
  ! step3d_uv2; the restart's reference m at the faces' ghost cells
  public :: imex_obc_faces, imex_obm_rst_may
  public :: obcm3
  public :: pgu, pgv, ipg_nm1, dpw, hs1, hs2, hsol, pf_mq, bref_sav, ad_n2raw, hz_op, dza, ad_n2o
  integer, parameter, public :: imex_nsc = 17

  logical, public, protected :: imex_on = .false.     ! &imex_igw_settings present

  integer(ip), parameter, public :: imex_stage_pred = 1, imex_stage_corr = 2

  ! Namelist IMEX_IGW_SETTINGS: the CPU's keys and defaults (UCLA ROMS imex-modal 75d3b58b). The
  ! template keys (the ones a user sets): imex_verbose, imex_bref_hours, imex_rst_state,
  ! imex_obc_mode, imex_obc_tref; the others are solver and development settings (NAMELIST_COMPAT)
  real(rp)    :: imex_theta_p = 0.5_rp, imex_phi = 0.25_rp, imex_theta_c = 0.6_rp
  real(rp)    :: imex_mu = 0.0_rp, imex_nu = 0.3_rp, imex_rho = 0.1_rp
  real(rp)    :: imex_am3_crv = 1.0_rp / 6.0_rp
  real(rp)    :: imex_tol_pred = 1.e-2_rp, imex_tol_corr = 1.e-3_rp
  integer     :: imex_maxit = 200
  ! 0 quiet, 1 solver statistics (IMEX line, IMEX_SOLVE), 2 + the true FP64 residual of every
  ! solve (IMEX_PCHK) and max |dp|; 0, 1 and 2 give the same results
  integer     :: imex_verbose = 0
  integer     :: imex_selftest = 0
  logical, public, protected :: imex_solver_enabled = .true.
  logical     :: imex_allow_explicit = .false.
  ! CPU 941f00d2: the smoothing's neighbour test with a tolerance relative
  ! to the cell's thickness (a neighbour column whose top or bottom z_r lies
  ! within imex_bref_ztol Hz of the cell's z_r is taken, its b linearly
  ! extrapolated over that distance); without it columns of equal geometry
  ! decided the test on the last bit of z_r (the seamount discrepancy of
  ! the 203a1a7a port). Module parameter, as on the CPU.
  real(rp), parameter :: imex_bref_ztol = 1.e-8_rp
  real(rp)    :: imex_n2_taper = 1.e-7_rp
  ! CPU 94bbf5bc: 1e-6 (was 3e-7; a deep mode grew in weakly stratified abyssal water, SARGASSO_COOLING.md)
  real(rp)    :: imex_n2_ref = 1.e-6_rp
  integer     :: imex_warm = 1
  real(rp)    :: imex_bref_hours   ! default 24 (CPU 5267429e; was 6): set in imex_read_namelist
  integer     :: imex_setup_lag = 1
  integer     :: imex_pcg_start = 3
  integer     :: imex_pcg_fused = 1
  integer     :: imex_solver_prec = 4
  ! CPU imex_stage_fused (the stage rhs and the correction fused with the solve): read so that a CPU
  ! namelist can be read; only 1 (the GPU's own fused stage is imex_gpu_stage)
  integer     :: imex_stage_fused = 1
  ! GPU only: deterministic sums (column partials + one fixed-order block,
  ! pinned asynchronous copy; imex_red) in the fused solve and the b_ref
  ! domain mean (0: CUF-kernel scalar reductions, not bitwise reproducible).
  ! Default 1 since session 3 (whole-step cost +0.0% sargasso, +0.4-0.6% Pac).
  ! Session 6: also the imex_pcg_start 0 solve (pcg: <x,Mx>, <r,z>, <r,r>, the
  ! warm start's norm), so that its exact restart is bit-exact
  integer     :: imex_gpu_redu = 1
  ! GPU only (session 3): the stage outside the PCG in fewer passes - the
  ! rhs (stage increment, memory term) formed in the solve's start-up
  ! kernel, the solution kept in place (no warm-start copies, no dps/dpw
  ! arrays), the last x update and the stored W M x in one kernel, x
  ! exchanged one cell wide (GPU-aware MPI), the u- and v-corrections with
  ! the face values computed once, the predictor transport without the u*,
  ! v* copies, act = 0 columns and closed faces skipped. Same arithmetic
  ! (bit-identical with imex_gpu_redu = 1). 0: the session-2 kernels.
  ! Default 1 since session 3.
  integer     :: imex_gpu_stage = 1
  ! GPU only (session 3): the PCG operator sweep restructured with the same
  ! arithmetic - land columns and closed faces skipped (their values are
  ! zero, or multiplied by zero), the column kernel in one downward pass,
  ! the face kernel storing g and its Hz_f-weighted column mean only, the
  ! face transports F formed in the cell kernel; sweep 2 skips land
  ! columns. 0: the session-2 kernels. Default 1 since session 3.
  integer     :: imex_gpu_sweep = 1
  ! CPU 73a63afb (FS_SPEC 3.8, 9b-H): imex_rst_state 1 (default) the IMEX state in the EXACT_RESTART
  ! files (one time-less copy, with the second record of each pair), 0 none (small files, the old
  ! inexact restart); imex_redu (CPU: rank-count-independent integer-digit sums) read, only 0 on
  ! the GPU (imex_gpu_redu = 1 is the GPU's deterministic sum within a device)
  integer     :: imex_rst_state = 1
  integer     :: imex_redu = 0
  ! CPU 1b3132bd, default 1 since 903b3aca (IMEX_SPONGE_INSTABILITY.md): lateral mixing as a split step
  ! from the corrected state, 1 visc3d's Laplacian from u^n plus the baroclinic part of this step's
  ! corrected increment (visc3d.F90); 0 as before (visc3d from u^n added after the implicit
  ! correction). With the additive form the explicit Laplacian of a sponge band made the IMEX step
  ! unstable at C_igw > ~2.
  integer, public, protected :: imex_mix_split = 1
  ! The CPU's 'modal' open boundary (keys and defaults as the CPU's): 'auto' selects 'modal' with an
  ! open edge and 3-D boundary data (obm_decide). imex_obc_mode, imex_obc_tref: template keys;
  ! imex_obc_nmodes, imex_obc_q0 (0 off, else 0.2-1), imex_obc_cmin, imex_obc_mprint (OBCM/OBCF
  ! lines): solver and development settings
  character(len=8) :: imex_obc_mode = 'auto'
  integer     :: imex_obc_nmodes = 3
  real(rp)    :: imex_obc_q0 = 0.0_rp
  real(rp)    :: imex_obc_tref = 259200.0_rp
  integer     :: imex_obc_mprint = 0
  real(rp)    :: imex_obc_cmin = 0.3_rp

  ! Former keys, fixed at their former defaults (removed from the namelist 2026-11-13, UCLA ROMS
  ! IMEX_KEY_TRIM): the N2 regularization of the diagonal [s-2], the b_ref smoothing passes along
  ! geopotentials
  real(rp), parameter :: imex_n2_reg = 1.e-12_rp
  integer, parameter :: imex_bref_smooth = 4
  ! keys removed from IMEX_IGW_SETTINGS (the CPU's list): a namelist that sets one stops at the read
  ! with a note listing them (removed_keys_set)
  integer, parameter :: n_removed = 34
  character(len=20), parameter :: removed_keys(n_removed) = [character(len=20) :: &
       'imex_force_mode', 'imex_n2_min', 'imex_bt_mode', 'imex_bt_passes', 'imex_s2d_check', 'imex_rhs_full', &
       'imex_probe_steps', 'imex_pcols', 'imex_win', 'imex_precond', 'imex_pcg_variant', 'imex_pcg_overlap', &
       'imex_adj_fused', 'imex_bref', 'imex_bref_floor', 'imex_bref_every', 'imex_bref_smooth', 'imex_n2_reg', &
       'imex_obc_rhs', 'imex_obc_op', 'imex_obc_zero', 'imex_obc_diag', 'imex_pred_btfix', 'imex_tsmix_n', &
       'imex_sponge_eq', 'imex_trim3d', 'imex_obc_eig', 'imex_obc_dcols', 'imex_obc_scrit', 'imex_obc_sband', &
       'imex_obc_fref', 'imex_obc_trelax', 'imex_obc_cslow', 'imex_obc_supercrit']

  namelist /imex_igw_settings/ &
       imex_theta_p, imex_phi, imex_theta_c, imex_mu, imex_nu, imex_rho, &
       imex_am3_crv, imex_tol_pred, imex_tol_corr, imex_maxit, &
       imex_verbose, imex_solver_enabled, imex_allow_explicit, imex_selftest, &
       imex_n2_ref, imex_n2_taper, imex_warm, &
       imex_bref_hours, imex_setup_lag, imex_pcg_start, imex_pcg_fused, imex_solver_prec, imex_stage_fused, &
       imex_redu, imex_rst_state, imex_mix_split, &
       imex_obc_mode, imex_obc_nmodes, imex_obc_q0, imex_obc_tref, &
       imex_obc_mprint, imex_obc_cmin, &
       imex_gpu_redu, imex_gpu_stage, imex_gpu_sweep

  ! Weights (set_derived_weights): regular step and first step of a run
  real(rp) :: wpg_pred(2), wflx_pred(3), wpg_corr(3), wrhs_corr(4)
  real(rp), public :: wflx_corr(4) = 0.0_rp     ! step3d_uv2: on (u^n+1, Flx*, u^n, u^n-1)
  real(rp) :: wpg_pred_reg(2), wflx_pred_reg(3), wpg_corr_reg(3), wflx_corr_reg(4)
  real(rp) :: wpg_pred_1st(2), wflx_pred_1st(3), wpg_corr_1st(3), wflx_corr_1st(4)

  ! PG stores (Hz-weighted, as ru, rv): slots ipg_n, ipg_nm1 (swapped by
  ! imex_shift_levels) and 3 = level *
  real(rp), WORK_ATTR allocatable :: pgu(:,:,:,:), pgv(:,:,:,:)
  integer(ip) :: ipg_n = 1, ipg_nm1 = 2
  ! velocity at n-1 [m/s]; btfix copies of u*, v*
  real(rp), WORK_ATTR public, allocatable :: u_nm1(:,:,:), v_nm1(:,:,:)
  real(rp), WORK_ATTR allocatable :: pb_u(:,:,:), pb_v(:,:,:)

  ! Geometry (static): face coefficients, active cells, cell area weight
  real(rp), WORK_ATTR allocatable :: cx(:,:), cy(:,:), cxo(:,:), cyo(:,:), act(:,:), darea(:,:), ad_act(:,:)
  ! Per step: 1/sum_k face thickness; 1/(z_r(k+1)-z_r(k-1)) (centred N2)
  real(rp), WORK_ATTR allocatable :: rshu(:,:), rshv(:,:), dza(:,:,:)
  ! Operator coefficients (CPU names): background b (b_ref), its last build,
  ! raw N2 (taper), N2 at cells (floored), kernel factor, en2, 1/N2, the
  ! previous step's N2 (memory-term rescale), face/interface values, the
  ! 'coef' interface values and diag(D D^T)
  real(rp), WORK_ATTR allocatable :: ad_bb(:,:,:), bref_sav(:,:,:), ad_n2raw(:,:,:), ad_n2t(:,:,:)
  real(rp), WORK_ATTR allocatable :: ad_n2k(:,:,:), ad_en2(:,:,:), ad_n2c(:,:,:), ad_n2o(:,:,:)
  real(rp), WORK_ATTR allocatable :: ad_cu(:,:,:), ad_cv(:,:,:), ad_cz(:,:,:), ad_czt(:,:,:), ad_czb(:,:,:)
  real(rp), WORK_ATTR allocatable :: ad_dd(:,:,:)
  ! b_ref build scratch
  real(rp), WORK_ATTR allocatable :: bw0(:,:,:), bw1(:,:,:), bw2(:,:,:), bw3(:,:,:)
  integer(ip), WORK_ATTR allocatable :: bword(:,:,:)
  ! Stage solves: rhs transports, r0, b, solution, memory terms
  real(rp), WORK_ATTR allocatable :: wk_fu(:,:,:), wk_fv(:,:,:), wk_r0(:,:,:), wk_b(:,:,:), wk_dp(:,:,:)
  real(rp), WORK_ATTR allocatable :: dps(:,:,:), dpw(:,:,:)
  ! PCG work: residual, preconditioned residual, search direction, M p; the
  ! operator's column (zc, db) and face (fu, fv) intermediates; the warm-start
  ! store of each stage's last solution
  real(rp), WORK_ATTR allocatable :: wk_r(:,:,:), wk_z(:,:,:), wk_p(:,:,:), wk_q(:,:,:)
  real(rp), WORK_ATTR allocatable :: op_zc(:,:,:), op_db(:,:,:), op_fu(:,:,:), op_fv(:,:,:)
  real(rp), WORK_ATTR allocatable :: hsol(:,:,:,:)
  ! imex_gpu_stage = 1: the predictor's and the corrector's solutions, kept
  ! in place between steps (the warm starts; hs1 = dps, hs1 + hs2 = dpw)
  real(rp), WORK_ATTR allocatable :: hs1(:,:,:), hs2(:,:,:)
  integer :: nh(2) = 0, n_maxit_hits = 0
  ! In effect (check_settings): the setup lag, the start-up mode, the fused
  ! solve and its storage kind, the IMEX_PCHK check (fused, verbose >= 1)
  logical :: lag_on = .false., pf_on = .false., pf_chk = .false.
  ! imex_gpu_redu = 1 in effect
  logical :: redu_on = .false.
  ! imex_gpu_stage = 1 in effect; fs_on: the solve being run forms the rhs
  ! in its start-up (fs_dtl = dts, fs_cb = cfbak of the predictor)
  logical :: stage_on = .false., fs_on = .false.
  ! imex_gpu_sweep = 1 in effect; the one-cell x exchange of imex_gpu_stage
  ! (GPU-aware MPI or no MPI neighbour)
  logical :: sweep_on = .false., xring_on = .false.
  ! imex_gpu_sweep: Hz_f-weighted column means of g at u- and v-faces
  real(rp), WORK_ATTR allocatable :: op_sgu(:,:), op_sgv(:,:)
  real(rp) :: fs_dtl = 0.0_rp, fs_cb = 0.0_rp
  ! imex_gpu_stage: at a b_ref rebuild the memory term dpw is rescaled to
  ! the new N2 (eta_setup1); dpw = (hs1 + hs2) n2o/n2t is then formed there
  ! and used by that step's predictor (fs_dpw), as the original's dpw
  logical :: fs_dpw = .false.
  ! a zero the compiler cannot fold: PF_START_RHS forms r0 as 0 + increment,
  ! as the original's zeroed wk_r0 += increment (same result for -0)
  real(rp) :: fs_zero = 0.0_rp
  ! the one-cell x exchange of imex_gpu_stage (pf_ring1_8): send and receive buffers
#ifdef USE_CUDA
  real(8), device, allocatable :: rg8_sw(:), rg8_se(:), rg8_ss(:), rg8_sn(:), rg8_rw(:), rg8_re(:), rg8_rs(:), rg8_rn(:)
#else
  real(8), allocatable :: rg8_sw(:), rg8_se(:), rg8_ss(:), rg8_sn(:), rg8_rw(:), rg8_re(:), rg8_rs(:), rg8_rn(:)
#endif
  integer :: pcg_start = 0, pf_prec = 8
  ! Coefficient set counter (bumped when the coefficients are rebuilt: every
  ! step at imex_setup_lag 0, at b_ref rebuilds at 1)
  integer :: coef_version = 0
  ! imex_setup_lag = 1: Hz (with halos) and rshu, rshv of the last rebuild
  real(rp), WORK_ATTR allocatable :: hz_op(:,:,:), rshu_op(:,:), rshv_op(:,:)
  ! Fused solve (imex_pf.inc): FP64 set rczk = -act/n2k (with halo), wr =
  ! darea hv/n2c, dinv = act/(W diag M) per stage; FP32 set (prec 4): the
  ! same plus czt, czb, cu, cv, hv in real(4), the FP64 projection sums of
  ! the real(4) face thicknesses, and z, p (two buffers), q; pf_mq: the
  ! stored W M x of each stage (slots 1, 2) and the start-up W M x0 (3);
  ! wk_p2: the second p buffer of the FP64 solve
  integer :: pf_cver8 = -1, pf_cver4 = -1, pf_dver8(2) = -1, pf_dver4(2) = -1, ps_mver(2) = -1
  real(rp) :: pf_dc28(2) = -1.0_rp, pf_dc24(2) = -1.0_rp, ps_mc2(2) = -1.0_rp
  real(rp), WORK_ATTR allocatable :: pf_rczk8(:,:,:), pf_wr8(:,:,:), pf_dinv8(:,:,:,:), pf_mq(:,:,:,:), wk_p2(:,:,:)
  real(4), WORK_ATTR allocatable :: pf_rczk4(:,:,:), pf_wr4(:,:,:), pf_czt4(:,:,:), pf_czb4(:,:,:)
  real(4), WORK_ATTR allocatable :: pf_cu4(:,:,:), pf_cv4(:,:,:), pf_hz4(:,:,:), pf_dinv4(:,:,:,:)
  real(4), WORK_ATTR allocatable :: pf_z4(:,:,:), pf_pa4(:,:,:), pf_pb4(:,:,:), pf_q4(:,:,:)
  real(rp), WORK_ATTR allocatable :: pf_ru4(:,:), pf_rv4(:,:)

  logical :: first_step = .true.
  ! Exact restart: rst_read the state was read from the initial file (the first step is an
  ! ordinary step), rst_pend rst_replay due at the first imex_step_setup, rst_on the replay's
  ! adj_setup (b_ref from bref_sav, no rescale), rst_mvalid the stored W M x of a stage applies
  ! to the replayed operator
  logical :: rst_read = .false., rst_pend = .false., rst_on = .false.
  integer :: rst_mvalid(2) = 0
  logical :: geometry_ready = .false., eta_old = .false., have_bref = .false.
  integer(ip) :: coef_iic = -huge(1_ip), bref_last = -100000000
  integer :: nbref_builds = 0
  ! Statistics since the last diagnostics line (CPU st_*)
  integer :: st_n(2) = 0, st_it(2) = 0, st_itmax(2) = 0, st_bad = 0
  real(rp) :: st_dp(2) = 0.0_rp, st_du(2) = 0.0_rp
  logical :: talk = .false.

  ! 'modal' open boundary (imex_obm.inc; CPU names ob_* where they exist). imex_obc_modal: in effect
  ! (obm_decide, first IMEX step); obm_gpu_ok: the solver path supports it (check_settings);
  ! rst_md_file: imex_rst_scal(16) of the restart whose state was read
  logical, public, protected :: imex_obc_modal = .false.
  logical :: obm_decided = .false., obm_gpu_ok = .false.
  integer :: rst_md_file = 1
  integer, parameter :: OBM_MMAX = 20, OBM_NT = 64, OBM_BT = 32   ! OBM_BT: threads per column (per-stage, per-sweep kernels)
  ! obm_modes_d (CPU obm_modes): multisection steps, Rayleigh-quotient passes, shifts per bracket and pass,
  ! bracket and near-degeneracy tolerances
  integer, parameter :: OBM_MBIS = 160, OBM_MRQ = 12, OBM_NSEC = 1
  real(rp), parameter :: OBM_TOLB = 1.e-2_rp, OBM_TOLG = 1.e-6_rp
  real(rp), parameter :: obm_tramp = 86400.0_rp
  ! packed open-edge face columns (obm_init) and the cells next to the faces
  integer :: ob_nc = 0, ob_np = 0, ob_m = 0, ob_nact = 0, ob_nbuild = 0, ob_mn = 0
  logical :: ob_edge(4) = .false., ob_mr_ok = .false., ob_rst = .false.
  integer(ip), WORK_ATTR allocatable :: ob_ce(:), ob_cl(:), ob_ci(:), ob_cj(:), ob_fi(:), ob_fj(:), ob_gi(:), ob_gj(:)
  integer(ip), WORK_ATTR allocatable :: ob_pi(:), ob_pj(:), ob_pc(:,:), ob_wi(:,:,:), ob_dg(:,:)
  ! the face columns' data (CPU ob_act, ob_yr, ob_w, ob_hn, ob_ub, ob_a, ob_c, ob_phi, ob_mr;
  ! ob_kdc the Jacobi column of each face, ob_dd the mismatch for the OBCM lines, ob_pd diagnostics
  ! partials), work arrays ob_gk, ob_sc, ob_z, ob_wr
  real(rp), WORK_ATTR allocatable :: ob_act(:), ob_yr(:), ob_w(:,:), ob_hn(:,:), ob_ub(:,:), ob_kdc(:,:)
  real(rp), WORK_ATTR allocatable :: ob_a(:,:), ob_c(:,:), ob_phi(:,:,:), ob_mr(:,:), ob_dd(:,:)
  real(rp), WORK_ATTR allocatable :: ob_pd(:,:), ob_gk(:,:), ob_sc(:,:,:), ob_z(:,:,:,:), ob_wr(:,:,:)
  ! the operator sweep's face coefficients (cxo, cyo, and dn_u, dm_v at the active open faces under
  ! 'modal'); the column of each open face (0: none), always allocated (rhs_transport reads them)
  real(rp), WORK_ATTR allocatable :: cxm(:,:), cym(:,:)
  integer(ip), WORK_ATTR allocatable :: obcol_u(:,:), obcol_v(:,:)
  ! restart: imex_obcm (the reference at the faces' ghost cells)
  real(rp), WORK_ATTR allocatable :: obcm3(:,:,:)

contains

  !======================================================================
  ! Namelist, settings
  !======================================================================
  subroutine imex_read_namelist(unit)
    ! &imex_igw_settings present: IMEX on. Absent: stock model. A group the
    ! namelist read rejects (unknown key) stops the run, as every group; when
    ! it sets keys removed on 2026-11-13 the message lists them (the CPU's)
    integer, intent(in) :: unit
    integer :: ios, p
    character(len=512) :: msg
    character(len=1024) :: found
    talk = (comm_world_rank() == 0)
    imex_bref_hours = 24.0_rp            ! the default (see the declaration)
    rewind(unit)
    msg = ''
    read(unit, nml=imex_igw_settings, iostat=ios, iomsg=msg)
    if (ios < 0) then
      imex_on = .false.
      return
    endif
    if (ios > 0) then
      do p = 1, len(msg)                   ! (the compiler pads iomsg with NUL characters)
        if (iachar(msg(p:p)) < 32) msg(p:p) = ' '
      enddo
      call removed_keys_set(unit, found)
      if (talk) then
        if (len_trim(found) > 0) then
          write(*,'(a)') 'ERROR: could not read the IMEX_IGW_SETTINGS section of the namelist file (' // &
               trim(msg) // '). It sets keys that were removed (2026-11-13; their former defaults are now' // &
               ' fixed): ' // trim(found) // '. Delete them from the namelist: NAMELIST_COMPAT (IMEX' // &
               ' section; the same line as in UCLA ROMS''s src/namelist.nml) gives a sed line that does it'
        else
          write(*,'(a)') 'ERROR: could not read the IMEX_IGW_SETTINGS section of the namelist file (' // &
               trim(msg) // ')'
        endif
      endif
      error stop 'read_namelist: unreadable namelist group (see above)'
    endif
    imex_on = .true.
    call check_settings()
    call set_derived_weights()
    if (talk) call print_settings()
  end subroutine imex_read_namelist

  subroutine removed_keys_set(unit, found)
    ! (CPU removed_keys_set) After a failed read: the removed keys (removed_keys) that the namelist
    ! file's IMEX_IGW_SETTINGS group sets, as a comma-separated list (empty if none). A key counts
    ! where it starts an entry (after blanks or a comma) and is followed by '=' or '(' (comments
    ! after '!' are skipped)
    integer, intent(in) :: unit
    character(len=*), intent(out) :: found
    character(len=1024) :: ln
    integer :: ios, k, p, q, c, n
    logical :: ingrp, last, hit(n_removed)
    found = ''; hit = .false.; ingrp = .false.
    rewind(unit)
    do
      read(unit, '(A)', iostat=ios) ln
      if (ios /= 0) exit
      c = index(ln, '!')
      if (c > 0) ln(c:) = ' '
      do p = 1, len_trim(ln)                    ! lower case
        c = iachar(ln(p:p))
        if (c >= iachar('A') .and. c <= iachar('Z')) ln(p:p) = achar(c + 32)
      enddo
      if (.not. ingrp) then                     ! the group starts (entries may follow on the line)
        p = index(ln, '&imex_igw_settings')
        if (p == 0) cycle
        ingrp = .true.
        ln = ln(p+18:)
      endif
      last = index(ln, '/') > 0                 ! the group ends on this line
      if (last) ln(index(ln, '/'):) = ' '
      do k = 1, n_removed
        n = len_trim(removed_keys(k))
        p = 0
        do
          q = index(ln(p+1:), trim(removed_keys(k)))
          if (q == 0) exit
          p = p + q                             ! the key starts at p
          if (p > 1) then
            if (ln(p-1:p-1) /= ' ' .and. ln(p-1:p-1) /= ',') cycle
          endif
          c = verify(ln(p+n:), ' ')             ! the first non-blank after it
          if (c == 0) cycle
          if (ln(p+n+c-1:p+n+c-1) == '=' .or. ln(p+n+c-1:p+n+c-1) == '(') hit(k) = .true.
        enddo
      enddo
      if (last) exit
    enddo
    do k = 1, n_removed
      if (.not. hit(k)) cycle
      if (len_trim(found) > 0) then
        found = trim(found) // ', ' // trim(removed_keys(k))
      else
        found = trim(removed_keys(k))
      endif
    enddo
  end subroutine removed_keys_set

  subroutine check_settings()
    ! The CPU's constraint checks (check_settings), then the values this
    ! port supports.
    real(rp), parameter :: tol = 1.e-12_rp
    real(rp) :: theta_min
    integer :: nerr
    nerr = 0
    if (abs(imex_theta_c + 0.5_rp * imex_mu - imex_rho - 0.5_rp) > tol) &
      call bad('imex_theta_c + imex_mu/2 - imex_rho must be 1/2 (corrector centred at n+1/2)')
    if (abs(imex_theta_c + imex_mu + imex_nu + imex_rho - 1.0_rp) > tol) &
      call bad('imex_theta_c + imex_mu + imex_nu + imex_rho must be 1')
    theta_min = merge(0.0_rp, 0.5_rp, imex_allow_explicit)
    call range('imex_theta_p', imex_theta_p, theta_min, 1.0_rp)
    call range('imex_theta_c', imex_theta_c, theta_min, 1.0_rp)
    call range('imex_phi', imex_phi, 0.0_rp, 1.0_rp)
    call range('imex_nu', imex_nu, 0.0_rp, 1.0_rp)
    call range('imex_rho', imex_rho, 0.0_rp, 0.5_rp)
    call range('imex_mu', imex_mu, -0.5_rp, 1.0_rp)
    if (imex_theta_p + imex_phi > 1.0_rp + tol) call bad('imex_theta_p + imex_phi must be <= 1')
    if (imex_tol_pred <= 0.0_rp .or. imex_tol_corr <= 0.0_rp .or. imex_maxit < 1) &
      call bad('imex_tol_pred, imex_tol_corr and imex_maxit must be positive')
    if (imex_n2_ref < 0.0_rp .or. imex_n2_taper < 0.0_rp) call bad('imex_n2_ref and imex_n2_taper must be >= 0')
    if (imex_verbose < 0 .or. imex_verbose > 2) call bad('imex_verbose must be 0, 1 or 2')
    if (imex_mix_split < 0 .or. imex_mix_split > 1) call bad('imex_mix_split must be 0 or 1')
    if (imex_warm < 0 .or. imex_warm > 1) call bad('imex_warm must be 0 or 1')
    if (imex_bref_hours <= 0.0_rp) call bad('imex_bref_hours must be positive')
    if (imex_setup_lag < 0 .or. imex_setup_lag > 1) call bad('imex_setup_lag must be 0 or 1')
    if (imex_pcg_start < 0 .or. imex_pcg_start > 3) call bad('imex_pcg_start must be 0, 1, 2 or 3')
    if (imex_pcg_fused < 0 .or. imex_pcg_fused > 1) call bad('imex_pcg_fused must be 0 or 1')
    if (imex_solver_prec /= 4 .and. imex_solver_prec /= 8) call bad('imex_solver_prec must be 8 or 4')
    if (imex_rst_state < 0 .or. imex_rst_state > 1) call bad('imex_rst_state must be 0 or 1')
    if (trim(imex_obc_mode) /= 'fixed' .and. trim(imex_obc_mode) /= 'modal' .and. trim(imex_obc_mode) /= 'auto') &
      call bad("imex_obc_mode must be 'fixed', 'modal' or 'auto'")
    ! (the CPU checks these when the run is 'modal'; here 'auto' is resolved at the first step)
    if (trim(imex_obc_mode) /= 'fixed' .and. (imex_obc_nmodes < 1 .or. imex_obc_nmodes > 20 .or. &
        (imex_obc_q0 /= 0.0_rp .and. (imex_obc_q0 < 0.2_rp .or. imex_obc_q0 > 1.0_rp)) .or. imex_obc_tref < 0.0_rp &
        .or. imex_obc_mprint < 0)) call bad('imex_obc_nmodes must be in 1..20, imex_obc_q0 0 (off) or in [0.2, 1]' &
         // ' (the admittances are divided by q0), imex_obc_tref >= 0 and imex_obc_mprint >= 0')
    if (imex_obc_cmin < 0.0_rp) call bad('imex_obc_cmin must be >= 0')
    ! What the GPU port has (the CPU's default path); the other values of the CPU's solver and
    ! development settings are CPU-only
    if (abs(imex_am3_crv - AM3_crv) > tol) call bad('imex_am3_crv: only 1/6 (the stock LF-AM3) on the GPU')
    if (imex_selftest /= 0) call bad('imex_selftest: not on the GPU (0)')
    if (imex_stage_fused /= 1) call bad('imex_stage_fused: only 1 on the GPU (the GPU''s fused stage is' &
         // ' imex_gpu_stage)')
    if (imex_redu /= 0) call bad('imex_redu: only 0 on the GPU (the CPU''s integer-digit sums are not ported;'// &
         ' imex_gpu_redu = 1 is the GPU''s deterministic sum)')
    ! The CPU's resolution (check_settings): the lag, the start-up and the
    ! fused solve need the default path, which is all the GPU runs
    lag_on = (imex_setup_lag == 1)
    pcg_start = imex_pcg_start
    pf_on = (imex_pcg_fused == 1) .and. pcg_start >= 2
    pf_prec = 8
    if (pf_on .and. imex_solver_prec == 4) pf_prec = 4
    pf_chk = pf_on .and. imex_verbose >= 2           ! CPU 24967a92: IMEX_PCHK at imex_verbose >= 2
    ! What the GPU has: imex_pcg_start 0 (imex_pcg_warm/imex_pcg) or the
    ! fused solve (start 2, 3); the CPU's imex_pcg_sc (start 1, or 2/3
    ! unfused) is not ported
    if (pcg_start > 0 .and. .not. pf_on) &
      call bad('imex_pcg_start 1, or 2/3 with imex_pcg_fused 0 (CPU imex_pcg_sc): not on the GPU')
    if (imex_gpu_redu < 0 .or. imex_gpu_redu > 1) call bad('imex_gpu_redu must be 0 or 1')
    redu_on = (imex_gpu_redu == 1)
    if (imex_gpu_stage < 0 .or. imex_gpu_stage > 1) call bad('imex_gpu_stage must be 0 or 1')
    ! the fused stage needs the fused solve with warm starts and both
    ! stages implicit (c2 > 0)
    stage_on = (imex_gpu_stage == 1) .and. pf_on .and. imex_warm == 1 &
               .and. imex_theta_p > 0.0_rp .and. imex_theta_c > 0.0_rp
    if (imex_gpu_sweep < 0 .or. imex_gpu_sweep > 1) call bad('imex_gpu_sweep must be 0 or 1')
    sweep_on = (imex_gpu_sweep == 1) .and. pf_on
    ! 'modal' on the GPU (obm_decide): the face term lives in the restructured sweep (PF_OPS), the face
    ! values in the fused stage (solve_stage_fast); imex_pcg_start 0 (apply_m, CPU's unfused operator)
    ! is not ported ('modal' stops, 'auto' runs 'fixed' with a note)
    obm_gpu_ok = pf_on .and. sweep_on .and. stage_on .and. imex_solver_enabled
    if (trim(imex_obc_mode) == 'modal' .and. .not. obm_gpu_ok) call bad("imex_obc_mode = 'modal' on the GPU" &
         // ' needs imex_pcg_start 2 or 3, imex_pcg_fused 1, imex_warm 1, theta_p, theta_c > 0, imex_gpu_stage 1' &
         // ' and imex_gpu_sweep 1')
    if (nerr > 0) error stop 'imex_igw: unsupported &imex_igw_settings (see above)'
  contains
    subroutine bad(what)
      character(len=*), intent(in) :: what
      nerr = nerr + 1
      if (talk) write(*,'(a,a)') 'ERROR: &imex_igw_settings: ', what
    end subroutine bad
    subroutine range(name, val, lo, hi)
      character(len=*), intent(in) :: name
      real(rp), intent(in) :: val, lo, hi
      if (val < lo - tol .or. val > hi + tol) call bad(name // ' out of range')
    end subroutine range
  end subroutine check_settings

  subroutine set_derived_weights()
    ! CPU set_derived_weights: regular step and first step of a run.
    real(rp) :: phi_1st, nu_1st
    wpg_pred_reg  = [imex_theta_p + imex_phi, 1.0_rp - imex_theta_p - imex_phi]
    wflx_pred_reg = [imex_theta_p, imex_phi, 1.0_rp - imex_theta_p - imex_phi]
    wpg_corr_reg  = [imex_theta_c + imex_mu, imex_nu, imex_rho]
    wflx_corr_reg = [imex_theta_c, imex_mu, imex_nu, imex_rho]
    phi_1st = 1.0_rp - imex_theta_p
    nu_1st  = 1.0_rp - imex_theta_c - imex_mu
    wpg_pred_1st  = [1.0_rp, 0.0_rp]
    wflx_pred_1st = [imex_theta_p, phi_1st, 0.0_rp]
    wpg_corr_1st  = [imex_theta_c + imex_mu, nu_1st, 0.0_rp]
    wflx_corr_1st = [imex_theta_c, imex_mu, nu_1st, 0.0_rp]
    wpg_pred = wpg_pred_1st; wflx_pred = wflx_pred_1st
    wpg_corr = wpg_corr_1st; wflx_corr = wflx_corr_1st; wrhs_corr = wflx_corr_1st
  end subroutine set_derived_weights

  subroutine print_settings()
    write(*,'(/1x,a)') 'IMEX_IGW: implicit internal-wave terms (GPU)'
    write(*,'(3x,a,3f10.5)') 'predictor theta_p, phi, 1-theta_p-phi   =', &
         imex_theta_p, imex_phi, 1.0_rp - imex_theta_p - imex_phi
    write(*,'(3x,a,4f10.5)') 'corrector theta_c, mu, nu, rho          =', &
         imex_theta_c, imex_mu, imex_nu, imex_rho
    write(*,'(3x,a,4f10.5)') 'corrector transport (n+1, *, n, n-1)    =', wflx_corr_reg
    write(*,'(3x,a,es10.3,a,es10.3,a,i4,a,i2)') 'imex_tol_pred =', imex_tol_pred, '  imex_tol_corr =', &
         imex_tol_corr, '  imex_maxit =', imex_maxit, '  imex_verbose =', imex_verbose
    write(*,'(3x,a,l2,a,l2,a,i2)') 'imex_solver_enabled =', imex_solver_enabled, &
         '  imex_allow_explicit =', imex_allow_explicit, '  imex_warm =', imex_warm
    write(*,'(3x,a,es10.2,a,es10.2)') 'imex_n2_taper =', imex_n2_taper, '  imex_n2_ref =', imex_n2_ref
    write(*,'(3x,a,f7.3,a,i2,a,i2,a,i2,a,i2)') 'imex_bref_hours =', imex_bref_hours, '  imex_setup_lag =', &
         merge(1, 0, lag_on), '  imex_pcg_start =', pcg_start, '  imex_pcg_fused =', merge(1, 0, pf_on), &
         '  imex_solver_prec =', pf_prec
    if (imex_solver_prec /= pf_prec) write(*,'(3x,a)') 'imex_solver_prec = 4 needs imex_pcg_fused = 1 in effect:' &
         // ' imex_solver_prec = 8'
    write(*,'(3x,a,i2,a,i2,a,i2)') 'imex_gpu_redu =', imex_gpu_redu, &
         '  imex_gpu_stage =', merge(1, 0, stage_on), '  imex_gpu_sweep =', merge(1, 0, sweep_on)
    write(*,'(3x,a,i2,a,i2,a,i2)') 'imex_redu =', imex_redu, '  imex_rst_state =', imex_rst_state, &
         '  imex_mix_split =', imex_mix_split
    write(*,'(3x,a,a,a)') "imex_obc_mode = '", trim(imex_obc_mode), "' (resolved at the first step: 'auto'" &
         // " selects 'modal' with an open edge and 3-D boundary data)"
    if (imex_gpu_stage == 1 .and. .not. stage_on) write(*,'(3x,a)') 'imex_gpu_stage = 1 needs the fused solve,' &
         // ' imex_warm = 1 and theta_p, theta_c > 0: the session-2 stage is used'
    write(*,*)
  end subroutine print_settings

  logical function imex_tsmix_active()
    ! t3dmix (and the passives' mixing) from t^n instead of t^n+1/2 (the CPU's, always with the solves)
    imex_tsmix_active = imex_on .and. imex_solver_enabled
  end function imex_tsmix_active

  !======================================================================
  ! Allocation
  !======================================================================
  subroutine imex_alloc()
    integer(ip) :: is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    allocate(pgu(is:ie, js:je, nz, 3), pgv(is:ie, js:je, nz, 3)); pgu = 0.0_rp; pgv = 0.0_rp
    allocate(u_nm1(is:ie, js:je, nz), v_nm1(is:ie, js:je, nz)); u_nm1 = 0.0_rp; v_nm1 = 0.0_rp
    allocate(pb_u(is:ie, js:je, nz), pb_v(is:ie, js:je, nz)); pb_u = 0.0_rp; pb_v = 0.0_rp
    allocate(cx(is:ie, js:je), cy(is:ie, js:je), cxo(is:ie, js:je), cyo(is:ie, js:je))
    allocate(act(is:ie, js:je), darea(is:ie, js:je), ad_act(is:ie, js:je))
    allocate(rshu(is:ie, js:je), rshv(is:ie, js:je))
    cx = 0.0_rp; cy = 0.0_rp; cxo = 0.0_rp; cyo = 0.0_rp; act = 0.0_rp; darea = 0.0_rp; ad_act = 0.0_rp
    allocate(cxm(is:ie, js:je), cym(is:ie, js:je), obcol_u(is:ie, js:je), obcol_v(is:ie, js:je))
    cxm = 0.0_rp; cym = 0.0_rp; obcol_u = 0; obcol_v = 0
    allocate(ob_ub(nz, 1), ob_act(1)); ob_ub = 0.0_rp; ob_act = 0.0_rp   ! (read by rhs_transport; obm_init
                                                                         ! reallocates them)
    rshu = 0.0_rp; rshv = 0.0_rp
    allocate(dza(is:ie, js:je, nz)); dza = 0.0_rp
    allocate(ad_bb(is:ie, js:je, nz), bref_sav(is:ie, js:je, nz), ad_n2raw(is:ie, js:je, nz))
    allocate(ad_n2t(is:ie, js:je, nz), ad_n2k(is:ie, js:je, nz), ad_en2(is:ie, js:je, nz))
    allocate(ad_n2c(is:ie, js:je, nz), ad_n2o(is:ie, js:je, nz))
    allocate(ad_cu(is:ie, js:je, nz), ad_cv(is:ie, js:je, nz), ad_cz(is:ie, js:je, 0:nz))
    allocate(ad_czt(is:ie, js:je, nz), ad_czb(is:ie, js:je, nz), ad_dd(is:ie, js:je, nz))
    ad_bb = 0.0_rp; bref_sav = 0.0_rp; ad_n2raw = 0.0_rp
    ad_n2t = 1.0_rp; ad_n2k = 1.0_rp; ad_en2 = 0.0_rp; ad_n2c = 1.0_rp; ad_n2o = 1.0_rp
    ad_cu = 0.0_rp; ad_cv = 0.0_rp; ad_cz = 0.0_rp; ad_czt = 0.0_rp; ad_czb = 0.0_rp; ad_dd = 0.0_rp
    allocate(bw0(is:ie, js:je, nz), bw1(is:ie, js:je, nz), bw2(is:ie, js:je, nz), bw3(is:ie, js:je, nz))
    allocate(bword(is:ie, js:je, nz))
    bw0 = 0.0_rp; bw1 = 0.0_rp; bw2 = 0.0_rp; bw3 = 0.0_rp; bword = 0
    allocate(wk_fu(is:ie, js:je, nz), wk_fv(is:ie, js:je, nz), wk_r0(is:ie, js:je, nz))
    allocate(wk_b(is:ie, js:je, nz), wk_dp(is:ie, js:je, nz), dps(is:ie, js:je, nz), dpw(is:ie, js:je, nz))
    wk_fu = 0.0_rp; wk_fv = 0.0_rp; wk_r0 = 0.0_rp; wk_b = 0.0_rp; wk_dp = 0.0_rp
    dps = 0.0_rp; dpw = 0.0_rp
    allocate(wk_r(is:ie, js:je, nz), wk_z(is:ie, js:je, nz), wk_p(is:ie, js:je, nz), wk_q(is:ie, js:je, nz))
    allocate(op_zc(is:ie, js:je, nz), op_db(is:ie, js:je, nz), op_fu(is:ie, js:je, nz), op_fv(is:ie, js:je, nz))
    allocate(hsol(is:ie, js:je, nz, 2))
    if (stage_on) then
      allocate(hs1(is:ie, js:je, nz), hs2(is:ie, js:je, nz)); hs1 = 0.0_rp; hs2 = 0.0_rp
    endif
    wk_r = 0.0_rp; wk_z = 0.0_rp; wk_p = 0.0_rp; wk_q = 0.0_rp
    op_zc = 0.0_rp; op_db = 0.0_rp; op_fu = 0.0_rp; op_fv = 0.0_rp; hsol = 0.0_rp
    if (lag_on) then
      allocate(hz_op(is:ie, js:je, nz), rshu_op(is:ie, js:je), rshv_op(is:ie, js:je))
      hz_op = 0.0_rp; rshu_op = 0.0_rp; rshv_op = 0.0_rp
    endif
    if (redu_on) call red_ensure(decomp%nx * decomp%ny)
    xring_on = stage_on .and. (comm_gpu_direct .or. .not. any([decomp%nbr_w, decomp%nbr_e, &
               decomp%nbr_s, decomp%nbr_n] >= 0))
    if (sweep_on) then
      allocate(op_sgu(is:ie, js:je), op_sgv(is:ie, js:je)); op_sgu = 0.0_rp; op_sgv = 0.0_rp
    endif
    if (pf_on) then
      allocate(pf_mq(is:ie, js:je, nz, 3)); pf_mq = 0.0_rp
      if (pf_prec == 8 .or. pf_chk) then
        allocate(pf_rczk8(is:ie, js:je, nz), pf_wr8(is:ie, js:je, nz), pf_dinv8(is:ie, js:je, nz, 2))
        pf_rczk8 = 0.0_rp; pf_wr8 = 0.0_rp; pf_dinv8 = 0.0_rp
      endif
      if (pf_prec == 8) then
        allocate(wk_p2(is:ie, js:je, nz)); wk_p2 = 0.0_rp
      else
        allocate(pf_rczk4(is:ie, js:je, nz), pf_wr4(is:ie, js:je, nz), pf_czt4(is:ie, js:je, nz))
        allocate(pf_czb4(is:ie, js:je, nz), pf_cu4(is:ie, js:je, nz), pf_cv4(is:ie, js:je, nz))
        allocate(pf_hz4(is:ie, js:je, nz), pf_dinv4(is:ie, js:je, nz, 2), pf_z4(is:ie, js:je, nz))
        allocate(pf_pa4(is:ie, js:je, nz), pf_pb4(is:ie, js:je, nz), pf_q4(is:ie, js:je, nz))
        allocate(pf_ru4(is:ie, js:je), pf_rv4(is:ie, js:je))
        pf_rczk4 = 0.0; pf_wr4 = 0.0; pf_czt4 = 0.0; pf_czb4 = 0.0; pf_cu4 = 0.0; pf_cv4 = 0.0
        pf_hz4 = 0.0; pf_dinv4 = 0.0; pf_z4 = 0.0; pf_pa4 = 0.0; pf_pb4 = 0.0; pf_q4 = 0.0
        pf_ru4 = 0.0_rp; pf_rv4 = 0.0_rp
      endif
    endif
  end subroutine imex_alloc

  !======================================================================
  ! Step bookkeeping
  !======================================================================
  subroutine imex_begin_step()
    ! CPU imex_begin_step: this step's weights; on the first step of a run
    ! the first-step weights and u_nm1, v_nm1 := level n.
    if (.not. imex_on) return
    if (.not. allocated(pgu)) call imex_alloc()
    if (.not. obm_decided) call obm_decide()        ! 'modal' or 'fixed' (may turn the restart inexact)
    ! (exact restart with the IMEX state read, rst_read: an ordinary step; u^n-1 is the
    ! restart's earlier record, in slot 3-nstp; pg_nm1 and the solver state come from the file)
    first_step = ((iic == ntstart) .and. .not. rst_read) .or. is_first_time_step()
    if (first_step) then
      wpg_pred = wpg_pred_1st; wflx_pred = wflx_pred_1st
      wpg_corr = wpg_corr_1st; wflx_corr = wflx_corr_1st; wrhs_corr = wflx_corr_1st
      call copy_level_n()
    else
      wpg_pred = wpg_pred_reg; wflx_pred = wflx_pred_reg
      wpg_corr = wpg_corr_reg; wflx_corr = wflx_corr_reg; wrhs_corr = wflx_corr_reg
      if (iic == ntstart) call copy_level_nm1()       ! rst_read
    endif
  end subroutine imex_begin_step

  subroutine copy_level_n()
    ! u_nm1, v_nm1 := u, v(nstp) (whole arrays, halos included)
    integer(ip) :: i, j, k, is, ie, js, je, nz, n
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    n = nstp
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          u_nm1(i, j, k) = u(i, j, k, n)
          v_nm1(i, j, k) = v(i, j, k, n)
        enddo
      enddo
    enddo
  end subroutine copy_level_n

  subroutine copy_level_nm1()
    ! u_nm1, v_nm1 := u, v(3 - nstp) (the exact restart's earlier record)
    integer(ip) :: i, j, k, is, ie, js, je, nz, n
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    n = 3 - nstp
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          u_nm1(i, j, k) = u(i, j, k, n)
          v_nm1(i, j, k) = v(i, j, k, n)
        enddo
      enddo
    enddo
  end subroutine copy_level_nm1

  subroutine imex_shift_levels()
    ! CPU imex_shift_levels (end of the step): level n becomes n-1; the
    ! diagnostics line.
    integer(ip) :: tmp
    if (.not. imex_on) return
    tmp = ipg_nm1; ipg_nm1 = ipg_n; ipg_n = tmp
    call copy_level_n()
    call imex_diag_line()
  end subroutine imex_shift_levels

  subroutine imex_diag_line()
    ! CPU imex_diag_line: stage statistics since the last line (cadence as
    ! the CPU: the ninfo steps, densified to powers of 2 at the start).
    real(8) :: loc(4), glob(4)
    integer :: i, j, ni
    if (.not. imex_solver_enabled) return
    ni = int(ninfo_steps)
    j = min(int(iic), 2 * ni) - 1
    i = 1
    do while (i < j)
      i = 2 * i
    enddo
    if (mod(int(iic), min(i, ni)) /= 0) return
    if (imex_verbose >= 1) then
      loc = [st_dp(1), st_du(1), st_dp(2), st_du(2)]
      call comm_allreduce_max(4, loc, glob)
      ! CPU 24967a92: max |dp| is tracked at imex_verbose >= 2 only (level 1: two values)
      if (imex_verbose >= 2) then
        if (talk) write(*,'(1x,a,f6.1,a,i3,a,f6.1,a,i3,a,4es24.16,a,i4)') 'IMEX it pred', &
             dble(st_it(1)) / max(st_n(1), 1), '/', st_itmax(1), ' corr', dble(st_it(2)) / max(st_n(2), 1), &
             '/', st_itmax(2), '  max|dp| |du| pred, corr', glob, '  unconverged', st_bad
      else
        if (talk) write(*,'(1x,a,f6.1,a,i3,a,f6.1,a,i3,a,2es24.16,a,i4)') 'IMEX it pred', &
             dble(st_it(1)) / max(st_n(1), 1), '/', st_itmax(1), ' corr', dble(st_it(2)) / max(st_n(2), 1), &
             '/', st_itmax(2), '  max|du| pred, corr', glob(2), glob(4), '  unconverged', st_bad
      endif
    endif
    st_n = 0; st_it = 0; st_itmax = 0; st_bad = 0; st_dp = 0.0_rp; st_du = 0.0_rp
  end subroutine imex_diag_line

  !======================================================================
  ! PG stores and the explicit combination (end of prsgrd)
  !======================================================================
  subroutine imex_pg_store_combine(i_ru_lo, i_ru_hi, j_rv_lo, j_rv_hi)
    ! CPU imex_pg_store_combine on the faces prsgrd computed: store ru, rv
    ! as this stage's level (n in the predictor, * in the corrector), replace
    ! them by the weighted combination, and restore the current level's
    ! depth mean (the weights act on the baroclinic part only).
    integer(ip), intent(in) :: i_ru_lo, i_ru_hi, j_rv_lo, j_rv_hi
    integer(ip) :: i, j, k, nx, ny, nz, sc, s1, s2
    real(rp) :: w1, w2, w3, sc_u, sr_u, sh_u, hf
    if (.not. imex_on) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    if (nrhs == 3) then            ! corrector: level *, combine (*, n, n-1)
      sc = 3; s1 = ipg_n; s2 = ipg_nm1
      w1 = wpg_corr(1); w2 = wpg_corr(2); w3 = wpg_corr(3)
    else                           ! predictor: level n, combine (n, n-1)
      sc = ipg_n; s1 = ipg_nm1; s2 = ipg_nm1
      w1 = wpg_pred(1); w2 = wpg_pred(2); w3 = 0.0_rp
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_ru_lo, i_ru_hi
        sc_u = 0.0_rp; sr_u = 0.0_rp; sh_u = 0.0_rp
        do k = 1, nz
          pgu(i, j, k, sc) = ru(i, j, k)
          if (sc == 3) then
            ru(i, j, k) = w1 * pgu(i, j, k, sc) + w2 * pgu(i, j, k, s1) + w3 * pgu(i, j, k, s2)
          else
            ru(i, j, k) = w1 * pgu(i, j, k, sc) + w2 * pgu(i, j, k, s1)
          endif
          sc_u = sc_u + pgu(i, j, k, sc)
          sr_u = sr_u + ru(i, j, k)
          sh_u = sh_u + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
        enddo
        sc_u = (sc_u - sr_u) / sh_u
        do k = 1, nz
          ru(i, j, k) = ru(i, j, k) + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * sc_u
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_rv_lo, j_rv_hi
      do i = 1, nx
        sc_u = 0.0_rp; sr_u = 0.0_rp; sh_u = 0.0_rp
        do k = 1, nz
          pgv(i, j, k, sc) = rv(i, j, k)
          if (sc == 3) then
            rv(i, j, k) = w1 * pgv(i, j, k, sc) + w2 * pgv(i, j, k, s1) + w3 * pgv(i, j, k, s2)
          else
            rv(i, j, k) = w1 * pgv(i, j, k, sc) + w2 * pgv(i, j, k, s1)
          endif
          sc_u = sc_u + pgv(i, j, k, sc)
          sr_u = sr_u + rv(i, j, k)
          sh_u = sh_u + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
        enddo
        sc_u = (sc_u - sr_u) / sh_u
        do k = 1, nz
          rv(i, j, k) = rv(i, j, k) + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * sc_u
        enddo
      enddo
    enddo
  end subroutine imex_pg_store_combine

  !======================================================================
  ! Predictor tracer transport (between the momentum and tracer predictors)
  !======================================================================
  subroutine imex_predictor_transport()
    ! CPU imex_predictor_transport: exchange u*, v*; Flx_pred = Hz_u dn_u
    ! (theta_p u* + phi u^n + (1-theta_p-phi) u^n-1) into FlxU, FlxV over
    ! set_HUV's ranges (u* with set_HUV1's barotropic mean, the CPU's
    ! imex_pred_btfix_tile); exchange; omega (We, Wi). set_HUV1 overwrites
    ! FlxU, FlxV later as usual.
    use geom, only: omega
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je, n_new, n_stp, first
    integer(ip) :: i_v0, i_v1, j_u0, j_u1
    real(rp) :: w1, w2, w3, sd, sf, dcc, mis, pbx
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    n_new = nnew; n_stp = nstp
    w1 = wflx_pred(1); w2 = wflx_pred(2); w3 = wflx_pred(3)
    first = merge(1_ip, 0_ip, is_first_time_step())
    i_v0 = 1; i_v1 = nx; j_u0 = 1; j_u1 = ny
    if (decomp%edge_w) i_v0 = 0
    if (decomp%edge_e) i_v1 = nx + 1
    if (decomp%edge_s) j_u0 = 0
    if (decomp%edge_n) j_u1 = ny + 1
    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_end()
    if (stage_on) then
      ! imex_gpu_stage = 1: the btfix velocity (u* - mis) umask formed in the
      ! transport kernel itself (no pb_u, pb_v copies; same expressions)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
      do j = j_u0, j_u1
        do i = 1, nx + 1
          sd = 0.0_rp; sf = 0.0_rp
          do k = 1, nz
            dcc = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)
            sd = sd + dcc; sf = sf + dcc * u(i, j, k, n_new)
          enddo
          if (first == 1) then
            mis = (sf - DU_avg1(i, j)) / sd
          else
            mis = (sf - 3.63_rp * DU_avg1(i, j) + 4.47_rp * DU_avg2(i, j) - 2.05_rp * DU_avg_bak(i, j)) / sd
          endif
          do k = 1, nz
            pbx = (u(i, j, k, n_new) - mis) * umask(i, j)
            FlxU(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j) &
                 * (w1 * pbx + w2 * u(i, j, k, n_stp) + w3 * u_nm1(i, j, k))
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
      do j = 1, ny + 1
        do i = i_v0, i_v1
          sd = 0.0_rp; sf = 0.0_rp
          do k = 1, nz
            dcc = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)
            sd = sd + dcc; sf = sf + dcc * v(i, j, k, n_new)
          enddo
          if (first == 1) then
            mis = (sf - DV_avg1(i, j)) / sd
          else
            mis = (sf - 3.63_rp * DV_avg1(i, j) + 4.47_rp * DV_avg2(i, j) - 2.05_rp * DV_avg_bak(i, j)) / sd
          endif
          do k = 1, nz
            pbx = (v(i, j, k, n_new) - mis) * vmask(i, j)
            FlxV(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) &
                 * (w1 * pbx + w2 * v(i, j, k, n_stp) + w3 * v_nm1(i, j, k))
          enddo
        enddo
      enddo
    else
      ! pb = u* everywhere, with set_HUV1's depth mean over its ranges
      ! (CPU imex_pred_btfix_tile; EXTRAP_BAR_FLUXES weights, DU_avg1 alone
      ! on the first step), masked
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            pb_u(i, j, k) = u(i, j, k, n_new)
            pb_v(i, j, k) = v(i, j, k, n_new)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = j_u0, j_u1
        do i = 1, nx + 1
          sd = 0.0_rp; sf = 0.0_rp
          do k = 1, nz
            dcc = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)
            sd = sd + dcc; sf = sf + dcc * u(i, j, k, n_new)
          enddo
          if (first == 1) then
            mis = (sf - DU_avg1(i, j)) / sd
          else
            mis = (sf - 3.63_rp * DU_avg1(i, j) + 4.47_rp * DU_avg2(i, j) - 2.05_rp * DU_avg_bak(i, j)) / sd
          endif
          do k = 1, nz
            pb_u(i, j, k) = (u(i, j, k, n_new) - mis) * umask(i, j)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny + 1
        do i = i_v0, i_v1
          sd = 0.0_rp; sf = 0.0_rp
          do k = 1, nz
            dcc = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)
            sd = sd + dcc; sf = sf + dcc * v(i, j, k, n_new)
          enddo
          if (first == 1) then
            mis = (sf - DV_avg1(i, j)) / sd
          else
            mis = (sf - 3.63_rp * DV_avg1(i, j) + 4.47_rp * DV_avg2(i, j) - 2.05_rp * DV_avg_bak(i, j)) / sd
          endif
          do k = 1, nz
            pb_v(i, j, k) = (v(i, j, k, n_new) - mis) * vmask(i, j)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = j_u0, j_u1
        do i = 1, nx + 1
          do k = 1, nz
            FlxU(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j) &
                 * (w1 * pb_u(i, j, k) + w2 * u(i, j, k, n_stp) + w3 * u_nm1(i, j, k))
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny + 1
        do i = i_v0, i_v1
          do k = 1, nz
            FlxV(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) &
                 * (w1 * pb_v(i, j, k) + w2 * v(i, j, k, n_stp) + w3 * v_nm1(i, j, k))
          enddo
        enddo
      enddo
    endif
    ! river faces: the river transport at time n (CPU imex_pred_riv_tile, 59a2e72c and
    ! 1648c059; FS_SPEC 9b-J item 1)
    if (river_on) call rivers_pred_flx()
    call hg_begin()
    call hg_add(F_FLXU)
    call hg_add(F_FLXV)
    call hg_end()
    call omega()
  end subroutine imex_predictor_transport

  !======================================================================
  ! Stage solve (rhs, elliptic solve, correction)
  !======================================================================
  subroutine imex_solve_stage(stage, dtau, cfbak)
    ! CPU imex_solve_stage. Predictor: after the momentum predictor's
    ! vertical solve, u,v(nnew) = u*_expl [m/s]; corrector: after
    ! step3d_uv1, u,v(nnew) = Hz_u u^n+1_expl.
    use obc, only: u3dbc, v3dbc
    integer, intent(in) :: stage
    real(rp), intent(in), optional :: dtau, cfbak
    real(rp) :: dts, cb, theta, tol, c2, resid, rdp
    integer :: is_, iters
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je
    if (.not. imex_on) return
    if (.not. imex_solver_enabled) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call imex_step_setup()
    if (stage_on) then
      call solve_stage_fast(stage, dtau, cfbak)
      return
    endif
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          wk_r0(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo
    if (stage == imex_stage_pred) then
      is_ = 1
      dts = dt; cb = 0.0_rp
      if (present(dtau)) dts = dtau
      if (present(cfbak)) cb = cfbak
      theta = imex_theta_p; tol = imex_tol_pred
      call u3dbc()                         ! 'fixed': OBC values of the explicit u*
      call v3dbc()
      call hg_begin()
      call hg_add(F_U, nnew)
      call hg_add(F_V, nnew)
      call hg_end()
      call rhs_transport(wflx_pred, nnew, nstp, .false., 0.0_rp, 0_ip)
      call stage_increment(dts)
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            wk_r0(i, j, k) = wk_r0(i, j, k) - cb * dpw(i, j, k)
          enddo
        enddo
      enddo
    else
      is_ = 2
      dts = dt
      theta = imex_theta_c; tol = imex_tol_corr
      call hg_begin()
      call hg_add(F_U, nnew)
      call hg_add(F_V, nnew)
      call hg_end()
      call rhs_transport(wrhs_corr(2:4), 3_ip, nstp, .true., wrhs_corr(1), nnew)
      call stage_increment(dts)
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            wk_r0(i, j, k) = wk_r0(i, j, k) - dps(i, j, k)
          enddo
        enddo
      enddo
    endif
    c2 = (theta * dts)**2
    iters = 0; resid = 0.0_rp
    if (c2 > 0.0_rp) then
      ! (N2 + c2 D_b D_b^T) eta = r0, as M eta = act r0/n2c
      if (pf_on) then                      ! fused solve (imex_pcg_start 2, 3; b formed in it)
        call solve_fused(is_, c2, tol, iters, resid)
      else
        CUF_DO3
        do k = 1, nz
          do j = 1, ny
            do i = 1, nx
              wk_b(i, j, k) = act(i, j) * wk_r0(i, j, k) / ad_n2c(i, j, k)
            enddo
          enddo
        enddo
        if (lag_on) then                   ! operator, Jacobi and inner products of the rebuild step
          call solve_warm(is_, c2, tol, iters, resid, hz_op, rshu_op, rshv_op)
        else
          call solve_warm(is_, c2, tol, iters, resid, Hz, rshu, rshv)
        endif
      endif
      if (imex_verbose >= 1 .and. talk) write(*,'(1x,a,i6,i2,a,i5,a,es24.16)') 'IMEX_SOLVE', iic, is_, &
           ' its', iters, ' resid', resid
      call correct(stage, theta * dts)
    else
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            wk_dp(i, j, k) = wk_r0(i, j, k)
          enddo
        enddo
      enddo
    endif
    if (stage == imex_stage_pred) then
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            dps(i, j, k) = wk_dp(i, j, k)
          enddo
        enddo
      enddo
    else
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            dpw(i, j, k) = dps(i, j, k) + wk_dp(i, j, k)
          enddo
        enddo
      enddo
    endif
    st_n(is_) = st_n(is_) + 1
    st_it(is_) = st_it(is_) + iters
    st_itmax(is_) = max(st_itmax(is_), iters)
    if (resid > tol) st_bad = st_bad + 1
    if (imex_verbose >= 2) then      ! max |dp| (CPU 24967a92: imex_verbose >= 2)
      rdp = 0.0_rp
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            rdp = max(rdp, abs(wk_dp(i, j, k)))
          enddo
        enddo
      enddo
      st_dp(is_) = max(st_dp(is_), rdp)
    endif
  end subroutine imex_solve_stage

  subroutine solve_stage_fast(stage, dtau, cfbak)
    ! imex_gpu_stage = 1: imex_solve_stage with the same arithmetic in fewer
    ! passes. The rhs transports as before; the stage increment, the memory
    ! term (predictor - cfbak (dps + dp_corr) of the last step = - cfbak
    ! (hsol1 + hsol2); corrector - dps = - hsol1) and r0 are formed in the
    ! solve's start-up kernel; the solve runs in hsol(:,:,:,stage) (the
    ! warm start in place); the correction reads it (correct_fast).
    use obc, only: u3dbc, v3dbc
    integer, intent(in) :: stage
    real(rp), intent(in), optional :: dtau, cfbak
    real(rp) :: dts, cb, theta, tol, c2, resid, rdp
    integer :: is_, iters
    integer(ip) :: i, j, k, nx, ny, nz
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    if (stage == imex_stage_pred) then
      is_ = 1
      dts = dt; cb = 0.0_rp
      if (present(dtau)) dts = dtau
      if (present(cfbak)) cb = cfbak
      theta = imex_theta_p; tol = imex_tol_pred
      call u3dbc()                         ! 'fixed': OBC values of the explicit u*
      call v3dbc()
      ! 'modal': the explicit face values, written into u, v(nnew) at the open faces (CPU obm_explicit,
      ! imex_obc_faces in imex_obc_pre; obm_explicit reads nothing u3dbc/v3dbc write)
      if (imex_obc_modal) call obm_explicit(imex_stage_pred, .true.)
      call hg_begin()
      call hg_add(F_U, nnew)
      call hg_add(F_V, nnew)
      call hg_end()
      call rhs_transport(wflx_pred, nnew, nstp, .false., 0.0_rp, 0_ip)
    else
      is_ = 2
      dts = dt; cb = 0.0_rp
      theta = imex_theta_c; tol = imex_tol_corr
      if (imex_obc_modal) call obm_explicit(imex_stage_corr, .false.)   ! 'modal': face values (rhs bfix)
      call hg_begin()
      call hg_add(F_U, nnew)
      call hg_add(F_V, nnew)
      call hg_end()
      call rhs_transport(wrhs_corr(2:4), 3_ip, nstp, .true., wrhs_corr(1), nnew)
    endif
    c2 = (theta * dts)**2
    fs_on = .true.; fs_dtl = dts; fs_cb = cb
    call solve_fused(is_, c2, tol, iters, resid)
    fs_on = .false.
    if (is_ == 1) fs_dpw = .false.
    if (imex_verbose >= 1 .and. talk) write(*,'(1x,a,i6,i2,a,i5,a,es24.16)') 'IMEX_SOLVE', iic, is_, &
         ' its', iters, ' resid', resid
    if (is_ == 1) then
      call correct_fast(stage, theta * dts, hs1)
    else
      call correct_fast(stage, theta * dts, hs2)
    endif
    st_n(is_) = st_n(is_) + 1
    st_it(is_) = st_it(is_) + iters
    st_itmax(is_) = max(st_itmax(is_), iters)
    if (resid > tol) st_bad = st_bad + 1
    if (imex_verbose >= 2) then      ! max |dp| (CPU 24967a92: imex_verbose >= 2)
      rdp = 0.0_rp
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            rdp = max(rdp, abs(merge(hs1(i, j, k), hs2(i, j, k), is_ == 1)))
          enddo
        enddo
      enddo
      st_dp(is_) = max(st_dp(is_), rdp)
    endif
  end subroutine solve_stage_fast

  subroutine rhs_transport(w, sa, sb, mom, wm, sm)
    ! CPU imex_rhs_transport: projected rhs transports at the faces of the
    ! tile's cells: F = Hz_u dn_u (w1 u(sa) + w2 u(sb) + w3 u_nm1)
    ! [+ wm dn_u u(sm), Hz_u-weighted momentum]; at open-boundary faces the
    ! momentum term takes Hz_u u(sa) (the OBC's value: u(nnew) there is not
    ! Hz-weighted); F -= Hz_u sum_k F / sum_k Hz_u; closed faces 0.
    real(rp), intent(in) :: w(3), wm
    integer(ip), intent(in) :: sa, sb, sm
    logical, intent(in) :: mom
    integer(ip) :: i, j, k, nx, ny, nz, s_a, s_b, s_m, imo, bnd, skp, mdl, oc
    real(rp) :: w1, w2, w3, wmom, sf, d, hf, ua
    logical :: ew, ee, es, en
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    s_a = sa; s_b = sb; s_m = max(sm, 1_ip)
    w1 = w(1); w2 = w(2); w3 = w(3)
    imo = merge(1_ip, 0_ip, mom); wmom = wm
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    skp = merge(1_ip, 0_ip, stage_on)    ! imex_gpu_stage: closed faces (F = 0 on every call) skipped
    mdl = merge(1_ip, 0_ip, imex_obc_modal)   ! 'modal': the face's explicit value instead of u* (CPU bfix)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        if (skp == 0 .or. cx(i, j) > 0.0_rp) then
        sf = 0.0_rp
        do k = 1, nz
          hf = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
          wk_fu(i, j, k) = hf * dn_u(i, j) * (w1 * u(i, j, k, s_a) + w2 * u(i, j, k, s_b) + w3 * u_nm1(i, j, k))
          if (imo == 1) wk_fu(i, j, k) = wk_fu(i, j, k) + wmom * dn_u(i, j) * u(i, j, k, s_m)
          sf = sf + wk_fu(i, j, k)
        enddo
        bnd = 0
        if (ew .and. i == 1) bnd = 1
        if (ee .and. i == nx + 1) bnd = 1
        if (imo == 1 .and. bnd == 1) then
          oc = 0
          if (mdl == 1) oc = obcol_u(i, j)
          do k = 1, nz
            hf = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
            ua = u(i, j, k, s_a)
            if (oc > 0) then
              if (ob_act(oc) > 0.0_rp) ua = ob_ub(k, oc)
            endif
            d = wmom * dn_u(i, j) * (hf * ua - u(i, j, k, s_m))
            wk_fu(i, j, k) = wk_fu(i, j, k) + d
            sf = sf + d
          enddo
        endif
        do k = 1, nz
          if (cx(i, j) > 0.0_rp) then
            wk_fu(i, j, k) = wk_fu(i, j, k) - 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * sf * rshu(i, j)
          else
            wk_fu(i, j, k) = 0.0_rp
          endif
        enddo
        endif
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        if (skp == 0 .or. cy(i, j) > 0.0_rp) then
        sf = 0.0_rp
        do k = 1, nz
          hf = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
          wk_fv(i, j, k) = hf * dm_v(i, j) * (w1 * v(i, j, k, s_a) + w2 * v(i, j, k, s_b) + w3 * v_nm1(i, j, k))
          if (imo == 1) wk_fv(i, j, k) = wk_fv(i, j, k) + wmom * dm_v(i, j) * v(i, j, k, s_m)
          sf = sf + wk_fv(i, j, k)
        enddo
        bnd = 0
        if (es .and. j == 1) bnd = 1
        if (en .and. j == ny + 1) bnd = 1
        if (imo == 1 .and. bnd == 1) then
          oc = 0
          if (mdl == 1) oc = obcol_v(i, j)
          do k = 1, nz
            hf = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
            ua = v(i, j, k, s_a)
            if (oc > 0) then
              if (ob_act(oc) > 0.0_rp) ua = ob_ub(k, oc)
            endif
            d = wmom * dm_v(i, j) * (hf * ua - v(i, j, k, s_m))
            wk_fv(i, j, k) = wk_fv(i, j, k) + d
            sf = sf + d
          enddo
        endif
        do k = 1, nz
          if (cy(i, j) > 0.0_rp) then
            wk_fv(i, j, k) = wk_fv(i, j, k) - 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * sf * rshv(i, j)
          else
            wk_fv(i, j, k) = 0.0_rp
          endif
        enddo
        endif
      enddo
    enddo
  end subroutine rhs_transport

  subroutine stage_increment(dts)
    ! CPU imex_stage_increment_coef (fused): wk_r0 += act dts n2c D(F),
    ! D(F) = -en2 [div_h(c F) + czt W(k) - czb W(k-1)], W the omega of F
    ! (W(0) = W(nz) = 0).
    real(rp), intent(in) :: dts
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: dtl, w, wm, d
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    dtl = dts
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        wm = 0.0_rp
        do k = 1, nz
          w = 0.0_rp
          if (k < nz) w = wm - (wk_fu(i + 1, j, k) - wk_fu(i, j, k) + wk_fv(i, j + 1, k) - wk_fv(i, j, k))
          d = -ad_en2(i, j, k) * (ad_cu(i + 1, j, k) * wk_fu(i + 1, j, k) - ad_cu(i, j, k) * wk_fu(i, j, k) &
              + ad_cv(i, j + 1, k) * wk_fv(i, j + 1, k) - ad_cv(i, j, k) * wk_fv(i, j, k) &
              + ad_czt(i, j, k) * w - ad_czb(i, j, k) * wm)
          wm = w
          wk_r0(i, j, k) = wk_r0(i, j, k) + act(i, j) * dtl * ad_n2c(i, j, k) * d
        enddo
      enddo
    enddo
  end subroutine stage_increment

  !======================================================================
  ! Per-step setup: geometry, b_ref, coefficients
  !======================================================================
  subroutine imex_step_setup()
    ! CPU imex_step_setup (adjoint mode): once per step; rshu, rshv, dza;
    ! then imex_adj_setup. imex_setup_lag = 1: dza and imex_adj_setup only
    ! on the steps that rebuild b_ref (coef_version bumped there, Hz, rshu,
    ! rshv saved as the operator's hz_op, rshu_op, rshv_op).
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je
    logical :: lag_skip
    if (coef_iic == iic) return
    lag_skip = .false.
    if (lag_on) lag_skip = .not. bref_due()
    if (.not. geometry_ready) call setup_geometry()
    if (rst_pend) then                     ! exact restart: the operator of the last rebuild
      if (lag_on) call rst_replay()
      rst_pend = .false.
    endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        rshu(i, j) = 0.0_rp
        do k = 1, nz
          rshu(i, j) = rshu(i, j) + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
        enddo
        rshu(i, j) = 1.0_rp / rshu(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        rshv(i, j) = 0.0_rp
        do k = 1, nz
          rshv(i, j) = rshv(i, j) + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
        enddo
        rshv(i, j) = 1.0_rp / rshv(i, j)
      enddo
    enddo
    if (lag_skip) then                     ! imex_setup_lag: the operator of the last rebuild
      coef_iic = iic
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 0, ny + 1
        do i = 0, nx + 1
          dza(i, j, k) = 1.0_rp / (z_r(i, j, min(k + 1, nz)) - z_r(i, j, max(k - 1, 1_ip)))
        enddo
      enddo
    enddo
    call adj_setup()
    coef_iic = iic
    coef_version = coef_version + 1
    if (lag_on) then                       ! the operator's Hz and face sums of this step
      is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
      call copy3(Hz, hz_op)
      CUF_DO2
      do j = js, je
        do i = is, ie
          rshu_op(i, j) = rshu(i, j)
          rshv_op(i, j) = rshv(i, j)
        enddo
      enddo
    endif
  end subroutine imex_step_setup

  integer function imex_nbref()
    ! CPU imex_nbref: the b_ref cadence in steps
    imex_nbref = max(1, nint(3600.0_rp * imex_bref_hours / dt))
  end function imex_nbref

  logical function bref_due()
    ! CPU imex_bref_due: this step's adj_setup rebuilds b_ref
    integer :: nb
    nb = imex_nbref()
    bref_due = .not. (nb > 1 .and. have_bref .and. iic - bref_last < nb .and. iic >= bref_last)
  end function bref_due

  subroutine setup_geometry()
    ! CPU imex_setup_geometry_tile: cx = umask dn_u/dm_u, cy = vmask
    ! dm_v/dn_v on the tile's faces (open edges keep their faces; closed
    ! edges are the masks' zeros), act = wet with an open face, darea =
    ! act/(pm pn); cxo, cyo: open-boundary faces out of the operator
    ! ('fixed'); ad_act = act with halos.
    use gpu_state, only: obc_west, obc_east, obc_south, obc_north
    integer(ip) :: i, j, nx, ny
    logical :: ew, ee, es, en, ow, oe, os, on
    nx = decomp%nx; ny = decomp%ny
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (j <= ny) then
          ! CPU 820bb8f3: the metric ratio only on open faces (MPI_MASKING: Inf/Inf x 0 = NaN at
          ! faces towards a dropped tile there; roms_gpu has no dropped tiles, same values)
          cx(i, j) = 0.0_rp
          if (umask(i, j) > 0.0_rp) cx(i, j) = dn_u(i, j) / dm_u(i, j) * umask(i, j)
          if (ew .and. i == 1 .and. .not. ow) cx(i, j) = 0.0_rp        ! closed edges
          if (ee .and. i == nx + 1 .and. .not. oe) cx(i, j) = 0.0_rp
          cxo(i, j) = cx(i, j)
          if (ew .and. i == 1) cxo(i, j) = 0.0_rp
          if (ee .and. i == nx + 1) cxo(i, j) = 0.0_rp
          cxm(i, j) = cxo(i, j)
        endif
        if (i <= nx) then
          cy(i, j) = 0.0_rp
          if (vmask(i, j) > 0.0_rp) cy(i, j) = dm_v(i, j) / dn_v(i, j) * vmask(i, j)
          if (es .and. j == 1 .and. .not. os) cy(i, j) = 0.0_rp
          if (en .and. j == ny + 1 .and. .not. on) cy(i, j) = 0.0_rp
          cyo(i, j) = cy(i, j)
          if (es .and. j == 1) cyo(i, j) = 0.0_rp
          if (en .and. j == ny + 1) cyo(i, j) = 0.0_rp
          cym(i, j) = cyo(i, j)
        endif
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        act(i, j) = 0.0_rp
        if (cx(i, j) + cx(i + 1, j) + cy(i, j) + cy(i, j + 1) > 0.0_rp) act(i, j) = 1.0_rp
        act(i, j) = act(i, j) * rmask(i, j)
        darea(i, j) = act(i, j) / (pm(i, j) * pn(i, j))
        ad_act(i, j) = act(i, j)
      enddo
    enddo
    call halo_fill_2d(ad_act, STAGGER_RHO)
    geometry_ready = .true.
  end subroutine setup_geometry

  subroutine adj_setup()
    ! CPU imex_adj_setup with imex_bref = 'ref', imex_bref_floor = 'coef':
    ! b_ref (rebuilt every nbref steps, else the saved one), eta set-up
    ! (N2 at cells, taper), face/interface values, 'coef' split, exact
    ! diagonal, 1/N2.
    integer(ip) :: i, j, k, nx, ny, nz, i0, i1, j0, j1, nbref
    real(rp) :: s1, s2, bref, grho, c1, c2s, rs2(2)
    real(8) :: loc(2), glob(2)
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nbref = imex_nbref()
    if (rst_on .or. (nbref > 1 .and. have_bref .and. iic - bref_last < nbref .and. iic >= bref_last)) then
      call copy3(bref_sav, ad_bb)          ! (rst_on: rst_replay, b_ref from the restart file)
    else
      ! bb relative to its domain mean over the active cells
      s1 = 0.0_rp; s2 = 0.0_rp
      if (redu_on) then                    ! column partials, deterministic sum (imex_red)
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny
          do i = 1, nx
            c1 = 0.0_rp; c2s = 0.0_rp
            do k = 1, nz
              c1 = c1 + act(i, j) * rho1(i, j, k)
              c2s = c2s + act(i, j)
            enddo
            red_part(i + (j - 1) * nx, 1) = c1; red_part(i + (j - 1) * nx, 2) = c2s
          enddo
        enddo
        call red_sum(2, rs2)
        s1 = rs2(1); s2 = rs2(2)
      else
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            s1 = s1 + act(i, j) * rho1(i, j, k)
            s2 = s2 + act(i, j)
          enddo
        enddo
      enddo
      endif
      loc = [real(s1, 8), real(s2, 8)]
      call comm_allreduce_sum(2, loc, glob)
      bref = glob(1) / max(glob(2), 1.0_8)
      grho = g / rho0
      i0 = 1; i1 = nx; j0 = 1; j1 = ny
      if (decomp%edge_w) i0 = 0
      if (decomp%edge_e) i1 = nx + 1
      if (decomp%edge_s) j0 = 0
      if (decomp%edge_n) j1 = ny + 1
      call zero3(ad_bb)
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = j0, j1
          do i = i0, i1
            ad_bb(i, j, k) = -g / rho0 * (rho1(i, j, k) - bref)
          enddo
        enddo
      enddo
      call halo_fill_3d(ad_bb, STAGGER_RHO)
      call bref_build()
      call copy3(ad_bb, bref_sav)
      have_bref = .true.
      bref_last = iic
    endif
    call eta_setup1()
    call bcoef()
    call cz_split()
    call adj_diag_coef()
    ! eta_setup(2): 1/N2
    call recip3(ad_n2t, ad_n2c)
    if (imex_obc_modal) call obm_build()  ! 'modal': modes, admittances and the Jacobi term (CPU obm_build)
  end subroutine adj_setup

  subroutine bref_build()
    ! CPU imex_bref_build ('coef' floor: no floor on b_ref itself):
    ! (1) per column, a volume-preserving stable sort to a stable profile
    ! (columns already monotone kept), on the tile's columns and the
    ! open-edge ghosts; exchange; (2) imex_bref_smooth passes of the masked
    ! 9-point filter along geopotentials (neighbour b interpolated linearly
    ! to the cell's z_r), open-edge ghosts from the interior, exchange;
    ! N2raw for the taper; exchange.
    integer(ip) :: i, j, k, nx, ny, nz, i0, i1, j0, j1, n, m, kk, ii, jj, ip0, nsm, nn, mono, q
    real(rp) :: rem, acc, need, take, z0, ztol, wsum, ssum, w, bi, b0max, bdmax, g3, g4
    real(rp) :: s3, s4, bdm, enc, ax, fx, fy
    real(8) :: loc(4), glob(4), b0max_g
    integer(ip) :: gi0, gj0, gnx, gny, iloc, jloc, kloc
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    i0 = 1; i1 = nx; j0 = 1; j1 = ny
    if (decomp%edge_w) i0 = 0
    if (decomp%edge_e) i1 = nx + 1
    if (decomp%edge_s) j0 = 0
    if (decomp%edge_n) j1 = ny + 1
    if (imex_verbose >= 1 .or. nbref_builds == 0) call copy3(ad_bb, bw3)   ! b before the build (diagnostics)
    ! (1) sort: bword = order, bw0 = sorted b, bw1 = sorted Hz, bw2 = laid-in b
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j0, j1
      do i = i0, i1
        if (rmask(i, j) > 0.0_rp) then
          mono = 1
          do k = 1, nz - 1
            if (ad_bb(i, j, k + 1) < ad_bb(i, j, k)) mono = 0
          enddo
          if (mono == 0) then
            do k = 1, nz
              bword(i, j, k) = k
            enddo
            do k = 2, nz
              m = bword(i, j, k); n = k - 1
              do q = 1, k - 1                       ! CPU: do while (n >= 1)
                if (ad_bb(i, j, bword(i, j, n)) >= ad_bb(i, j, m)) exit
                bword(i, j, n + 1) = bword(i, j, n); n = n - 1
              enddo
              bword(i, j, n + 1) = m
            enddo
            do k = 1, nz
              bw0(i, j, k) = ad_bb(i, j, bword(i, j, k)); bw1(i, j, k) = Hz(i, j, bword(i, j, k))
            enddo
            ip0 = 1; rem = bw1(i, j, 1)
            do kk = nz, 1, -1
              need = Hz(i, j, kk); acc = 0.0_rp
              do q = 1, nz + 2                      ! CPU: do while (need > 0 .and. ip <= nz)
                if (.not. (need > 0.0_rp .and. ip0 <= nz)) exit
                take = min(need, rem)
                acc = acc + bw0(i, j, ip0) * take; need = need - take; rem = rem - take
                if (rem <= 0.0_rp) then
                  ip0 = ip0 + 1
                  if (ip0 <= nz) rem = bw1(i, j, ip0)
                endif
              enddo
              if (need > 0.0_rp) acc = acc + bw0(i, j, nz) * need
              bw2(i, j, kk) = acc / Hz(i, j, kk)
            enddo
            do k = 1, nz
              ad_bb(i, j, k) = bw2(i, j, k)
            enddo
          endif
        endif
      enddo
    enddo
    call halo_fill_3d(ad_bb, STAGGER_RHO)
    ! (2) smoothing along geopotentials
    nsm = imex_bref_smooth
    do nn = 1, nsm
      call copy3(ad_bb, bw0)          ! bw0 = bb of this pass (read); ad_bb = smoothed
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            if (rmask(i, j) > 0.0_rp) then
              z0 = z_r(i, j, k)
              ztol = imex_bref_ztol * Hz(i, j, k)   ! neighbour test tolerance (CPU 941f00d2)
              wsum = 4.0_rp; ssum = 4.0_rp * bw0(i, j, k)
              do jj = j - 1, j + 1
                do ii = i - 1, i + 1
                  if (ii == i .and. jj == j) cycle
                  if (rmask(ii, jj) <= 0.0_rp) cycle
                  if (z0 < z_r(ii, jj, 1) - ztol .or. z0 > z_r(ii, jj, nz) + ztol) cycle
                  ! CPU: kk = 1; do while (kk < nz-1 .and. z_r(kk+1) < z0) kk = kk+1.
                  ! z_r increases strictly with k (Hz > 0), so that kk is the
                  ! one in [1, nz-1] with (kk = 1 or z_r(kk) < z0) and (kk =
                  ! nz-1 or z_r(kk+1) >= z0); found here from the cell's own
                  ! level k downwards, then upwards: the same kk in O(|kk-k|)
                  ! steps instead of O(kk) (session 3; the search dominated
                  ! the b_ref rebuild on Pac)
                  kk = min(k, nz - 1)
                  do q = 1, nz
                    if (.not. (kk > 1 .and. z_r(ii, jj, kk) >= z0)) exit
                    kk = kk - 1
                  enddo
                  do q = 1, nz
                    if (.not. (kk < nz - 1 .and. z_r(ii, jj, kk + 1) < z0)) exit
                    kk = kk + 1
                  enddo
                  bi = bw0(ii, jj, kk) + (bw0(ii, jj, kk + 1) - bw0(ii, jj, kk)) &
                       * (z0 - z_r(ii, jj, kk)) / (z_r(ii, jj, kk + 1) - z_r(ii, jj, kk))
                  w = 2.0_rp
                  if (ii /= i .and. jj /= j) w = 1.0_rp
                  wsum = wsum + w; ssum = ssum + w * bi
                enddo
              enddo
              ad_bb(i, j, k) = ssum / wsum
            endif
          enddo
        enddo
      enddo
      call bref_ghosts()
      call halo_fill_3d(ad_bb, STAGGER_RHO)
    enddo
    ! N2 of b_ref before any floor (taper)
    if (imex_n2_taper > 0.0_rp) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 0, ny + 1
          do i = 0, nx + 1
            ad_n2raw(i, j, k) = dza(i, j, k) * (ad_bb(i, j, min(k + 1, nz)) - ad_bb(i, j, max(k - 1, 1_ip)))
          enddo
        enddo
      enddo
    endif
    call halo_fill_3d(ad_bb, STAGGER_RHO)   ! ('coef': no floor here; the CPU exchanges again)
    nbref_builds = nbref_builds + 1
    if (imex_verbose == 0 .and. nbref_builds > 1) return
    ! IMEX_BREF line: max|b - b_ref|/max|b|, max |grad_s b| and |grad_s b_ref| at open faces
    b0max = 0.0_rp; bdmax = 0.0_rp; s3 = 0.0_rp; s4 = 0.0_rp
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          ! (one reduction statement per variable: nested conditional updates
          ! were not reduced by the CUF kernel)
          ax = merge(1.0_rp, 0.0_rp, act(i, j) > 0.0_rp)
          fx = merge(ax, 0.0_rp, cx(i, j) > 0.0_rp)
          fy = merge(ax, 0.0_rp, cy(i, j) > 0.0_rp)
          b0max = max(b0max, ax * abs(bw3(i, j, k)))
          bdmax = max(bdmax, ax * abs(bw3(i, j, k) - ad_bb(i, j, k)))
          s3 = max(s3, fx * abs(bw3(i, j, k) - bw3(i - 1, j, k)) * 0.5_rp * (pm(i, j) + pm(i - 1, j)), &
                       fy * abs(bw3(i, j, k) - bw3(i, j - 1, k)) * 0.5_rp * (pn(i, j) + pn(i, j - 1)))
          s4 = max(s4, fx * abs(ad_bb(i, j, k) - ad_bb(i - 1, j, k)) * 0.5_rp * (pm(i, j) + pm(i - 1, j)), &
                       fy * abs(ad_bb(i, j, k) - ad_bb(i, j - 1, k)) * 0.5_rp * (pn(i, j) + pn(i, j - 1)))
        enddo
      enddo
    enddo
    loc = [real(bdmax, 8), real(b0max, 8), real(s3, 8), real(s4, 8)]
    call comm_allreduce_max(4, loc, glob)
    ! where (global i, j, k): the smallest k-j-i index among the cells at the maximum
    bdm = glob(1); b0max_g = glob(2); gi0 = decomp%i0 - 1; gj0 = decomp%j0 - 1; gnx = decomp%gnx + 2; gny = decomp%gny + 2
    enc = huge(1.0_rp)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then
            if (abs(bw3(i, j, k) - ad_bb(i, j, k)) == bdm) &
              enc = min(enc, real((k * gny + (gj0 + j)) * gnx + (gi0 + i), rp))
          endif
        enddo
      enddo
    enddo
    loc(1) = real(enc, 8)
    call comm_allreduce_min(1, loc(1:1), glob(2:2))
    kloc = int(glob(2) / (real(gnx, 8) * gny)); jloc = int(mod(glob(2), real(gnx, 8) * gny) / gnx)
    iloc = int(mod(glob(2), real(gnx, 8)))
    if (imex_verbose >= 1 .and. talk) write(*,'(1x,a,i7,a,es24.16,a,3i5,a,2es24.16)') 'IMEX_BREF', iic, &
         ' max|b-b_ref|/max|b|', bdm / max(b0max_g, 1.e-300_8), ' at', iloc, jloc, kloc, &
         ' max|grad_s b|, b_ref', glob(3), glob(4)
    if (nbref_builds == 1) then
      g3 = 0.0_rp; g4 = 0.0_rp
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then
            g3 = max(g3, h(i, j)); g4 = max(g4, sqrt(pm(i, j)**2 + pn(i, j)**2))
          endif
        enddo
      enddo
      loc(1:2) = [real(g3, 8), real(g4, 8)]
      call comm_allreduce_max(2, loc(1:2), glob(1:2))
      if (talk) write(*,'(1x,a,es10.3,a,f7.1,a,f8.1,a,f7.3)') 'IMEX_BREF imex_n2_ref =', imex_n2_ref, &
           ': mode-1 Courant N_ref H dt sqrt(pm2+pn2)/pi at H max', glob(1), ' m, dt', dt, ' s:', &
           sqrt(imex_n2_ref) * glob(1) * dt * glob(2) / 3.14159265358979_8
    endif
  end subroutine bref_build

  subroutine bref_ghosts()
    ! open-edge ghost cells take their interior neighbour's value (W, E; then S, N)
    integer(ip) :: i, j, k, nx, ny, nz
    logical :: ew, ee, es, en
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    if (ew .or. ee) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          if (ew) ad_bb(0, j, k) = ad_bb(1, j, k)
          if (ee) ad_bb(nx + 1, j, k) = ad_bb(nx, j, k)
        enddo
      enddo
    endif
    if (es .or. en) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 1, nz
        do i = 0, nx + 1
          if (es) ad_bb(i, 0, k) = ad_bb(i, 1, k)
          if (en) ad_bb(i, ny + 1, k) = ad_bb(i, ny, k)
        enddo
      enddo
    endif
  end subroutine bref_ghosts

  subroutine eta_setup1()
    ! CPU imex_eta_setup(1) on the tile and one halo ring: N2 of b_ref
    ! (centred, >= imex_n2_reg, 'coef': >= imex_n2_ref), kernel factor 1,
    ! en2 = act pm pn/Hz; the memory terms rescaled to this step's N2;
    ! then the taper.
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je
    real(rp) :: reg, nref, tap, n2, sq
    logical :: resc
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    reg = imex_n2_reg; nref = imex_n2_ref; tap = imex_n2_taper
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 0, ny + 1
        do i = 0, nx + 1
          ad_n2t(i, j, k) = max(dza(i, j, k) * (ad_bb(i, j, min(k + 1, nz)) - ad_bb(i, j, max(k - 1, 1_ip))), reg)
          ad_n2k(i, j, k) = 1.0_rp
          ad_en2(i, j, k) = ad_act(i, j) * pm(i, j) * pn(i, j) / Hz(i, j, k)
          ad_n2t(i, j, k) = max(ad_n2t(i, j, k), nref)
        enddo
      enddo
    enddo
    resc = eta_old
    if (resc .and. stage_on) then
      ! imex_gpu_stage: dps is not kept (the predictor's solution replaces
      ! it before the corrector reads it); dpw = (hs1 + hs2) rescaled, the
      ! original's dpw = dps + dp_corr of the last step, rescaled
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            dpw(i, j, k) = (hs1(i, j, k) + hs2(i, j, k)) * (ad_n2o(i, j, k) / ad_n2t(i, j, k))
          enddo
        enddo
      enddo
      fs_dpw = .true.
    else if (resc) then
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            dps(i, j, k) = dps(i, j, k) * (ad_n2o(i, j, k) / ad_n2t(i, j, k))
            dpw(i, j, k) = dpw(i, j, k) * (ad_n2o(i, j, k) / ad_n2t(i, j, k))
          enddo
        enddo
      enddo
    endif
    call copy3(ad_n2t, ad_n2o)
    eta_old = .true.
    if (tap > 0.0_rp) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 0, ny + 1
          do i = 0, nx + 1
            n2 = max(ad_n2raw(i, j, k), 0.0_rp)
            sq = sqrt(max(n2 / (n2 + tap), 1.e-8_rp))
            ad_en2(i, j, k) = ad_en2(i, j, k) * sq
            ad_n2k(i, j, k) = 1.0_rp / sq
          enddo
        enddo
      enddo
      if (imex_verbose >= 1 .and. (iic == ntstart .or. mod(iic, 30_ip) == 0)) call taper_line()
    endif
  end subroutine eta_setup1

  subroutine taper_line()
    ! CPU IMEX_TAPER line: fractions of the active cells floored, s < 1/2,
    ! s < 1/10, N2raw <= 0
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: c1, c2, c3, c4, c5, n2, nref, tap
    real(8) :: loc(5), glob(5)
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nref = imex_n2_ref; tap = imex_n2_taper
    c1 = 0.0_rp; c2 = 0.0_rp; c3 = 0.0_rp; c4 = 0.0_rp; c5 = 0.0_rp
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then
            n2 = ad_n2raw(i, j, k)
            c1 = c1 + 1.0_rp
            if (n2 < nref) c2 = c2 + 1.0_rp
            if (n2 < tap) c3 = c3 + 1.0_rp
            if (n2 < tap / 9.0_rp) c4 = c4 + 1.0_rp
            if (n2 <= 0.0_rp) c5 = c5 + 1.0_rp
          endif
        enddo
      enddo
    enddo
    loc = [real(c1, 8), real(c2, 8), real(c3, 8), real(c4, 8), real(c5, 8)]
    call comm_allreduce_sum(5, loc, glob)
    if (talk) write(*,'(1x,a,i7,a,i10,a,5es24.16)') 'IMEX_TAPER', iic, ' active cells', nint(glob(1)), &
         ' fraction floored, s<1/2, s<1/10, N2raw<=0:', glob(2:5) / max(glob(1), 1.0_8)
  end subroutine taper_line

  subroutine bcoef()
    ! CPU imex_bcoef on ad_bb: thickness-aware C4 interface values cz
    ! (t_vadv_c4) on the tile's columns and one halo ring; centred
    ! 4th-order face values cu, cv (compute_horiz_tracer_fluxes.h without
    ! UPSTREAM_TS: masked differences, open-edge extrapolation of the
    ! difference) at the faces 1:nx+1 x 1:ny and 1:nx x 1:ny+1.
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: hk, hkp, hkm, hkp2, rh, tlin, tp, tm, dlm, dl0, dlp
    real(rp) :: fm1, f0, fp1, fm2, gr0, grm
    logical :: ew, ee, es, en
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 0, nx + 1
        ad_cz(i, j, 0) = 0.0_rp; ad_cz(i, j, nz) = 0.0_rp
        do k = 1, nz - 1
          ! dlt(k-1), dlt(k), dlt(k+1) with dlt(0) = dlt(nz) = 0
          dlm = 0.0_rp; dlp = 0.0_rp
          if (k >= 2) dlm = (ad_bb(i, j, k) - ad_bb(i, j, k - 1)) / max(Hz(i, j, k - 1) + Hz(i, j, k), 1.d-30)
          dl0 = (ad_bb(i, j, k + 1) - ad_bb(i, j, k)) / max(Hz(i, j, k) + Hz(i, j, k + 1), 1.d-30)
          if (k <= nz - 2) dlp = (ad_bb(i, j, k + 2) - ad_bb(i, j, k + 1)) / max(Hz(i, j, k + 1) + Hz(i, j, k + 2), 1.d-30)
          hk = Hz(i, j, k); hkp = Hz(i, j, k + 1)
          hkm = hk
          if (k >= 2) hkm = Hz(i, j, k - 1)
          hkp2 = hkp
          if (k <= nz - 2) hkp2 = Hz(i, j, k + 2)
          rh = 1.0_rp / max(hk + hkp, 1.d-30)
          tlin = (hkp * ad_bb(i, j, k) + hk * ad_bb(i, j, k + 1)) * rh
          tp = tlin - hk * hkp / max(hkm + hk + hkp, 1.d-30) * (dl0 - dlm)
          tm = tlin - hk * hkp / max(hk + hkp + hkp2, 1.d-30) * (dlp - dl0)
          ad_cz(i, j, k) = 0.5_rp * (tp + tm)
        enddo
      enddo
    enddo
    ! u-faces i = 1..nx+1, j = 1..ny: fd(i) = (b(i) - b(i-1)) umask(i);
    ! gr(i) = (fd(i+1) + fd(i))/2; cu = (b(i) + b(i-1) - (gr(i) - gr(i-1))/3)/2;
    ! at a west edge fd(0) := fd(1), at an east edge fd(nx+2) := fd(nx+1)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx + 1
          f0 = (ad_bb(i, j, k) - ad_bb(i - 1, j, k)) * umask(i, j)
          if (ew .and. i == 1) then
            fm1 = f0
          else
            fm1 = (ad_bb(i - 1, j, k) - ad_bb(i - 2, j, k)) * umask(i - 1, j)
          endif
          if (ee .and. i == nx + 1) then
            fp1 = f0
          else
            fp1 = (ad_bb(i + 1, j, k) - ad_bb(i, j, k)) * umask(i + 1, j)
          endif
          gr0 = 0.5_rp * (fp1 + f0)
          grm = 0.5_rp * (f0 + fm1)
          ad_cu(i, j, k) = 0.5_rp * (ad_bb(i, j, k) + ad_bb(i - 1, j, k) - 0.3333333333333333_rp * (gr0 - grm))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny + 1
        do i = 1, nx
          f0 = (ad_bb(i, j, k) - ad_bb(i, j - 1, k)) * vmask(i, j)
          if (es .and. j == 1) then
            fm1 = f0
          else
            fm1 = (ad_bb(i, j - 1, k) - ad_bb(i, j - 2, k)) * vmask(i, j - 1)
          endif
          if (en .and. j == ny + 1) then
            fp1 = f0
          else
            fp1 = (ad_bb(i, j + 1, k) - ad_bb(i, j, k)) * vmask(i, j + 1)
          endif
          gr0 = 0.5_rp * (fp1 + f0)
          grm = 0.5_rp * (f0 + fm1)
          ad_cv(i, j, k) = 0.5_rp * (ad_bb(i, j, k) + ad_bb(i, j - 1, k) - 0.3333333333333333_rp * (gr0 - grm))
        enddo
      enddo
    enddo
  end subroutine bcoef

  subroutine cz_split()
    ! CPU imex_cz_split(.true.) with 'coef': czt(k) = cz(k), czb(k) = cz(k-1),
    ! floored where active: czt(k) - b(k) >= n2_ref Hz/2 (k < nz),
    ! b(k) - czb(k) >= n2_ref Hz/2 (k > 1), on the tile and one halo ring
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: nref, fh
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nref = imex_n2_ref
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 0, ny + 1
        do i = 0, nx + 1
          ad_czt(i, j, k) = ad_cz(i, j, k); ad_czb(i, j, k) = ad_cz(i, j, k - 1)
          if (ad_act(i, j) > 0.0_rp) then
            fh = 0.5_rp * nref * Hz(i, j, k)
            if (k < nz) then
              if (ad_cz(i, j, k) - ad_bb(i, j, k) < fh) ad_czt(i, j, k) = ad_bb(i, j, k) + fh
            endif
            if (k > 1) then
              if (ad_bb(i, j, k) - ad_cz(i, j, k - 1) < fh) ad_czb(i, j, k) = ad_bb(i, j, k) - fh
            endif
          endif
        enddo
      enddo
    enddo
  end subroutine cz_split

  subroutine adj_diag_coef()
    ! CPU adj_diag_coef: dd = diag(D D^T), exact, per active cell from the
    ! column structure of its four faces (open-boundary faces excluded,
    ! cxo, cyo), scaled by pm pn/(Hz n2k^2). The prefix sums qq, pr of the
    ! CPU's arrays are carried as running sums (same order, same values).
    integer(ip) :: i, j, k, nx, ny, nz, m, ii, jj
    real(rp) :: rs, qq, qqp, qtot, pr, prp, hf, cf, q, gam, aa, bq, co
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        do k = 1, nz
          ad_dd(i, j, k) = 0.0_rp
        enddo
        if (act(i, j) > 0.0_rp) then
          do m = 1, 4
            if (m <= 2) then
              ii = i + (m - 1); jj = j
              co = cxo(ii, jj); rs = rshu(ii, jj)
            else
              ii = i; jj = j + (m - 3)
              co = cyo(ii, jj); rs = rshv(ii, jj)
            endif
            if (co > 0.0_rp) then
              ! total qq(nz)
              qtot = 0.0_rp
              do k = 1, nz
                if (m <= 2) then
                  hf = 0.5_rp * (Hz(ii, jj, k) + Hz(ii - 1, jj, k))
                else
                  hf = 0.5_rp * (Hz(ii, jj, k) + Hz(ii, jj - 1, k))
                endif
                qtot = qtot + hf * co
              enddo
              qqp = 0.0_rp; prp = 0.0_rp
              do k = 1, nz
                if (m <= 2) then
                  hf = 0.5_rp * (Hz(ii, jj, k) + Hz(ii - 1, jj, k)); cf = ad_cu(ii, jj, k)
                else
                  hf = 0.5_rp * (Hz(ii, jj, k) + Hz(ii, jj - 1, k)); cf = ad_cv(ii, jj, k)
                endif
                q = hf * co
                qq = qqp + q
                pr = prp + hf * rs
                gam = -cf * hf * rs + ad_czt(i, j, k) * pr - ad_czb(i, j, k) * prp
                aa = gam - ad_czt(i, j, k) + ad_czb(i, j, k)
                bq = gam + cf - ad_czt(i, j, k)
                ad_dd(i, j, k) = ad_dd(i, j, k) + aa * aa * qqp + bq * bq * q + gam * gam * (qtot - qq)
                qqp = qq; prp = pr
              enddo
            endif
          enddo
          do k = 1, nz
            ad_dd(i, j, k) = ad_dd(i, j, k) * pm(i, j) * pn(i, j) / (Hz(i, j, k) * ad_n2k(i, j, k)**2)
          enddo
        endif
      enddo
    enddo
  end subroutine adj_diag_coef

  !======================================================================
  ! Elliptic solver: (N2 + c2 D_b D_b^T) eta = r0 as M eta = act r0/n2c,
  ! M = 1/n2c + c2 D D^T (CPU imex_pcg, adjoint mode, fused operator,
  ! Jacobi preconditioner), warm start, correction
  !======================================================================
  subroutine solve_warm(is_, c2, tol, iters, resid, hv, rsu, rsv)
    ! CPU imex_pcg_warm (imex_warm = 1): the stage's previous solution as
    ! the initial guess (cold on the stage's first solve); the solution is
    ! kept for the next solve of this stage. hv, rsu, rsv: the operator's Hz
    ! and face-thickness sums (Hz, rshu, rshv; imex_setup_lag: hz_op,
    ! rshu_op, rshv_op, the CPU's imex_pcg_l and *_l routines)
    integer, intent(in) :: is_
    real(rp), intent(in) :: c2, tol
    integer, intent(out) :: iters
    real(rp), intent(out) :: resid
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: rsu(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rsv(decomp%is:decomp%ie, decomp%js:decomp%je)
    logical :: warm
    warm = (imex_warm == 1 .and. nh(is_) > 0)
    if (warm) then
      call copy3s(hsol, is_, wk_dp)
    endif
    call pcg(c2, tol, warm, iters, resid, hv, rsu, rsv)
    call store3s(wk_dp, hsol, is_)
    nh(is_) = min(nh(is_) + 1, 2)
  end subroutine solve_warm

  subroutine pcg(c2, tol, warm, iters, resid, hv, rsu, rsv)
    ! CPU imex_pcg (force_adj, fused operator, 'diag' preconditioner): PCG in
    ! the volume-weighted product, stop on the preconditioned residual
    ! sqrt(<r, M^-1 r>/<r0, M^-1 r0>) <= tol (warm: tol scaled to the cold
    ! solve's absolute stop, at most 1/2), at most imex_maxit iterations;
    ! two reductions per iteration; the search direction's halo one cell
    ! wide (here: the full halo exchange, same values). x = wk_dp returned
    ! with current halos.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(rp), intent(in) :: c2, tol
    logical, intent(in) :: warm
    integer, intent(out) :: iters
    real(rp), intent(out) :: resid
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: rsu(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rsv(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je
    integer :: it
    real(rp) :: c2l, s1, s2, s3, rho, rho0, rr, pq, alpha, beta, tole, bnorm, dum, cs3, rs3(3)
    real(8) :: loc(3), glob(3)
    logical :: broke
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    c2l = c2
    iters = 0; resid = 0.0_rp; rr = 0.0_rp; tole = tol; bnorm = 0.0_rp; broke = .false.
    if (.not. warm) call zero3(wk_dp)
    if (warm) then
      call apply_m(wk_dp, wk_q, c2l, dum, hv, rsu, rsv)   ! x has current halos
      s3 = 0.0_rp
      if (redu_on) then
        ! imex_gpu_redu = 1 (session 6): column partials (k ascending), deterministic sums
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny
          do i = 1, nx
            cs3 = 0.0_rp
            do k = 1, nz
              wk_r(i, j, k) = act(i, j) * wk_b(i, j, k) - wk_q(i, j, k)
              cs3 = cs3 + darea(i, j) * hv(i, j, k) * act(i, j) * wk_b(i, j, k)**2 &
                   / (1.0_rp / ad_n2c(i, j, k) + c2l * ad_dd(i, j, k))
            enddo
            red_part(i + (j - 1) * nx, 3) = cs3
          enddo
        enddo
        call precond(c2l, s1, s2, hv)
        call red_sum(3, rs3)
        s1 = rs3(1); s2 = rs3(2); s3 = rs3(3)
      else
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            wk_r(i, j, k) = act(i, j) * wk_b(i, j, k) - wk_q(i, j, k)
            s3 = s3 + darea(i, j) * hv(i, j, k) * act(i, j) * wk_b(i, j, k)**2 &
                 / (1.0_rp / ad_n2c(i, j, k) + c2l * ad_dd(i, j, k))
          enddo
        enddo
      enddo
      call precond(c2l, s1, s2, hv)
      endif
      loc = [real(s1, 8), real(s2, 8), real(s3, 8)]
      call comm_allreduce_sum(3, loc, glob)
      bnorm = glob(3)
      tole = min(0.5_rp, tol * sqrt(bnorm / max(glob(1), 1.e-300_8)))
    else
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            wk_r(i, j, k) = act(i, j) * wk_b(i, j, k)
          enddo
        enddo
      enddo
      call precond(c2l, s1, s2, hv)
      if (redu_on) then
        call red_sum(2, rs3(1:2)); s1 = rs3(1); s2 = rs3(2)
      endif
      loc(1:2) = [real(s1, 8), real(s2, 8)]
      call comm_allreduce_sum(2, loc(1:2), glob(1:2))
    endif
    rho = glob(1); rho0 = glob(1)
    if (.not. ieee_is_finite(rho0)) then
      resid = 1.0_rp
      call breakdown('<r0,M^-1 r0>', rho0, 0)
      n_maxit_hits = n_maxit_hits + 1
      return
    endif
    if (rho0 <= 0.0_rp) return                 ! zero rhs: x as it is
    call copy3(wk_z, wk_p)
    resid = 1.0_rp
    do it = 1, imex_maxit
      call halo_fill_3d(wk_p, STAGGER_RHO)
      call apply_m(wk_p, wk_q, c2l, s1, hv, rsu, rsv)
      if (redu_on) then
        call red_sum(1, rs3(1:1)); s1 = rs3(1)
      endif
      loc(1) = real(s1, 8)
      call comm_allreduce_sum(1, loc(1:1), glob(1:1))
      pq = glob(1)
      if (.not. ieee_is_finite(pq) .or. pq <= 0.0_rp) then
        call breakdown('<p,Ap>', pq, it)
        exit
      endif
      alpha = rho / pq
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            wk_dp(i, j, k) = wk_dp(i, j, k) + alpha * wk_p(i, j, k)
            wk_r(i, j, k) = wk_r(i, j, k) - alpha * wk_q(i, j, k)
          enddo
        enddo
      enddo
      call precond(c2l, s1, s2, hv)
      if (redu_on) then
        call red_sum(2, rs3(1:2)); s1 = rs3(1); s2 = rs3(2)
      endif
      loc(1:2) = [real(s1, 8), real(s2, 8)]
      call comm_allreduce_sum(2, loc(1:2), glob(1:2))
      iters = it
      if (.not. ieee_is_finite(glob(1)) .or. glob(1) < 0.0_8) then
        call breakdown('<r,M^-1 r>', glob(1), it)
        exit
      endif
      resid = sqrt(glob(1) / rho0)
      rr = glob(2)
      if (resid <= tole) exit
      beta = glob(1) / rho
      rho = glob(1)
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            wk_p(i, j, k) = wk_z(i, j, k) + beta * wk_p(i, j, k)
          enddo
        enddo
      enddo
    enddo
    if (broke .or. resid > tole) n_maxit_hits = n_maxit_hits + 1
    if (warm) resid = resid * sqrt(rho0 / max(bnorm, 1.e-300_8))
    call halo_fill_3d(wk_dp, STAGGER_RHO)
  contains
    subroutine breakdown(what, val, iter)
      character(len=*), intent(in) :: what
      real(rp), intent(in) :: val
      integer, intent(in) :: iter
      broke = .true.
      if (talk) write(*,'(1x,a,a,a,es12.4,a,i4,a,es12.4)') 'IMEX_IGW WARNING: PCG breakdown, ', &
           what, ' =', val, ' at iteration', iter, '; stopped with relative residual', resid
    end subroutine breakdown
  end subroutine pcg

  subroutine precond(c2, rz, rr, hv)
    ! CPU imex_adj_precond ('diag'): z = act r/(1/n2c + c2 dd) (exact
    ! Jacobi diagonal of M); local <r,z>, <r,r>
    real(rp), intent(in) :: c2
    real(rp), intent(out) :: rz, rr
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: c2l, a, b, ca, cb
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    c2l = c2; a = 0.0_rp; b = 0.0_rp
    if (redu_on) then
      ! imex_gpu_redu = 1 (session 6): column partials into red_part(:, 1:2) (the caller sums them)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          ca = 0.0_rp; cb = 0.0_rp
          do k = 1, nz
            wk_z(i, j, k) = act(i, j) * wk_r(i, j, k) / (1.0_rp / ad_n2c(i, j, k) + c2l * ad_dd(i, j, k))
            ca = ca + darea(i, j) * hv(i, j, k) * wk_r(i, j, k) * wk_z(i, j, k)
            cb = cb + darea(i, j) * hv(i, j, k) * wk_r(i, j, k) * wk_r(i, j, k)
          enddo
          red_part(i + (j - 1) * nx, 1) = ca; red_part(i + (j - 1) * nx, 2) = cb
        enddo
      enddo
      rz = 0.0_rp; rr = 0.0_rp
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          wk_z(i, j, k) = act(i, j) * wk_r(i, j, k) / (1.0_rp / ad_n2c(i, j, k) + c2l * ad_dd(i, j, k))
          a = a + darea(i, j) * hv(i, j, k) * wk_r(i, j, k) * wk_z(i, j, k)
          b = b + darea(i, j) * hv(i, j, k) * wk_r(i, j, k) * wk_r(i, j, k)
        enddo
      enddo
    enddo
    rz = a; rr = b
  end subroutine precond

  subroutine col_adjoint(x)
    ! The transpose's column quantities of x (current halos) on the tile and
    ! one halo ring: zc = -act x/n2k, db(k) = -sum_{k'>=k} (czt(k') zc(k') -
    ! czb(k'+1) zc(k'+1)) (db(nz) = 0), into op_zc, op_db (CPU col_row).
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: sg
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 0, nx + 1
        do k = 1, nz
          op_zc(i, j, k) = -ad_act(i, j) * x(i, j, k) / ad_n2k(i, j, k)
        enddo
        op_db(i, j, nz) = 0.0_rp
        sg = 0.0_rp
        do k = nz - 1, 1, -1
          sg = sg + ad_czt(i, j, k) * op_zc(i, j, k) - ad_czb(i, j, k + 1) * op_zc(i, j, k + 1)
          op_db(i, j, k) = -sg
        enddo
      enddo
    enddo
  end subroutine col_adjoint

  subroutine apply_m(x, mx, c2, xmx, hv, rsu, rsv)
    ! CPU imex_adj_apply_fused_coef: M x = x/n2c + c2 D D^T x on the active
    ! cells (x with current halos, one cell), local <x, M x>. D^T x at the
    ! faces: g = c (zc(L) - zc(R)) + db(L) - db(R), F = Hz_f cxo (g - gbar)
    ! (gbar the Hz_f-weighted column mean); D F on the cells with the
    ! vertical recursion of W.
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: mx(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), intent(in) :: c2
    real(rp), intent(out) :: xmx
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: rsu(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rsv(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: c2l, sg, hf, w, wm, dv, dd, a, ca
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    c2l = c2
    call col_adjoint(x)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        sg = 0.0_rp
        do k = 1, nz
          op_fu(i, j, k) = ad_cu(i, j, k) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
          sg = sg + 0.5_rp * (hv(i, j, k) + hv(i - 1, j, k)) * op_fu(i, j, k)
        enddo
        sg = sg * rsu(i, j)
        do k = 1, nz
          op_fu(i, j, k) = 0.5_rp * (hv(i, j, k) + hv(i - 1, j, k)) * cxo(i, j) * (op_fu(i, j, k) - sg)
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        sg = 0.0_rp
        do k = 1, nz
          op_fv(i, j, k) = ad_cv(i, j, k) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
          sg = sg + 0.5_rp * (hv(i, j, k) + hv(i, j - 1, k)) * op_fv(i, j, k)
        enddo
        sg = sg * rsv(i, j)
        do k = 1, nz
          op_fv(i, j, k) = 0.5_rp * (hv(i, j, k) + hv(i, j - 1, k)) * cyo(i, j) * (op_fv(i, j, k) - sg)
        enddo
      enddo
    enddo
    a = 0.0_rp
    if (redu_on) then
      ! imex_gpu_redu = 1 (session 6): the column partial of <x, M x> into red_part(:, 1) (the
      ! caller sums it); the same M x
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          wm = 0.0_rp; ca = 0.0_rp
          do k = 1, nz
            dv = op_fu(i + 1, j, k) - op_fu(i, j, k) + op_fv(i, j + 1, k) - op_fv(i, j, k)
            w = 0.0_rp
            if (k < nz) w = wm - dv
            hf = -ad_en2(i, j, k) * (ad_cu(i + 1, j, k) * op_fu(i + 1, j, k) - ad_cu(i, j, k) * op_fu(i, j, k) &
                 + ad_cv(i, j + 1, k) * op_fv(i, j + 1, k) - ad_cv(i, j, k) * op_fv(i, j, k) &
                 + ad_czt(i, j, k) * w - ad_czb(i, j, k) * wm)
            wm = w
            mx(i, j, k) = act(i, j) * (x(i, j, k) / ad_n2c(i, j, k) + c2l * hf)
            ca = ca + darea(i, j) * hv(i, j, k) * x(i, j, k) * mx(i, j, k)
          enddo
          red_part(i + (j - 1) * nx, 1) = ca
        enddo
      enddo
      xmx = 0.0_rp
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        wm = 0.0_rp
        do k = 1, nz
          dv = op_fu(i + 1, j, k) - op_fu(i, j, k) + op_fv(i, j + 1, k) - op_fv(i, j, k)
          w = 0.0_rp
          if (k < nz) w = wm - dv
          hf = -ad_en2(i, j, k) * (ad_cu(i + 1, j, k) * op_fu(i + 1, j, k) - ad_cu(i, j, k) * op_fu(i, j, k) &
               + ad_cv(i, j + 1, k) * op_fv(i, j + 1, k) - ad_cv(i, j, k) * op_fv(i, j, k) &
               + ad_czt(i, j, k) * w - ad_czb(i, j, k) * wm)
          wm = w
          mx(i, j, k) = act(i, j) * (x(i, j, k) / ad_n2c(i, j, k) + c2l * hf)
          a = a + darea(i, j) * hv(i, j, k) * x(i, j, k) * mx(i, j, k)
        enddo
      enddo
    enddo
    xmx = a
  end subroutine apply_m

  subroutine correct(stage, tdt)
    ! CPU imex_correct_fused_coef: du = -tdt fm (g - gbar)/dm_u umask at the
    ! u-faces the stage updates (2:nx at a west edge, else 1:nx; rows 1:ny)
    ! and the v analogue, g the face values of D^T wk_dp; added to u, v(nnew)
    ! as velocity (predictor) or as Hz_f du (corrector); max |du| for the
    ! diagnostics line.
    integer, intent(in) :: stage
    real(rp), intent(in) :: tdt
    integer(ip) :: i, j, k, nx, ny, nz, i_u0, j_v0, n_new, pred, stat
    real(rp) :: sg, g, du, hf, fm, tl, dmx
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    i_u0 = 1; if (decomp%edge_w) i_u0 = 2
    j_v0 = 1; if (decomp%edge_s) j_v0 = 2
    n_new = nnew; tl = tdt
    pred = merge(1_ip, 0_ip, stage == imex_stage_pred)
    call col_adjoint(wk_dp)
    dmx = 0.0_rp
    stat = merge(1_ip, 0_ip, imex_verbose >= 1)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, nx
        sg = 0.0_rp
        do k = 1, nz
          g = ad_cu(i, j, k) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
          sg = sg + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * g
        enddo
        sg = sg * rshu(i, j)
        fm = 0.0_rp
        if (cxo(i, j) > 0.0_rp) fm = 1.0_rp
        do k = 1, nz
          g = ad_cu(i, j, k) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
          du = -tl * fm * (g - sg) / dm_u(i, j)
          du = du * umask(i, j)
          if (pred == 1) then
            u(i, j, k, n_new) = u(i, j, k, n_new) + du
          else
            hf = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
            u(i, j, k, n_new) = u(i, j, k, n_new) + hf * du
          endif
          if (stat == 1) dmx = max(dmx, abs(du))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        sg = 0.0_rp
        do k = 1, nz
          g = ad_cv(i, j, k) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
          sg = sg + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * g
        enddo
        sg = sg * rshv(i, j)
        fm = 0.0_rp
        if (cyo(i, j) > 0.0_rp) fm = 1.0_rp
        do k = 1, nz
          g = ad_cv(i, j, k) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
          du = -tl * fm * (g - sg) / dn_v(i, j)
          du = du * vmask(i, j)
          if (pred == 1) then
            v(i, j, k, n_new) = v(i, j, k, n_new) + du
          else
            hf = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
            v(i, j, k, n_new) = v(i, j, k, n_new) + hf * du
          endif
          if (stat == 1) dmx = max(dmx, abs(du))
        enddo
      enddo
    enddo
    if (stage == imex_stage_pred) then
      st_du(1) = max(st_du(1), dmx)
    else
      st_du(2) = max(st_du(2), dmx)
    endif
  end subroutine correct

  subroutine correct_fast(stage, tdt, x)
    ! imex_gpu_stage = 1: correct() for the solution x (hs1 or hs2): the
    ! column quantities without the act = 0 columns (zc, dbar are 0 there),
    ! then the u- and the v-correction, each skipping closed faces (cxo,
    ! cyo = 0: du = 0) and computing g once (op_fu, op_fv between the
    ! column-mean and the update pass; same arithmetic); the max |du|
    ! reduction only with imex_verbose >= 1 (a CUF reduction makes the
    ! kernel synchronous).
    integer, intent(in) :: stage
    real(rp), intent(in) :: tdt
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, i_u0, j_v0, n_new, pred, vb
    real(rp) :: sg, g, du, hf, tl, dmu, dmv
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    i_u0 = 1; if (decomp%edge_w) i_u0 = 2
    j_v0 = 1; if (decomp%edge_s) j_v0 = 2
    n_new = nnew; tl = tdt
    pred = merge(1_ip, 0_ip, stage == imex_stage_pred)
    vb = merge(1_ip, 0_ip, imex_verbose >= 1)
    CUF_DO2
    do j = 0, ny + 1
      do i = 0, nx + 1
        if (ad_act(i, j) > 0.0_rp) then
          do k = 1, nz
            op_zc(i, j, k) = -ad_act(i, j) * x(i, j, k) / ad_n2k(i, j, k)
          enddo
          op_db(i, j, nz) = 0.0_rp
          sg = 0.0_rp
          do k = nz - 1, 1, -1
            sg = sg + ad_czt(i, j, k) * op_zc(i, j, k) - ad_czb(i, j, k + 1) * op_zc(i, j, k + 1)
            op_db(i, j, k) = -sg
          enddo
        endif
      enddo
    enddo
    dmu = 0.0_rp; dmv = 0.0_rp
    if (vb == 1) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = i_u0, nx
          if (cxo(i, j) > 0.0_rp) then
            sg = 0.0_rp
            do k = 1, nz
              g = ad_cu(i, j, k) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
              op_fu(i, j, k) = g
              sg = sg + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * g
            enddo
            sg = sg * rshu(i, j)
            do k = 1, nz
              du = -tl * 1.0_rp * (op_fu(i, j, k) - sg) / dm_u(i, j)
              du = du * umask(i, j)
              if (pred == 1) then
                u(i, j, k, n_new) = u(i, j, k, n_new) + du
              else
                hf = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
                u(i, j, k, n_new) = u(i, j, k, n_new) + hf * du
              endif
              dmu = max(dmu, abs(du))
            enddo
          endif
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = j_v0, ny
        do i = 1, nx
          if (cyo(i, j) > 0.0_rp) then
            sg = 0.0_rp
            do k = 1, nz
              g = ad_cv(i, j, k) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
              op_fv(i, j, k) = g
              sg = sg + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * g
            enddo
            sg = sg * rshv(i, j)
            do k = 1, nz
              du = -tl * 1.0_rp * (op_fv(i, j, k) - sg) / dn_v(i, j)
              du = du * vmask(i, j)
              if (pred == 1) then
                v(i, j, k, n_new) = v(i, j, k, n_new) + du
              else
                hf = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
                v(i, j, k, n_new) = v(i, j, k, n_new) + hf * du
              endif
              dmv = max(dmv, abs(du))
            enddo
          endif
        enddo
      enddo
      if (stage == imex_stage_pred) then
        st_du(1) = max(st_du(1), dmu, dmv)
      else
        st_du(2) = max(st_du(2), dmu, dmv)
      endif
      return
    endif
    CUF_DO2
    do j = 1, ny
      do i = i_u0, nx
        if (cxo(i, j) > 0.0_rp) then
          sg = 0.0_rp
          do k = 1, nz
            g = ad_cu(i, j, k) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
            op_fu(i, j, k) = g
            sg = sg + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * g
          enddo
          sg = sg * rshu(i, j)
          do k = 1, nz
            du = -tl * 1.0_rp * (op_fu(i, j, k) - sg) / dm_u(i, j)
            du = du * umask(i, j)
            if (pred == 1) then
              u(i, j, k, n_new) = u(i, j, k, n_new) + du
            else
              hf = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
              u(i, j, k, n_new) = u(i, j, k, n_new) + hf * du
            endif
          enddo
        endif
      enddo
    enddo
    CUF_DO2
    do j = j_v0, ny
      do i = 1, nx
        if (cyo(i, j) > 0.0_rp) then
          sg = 0.0_rp
          do k = 1, nz
            g = ad_cv(i, j, k) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
            op_fv(i, j, k) = g
            sg = sg + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * g
          enddo
          sg = sg * rshv(i, j)
          do k = 1, nz
            du = -tl * 1.0_rp * (op_fv(i, j, k) - sg) / dn_v(i, j)
            du = du * vmask(i, j)
            if (pred == 1) then
              v(i, j, k, n_new) = v(i, j, k, n_new) + du
            else
              hf = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
              v(i, j, k, n_new) = v(i, j, k, n_new) + hf * du
            endif
          enddo
        endif
      enddo
    enddo
  end subroutine correct_fast

  !======================================================================
  ! Fused solve (imex_pcg_fused = 1; CPU imex_pcg_s, imex_pcg_f,
  ! imex_pf_setup8/4, imex_pf_check; the PCG itself in imex_pf.inc)
  !======================================================================
  subroutine solve_fused(is_, c2, tol, iters, resid)
    ! CPU imex_pcg_s with imex_pcg_fused: the stage's previous solution as
    ! the guess (imex_warm = 1; cold on the stage's first solve); with
    ! imex_pcg_start = 3 and the operator unchanged since that solve (same
    ! coef_version and c2) its stored W M x is reused. The solution is kept
    ! for the next solve of the stage.
    integer, intent(in) :: is_
    real(rp), intent(in) :: c2, tol
    integer, intent(out) :: iters
    real(rp), intent(out) :: resid
    logical :: warm, reuse
    warm = (imex_warm == 1 .and. nh(is_) > 0)
    reuse = pcg_start == 3 .and. warm .and. ps_mver(is_) == coef_version .and. ps_mc2(is_) == c2
    if (fs_on) then                      ! imex_gpu_stage: the solve in hs1 / hs2 itself
      if (lag_on .and. is_ == 1) then
        call solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, hz_op, rshu_op, rshv_op, hs1)
      else if (lag_on) then
        call solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, hz_op, rshu_op, rshv_op, hs2)
      else if (is_ == 1) then
        call solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, Hz, rshu, rshv, hs1)
      else
        call solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, Hz, rshu, rshv, hs2)
      endif
      nh(is_) = min(nh(is_) + 1, 2)
      return
    endif
    if (warm) call copy3s(hsol, is_, wk_dp)
    if (lag_on) then
      call solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, hz_op, rshu_op, rshv_op, wk_dp)
    else
      call solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, Hz, rshu, rshv, wk_dp)
    endif
    call store3s(wk_dp, hsol, is_)
    nh(is_) = min(nh(is_) + 1, 2)
  end subroutine solve_fused

  subroutine solve_fused_hv(is_, c2, tol, iters, resid, warm, reuse, hv, rsu, rsv, x)
    ! CPU imex_pcg_f: the coefficient set of the storage kind, the solve,
    ! the IMEX_PCHK check (imex_verbose >= 2); x the solution (wk_dp, or
    ! hsol(:,:,:,is_) under imex_gpu_stage)
    integer, intent(in) :: is_
    real(rp), intent(in) :: c2, tol
    integer, intent(out) :: iters
    real(rp), intent(out) :: resid
    logical, intent(in) :: warm, reuse
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: rsu(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rsv(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    if (pf_prec == 8 .or. pf_chk) call pf_setup8(is_, c2, hv)
    if (pf_prec == 4) then
      call pf_setup4(is_, c2, hv)
      call pf_solve4(wk_r0, x, c2, tol, iters, resid, warm, is_, hv, reuse, &
                     pf_rczk4, pf_czt4, pf_czb4, pf_cu4, pf_cv4, pf_hz4, pf_wr4, pf_ru4, pf_rv4, pf_dinv4, &
                     pf_z4, pf_pa4, pf_pb4, pf_q4)
    else
      call pf_solve8(wk_r0, x, c2, tol, iters, resid, warm, is_, hv, reuse, &
                     pf_rczk8, ad_czt, ad_czb, ad_cu, ad_cv, hv, pf_wr8, rsu, rsv, pf_dinv8, &
                     wk_z, wk_p, wk_p2, wk_q)
    endif
    if (pf_chk) call pf_check(is_, c2, iters, resid, hv, rsu, rsv, x)
  end subroutine solve_fused_hv

  subroutine pf_setup8(stage, c2, hv)
    ! CPU imex_pf_setup8: rczk = -act/n2k (tile and one halo ring), wr =
    ! darea hv/n2c once per coefficient set; dinv = act/(darea hv diag M)
    ! per stage once per coefficient set and c2
    integer, intent(in) :: stage
    real(rp), intent(in) :: c2
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, s
    real(rp) :: c2l
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    s = stage; c2l = c2
    if (pf_cver8 /= coef_version) then
      CUF_DO3
      do k = 1, nz
        do j = 0, ny + 1
          do i = 0, nx + 1
            pf_rczk8(i, j, k) = -ad_act(i, j) / ad_n2k(i, j, k)
            if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) &
              pf_wr8(i, j, k) = darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k)
          enddo
        enddo
      enddo
      pf_cver8 = coef_version
    endif
    if (pf_dver8(s) /= coef_version .or. pf_dc28(s) /= c2) then
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            if (act(i, j) > 0.0_rp) then
              pf_dinv8(i, j, k, s) = 1.0_rp / (darea(i, j) * hv(i, j, k) * (1.0_rp / ad_n2c(i, j, k) &
                                     + c2l * ad_dd(i, j, k)))
            else
              pf_dinv8(i, j, k, s) = 0.0_rp
            endif
          enddo
        enddo
      enddo
      if (imex_obc_modal) call obm_dinv(s, c2, hv, 8)   ! 'modal': + theta dt ob_kd on the boundary cells
      pf_dver8(s) = coef_version; pf_dc28(s) = c2
    endif
  end subroutine pf_setup8

  subroutine pf_setup4(stage, c2, hv)
    ! CPU imex_pf_setup4: each FP32 value computed in FP64 and rounded once
    ! per coefficient set (dinv: per stage and c2); pf_ru4, pf_rv4 = 1/sum_k
    ! of the real(4) face thicknesses in FP64 (exact depth-mean removal)
    integer, intent(in) :: stage
    real(rp), intent(in) :: c2
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je, s
    real(rp) :: c2l, sh
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    s = stage; c2l = c2
    if (pf_cver4 /= coef_version) then
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            pf_czt4(i, j, k) = real(ad_czt(i, j, k), 4); pf_czb4(i, j, k) = real(ad_czb(i, j, k), 4)
            pf_cu4(i, j, k) = real(ad_cu(i, j, k), 4); pf_cv4(i, j, k) = real(ad_cv(i, j, k), 4)
            pf_hz4(i, j, k) = real(hv(i, j, k), 4)
          enddo
        enddo
      enddo
      CUF_DO3
      do k = 1, nz
        do j = 0, ny + 1
          do i = 0, nx + 1
            pf_rczk4(i, j, k) = real(-ad_act(i, j) / ad_n2k(i, j, k), 4)
            if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) &
              pf_wr4(i, j, k) = real(darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k), 4)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx + 1
          sh = 0.0_rp
          do k = 1, nz
            sh = sh + 0.5_rp * (real(pf_hz4(i, j, k), 8) + real(pf_hz4(i - 1, j, k), 8))
          enddo
          pf_ru4(i, j) = 1.0_rp / sh
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny + 1
        do i = 1, nx
          sh = 0.0_rp
          do k = 1, nz
            sh = sh + 0.5_rp * (real(pf_hz4(i, j, k), 8) + real(pf_hz4(i, j - 1, k), 8))
          enddo
          pf_rv4(i, j) = 1.0_rp / sh
        enddo
      enddo
      pf_cver4 = coef_version
    endif
    if (pf_dver4(s) /= coef_version .or. pf_dc24(s) /= c2) then
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            if (act(i, j) > 0.0_rp) then
              pf_dinv4(i, j, k, s) = real(1.0_rp / (darea(i, j) * hv(i, j, k) * (1.0_rp / ad_n2c(i, j, k) &
                                     + c2l * ad_dd(i, j, k))), 4)
            else
              pf_dinv4(i, j, k, s) = 0.0
            endif
          enddo
        enddo
      enddo
      if (imex_obc_modal) call obm_dinv(s, c2, hv, 4)   ! 'modal': + theta dt ob_kd on the boundary cells
      pf_dver4(s) = coef_version; pf_dc24(s) = c2
    endif
  end subroutine pf_setup4

  subroutine pf_check(stage, c2, iters, resid, hv, rsu, rsv, x)
    ! CPU imex_pf_check (imex_verbose >= 2, diagnostic): the true residual of
    ! the solution wk_dp (halos current) with the FP64 operator, rt_true =
    ! W act b - W M x, against the solve's recursive rt (wk_r); the IMEX_PCHK
    ! line: stage, iterations, the solve's residual, the true residual and
    ! the gap, both in the stop's norm sqrt(<., D^-1 .>_W / <b, D^-1 b>_W)
    integer, intent(in) :: stage, iters
    real(rp), intent(in) :: c2, resid
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: rsu(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rsv(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, s
    real(rp) :: pq, bt, rt, a1, a2, a3
    real(8) :: loc(3), glob(3)
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    s = stage
    call pf_setup8(stage, c2, hv)
    call pf_op8(0, 0.0_rp, 0.0_rp, c2, x, pf_mq, wk_z, wk_q, wk_q, x, wk_q, pf_rczk8, ad_czt, ad_czb, &
                ad_cu, ad_cv, hv, pf_wr8, rsu, rsv, pq)
    a1 = 0.0_rp; a2 = 0.0_rp; a3 = 0.0_rp
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          bt = act(i, j) * darea(i, j) * hv(i, j, k) * (act(i, j) * wk_r0(i, j, k) / ad_n2c(i, j, k))
          rt = bt - pf_mq(i, j, k, 3)
          a1 = a1 + pf_dinv8(i, j, k, s) * rt * rt
          a2 = a2 + pf_dinv8(i, j, k, s) * (rt - wk_r(i, j, k))**2
          a3 = a3 + pf_dinv8(i, j, k, s) * bt * bt
        enddo
      enddo
    enddo
    loc = [real(a1, 8), real(a2, 8), real(a3, 8)]
    call comm_allreduce_sum(3, loc, glob)
    if (talk) write(*,'(1x,a,i2,i5,a,i2,3es24.16)') 'IMEX_PCHK', stage, iters, ' prec', pf_prec, &
         resid, sqrt(glob(1) / max(glob(3), 1.e-300_8)), sqrt(glob(2) / max(glob(3), 1.e-300_8))
  end subroutine pf_check

#include "imex_obm.inc"

#define PFK 8
#define PF_SOLVE pf_solve8
#define PF_START pf_start8
#define PF_START_RHS pf_start_rhs8
#define PF_XMQ pf_xmq8
#define PF_OPS pf_ops8
#define PF_OP pf_op8
#define PF_UP pf_up8
#define PF_XUPD pf_xupd8
#define PF_ZEXCH pf_zexch8
#define PF_RING1 pf_ring1_8
#define PF_COMMRING comm_ring_dev8
#define RG_SW rg8_sw
#define RG_SE rg8_se
#define RG_SS rg8_ss
#define RG_SN rg8_sn
#define RG_RW rg8_rw
#define RG_RE rg8_re
#define RG_RS rg8_rs
#define RG_RN rg8_rn
#include "imex_pf.inc"
#undef PFK
#undef PF_SOLVE
#undef PF_START
#undef PF_START_RHS
#undef PF_XMQ
#undef PF_OPS
#undef PF_OP
#undef PF_UP
#undef PF_XUPD
#undef PF_ZEXCH
#undef PF_RING1
#undef PF_COMMRING
#undef RG_SW
#undef RG_SE
#undef RG_SS
#undef RG_SN
#undef RG_RW
#undef RG_RE
#undef RG_RS
#undef RG_RN
#define PFK 4
#define PF_SOLVE pf_solve4
#define PF_START pf_start4
#define PF_START_RHS pf_start_rhs4
#define PF_XMQ pf_xmq4
#define PF_OPS pf_ops4
#define PF_OP pf_op4
#define PF_UP pf_up4
#define PF_XUPD pf_xupd4
#define PF_ZEXCH pf_zexch4
#include "imex_pf.inc"

  !======================================================================
  ! Whole-array helpers (halos included)
  !======================================================================
  subroutine copy3(a, b)
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: b(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          b(i, j, k) = a(i, j, k)
        enddo
      enddo
    enddo
  end subroutine copy3

  subroutine zero3(a)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          a(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine zero3

  subroutine recip3(a, b)
    ! b = 1/a (whole array)
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: b(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          b(i, j, k) = 1.0_rp / a(i, j, k)
        enddo
      enddo
    enddo
  end subroutine recip3

  subroutine copy3s(a, s, b)
    ! b = a(:,:,:,s) (whole array)
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 2)
    integer, intent(in) :: s
    real(rp), WORK_ATTR intent(inout) :: b(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, is, ie, js, je, nz, sl
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; sl = s
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          b(i, j, k) = a(i, j, k, sl)
        enddo
      enddo
    enddo
  end subroutine copy3s

  subroutine store3s(b, a, s)
    ! a(:,:,:,s) = b (whole array)
    real(rp), WORK_ATTR intent(in) :: b(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 2)
    integer, intent(in) :: s
    integer(ip) :: i, j, k, is, ie, js, je, nz, sl
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; sl = s
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          a(i, j, k, sl) = b(i, j, k)
        enddo
      enddo
    enddo
  end subroutine store3s

  !======================================================================
  ! Exact restart (CPU 22c05bc1 imex_rst_def/wrt/wrt_scal/get, imex_rst_replay; FS_SPEC 9b-C).
  ! The state a step reads from earlier steps, written after the step (his_io; after
  ! imex_shift_levels, so the pg slot ipg_nm1 is the next step's pg_nm1) and read by
  ! case_io on an exact restart, under the CPU's names:
  !   imex_pgu, imex_pgv  pgu, pgv(:,:,:,ipg_nm1) (u, v faces; the next step's pg_nm1)
  !   imex_dpw            the memory term dPw (imex_gpu_stage: hs1 + hs2, formed for the write)
  !   imex_xp, imex_xc    each stage's last solution (hs1, hs2; or hsol(:,:,:,1:2))
  !   imex_mxp, imex_mxc  W M x of each stage's last solve (pf_mq(:,:,:,1:2))
  !   imex_bref           b_ref of the last rebuild, imex_n2raw its N2 before the floor (taper)
  !   imex_hzop, imex_dza Hz and dza of the last rebuild step (setup lag: rst_replay rebuilds the
  !                       frozen operator from them); imex_n2o (no lag: the last setup's N2)
  !   imex_rst_scal(17)   the CPU's scalars (imex_rst_scal_vals), per record
  ! u^n-1 needs no variable (the earlier record, slot 3-nstp).
  ! Layout of CPU 73a63afb (FS_SPEC 3.8, 9b-H item 1): the 3-D imex_* variables have no time
  ! dimension (one copy per file) and are written with the second record of the EXACT_RESTART pair
  ! only (rec 2, the record a restart continues from); imex_rst_scal stays per record, flag 1 = 1
  ! only on the record whose state is in the file, 15 = layout (2; 0 in the files of e8fca7b-aeceaec,
  ! a copy per record, still read), 16 = mode (rst_mode: 1, 'modal' + 100 + its settings; files with
  ! removed options are read as inexact restarts, with a note; the boundary part is checked by
  ! obm_decide), 17 = the 'modal' reference's update count (ob_mn).
  !   imex_obcm           'modal': the reference m at the ghost cell beyond each open face
  ! Only the variables the settings in effect use are written (imex_rst_has): x with imex_warm > 0,
  ! mx with imex_pcg_start 3 + imex_warm 1 + the setup lag, n2raw whenever the taper uses it (CPU
  ! 72539140; before it, with the lag and 'coef', n2raw was recomputed on read from imex_bref and
  ! imex_dza, which did not reproduce the rebuild's values to the bit: restarts off the b_ref
  ! cadence were inexact; files without it still get that recompute, logged).
  ! imex_rst_state = 0: imex_rst_scal only (flag 1 = 0; inexact restart, logged).
  ! Every mode of the GPU restarts exactly (imex_pcg_start 0 too: its warm start is hsol, nh).
  !======================================================================
  logical function imex_rst_has(name)
    ! the variable is in this run's restart files (CPU imex_rst_def, rst_has_*)
    character(len=*), intent(in) :: name
    imex_rst_has = .false.
    if (.not. (imex_on .and. imex_solver_enabled)) return
    if (trim(name) == 'imex_rst_scal') then
      imex_rst_has = .true.
      return
    endif
    if (imex_rst_state == 0) return
    select case (trim(name))
    case ('imex_pgu', 'imex_pgv', 'imex_dpw', 'imex_bref')
      imex_rst_has = .true.
    case ('imex_xp', 'imex_xc')
      imex_rst_has = (imex_warm > 0)
    case ('imex_mxp', 'imex_mxc')
      imex_rst_has = (pcg_start == 3 .and. imex_warm == 1 .and. lag_on)
    case ('imex_n2raw')
      imex_rst_has = (imex_n2_taper > 0.0_rp)      ! whenever the taper uses it (CPU 72539140)
    case ('imex_hzop', 'imex_dza')
      imex_rst_has = lag_on
    case ('imex_n2o')
      imex_rst_has = .not. lag_on
    case ('imex_obcm')                             ! 'modal' with the reference (CPU rst_has_obcm)
      if (.not. obm_decided) call obm_decide()
      imex_rst_has = imex_obc_modal .and. imex_obc_tref > 0.0_rp
    end select
  end function imex_rst_has

  logical function imex_obm_rst_may(name)
    ! reading the initial file (before obm_decide): imex_obcm is read when the run may be 'modal' with
    ! the reference; obm_decide keeps it only for a 'modal' run of the same settings
    character(len=*), intent(in) :: name
    imex_obm_rst_may = imex_on .and. imex_solver_enabled .and. imex_rst_state == 1 .and. &
         trim(imex_obc_mode) /= 'fixed' .and. imex_obc_tref > 0.0_rp .and. trim(name) == 'imex_obcm'
  end function imex_obm_rst_may

  subroutine imex_obc_faces(stage)
    ! 'modal' (CPU imex_obc_faces after u3dbc/v3dbc): pre_step3d (stage 1, with the barotropic mean of
    ! the predictor transport) and step3d_uv2 (stage 2); the stage's face increment (CPU obm_correct)
    ! is applied here, fused with the face write
    integer, intent(in) :: stage
    if (.not. imex_on) return
    if (imex_obc_modal) call obm_correct_faces(stage)
  end subroutine imex_obc_faces

  logical function imex_stage_on()
    ! imex_gpu_stage = 1 in effect: the stage solutions are hs1, hs2 (else hsol(:,:,:,1:2))
    imex_stage_on = stage_on
  end function imex_stage_on

  subroutine imex_rst_alloc()
    ! the arrays exist (a restart written or read before the first IMEX step)
    if (.not. imex_on) return
    if (.not. allocated(pgu)) call imex_alloc()
    if (.not. allocated(pf_mq)) then
      allocate(pf_mq(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz, 3)); pf_mq = 0.0_rp
    endif
    if (.not. allocated(obcm3) .and. imex_obm_rst_may('imex_obcm')) then
      allocate(obcm3(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)); obcm3 = 0.0_rp
    endif
  end subroutine imex_rst_alloc

  subroutine imex_rst_prepare()
    ! before a restart record is written: imex_gpu_stage keeps no dpw array between steps,
    ! dpw = hs1 + hs2 (the CPU's dps + dp_corr; dpw is only read after eta_setup1 rewrites
    ! it); the device is drained so the host can read the managed arrays
    integer(ip) :: i, j, k, is, ie, js, je, nz
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (.not. (imex_on .and. imex_solver_enabled)) return
    call imex_rst_alloc()
    if (stage_on) then
      is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            dpw(i, j, k) = hs1(i, j, k) + hs2(i, j, k)
          enddo
        enddo
      enddo
    endif
    if (imex_rst_has('imex_obcm')) call obm_m_to3()   ! 'modal': m at the faces' ghost cells
#ifdef USE_CUDA
    ierr = cudaDeviceSynchronize()
#endif
  end subroutine imex_rst_prepare

  subroutine imex_rst_scal_vals(sc, rec)
    ! imex_rst_scal of record rec of the file (the CPU's layout): 1 state complete in this file
    ! (only with rec 2, the record the 3-D state is written with, and imex_rst_state 1), 2 lag on,
    ! 3 b_ref age in steps, 4 eta_old, 5-6 warm-start history count, 7-8 stored W M x valid for the
    ! operator in use, 9-10 its c2, 11 imex_warm, 12 imex_pcg_start in effect, 13 b_ref cadence,
    ! 14 solver storage, 15 layout (2: one time-less copy per file), 16 mode (rst_mode), 17 'modal'
    ! reference: predictor updates of m since it was set (ob_mn)
    real(8), intent(out) :: sc(imex_nsc)
    integer, intent(in) :: rec
    integer :: s
    sc = 0.0d0
    if (.not. (imex_on .and. imex_solver_enabled)) return
    if (allocated(pgu) .and. have_bref .and. coef_version > 0 .and. imex_rst_state == 1 .and. rec == 2) sc(1) = 1.0d0
    sc(2) = merge(1.0d0, 0.0d0, lag_on)
    if (have_bref) sc(3) = dble(iic - bref_last)
    sc(4) = merge(1.0d0, 0.0d0, eta_old)
    sc(5:6) = dble(nh)
    do s = 1, 2
      if (pcg_start == 3 .and. imex_warm == 1 .and. lag_on .and. ps_mver(s) == coef_version) sc(6 + s) = 1.0d0
    enddo
    sc(9:10) = ps_mc2
    sc(11) = dble(imex_warm); sc(12) = dble(pcg_start); sc(13) = dble(imex_nbref())
    sc(14) = dble(merge(pf_prec, 0, pf_on))
    sc(15) = 2.0d0; sc(16) = dble(rst_mode()); sc(17) = dble(ob_mn)
  end subroutine imex_rst_scal_vals

  integer function rst_mode()
    ! imex_rst_scal(16) (CPU rst_mode): 1; 'modal' + 100 + 1000 imex_obc_nmodes + 1000000 imex_obc_tref in
    ! hours (at least 1 for tref > 0), so that a restart across a change of these is refused as inexact
    ! (imex_obc_q0 is not encoded: it changes Y, which the replay rebuilds with the new value). Files
    ! written before the key trim of 2026-11-13 may also carry the removed options: 2 in place of 1
    ! imex_force_mode 'gradient', + 10 imex_bt_mode 'step2d', + 200000 imex_obc_fref = 2 (obc_mode_change)
    rst_mode = 1
    if (imex_obc_modal) rst_mode = rst_mode + 100 + 1000 * imex_obc_nmodes &
         + 1000000 * merge(max(1, nint(imex_obc_tref / 3600.0_rp)), 0, imex_obc_tref > 0.0_rp)
  end function rst_mode

  logical function rst_removed(mold)
    ! the rst_mode of a file written with an option removed on 2026-11-13
    integer, intent(in) :: mold
    rst_removed = mod(mold, 10) == 2 .or. mod(mold / 10, 10) == 1 .or. &
                  (mod(mold / 100, 10) == 1 .and. mod(mold / 100000, 10) == 2)
  end function rst_removed

  function obc_mode_change(mold, mnew) result(txt)
    ! a rst_mode change in words (CPU obc_mode_change, the inexact-restart note): the removed options of
    ! an older file, or the open-boundary settings
    integer, intent(in) :: mold, mnew
    character(len=256) :: txt
    character(len=128) :: rm
    logical :: mo, mn
    rm = ''
    if (mod(mold, 10) == 2) rm = "imex_force_mode 'gradient'"
    if (mod(mold / 10, 10) == 1) rm = trim(rm) // "  imex_bt_mode 'step2d'"
    if (mod(mold / 100, 10) == 1 .and. mod(mold / 100000, 10) == 2) rm = trim(rm) // "  imex_obc_fref = 2"
    if (len_trim(rm) > 0) then
      txt = 'the file was written with options removed on 2026-11-13: ' // trim(adjustl(rm))
      return
    endif
    mo = mod(mold / 100, 10) == 1; mn = mod(mnew / 100, 10) == 1
    if (mo .neqv. mn) then
      txt = "the open-boundary condition changes from imex_obc_mode '" // trim(merge('modal', 'fixed', mo)) &
            // "' to '" // trim(merge('modal', 'fixed', mn)) // "'"
      if (mn) txt = trim(txt) // " (the default 'auto' selects 'modal')"
    else if (mn) then
      txt = "the 'modal' boundary settings differ (imex_obc_nmodes or imex_obc_tref): the reference m" &
            // ' restarts from the current mismatch'
    else
      txt = 'restart mode'
    endif
  end function obc_mode_change

  logical function imex_rst_accept(exact, have, sc)
    ! case_io, before reading the fields: the state is read when the restart is exact and the
    ! record has a complete state for this run's settings; otherwise the first step starts the
    ! IMEX stores as before (first-step weights; inexact), with the CPU's note
    logical, intent(in) :: exact, have
    real(8), intent(in) :: sc(imex_nsc)
    character(len=512) :: why
    integer :: md
    imex_rst_accept = .false.
    if (.not. (imex_on .and. imex_solver_enabled)) return
    why = ''
    md = nint(sc(16))
    if (md == 0) md = 1                       ! (files of e8fca7b-aeceaec: adjoint)
    if (.not. exact) then
      why = 'not an exact restart'
    else if (.not. have) then
      why = 'no IMEX state in the initial file'
    else if (imex_rst_state == 0) then
      why = 'imex_rst_state = 0'
    else if (nint(sc(1)) /= 1) then
      why = 'the record has no IMEX state (written with imex_rst_state = 0, incomplete, or not the second'// &
            ' record of an exact-restart pair)'
    else if (nint(sc(2)) /= merge(1, 0, lag_on) .or. nint(sc(11)) /= imex_warm .or. nint(sc(12)) /= pcg_start &
             .or. nint(sc(13)) /= imex_nbref() .or. nint(sc(14)) /= merge(pf_prec, 0, pf_on) .or. rst_removed(md)) then
      why = 'IMEX settings differ from the run that wrote it (setup lag, warm start, b_ref cadence, solver, mode'
      if (nint(sc(11)) >= 2) why = trim(why) // '; the file was written with imex_warm = 2 or 3, removed on' // &
           ' 2026-11-13'
      if (rst_removed(md)) why = trim(why) // '; ' // trim(obc_mode_change(md, rst_mode()))
      why = trim(why) // ')'
    endif
    if (why /= '') then
      if (talk) write(*,'(1x,A)') 'IMEX_IGW: '//trim(why)//': the IMEX stores start as on a first step'// &
           ' (inexact restart)'
      return
    endif
    call imex_rst_alloc()
    ! the boundary part of the mode (CPU rst_mode: 'modal' and its settings) is checked by obm_decide at
    ! the first step, when the open sides and the boundary data are known
    rst_md_file = md
    imex_rst_accept = .true.
  end function imex_rst_accept

  subroutine imex_rst_loaded(sc, n2raw_read, obcm_read)
    ! after case_io has read the fields (imex_rst_accept true): halos, the scalars; the
    ! frozen operator is rebuilt at the first imex_step_setup (rst_replay). n2raw_read: imex_n2raw
    ! was in the file (else, with a taper, it is recomputed from b_ref and dza on the tile and its
    ! ring, as bref_build made it: files written with the setup lag before CPU 72539140; exact only
    ! when the restart step rebuilds b_ref, logged)
    real(8), intent(in) :: sc(imex_nsc)
    logical, intent(in) :: n2raw_read, obcm_read
    integer(ip) :: i, j, k, nx, ny, nz
    if (imex_warm > 0) then
      if (stage_on) then
        call halo_fill_3d(hs1, STAGGER_RHO)
        call halo_fill_3d(hs2, STAGGER_RHO)
      else
        call halo_fill_3d(hsol(:,:,:,1), STAGGER_RHO)
        call halo_fill_3d(hsol(:,:,:,2), STAGGER_RHO)
      endif
    endif
    nh = nint(sc(5:6))
    ps_mver = -1; rst_mvalid = 0
    if (imex_rst_has('imex_mxp')) then
      rst_mvalid = nint(sc(7:8)); ps_mc2 = sc(9:10)
    endif
    call halo_fill_3d(bref_sav, STAGGER_RHO)
    have_bref = .true.
    bref_last = ntstart - 1 - nint(sc(3))   ! the same age at the next step
    if (lag_on) then
      call halo_fill_3d(hz_op, STAGGER_RHO)
      call halo_fill_3d(dza, STAGGER_RHO)
    else
      eta_old = (nint(sc(4)) == 1)
    endif
    if (imex_n2_taper > 0.0_rp) then
      if (n2raw_read) then
        call halo_fill_3d(ad_n2raw, STAGGER_RHO)
      else
        if (talk) write(*,'(1x,A)') 'IMEX_IGW: no imex_n2raw in the initial file: N2 of b_ref'// &
             ' recomputed (exact only at a b_ref rebuild)'
        nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do k = 1, nz
          do j = 0, ny + 1
            do i = 0, nx + 1
              ad_n2raw(i, j, k) = dza(i, j, k) * (bref_sav(i, j, min(k + 1, nz)) - bref_sav(i, j, max(k - 1, 1_ip)))
            enddo
          enddo
        enddo
      endif
    endif
    ! 'modal' reference (CPU imex_rst_get): taken over by the first obm_explicit; the ramp count continues
    ! (0, a file without it: past the ramp); without imex_obcm m restarts from the current mismatch
    ob_rst = .false.
    if (imex_obm_rst_may('imex_obcm')) then
      if (obcm_read) then
        ob_rst = .true.
        ob_mn = nint(sc(17))
        if (ob_mn <= 0) ob_mn = 1000000000 - 1
      else if (talk .and. mod(nint(sc(16)) / 100, 10) == 1) then
        write(*,'(1x,A)') "IMEX_IGW: no imex_obcm in the initial file: the 'modal' reference starts from the" &
             // ' current mismatch'
      endif
    endif
    rst_read = .true.; rst_pend = .true.
    if (talk) write(*,'(1x,A,I4,A,2I3)') 'IMEX_IGW: IMEX state read from the initial file (exact restart);'// &
         ' b_ref age', nint(sc(3)), ' steps, warm-start history', nh
  end subroutine imex_rst_loaded

  subroutine rst_replay()
    ! imex_setup_lag = 1, first imex_step_setup after an exact restart: the operator of the last
    ! b_ref rebuild step rebuilt from that step's inputs, as that step built it (the CPU's
    ! imex_rst_replay): Hz := hz_op for the call, rshu, rshv from it, dza as read, adj_setup
    ! with b_ref from bref_sav and no rescale of the memory terms (eta_old false; ad_n2o := its
    ! n2t); then a new coef_version, and the stored W M x of a stage valid for it when it was
    ! for the operator in use when written. bw0 (b_ref build scratch) holds the step's Hz.
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je
    integer :: s
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call copy3(Hz, bw0)
    call copy3(hz_op, Hz)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        rshu(i, j) = 0.0_rp
        do k = 1, nz
          rshu(i, j) = rshu(i, j) + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
        enddo
        rshu(i, j) = 1.0_rp / rshu(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        rshv(i, j) = 0.0_rp
        do k = 1, nz
          rshv(i, j) = rshv(i, j) + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
        enddo
        rshv(i, j) = 1.0_rp / rshv(i, j)
      enddo
    enddo
    CUF_DO2
    do j = js, je
      do i = is, ie
        rshu_op(i, j) = rshu(i, j)
        rshv_op(i, j) = rshv(i, j)
      enddo
    enddo
    eta_old = .false.; rst_on = .true.
    call adj_setup()
    rst_on = .false.
    call copy3(bw0, Hz)
    coef_version = coef_version + 1
    do s = 1, 2
      if (rst_mvalid(s) == 1) ps_mver(s) = coef_version
    enddo
  end subroutine rst_replay

end module imex_igw

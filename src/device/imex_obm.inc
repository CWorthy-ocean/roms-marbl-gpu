  !======================================================================
  ! imex_obc_mode = 'modal': the characteristic (per-mode Flather) open
  ! boundary of the internal modes. GPU port of the CPU's obm_* routines
  ! (UCLA ROMS imex-modal 75d3b58b, src/imex_igw.F90; physics in the CPU
  ! header "imex_obc_mode = 'modal'" and IMEX_MODAL_OBC_DESIGN.md). Per open
  ! face column: u_b = u_ext - <u_ext> + s Y (P(b_cur - b_ext) - m),
  ! Y g = y_rem (g - <g>) + sum_m a_m phi_m <phi_m, g>, <.> the hf-weighted
  ! mean; implicit in the stage solves through the face transport
  ! F_b = hf dn (Y g)/(theta dt) in every operator sweep and its Jacobi term.
  !
  ! Layout: the open-edge face columns of this tile in one packed list
  ! c = 1..ob_nc (edges in the order W, E, S, N, as the CPU's loops; every
  ! column of an open physical edge, active or not; ob_act(c) = 1 on the
  ! active ones, set by the build); per column the cell (ob_ci, ob_cj), the
  ! face (ob_fi, ob_fj), the ghost cell beyond it (ob_gi, ob_gj), the edge
  ! ob_ce and the along-edge index ob_cl (j at W/E, i at S/N); the tile's
  ! cells next to an open face (ob_pi, ob_pj; ob_pc(e, p) the column of edge
  ! e at that cell, 0 if none) for the Jacobi term.
  !
  ! Kernels, one thread per column (the CPU routine in brackets):
  ! - obm_build_k: at each b_ref rebuild (end of adj_setup) the modes with the
  !   few-modes solver (obm_modes_d, obm_sturm_d, obm_twist_d = obm_modes,
  !   obm_sturm, obm_twist with one shift per bracket and pass), the weights,
  !   admittances and the closed-form Jacobi column (obm_kd_d = obm_kd_col)
  !   [obm_build, the CPU's few-modes solver];
  ! - obm_face_k: per operator sweep, after the face kernel has stored g at
  !   the open faces: g := (Y g)/sqrt(c2), mean 0; the cell kernel forms F
  !   with the face coefficient cxm, cym (= cxo, cyo, and dn_u, dm_v at the
  !   active open faces) [obm_ufx, obm_vfx];
  ! - obm_expl_k: start of a stage, the explicit face values and the
  !   reference m; in the predictor also the face write into u, v(nnew)
  !   [obm_explicit + imex_obc_faces of imex_obc_pre];
  ! - obm_corr_k: after u3dbc/v3dbc of pre_step3d (predictor, with the
  !   barotropic mean) and of step3d_uv2 (corrector): the face increment of
  !   the stage solution and the face write [obm_correct + imex_obc_faces];
  ! - the Jacobi term in pf_setup8/4 (per stage and coefficient set), over
  !   the cells next to open faces; the diagnostics (OBCM, OBCF lines, the
  !   start note) through imex_red's deterministic sums;
  ! - the density of both sides of the mismatch d through the model's EOS
  !   (obm_rho_d = rho_eos's rho at one point, the CPU's rho_eos_point).
  ! Work arrays (fixed per column, no automatic arrays in device code):
  ! ob_sc(nz, 6, nc), ob_z(nz, m, 4, nc), ob_wr(m, 9, nc), ob_wi(m, 5, nc).
  !
  ! Recommended use of 'modal' (the CPU's; nest and Pacific tests, IMEX_OBC_REMAINING.md sec. 6.6):
  ! - defaults: imex_obc_mode 'auto' (selects 'modal'), imex_obc_nmodes 3, imex_obc_cmin 0.3 m/s,
  !   imex_obc_tref 3 d;
  ! - imex_obc_q0 is off by default; where the semidiurnal tide is close to inertial (f/omega0 > 0.6) it
  !   has been set to sqrt(1 - f^2/omega_M2^2). It returns part of a Kelvin wave that leaves along a
  !   coast (13 % at 0.5);
  ! - sponge: a nest whose boundary data carry the internal tide runs best with a quarter of the v_sponge
  !   used with the stock boundary (the sponge damps the tide that comes in); an outermost domain, whose
  !   boundary data have no internal tide, keeps the stock value. Keep v_sponge dt/dx^2 <= 0.1;
  ! - SPONGE_TUNE's ub_tune has no effect on the normal velocity under 'modal': leave it off;
  ! - start: a run that does not begin from the state its boundary data come from (a nest not initialised
  !   from its parent) should take its first ~10 days with imex_obc_mode = 'fixed' and switch to 'modal'
  !   at a restart (inexact by design: m restarts from the mismatch of that day); see obm_startnote;
  ! - step: the limit is explicit horizontal advection, not the boundary. Keep the horizontal part of the
  !   Courant number below about 0.8 over the whole run;
  ! - known limits: a wave that the boundary data do not contain and that leaves at more than about 60
  !   degrees from the normal returns at the rate of any one-direction condition (36 % at 75 degrees);
  !   the child's depths must match the parent's along the open edges.
  !======================================================================

  subroutine obm_decide()
    ! Once, at the first IMEX step (the open sides and the boundary data are
    ! known; init_bry runs after the initial file is read): 'modal' or 'fixed'
    ! as the CPU (check_settings): 'auto' selects 'modal' with an open edge
    ! and 3-D boundary data (the CPU's M3_FRC_BRY and T_FRC_BRY: normal
    ! velocity, temperature and salinity on every open side). The GPU's own
    ! conditions (check_settings, obm_gpu_ok): with 'auto' a NOTE and
    ! 'fixed', with 'modal' a stop. A restart whose IMEX state was written
    ! with another boundary mode is turned into the inexact restart, as the
    ! CPU does (rst_mode, obc_mode_change).
    use gpu_state, only: obc_west, obc_east, obc_south, obc_north, obc_3d
    logical :: opn, d3, cpu_modal
    character(len=256) :: why
    if (obm_decided) return
    obm_decided = .true.
    if (.not. (imex_on .and. imex_solver_enabled)) return
    opn = obc_west .or. obc_east .or. obc_south .or. obc_north
    d3 = opn
    if (obc_west .and. .not. obc_3d(1)) d3 = .false.
    if (obc_east .and. .not. obc_3d(2)) d3 = .false.
    if (obc_south .and. .not. obc_3d(3)) d3 = .false.
    if (obc_north .and. .not. obc_3d(4)) d3 = .false.
    cpu_modal = trim(imex_obc_mode) == 'modal' .or. (trim(imex_obc_mode) == 'auto' .and. d3)
    imex_obc_modal = .false.
    if (cpu_modal) then
      if (trim(imex_obc_mode) == 'modal' .and. .not. d3) then
        if (talk) write(*,'(a)') "ERROR: &imex_igw_settings: imex_obc_mode = 'modal' needs an open edge with 3-D" &
             // ' boundary data (normal velocity, temperature, salinity) on every open side'
        error stop 'imex_igw: unsupported &imex_igw_settings (see above)'
      endif
      if (obm_gpu_ok) then
        imex_obc_modal = .true.
      else if (trim(imex_obc_mode) == 'modal') then
        if (talk) write(*,'(a)') "ERROR: &imex_igw_settings: 'modal' on the GPU needs the fused solve" &
             // ' (imex_pcg_start 2 or 3, imex_pcg_fused 1, imex_warm 1)' &
             // ' with imex_gpu_stage 1 and imex_gpu_sweep 1'
        error stop 'imex_igw: unsupported &imex_igw_settings (see above)'
      else
        if (talk) write(*,'(1x,a)') "IMEX_IGW NOTE: imex_obc_mode = 'auto' runs 'fixed' on the GPU with these" &
             // " settings ('modal' needs imex_pcg_start 2 or 3 with the fused solve, imex_warm 1," &
             // " imex_gpu_stage 1, imex_gpu_sweep 1); the CPU selects 'modal' for this namelist, so the" &
             // " two codes differ"
      endif
    endif
    if (talk) then
      if (imex_obc_modal) then
        write(*,'(1x,a,a,a,i3,a,f7.4,a,f10.1,a)') "IMEX_IGW: imex_obc_mode = '", trim(imex_obc_mode), &
             "' -> 'modal': imex_obc_nmodes =", imex_obc_nmodes, '  imex_obc_q0 =', imex_obc_q0, &
             '  imex_obc_tref =', imex_obc_tref, ' s'
        if (imex_obc_q0 > 0.0_rp) write(*,'(3x,a,f7.4,a,f7.4,a)') 'imex_obc_q0: admittances 1/(q0 c_m) =', &
             1.0_rp / imex_obc_q0, ' / c_m (omega0 =', 1.0_rp / sqrt(1.0_rp - imex_obc_q0**2), ' f)'
        if (imex_obc_cmin > 0.0_rp) write(*,'(3x,a,f8.4,a)') 'imex_obc_cmin =', imex_obc_cmin, &
             ' m/s (cap on the admittances)'
        write(*,'(3x,a)') 'mode build: few-modes solver, closed-form Jacobi; one device thread per face column'
      else
        write(*,'(1x,a,a,a)') "IMEX_IGW: imex_obc_mode = '", trim(imex_obc_mode), "' -> 'fixed'"
      endif
    endif
    ! the restart's IMEX state (accepted in imex_rst_accept without the boundary part of the mode)
    if (rst_read .and. rst_md_file /= rst_mode()) then
      why = obc_mode_change(rst_md_file, rst_mode())
      if (talk) write(*,'(1x,a)') 'IMEX_IGW: IMEX settings differ from the run that wrote it ('//trim(why)// &
           '): the IMEX stores start as on a first step (inexact restart)'
      call obm_rst_revert()
    endif
  end subroutine obm_decide

  subroutine obm_rst_revert()
    ! the restart's IMEX state is not used (obm_decide): as if imex_rst_accept had refused it
    integer(ip) :: i, j, k, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    rst_read = .false.; rst_pend = .false.
    nh = 0; ps_mver = -1; rst_mvalid = 0
    have_bref = .false.; bref_last = -100000000; eta_old = .false.
    ob_rst = .false.; ob_mn = 0
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          dpw(i, j, k) = 0.0_rp
          hsol(i, j, k, 1) = 0.0_rp
          hsol(i, j, k, 2) = 0.0_rp
        enddo
      enddo
    enddo
    if (stage_on) then
      call zero3(hs1)
      call zero3(hs2)
    endif
  end subroutine obm_rst_revert

  !----------------------------------------------------------------------
  subroutine obm_init()
    ! first build: the packed list of the open-edge face columns, the cells
    ! next to them, the face maps, the arrays (host integer arithmetic only)
    use gpu_state, only: obc_west, obc_east, obc_south, obc_north
    integer(ip) :: nx, ny, nz, c, e, l, n, p, i, j
    integer(ip), allocatable :: pid(:,:)
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ob_edge = [decomp%edge_w .and. obc_west, decomp%edge_e .and. obc_east, &
               decomp%edge_s .and. obc_south, decomp%edge_n .and. obc_north]
    ob_m = min(imex_obc_nmodes, int(nz) - 1)
    n = 0
    if (ob_edge(1)) n = n + ny
    if (ob_edge(2)) n = n + ny
    if (ob_edge(3)) n = n + nx
    if (ob_edge(4)) n = n + nx
    ob_nc = int(n)
    if (n > nx * ny) then
      write(*,'(a,i0,a,i0,a,i0)') 'ERROR: imex_obc_mode modal: rank ', comm_world_rank(), ': tile ', nx, ' x ', ny
      error stop 'imex_igw: the modal open boundary needs tiles with more cells than open-edge faces'
    endif
    call red_ensure(nx * ny)
    n = max(n, 1_ip)
    call sync_arch()
    if (allocated(ob_ub)) deallocate(ob_ub)
    if (allocated(ob_act)) deallocate(ob_act)
    allocate(ob_ce(n), ob_cl(n), ob_ci(n), ob_cj(n), ob_fi(n), ob_fj(n), ob_gi(n), ob_gj(n))
    allocate(ob_act(n), ob_yr(n), ob_w(nz, n), ob_hn(nz, n), ob_ub(nz, n), ob_kdc(nz, n))
    allocate(ob_a(ob_m, n), ob_c(ob_m, n), ob_phi(nz, ob_m, n), ob_mr(nz, n), ob_dd(nz, n))
    allocate(ob_gk(0:nz, n), ob_sc(nz, 6, n), ob_z(nz, ob_m, 4, n), ob_wr(ob_m, 9, n), ob_wi(ob_m, 5, n))
    allocate(ob_dg(4, n), ob_pd(16, n))
    ob_ce = 0; ob_cl = 0; ob_ci = 1; ob_cj = 1; ob_fi = 1; ob_fj = 1; ob_gi = 1; ob_gj = 1
    ob_act = 0.0_rp; ob_yr = 0.0_rp; ob_w = 0.0_rp; ob_hn = 0.0_rp; ob_ub = 0.0_rp; ob_kdc = 0.0_rp
    ob_a = 0.0_rp; ob_c = 0.0_rp; ob_phi = 0.0_rp; ob_mr = 0.0_rp; ob_dd = 0.0_rp
    ob_gk = 0.0_rp; ob_sc = 0.0_rp; ob_z = 0.0_rp; ob_wr = 0.0_rp; ob_wi = 0; ob_dg = 0; ob_pd = 0.0_rp
    c = 0
    do e = 1, 4
      if (.not. ob_edge(e)) cycle
      do l = 1, merge(ny, nx, e <= 2)
        c = c + 1
        ob_ce(c) = e; ob_cl(c) = l
        select case (e)
        case (1)
          i = 1; j = l; ob_fi(c) = 1; ob_fj(c) = l; ob_gi(c) = 0; ob_gj(c) = l
        case (2)
          i = nx; j = l; ob_fi(c) = nx + 1; ob_fj(c) = l; ob_gi(c) = nx + 1; ob_gj(c) = l
        case (3)
          i = l; j = 1; ob_fi(c) = l; ob_fj(c) = 1; ob_gi(c) = l; ob_gj(c) = 0
        case default
          i = l; j = ny; ob_fi(c) = l; ob_fj(c) = ny + 1; ob_gi(c) = l; ob_gj(c) = ny + 1
        end select
        ob_ci(c) = i; ob_cj(c) = j
      enddo
    enddo
    ! the cells next to an open face, each once, with its faces' columns
    allocate(pid(nx, ny)); pid = 0
    p = 0
    do c = 1, ob_nc
      if (pid(ob_ci(c), ob_cj(c)) == 0) then
        p = p + 1; pid(ob_ci(c), ob_cj(c)) = p
      endif
    enddo
    ob_np = int(p)
    allocate(ob_pi(max(p, 1_ip)), ob_pj(max(p, 1_ip)), ob_pc(4, max(p, 1_ip)))
    ob_pi = 1; ob_pj = 1; ob_pc = 0
    do c = 1, ob_nc
      p = pid(ob_ci(c), ob_cj(c))
      ob_pi(p) = ob_ci(c); ob_pj(p) = ob_cj(c); ob_pc(ob_ce(c), p) = c
    enddo
    deallocate(pid)
    ! face maps (rhs_transport) and the column of each face
    do c = 1, ob_nc
      if (ob_ce(c) <= 2) then
        obcol_u(ob_fi(c), ob_fj(c)) = c
      else
        obcol_v(ob_fi(c), ob_fj(c)) = c
      endif
    enddo
  end subroutine obm_init

  !----------------------------------------------------------------------
  subroutine obm_build()
    ! end of adj_setup (b_ref, the step's Hz - hz_op in rst_replay -, the
    ! coefficients of D): CPU obm_build (the few-modes solver). The printed
    ! statistics as the CPU's (min/max of c_m over the domain, failed columns;
    ! first build: the reference filter and sponge notes).
    use gpu_state, only: f, visc2_r
    use pflx_mod, only: ub_tune
    integer(ip) :: c, i, j, nx, ny, m
    integer :: nb
    real(rp) :: q, ycap, cmn(OBM_MMAX), cmx(OBM_MMAX), fmn, sdn, wt, hres
    real(8) :: lmn(OBM_MMAX + 2), gmn(OBM_MMAX + 2), lmx(OBM_MMAX + 1), gmx(OBM_MMAX + 1), ls(1), gs(1)
#ifdef USE_CUDA
    integer :: nblk
#endif
    if (.not. imex_obc_modal) return
    if (.not. allocated(ob_ce)) call obm_init()
    nx = decomp%nx; ny = decomp%ny
    q = 1.0_rp
    if (imex_obc_q0 > 0.0_rp) q = 1.0_rp / imex_obc_q0
    ycap = 0.0_rp
    if (imex_obc_cmin > 0.0_rp) ycap = 1.0_rp / imex_obc_cmin
    if (ob_nc > 0) then
#ifdef USE_CUDA
      nblk = (ob_nc + OBM_NT - 1) / OBM_NT
      call obm_build_k<<<nblk, OBM_NT, 0, work_stream>>>(ob_nc, int(decomp%nz), ob_m, imex_n2_ref, imex_n2_reg, &
           q, imex_obc_cmin, ycap)
      call obm_cxm_k<<<nblk, OBM_NT, 0, work_stream>>>(ob_nc)
#else
      error stop 'obm_build: CUDA build required'
#endif
    endif
    ob_nbuild = ob_nbuild + 1
    if (ob_nbuild == 1 .or. imex_verbose >= 1) then
      ! (collective: adj_setup is) the active columns, min/max c_m, columns without convergence
      call sync_arch()
      m = ob_m
      cmn = huge(1.0_rp); cmx = -huge(1.0_rp); nb = 0; fmn = huge(1.0_rp)
      ob_nact = 0
      do c = 1, ob_nc
        if (ob_act(c) <= 0.0_rp) cycle
        ob_nact = ob_nact + 1
        cmn(1:m) = min(cmn(1:m), ob_c(1:m, c)); cmx(1:m) = max(cmx(1:m), ob_c(1:m, c))
        if (ob_dg(4, c) /= 1) nb = nb + 1
        fmn = min(fmn, abs(f(ob_ci(c), ob_cj(c))))
      enddo
      lmn(1:m) = cmn(1:m); lmn(m + 1) = fmn; lmn(m + 2) = 0.0d0
      lmx(1:m) = cmx(1:m); lmx(m + 1) = dble(nb)
      call comm_allreduce_min(m + 2, lmn(1:m + 2), gmn(1:m + 2))
      call comm_allreduce_max(m + 1, lmx(1:m + 1), gmx(1:m + 1))
      ls(1) = dble(nb)
      call comm_allreduce_sum(1, ls, gs)
      if (talk) then
        write(*,'(1x,A,I7,A,I3,A,20(F8.4,A,F8.4))') 'IMEX_OBCM', iic, ' modal build', m, ' modes, c_m' &
             // ' min-max [m/s]:', (gmn(i), '-', gmx(i), i = 1, m)
        if (gs(1) > 0.5d0) write(*,'(1x,A,I6,A)') 'IMEX_IGW WARNING: modal OBC eigen-solver not converged in', &
             nint(gs(1)), ' face columns'
      endif
      if (ob_nbuild == 1) then
        ! the explicit (sponge) viscosity's diffusion number visc2 dt / min(dx, dy)^2 (CPU first-build note)
        sdn = 0.0_rp
        do j = 1, ny
          do i = 1, nx
            sdn = max(sdn, visc2_r(i, j) * dt * max(pm(i, j), pn(i, j))**2)
          enddo
        enddo
        ls(1) = sdn
        call comm_allreduce_max(1, ls, gs)
        if (talk .and. imex_obc_tref > 0.0_rp .and. gs(1) > 0.1d0) write(*,'(1x,A,F6.3,A)') &
             'IMEX_IGW NOTE: the explicit viscosity (sponge) reaches visc2 dt/dx^2 =', gs(1), ' (> 0.1): with the' &
             // ' modal reference an edge mode can grow at a rate ~1/tref (N3 Hawaii nest: 0.17 blew up for tref 6 h' &
             // ' to 3 d, 0.085 ran 29 d); reduce v_sponge or dt'
        if (talk .and. imex_obc_tref > 0.0_rp) then
          wt = gmn(m + 1) * imex_obc_tref
          hres = 1.0_rp / sqrt(1.0_rp + wt**2)
          if (gmn(m + 1) > 0.0d0) then
            write(*,'(1x,A,F9.2,A,F9.3,A,F7.4,A)') 'IMEX_OBCM reference: longest inertial period on the' &
                 // ' open edges', 2.0_rp * acos(-1.0_rp) / gmn(m + 1) / 86400.0_rp, ' d; tref', &
                 imex_obc_tref / 86400.0_rp, ' d: |H| at that frequency', hres, &
                 ' (rule: <= 0.053, tref >= 3 inertial periods)'
          else
            write(*,'(1x,A)') 'IMEX_OBCM reference: f = 0 on an open edge (no inertial period)'
          endif
          if (imex_obc_tref < obm_tramp) write(*,'(1x,A)') 'IMEX_IGW NOTE: imex_obc_tref is below 1 d: a' &
               // ' reference that follows the mismatch within hours stops radiating edge transients (N3 Hawaii' &
               // ' nest: an edge mode grew at ~1/tref, tref 6 h blew up in 0.8 d); see the key description'
          if (hres > 0.0531_rp) write(*,'(1x,A)') 'IMEX_IGW NOTE: the reference filter holds more than 5 % of' &
               // ' the near-inertial band at the lowest |f| of the open edges: a longer imex_obc_tref' &
               // ' radiates it; see the key description'
        endif
        if (talk .and. ub_tune) write(*,'(1x,A)') "IMEX_IGW NOTE: SPONGE_TUNE ub_tune is active with" &
             // " imex_obc_mode 'modal': the modal condition sets the normal velocity at the open faces, so the" &
             // " tuned nudging acts on the tangential velocity and the tracers only (ub may drift to its clamp);" &
             // " ub_tune = .false. is recommended"
      endif
    endif
  end subroutine obm_build

#ifdef USE_CUDA
  attributes(global) subroutine obm_build_k(nc, nz, m, n2ref, n2reg, q, cmin, ycap)
    ! one thread per face column: CPU obm_build's column loop (the few-modes solver)
    integer, value :: nc, nz, m
    real(rp), value :: n2ref, n2reg, q, cmin, ycap
    integer :: c, e, i, j, fi, fj, gi, gj, k, mm
    real(rp) :: hh, dz, n2, ym, yrem
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    e = ob_ce(c); i = ob_ci(c); j = ob_cj(c); fi = ob_fi(c); fj = ob_fj(c); gi = ob_gi(c); gj = ob_gj(c)
    ob_act(c) = 0.0_rp
    do k = 1, nz
      ob_kdc(k, c) = 0.0_rp
    enddo
    if (act(i, j) <= 0.0_rp) return
    if (e <= 2) then
      if (cx(fi, fj) <= 0.0_rp) return
    else
      if (cy(fi, fj) <= 0.0_rp) return
    endif
    ob_act(c) = 1.0_rp
    hh = 0.0_rp
    do k = 1, nz
      ob_sc(k, 1, c) = 0.5_rp * (Hz(i, j, k) + Hz(gi, gj, k)); hh = hh + ob_sc(k, 1, c)
    enddo
    ob_gk(0, c) = 0.0_rp; ob_gk(nz, c) = 0.0_rp
    do k = 1, nz - 1
      dz = 0.5_rp * (Hz(i, j, k) + Hz(i, j, k + 1))
      n2 = max((ad_bb(i, j, k + 1) - ad_bb(i, j, k)) / dz, n2ref, n2reg)
      ob_gk(k, c) = 1.0_rp / (n2 * dz)
    enddo
    call obm_modes_d(nz, m, ob_sc(1, 1, c), ob_gk(0, c), ob_c(1, c), ob_phi(1, 1, c), &
         ob_wr(1, 1, c), ob_wr(1, 2, c), ob_wr(1, 3, c), ob_wr(1, 4, c), ob_wr(1, 5, c), ob_wr(1, 6, c), &
         ob_wi(1, 1, c), ob_wi(1, 2, c), ob_wi(1, 3, c), ob_wi(1, 4, c), ob_wi(1, 5, c), &
         ob_z(1, 1, 1, c), ob_z(1, 1, 2, c), ob_z(1, 1, 3, c), ob_z(1, 1, 4, c), ob_dg(1, c), ob_dg(2, c), &
         ob_dg(3, c), ob_wr(1, 9, c), ob_dg(4, c))
    do k = 1, nz
      ob_w(k, c) = ob_sc(k, 1, c) / hh
      if (e <= 2) then
        ob_hn(k, c) = ob_sc(k, 1, c) * dn_u(fi, fj)
      else
        ob_hn(k, c) = ob_sc(k, 1, c) * dm_v(fi, fj)
      endif
    enddo
    yrem = q / ob_c(m, c)
    if (cmin > 0.0_rp) yrem = min(yrem, ycap)            ! cap on the admittance itself
    ob_yr(c) = yrem
    do mm = 1, m
      ym = q / ob_c(mm, c)
      if (cmin > 0.0_rp) ym = min(ym, ycap)
      ob_a(mm, c) = ym - yrem
    enddo
    do k = 1, nz                                  ! the cell's and face's coefficients for the Jacobi term
      ob_sc(k, 2, c) = -ad_act(i, j) / ad_n2k(i, j, k)
      ob_sc(k, 3, c) = ad_czt(i, j, k)
      ob_sc(k, 4, c) = ad_czb(i, j, k)
      if (e <= 2) then
        ob_sc(k, 5, c) = ad_cu(fi, fj, k)
      else
        ob_sc(k, 5, c) = ad_cv(fi, fj, k)
      endif
      ob_sc(k, 6, c) = ad_en2(i, j, k)
    enddo
    call obm_kd_d(nz, m, merge(1.0_rp, -1.0_rp, e == 2 .or. e == 4), ob_yr(c), ob_w(1, c), ob_hn(1, c), ob_a(1, c), &
         ob_phi(1, 1, c), ob_sc(1, 2, c), ob_wr(1, 7, c), ob_wr(1, 8, c), ob_kdc(1, c))
  end subroutine obm_build_k

  attributes(global) subroutine obm_cxm_k(nc)
    ! the face coefficient of the operator sweep's cell kernel: dn_u, dm_v at the active open faces
    integer, value :: nc
    integer :: c, fi, fj
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    fi = ob_fi(c); fj = ob_fj(c)
    if (ob_ce(c) <= 2) then
      cxm(fi, fj) = cxo(fi, fj)
      if (ob_act(c) > 0.0_rp) cxm(fi, fj) = dn_u(fi, fj)
    else
      cym(fi, fj) = cyo(fi, fj)
      if (ob_act(c) > 0.0_rp) cym(fi, fj) = dm_v(fi, fj)
    endif
  end subroutine obm_cxm_k

  attributes(device) subroutine obm_modes_d(n, m, hf, gk, cm, phi, lo, hi, sg, s, lam, gam, clo, chi, cnt, iact, &
                                            iconv, z, wp, wq, wu, nps, nrq, nfb, gmin, iok)
    ! CPU obm_modes with OBM_NSEC = 1 (one shift per unresolved bracket and pass), work arrays from the
    ! caller; iact, iconv, iok as integers (1 true); the scalar results in the caller's arrays. The m
    ! slowest baroclinic modes of the column: the
    ! eigenpairs 2..m+1 of B phi = lam W phi, c = 1/sqrt(lam), sum_k hf phi^2 = sum_k hf, phi(n) >= 0.
    integer, value :: n, m
    real(rp) :: hf(n), gk(0:n), cm(m), phi(n, m), lo(m), hi(m), sg(m), s(m), lam(m), gam(m)
    integer :: clo(m), chi(m), cnt(m), iact(m), iconv(m)
    real(rp) :: z(n, m), wp(n, m), wq(n, m), wu(n, m)
    integer :: nps, nrq, nfb, iok
    real(rp) :: gmin
    real(rp) :: hh, rs, lb, ub, nw, dl, sn, pr, fr
    integer :: j, jj, k, it, ns, i, nact
    ! (the bounds OBM_MBIS 160, OBM_MRQ 12, tolerances OBM_TOLB 1e-2, OBM_TOLG 1e-6 and OBM_NSEC 1 are module
    ! parameters: local named constants of an inlined device routine made nvfortran 25.9 fail, llgetsym)
    iok = 1; nps = 0; nrq = 0; nfb = 0
    hh = 0.0_rp; rs = 0.0_rp; ub = 0.0_rp
    do k = 1, n
      hh = hh + hf(k)
      ub = max(ub, 2.0_rp * (gk(k - 1) + gk(k)) / hf(k))
    enddo
    do k = 1, n - 1
      rs = rs + 1.0_rp / gk(k)
    enddo
    lb = 0.5_rp / (hh * rs)
    ub = 2.0_rp * ub
    do j = 1, m
      lo(j) = lb; hi(j) = ub; clo(j) = 1; chi(j) = n
    enddo
    do it = 1, OBM_MBIS
      ns = 0
      do j = 1, m
        if (clo(j) == j .and. chi(j) == j + 1 .and. hi(j) - lo(j) <= OBM_TOLB * hi(j)) cycle
        do i = 1, OBM_NSEC
          fr = real(i, rp) / real(OBM_NSEC + 1, rp)
          if (hi(j) > 2.0_rp * lo(j)) then
            sn = lo(j) * (hi(j) / lo(j))**fr
          else
            sn = lo(j) + fr * (hi(j) - lo(j))
          endif
          if (sn > lo(j) .and. sn < hi(j)) then
            ns = ns + 1; sg(ns) = sn
          endif
        enddo
      enddo
      if (ns == 0) exit
      nps = nps + 1
      do i = 1, ns
        cnt(i) = obm_sturm_d(n, hf, gk, sg(i))
      enddo
      do i = 1, ns
        do jj = 1, m
          if (cnt(i) >= jj + 1) then
            if (sg(i) < hi(jj)) then
              hi(jj) = sg(i); chi(jj) = cnt(i)
            endif
          else
            if (sg(i) > lo(jj)) then
              lo(jj) = sg(i); clo(jj) = cnt(i)
            endif
          endif
        enddo
      enddo
    enddo
    do j = 1, m
      s(j) = 0.5_rp * (lo(j) + hi(j)); iconv(j) = 0; gam(j) = 0.0_rp; lam(j) = s(j)
    enddo
    do it = 1, OBM_MRQ
      nact = 0
      do j = 1, m
        iact(j) = 1 - iconv(j); nact = nact + iact(j)
      enddo
      if (nact == 0) exit
      nrq = nrq + 1
      call obm_twist_d(n, m, hf, gk, s, iact, cnt, gam, z, wp, wq, wu)
      do j = 1, m
        if (iact(j) == 0) cycle
        if (cnt(j) >= j + 1) then
          hi(j) = min(hi(j), s(j))
        else
          lo(j) = max(lo(j), s(j))
        endif
        nw = 0.0_rp
        do k = 1, n
          nw = nw + hf(k) * z(k, j) * z(k, j)
        enddo
        dl = gam(j) / nw
        if (abs(dl) <= 16.0_rp * epsilon(1.0_rp) * s(j)) then
          iconv(j) = 1; lam(j) = s(j) + dl
        else
          sn = s(j) + dl
          if (sn <= lo(j) .or. sn >= hi(j)) sn = 0.5_rp * (lo(j) + hi(j))
          s(j) = sn
        endif
      enddo
    enddo
    do j = 1, m
      if (iconv(j) == 1) cycle                     ! bisection to the rounding level, vector at the midpoint
      nfb = nfb + 1
      do it = 1, OBM_MBIS
        sn = 0.5_rp * (lo(j) + hi(j))
        if (sn <= lo(j) .or. sn >= hi(j)) exit
        sg(1) = sn
        cnt(1) = obm_sturm_d(n, hf, gk, sg(1))
        if (cnt(1) >= j + 1) then
          hi(j) = sn
        else
          lo(j) = sn
        endif
      enddo
      do jj = 1, m
        iact(jj) = 0
      enddo
      iact(j) = 1; s(j) = 0.5_rp * (lo(j) + hi(j))
      call obm_twist_d(n, m, hf, gk, s, iact, cnt, gam, z, wp, wq, wu)
      nw = 0.0_rp
      do k = 1, n
        nw = nw + hf(k) * z(k, j) * z(k, j)
      enddo
      lam(j) = s(j) + gam(j) / nw
    enddo
    gmin = 1.0_rp
    do j = 1, m
      nw = 0.0_rp
      do k = 1, n
        nw = nw + hf(k) * z(k, j) * z(k, j)
      enddo
      if (.not. (nw > 0.0_rp .and. nw <= huge(1.0_rp))) then
        iok = 0; nw = hh
        do k = 1, n
          z(k, j) = 0.0_rp
        enddo
      endif
      nw = sqrt(hh / nw)
      do k = 1, n
        phi(k, j) = z(k, j) * nw
      enddo
      do jj = 1, j - 1                            ! nearly equal eigenvalues: W-orthogonal to the earlier
        gmin = min(gmin, abs(lam(j) - lam(jj)) / lam(j))
        if (abs(lam(j) - lam(jj)) > OBM_TOLG * lam(j)) cycle
        pr = 0.0_rp
        do k = 1, n
          pr = pr + hf(k) * phi(k, j) * phi(k, jj)
        enddo
        pr = pr / hh
        do k = 1, n
          phi(k, j) = phi(k, j) - pr * phi(k, jj)
        enddo
        nw = 0.0_rp
        do k = 1, n
          nw = nw + hf(k) * phi(k, j)**2
        enddo
        if (.not. (nw > 0.0_rp)) then
          iok = 0; nw = hh
        endif
        nw = sqrt(hh / nw)
        do k = 1, n
          phi(k, j) = phi(k, j) * nw
        enddo
      enddo
      if (phi(n, j) < 0.0_rp) then
        do k = 1, n
          phi(k, j) = -phi(k, j)
        enddo
      endif
      cm(j) = 1.0_rp / sqrt(max(lam(j), 1.e-30_rp))
    enddo
  end subroutine obm_modes_d

  attributes(device) integer function obm_sturm_d(n, hf, gk, sh)
    ! CPU obm_sturm for one shift sh (the CPU runs its ns shifts in one loop over k; each shift's
    ! operations are the same): the number of generalized eigenvalues below sh, from the signs of the
    ! pivots of B - sh W in differential form; a zero pivot replaced by -eps gk(k). (A function: the
    ! subroutine form, called from obm_modes_d, made nvfortran 25.9 fail with an internal error.)
    integer, value :: n
    real(rp), value :: sh
    real(rp) :: hf(n), gk(0:n)
    real(rp) :: ts_, pv_
    integer :: k, cn
    ts_ = -sh * hf(1); cn = 0
    do k = 1, n - 1
      pv_ = gk(k) + ts_
      if (pv_ == 0.0_rp) pv_ = -epsilon(1.0_rp) * gk(k)
      if (pv_ < 0.0_rp) cn = cn + 1
      ts_ = -sh * hf(k + 1) + gk(k) * ts_ / pv_
    enddo
    if (ts_ < 0.0_rp) cn = cn + 1
    obm_sturm_d = cn
  end function obm_sturm_d

  attributes(device) subroutine obm_twist_d(n, m, hf, gk, s, iact, cnt, gam, z, tp, tq, tu)
    ! CPU obm_twist: top-down (p) and bottom-up (u) differential factorizations of B - s W, the twist
    ! index of the smallest |gamma_k|, gam, the vector z(:,j) (z(r,j) = 1), the Sturm count
    integer, value :: n, m
    real(rp) :: hf(n), gk(0:n), s(m), gam(m), z(n, m), tp(n, m), tq(n, m), tu(n, m)
    integer :: iact(m), cnt(m)
    real(rp) :: ga, gb
    integer :: j, k, kb, r
    do j = 1, m
      tp(1, j) = -s(j) * hf(1); tu(n, j) = -s(j) * hf(n)
    enddo
    do k = 1, n - 1
      kb = n - k
      do j = 1, m
        ga = tp(k, j) + gk(k)
        if (ga == 0.0_rp) ga = -epsilon(1.0_rp) * gk(k)
        tp(k + 1, j) = -s(j) * hf(k + 1) + gk(k) * tp(k, j) / ga
        gb = tu(kb + 1, j) + gk(kb)
        if (gb == 0.0_rp) gb = -epsilon(1.0_rp) * gk(kb)
        tu(kb, j) = -s(j) * hf(kb) + gk(kb) * tu(kb + 1, j) / gb
      enddo
    enddo
    do j = 1, m
      if (iact(j) == 0) cycle
      cnt(j) = 0; r = 1; gam(j) = huge(1.0_rp)
      do k = 1, n
        ga = tp(k, j) + tu(k, j) + s(j) * hf(k)
        if (abs(ga) < abs(gam(j))) then
          gam(j) = ga; r = k
        endif
        tq(k, j) = tu(k, j) + gk(k - 1)
        if (tq(k, j) == 0.0_rp) tq(k, j) = -epsilon(1.0_rp) * gk(k - 1)
        ga = tp(k, j) + gk(k)
        if (ga == 0.0_rp) ga = -epsilon(1.0_rp) * gk(k)
        tp(k, j) = ga
        if (ga < 0.0_rp) cnt(j) = cnt(j) + 1
      enddo
      z(r, j) = 1.0_rp
      do k = r - 1, 1, -1
        z(k, j) = gk(k) / tp(k, j) * z(k + 1, j)
      enddo
      do k = r + 1, n
        z(k, j) = gk(k - 1) / tq(k, j) * z(k - 1, j)
      enddo
    enddo
  end subroutine obm_twist_d

  attributes(device) subroutine obm_kd_d(nz, m, sg, yr, w, hn, a, phi, cq, sp, shp, kd)
    ! CPU obm_kd_col (cz_split): the Jacobi column of a face, kd(q) = (the boundary term of e_q)(q), in
    ! closed form from prefix sums. Column inputs (the kernel's copies) cq(:, 1) = -act/n2k, 2 czt,
    ! 3 czb, 4 the face's cu or cv, 5 en2; sp, shp work (m)
    integer, value :: nz, m
    real(rp), value :: sg, yr
    real(rp) :: w(nz), hn(nz), a(m), phi(nz, m), cq(nz, 5), sp(m), shp(m), kd(nz)
    real(rp) :: sw, sh, al, ct, cb, cf, g1, g2, gm, tq, fs, dm, fb, wm, ww, sv
    integer :: q, mm
    sw = 0.0_rp; sh = 0.0_rp
    do mm = 1, m
      sp(mm) = 0.0_rp; shp(mm) = 0.0_rp
    enddo
    do q = 1, nz
      al = cq(q, 1)
      ct = 0.0_rp; cb = 0.0_rp
      if (q < nz) ct = cq(q, 2)
      if (q > 1) cb = cq(q, 3)
      cf = cq(q, 4)
      g1 = sg * al * (cb - ct)                    ! g below q
      g2 = sg * al * (cf - ct)                    ! g at q
      gm = g1 * sw + g2 * w(q)                    ! <g>
      tq = yr * (g2 - gm)                         ! (Y g)(q)
      fs = yr * (g1 - gm) * sh                    ! sum_(k<q) hn (Y g)
      do mm = 1, m
        dm = a(mm) * (g1 * sp(mm) + g2 * w(q) * phi(q, mm))
        tq = tq + dm * phi(q, mm)
        fs = fs + dm * shp(mm)
      enddo
      fb = hn(q) * tq
      wm = -sg * fs
      ww = 0.0_rp
      if (q < nz) ww = wm - sg * fb
      sv = sg * cf * fb + cq(q, 2) * ww - cq(q, 3) * wm
      kd(q) = -cq(q, 5) * sv
      sw = sw + w(q); sh = sh + hn(q)
      do mm = 1, m
        sp(mm) = sp(mm) + w(q) * phi(q, mm)
        shp(mm) = shp(mm) + hn(q) * phi(q, mm)
      enddo
    enddo
  end subroutine obm_kd_d

  attributes(device) subroutine obm_y_d(nz, m, yr, w, a, phi, gc, tc, dmw)
    ! CPU obm_y: tc = Y gc = y_rem (gc - <gc>) + sum_m a_m phi_m <phi_m, gc> (gc, tc distinct)
    integer, value :: nz, m
    real(rp), value :: yr
    real(rp) :: w(nz), a(m), phi(nz, m), gc(nz), tc(nz), dmw(m)
    real(rp) :: gm, dm
    integer :: k, mm
    gm = 0.0_rp
    do k = 1, nz
      gm = gm + w(k) * gc(k)
    enddo
    do mm = 1, m
      dm = 0.0_rp
      if (a(mm) /= 0.0_rp) then
        do k = 1, nz
          dm = dm + w(k) * phi(k, mm) * gc(k)
        enddo
        dm = dm * a(mm)
      endif
      dmw(mm) = dm
    enddo
    do k = 1, nz
      dm = yr * (gc(k) - gm)
      do mm = 1, m
        if (a(mm) /= 0.0_rp) dm = dm + dmw(mm) * phi(k, mm)
      enddo
      tc(k) = dm
    enddo
  end subroutine obm_y_d

  attributes(device) real(rp) function obm_rho_d(tt, ts, zr)
    ! the model's density at depth -zr (CPU rho_eos_point): roms_gpu's rho_eos at one point, unmasked,
    ! rho = rho1 + qp1 dpth (1 - qp2 dpth) with rho1 = jackett_rho1(T, S) and the DUKO_2001
    ! compressibility qp1 = jackett_qp1(T, S, rho1) (formulas.F90: the same expressions and the same
    ! coefficients, the reference T, S of K0_Duk included; written out here because a call of those
    ! host/device functions from this kernel made nvlink fail, 'Prototype doesn't match')
    real(rp), value :: tt, ts, zr
    real(rp) :: r1, sq, k0, kd, q1, dpth
    sq = sqrt(max(0.0_rp, ts))
    r1 = (eos_r00 - rho0) + tt * (eos_r01 + tt * (eos_r02 + tt * (eos_r03 + tt * ( &
         eos_r04 + tt * eos_r05)))) &
         + ts * (eos_r10 + tt * (eos_r11 + tt * (eos_r12 + tt * ( &
         eos_r13 + tt * eos_r14))) &
         + sq * (eos_rS0 + tt * (eos_rS1 + tt * eos_rS2)) &
         + ts * eos_r20)
    kd = eos_Ttr * (eos_K01 + eos_Ttr * (eos_K02 + eos_Ttr * (eos_K03 + eos_Ttr * eos_K04))) &
         + eos_Tsr * (eos_K10 + eos_Ttr * (eos_K11 + eos_Ttr * (eos_K12 + eos_Ttr * eos_K13)) &
         + sqrt(eos_Tsr) * (eos_KS0 + eos_Ttr * (eos_KS1 + eos_Ttr * eos_KS2)))
    k0 = tt * (eos_K01 + tt * (eos_K02 + tt * (eos_K03 + tt * eos_K04))) &
         + ts * (eos_K10 + tt * (eos_K11 + tt * (eos_K12 + tt * eos_K13)) &
         + sq * (eos_KS0 + tt * (eos_KS1 + tt * eos_KS2)))
    q1 = 0.1_rp * (rho0 + r1) * (kd - k0) / max((eos_K00 + k0) * (eos_K00 + kd), 1.0e-30_rp)
    dpth = -zr
    obm_rho_d = r1 + q1 * dpth * (1.0_rp - qp2 * dpth)
  end function obm_rho_d

  attributes(device) subroutine obm_yb(nz, m, yr, w, a, phi, gc, tc)
    ! obm_y_d by a block of OBM_BT threads (one column; every thread calls it): tc = Y gc; the sums over k
    ! in lane-strided partials and a fixed tree (deterministic; the CPU sums k in order, so the result
    ! differs from the CPU's at rounding level); gc, tc distinct
    integer, value :: nz, m
    real(rp), value :: yr
    real(rp) :: w(nz), a(m), phi(nz, m), gc(nz), tc(nz)
    real(rp), shared :: sh(OBM_BT, OBM_MMAX + 1)
    real(rp) :: x, t
    integer :: l, k, mm, s
    l = threadIdx%x
    do mm = 1, m + 1
      sh(l, mm) = 0.0_rp
    enddo
    do k = l, nz, OBM_BT
      x = gc(k)
      sh(l, 1) = sh(l, 1) + w(k) * x
      do mm = 1, m
        sh(l, 1 + mm) = sh(l, 1 + mm) + w(k) * phi(k, mm) * x
      enddo
    enddo
    call syncthreads()
    s = OBM_BT / 2
    do while (s >= 1)
      if (l <= s) then
        do mm = 1, m + 1
          sh(l, mm) = sh(l, mm) + sh(l + s, mm)
        enddo
      endif
      call syncthreads()
      s = s / 2
    enddo
    do k = l, nz, OBM_BT
      t = yr * (gc(k) - sh(1, 1))
      do mm = 1, m
        if (a(mm) /= 0.0_rp) t = t + (sh(1, 1 + mm) * a(mm)) * phi(k, mm)
      enddo
      tc(k) = t
    enddo
    call syncthreads()
  end subroutine obm_yb

  attributes(device) real(rp) function obm_bsum(nz, w, x)
    ! sum_k w(k) x(k) by a block of OBM_BT threads (every thread calls it; all get the result); fixed tree
    integer, value :: nz
    real(rp) :: w(nz), x(nz)
    real(rp), shared :: sh(OBM_BT)
    integer :: l, k, s
    l = threadIdx%x
    sh(l) = 0.0_rp
    do k = l, nz, OBM_BT
      sh(l) = sh(l) + w(k) * x(k)
    enddo
    call syncthreads()
    s = OBM_BT / 2
    do while (s >= 1)
      if (l <= s) sh(l) = sh(l) + sh(l + s)
      call syncthreads()
      s = s / 2
    enddo
    obm_bsum = sh(1)
    call syncthreads()
  end function obm_bsum

  attributes(global) subroutine obm_face_k(nz, m, rtc)
    ! per operator sweep (PF_OPS, after the face kernel stored g at the active open faces), one block of
    ! OBM_BT threads per column: g := (Y g)/sqrt(c2), the face's column mean 0; the cell kernel forms
    ! F = Hz_f dn (Y g)/sqrt(c2) [CPU obm_ufx, obm_vfx: F_b = hf dn (Y g)/sqrt(c2)]
    integer, value :: nz, m
    real(rp), value :: rtc
    integer :: c, k, fi, fj, l
    c = blockIdx%x; l = threadIdx%x
    if (ob_act(c) <= 0.0_rp) return
    fi = ob_fi(c); fj = ob_fj(c)
    if (ob_ce(c) <= 2) then
      do k = l, nz, OBM_BT
        ob_sc(k, 1, c) = op_fu(fi, fj, k)
      enddo
    else
      do k = l, nz, OBM_BT
        ob_sc(k, 1, c) = op_fv(fi, fj, k)
      enddo
    endif
    call syncthreads()
    call obm_yb(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 1, c), ob_sc(1, 2, c))
    if (ob_ce(c) <= 2) then
      do k = l, nz, OBM_BT
        op_fu(fi, fj, k) = rtc * ob_sc(k, 2, c)
      enddo
      if (l == 1) op_sgu(fi, fj) = 0.0_rp
    else
      do k = l, nz, OBM_BT
        op_fv(fi, fj, k) = rtc * ob_sc(k, 2, c)
      enddo
      if (l == 1) op_sgv(fi, fj) = 0.0_rp
    endif
  end subroutine obm_face_k

  attributes(global) subroutine obm_expl_k(nz, m, nr, nn, pred, mref, minit, rcp, wr, wface, wdd)
    ! start of a stage (CPU obm_explicit), one block of OBM_BT threads per column: ob_ub =
    ! u_ext - <u_ext> + s Y (d - m), d = P(b_cur - b_ext) (baroclinic with the reference), b through the
    ! model's EOS (obm_rho_d) on the cell's z_r, tracers at nrhs; the reference m (once per step in the predictor;
    ! set to d at minit; rcp: taken from the restart's ghost cells first); wface: u, v(nnew) := ob_ub at
    ! the face (imex_obc_faces of imex_obc_pre, after u3dbc/v3dbc); wdd: d into ob_dd (OBCM lines). The
    ! pressure integral in the CPU's order (one thread); the column sums in a fixed tree
    integer, value :: nz, m, nr, nn, pred, mref, minit, rcp, wface, wdd
    real(rp), value :: wr
    integer :: c, e, i, j, l, fi, fj, gi, gj, k, lt
    real(rp) :: ti, si, te, se, zr, acc, db, um, sgn, ubt, uu
    c = blockIdx%x; lt = threadIdx%x
    if (ob_act(c) <= 0.0_rp) then
      do k = lt, nz, OBM_BT
        ob_ub(k, c) = 0.0_rp
      enddo
      return
    endif
    e = ob_ce(c); l = ob_cl(c); i = ob_ci(c); j = ob_cj(c); fi = ob_fi(c); fj = ob_fj(c)
    gi = ob_gi(c); gj = ob_gj(c)
    sgn = merge(1.0_rp, -1.0_rp, e == 2 .or. e == 4)
    do k = lt, nz, OBM_BT
      if (rcp == 1) ob_mr(k, c) = obcm3(gi, gj, k)   ! restart: m from the file (faces' ghost cells)
      ti = t(i, j, k, nr, itemp); si = t(i, j, k, nr, isalt)
      select case (e)
      case (1)
        te = t_west(l, k, itemp); se = t_west(l, k, isalt); ob_sc(k, 2, c) = u_west(l, k)
      case (2)
        te = t_east(l, k, itemp); se = t_east(l, k, isalt); ob_sc(k, 2, c) = u_east(l, k)
      case (3)
        te = t_south(l, k, itemp); se = t_south(l, k, isalt); ob_sc(k, 2, c) = v_south(l, k)
      case default
        te = t_north(l, k, itemp); se = t_north(l, k, isalt); ob_sc(k, 2, c) = v_north(l, k)
      end select
      zr = z_r(i, j, k)
      ob_sc(k, 1, c) = -g / rho0 * (obm_rho_d(ti, si, zr) - obm_rho_d(te, se, zr))
      ob_sc(k, 4, c) = Hz(i, j, k)
    enddo
    call syncthreads()
    if (lt == 1) then                             ! p = -int_z^0 db, on the cell column (CPU order)
      acc = 0.0_rp
      do k = nz, 1, -1
        db = ob_sc(k, 1, c)
        ob_sc(k, 1, c) = -(acc + 0.5_rp * db * ob_sc(k, 4, c))
        acc = acc + db * ob_sc(k, 4, c)
      enddo
    endif
    call syncthreads()
    if (wdd == 1) then
      do k = lt, nz, OBM_BT
        ob_dd(k, c) = ob_sc(k, 1, c)
      enddo
    endif
    if (mref == 1) then                           ! reference: radiate d - m (baroclinic)
      um = obm_bsum(nz, ob_w(1, c), ob_sc(1, 1, c))
      do k = lt, nz, OBM_BT
        ob_sc(k, 1, c) = ob_sc(k, 1, c) - um
        if (minit == 1) then
          ob_mr(k, c) = ob_sc(k, 1, c)
        else if (pred == 1) then
          ob_mr(k, c) = ob_mr(k, c) + wr * (ob_sc(k, 1, c) - ob_mr(k, c))
        endif
        ob_sc(k, 1, c) = ob_sc(k, 1, c) - ob_mr(k, c)
      enddo
      call syncthreads()
    endif
    call obm_yb(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 1, c), ob_sc(1, 3, c))
    um = obm_bsum(nz, ob_w(1, c), ob_sc(1, 2, c))
    ubt = 0.0_rp
    do k = lt, nz, OBM_BT
      ob_ub(k, c) = (ob_sc(k, 2, c) - um) + sgn * ob_sc(k, 3, c)
      if (wface == 1) then                        ! imex_obc_faces (no mean): the normal face velocity
        uu = ob_ub(k, c) + ubt
        if (e <= 2) then
          u(fi, fj, k, nn) = uu * umask(fi, fj)
        else
          v(fi, fj, k, nn) = uu * vmask(fi, fj)
        endif
      endif
    enddo
  end subroutine obm_expl_k

  attributes(global) subroutine obm_corr_k(nz, m, stg, nn, amean, first)
    ! CPU obm_correct of the stage solution (hs1 predictor, hs2 corrector; the interior cell's column),
    ! one block of OBM_BT threads per column: ob_ub -= Y g(x), g = s (cf zc + dbar), zc = -act x/n2k (dbar
    ! in the CPU's order, one thread); then imex_obc_faces: u, v(nnew) := ob_ub (+ amean: the barotropic
    ! velocity of set_HUV1's predictor transport, DU_avg1 or the EXTRAP_BAR_FLUXES combination over the
    ! face's current Hz dn, its depth sum in the CPU's order)
    integer, value :: nz, m, stg, nn, amean, first
    integer :: c, e, i, j, fi, fj, gi, gj, k, l
    real(rp) :: acc, sgn, cf, sd, tr, uu, xv
    real(rp), shared :: ubt
    c = blockIdx%x; l = threadIdx%x
    if (ob_act(c) <= 0.0_rp) return
    e = ob_ce(c); i = ob_ci(c); j = ob_cj(c); fi = ob_fi(c); fj = ob_fj(c); gi = ob_gi(c); gj = ob_gj(c)
    sgn = merge(1.0_rp, -1.0_rp, e == 2 .or. e == 4)
    do k = l, nz, OBM_BT
      if (stg == 1) then
        xv = hs1(i, j, k)
      else
        xv = hs2(i, j, k)
      endif
      ob_sc(k, 1, c) = -ad_act(i, j) * xv / ad_n2k(i, j, k)
      ob_sc(k, 4, c) = ad_czt(i, j, k)
      ob_sc(k, 5, c) = ad_czb(i, j, k)
      ob_sc(k, 6, c) = 0.5_rp * (Hz(i, j, k) + Hz(gi, gj, k))
    enddo
    call syncthreads()
    if (l == 1) then
      ob_sc(nz, 2, c) = 0.0_rp; acc = 0.0_rp
      do k = nz - 1, 1, -1
        acc = acc + ob_sc(k, 4, c) * ob_sc(k, 1, c) - ob_sc(k + 1, 5, c) * ob_sc(k + 1, 1, c)
        ob_sc(k, 2, c) = -acc
      enddo
      ubt = 0.0_rp
      if (amean == 1) then
        sd = 0.0_rp
        do k = 1, nz
          sd = sd + ob_sc(k, 6, c)
        enddo
        if (e <= 2) then
          sd = sd * dn_u(fi, fj)
          if (first == 1) then
            tr = DU_avg1(fi, fj)
          else
            tr = 3.63_rp * DU_avg1(fi, fj) - 4.47_rp * DU_avg2(fi, fj) + 2.05_rp * DU_avg_bak(fi, fj)
          endif
        else
          sd = sd * dm_v(fi, fj)
          if (first == 1) then
            tr = DV_avg1(fi, fj)
          else
            tr = 3.63_rp * DV_avg1(fi, fj) - 4.47_rp * DV_avg2(fi, fj) + 2.05_rp * DV_avg_bak(fi, fj)
          endif
        endif
        ubt = tr / sd
      endif
    endif
    call syncthreads()
    do k = l, nz, OBM_BT
      if (e <= 2) then
        cf = ad_cu(fi, fj, k)
      else
        cf = ad_cv(fi, fj, k)
      endif
      ob_sc(k, 1, c) = sgn * (cf * ob_sc(k, 1, c) + ob_sc(k, 2, c))
    enddo
    call syncthreads()
    call obm_yb(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 1, c), ob_sc(1, 3, c))
    do k = l, nz, OBM_BT
      ob_ub(k, c) = ob_ub(k, c) - ob_sc(k, 3, c)
      uu = ob_ub(k, c) + ubt
      if (e <= 2) then
        u(fi, fj, k, nn) = uu * umask(fi, fj)
      else
        v(fi, fj, k, nn) = uu * vmask(fi, fj)
      endif
    enddo
  end subroutine obm_corr_k

  attributes(global) subroutine obm_dinv8_k(np, nz, s, c2, hv, is_, ie_, js_, je_)
    ! the Jacobi with the boundary term (CPU imex_pf_setup8): kd = 0 + the face columns' terms in the
    ! order W, E, S, N; dinv = 1/(darea hv (1/n2c + c2 dd + sqrt(c2) kd)) where kd > 0
    integer, value :: np, nz, s, is_, ie_, js_, je_
    real(rp), value :: c2
    real(rp), device :: hv(is_:ie_, js_:je_, nz)
    integer :: p, i, j, k, e, c
    real(rp) :: kd
    p = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (p > np) return
    i = ob_pi(p); j = ob_pj(p)
    if (act(i, j) <= 0.0_rp) return
    do k = 1, nz
      kd = 0.0_rp
      do e = 1, 4
        c = ob_pc(e, p)
        if (c > 0) then
          if (ob_act(c) > 0.0_rp) kd = kd + ob_kdc(k, c)
        endif
      enddo
      if (kd > 0.0_rp) pf_dinv8(i, j, k, s) = 1.0_rp / (darea(i, j) * hv(i, j, k) * (1.0_rp / ad_n2c(i, j, k) &
           + c2 * ad_dd(i, j, k) + sqrt(c2) * kd))
    enddo
  end subroutine obm_dinv8_k

  attributes(global) subroutine obm_dinv4_k(np, nz, s, c2, hv, is_, ie_, js_, je_)
    ! as obm_dinv8_k for the FP32 set (CPU imex_pf_setup4: computed in FP64, rounded once)
    integer, value :: np, nz, s, is_, ie_, js_, je_
    real(rp), value :: c2
    real(rp), device :: hv(is_:ie_, js_:je_, nz)
    integer :: p, i, j, k, e, c
    real(rp) :: kd
    p = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (p > np) return
    i = ob_pi(p); j = ob_pj(p)
    if (act(i, j) <= 0.0_rp) return
    do k = 1, nz
      kd = 0.0_rp
      do e = 1, 4
        c = ob_pc(e, p)
        if (c > 0) then
          if (ob_act(c) > 0.0_rp) kd = kd + ob_kdc(k, c)
        endif
      enddo
      if (kd > 0.0_rp) pf_dinv4(i, j, k, s) = real(1.0_rp / (darea(i, j) * hv(i, j, k) * (1.0_rp / ad_n2c(i, j, k) &
           + c2 * ad_dd(i, j, k) + sqrt(c2) * kd)), 4)
    enddo
  end subroutine obm_dinv4_k

  attributes(global) subroutine obm_m3_k(nc, nz)
    ! restart: the reference m of each active face at the face's ghost cell (CPU obm_m_to3)
    integer, value :: nc, nz
    integer :: c, k
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    if (ob_act(c) <= 0.0_rp) return
    do k = 1, nz
      obcm3(ob_gi(c), ob_gj(c), k) = ob_mr(k, c)
    enddo
  end subroutine obm_m3_k

  attributes(global) subroutine obm_mdiag_k(nc, nz, m, ns, lmid1, lmid2, mref)
    ! OBCM partials per column into ob_pd (CPU obm_mdiag): 1 count, 2-8 the hf-weighted column sums of
    ! d^2, m^2, (d-m)^2, (Y d)^2, (Y m)^2, (Y(d-m))^2, u_bc^2; 9 the edge's middle column; 10-13 its
    ! mode-1 values; 14-16 the column rms of Y(d-m), u_bc, d (maxima)
    integer, value :: nc, nz, m, ns, lmid1, lmid2, mref
    integer :: c, e, k, fi, fj, lm
    real(rp) :: s, s2, s3, s4, ub, ph
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    do k = 1, 16
      ob_pd(k, c) = 0.0_rp
    enddo
    if (ob_act(c) <= 0.0_rp) return
    e = ob_ce(c); fi = ob_fi(c); fj = ob_fj(c)
    s = 0.0_rp
    do k = 1, nz
      s = s + ob_w(k, c) * ob_dd(k, c)
    enddo
    do k = 1, nz
      ob_sc(k, 1, c) = ob_dd(k, c) - s           ! d
      ob_sc(k, 2, c) = 0.0_rp                      ! m (the radiated reference)
      if (mref == 1) ob_sc(k, 2, c) = ob_mr(k, c)
    enddo
    ob_pd(1, c) = 1.0_rp
    s2 = 0.0_rp; s3 = 0.0_rp; s4 = 0.0_rp
    do k = 1, nz
      s2 = s2 + ob_w(k, c) * ob_sc(k, 1, c)**2
      s3 = s3 + ob_w(k, c) * ob_sc(k, 2, c)**2
      s4 = s4 + ob_w(k, c) * (ob_sc(k, 1, c) - ob_sc(k, 2, c))**2
    enddo
    ob_pd(2, c) = s2; ob_pd(3, c) = s3; ob_pd(4, c) = s4
    ob_pd(16, c) = sqrt(s2)
    call obm_y_d(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 1, c), ob_sc(1, 3, c), &
         ob_wr(1, 9, c))                   ! Y d
    s = 0.0_rp
    do k = 1, nz
      s = s + ob_w(k, c) * ob_sc(k, 3, c)**2
    enddo
    ob_pd(5, c) = s
    call obm_y_d(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 2, c), ob_sc(1, 3, c), &
         ob_wr(1, 9, c))                   ! Y m
    s = 0.0_rp
    do k = 1, nz
      s = s + ob_w(k, c) * ob_sc(k, 3, c)**2
    enddo
    ob_pd(6, c) = s
    do k = 1, nz
      ob_sc(k, 4, c) = ob_sc(k, 1, c) - ob_sc(k, 2, c)
    enddo
    call obm_y_d(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 4, c), ob_sc(1, 3, c), &
         ob_wr(1, 9, c))                   ! Y (d - m)
    s = 0.0_rp
    do k = 1, nz
      s = s + ob_w(k, c) * ob_sc(k, 3, c)**2
    enddo
    ob_pd(7, c) = s; ob_pd(14, c) = sqrt(s)
    ub = 0.0_rp                                    ! u^n at the face, baroclinic
    do k = 1, nz
      if (e <= 2) then
        ob_sc(k, 3, c) = u(fi, fj, k, ns)
      else
        ob_sc(k, 3, c) = v(fi, fj, k, ns)
      endif
      ub = ub + ob_w(k, c) * ob_sc(k, 3, c)
    enddo
    s = 0.0_rp
    do k = 1, nz
      ob_sc(k, 3, c) = ob_sc(k, 3, c) - ub
      s = s + ob_w(k, c) * ob_sc(k, 3, c)**2
    enddo
    ob_pd(8, c) = s; ob_pd(15, c) = sqrt(s)
    lm = merge(lmid1, lmid2, e <= 2)
    if (ob_cl(c) == lm .and. m >= 1) then
      ob_pd(9, c) = 1.0_rp
      s = 0.0_rp; s2 = 0.0_rp; s3 = 0.0_rp; s4 = 0.0_rp
      do k = 1, nz
        ph = ob_w(k, c) * ob_phi(k, 1, c)
        s = s + ph * ob_sc(k, 1, c); s2 = s2 + ph * ob_sc(k, 2, c)
        s3 = s3 + ph * ob_sc(k, 4, c); s4 = s4 + ph * ob_sc(k, 3, c)
      enddo
      ob_pd(10, c) = s / ob_c(1, c); ob_pd(11, c) = s2 / ob_c(1, c); ob_pd(12, c) = s3 / ob_c(1, c)
      ob_pd(13, c) = s4
    endif
  end subroutine obm_mdiag_k

  attributes(global) subroutine obm_fdiag_k(nc, nz, nn)
    ! OBCF partials per column into ob_pd (CPU obm_fdiag, from u, v(nnew) at the faces): 1 levels,
    ! 2 u* inflow, 3 its depth mean inflow, 4 both agree, 5 columns, 6 ubt^2, 7 column sum w (u - ubt)^2
    integer, value :: nc, nz, nn
    integer :: c, e, k, fi, fj
    real(rp) :: ubt, sg, uc, s
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    do k = 1, 7
      ob_pd(k, c) = 0.0_rp
    enddo
    if (ob_act(c) <= 0.0_rp) return
    e = ob_ce(c); fi = ob_fi(c); fj = ob_fj(c)
    sg = merge(1.0_rp, -1.0_rp, e == 2 .or. e == 4)
    ubt = 0.0_rp
    do k = 1, nz
      if (e <= 2) then
        uc = u(fi, fj, k, nn)
      else
        uc = v(fi, fj, k, nn)
      endif
      ubt = ubt + ob_w(k, c) * uc
    enddo
    s = 0.0_rp
    do k = 1, nz
      if (e <= 2) then
        uc = u(fi, fj, k, nn)
      else
        uc = v(fi, fj, k, nn)
      endif
      ob_pd(1, c) = ob_pd(1, c) + 1.0_rp
      if (sg * uc < 0.0_rp) ob_pd(2, c) = ob_pd(2, c) + 1.0_rp
      if (sg * ubt < 0.0_rp) ob_pd(3, c) = ob_pd(3, c) + 1.0_rp
      if ((sg * uc < 0.0_rp) .eqv. (sg * ubt < 0.0_rp)) ob_pd(4, c) = ob_pd(4, c) + 1.0_rp
      s = s + ob_w(k, c) * (uc - ubt)**2
    enddo
    ob_pd(5, c) = 1.0_rp; ob_pd(6, c) = ubt**2; ob_pd(7, c) = s
  end subroutine obm_fdiag_k

  attributes(global) subroutine obm_start_k(nc, nz, m)
    ! start-note partials (CPU obm_startnote): 1 column, 2 sum w m^2, 3 sum w (Y m)^2
    integer, value :: nc, nz, m
    integer :: c, k
    real(rp) :: s2, s3
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    do k = 1, 3
      ob_pd(k, c) = 0.0_rp
    enddo
    if (ob_act(c) <= 0.0_rp) return
    do k = 1, nz
      ob_sc(k, 1, c) = ob_mr(k, c)
    enddo
    call obm_y_d(nz, m, ob_yr(c), ob_w(1, c), ob_a(1, c), ob_phi(1, 1, c), ob_sc(1, 1, c), ob_sc(1, 2, c), &
         ob_wr(1, 9, c))
    s2 = 0.0_rp; s3 = 0.0_rp
    do k = 1, nz
      s2 = s2 + ob_w(k, c) * ob_mr(k, c)**2
      s3 = s3 + ob_w(k, c) * ob_sc(k, 2, c)**2
    enddo
    ob_pd(1, c) = 1.0_rp; ob_pd(2, c) = s2; ob_pd(3, c) = s3
  end subroutine obm_start_k

  attributes(global) subroutine obm_red_k(n, nc, i1, e1, i2, e2, i3, e3, i4, e4)
    ! red_part(:, s) = ob_pd(i_s, c) on the columns of edge e_s (0 elsewhere), s = 1..4
    integer, value :: n, nc, i1, e1, i2, e2, i3, e3, i4, e4
    integer :: r
    r = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (r > n) return
    red_part(r, 1) = 0.0_rp; red_part(r, 2) = 0.0_rp; red_part(r, 3) = 0.0_rp; red_part(r, 4) = 0.0_rp
    if (r > nc) return
    if (ob_ce(r) == e1) red_part(r, 1) = ob_pd(i1, r)
    if (ob_ce(r) == e2) red_part(r, 2) = ob_pd(i2, r)
    if (ob_ce(r) == e3) red_part(r, 3) = ob_pd(i3, r)
    if (ob_ce(r) == e4) red_part(r, 4) = ob_pd(i4, r)
  end subroutine obm_red_k
#endif

  !----------------------------------------------------------------------
  subroutine obm_edge_sums(nv, iv, out)
    ! out(v, e) = the deterministic sum (imex_red) over this tile's columns of edge e of ob_pd(iv(v), :),
    ! then over the ranks; 4 sums per pass
    integer, intent(in) :: nv, iv(nv)
    real(8), intent(out) :: out(nv, 4)
    integer :: q0, q, s, ie_(4), iv_(4)
    real(rp) :: rs(4)
    real(8) :: loc(nv * 4), glob(nv * 4)
#ifdef USE_CUDA
    integer :: n
#endif
    out = 0.0d0
    ! pair q = 1..4 nv: value 1 + mod(q - 1, nv), edge 1 + (q - 1)/nv
    do q0 = 0, nv * 4 - 1, 4
      do s = 1, 4
        q = q0 + s
        iv_(s) = 1; ie_(s) = 99
        if (q <= nv * 4) then
          iv_(s) = iv(1 + mod(q - 1, nv)); ie_(s) = 1 + (q - 1) / nv
        endif
      enddo
#ifdef USE_CUDA
      n = int(decomp%nx * decomp%ny)
      call obm_red_k<<<(n + OBM_NT - 1) / OBM_NT, OBM_NT, 0, work_stream>>>(n, ob_nc, iv_(1), ie_(1), iv_(2), &
           ie_(2), iv_(3), ie_(3), iv_(4), ie_(4))
#endif
      call red_sum(4, rs)
      do s = 1, 4
        if (q0 + s <= nv * 4) loc(q0 + s) = rs(s)
      enddo
    enddo
    call comm_allreduce_sum(nv * 4, loc, glob)
    do q = 1, nv * 4
      out(1 + mod(q - 1, nv), 1 + (q - 1) / nv) = glob(q)
    enddo
  end subroutine obm_edge_sums

  subroutine obm_edge_max(nv, iv, out)
    ! out(v, e) = max over the columns of edge e of ob_pd(iv(v), :) (exact; host, after a sync)
    integer, intent(in) :: nv, iv(nv)
    real(8), intent(out) :: out(nv, 4)
    real(8) :: loc(nv * 4)
    integer :: c, v
    call sync_arch()
    loc = 0.0d0
    do c = 1, ob_nc
      if (ob_act(c) <= 0.0_rp) cycle
      do v = 1, nv
        loc(v + nv * (ob_ce(c) - 1)) = max(loc(v + nv * (ob_ce(c) - 1)), ob_pd(iv(v), c))
      enddo
    enddo
    call comm_allreduce_max(nv * 4, loc, out)
  end subroutine obm_edge_max

  !----------------------------------------------------------------------
  subroutine obm_explicit(stage, wface)
    ! the stage's explicit face values (obm_expl_k); the reference's bookkeeping on the host (CPU
    ! obm_explicit: ramped weight, restart copy, cold start and its note, the OBCM lines)
    integer, intent(in) :: stage
    logical, intent(in) :: wface
    logical :: mref, minit, rcp, pred, mpr
    real(rp) :: wr
#ifdef USE_CUDA
    integer :: nblk
#endif
    if (.not. imex_obc_modal) return
    if (.not. allocated(ob_ce)) return
    pred = (stage == imex_stage_pred)
    mref = imex_obc_tref > 0.0_rp
    rcp = .false.; minit = .false.; wr = 0.0_rp
    if (mref) then
      if (.not. ob_mr_ok .and. ob_rst) then        ! restart: m from the file
        rcp = .true.
        ob_mr_ok = .true.
      endif
      minit = .not. ob_mr_ok
      if (pred) then
        if (minit) ob_mn = 0
        ob_mn = min(ob_mn, 1000000000 - 1) + 1
      endif
      wr = max(1.0_rp - exp(-dt / imex_obc_tref), min(1.0_rp / dble(max(ob_mn, 1)), &
           1.0_rp - exp(-dt / min(imex_obc_tref, obm_tramp))))
    endif
    mpr = imex_obc_mprint > 0 .and. pred
    if (mpr) mpr = mod(int(iic), imex_obc_mprint) == 0
    if (ob_nc > 0) then
#ifdef USE_CUDA
      nblk = (ob_nc + OBM_NT - 1) / OBM_NT
      call obm_expl_k<<<ob_nc, OBM_BT, 0, work_stream>>>(int(decomp%nz), ob_m, int(nrhs), int(nnew), &
           merge(1, 0, pred), merge(1, 0, mref), merge(1, 0, minit), merge(1, 0, rcp), wr, merge(1, 0, wface), &
           merge(1, 0, mpr))
#endif
    endif
    if (rcp) ob_rst = .false.
    if (mref) ob_mr_ok = .true.
    if (mref .and. minit) call obm_startnote()
    if (mpr) call obm_mdiag()
  end subroutine obm_explicit

  subroutine obm_correct_faces(stage)
    ! after u3dbc/v3dbc of pre_step3d (stage 1: predictor, with the barotropic mean) or of step3d_uv2
    ! (stage 2): the face increment of the stage solution and the faces' normal velocity (obm_corr_k)
    integer, intent(in) :: stage
#ifdef USE_CUDA
    integer :: nblk
#endif
    if (.not. imex_obc_modal) return
    if (.not. allocated(ob_ce)) return
    if (ob_nc > 0) then
#ifdef USE_CUDA
      nblk = (ob_nc + OBM_NT - 1) / OBM_NT
      call obm_corr_k<<<ob_nc, OBM_BT, 0, work_stream>>>(int(decomp%nz), ob_m, stage, int(nnew), &
           merge(1, 0, stage == imex_stage_pred), merge(1, 0, is_first_time_step()))
#endif
    endif
    if (stage == imex_stage_pred .and. imex_obc_mprint > 0) then
      if (mod(int(iic), imex_obc_mprint) == 0) call obm_fdiag()
    endif
  end subroutine obm_correct_faces

  subroutine obm_face_op(c2)
    ! per operator sweep (PF_OPS): the open faces' term (obm_face_k)
    real(rp), intent(in) :: c2
#ifdef USE_CUDA
    if (ob_nact <= 0 .or. c2 <= 0.0_rp) return
    call obm_face_k<<<ob_nc, OBM_BT, 0, work_stream>>>(int(decomp%nz), ob_m, 1.0_rp / sqrt(c2))
#endif
  end subroutine obm_face_op

  subroutine obm_dinv(s, c2, hv, prec)
    ! pf_setup8/4: the Jacobi with the boundary term on the cells next to the open faces
    integer, intent(in) :: s, prec
    real(rp), intent(in) :: c2
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
#ifdef USE_CUDA
    integer :: nblk
    if (.not. imex_obc_modal .or. c2 <= 0.0_rp .or. ob_np <= 0 .or. ob_nc <= 0) return
    nblk = (ob_np + OBM_NT - 1) / OBM_NT
    if (prec == 8) then
      call obm_dinv8_k<<<nblk, OBM_NT, 0, work_stream>>>(ob_np, int(decomp%nz), s, c2, hv, int(decomp%is), &
           int(decomp%ie), int(decomp%js), int(decomp%je))
    else
      call obm_dinv4_k<<<nblk, OBM_NT, 0, work_stream>>>(ob_np, int(decomp%nz), s, c2, hv, int(decomp%is), &
           int(decomp%ie), int(decomp%js), int(decomp%je))
    endif
#endif
  end subroutine obm_dinv

  subroutine obm_m_to3()
    ! restart: imex_obcm at the faces' ghost cells, 0 elsewhere
    if (.not. allocated(obcm3)) return
    call zero3(obcm3)
    if (.not. allocated(ob_ce)) return
#ifdef USE_CUDA
    if (ob_nc > 0) call obm_m3_k<<<(ob_nc + OBM_NT - 1) / OBM_NT, OBM_NT, 0, work_stream>>>(ob_nc, &
         int(decomp%nz))
#endif
  end subroutine obm_m_to3

  !----------------------------------------------------------------------
  subroutine obm_startnote()
    ! CPU obm_startnote (the reference was set to the present mismatch): per open edge the rms of the
    ! held mismatch and of Y m; the start NOTE; the WARNING above 0.7 m2/s2
    real(rp), parameter :: obm_dwarn = 0.7_rp
    character(len=1), parameter :: en(4) = ['W', 'E', 'S', 'N']
    real(8) :: sm(3, 4)
    real(rp) :: rd, ry
    integer :: e
    logical :: big
#ifdef USE_CUDA
    if (ob_nc > 0) call obm_start_k<<<(ob_nc + OBM_NT - 1) / OBM_NT, OBM_NT, 0, work_stream>>>(ob_nc, &
         int(decomp%nz), ob_m)
#endif
    call obm_edge_sums(3, [1, 2, 3], sm)
    if (.not. talk) return
    big = .false.
    write(*,'(1x,A)') "IMEX_OBC start: the 'modal' reference is set to the present difference between the" &
         // ' model and its boundary data (held, not radiated)'
    do e = 1, 4
      if (sm(1, e) <= 0.0d0) cycle
      rd = sqrt(sm(2, e) / sm(1, e)); ry = sqrt(sm(3, e) / sm(1, e))
      write(*,'(1x,A,A,A,F9.4,A,F8.4,A)') 'IMEX_OBC start: edge ', en(e), ': held pressure difference, rms', &
           rd, ' m2/s2; as a face velocity', ry, ' m/s'
      big = big .or. rd > obm_dwarn
    enddo
    write(*,'(1x,A)') "IMEX_IGW NOTE: a run that does not start from the state its boundary data come from" &
         // " (a nest not initialised from its parent) should take its first ~10 days with imex_obc_mode =" &
         // " 'fixed' and switch to 'modal' at a restart; otherwise the adjustment of the edge zones is" &
         // ' carried into the interior as a low-frequency error'
    if (big) write(*,'(1x,A,F5.1,A)') 'IMEX_IGW WARNING: the held pressure difference exceeds', obm_dwarn, &
         ' m2/s2 on an open edge: check the boundary data and that the depths of this grid match those of' &
         // ' the grid the boundary data come from along the open edges'
  end subroutine obm_startnote

  subroutine obm_mdiag()
    ! CPU obm_mdiag (imex_obc_mprint, predictor, after the reference update): an OBCM line per open edge
    use time_state, only: time
    real(8) :: ds(13, 4), dx(3, 4)
    integer :: e, k
    integer :: lmid1, lmid2
    lmid1 = int((decomp%gny + 1) / 2 - (decomp%j0 - 1))
    lmid2 = int((decomp%gnx + 1) / 2 - (decomp%i0 - 1))
#ifdef USE_CUDA
    if (ob_nc > 0) call obm_mdiag_k<<<(ob_nc + OBM_NT - 1) / OBM_NT, OBM_NT, 0, work_stream>>>(ob_nc, &
         int(decomp%nz), ob_m, int(nstp), lmid1, lmid2, merge(1, 0, imex_obc_tref > 0.0_rp))
#endif
    call obm_edge_sums(13, [(k, k = 1, 13)], ds)
    call obm_edge_max(3, [14, 15, 16], dx)
    if (.not. talk) return
    do e = 1, 4
      if (ds(1, e) <= 0.0d0) cycle
      write(*,'(A,I8,F11.5,I2,I6,14ES11.3)') 'OBCM', int(iic), time / 86400.0_rp, e, nint(ds(1, e)), &
           (sqrt(ds(k, e) / ds(1, e)), k = 2, 8), dx(1:3, e), ds(10:13, e) * merge(1.0d0, 0.0d0, ds(9, e) > 0.0d0)
    enddo
  end subroutine obm_mdiag

  subroutine obm_fdiag()
    ! CPU obm_fdiag (imex_obc_mprint, pre_step3d after the face write): an OBCF line per open edge
    use time_state, only: time
    real(8) :: fs(7, 4)
    integer :: e
#ifdef USE_CUDA
    if (ob_nc > 0) call obm_fdiag_k<<<(ob_nc + OBM_NT - 1) / OBM_NT, OBM_NT, 0, work_stream>>>(ob_nc, &
         int(decomp%nz), int(nnew))
#endif
    call obm_edge_sums(7, [1, 2, 3, 4, 5, 6, 7], fs)
    if (.not. talk) return
    do e = 1, 4
      if (fs(1, e) <= 0.0d0) cycle
      write(*,'(A,I8,F11.5,I2,I9,5ES11.3)') 'OBCF', int(iic), time / 86400.0_rp, e, nint(fs(1, e)), &
           fs(2:4, e) / fs(1, e), sqrt(fs(6, e) / fs(5, e)), sqrt(fs(7, e) / fs(5, e))
    enddo
  end subroutine obm_fdiag

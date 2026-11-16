  !======================================================================
  ! imex_pc = 'modes': the mode-split preconditioner of the stage solves (CPU bf897292, imex_igw.F90 section
  ! 'mode-split', pcm_*; PRECOND_SCOPE.md sec. 3.1, 8.2 stage 5).
  ! The stage operator A = W M (W = darea hv) is, in a column, nearly a 2-D Helmholtz operator per vertical
  ! mode, and the Jacobi diagonal averages over the modes. Per column the K fastest modes Phi (in the unknown,
  ! wr-orthonormal, wr = W/n2c) are taken out of the Jacobi step ("one projection" form):
  !   rc = Phi^T W r                       (K 2-D fields)
  !   z  = r/d_K + Phi (V(rc) - G^-1 rc)   d_K = diag(M) less the K modes' stiff part, G = Phi^T W d_K Phi
  ! V = one V-cycle per mode (2 or 3 levels, 2 x 2 aggregates inside the tile, red-black sweeps with tile-local
  ! colours, one halo exchange per sweep for all K modes, no gathered level: the CPU's pcm_vcycle on the level
  ! kernels of implicit_fs_mod, here kernels over the K modes at once in the format of the free surface's fs_mg
  ! levels, the exchanges by the free surface's CUDA IPC rings when it has them, else one GPU-aware MPI message
  ! per side for all modes), so the iterations depend on the tiling at the tolerance level.
  ! Build (pcm_build, once per coef_version: b_ref rebuild and restart replay): K (imex_pc_modes, or automatic: a
  ! WKB switch on the median mode-1 Courant number, then the median Courant numbers of the 4 fastest modes; K = 0
  ! is the plain Jacobi path), the modes of each column (obm_modes_d, the column's own hv as weights, one device
  ! thread per column), mapped to the unknown, Gram-Schmidt in wr, rounded to FP32; a_m's interior part from 5 K
  ! applications of the FP64 operator sweep pf_op8 (mode 0) on a 5-colouring of the columns at c2 = (theta_c
  ! dt)^2 (the 'modal' faces off: cxm = cxo) and the 'modal' part from 5 K more with them on. Per stage and c2
  ! (pcm_stage): d_K (floored at 1/n2c), G^-1 per column, the levels per mode. In the fused solve (PF_SOLVE)
  ! act/(W d_K) replaces the Jacobi reciprocal (pcm_dinv), the start-up and sweep 2 form rc, pcm_post runs the
  ! V-cycles and a, PCM_ZADD forms z = z_J + Phi a. The same expressions in the same order as the CPU's.

  subroutine pcm_setup(stage, c2, hv, ru, rv)
    ! (CPU pcm_setup) the build at a new operator (the probe at the fixed c2 = (theta_c dt)^2, so that the parts
    ! per unit c2 do not depend on which solve triggers it: exact restarts), the stage's d_K, G^-1 and levels
    ! at a new operator or c2
    integer, intent(in) :: stage
    real(rp), intent(in) :: c2
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: ru(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rv(decomp%is:decomp%ie, decomp%js:decomp%je)
    if (pcm_ver /= coef_version) then
      call itw(31, 0)
      call pcm_build((imex_theta_c * dt)**2, hv, ru, rv)
      call itw(31, 1)
      pcm_ver = coef_version; pcm_sver = -1
    endif
    if (pcm_k > 0 .and. (pcm_sver(stage) /= pcm_ver .or. pcm_sc2(stage) /= c2)) then
      call pcm_stage(stage, c2, hv)
      pcm_sver(stage) = pcm_ver; pcm_sc2(stage) = c2
    endif
  end subroutine pcm_setup

  subroutine pcm_alloc_small()
    ! first build: K max, the level sizes, the coarse face factors of levels 1 and 2 (the free surface's
    ! 'width' rule on the tiles' true sizes, as fs_mg's mg_faces: the CPU's pcm_faces_dir for the same tiles),
    ! the counters
    integer :: l, np, k, npx, npy
    integer, allocatable :: info(:,:), cxs(:), cys(:)
    real(rp), allocatable :: fx(:,:), fy(:,:)
    logical :: ok
    pcm_kmax = 4
    if (imex_pc_modes > 0) pcm_kmax = imex_pc_modes
    pcm_kmax = min(pcm_kmax, int(decomp%nz) - 1)
    pcm_nx(0) = int(decomp%nx); pcm_ny(0) = int(decomp%ny)
    do l = 1, pcm_nl - 1
      pcm_nx(l) = (pcm_nx(l - 1) + 1) / 2; pcm_ny(l) = (pcm_ny(l - 1) + 1) / 2
    enddo
    np = int(comm_size())
    allocate(info(4, np))
    call comm_allgather_i4(4, [int(decomp%inode), int(decomp%jnode), int(decomp%nx), int(decomp%ny)], info)
    npx = maxval(info(1, :)) + 1; npy = maxval(info(2, :)) + 1
    allocate(cxs(0:npx - 1), cys(0:npy - 1))
    cxs = -1; cys = -1
    ok = .true.
    do k = 1, np                           ! the tile sizes per column / row (a tensor-product tiling)
      if (cxs(info(1, k)) < 0) cxs(info(1, k)) = info(3, k)
      if (cys(info(2, k)) < 0) cys(info(2, k)) = info(4, k)
      ok = ok .and. cxs(info(1, k)) == info(3, k) .and. cys(info(2, k)) == info(4, k)
    enddo
    if (.not. ok .or. any(cxs < 0) .or. any(cys < 0)) &
         error stop "imex_igw: imex_pc = 'modes' needs a tensor-product tiling"
    allocate(fx(pcm_nx(0) + 1, 2), fy(pcm_ny(0) + 1, 2))
    call pcm_faces_dir(cxs, npx, int(decomp%inode), pcm_nx, decomp%periodic_x, fx)
    call pcm_faces_dir(cys, npy, int(decomp%jnode), pcm_ny, decomp%periodic_y, fy)
    allocate(pcm_ffx(pcm_nx(0) + 1, 2), pcm_ffy(pcm_ny(0) + 1, 2))
    pcm_ffx = fx; pcm_ffy = fy
    deallocate(info, cxs, cys, fx, fy)
    allocate(pcm_hist(PCM_NHB, 4), pcm_cnt(2))
    call pcm_zero_hist()
  end subroutine pcm_alloc_small

  subroutine pcm_faces_dir(c0, npn, me, nl, per, ff)
    ! (CPU pcm_faces_dir) factors of the coarse face coefficients of levels 1 and 2 along one direction: for the
    ! coarse face between cells I-1 and I (fine face fi = min(2I-1, nf+1)), (w_f(fi-1) + w_f(fi)) / (w_c(I-1) +
    ! w_c(I)), widths in fine-grid cells on the global index of the level (beyond a physical edge the edge
    ! cell's width, periodic wrap with per)
    integer, intent(in) :: npn, me
    integer, intent(in) :: c0(0:npn - 1), nl(0:pcm_nl - 1)
    logical, intent(in) :: per
    real(rp), intent(out) :: ff(:, :)
    integer :: c(0:npn - 1), cp(0:npn - 1), t, ic, k0f, k0c, g0f, g0c, nf, nc, fi, l
    real(rp), allocatable :: wf(:), wc(:)
    ff = 0.0_rp
    c = c0
    allocate(wf(sum(c))); wf = 1.0_rp
    do l = 1, 2
      cp = c; c = (c + 1) / 2
      allocate(wc(sum(c)))
      k0f = 0; k0c = 0
      do t = 0, npn - 1
        do ic = 1, c(t)
          wc(k0c + ic) = wf(k0f + 2 * ic - 1)
          if (2 * ic <= cp(t)) wc(k0c + ic) = wc(k0c + ic) + wf(k0f + 2 * ic)
        enddo
        k0f = k0f + cp(t); k0c = k0c + c(t)
      enddo
      g0f = sum(cp(0:me - 1)); g0c = sum(c(0:me - 1)); nf = cp(me); nc = c(me)
      if (nc /= nl(l)) error stop 'imex_igw: pcm_faces_dir: level size'
      do ic = 1, nc + 1
        fi = min(2 * ic - 1, nf + 1)
        ff(ic, l) = (pcm_wget(wf, g0f + fi - 1, per) + pcm_wget(wf, g0f + fi, per)) &
             / (pcm_wget(wc, g0c + ic - 1, per) + pcm_wget(wc, g0c + ic, per))
      enddo
      call move_alloc(wc, wf)
    enddo
    deallocate(wf)
  end subroutine pcm_faces_dir

  real(rp) function pcm_wget(w, k, per)
    ! width of cell k of a level (beyond the ends: periodic wrap or the end cell)
    real(rp), intent(in) :: w(:)
    integer, intent(in) :: k
    logical, intent(in) :: per
    integer :: n
    n = size(w)
    if (k < 1) then
      pcm_wget = w(1)
      if (per) pcm_wget = w(n)
    else if (k > n) then
      pcm_wget = w(n)
      if (per) pcm_wget = w(1)
    else
      pcm_wget = w(k)
    endif
  end function pcm_wget

  subroutine pcm_alloc_big()
    ! the first build with modes: the per-mode arrays, the levels, the MPI ring buffers
    integer(ip) :: is, ie, js, je, nz
    integer :: kx, mx, my
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    kx = pcm_kmax; mx = pcm_nx(0); my = pcm_ny(0)
    allocate(pcm_phi(is:ie, js:je, nz, kx), pcm_ci(is:ie, js:je, 3, kx), pcm_act(is:ie, js:je, kx))
    allocate(pcm_cm(is:ie, js:je, kx), pcm_cb(is:ie, js:je, kx), pcm_dki(is:ie, js:je, nz, 2))
    allocate(pcm_gi(is:ie, js:je, kx, kx, 2), pcm_t3(is:ie, js:je, nz), pcm_cmc(mx * my, kx))
    pcm_phi = 0.0_rp; pcm_ci = 0.0_rp; pcm_act = 0.0_rp; pcm_cm = 0.0_rp; pcm_cb = 0.0_rp; pcm_dki = 0.0_rp
    pcm_gi = 0.0_rp; pcm_t3 = 0.0_rp; pcm_cmc = 0.0_rp
    if (pf_prec == 4) then
      allocate(pcm_phi4(is:ie, js:je, nz, kx)); pcm_phi4 = 0.0
    endif
    allocate(lv_x(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1), lv_b(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1))
    allocate(lv_t(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1))
    allocate(lv_e(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1, 2), lv_w(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1, 2))
    allocate(lv_s(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1, 2), lv_d(0:mx + 1, 0:my + 1, kx, 0:pcm_nl - 1, 2))
    lv_x = 0.0_rp; lv_b = 0.0_rp; lv_t = 0.0_rp; lv_e = 0.0_rp; lv_w = 0.0_rp; lv_s = 0.0_rp; lv_d = 0.0_rp
    allocate(pcm_rsw(kx * my), pcm_rse(kx * my), pcm_rrw(kx * my), pcm_rre(kx * my))
    allocate(pcm_rss(kx * mx), pcm_rsn(kx * mx), pcm_rrs(kx * mx), pcm_rrn(kx * mx))
    pcm_rsw = 0; pcm_rse = 0; pcm_rrw = 0; pcm_rre = 0; pcm_rss = 0; pcm_rsn = 0; pcm_rrs = 0; pcm_rrn = 0
    pcm_big = .true.
  end subroutine pcm_alloc_big

  subroutine pcm_build(c2p, hv, ru, rv)
    ! (CPU pcm_build) per operator: K (imex_pc_modes, or automatic from the median mode Courant numbers, 0 =
    ! Jacobi), the modes of every active column (pcm_modes_k), mapped to the unknown, Gram-Schmidt in wr
    ! (fastest first), rounded to FP32; the parts of a_m = Phi_m^T A Phi_m by probing (pcm_probe); the
    ! IMEX_PCM lines
    real(rp), intent(in) :: c2p
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: ru(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rv(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: hist(PCM_NHB, 4), ghist(PCM_NHB, 4), cnt(2), ib, m, kb, ntot, acc, nxi, nzi, c0, n, nch
    real(8) :: hr(PCM_NHB * 4), hg(PCM_NHB * 4), med(4), cmx(4), cmg(4), x1(2), x2(2)
    real(rp) :: cou, t0, cm1
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je, mm, c, nc
    character(len=8) :: env
#ifdef USE_CUDA
    real(rp), device, allocatable :: wh(:,:), wg(:,:), wcm(:,:), wph(:,:,:), wr(:,:,:), wz(:,:,:,:), wxp(:,:,:)
    real(rp), device, allocatable :: wxs(:,:), ww(:,:)
    integer, device, allocatable :: wi(:,:,:), wd(:,:)
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call sync_arch()
    t0 = wall_time()
    if (.not. allocated(pcm_ffx)) then
      call pcm_alloc_small()
      call get_environment_variable('ROMS_GPU_PCM_GRAPH', env)
      pcm_gon = trim(env) /= '0'
    endif
    pcm_gready = .false.                         ! the graphs (PCM_APPLY) are recorded again (K, levels may change)
    kb = pcm_kmax
    nxi = int(nx); nzi = int(nz); nc = nx * ny
    call pcm_zero_hist()
    if (imex_pc_modes == 0) then
      ! automatic: first the switch from the WKB mode-1 speed c_1 ~ sum_k N dz / pi (N2 floored as below; no
      ! eigen-solve), median over the active columns; below imex_pc_cou_on no modes are built (Jacobi)
#ifdef USE_CUDA
      call pcm_wkb_k<<<(nc + PCM_NT - 1) / PCM_NT, PCM_NT, 0, work_stream>>>(int(nc), nxi, nzi, imex_theta_c, dt, &
           acos(-1.0_rp), imex_n2_ref, imex_n2_reg, hv, int(is), int(ie), int(js), int(je))
#else
      error stop 'imex_igw: imex_pc = modes needs the CUDA build'
#endif
      call sync_arch()
      hist = pcm_hist
      hr = real(reshape(hist, [PCM_NHB * 4]), 8)
      call comm_allreduce_sum(PCM_NHB * 4, hr, hg)
      ghist = reshape(nint(hg), [PCM_NHB, 4])
      call pcm_zero_hist()
      ntot = sum(ghist(:, 1)); acc = 0; cou = 0.0_rp
      do ib = 1, PCM_NHB
        acc = acc + ghist(ib, 1)
        if (2 * acc >= ntot) then
          cou = 10.0_rp**(-3.0_rp + 0.05_rp * (real(ib, 8) - 0.5_rp))
          exit
        endif
      enddo
      if (cou < imex_pc_cou_on) then
        pcm_k = 0
        pcm_nbuild = pcm_nbuild + 1
        if (talk) write(*,'(1x,A,I7,A,F8.3,A,F6.2,A)') 'IMEX_PCM', iic, ' K 0: median WKB mode-1' &
             // ' (theta_c dt c_1/dx)^2', cou, ' <', imex_pc_cou_on, ' -> Jacobi'
        return
      endif
    endif
    if (.not. pcm_big) call pcm_alloc_big()
    call pcm_zero_build()
#ifdef USE_CUDA
    nch = min(int(nc), 16384)
    allocate(wh(nz, nch), wg(0:nz, nch), wcm(4, nch), wph(nz, 4, nch), wr(4, 7, nch), wz(nz, 4, 4, nch))
    allocate(wxp(nz, 4, nch), wxs(nz, nch), ww(0:nz, nch), wi(4, 5, nch), wd(4, nch))
    do c0 = 1, int(nc), nch
      n = min(nch, int(nc) - c0 + 1)
      call pcm_modes_k<<<(n + PCM_NT - 1) / PCM_NT, PCM_NT, 0, work_stream>>>(c0, n, nch, nxi, nzi, kb, &
           imex_theta_c, dt, imex_n2_ref, imex_n2_reg, hv, int(is), int(ie), int(js), int(je), &
           wh, wg, wcm, wph, wr, wz, wxp, wxs, ww, wi, wd)
    enddo
    call sync_arch()
    deallocate(wh, wg, wcm, wph, wr, wz, wxp, wxs, ww, wi, wd)
#endif
    ! K: medians of the mode Courant numbers (global histograms)
    hist = pcm_hist
    hr = real(reshape(hist, [PCM_NHB * 4]), 8)
    call comm_allreduce_sum(PCM_NHB * 4, hr, hg)
    ghist = reshape(nint(hg), [PCM_NHB, 4])
    med = 0.0d0
    do m = 1, kb
      ntot = sum(ghist(:, m)); acc = 0
      do ib = 1, PCM_NHB
        acc = acc + ghist(ib, m)
        if (2 * acc >= ntot) then
          med(m) = 10.0_rp**(-3.0_rp + 0.05_rp * (real(ib, 8) - 0.5_rp))
          exit
        endif
      enddo
    enddo
    if (imex_pc_modes > 0) then
      pcm_k = kb
    else if (med(1) < imex_pc_cou_on) then
      pcm_k = 0
    else
      pcm_k = 1
      do m = 2, kb
        if (med(m) >= imex_pc_cou_k) pcm_k = m
      enddo
    endif
    pcm_lev = 3
    if (med(1) <= pcm_cou_l3) pcm_lev = 2
    if (imex_pc_levels > 0) pcm_lev = imex_pc_levels
    do m = pcm_k + 1, kb                           ! modes not used
      mm = m
      CUF_DO3
      do k = 1, nz
        do j = js, je
          do i = is, ie
            pcm_phi(i, j, k, mm) = 0.0_rp
          enddo
        enddo
      enddo
    enddo
    do m = 1, pcm_k
      mm = m
      call halo_fill_3d(pcm_phi(:, :, :, m), STAGGER_RHO)
      CUF_DO2
      do j = js, je
        do i = is, ie
          cm1 = 0.0_rp
          do k = 1, nz
            if (pcm_phi(i, j, k, mm) /= 0.0_rp) cm1 = 1.0_rp
          enddo
          pcm_act(i, j, mm) = cm1 * ad_act(i, j)
        enddo
      enddo
      CUF_DO2
      do j = 1, ny                                 ! mass part: Phi^T diag(wr) Phi (centre)
        do i = 1, nx
          cm1 = 0.0_rp
          do k = 1, nz
            cm1 = cm1 + darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k) * pcm_phi(i, j, k, mm)**2
          enddo
          pcm_cm(i, j, mm) = cm1
        enddo
      enddo
      if (pf_prec == 4) then
        CUF_DO3
        do k = 1, nz
          do j = js, je
            do i = is, ie
              pcm_phi4(i, j, k, mm) = real(pcm_phi(i, j, k, mm), 4)
            enddo
          enddo
        enddo
      endif
    enddo
    if (pcm_k > 0) call pcm_probe(c2p, hv, ru, rv)
    pcm_nbuild = pcm_nbuild + 1
    call sync_arch()
    pcm_tbuild = wall_time() - t0
    ! (collective) the build line: K, the median and maximum mode Courant numbers (theta_c dt c_m max(pm, pn))^2
    do m = 1, kb
      mm = m; cm1 = 0.0_rp
      CUF_DO1
      do c = 1, nc
        cm1 = max(cm1, pcm_cmc(c, mm))
      enddo
      cmx(m) = cm1
    enddo
    call comm_allreduce_max(kb, cmx(1:kb), cmg(1:kb))
    cnt = pcm_cnt
    x1 = real(cnt, 8)
    call comm_allreduce_sum(2, x1, x2)
    x1(1) = pcm_tbuild
    call comm_allreduce_max(1, x1(1:1), cmx(1:1))
    if (talk) then
      write(*,'(1x,A,I7,A,I2,A,I2,A,4F8.3)') 'IMEX_PCM', iic, ' K', pcm_k, ' levels', pcm_lev, &
           '  median (theta_c dt c_m/dx)^2 of modes 1..', (med(m), m = 1, kb)
      write(*,'(1x,A,I7,A,4F8.3)') 'IMEX_PCM', iic, ' max (theta_c dt c_m/dx)^2:', &
           ((imex_theta_c * dt * cmg(m))**2, m = 1, kb)
      write(*,'(1x,A,I7,A,I6,A,I8,A,F9.2,A,A)') 'IMEX_PCM', iic, ' modes not converged', &
           nint(x2(1)), ', dropped (column, mode)', nint(x2(2)), ', build', 1.e3_rp * cmx(1), ' ms (max rank)', &
           merge(' -> Jacobi (no mode fast enough)', '                                ', pcm_k == 0)
    endif
  end subroutine pcm_build

  subroutine pcm_zero_hist()
    ! the build's counters = 0 (a kernel: nvfortran 25.9 left the runtime routine of the integer device-array
    ! assignment undeclared in this module)
    integer :: ib, m
    CUF_DO2
    do m = 1, 4
      do ib = 1, PCM_NHB
        pcm_hist(ib, m) = 0
        if (ib == 1 .and. m <= 2) pcm_cnt(m) = 0
      enddo
    enddo
  end subroutine pcm_zero_hist

  subroutine pcm_zero_build()
    ! a new build: the modes, their activity and the operator parts are 0 (CPU pcm_build's first statement)
    integer(ip) :: i, j, k, m, is, ie, js, je, nz, kx
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; kx = pcm_kmax
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          do m = 1, kx
            pcm_phi(i, j, k, m) = 0.0_rp
          enddo
        enddo
      enddo
    enddo
    CUF_DO2
    do j = js, je
      do i = is, ie
        do m = 1, kx
          pcm_act(i, j, m) = 0.0_rp; pcm_cm(i, j, m) = 0.0_rp; pcm_cb(i, j, m) = 0.0_rp
          pcm_ci(i, j, 1, m) = 0.0_rp; pcm_ci(i, j, 2, m) = 0.0_rp; pcm_ci(i, j, 3, m) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine pcm_zero_build

#ifdef USE_CUDA
  attributes(global) subroutine pcm_wkb_k(nc, nx, nz, th, dtv, pi, n2ref, n2reg, hv, is_, ie_, js_, je_)
    ! one thread per tile column: CPU pcm_build's automatic switch, the WKB mode-1 Courant number of the column
    ! into the histogram (log10 bins of 0.05 from 1e-3)
    integer, value :: nc, nx, nz, is_, ie_, js_, je_
    real(rp), value :: th, dtv, pi, n2ref, n2reg
    real(rp), device :: hv(is_:ie_, js_:je_, nz)
    integer :: c, i, j, k, ib, io
    real(rp) :: cou, dz, n2
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nc) return
    i = mod(c - 1, nx) + 1; j = (c - 1) / nx + 1
    if (act(i, j) <= 0.0_rp) return
    cou = 0.0_rp
    do k = 1, nz - 1
      dz = 0.5_rp * (hv(i, j, k) + hv(i, j, k + 1))
      n2 = max((ad_bb(i, j, k + 1) - ad_bb(i, j, k)) / dz, n2ref, n2reg)
      cou = cou + sqrt(n2) * dz
    enddo
    cou = (th * dtv * cou / pi * max(pm(i, j), pn(i, j)))**2
    ib = min(PCM_NHB, max(1, 1 + int((log10(max(cou, 1.e-30_rp)) + 3.0_rp) / 0.05_rp)))
    io = atomicadd(pcm_hist(ib, 1), 1)
  end subroutine pcm_wkb_k

  attributes(global) subroutine pcm_modes_k(c0, n, nch, nx, nz, kb, th, dtv, n2ref, n2reg, hv, is_, ie_, js_, je_, &
                                            wh, wg, wcm, wph, wr, wz, wxp, wxs, ww, wi, wd)
    ! one thread per tile column c0 .. c0+n-1 (work slot t): CPU pcm_build's column loop: the kb fastest modes
    ! of L_z (obm_modes_d), each mapped to the unknown by the column part of D, x_m(k) = rczk(k) [(czt - b) w(k)
    ! + (b - czb) w(k-1)] / wr(k), w = -cumsum(hv phi), made wr-orthogonal to the faster ones (FP64), normalized
    ! (dropped below 1e-6 of its norm), stored rounded to FP32; c_m max(pm, pn) and the Courant histogram
    integer, value :: c0, n, nch, nx, nz, kb, is_, ie_, js_, je_
    real(rp), value :: th, dtv, n2ref, n2reg
    real(rp), device :: hv(is_:ie_, js_:je_, nz)
    real(rp), device :: wh(nz, nch), wg(0:nz, nch), wcm(4, nch), wph(nz, 4, nch), wr(4, 7, nch)
    real(rp), device :: wz(nz, 4, 4, nch), wxp(nz, 4, nch), wxs(nz, nch), ww(0:nz, nch)
    integer, device :: wi(4, 5, nch), wd(4, nch)
    integer :: t, c, i, j, k, m, nn, ib, io
    real(rp) :: dz, n2, wrr, nrm0, nrm, pr, cou
    t = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (t > n) return
    c = c0 - 1 + t
    i = mod(c - 1, nx) + 1; j = (c - 1) / nx + 1
    do m = 1, kb
      pcm_cmc(c, m) = 0.0_rp
    enddo
    if (act(i, j) <= 0.0_rp) return
    do k = 1, nz
      wh(k, t) = hv(i, j, k)
    enddo
    wg(0, t) = 0.0_rp; wg(nz, t) = 0.0_rp
    do k = 1, nz - 1
      dz = 0.5_rp * (hv(i, j, k) + hv(i, j, k + 1))
      n2 = max((ad_bb(i, j, k + 1) - ad_bb(i, j, k)) / dz, n2ref, n2reg)
      wg(k, t) = 1.0_rp / (n2 * dz)
    enddo
    call obm_modes_d(nz, kb, wh(1, t), wg(0, t), wcm(1, t), wph(1, 1, t), &
         wr(1, 1, t), wr(1, 2, t), wr(1, 3, t), wr(1, 4, t), wr(1, 5, t), wr(1, 6, t), &
         wi(1, 1, t), wi(1, 2, t), wi(1, 3, t), wi(1, 4, t), wi(1, 5, t), &
         wz(1, 1, 1, t), wz(1, 1, 2, t), wz(1, 1, 3, t), wz(1, 1, 4, t), wd(1, t), wd(2, t), &
         wd(3, t), wr(1, 7, t), wd(4, t))
    if (wd(4, t) /= 1) io = atomicadd(pcm_cnt(1), 1)
    do m = 1, kb
      ww(0, t) = 0.0_rp
      do k = 1, nz
        ww(k, t) = ww(k - 1, t) - wh(k, t) * wph(k, m, t)
      enddo
      ww(nz, t) = 0.0_rp
      nrm0 = 0.0_rp
      do k = 1, nz
        wrr = darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k)
        wxs(k, t) = -act(i, j) / ad_n2k(i, j, k) * ((ad_czt(i, j, k) - ad_bb(i, j, k)) * ww(k, t) &
             + (ad_bb(i, j, k) - ad_czb(i, j, k)) * ww(k - 1, t)) / wrr
        nrm0 = nrm0 + wrr * wxs(k, t)**2
      enddo
      nrm0 = sqrt(nrm0)
      do nn = 1, m - 1                             ! wr-orthogonal to the faster modes (FP64)
        pr = 0.0_rp
        do k = 1, nz
          pr = pr + darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k) * wxp(k, nn, t) * wxs(k, t)
        enddo
        do k = 1, nz
          wxs(k, t) = wxs(k, t) - pr * wxp(k, nn, t)
        enddo
      enddo
      nrm = 0.0_rp
      do k = 1, nz
        nrm = nrm + darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k) * wxs(k, t)**2
      enddo
      nrm = sqrt(nrm)
      if (nrm0 > 0.0_rp .and. nrm > 1.e-6_rp * nrm0) then
        do k = 1, nz
          wxs(k, t) = wxs(k, t) / nrm
        enddo
      else
        do k = 1, nz
          wxs(k, t) = 0.0_rp
        enddo
        io = atomicadd(pcm_cnt(2), 1)
      endif
      do k = 1, nz
        wxp(k, m, t) = wxs(k, t)
        pcm_phi(i, j, k, m) = real(real(wxs(k, t), 4), 8)   ! modes stored in FP32
      enddo
      pcm_cmc(c, m) = wcm(m, t) * max(pm(i, j), pn(i, j))
      cou = (th * dtv * wcm(m, t) * max(pm(i, j), pn(i, j)))**2
      ib = min(PCM_NHB, max(1, 1 + int((log10(max(cou, 1.e-30_rp)) + 3.0_rp) / 0.05_rp)))
      io = atomicadd(pcm_hist(ib, m), 1)
    enddo
  end subroutine pcm_modes_k
#endif

  subroutine pcm_probe(c2p, hv, ru, rv)
    ! (CPU pcm_build's probing) the parts of a_m by the FP64 operator sweep pf_op8 (mode 0: W M of an FP64
    ! operand) with its own FP64 set (rczk = -act/n2k on the tile and its ring, wr = W/n2c): colour c's columns of
    ! mode m (colour (i + 2 j) mod 5 on global indices); at a column of colour col the stencil member of colour c
    ! is offset o = (c - col) mod 5: 0 centre, 1 east (i+1), 2 north (j+1) (3 south, 4 west not stored). The
    ! interior part with the 'modal' faces off (imex_obc_modal off, cxm, cym = cxo, cyo: the CPU's operator
    ! without obm_ufx/obm_vfx), the 'modal' part (column-local, centre only) as the difference with them on
    real(rp), intent(in) :: c2p
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: ru(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rv(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR allocatable :: rcz(:,:,:), wrz(:,:,:), y2(:,:,:,:), cxs(:,:), cys(:,:)
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je, mm, cc, gi0, gj0, o, sm2
    integer :: m, c
    real(rp) :: sc, pq, s1, s2, c2l
    logical :: smod
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    gi0 = decomp%i0 - 1; gj0 = decomp%j0 - 1      ! the CPU's iSW_corn, jSW_corn
    allocate(rcz(is:ie, js:je, nz), wrz(is:ie, js:je, nz))
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          rcz(i, j, k) = 0.0_rp; wrz(i, j, k) = 0.0_rp
          if (i >= 0 .and. i <= nx + 1 .and. j >= 0 .and. j <= ny + 1) rcz(i, j, k) = -ad_act(i, j) / ad_n2k(i, j, k)
          if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) &
               wrz(i, j, k) = darea(i, j) * hv(i, j, k) / ad_n2c(i, j, k)
        enddo
      enddo
    enddo
    sc = sqrt(c2p); c2l = c2p
    smod = imex_obc_modal
    sm2 = merge(1_ip, 0_ip, smod)
    allocate(y2(is:ie, js:je, nz, 3))              ! (read by the kernel below only with smod)
    if (smod) then
      allocate(cxs(is:ie, js:je), cys(is:ie, js:je))
      CUF_DO2
      do j = js, je
        do i = is, ie
          cxs(i, j) = cxm(i, j); cys(i, j) = cym(i, j)
        enddo
      enddo
    endif
    do m = 1, pcm_k
      mm = m
      do c = 0, 4
        cc = c
        CUF_DO3
        do k = 1, nz
          do j = 0, ny + 1
            do i = 0, nx + 1
              pcm_t3(i, j, k) = 0.0_rp
              if (modulo(gi0 + i + 2 * (gj0 + j), 5_ip) == cc) pcm_t3(i, j, k) = pcm_phi(i, j, k, mm)
            enddo
          enddo
        enddo
        if (smod) then                             ! interior part: the operator without the 'modal' faces
          imex_obc_modal = .false.
          CUF_DO2
          do j = js, je
            do i = is, ie
              cxm(i, j) = cxo(i, j); cym(i, j) = cyo(i, j)
            enddo
          enddo
        endif
        call pf_op8(0, 0.0_rp, 0.0_rp, c2l, pcm_t3, pf_mq, wk_z, wk_p, wk_p, wk_dp, wk_q, rcz, ad_czt, ad_czb, &
                    ad_cu, ad_cv, hv, wrz, ru, rv, pq)
        if (smod) then                             ! + the 'modal' faces
          CUF_DO2
          do j = js, je
            do i = is, ie
              cxm(i, j) = cxs(i, j); cym(i, j) = cys(i, j)
            enddo
          enddo
          imex_obc_modal = .true.
          call pf_op8(0, 0.0_rp, 0.0_rp, c2l, pcm_t3, y2, wk_z, wk_p, wk_p, wk_dp, wk_q, rcz, ad_czt, ad_czb, &
                      ad_cu, ad_cv, hv, wrz, ru, rv, pq)
        endif
        CUF_DO2
        do j = 1, ny
          do i = 1, nx
            o = modulo(cc - modulo(gi0 + i + 2 * (gj0 + j), 5_ip), 5_ip)
            if (o <= 2) then
              s1 = 0.0_rp; s2 = 0.0_rp
              do k = 1, nz
                s1 = s1 + pcm_phi(i, j, k, mm) * (pf_mq(i, j, k, 3) - wrz(i, j, k) * pcm_t3(i, j, k))
                if (sm2 == 1) s2 = s2 + pcm_phi(i, j, k, mm) * (y2(i, j, k, 3) - pf_mq(i, j, k, 3))
              enddo
              pcm_ci(i, j, o + 1, mm) = s1 / c2l
              if (o == 0 .and. sm2 == 1) pcm_cb(i, j, mm) = s2 / sc
            endif
          enddo
        enddo
      enddo
      call halo_fill_2d(pcm_ci(:, :, 2, m), STAGGER_RHO)   ! east, north couplings of the halo columns
      call halo_fill_2d(pcm_ci(:, :, 3, m), STAGGER_RHO)
    enddo
    call sync_arch()
    deallocate(rcz)
    deallocate(wrz)
    deallocate(y2)
    if (smod) then
      deallocate(cxs)
      deallocate(cys)
    endif
  end subroutine pcm_probe

  subroutine pcm_stage(stage, c2, hv)
    ! (CPU pcm_stage) per stage and c2: d_K = diag(M) - sum_m lam_m W (Phi_m/n2c)^2, lam_m = c2 a_int(centre) +
    ! sqrt(c2) a_modal(centre), floored at 1/n2c, its reciprocal act/d_K (pcm_dki); G = Phi^T W d_K Phi and G^-1
    ! per column; the V-cycle levels of each mode (level 0: pcm_fine; 1, 2: the free surface's coarsening with
    ! the 'width' face factors)
    integer, intent(in) :: stage
    real(rp), intent(in) :: c2
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, m
    real(rp) :: c2l, sc
    integer :: l
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    c2l = c2; sc = sqrt(c2)
    CUF_DO3
    do k = 1, nz                                   ! diag(M) (+ the 'modal' boundary term below)
      do j = 1, ny
        do i = 1, nx
          pcm_t3(i, j, k) = 1.0_rp / ad_n2c(i, j, k) + c2l * ad_dd(i, j, k)
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    if (imex_obc_modal .and. ob_np > 0 .and. ob_nc > 0) &
      call pcm_kd_k<<<(ob_np + OBM_NT - 1) / OBM_NT, OBM_NT, 0, work_stream>>>(ob_np, int(nz), sc)
    call pcm_gk_k<<<(int(nx * ny) + PCM_NT - 1) / PCM_NT, PCM_NT, 0, work_stream>>>(int(nx), int(ny), int(nz), &
         pcm_k, stage, c2l, sc, hv, int(decomp%is), int(decomp%ie), int(decomp%js), int(decomp%je))
#else
    error stop 'imex_igw: imex_pc = modes needs the CUDA build'
#endif
    call pcm_fine(stage, c2l, sc)
    do l = 0, pcm_nl - 2
      call pcm_coarsen(l, stage)
    enddo
  end subroutine pcm_stage

#ifdef USE_CUDA
  attributes(global) subroutine pcm_kd_k(np, nz, sc)
    ! the 'modal' boundary term of diag(M) on the cells next to the open faces: kd = 0 + the face columns'
    ! terms in the order W, E, S, N (as obm_dinv8_k); diag += sqrt(c2) kd where kd > 0 (CPU pcm_stage's ob_kd)
    integer, value :: np, nz
    real(rp), value :: sc
    integer :: p, i, j, k, e, c
    real(rp) :: kd
    p = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (p > np) return
    i = ob_pi(p); j = ob_pj(p)
    do k = 1, nz
      kd = 0.0_rp
      do e = 1, 4
        c = ob_pc(e, p)
        if (c > 0) then
          if (ob_act(c) > 0.0_rp) kd = kd + ob_kdc(k, c)
        endif
      enddo
      if (kd > 0.0_rp) pcm_t3(i, j, k) = pcm_t3(i, j, k) + sc * kd
    enddo
  end subroutine pcm_kd_k

  attributes(global) subroutine pcm_gk_k(nx, ny, nz, kk, s, c2, sc, hv, is_, ie_, js_, je_)
    ! one thread per tile column: d_K and its reciprocal, G and G^-1 (Gauss-Jordan without pivoting, CPU
    ! pcm_inv; inactive modes: unit row and column, zero in G^-1)
    integer, value :: nx, ny, nz, kk, s, is_, ie_, js_, je_
    real(rp), value :: c2, sc
    real(rp), device :: hv(is_:ie_, js_:je_, nz)
    integer :: c, i, j, k, m, n, ii, jj
    real(rp) :: lam(4), g(4, 4), w(4, 8), dm, dk, ww, p, f
    logical :: am(4)
    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nx * ny) return
    i = mod(c - 1, nx) + 1; j = (c - 1) / nx + 1
    do m = 1, kk
      lam(m) = c2 * pcm_ci(i, j, 1, m) + sc * pcm_cb(i, j, m)
      am(m) = pcm_act(i, j, m) > 0.0_rp
    enddo
    do m = 1, 4
      do n = 1, 4
        g(n, m) = 0.0_rp
      enddo
    enddo
    do k = 1, nz
      dm = pcm_t3(i, j, k)
      ww = darea(i, j) * hv(i, j, k)
      dk = dm
      do m = 1, kk
        dk = dk - lam(m) * ww * (pcm_phi(i, j, k, m) / ad_n2c(i, j, k))**2
      enddo
      dk = max(dk, 1.0_rp / ad_n2c(i, j, k))
      pcm_dki(i, j, k, s) = act(i, j) / dk
      do m = 1, kk
        do n = 1, kk
          g(n, m) = g(n, m) + pcm_phi(i, j, k, n) * pcm_phi(i, j, k, m) * ww * dk
        enddo
      enddo
    enddo
    do m = 1, kk                                   ! inactive modes: unit row and column
      if (am(m)) cycle
      do n = 1, kk
        g(m, n) = 0.0_rp; g(n, m) = 0.0_rp
      enddo
      g(m, m) = 1.0_rp
    enddo
    do ii = 1, 4
      do jj = 1, 8
        w(ii, jj) = 0.0_rp
      enddo
    enddo
    do ii = 1, kk
      do jj = 1, kk
        w(ii, jj) = g(ii, jj)
      enddo
      w(ii, kk + ii) = 1.0_rp
    enddo
    do ii = 1, kk
      p = 1.0_rp / w(ii, ii)
      do jj = 1, 2 * kk
        w(ii, jj) = w(ii, jj) * p
      enddo
      do n = 1, kk
        if (n == ii) cycle
        f = w(n, ii)
        do jj = 1, 2 * kk
          w(n, jj) = w(n, jj) - f * w(ii, jj)
        enddo
      enddo
    enddo
    do m = 1, kk
      do n = 1, kk
        if (am(m) .and. am(n)) then
          pcm_gi(i, j, n, m, s) = w(n, kk + m)
        else
          pcm_gi(i, j, n, m, s) = 0.0_rp
        endif
      enddo
    enddo
  end subroutine pcm_gk_k
#endif

  subroutine pcm_fine(s, c2, sc)
    ! (CPU pcm_fine) level 0 of each mode in the format of the level kernels: A x = e x + sum_faces c (x - x_nb),
    ! c the face conductance = minus the coupling (w west faces 1..nx+1, s south faces 1..ny+1), e the row sum
    ! (floored at 1e-12 diag on active cells), d = 1/diag (0 on inactive cells); the copies with the row sums
    ! and conductances clamped at 0 that level 1 is coarsened from in lv_x, lv_b, lv_t (level 0 planes, free
    ! until the first V-cycle)
    integer, intent(in) :: s
    real(rp), intent(in) :: c2, sc
    integer(ip) :: i, j, m, nx, ny, kk, ss
    real(rp) :: c2l, scl, d, rs, cwv, csv
    nx = pcm_nx(0); ny = pcm_ny(0); kk = pcm_k; ss = s; c2l = c2; scl = sc
    CUF_DO3
    do m = 1, kk
      do j = 0, ny + 1
        do i = 0, nx + 1
          lv_e(i, j, m, 0, ss) = 0.0_rp; lv_d(i, j, m, 0, ss) = 0.0_rp; lv_x(i, j, m, 0) = 0.0_rp
          cwv = 0.0_rp; csv = 0.0_rp
          if (j >= 1 .and. j <= ny .and. i >= 1 .and. i <= nx + 1) &
               cwv = -c2l * pcm_ci(i - 1, j, 2, m) * pcm_act(i - 1, j, m) * pcm_act(i, j, m)
          if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny + 1) &
               csv = -c2l * pcm_ci(i, j - 1, 3, m) * pcm_act(i, j - 1, m) * pcm_act(i, j, m)
          lv_w(i, j, m, 0, ss) = cwv; lv_s(i, j, m, 0, ss) = csv
          lv_b(i, j, m, 0) = 0.0_rp; lv_t(i, j, m, 0) = 0.0_rp
          if (i >= 1 .and. i <= nx + 1 .and. j >= 1 .and. j <= ny + 1) then
            lv_b(i, j, m, 0) = max(cwv, 0.0_rp); lv_t(i, j, m, 0) = max(csv, 0.0_rp)
          endif
        enddo
      enddo
    enddo
    CUF_DO3
    do m = 1, kk
      do j = 1, ny
        do i = 1, nx
          if (pcm_act(i, j, m) > 0.0_rp) then
            d = pcm_cm(i, j, m) + c2l * pcm_ci(i, j, 1, m) + scl * pcm_cb(i, j, m)
            if (d > 0.0_rp) then
              rs = d - (lv_w(i, j, m, 0, ss) + lv_w(i + 1, j, m, 0, ss) + lv_s(i, j, m, 0, ss) &
                   + lv_s(i, j + 1, m, 0, ss))
              lv_x(i, j, m, 0) = max(rs, 0.0_rp)
              lv_e(i, j, m, 0, ss) = max(rs, 1.e-12_rp * d)
              lv_d(i, j, m, 0, ss) = 1.0_rp / d
            endif
          endif
        enddo
      enddo
    enddo
  end subroutine pcm_fine

  subroutine pcm_coarsen(l, s)
    ! (implicit_fs_mod's mg_coarsen) level l+1 of each mode from level l (l = 0: the clamped copies of
    ! pcm_fine): e the sum of the aggregate's row sums, the coarse face conductances the factor (pcm_ffx, pcm_ffy:
    ! the 'width' rule) times the sum of the fine faces on the aggregate's side, d = 1/(e + sum of the faces) on
    ! cells with e > 0 (mg_diag); the sums in the CPU's order
    integer, intent(in) :: l, s
    integer(ip) :: ic, jc, i, j, m, nxf, nyf, nxc, nyc, kk, ss, ll, fi, fj, l0
    real(rp) :: ev, cwv, csv
    nxf = pcm_nx(l); nyf = pcm_ny(l); nxc = pcm_nx(l + 1); nyc = pcm_ny(l + 1)
    kk = pcm_k; ss = s; ll = l; l0 = merge(1_ip, 0_ip, l == 0)
    CUF_DO3
    do m = 1, kk
      do jc = 0, nyc + 1
        do ic = 0, nxc + 1
          ev = 0.0_rp; cwv = 0.0_rp; csv = 0.0_rp
          if (jc >= 1 .and. jc <= nyc .and. ic >= 1 .and. ic <= nxc) then
            do j = 2 * jc - 1, min(2 * jc, nyf)
              do i = 2 * ic - 1, min(2 * ic, nxf)
                if (l0 == 1) then
                  ev = ev + lv_x(i, j, m, 0)
                else
                  ev = ev + lv_e(i, j, m, ll, ss)
                endif
              enddo
            enddo
          endif
          if (jc >= 1 .and. jc <= nyc .and. ic >= 1 .and. ic <= nxc + 1) then
            fi = min(2 * ic - 1, nxf + 1)
            do j = 2 * jc - 1, min(2 * jc, nyf)
              if (l0 == 1) then
                cwv = cwv + pcm_ffx(ic, ll + 1) * lv_b(fi, j, m, 0)
              else
                cwv = cwv + pcm_ffx(ic, ll + 1) * lv_w(fi, j, m, ll, ss)
              endif
            enddo
          endif
          if (ic >= 1 .and. ic <= nxc .and. jc >= 1 .and. jc <= nyc + 1) then
            fj = min(2 * jc - 1, nyf + 1)
            do i = 2 * ic - 1, min(2 * ic, nxf)
              if (l0 == 1) then
                csv = csv + pcm_ffy(jc, ll + 1) * lv_t(i, fj, m, 0)
              else
                csv = csv + pcm_ffy(jc, ll + 1) * lv_s(i, fj, m, ll, ss)
              endif
            enddo
          endif
          lv_e(ic, jc, m, ll + 1, ss) = ev; lv_w(ic, jc, m, ll + 1, ss) = cwv; lv_s(ic, jc, m, ll + 1, ss) = csv
        enddo
      enddo
    enddo
    CUF_DO3
    do m = 1, kk
      do jc = 0, nyc + 1
        do ic = 0, nxc + 1
          lv_d(ic, jc, m, ll + 1, ss) = 0.0_rp
          if (jc >= 1 .and. jc <= nyc .and. ic >= 1 .and. ic <= nxc) then
            if (lv_e(ic, jc, m, ll + 1, ss) > 0.0_rp) lv_d(ic, jc, m, ll + 1, ss) = 1.0_rp &
                 / (lv_e(ic, jc, m, ll + 1, ss) + lv_w(ic, jc, m, ll + 1, ss) + lv_w(ic + 1, jc, m, ll + 1, ss) &
                 + lv_s(ic, jc, m, ll + 1, ss) + lv_s(ic, jc + 1, m, ll + 1, ss))
          endif
        enddo
      enddo
    enddo
  end subroutine pcm_coarsen

  subroutine pcm_dinv(s, hv, prec)
    ! pf_setup8/4 with 'modes' and K > 0: the deflated diagonal act/(W d_K) replaces the Jacobi reciprocal on
    ! the active cells (CPU imex_pf_setup8/4)
    integer, intent(in) :: s, prec
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, ss
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ss = s
    if (prec == 8) then
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            if (act(i, j) > 0.0_rp) pf_dinv8(i, j, k, ss) = pcm_dki(i, j, k, ss) / (darea(i, j) * hv(i, j, k))
          enddo
        enddo
      enddo
    else
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            if (act(i, j) > 0.0_rp) pf_dinv4(i, j, k, ss) = real(pcm_dki(i, j, k, ss) / (darea(i, j) * hv(i, j, k)), 4)
          enddo
        enddo
      enddo
    endif
  end subroutine pcm_dinv

  !----------------------------------------------------------------------
  ! the V-cycle (CPU pcm_vcycle on implicit_fs_mod's mg_smooth, mg_resid, mg_restrict, mg_prolong; pcm_exchk)
  !----------------------------------------------------------------------
  subroutine pcm_vcycle(s)
    ! lv_x(:, :, m, 0) = one V-cycle on mode m's 2-D operator applied to rc = lv_b(:, :, m, 0), from zero, for
    ! the K modes together: per level a pre-sweep from zero (no exchange), exchange, residual and sum
    ! restriction; imex_pc_ncoarse pre- and post-sweeps on the coarsest level (an exchange before each but the
    ! first); prolongation, exchange, a post-sweep. 'rbgs': each red-black sweep one kernel from one buffer into
    ! the other (pcm_sw2; lv_x, lv_t alternate per level, the last sweep of level 0 lands in lv_x); 'jacobi':
    ! pcm_vcycle_j
    integer, intent(in) :: s
    integer :: k, b1, b2
    if (pcm_jac) then
      call pcm_vcycle_j(s)
      return
    endif
    if (.not. pcm_capt) call itw2(32, 0)
    call pcm_sw2(0, 1, s, .true., lv_x, lv_t)        ! level 0 pre-sweep from zero into lv_t
    call pcm_exch2(0, lv_t)
    call pcm_rr2(0, s, lv_t)
    call pcm_sw2(1, 1, s, .true., lv_t, lv_x)        ! level 1 pre-sweep from zero into lv_x
    b1 = 1                                           ! level 1 in lv_x (1) or lv_t (2)
    if (pcm_lev == 2) then                           ! level 1 is the coarsest
      do k = 2, 2 * imex_pc_ncoarse
        if (b1 == 1) then
          call pcm_exch2(1, lv_x)
          call pcm_sw2(1, merge(1, 2, k <= imex_pc_ncoarse), s, .false., lv_x, lv_t)
        else
          call pcm_exch2(1, lv_t)
          call pcm_sw2(1, merge(1, 2, k <= imex_pc_ncoarse), s, .false., lv_t, lv_x)
        endif
        b1 = 3 - b1
      enddo
      if (b1 == 1) then
        call pcm_prolong2(0, s, lv_t, lv_x)
      else
        call pcm_prolong2(0, s, lv_t, lv_t)
      endif
      call pcm_exch2(0, lv_t)
      call pcm_sw2(0, 2, s, .false., lv_t, lv_x)
      if (.not. pcm_capt) call itw2(32, 1)
      return
    endif
    call pcm_exch2(1, lv_x)
    call pcm_rr2(1, s, lv_x)
    call pcm_sw2(2, 1, s, .true., lv_t, lv_x)        ! level 2 pre-sweep from zero into lv_x
    b2 = 1
    do k = 2, 2 * imex_pc_ncoarse
      if (b2 == 1) then
        call pcm_exch2(2, lv_x)
        call pcm_sw2(2, merge(1, 2, k <= imex_pc_ncoarse), s, .false., lv_x, lv_t)
      else
        call pcm_exch2(2, lv_t)
        call pcm_sw2(2, merge(1, 2, k <= imex_pc_ncoarse), s, .false., lv_t, lv_x)
      endif
      b2 = 3 - b2
    enddo
    if (b2 == 1) then
      call pcm_prolong2(1, s, lv_x, lv_x)
    else
      call pcm_prolong2(1, s, lv_x, lv_t)
    endif
    call pcm_exch2(1, lv_x)
    call pcm_sw2(1, 2, s, .false., lv_x, lv_t)       ! level 1 post-sweep into lv_t
    call pcm_prolong2(0, s, lv_t, lv_t)
    call pcm_exch2(0, lv_t)
    call pcm_sw2(0, 2, s, .false., lv_t, lv_x)
    if (.not. pcm_capt) call itw2(32, 1)
  end subroutine pcm_vcycle

  subroutine pcm_vcycle_j(s)
    ! pcm_vcycle with imex_pc_smoother = 'jacobi' (in place in lv_x, lv_t the smoother's work array)
    integer, intent(in) :: s
    integer :: k
    if (.not. pcm_capt) call itw2(32, 0)
    call pcm_zero0()
    call pcm_smooth(0, 1, s)
    call pcm_exch(0)
    call pcm_rr(0, s)
    call pcm_smooth(1, 1, s)
    if (pcm_lev == 2) then
      do k = 2, 2 * imex_pc_ncoarse
        call pcm_exch(1)
        call pcm_smooth(1, merge(1, 2, k <= imex_pc_ncoarse), s)
      enddo
      call pcm_prolong(0, s)
      call pcm_exch(0)
      call pcm_smooth(0, 2, s)
      if (.not. pcm_capt) call itw2(32, 1)
      return
    endif
    call pcm_exch(1)
    call pcm_rr(1, s)
    call pcm_smooth(2, 1, s)
    do k = 2, 2 * imex_pc_ncoarse
      call pcm_exch(2)
      call pcm_smooth(2, merge(1, 2, k <= imex_pc_ncoarse), s)
    enddo
    call pcm_prolong(1, s)
    call pcm_exch(1)
    call pcm_smooth(1, 2, s)
    call pcm_prolong(0, s)
    call pcm_exch(0)
    call pcm_smooth(0, 2, s)
    if (.not. pcm_capt) call itw2(32, 1)
  end subroutine pcm_vcycle_j

  subroutine pcm_sw2(l, mode, s, fz, xs, xd)
    ! 'rbgs': one red-black sweep of level l of the K modes in one kernel, from xs (halo current; fz: from zero,
    ! xs's values taken as 0) into xd (interior): a first-colour cell gets mg_smooth's expression on xs; a
    ! second-colour cell forms the new values of its four first-colour neighbours with the same expression (xs's
    ! value where the neighbour is a halo cell), then its own (as fs_mg's sweep_c2): the values of mg_smooth's two
    ! colour passes (mode 1: colour 0 first, mode 2: colour 1 first)
    integer, intent(in) :: l, mode, s
    logical, intent(in) :: fz
    real(rp), PCMDEV intent(in) :: xs(0:pcm_nx(0) + 1, 0:pcm_ny(0) + 1, pcm_kmax, 0:pcm_nl - 1)
    real(rp), PCMDEV intent(inout) :: xd(0:pcm_nx(0) + 1, 0:pcm_ny(0) + 1, pcm_kmax, 0:pcm_nl - 1)
    integer(ip) :: i, j, m, nxl, nyl, kk, ll, ss, p1, z
    real(rp) :: zr, vw, ve, vs, vn, a1, a2, a3, a4
    nxl = pcm_nx(l); nyl = pcm_ny(l); kk = pcm_k; ll = l; ss = s; p1 = mod(1 + mode, 2)
    z = merge(1_ip, 0_ip, fz); zr = fs_zero
    CUF_DO3
    do m = 1, kk
      do j = 1, nyl
        do i = 1, nxl
          if (mod(i + j + p1, 2_ip) == 1) then
            a1 = merge(zr, xs(i - 1, j, m, ll), z == 1); a2 = merge(zr, xs(i + 1, j, m, ll), z == 1)
            a3 = merge(zr, xs(i, j - 1, m, ll), z == 1); a4 = merge(zr, xs(i, j + 1, m, ll), z == 1)
            xd(i, j, m, ll) = lv_d(i, j, m, ll, ss) * (lv_b(i, j, m, ll) + lv_w(i, j, m, ll, ss) * a1 &
                 + lv_w(i + 1, j, m, ll, ss) * a2 + lv_s(i, j, m, ll, ss) * a3 &
                 + lv_s(i, j + 1, m, ll, ss) * a4)
          else
            if (i > 1) then
              a1 = merge(zr, xs(i - 2, j, m, ll), z == 1); a2 = merge(zr, xs(i, j, m, ll), z == 1)
              a3 = merge(zr, xs(i - 1, j - 1, m, ll), z == 1); a4 = merge(zr, xs(i - 1, j + 1, m, ll), z == 1)
              vw = lv_d(i - 1, j, m, ll, ss) * (lv_b(i - 1, j, m, ll) + lv_w(i - 1, j, m, ll, ss) * a1 &
                   + lv_w(i, j, m, ll, ss) * a2 + lv_s(i - 1, j, m, ll, ss) * a3 &
                   + lv_s(i - 1, j + 1, m, ll, ss) * a4)
            else
              vw = merge(zr, xs(i - 1, j, m, ll), z == 1)
            endif
            if (i < nxl) then
              a1 = merge(zr, xs(i, j, m, ll), z == 1); a2 = merge(zr, xs(i + 2, j, m, ll), z == 1)
              a3 = merge(zr, xs(i + 1, j - 1, m, ll), z == 1); a4 = merge(zr, xs(i + 1, j + 1, m, ll), z == 1)
              ve = lv_d(i + 1, j, m, ll, ss) * (lv_b(i + 1, j, m, ll) + lv_w(i + 1, j, m, ll, ss) * a1 &
                   + lv_w(i + 2, j, m, ll, ss) * a2 + lv_s(i + 1, j, m, ll, ss) * a3 &
                   + lv_s(i + 1, j + 1, m, ll, ss) * a4)
            else
              ve = merge(zr, xs(i + 1, j, m, ll), z == 1)
            endif
            if (j > 1) then
              a1 = merge(zr, xs(i - 1, j - 1, m, ll), z == 1); a2 = merge(zr, xs(i + 1, j - 1, m, ll), z == 1)
              a3 = merge(zr, xs(i, j - 2, m, ll), z == 1); a4 = merge(zr, xs(i, j, m, ll), z == 1)
              vs = lv_d(i, j - 1, m, ll, ss) * (lv_b(i, j - 1, m, ll) + lv_w(i, j - 1, m, ll, ss) * a1 &
                   + lv_w(i + 1, j - 1, m, ll, ss) * a2 + lv_s(i, j - 1, m, ll, ss) * a3 &
                   + lv_s(i, j, m, ll, ss) * a4)
            else
              vs = merge(zr, xs(i, j - 1, m, ll), z == 1)
            endif
            if (j < nyl) then
              a1 = merge(zr, xs(i - 1, j + 1, m, ll), z == 1); a2 = merge(zr, xs(i + 1, j + 1, m, ll), z == 1)
              a3 = merge(zr, xs(i, j, m, ll), z == 1); a4 = merge(zr, xs(i, j + 2, m, ll), z == 1)
              vn = lv_d(i, j + 1, m, ll, ss) * (lv_b(i, j + 1, m, ll) + lv_w(i, j + 1, m, ll, ss) * a1 &
                   + lv_w(i + 1, j + 1, m, ll, ss) * a2 + lv_s(i, j + 1, m, ll, ss) * a3 &
                   + lv_s(i, j + 2, m, ll, ss) * a4)
            else
              vn = merge(zr, xs(i, j + 1, m, ll), z == 1)
            endif
            xd(i, j, m, ll) = lv_d(i, j, m, ll, ss) * (lv_b(i, j, m, ll) + lv_w(i, j, m, ll, ss) * vw &
                 + lv_w(i + 1, j, m, ll, ss) * ve + lv_s(i, j, m, ll, ss) * vs + lv_s(i, j + 1, m, ll, ss) * vn)
          endif
        enddo
      enddo
    enddo
  end subroutine pcm_sw2

  subroutine pcm_rr2(l, s, x)
    ! pcm_rr on the buffer x (the coarse level's x is set by its pre-sweep from zero)
    integer, intent(in) :: l, s
    real(rp), PCMDEV intent(in) :: x(0:pcm_nx(0) + 1, 0:pcm_ny(0) + 1, pcm_kmax, 0:pcm_nl - 1)
    integer(ip) :: ic, jc, i, j, m, nxf, nyf, nxc, nyc, kk, ll, ss
    real(rp) :: sm, r
    nxf = pcm_nx(l); nyf = pcm_ny(l); nxc = pcm_nx(l + 1); nyc = pcm_ny(l + 1)
    kk = pcm_k; ll = l; ss = s
    CUF_DO3
    do m = 1, kk
      do jc = 1, nyc
        do ic = 1, nxc
          sm = 0.0_rp
          do j = 2 * jc - 1, min(2 * jc, nyf)
            do i = 2 * ic - 1, min(2 * ic, nxf)
              r = 0.0_rp
              if (lv_e(i, j, m, ll, ss) > 0.0_rp) r = lv_b(i, j, m, ll) - (lv_e(i, j, m, ll, ss) * x(i, j, m, ll) &
                   + lv_w(i, j, m, ll, ss) * (x(i, j, m, ll) - x(i - 1, j, m, ll)) &
                   + lv_w(i + 1, j, m, ll, ss) * (x(i, j, m, ll) - x(i + 1, j, m, ll)) &
                   + lv_s(i, j, m, ll, ss) * (x(i, j, m, ll) - x(i, j - 1, m, ll)) &
                   + lv_s(i, j + 1, m, ll, ss) * (x(i, j, m, ll) - x(i, j + 1, m, ll)))
              sm = sm + r
            enddo
          enddo
          lv_b(ic, jc, m, ll + 1) = sm
        enddo
      enddo
    enddo
  end subroutine pcm_rr2

  subroutine pcm_prolong2(l, s, xf, xc)
    ! pcm_prolong: xf of level l += xc of level l+1 (xf and xc may be the same buffer: other levels)
    integer, intent(in) :: l, s
    real(rp), PCMDEV intent(inout) :: xf(0:pcm_nx(0) + 1, 0:pcm_ny(0) + 1, pcm_kmax, 0:pcm_nl - 1)
    real(rp), PCMDEV intent(in) :: xc(0:pcm_nx(0) + 1, 0:pcm_ny(0) + 1, pcm_kmax, 0:pcm_nl - 1)
    integer(ip) :: i, j, m, nxf, nyf, kk, ll, ss
    nxf = pcm_nx(l); nyf = pcm_ny(l); kk = pcm_k; ll = l; ss = s
    CUF_DO3
    do m = 1, kk
      do j = 1, nyf
        do i = 1, nxf
          if (lv_e(i, j, m, ll, ss) > 0.0_rp) xf(i, j, m, ll) = xf(i, j, m, ll) &
               + xc((i + 1) / 2, (j + 1) / 2, m, ll + 1)
        enddo
      enddo
    enddo
  end subroutine pcm_prolong2

  subroutine pcm_exch2(l, x)
    ! pcm_exch on the buffer x
    integer, intent(in) :: l
    real(rp), PCMDEV intent(inout) :: x(0:pcm_nx(0) + 1, 0:pcm_ny(0) + 1, pcm_kmax, 0:pcm_nl - 1)
    integer(ip) :: nxl, nyl, mx, m, kk, ll, w, e, s, n, nb(4), q
    integer :: mi, ierr
    nxl = pcm_nx(l); nyl = pcm_ny(l); kk = pcm_k; ll = l
    nb = [decomp%nbr_w, decomp%nbr_e, decomp%nbr_s, decomp%nbr_n]
    mx = max(nxl, nyl)
    if (all(nb == NBR_NONE)) then
      CUF_DO2
      do m = 1, kk
        do q = 1, mx
          if (q <= nyl) then
            x(0, q, m, ll) = 0.0_rp; x(nxl + 1, q, m, ll) = 0.0_rp
          endif
          if (q <= nxl) then
            x(q, 0, m, ll) = 0.0_rp; x(q, nyl + 1, m, ll) = 0.0_rp
          endif
        enddo
      enddo
      return
    endif
#ifdef USE_CUDA
    if (ipc_on) then
      do mi = 1, pcm_k
        call ipc_ring_plane(x, pcm_nx(0), pcm_ny(0), mi + l * pcm_kmax, int(nxl), int(nyl))
      enddo
      return
    endif
#endif
    CUF_DO2
    do m = 1, kk
      do q = 1, mx
        if (q <= nyl) then
          pcm_rsw(q + (m - 1) * nyl) = x(1, q, m, ll); pcm_rse(q + (m - 1) * nyl) = x(nxl, q, m, ll)
        endif
        if (q <= nxl) then
          pcm_rss(q + (m - 1) * nxl) = x(q, 1, m, ll); pcm_rsn(q + (m - 1) * nxl) = x(q, nyl, m, ll)
        endif
      enddo
    enddo
    if (any(nb >= 0)) then
#ifdef USE_CUDA
      ierr = cudaStreamSynchronize(work_stream)
      if (ierr /= 0) error stop 'imex_igw: pcm_exch2 stream sync'
#endif
      call comm_ring_dev8(nb, int(nyl * kk), int(nxl * kk), pcm_rsw, pcm_rse, pcm_rss, pcm_rsn, &
                          pcm_rrw, pcm_rre, pcm_rrs, pcm_rrn)
    endif
    w = nb(1); e = nb(2); s = nb(3); n = nb(4)
    CUF_DO2
    do m = 1, kk
      do q = 1, mx
        if (q <= nyl) then
          if (w == NBR_NONE) then
            x(0, q, m, ll) = 0.0_rp
          else if (w == NBR_SELF) then
            x(0, q, m, ll) = pcm_rse(q + (m - 1) * nyl)
          else
            x(0, q, m, ll) = pcm_rrw(q + (m - 1) * nyl)
          endif
          if (e == NBR_NONE) then
            x(nxl + 1, q, m, ll) = 0.0_rp
          else if (e == NBR_SELF) then
            x(nxl + 1, q, m, ll) = pcm_rsw(q + (m - 1) * nyl)
          else
            x(nxl + 1, q, m, ll) = pcm_rre(q + (m - 1) * nyl)
          endif
        endif
        if (q <= nxl) then
          if (s == NBR_NONE) then
            x(q, 0, m, ll) = 0.0_rp
          else if (s == NBR_SELF) then
            x(q, 0, m, ll) = pcm_rsn(q + (m - 1) * nxl)
          else
            x(q, 0, m, ll) = pcm_rrs(q + (m - 1) * nxl)
          endif
          if (n == NBR_NONE) then
            x(q, nyl + 1, m, ll) = 0.0_rp
          else if (n == NBR_SELF) then
            x(q, nyl + 1, m, ll) = pcm_rss(q + (m - 1) * nxl)
          else
            x(q, nyl + 1, m, ll) = pcm_rrn(q + (m - 1) * nxl)
          endif
        endif
      enddo
    enddo
  end subroutine pcm_exch2

  subroutine pcm_zero0()
    ! x of level 0 (halo included) = 0: the V-cycle starts from zero
    integer(ip) :: i, j, m, nx, ny, kk
    nx = pcm_nx(0); ny = pcm_ny(0); kk = pcm_k
    CUF_DO3
    do m = 1, kk
      do j = 0, ny + 1
        do i = 0, nx + 1
          lv_x(i, j, m, 0) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine pcm_zero0

  subroutine pcm_smooth(l, mode, s)
    ! (mg_smooth with imex_pc_smoother) one sweep on level l of the K modes (halo of x current): 'rbgs' red
    ! then black (mode 1, pre) or black then red (mode 2, post), colours tile-local; 'jacobi' damped Jacobi
    ! (omega 0.8). Inactive cells: d = 0 -> x stays 0
    integer, intent(in) :: l, mode, s
    integer(ip) :: i, j, m, nxl, nyl, kk, ll, ss, c, col
    real(rp) :: w
    nxl = pcm_nx(l); nyl = pcm_ny(l); kk = pcm_k; ll = l; ss = s
    if (pcm_jac) then
      CUF_DO3
      do m = 1, kk
        do j = 1, nyl
          do i = 1, nxl
            lv_t(i, j, m, ll) = lv_d(i, j, m, ll, ss) * (lv_b(i, j, m, ll) &
                 + lv_w(i, j, m, ll, ss) * lv_x(i - 1, j, m, ll) &
                 + lv_w(i + 1, j, m, ll, ss) * lv_x(i + 1, j, m, ll) + lv_s(i, j, m, ll, ss) * lv_x(i, j - 1, m, ll) &
                 + lv_s(i, j + 1, m, ll, ss) * lv_x(i, j + 1, m, ll))
          enddo
        enddo
      enddo
      w = 0.8_rp
      CUF_DO3
      do m = 1, kk
        do j = 1, nyl
          do i = 1, nxl
            lv_x(i, j, m, ll) = (1.0_rp - w) * lv_x(i, j, m, ll) + w * lv_t(i, j, m, ll)
          enddo
        enddo
      enddo
      return
    endif
    do c = 1, 2
      col = mod(c + mode, 2)                       ! mode 1: 0 then 1; mode 2: 1 then 0
      CUF_DO3
      do m = 1, kk
        do j = 1, nyl
          do i = 1, nxl
            if (mod(i + j + col, 2_ip) == 1) &
                 lv_x(i, j, m, ll) = lv_d(i, j, m, ll, ss) * (lv_b(i, j, m, ll) &
                 + lv_w(i, j, m, ll, ss) * lv_x(i - 1, j, m, ll) &
                 + lv_w(i + 1, j, m, ll, ss) * lv_x(i + 1, j, m, ll) + lv_s(i, j, m, ll, ss) * lv_x(i, j - 1, m, ll) &
                 + lv_s(i, j + 1, m, ll, ss) * lv_x(i, j + 1, m, ll))
          enddo
        enddo
      enddo
    enddo
  end subroutine pcm_smooth

  subroutine pcm_rr(l, s)
    ! (mg_resid + mg_restrict) the residual of level l summed over each 2 x 2 aggregate (the CPU's order) into
    ! b of level l+1; x of level l+1 (halo included) = 0 for its pre-sweep from zero
    integer, intent(in) :: l, s
    integer(ip) :: ic, jc, i, j, m, nxf, nyf, nxc, nyc, kk, ll, ss
    real(rp) :: sm, r
    nxf = pcm_nx(l); nyf = pcm_ny(l); nxc = pcm_nx(l + 1); nyc = pcm_ny(l + 1)
    kk = pcm_k; ll = l; ss = s
    CUF_DO3
    do m = 1, kk
      do jc = 0, nyc + 1
        do ic = 0, nxc + 1
          lv_x(ic, jc, m, ll + 1) = 0.0_rp
          if (ic >= 1 .and. ic <= nxc .and. jc >= 1 .and. jc <= nyc) then
            sm = 0.0_rp
            do j = 2 * jc - 1, min(2 * jc, nyf)
              do i = 2 * ic - 1, min(2 * ic, nxf)
                r = 0.0_rp
                if (lv_e(i, j, m, ll, ss) > 0.0_rp) r = lv_b(i, j, m, ll) - (lv_e(i, j, m, ll, ss) * lv_x(i, j, m, ll) &
                     + lv_w(i, j, m, ll, ss) * (lv_x(i, j, m, ll) - lv_x(i - 1, j, m, ll)) &
                     + lv_w(i + 1, j, m, ll, ss) * (lv_x(i, j, m, ll) - lv_x(i + 1, j, m, ll)) &
                     + lv_s(i, j, m, ll, ss) * (lv_x(i, j, m, ll) - lv_x(i, j - 1, m, ll)) &
                     + lv_s(i, j + 1, m, ll, ss) * (lv_x(i, j, m, ll) - lv_x(i, j + 1, m, ll)))
                sm = sm + r
              enddo
            enddo
            lv_b(ic, jc, m, ll + 1) = sm
          endif
        enddo
      enddo
    enddo
  end subroutine pcm_rr

  subroutine pcm_prolong(l, s)
    ! (mg_prolong) x of level l += x of level l+1 (its aggregate) on the active cells (e > 0)
    integer, intent(in) :: l, s
    integer(ip) :: i, j, m, nxf, nyf, kk, ll, ss
    nxf = pcm_nx(l); nyf = pcm_ny(l); kk = pcm_k; ll = l; ss = s
    CUF_DO3
    do m = 1, kk
      do j = 1, nyf
        do i = 1, nxf
          if (lv_e(i, j, m, ll, ss) > 0.0_rp) lv_x(i, j, m, ll) = lv_x(i, j, m, ll) &
               + lv_x((i + 1) / 2, (j + 1) / 2, m, ll + 1)
        enddo
      enddo
    enddo
  end subroutine pcm_prolong

  subroutine pcm_exch(l)
    ! (CPU pcm_exchk) the one-cell halo (no corners) of x on level l for the K modes: through the free
    ! surface's CUDA IPC rings (fs_ipc; one ring per mode) when it has them, else one GPU-aware MPI message per
    ! side for all modes (comm_ring_dev8); a side without a neighbour gets a zero halo, a periodic self-wrap
    ! the tile's own opposite edge (as fs_mg's ring_mg)
    integer, intent(in) :: l
    integer(ip) :: nxl, nyl, mx, m, kk, ll, w, e, s, n, nb(4), q, kx
    integer :: mi, ierr
    nxl = pcm_nx(l); nyl = pcm_ny(l); kk = pcm_k; ll = l; kx = pcm_kmax
    nb = [decomp%nbr_w, decomp%nbr_e, decomp%nbr_s, decomp%nbr_n]
    mx = max(nxl, nyl)
    if (all(nb == NBR_NONE)) then
      CUF_DO2
      do m = 1, kk
        do q = 1, mx
          if (q <= nyl) then
            lv_x(0, q, m, ll) = 0.0_rp; lv_x(nxl + 1, q, m, ll) = 0.0_rp
          endif
          if (q <= nxl) then
            lv_x(q, 0, m, ll) = 0.0_rp; lv_x(q, nyl + 1, m, ll) = 0.0_rp
          endif
        enddo
      enddo
      return
    endif
#ifdef USE_CUDA
    if (ipc_on) then
      do mi = 1, pcm_k
        call ipc_ring_plane(lv_x, pcm_nx(0), pcm_ny(0), mi + l * pcm_kmax, int(nxl), int(nyl))
      enddo
      return
    endif
#endif
    CUF_DO2
    do m = 1, kk
      do q = 1, mx
        if (q <= nyl) then
          pcm_rsw(q + (m - 1) * nyl) = lv_x(1, q, m, ll); pcm_rse(q + (m - 1) * nyl) = lv_x(nxl, q, m, ll)
        endif
        if (q <= nxl) then
          pcm_rss(q + (m - 1) * nxl) = lv_x(q, 1, m, ll); pcm_rsn(q + (m - 1) * nxl) = lv_x(q, nyl, m, ll)
        endif
      enddo
    enddo
    if (any(nb >= 0)) then
#ifdef USE_CUDA
      ierr = cudaStreamSynchronize(work_stream)
      if (ierr /= 0) error stop 'imex_igw: pcm_exch stream sync'
#endif
      call comm_ring_dev8(nb, int(nyl * kk), int(nxl * kk), pcm_rsw, pcm_rse, pcm_rss, pcm_rsn, &
                          pcm_rrw, pcm_rre, pcm_rrs, pcm_rrn)
    endif
    w = nb(1); e = nb(2); s = nb(3); n = nb(4)
    CUF_DO2
    do m = 1, kk
      do q = 1, mx
        if (q <= nyl) then
          if (w == NBR_NONE) then
            lv_x(0, q, m, ll) = 0.0_rp
          else if (w == NBR_SELF) then
            lv_x(0, q, m, ll) = pcm_rse(q + (m - 1) * nyl)
          else
            lv_x(0, q, m, ll) = pcm_rrw(q + (m - 1) * nyl)
          endif
          if (e == NBR_NONE) then
            lv_x(nxl + 1, q, m, ll) = 0.0_rp
          else if (e == NBR_SELF) then
            lv_x(nxl + 1, q, m, ll) = pcm_rsw(q + (m - 1) * nyl)
          else
            lv_x(nxl + 1, q, m, ll) = pcm_rre(q + (m - 1) * nyl)
          endif
        endif
        if (q <= nxl) then
          if (s == NBR_NONE) then
            lv_x(q, 0, m, ll) = 0.0_rp
          else if (s == NBR_SELF) then
            lv_x(q, 0, m, ll) = pcm_rsn(q + (m - 1) * nxl)
          else
            lv_x(q, 0, m, ll) = pcm_rrs(q + (m - 1) * nxl)
          endif
          if (n == NBR_NONE) then
            lv_x(q, nyl + 1, m, ll) = 0.0_rp
          else if (n == NBR_SELF) then
            lv_x(q, nyl + 1, m, ll) = pcm_rss(q + (m - 1) * nxl)
          else
            lv_x(q, nyl + 1, m, ll) = pcm_rrn(q + (m - 1) * nxl)
          endif
        endif
      enddo
    enddo
  end subroutine pcm_exch

  subroutine pcm_post(s, slot)
    ! (CPU pcm_post) after rc = Phi^T rt is formed (the start-up or sweep 2): one V-cycle per mode, a = act V(rc)
    ! - G^-1 rc and the column's <rc, a>; the column partials of <rt, z_J> in red_part(:, 1) are copied to
    ! red_part(:, slot) (the d_K stopping norm) and the column's <rc, a> is added to red_part(:, 1) (<rt, z>)
    integer, intent(in) :: s, slot
    if (.not. pcm_capt) call itw2(33, 0)
    call pcm_vcycle(s)
    call pcm_post_c(s, slot)
    if (.not. pcm_capt) call itw2(33, 1)
  end subroutine pcm_post

  logical function pcm_gdev()
    ! the V-cycle's exchanges are device-only (the free surface's IPC rings, or no neighbour): it can be a graph
    pcm_gdev = .false.
#ifdef USE_CUDA
    pcm_gdev = ipc_on .or. all([decomp%nbr_w, decomp%nbr_e, decomp%nbr_s, decomp%nbr_n] == NBR_NONE)
#endif
  end function pcm_gdev

  subroutine pcm_post_c(s, slot)
    ! pcm_post's column pass (CPU pcm_post's loop): rc(m) = rc of the column, a(n) = act_n V(rc)_n - sum_m
    ! G^-1(n, m) rc(m), <rc, a> summed over n; a into lv_x(:, :, n, 0); the partials (written out for K <= 4: a
    ! CUF kernel holds no local arrays)
    integer, intent(in) :: s, slot
    integer(ip) :: i, j, nx, ny, kk, ss, sl, c
    real(rp) :: rc1, rc2, rc3, rc4, a1, a2, a3, a4, cs
    nx = pcm_nx(0); ny = pcm_ny(0); kk = pcm_k; ss = s; sl = slot
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        c = i + (j - 1) * nx
        rc1 = lv_b(i, j, 1, 0); rc2 = 0.0_rp; rc3 = 0.0_rp; rc4 = 0.0_rp
        if (kk >= 2) rc2 = lv_b(i, j, 2, 0)
        if (kk >= 3) rc3 = lv_b(i, j, 3, 0)
        if (kk >= 4) rc4 = lv_b(i, j, 4, 0)
        cs = 0.0_rp
        if (kk >= 1) then
          a1 = lv_x(i, j, 1, 0) * pcm_act(i, j, 1)
          a1 = a1 - pcm_gi(i, j, 1, 1, ss) * rc1
          if (kk >= 2) a1 = a1 - pcm_gi(i, j, 1, 2, ss) * rc2
          if (kk >= 3) a1 = a1 - pcm_gi(i, j, 1, 3, ss) * rc3
          if (kk >= 4) a1 = a1 - pcm_gi(i, j, 1, 4, ss) * rc4
          cs = cs + rc1 * a1
          lv_x(i, j, 1, 0) = a1
        endif
        if (kk >= 2) then
          a2 = lv_x(i, j, 2, 0) * pcm_act(i, j, 2)
          a2 = a2 - pcm_gi(i, j, 2, 1, ss) * rc1
          if (kk >= 2) a2 = a2 - pcm_gi(i, j, 2, 2, ss) * rc2
          if (kk >= 3) a2 = a2 - pcm_gi(i, j, 2, 3, ss) * rc3
          if (kk >= 4) a2 = a2 - pcm_gi(i, j, 2, 4, ss) * rc4
          cs = cs + rc2 * a2
          lv_x(i, j, 2, 0) = a2
        endif
        if (kk >= 3) then
          a3 = lv_x(i, j, 3, 0) * pcm_act(i, j, 3)
          a3 = a3 - pcm_gi(i, j, 3, 1, ss) * rc1
          if (kk >= 2) a3 = a3 - pcm_gi(i, j, 3, 2, ss) * rc2
          if (kk >= 3) a3 = a3 - pcm_gi(i, j, 3, 3, ss) * rc3
          if (kk >= 4) a3 = a3 - pcm_gi(i, j, 3, 4, ss) * rc4
          cs = cs + rc3 * a3
          lv_x(i, j, 3, 0) = a3
        endif
        if (kk >= 4) then
          a4 = lv_x(i, j, 4, 0) * pcm_act(i, j, 4)
          a4 = a4 - pcm_gi(i, j, 4, 1, ss) * rc1
          if (kk >= 2) a4 = a4 - pcm_gi(i, j, 4, 2, ss) * rc2
          if (kk >= 3) a4 = a4 - pcm_gi(i, j, 4, 3, ss) * rc3
          if (kk >= 4) a4 = a4 - pcm_gi(i, j, 4, 4, ss) * rc4
          cs = cs + rc4 * a4
          lv_x(i, j, 4, 0) = a4
        endif
        red_part(c, sl) = red_part(c, 1)
        red_part(c, 1) = red_part(c, 1) + cs
      enddo
    enddo
  end subroutine pcm_post_c


  subroutine pcm_zadd8(z)
    ! (CPU pcm_zadd8) z = z_J + Phi a on the tile (storage real(8)): per element the modes in turn
    real(8), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, kk
    real(rp) :: zz
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; kk = pcm_k
    if (.not. pcm_capt) call itw2(34, 0)
    CUF_DO3
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then
            zz = z(i, j, k)
            zz = zz + pcm_phi(i, j, k, 1) * lv_x(i, j, 1, 0)
            if (kk >= 2) zz = zz + pcm_phi(i, j, k, 2) * lv_x(i, j, 2, 0)
            if (kk >= 3) zz = zz + pcm_phi(i, j, k, 3) * lv_x(i, j, 3, 0)
            if (kk >= 4) zz = zz + pcm_phi(i, j, k, 4) * lv_x(i, j, 4, 0)
            z(i, j, k) = zz
          endif
        enddo
      enddo
    enddo
    if (.not. pcm_capt) call itw2(34, 1)
  end subroutine pcm_zadd8

  subroutine pcm_zadd4(z)
    ! (CPU pcm_zadd4) z = z_J + Phi a on the tile (storage real(4), the sum in FP64, rounded once)
    real(4), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, kk
    real(rp) :: zz
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; kk = pcm_k
    if (.not. pcm_capt) call itw2(34, 0)
    CUF_DO3
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then
            zz = real(z(i, j, k), 8)
            zz = zz + real(pcm_phi4(i, j, k, 1), 8) * lv_x(i, j, 1, 0)
            if (kk >= 2) zz = zz + real(pcm_phi4(i, j, k, 2), 8) * lv_x(i, j, 2, 0)
            if (kk >= 3) zz = zz + real(pcm_phi4(i, j, k, 3), 8) * lv_x(i, j, 3, 0)
            if (kk >= 4) zz = zz + real(pcm_phi4(i, j, k, 4), 8) * lv_x(i, j, 4, 0)
            z(i, j, k) = real(zz, 4)
          endif
        enddo
      enddo
    enddo
    if (.not. pcm_capt) call itw2(34, 1)
  end subroutine pcm_zadd4


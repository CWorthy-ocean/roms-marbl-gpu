module geom
  ! Geometry kernels: set_depth (SM09), set_HUV, omega (We/Wi split),
  ! set_HUV1 (named coupling). Interior loops launch on the GPU when
  ! USE_CUDA; halo stays on the host after the kernel.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use formulas
  use halo_mod
  use arch_mod, only: sync_arch
  use cdr_src_mod, only: cdr_vol_on, cdr_cmap, cd_vol, cq_w, cc_p0, cq_rel
  use rivers_mod, only: river_on, rivers_huv
  implicit none
  private

  public :: set_depth
  public :: set_HUV
  public :: set_HUV1
  public :: omega
  public :: apply_volume_sources
  public :: check_mode_coupling
  public :: step_maxima
  public :: setup_metrics
  public :: dump_huv_probe
  public :: dump_p1_fun
  public :: dump_rim
  public :: dump_bt2d_probe
  public :: dump_fast_probe
  public :: dump_fast_zeta_inc
  public :: dump_rvS
  public :: dump_rv3d
  public :: dump_rufrc_build
  public :: dump_zeta_obc
  public :: dump_vbar_obc
  public :: dump_vic
  public :: dump_we_b
  public :: dump_swflx_b

contains

  subroutine set_depth()
    ! z_w, z_r, Hz are formed over the whole array (halo included) from
    ! zeta(knew), whose halo is current at every call (step start, last fast
    ! step, load), so they equal the neighbour's values: no exchange (the
    ! 2026-10-08 audit found it never changed a halo value).
    call set_depth_interior()
  end subroutine set_depth

  subroutine set_depth_interior()
    ! Parallel (i,j), sequential k. Halo is not called from here.
    integer(ip) :: i, j, k, is, ie, js, je, nz, kn, lim
    real(rp) :: ds, s_w, s_r, hc_loc

    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nz = decomp%nz
    kn = knew
    hc_loc = hc
    ds = 1.0_rp / real(nz, rp)
    lim = dry_lim(G_GEOM)

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        hinv(i, j) = 1.0_rp / (h(i, j) + hc_loc)
        z_w(i, j, 0) = -h(i, j)
        do k = 1, nz
          s_w = ds * real(k - nz, rp)
          s_r = ds * (real(k - nz, rp) - 0.5_rp)
          z_w(i, j, k) = sm09_z(zeta(i, j, kn), h(i, j), hc_loc, s_w, Cs_w(k))
          z_r(i, j, k) = sm09_z(zeta(i, j, kn), h(i, j), hc_loc, s_r, Cs_r(k))
          ! Keep land Hz, matching UCLA set_depth. visc3d psi fluxes
          ! average the four surrounding Hz, including land corners.
          Hz(i, j, k)  = z_w(i, j, k) - z_w(i, j, k - 1)
        enddo
      enddo
    enddo

    call sync_arch()
  end subroutine set_depth_interior

  subroutine set_HUV()
    integer(ip) :: i, j, k, nx, ny, nz, n_rhs, lim
    integer(ip) :: i_v0, i_v1, j_u0, j_u1
    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_rhs = nrhs
    ! CPU istrR:iendR / jstrR:jendR on open edges so UFe/VFx have the
    ! halo flux columns (FlxV at i=0,nx+1 and FlxU at j=0,ny+1).
    i_v0 = 1
    i_v1 = nx
    if (decomp%edge_w) i_v0 = 0
    if (decomp%edge_e) i_v1 = nx + 1
    j_u0 = 1
    j_u1 = ny
    if (decomp%edge_s) j_u0 = 0
    if (decomp%edge_n) j_u1 = ny + 1
    lim = dry_lim(G_GEOM)

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_u0, j_u1
      do i = 1, nx + 1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        do k = 1, nz
          Hz_u(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
          FlxU(i, j, k) = apply_mask(Hz_u(i, j, k) * dn_u(i, j) * u(i, j, k, n_rhs), umask(i, j) + riv_umask(i, j))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = i_v0, i_v1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        do k = 1, nz
          Hz_v(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
          FlxV(i, j, k) = apply_mask(Hz_v(i, j, k) * dm_v(i, j) * v(i, j, k, n_rhs), vmask(i, j) + riv_vmask(i, j))
        enddo
      enddo
    enddo
    ! river faces: the river transport at time n (CPU 2805cbef)
    if (river_on) call rivers_huv(1_ip, nx + 1, j_u0, j_u1, i_v0, i_v1, 1_ip, ny + 1)
    call hg_begin()
    call hg_add(F_FLXU)
    call hg_add(F_FLXV)
    call hg_end()
  end subroutine set_HUV

  subroutine set_HUV1()
    ! Predictor coupling: remove barotropic mismatch vs extrapolated
    ! DU_avg target, then rebuild FlxU/FlxV. FIRST step uses DU_avg1 only.
    integer(ip) :: i, j, k, nx, ny, nz, n_new, lim
    integer(ip) :: i_v0, i_v1, j_u0, j_u1
    real(rp) :: D, intF, target, mis
    integer(ip) :: first

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_new = nnew
    first = merge(1_ip, 0_ip, is_first_time_step())
    i_v0 = 1
    i_v1 = nx
    if (decomp%edge_w) i_v0 = 0
    if (decomp%edge_e) i_v1 = nx + 1
    j_u0 = 1
    j_u1 = ny
    if (decomp%edge_s) j_u0 = 0
    if (decomp%edge_n) j_u1 = ny + 1
    lim = dry_lim(G_GEOM)

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_u0, j_u1
      do i = 1, nx + 1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        D = 0.0_rp
        intF = 0.0_rp
        do k = 1, nz
          Hz_u(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
          D    = D + Hz_u(i, j, k) * dn_u(i, j)
          intF = intF + Hz_u(i, j, k) * dn_u(i, j) * u(i, j, k, n_new)
        enddo
        if (first == 1) then
          target = DU_avg1(i, j)
        else
          target = EXTRAP_NOW * DU_avg1(i, j) - EXTRAP_MID * DU_avg2(i, j) &
                 + EXTRAP_BAK * DU_avg_bak(i, j)
        endif
        mis = flux_mismatch(intF, D, target)
        do k = 1, nz
          u(i, j, k, n_new) = apply_mask(u(i, j, k, n_new) - mis, umask(i, j))
          FlxU(i, j, k) = Hz_u(i, j, k) * dn_u(i, j) * u(i, j, k, n_new)
        enddo
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = i_v0, i_v1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        D = 0.0_rp
        intF = 0.0_rp
        do k = 1, nz
          Hz_v(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
          D    = D + Hz_v(i, j, k) * dm_v(i, j)
          intF = intF + Hz_v(i, j, k) * dm_v(i, j) * v(i, j, k, n_new)
        enddo
        if (first == 1) then
          target = DV_avg1(i, j)
        else
          target = EXTRAP_NOW * DV_avg1(i, j) - EXTRAP_MID * DV_avg2(i, j) &
                 + EXTRAP_BAK * DV_avg_bak(i, j)
        endif
        mis = flux_mismatch(intF, D, target)
        do k = 1, nz
          v(i, j, k, n_new) = apply_mask(v(i, j, k, n_new) - mis, vmask(i, j))
          FlxV(i, j, k) = Hz_v(i, j, k) * dm_v(i, j) * v(i, j, k, n_new)
        enddo
      enddo
    enddo

    call hg_begin()
    call hg_add(F_FLXU)
    call hg_add(F_FLXV)
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_end()
  end subroutine set_HUV1

  subroutine omega()
    integer(ip) :: i, j, k, nx, ny, nz, lim, cvo, cc, q, r
    real(rp) :: dtau, wrk, c2d, dh, cw, cw_max, cw_max2, cw_min, cff, cx0, cxk, cxkp, wacc, cfv
    real(rp), parameter :: cmnx_ratio = cu_min / cu_max
    real(rp), parameter :: cutoff = 2.0_rp - cmnx_ratio
    real(rp), parameter :: r4cmx = 0.25_rp / (1.0_rp - cmnx_ratio)

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    if (nrhs == 3) then
      dtau = dt
    elseif (is_first_time_step()) then
      dtau = 0.5_rp * dt
    else
      dtau = 0.6_rp * dt
    endif
    lim = dry_lim(G_GEOM)
    cvo = merge(1_ip, 0_ip, cdr_vol_on)

    ! Zero Wi, We except where the kernel below assigns them: in its
    ! columns (1:nx, 1:ny with the same dry skip) it writes Wi(1:nz) and
    ! We(0:nz), so only Wi(0) is zeroed there; halo-frame columns are
    ! zeroed in full, skipped dry columns are left alone (as before).
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) then
          Wi(i, j, 0) = 0.0_rp
        else
          do k = 0, nz
            Wi(i, j, k) = 0.0_rp
            We(i, j, k) = 0.0_rp
          enddo
        endif
      enddo
    enddo

    ! Two copies of the column kernel: without CDR volume releases the
    ! original one (its registers unchanged), with them the same plus the
    ! releases' water.
    if (cvo == 0) then
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        wacc = 0.0_rp
        do k = 1, nz
          wacc = wacc - FlxU(i + 1, j, k) + FlxU(i, j, k) &
                 - FlxV(i, j + 1, k) + FlxV(i, j, k)
          Wi(i, j, k) = wacc
          col_fc(i, j, k) = max(FlxU(i + 1, j, k), 0.0_rp) - min(FlxU(i, j, k), 0.0_rp) &
                          + max(FlxV(i, j + 1, k), 0.0_rp) - min(FlxV(i, j, k), 0.0_rp)
        enddo
        Wi(i, j, nz) = Wi(i, j, nz) + swflx(i, j) * dm_r(i, j) * dn_r(i, j)
        wrk = Wi(i, j, nz) / max(z_w(i, j, nz) - z_w(i, j, 0), 1.0e-30_rp)
        Wi(i, j, nz) = 0.0_rp
        We(i, j, nz) = 0.0_rp
        We(i, j, 0) = 0.0_rp
        cx0 = dtau * pm(i, j) * pn(i, j)
        do k = nz - 1, 1, -1
          Wi(i, j, k) = Wi(i, j, k) - wrk * (z_w(i, j, k) - z_w(i, j, 0))
          cxk = col_fc(i, j, k)
          cxkp = col_fc(i, j, k + 1)
          c2d = max(cxk, cxkp)
          dh  = min(Hz(i, j, k), Hz(i, j, k + 1))
          cw_max = cu_max * dh - c2d * cx0
          if (cw_max > 0.0_rp) then
            cw_max2 = cw_max * cw_max
            cw_min = cw_max * cmnx_ratio
            cw = abs(Wi(i, j, k)) * cx0
            if (cw < cw_min) then
              cff = cw_max2
            elseif (cw < cutoff * cw_max) then
              cff = cw_max2 + r4cmx * (cw - cw_min)**2
            else
              cff = cw_max * cw
            endif
            We(i, j, k) = apply_mask(cw_max2 * Wi(i, j, k) / cff, rmask(i, j))
            Wi(i, j, k) = apply_mask(Wi(i, j, k) - We(i, j, k), rmask(i, j))
          else
            We(i, j, k) = 0.0_rp
            Wi(i, j, k) = apply_mask(Wi(i, j, k), rmask(i, j))
          endif
        enddo
      enddo
    enddo

    else
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        wacc = 0.0_rp
        do k = 1, nz
          wacc = wacc - FlxU(i + 1, j, k) + FlxU(i, j, k) &
                 - FlxV(i, j + 1, k) + FlxV(i, j, k)
          Wi(i, j, k) = wacc
          col_fc(i, j, k) = max(FlxU(i + 1, j, k), 0.0_rp) - min(FlxU(i, j, k), 0.0_rp) &
                          + max(FlxV(i, j + 1, k), 0.0_rp) - min(FlxV(i, j, k), 0.0_rp)
        enddo
        ! CDR volume releases (CPU omega): each release's water, accumulated
        ! bottom up with its weights, before the rain and the grid motion
        if (.true.) then
          cc = cdr_cmap(i, j)
          if (cc > 0) then
            do q = cc_p0(cc), cc_p0(cc + 1) - 1
              r = cq_rel(q)
              cfv = 0.0_rp
              do k = 1, nz
                cfv = cfv + cd_vol(r) * cq_w(k, q)
                Wi(i, j, k) = Wi(i, j, k) + cfv
              enddo
            enddo
          endif
        endif
        Wi(i, j, nz) = Wi(i, j, nz) + swflx(i, j) * dm_r(i, j) * dn_r(i, j)
        wrk = Wi(i, j, nz) / max(z_w(i, j, nz) - z_w(i, j, 0), 1.0e-30_rp)
        Wi(i, j, nz) = 0.0_rp
        We(i, j, nz) = 0.0_rp
        We(i, j, 0) = 0.0_rp
        cx0 = dtau * pm(i, j) * pn(i, j)
        do k = nz - 1, 1, -1
          Wi(i, j, k) = Wi(i, j, k) - wrk * (z_w(i, j, k) - z_w(i, j, 0))
          cxk = col_fc(i, j, k)
          cxkp = col_fc(i, j, k + 1)
          c2d = max(cxk, cxkp)
          dh  = min(Hz(i, j, k), Hz(i, j, k + 1))
          cw_max = cu_max * dh - c2d * cx0
          if (cw_max > 0.0_rp) then
            cw_max2 = cw_max * cw_max
            cw_min = cw_max * cmnx_ratio
            cw = abs(Wi(i, j, k)) * cx0
            if (cw < cw_min) then
              cff = cw_max2
            elseif (cw < cutoff * cw_max) then
              cff = cw_max2 + r4cmx * (cw - cw_min)**2
            else
              cff = cw_max * cw
            endif
            We(i, j, k) = apply_mask(cw_max2 * Wi(i, j, k) / cff, rmask(i, j))
            Wi(i, j, k) = apply_mask(Wi(i, j, k) - We(i, j, k), rmask(i, j))
          else
            We(i, j, k) = 0.0_rp
            Wi(i, j, k) = apply_mask(Wi(i, j, k), rmask(i, j))
          endif
        enddo
      enddo
    enddo

    endif

    call apply_volume_sources()
    call fill_omega_edge_ghosts()
    call hg_begin()
    call hg_add(F_WE)
    call hg_add(F_WI)
    call hg_end()
  end subroutine omega

  subroutine fill_omega_edge_ghosts()
    ! CPU omega_mod after the We/Wi split: copy into physical-boundary
    ! ghosts. Halo exchange does not fill those on an open/closed edge,
    ! and SPLINE_UV at j=ny reads We(i,ny+1). All device kernels.
    integer(ip) :: i, j, k, nx, ny, nz
    logical :: sw, nw, se, ne
    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    if (decomp%edge_w) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 0, nz
        do j = 1, ny
          We(0, j, k) = We(1, j, k)
          Wi(0, j, k) = Wi(1, j, k)
        enddo
      enddo
    endif
    if (decomp%edge_e) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 0, nz
        do j = 1, ny
          We(nx + 1, j, k) = We(nx, j, k)
          Wi(nx + 1, j, k) = Wi(nx, j, k)
        enddo
      enddo
    endif
    if (decomp%edge_s) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 0, nz
        do i = 1, nx
          We(i, 0, k) = We(i, 1, k)
          Wi(i, 0, k) = Wi(i, 1, k)
        enddo
      enddo
    endif
    if (decomp%edge_n) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 0, nz
        do i = 1, nx
          We(i, ny + 1, k) = We(i, ny, k)
          Wi(i, ny + 1, k) = Wi(i, ny, k)
        enddo
      enddo
    endif
    sw = decomp%edge_w .and. decomp%edge_s; nw = decomp%edge_w .and. decomp%edge_n
    se = decomp%edge_e .and. decomp%edge_s; ne = decomp%edge_e .and. decomp%edge_n
    if (sw .or. nw .or. se .or. ne) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do k = 0, nz
        if (sw) then
          We(0, 0, k) = We(1, 1, k)
          Wi(0, 0, k) = Wi(1, 1, k)
        endif
        if (nw) then
          We(0, ny + 1, k) = We(1, ny, k)
          Wi(0, ny + 1, k) = Wi(1, ny, k)
        endif
        if (se) then
          We(nx + 1, 0, k) = We(nx, 1, k)
          Wi(nx + 1, 0, k) = Wi(nx, 1, k)
        endif
        if (ne) then
          We(nx + 1, ny + 1, k) = We(nx, ny, k)
          Wi(nx + 1, ny + 1, k) = Wi(nx, ny, k)
        endif
      enddo
    endif
  end subroutine fill_omega_edge_ghosts

  subroutine setup_metrics()
    ! setup_grid1_tile: dm/dn, pmon_u, pnom_v, umask/vmask/pmask.
    integer(ip) :: i, j, is, ie, js, je
    real(rp) :: cff1, cff2

    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    cff1 = 1.0_rp
    cff2 = 2.0_rp

    do j = js, je
      do i = is, ie
        pm(i, j) = max(pm(i, j), 1.0e-30_rp)
        pn(i, j) = max(pn(i, j), 1.0e-30_rp)
        dm_r(i, j) = 1.0_rp / pm(i, j)
        dn_r(i, j) = 1.0_rp / pn(i, j)
      enddo
    enddo

    ! CURVGRID: d(1/n)/d(xi) and d(1/m)/d(eta) at rho points.
    dndx = 0.0_rp
    dmde = 0.0_rp
    do j = js, je
      do i = is + 1, ie - 1
        dndx(i, j) = 0.5_rp / pn(i + 1, j) - 0.5_rp / pn(i - 1, j)
      enddo
    enddo
    do j = js + 1, je - 1
      do i = is, ie
        dmde(i, j) = 0.5_rp / pm(i, j + 1) - 0.5_rp / pm(i, j - 1)
      enddo
    enddo

    do j = js, je
      do i = is + 1, ie
        pmon_u(i, j) = (pm(i, j) + pm(i - 1, j)) / max(pn(i, j) + pn(i - 1, j), 1.0e-30_rp)
        dm_u(i, j) = 2.0_rp / max(pm(i, j) + pm(i - 1, j), 1.0e-30_rp)
        dn_u(i, j) = 2.0_rp / max(pn(i, j) + pn(i - 1, j), 1.0e-30_rp)
        umask(i, j) = rmask(i, j) * rmask(i - 1, j)
      enddo
    enddo

    do j = js + 1, je
      do i = is, ie
        pnom_v(i, j) = (pn(i, j) + pn(i, j - 1)) / max(pm(i, j) + pm(i, j - 1), 1.0e-30_rp)
        dm_v(i, j) = 2.0_rp / max(pm(i, j) + pm(i, j - 1), 1.0e-30_rp)
        dn_v(i, j) = 2.0_rp / max(pn(i, j) + pn(i, j - 1), 1.0e-30_rp)
        vmask(i, j) = rmask(i, j) * rmask(i, j - 1)
      enddo
    enddo

    do j = js + 1, je
      do i = is + 1, ie
        dm_p(i, j) = 4.0_rp / max(pm(i, j) + pm(i, j - 1) + pm(i - 1, j) + pm(i - 1, j - 1), 1.0e-30_rp)
        dn_p(i, j) = 4.0_rp / max(pn(i, j) + pn(i, j - 1) + pn(i - 1, j) + pn(i - 1, j - 1), 1.0e-30_rp)
        if (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp .and. &
            rmask(i - 1, j - 1) > 0.5_rp .and. rmask(i, j - 1) > 0.5_rp) then
          pmask(i, j) = 1.0_rp
        elseif ((rmask(i - 1, j) < 0.5_rp .and. rmask(i, j) > 0.5_rp .and. &
                 rmask(i - 1, j - 1) > 0.5_rp .and. rmask(i, j - 1) > 0.5_rp) .or. &
                (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) < 0.5_rp .and. &
                 rmask(i - 1, j - 1) > 0.5_rp .and. rmask(i, j - 1) > 0.5_rp) .or. &
                (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp .and. &
                 rmask(i - 1, j - 1) < 0.5_rp .and. rmask(i, j - 1) > 0.5_rp) .or. &
                (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp .and. &
                 rmask(i - 1, j - 1) > 0.5_rp .and. rmask(i, j - 1) < 0.5_rp)) then
          pmask(i, j) = cff1
        elseif ((rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) < 0.5_rp .and. &
                 rmask(i - 1, j - 1) > 0.5_rp .and. rmask(i, j - 1) < 0.5_rp) .or. &
                (rmask(i - 1, j) < 0.5_rp .and. rmask(i, j) > 0.5_rp .and. &
                 rmask(i - 1, j - 1) < 0.5_rp .and. rmask(i, j - 1) > 0.5_rp) .or. &
                (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp .and. &
                 rmask(i - 1, j - 1) < 0.5_rp .and. rmask(i, j - 1) < 0.5_rp) .or. &
                (rmask(i - 1, j) < 0.5_rp .and. rmask(i, j) < 0.5_rp .and. &
                 rmask(i - 1, j - 1) > 0.5_rp .and. rmask(i, j - 1) > 0.5_rp)) then
          pmask(i, j) = cff2
        else
          pmask(i, j) = 0.0_rp
        endif
      enddo
    enddo
  end subroutine setup_metrics

  subroutine apply_volume_sources()
    ! Sparse river/pipe/CDR list. Empty in v1; do not put these branches
    ! inside the fused step2d or omega structured loops.
  end subroutine apply_volume_sources

  subroutine check_mode_coupling(max_u_mis, max_v_mis, volume)
    ! Barotropic/baroclinic coupling check as device reductions: max
    ! |sum_k Hz*dn*u - DU_avg1| over wet U faces (V likewise) and the wet
    ! volume. A second pass finds where the maximum sits (largest index on
    ! ties) and DU/DV_avg1 there, so no managed field is read on the host.
    real(rp), intent(out) :: max_u_mis, max_v_mis, volume
    integer(ip) :: i, j, k, nx, ny, nz, n_new, ld, iloc
    real(rp) :: intF, mis, umx, vmx, vol, avg_at

    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    n_new = nnew
    ld = nx + 2
    umx = 0.0_rp; vmx = 0.0_rp; vol = 0.0_rp

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        intF = 0.0_rp
        do k = 1, nz
          intF = intF + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j) * u(i, j, k, n_new)
        enddo
        mis = merge(abs(intF - DU_avg1(i, j)), 0.0_rp, umask(i, j) > 0.5_rp)
        umx = max(umx, mis)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        intF = 0.0_rp
        do k = 1, nz
          intF = intF + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) * v(i, j, k, n_new)
        enddo
        mis = merge(abs(intF - DV_avg1(i, j)), 0.0_rp, vmask(i, j) > 0.5_rp)
        vmx = max(vmx, mis)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        do k = 1, nz
          vol = vol + Hz(i, j, k) / (pm(i, j) * pn(i, j)) * rmask(i, j)
        enddo
      enddo
    enddo
    max_u_mis = umx; max_v_mis = vmx; volume = vol

    if (umx > 0.0_rp) then
      iloc = 0; avg_at = -huge(1.0_rp)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx + 1
          intF = 0.0_rp
          do k = 1, nz
            intF = intF + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j) * u(i, j, k, n_new)
          enddo
          mis = merge(abs(intF - DU_avg1(i, j)), 0.0_rp, umask(i, j) > 0.5_rp)
          iloc = max(iloc, merge(i + ld * j, 0, mis == umx))
          avg_at = max(avg_at, merge(DU_avg1(i, j), -huge(1.0_rp), mis == umx))
        enddo
      enddo
      write(*,'(a,2i6,a,es14.6)') 'coupling: max u_mis at i,j=', mod(iloc, ld), iloc / ld, &
           '  DU_avg1=', avg_at
    endif
    if (vmx > 0.0_rp) then
      iloc = 0; avg_at = -huge(1.0_rp)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny + 1
        do i = 1, nx
          intF = 0.0_rp
          do k = 1, nz
            intF = intF + 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) * v(i, j, k, n_new)
          enddo
          mis = merge(abs(intF - DV_avg1(i, j)), 0.0_rp, vmask(i, j) > 0.5_rp)
          iloc = max(iloc, merge(i + ld * j, 0, mis == vmx))
          avg_at = max(avg_at, merge(DV_avg1(i, j), -huge(1.0_rp), mis == vmx))
        enddo
      enddo
      write(*,'(a,2i6,a,es14.6)') 'coupling: max v_mis at i,j=', mod(iloc, ld), iloc / ld, &
           '  DV_avg1=', avg_at
    endif
  end subroutine check_mode_coupling

  subroutine step_maxima(umax, akvmax, hblmax)
    ! Device reductions for the end-of-step log: max |u(nnew)| over wet U
    ! faces, max Akv and max hbls over wet rho points (as the host maxvals).
    real(rp), intent(out) :: umax, akvmax, hblmax
    integer(ip) :: i, j, k, nx, ny, nz, n_new, k0, k1
    real(rp) :: um, am, hm

    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    n_new = nnew
    k0 = lbound(Akv, 3); k1 = ubound(Akv, 3)
    um = -huge(1.0_rp); am = -huge(1.0_rp); hm = -huge(1.0_rp)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx + 1
          um = max(um, merge(abs(u(i, j, k, n_new)), -huge(1.0_rp), umask(i, j) > 0.5_rp))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = k0, k1
      do j = 1, ny
        do i = 1, nx
          am = max(am, merge(Akv(i, j, k), -huge(1.0_rp), rmask(i, j) > 0.5_rp))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        hm = max(hm, merge(hbls(i, j), -huge(1.0_rp), rmask(i, j) > 0.5_rp))
      enddo
    enddo
    umax = um; akvmax = am; hblmax = hm
  end subroutine step_maxima

  subroutine dump_huv_probe(tag)
    ! Face u/v, thickness, and mass flux at the coastal cell that climbs.
    ! Call after set_HUV, set_HUV1, and step3d_uv2.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, nz
    real(rp) :: hzu_w, hzu_e, hzv_s, hzv_n
    real(rp) :: flu_w, flu_e, flv_s, flv_n

    if (.not. probe_on) return
    call sync_arch()
    i = iprb_x(1)   ! X1 (154,136), rank-local
    j = jprb_x(1)
    nz = decomp%nz
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return

    write(*,'(a,a,a,i8,a,2i6,a,3i3)') 'diag: huv B  ', trim(tag), &
         '  iic=', iic, '  i,j=', i, j, '  nstp,nrhs,nnew=', nstp, nrhs, nnew
    write(*,'(a,a,a,9es11.3)') 'diag: huv B  ', trim(tag), &
         '  mask umW,umE,vmS,vmN,rm WCESN=', &
         umask(i, j), umask(i + 1, j), vmask(i, j), vmask(i, j + 1), &
         rmask(i - 1, j), rmask(i, j), rmask(i + 1, j), rmask(i, j - 1), rmask(i, j + 1)
    write(*,'(a,a,a,4es14.6)') 'diag: huv B  ', trim(tag), &
         '  dn_u W/E, dm_v S/N=', dn_u(i, j), dn_u(i + 1, j), dm_v(i, j), dm_v(i, j + 1)
    write(*,'(a,a,a,6es14.6)') 'diag: huv B  ', trim(tag), &
         '  DU_avg1/2/bak W,E=', &
         DU_avg1(i, j), DU_avg2(i, j), DU_avg_bak(i, j), &
         DU_avg1(i + 1, j), DU_avg2(i + 1, j), DU_avg_bak(i + 1, j)
    write(*,'(a,a,a,6es14.6)') 'diag: huv B  ', trim(tag), &
         '  DV_avg1/2/bak S,N=', &
         DV_avg1(i, j), DV_avg2(i, j), DV_avg_bak(i, j), &
         DV_avg1(i, j + 1), DV_avg2(i, j + 1), DV_avg_bak(i, j + 1)
    write(*,'(a,a,a,4es14.6)') 'diag: huv B  ', trim(tag), &
         '  ubar W/E, vbar S/N=', &
         ubar(i, j, knew), ubar(i + 1, j, knew), vbar(i, j, knew), vbar(i, j + 1, knew)

    do k = 1, nz
      hzu_w = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
      hzu_e = 0.5_rp * (Hz(i + 1, j, k) + Hz(i, j, k))
      hzv_s = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
      hzv_n = 0.5_rp * (Hz(i, j + 1, k) + Hz(i, j, k))
      write(*,'(a,a,a,i4,11es13.5)') 'diag: huv B  ', trim(tag), ' allk', k, &
           Hz(i, j, k), hzu_w, hzv_n, &
           u(i, j, k, nrhs), u(i, j, k, nnew), &
           v(i, j + 1, k, nrhs), v(i, j + 1, k, nnew), &
           FlxU(i, j, k), FlxU(i + 1, j, k), FlxV(i, j, k), FlxV(i, j + 1, k)
      if (k /= 1 .and. k /= nz) cycle
      flu_w = hzu_w * dn_u(i, j) * u(i, j, k, nrhs)
      flu_e = hzu_e * dn_u(i + 1, j) * u(i + 1, j, k, nrhs)
      flv_s = hzv_s * dm_v(i, j) * v(i, j, k, nrhs)
      flv_n = hzv_n * dm_v(i, j + 1) * v(i, j + 1, k, nrhs)
      write(*,'(a,a,a,i4,a,4es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  Hz*d*u_nrhs WESN=', flu_w, flu_e, flv_s, flv_n
      flu_w = hzu_w * dn_u(i, j) * u(i, j, k, nnew)
      flu_e = hzu_e * dn_u(i + 1, j) * u(i + 1, j, k, nnew)
      flv_s = hzv_s * dm_v(i, j) * v(i, j, k, nnew)
      flv_n = hzv_n * dm_v(i, j + 1) * v(i, j + 1, k, nnew)
      write(*,'(a,a,a,i4,a,5es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  Hz WCESN=', &
           Hz(i - 1, j, k), Hz(i, j, k), Hz(i + 1, j, k), Hz(i, j - 1, k), Hz(i, j + 1, k)
      write(*,'(a,a,a,i4,a,8es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  Hz_u/v store WESN, recomp WESN=', &
           Hz_u(i, j, k), Hz_u(i + 1, j, k), Hz_v(i, j, k), Hz_v(i, j + 1, k), &
           hzu_w, hzu_e, hzv_s, hzv_n
      write(*,'(a,a,a,i4,a,6es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  uW nstp/nrhs/nnew, uE nstp/nrhs/nnew=', &
           u(i, j, k, nstp), u(i, j, k, nrhs), u(i, j, k, nnew), &
           u(i + 1, j, k, nstp), u(i + 1, j, k, nrhs), u(i + 1, j, k, nnew)
      write(*,'(a,a,a,i4,a,6es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  vS nstp/nrhs/nnew, vN nstp/nrhs/nnew=', &
           v(i, j, k, nstp), v(i, j, k, nrhs), v(i, j, k, nnew), &
           v(i, j + 1, k, nstp), v(i, j + 1, k, nrhs), v(i, j + 1, k, nnew)
      write(*,'(a,a,a,i4,a,4es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  FlxU_w,FlxU_e,FlxV_s,FlxV_n=', &
           FlxU(i, j, k), FlxU(i + 1, j, k), FlxV(i, j, k), FlxV(i, j + 1, k)
      write(*,'(a,a,a,i4,a,4es14.6)') 'diag: huv B  ', trim(tag), &
           '  k=', k, '  Hz*d*u_nnew WESN=', flu_w, flu_e, flv_s, flv_n
    enddo
    flush(6)
  end subroutine dump_huv_probe

  subroutine dump_p1_fun(tag)
    ! Family-B west U-face FlxU(154,136) and south V-face FlxV(154,136).
    ! Call after set_HUV, after set_HUV1, and before step3d_uv1.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, nz, n_rhs, n_new
    real(rp) :: hzw, hze, hzu, dnu, um, flu, reb_r, reb_n
    real(rp) :: hzs, hzn, hzv, dmv, vm, flv

    if (.not. probe_on) return
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < decomp%is + 1 .or. i > decomp%ie) return
    if (j < decomp%js + 1 .or. j > decomp%je) return
    call sync_arch()
    nz = decomp%nz
    n_rhs = nrhs
    n_new = nnew
    dnu = dn_u(i, j)
    um = umask(i, j)
    dmv = dm_v(i, j)
    vm = vmask(i, j)
    do k = 1, nz
      hzw = Hz(i - 1, j, k)
      hze = Hz(i, j, k)
      hzu = 0.5_rp * (hzw + hze)
      flu = FlxU(i, j, k)
      reb_r = hzu * dnu * u(i, j, k, n_rhs)
      reb_n = hzu * dnu * u(i, j, k, n_new)
      write(*,'(a,i4,a,i8,a,a,a,12es26.17)') 'diag: fun B  k=', k, &
           '  iic=', iic, '  ', trim(tag), &
           '  FlxU,rebNrhs,rebNnew,uNrhs,uNnew,HzW,HzE,HzU,dn_u,umask,DU1,DU2=', &
           flu, reb_r, reb_n, u(i, j, k, n_rhs), u(i, j, k, n_new), &
           hzw, hze, hzu, dnu, um, DU_avg1(i, j), DU_avg2(i, j)
      hzs = Hz(i, j - 1, k)
      hzn = Hz(i, j, k)
      hzv = 0.5_rp * (hzs + hzn)
      flv = FlxV(i, j, k)
      reb_r = hzv * dmv * v(i, j, k, n_rhs)
      reb_n = hzv * dmv * v(i, j, k, n_new)
      write(*,'(a,i4,a,i8,a,a,a,12es26.17)') 'diag: fvn B  k=', k, &
           '  iic=', iic, '  ', trim(tag), &
           '  FlxV,rebNrhs,rebNnew,vNrhs,vNnew,HzS,HzN,HzV,dm_v,vmask,DV1,DV2=', &
           flv, reb_r, reb_n, v(i, j, k, n_rhs), v(i, j, k, n_new), &
           hzs, hzn, hzv, dmv, vm, DV_avg1(i, j), DV_avg2(i, j)
    enddo
    call dump_p1_huv1(tag)
    flush(6)
  end subroutine dump_p1_fun

  subroutine dump_we_b(tag)
    ! Wet rho P2 (455,56) We/Wi after omega, plus U/V-face interpolants
    ! and Courant-split ingredients. Matches CPU smoke_dump_we_b at P2.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, kk, nz
    real(rp) :: dtau, cx0, wrk, wsum, cxk, cxkp, c2d, dh, cw, cw_max, cff
    real(rp) :: wewu, wewv
    real(rp), parameter :: cmnx_ratio = cu_min / cu_max
    real(rp), parameter :: cutoff = 2.0_rp - cmnx_ratio
    real(rp), parameter :: r4cmx = 0.25_rp / (1.0_rp - cmnx_ratio)

    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(2)
    j = jprb_pt(2)
    if (i < 2 .or. i > decomp%nx - 1 .or. j < 2 .or. j > decomp%ny - 1) return
    nz = decomp%nz
    if (nrhs == 3) then
      dtau = dt
    elseif (is_first_time_step()) then
      dtau = 0.5_rp * dt
    else
      dtau = 0.6_rp * dt
    endif
    cx0 = dtau * pm(i, j) * pn(i, j)
    wrk = 0.0_rp
    do kk = 1, nz
      wrk = wrk - FlxU(i + 1, j, kk) + FlxU(i, j, kk) - FlxV(i, j + 1, kk) + FlxV(i, j, kk)
    enddo
    wrk = wrk + swflx(i, j) * dm_r(i, j) * dn_r(i, j)
    if (abs(z_w(i, j, nz) - z_w(i, j, 0)) > 1.0e-30_rp) then
      wrk = wrk / (z_w(i, j, nz) - z_w(i, j, 0))
    endif
    write(*,'(a,a,a,i8,a,11es26.17)') 'diag: we B 2d  ', trim(tag), &
         '  iic=', iic, '  nrhs,dtau,pm,pn,cx0,zw0,zwN,swflx,wrk,dmr,dnr=', &
         real(nrhs, rp), dtau, pm(i, j), pn(i, j), cx0, &
         z_w(i, j, 0), z_w(i, j, nz), swflx(i, j), wrk, dm_r(i, j), dn_r(i, j)
    do k = 1, nz
      wsum = We(i, j, k) + Wi(i, j, k)
      write(*,'(a,a,a,i8,a,i4,a,11es26.17)') 'diag: we B rho  ', trim(tag), &
           '  iic=', iic, '  k=', k, '  We,Wi,Wsum,Hz,zw,rmask,swflx,FuW,FuE,FvS,FvN=', &
           We(i, j, k), Wi(i, j, k), wsum, Hz(i, j, k), z_w(i, j, k), rmask(i, j), &
           swflx(i, j), FlxU(i, j, k), FlxU(i + 1, j, k), FlxV(i, j, k), FlxV(i, j + 1, k)
      wewu = 0.5_rp * (We(i, j, k) + We(i - 1, j, k) - 0.125_rp * ( &
           (We(i + 1, j, k) - We(i, j, k)) * umask(i + 1, j) - &
           (We(i - 1, j, k) - We(i - 2, j, k)) * umask(i - 1, j)))
      wewv = 0.5_rp * (We(i, j, k) + We(i, j - 1, k) - 0.125_rp * ( &
           (We(i, j + 1, k) - We(i, j, k)) * vmask(i, j + 1) - &
           (We(i, j - 1, k) - We(i, j - 2, k)) * vmask(i, j - 1)))
      write(*,'(a,a,a,i8,a,i4,a,11es26.17)') 'diag: we B u  ', trim(tag), &
           '  iic=', iic, '  k=', k, '  WeWW,WeW,WeC,WeE,umW,umC,umE,WeWu=', &
           We(i - 2, j, k), We(i - 1, j, k), We(i, j, k), We(i + 1, j, k), &
           umask(i - 1, j), umask(i, j), umask(i + 1, j), wewu
      write(*,'(a,a,a,i8,a,i4,a,11es26.17)') 'diag: we B v  ', trim(tag), &
           '  iic=', iic, '  k=', k, '  WeSS,WeS,WeC,WeN,vmS,vmC,vmN,WeWv=', &
           We(i, j - 2, k), We(i, j - 1, k), We(i, j, k), We(i, j + 1, k), &
           vmask(i, j - 1), vmask(i, j), vmask(i, j + 1), wewv
      cxk = max(FlxU(i + 1, j, k), 0.0_rp) - min(FlxU(i, j, k), 0.0_rp) + &
            max(FlxV(i, j + 1, k), 0.0_rp) - min(FlxV(i, j, k), 0.0_rp)
      cxkp = 0.0_rp
      if (k < nz) then
        cxkp = max(FlxU(i + 1, j, k + 1), 0.0_rp) - min(FlxU(i, j, k + 1), 0.0_rp) + &
               max(FlxV(i, j + 1, k + 1), 0.0_rp) - min(FlxV(i, j, k + 1), 0.0_rp)
      endif
      c2d = max(cxk, cxkp)
      dh = Hz(i, j, k)
      if (k < nz) dh = min(Hz(i, j, k), Hz(i, j, k + 1))
      cw_max = cu_max * dh - c2d * cx0
      cw = abs(wsum) * cx0
      cff = 0.0_rp
      if (cw_max > 0.0_rp) then
        if (cw < cw_max * cmnx_ratio) then
          cff = cw_max * cw_max
        elseif (cw < cutoff * cw_max) then
          cff = cw_max * cw_max + r4cmx * (cw - cw_max * cmnx_ratio)**2
        else
          cff = cw_max * cw
        endif
      endif
      write(*,'(a,a,a,i8,a,i4,a,10es26.17)') 'diag: we B split  ', trim(tag), &
           '  iic=', iic, '  k=', k, '  Wsum,cx0,CXk,CXkp,c2d,dh,cw,cw_max,cff,We=', &
           wsum, cx0, cxk, cxkp, c2d, dh, cw, cw_max, cff, We(i, j, k)
    enddo
    flush(6)
  end subroutine dump_we_b

  subroutine dump_swflx_b(tag)
    ! Family-B surface freshwater / heat flux and the interpolation
    ! clock used by set_forces (frc_slot / tneed).
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j
    real(rp) :: tneed
    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    tneed = frc_modtime()
    write(*,'(a,a,a,i8,a,10es26.17)') 'diag: swflx B  ', trim(tag), &
         '  iic=', iic, '  nrhs,slot,tdays,tneed,swflx,stflx,srflx,tair,dmr,dnr=', &
         real(nrhs, rp), real(frc_slot, rp), tdays, tneed, &
         swflx(i, j), stflx(i, j, itemp), srflx(i, j), tair(i, j), &
         dm_r(i, j), dn_r(i, j)
    flush(6)
  end subroutine dump_swflx_b

  subroutine dump_p1_huv1(tag)
    ! 2-D set_HUV1 ingredients at family-B west U and south V faces.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, nz
    real(rp) :: dcol, intf, tgt, mis, du1, du2, bak, rebt, bak_snap
    real(rp) :: dnu, um, first
    real(rp) :: hzu, hzv, dmv, vm, dv1, dv2, dvb

    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < decomp%is + 1 .or. i > decomp%ie) return
    if (j < decomp%js + 1 .or. j > decomp%je) return
    nz = decomp%nz
    dnu = dn_u(i, j)
    um = umask(i, j)
    du1 = DU_avg1(i, j)
    du2 = DU_avg2(i, j)
    bak = DU_avg_bak(i, j)
    dcol = 0.0_rp
    intf = 0.0_rp
    do k = 1, nz
      hzu = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
      dcol = dcol + hzu * dnu
      intf = intf + FlxU(i, j, k)
    enddo
    rebt = EXTRAP_NOW * du1 - EXTRAP_MID * du2 + EXTRAP_BAK * bak
    bak_snap = du1 - EXTRAP_AVG_BAK_FAC * du2
    if (is_first_time_step()) then
      first = 1.0_rp
      tgt = du1
    else
      first = 0.0_rp
      tgt = rebt
    endif
    mis = (intf - tgt) / max(dcol, 1.0e-30_rp)
    write(*,'(a,i8,a,a,a,12es18.10)') 'diag: huv1 B  iic=', iic, &
         '  ', trim(tag), &
         '  D,intF,target,mis,DU1,DU2,DUbak,rebT,bakSnap,first,umask,dn_u=', &
         dcol, intf, tgt, mis, du1, du2, bak, rebt, bak_snap, first, um, dnu

    dmv = dm_v(i, j)
    vm = vmask(i, j)
    dv1 = DV_avg1(i, j)
    dv2 = DV_avg2(i, j)
    dvb = DV_avg_bak(i, j)
    dcol = 0.0_rp
    intf = 0.0_rp
    do k = 1, nz
      hzv = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
      dcol = dcol + hzv * dmv
      intf = intf + FlxV(i, j, k)
    enddo
    rebt = EXTRAP_NOW * dv1 - EXTRAP_MID * dv2 + EXTRAP_BAK * dvb
    bak_snap = dv1 - EXTRAP_AVG_BAK_FAC * dv2
    if (is_first_time_step()) then
      first = 1.0_rp
      tgt = dv1
    else
      first = 0.0_rp
      tgt = rebt
    endif
    mis = (intf - tgt) / max(dcol, 1.0e-30_rp)
    write(*,'(a,i8,a,a,a,12es18.10)') 'diag: hvv1 B  iic=', iic, &
         '  ', trim(tag), &
         '  D,intF,target,mis,DV1,DV2,DVbak,rebT,bakSnap,first,vmask,dm_v=', &
         dcol, intf, tgt, mis, dv1, dv2, dvb, rebt, bak_snap, first, vm, dmv
  end subroutine dump_p1_huv1

  subroutine dump_rim(tag)
    ! Hour-1 leftover sites from the bak-fix day run (history python
    ! j,i on the 0-halo field). Halo rows/columns are included. The
    ! points are time_state's X2-X5, rank-local (prb_off off this rank):
    !   NeZ zeta (511,433)  north ghost, near east
    !   NtT temp (433,432)  last interior north, k=nz (and k=50 if nz >= 50)
    !   SeU u    (512,  0)  south ghost, east U-face, k=10 (if nz >= 10)
    !   NeV v    (513,432)  east ghost, last interior V, k=nz
    character(len=*), intent(in) :: tag
    integer(ip) :: nz, n_rhs, n_new, ku, i, j
    real(rp) :: hzu, hzv, t50

    if (.not. probe_on) return
    call sync_arch()
    nz = decomp%nz
    n_rhs = nrhs
    n_new = nnew
    ku = 10_ip

    i = iprb_x(2); j = jprb_x(2)
    if (in_alloc(i, j)) then
      write(*,'(a,i8,a,a,a,8es18.10)') 'diag: rim NeZ  iic=', iic, &
           '  ', trim(tag), &
           '  zeta,Zt_avg1,rmask,h,HzNz,Tnz,DUbak,DVbak=', &
           zeta(i, j, knew), zt_avg1(i, j), rmask(i, j), &
           h(i, j), Hz(i, j, nz), t(i, j, nz, n_new, itemp), &
           DU_avg_bak(i, j), DV_avg_bak(i, j)
    endif
    i = iprb_x(3); j = jprb_x(3)
    if (in_alloc(i, j)) then
      t50 = 0.0_rp
      if (nz >= 50) t50 = t(i, j, 50, n_new, itemp)
      write(*,'(a,i8,a,a,a,8es18.10)') 'diag: rim NtT  iic=', iic, &
           '  ', trim(tag), &
           '  Tnz,T50,hbls,zeta,HzNz,rmask,AktNz,WeNz=', &
           t(i, j, nz, n_new, itemp), t50, &
           hbls(i, j), zeta(i, j, knew), Hz(i, j, nz), &
           rmask(i, j), Akt(i, j, nz, itemp), We(i, j, nz)
    endif
    i = iprb_x(4); j = jprb_x(4)
    if (nz >= ku .and. in_alloc(i, j) .and. in_alloc(i - 1, j)) then
      hzu = 0.5_rp * (Hz(i, j, ku) + Hz(i - 1, j, ku))
      write(*,'(a,i8,a,a,a,8es18.10)') 'diag: rim SeU  iic=', iic, &
           '  ', trim(tag), &
           '  uNrhs,uNnew,ubar,umask,HzU,DU1,DU2,DUbak=', &
           u(i, j, ku, n_rhs), u(i, j, ku, n_new), ubar(i, j, knew), &
           umask(i, j), hzu, DU_avg1(i, j), DU_avg2(i, j), &
           DU_avg_bak(i, j)
    endif
    i = iprb_x(5); j = jprb_x(5)
    if (in_alloc(i, j) .and. in_alloc(i, j - 1)) then
      hzv = 0.5_rp * (Hz(i, j, nz) + Hz(i, j - 1, nz))
      write(*,'(a,i8,a,a,a,8es18.10)') 'diag: rim NeV  iic=', iic, &
           '  ', trim(tag), &
           '  vNrhs,vNnew,vbar,vmask,HzV,DV1,DV2,DVbak=', &
           v(i, j, nz, n_rhs), v(i, j, nz, n_new), &
           vbar(i, j, knew), vmask(i, j), hzv, &
           DV_avg1(i, j), DV_avg2(i, j), DV_avg_bak(i, j)
    endif
    flush(6)
  contains
    logical function in_alloc(i, j)
      integer(ip), intent(in) :: i, j
      in_alloc = i >= decomp%is .and. i <= decomp%ie .and. &
           j >= decomp%js .and. j <= decomp%je
    end function in_alloc
  end subroutine dump_rim

  subroutine dump_bt2d_probe(tag)
    ! West U-face / north V-face 2-D state at each probe point.
    ! Call after step3d_uv1+visc3d, after iif=1, and after the last fast step.
    character(len=*), intent(in) :: tag
    integer(ip) :: n, i, j
    character(len=16) :: pfx
    real(rp) :: zw_s, zc_s, zw_n, zc_n, Dw, Dc, Du

    if (.not. probe_on .and. .not. fast_probe_on) return
    call sync_arch()
    do n = 1, nprb_pt
      if (n > 1 .and. .not. probe_on) cycle
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(pfx, '(a,i1,a)') 'diag: btP', n, '  '
      zw_s = zeta(i - 1, j, kstp)
      zc_s = zeta(i, j, kstp)
      zw_n = zeta(i - 1, j, knew)
      zc_n = zeta(i, j, knew)
      Dw = h(i - 1, j) + zw_n
      Dc = h(i, j) + zc_n
      Du = 0.5_rp * (Dw + Dc)
      write(*,'(a,a,a,i8,a,2i6,a,i4,a,2i3)') trim(pfx)//'  ', trim(tag), &
           '  iic=', iic, '  i,j=', i, j, '  iif=', iif, '  kstp,knew=', kstp, knew
      write(*,'(a,a,a,5es26.17)') trim(pfx)//'  ', trim(tag), &
           '  rufrc,rubar,ru+rufrc,rvfrcN,rvbarN=', &
           rufrc(i, j), rubar(i, j), rubar(i, j) + rufrc(i, j), &
           rvfrc(i, j + 1), rvbar(i, j + 1)
      write(*,'(a,a,a,4es26.17)') trim(pfx)//'  ', trim(tag), &
           '  zeta W/C kstp,knew=', zw_s, zc_s, zw_n, zc_n
      write(*,'(a,a,a,5es26.17)') trim(pfx)//'  ', trim(tag), &
           '  D W/C/U, h W/C=', Dw, Dc, Du, h(i - 1, j), h(i, j)
      write(*,'(a,a,a,5es26.17)') trim(pfx)//'  ', trim(tag), &
           '  ubar kstp/knew, DU_avg1/2/bak=', &
           ubar(i, j, kstp), ubar(i, j, knew), &
           DU_avg1(i, j), DU_avg2(i, j), DU_avg_bak(i, j)
    enddo
    flush(6)
  end subroutine dump_bt2d_probe

  subroutine dump_fast_probe()
    ! Two lines per fast step at each probe point; matches CPU smoke_dump_fast.
    ! U/V values are the west / south faces of rho cell (i,j).
    integer(ip) :: n, i, j
    if (.not. probe_on .and. .not. fast_probe_on) return
    call sync_arch()
    do n = 1, nprb_pt
      if (n > 1 .and. .not. probe_on) cycle
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(*,'(a,i1,a,i8,a,i4,a,7es26.17)') 'diag: fast P', n, '  iic=', iic, &
           '  iif=', iif, '  zetaC,zetaW,ubar,vbar,rubar,DU_avg1,Zt_avg1=', &
           zeta(i, j, knew), zeta(i - 1, j, knew), ubar(i, j, knew), vbar(i, j, knew), &
           rubar(i, j), DU_avg1(i, j), zt_avg1(i, j)
      write(*,'(a,i1,a,i8,a,i4,a,5es26.17)') 'diag: fast2 P', n, '  iic=', iic, &
           '  iif=', iif, '  rvbar,rufrc,rvfrc,DV_avg1,zetaS=', &
           rvbar(i, j), rufrc(i, j), rvfrc(i, j), DV_avg1(i, j), zeta(i, j - 1, knew)
      write(*,'(a,i1,a,i8,a,i4,a,6es26.17)') 'diag: fastN P', n, '  iic=', iic, &
           '  iif=', iif, '  zetaN,vbarN,rvfrcN,rvbarN,DV_avg1N,DnewC=', &
           zeta(i, j + 1, knew), vbar(i, j + 1, knew), rvfrc(i, j + 1), &
           rvbar(i, j + 1), DV_avg1(i, j + 1), h(i, j) + zeta(i, j, knew)
    enddo
  end subroutine dump_fast_probe

  subroutine dump_fast_zeta_inc()
    ! Volume-flux ingredients of the zeta update, before DUon is reused.
    integer(ip) :: n, i, j
    real(rp) :: duw, due, dvs, dvn, div

    if (.not. probe_on .and. .not. fast_probe_on) return
    call sync_arch()
    do n = 1, nprb_pt
      if (n > 1 .and. .not. probe_on) cycle
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      duw = DUon(i, j)
      due = DUon(i + 1, j)
      dvs = DVom(i, j)
      dvn = DVom(i, j + 1)
      div = duw - due + dvs - dvn
      write(*,'(a,i1,a,i8,a,i4,a,8es26.17)') 'diag: fast3 P', n, '  iic=', iic, &
           '  iif=', iif, '  DUonW,DUonE,DVomS,DVomN,div,zeta_new,Dnew,swflx=', &
           duw, due, dvs, dvn, div, zeta_new(i, j), h(i, j) + zeta_new(i, j), &
           swflx(i, j)
    enddo
    flush(6)
  end subroutine dump_fast_zeta_inc

  subroutine dump_rvS(tag)
    ! South V-face of P1 (391,432): 3-D rvfrc vs first-step 2-D rvbar.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j

    if (.not. probe_on) return
    i = iprb_pt(1)
    j = jprb_pt(1)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    write(*,'(a,a,a,i8,a,i4,a,6es26.17)') 'diag: rvS P1  ', trim(tag), &
         '  iic=', iic, '  iif=', iif, '  rvfrc,rvbar,diff,vmask,vbar,DV_avg1=', &
         rvfrc(i, j), rvbar(i, j), rvfrc(i, j) - rvbar(i, j), vmask(i, j), &
         vbar(i, j, knew), DV_avg1(i, j)
    flush(6)
  end subroutine dump_rvS

  subroutine dump_rv3d(tag, extra)
    ! P1 south V-face 3-D rv build. extra is 0, (rvfrc-sum rv) after
    ! stresses, or the visc column increment after visc3d.
    character(len=*), intent(in) :: tag
    real(rp), intent(in) :: extra
    integer(ip) :: i, j, nz
    real(rp) :: rv50

    if (.not. probe_on) return
    i = iprb_pt(1)
    j = jprb_pt(1)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    nz = decomp%nz
    rv50 = 0.0_rp   ! level 50 only if nz >= 50
    if (nz >= 50) rv50 = rv(i, j, 50)
    write(*,'(a,a,a,i8,a,6es26.17)') 'diag: rv3d P1  ', trim(tag), &
         '  iic=', iic, '  sum_rv,rv1,rv50,rv100,rvfrc,extra=', &
         sum(rv(i, j, 1:nz)), rv(i, j, 1), rv50, rv(i, j, nz), &
         rvfrc(i, j), extra
    flush(6)
  end subroutine dump_rv3d

  subroutine dump_rufrc_build(tag)
    ! Mask / halo / OBC-face breakdown of rufrc. Matches CPU smoke_dump_rufrc_build.
    character(len=*), intent(in) :: tag
    integer(ip) :: n, i, j, k, ii, i0u, updated, nx, ny
    integer(ip) :: ifc(4)
    real(rp) :: rf(4), um(4), dmdn, hzu, u1, stress, drag
    real(rp) :: umW, umC, umE, rmW, rmC, sus, rDW, rDC, ruL, ruC, ruR

    if (.not. probe_on) return
    call sync_arch()
    nx = decomp%nx
    ny = decomp%ny
    i0u = 1
    if (decomp%edge_w) i0u = 2
    do n = 1, nprb_pt
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > nx .or. j < 1 .or. j > ny) cycle
      updated = 0
      if (i >= i0u .and. i <= nx) updated = 1
      umW = umask(i - 1, j)
      umC = umask(i, j)
      umE = umask(i + 1, j)
      rmW = rmask(i - 1, j)
      rmC = rmask(i, j)
      sus = sustr(i, j)
      rDW = r_D(i - 1, j)
      rDC = r_D(i, j)
      dmdn = dm_u(i, j) * dn_u(i, j)
      ruL = rufrc(i - 1, j)
      ruC = rufrc(i, j)
      ruR = rufrc(i + 1, j)
      u1 = u(i, j, 1, nnew)
      hzu = 0.5_rp * (Hz(i, j, 1) + Hz(i - 1, j, 1))
      stress = dmdn * sus
      drag = dmdn * 0.5_rp * (rDW + rDC) * u1 / max(hzu, 1.0e-30_rp)
      write(*,'(a,i1,a,a,a,i8,a,2i6,a,i2,a,12es26.17)') 'diag: ruB P', n, '  ', &
           trim(tag), '  iic=', iic, '  ig,jg=', i, j, '  upd=', updated, &
           '  umW,um,umE,rmW,rm,sustr,rDW,rD,dmdn,ruL,rufrc,ruR=', &
           umW, umC, umE, rmW, rmC, sus, rDW, rDC, dmdn, ruL, ruC, ruR
      write(*,'(a,i1,a,a,a,i8,a,6es26.17)') 'diag: ruC P', n, '  ', &
           trim(tag), '  iic=', iic, '  stress,drag,Hzu,u1,rufrc,umask=', &
           stress, drag, hzu, u1, ruC, umC
      do k = 1, 4
        ii = i - 3 + k
        ifc(k) = ii
        if (ii >= 0 .and. ii <= nx + 1) then
          rf(k) = rufrc(ii, j)
          um(k) = umask(ii, j)
        else
          rf(k) = 0.0_rp
          um(k) = -1.0_rp
        endif
      enddo
      write(*,'(a,i1,a,a,a,i8,a,4i6,a,8es26.17)') 'diag: ruF P', n, '  ', &
           trim(tag), '  iic=', iic, '  i0,i1,i2,i3=', ifc(1), ifc(2), ifc(3), ifc(4), &
           '  rufrc,umask x4=', rf(1), um(1), rf(2), um(2), rf(3), um(3), rf(4), um(4)
    enddo
    flush(6)
  end subroutine dump_rufrc_build

  subroutine dump_zeta_obc()
    ! Chapman zeta at west (P2 j), east (P3), north (P1) and south (P2 i).
    ! cx is the value this binary actually used (GPU: sqrt(g*max(h,1))).
    integer(ip) :: n, i, j, nx, ny
    real(rp) :: cx, hh, hfl

    if (.not. probe_on .and. .not. fast_probe_on) return
    if (probe_on .and. iif > 3 .and. iif /= nfast) return
    call sync_arch()
    nx = decomp%nx
    ny = decomp%ny

    if (decomp%edge_w .and. probe_on) then
      n = 2
      j = jprb_pt(n)
      i = 1
      if (j >= 0 .and. j <= ny + 1) then
        hh = h(i, j)
        hfl = max(hh, 1.0_rp)
        cx = dtfast * pm(i, j) * sqrt(g * hfl)
        write(*,'(a,i1,a,i8,a,i4,a,9es26.17)') 'diag: zobc W P', n, '  iic=', iic, &
             '  iif=', iif, '  cx,h,hfl,pm,rmaskG,zGkstp,zIkstp,zGknew,zIknew=', &
             cx, hh, hfl, pm(i, j), rmask(i - 1, j), &
             zeta(i - 1, j, kstp), zeta(i, j, kstp), &
             zeta(i - 1, j, knew), zeta(i, j, knew)
      endif
    endif

    if (decomp%edge_e .and. probe_on) then
      n = 3
      j = jprb_pt(n)
      i = nx
      if (j >= 0 .and. j <= ny + 1) then
        hh = h(i, j)
        hfl = max(hh, 1.0_rp)
        cx = dtfast * pm(i, j) * sqrt(g * hfl)
        write(*,'(a,i1,a,i8,a,i4,a,9es26.17)') 'diag: zobc E P', n, '  iic=', iic, &
             '  iif=', iif, '  cx,h,hfl,pm,rmaskG,zGkstp,zIkstp,zGknew,zIknew=', &
             cx, hh, hfl, pm(i, j), rmask(i + 1, j), &
             zeta(i + 1, j, kstp), zeta(i, j, kstp), &
             zeta(i + 1, j, knew), zeta(i, j, knew)
      endif
    endif

    if (decomp%edge_n) then
      n = 1
      i = iprb_pt(n)
      j = ny
      if (i >= 0 .and. i <= nx + 1) then
        hh = h(i, j)
        hfl = max(hh, 1.0_rp)
        cx = dtfast * pn(i, j) * sqrt(g * hfl)
        write(*,'(a,i1,a,i8,a,i4,a,9es26.17)') 'diag: zobc N P', n, '  iic=', iic, &
             '  iif=', iif, '  cx,h,hfl,pn,rmaskG,zGkstp,zIkstp,zGknew,zIknew=', &
             cx, hh, hfl, pn(i, j), rmask(i, j + 1), &
             zeta(i, j + 1, kstp), zeta(i, j, kstp), &
             zeta(i, j + 1, knew), zeta(i, j, knew)
      endif
    endif

    if (decomp%edge_s .and. probe_on) then
      n = 2
      i = iprb_pt(n)
      j = 1
      if (i >= 0 .and. i <= nx + 1) then
        hh = h(i, j)
        hfl = max(hh, 1.0_rp)
        cx = dtfast * pn(i, j) * sqrt(g * hfl)
        write(*,'(a,i1,a,i8,a,i4,a,9es26.17)') 'diag: zobc S P', n, '  iic=', iic, &
             '  iif=', iif, '  cx,h,hfl,pn,rmaskG,zGkstp,zIkstp,zGknew,zIknew=', &
             cx, hh, hfl, pn(i, j), rmask(i, j - 1), &
             zeta(i, j - 1, kstp), zeta(i, j, kstp), &
             zeta(i, j - 1, knew), zeta(i, j, knew)
      endif
    endif
    flush(6)
  end subroutine dump_zeta_obc

  subroutine dump_vbar_obc()
    ! Flather vbar at north (P1) and south (P2) after v2dbc.
    ! hx is the value this binary used (GPU: sqrt(g/max(cff,1))).
    integer(ip) :: n, i, nx, ny
    real(rp) :: cff, hx, cx, zx, hI, hG, pnI, pnG
    real(rp), parameter :: cx_lim = 0.292893218813452_rp

    if (.not. probe_on .and. .not. fast_probe_on) return
    if (probe_on .and. iif > 3 .and. iif /= nfast) return
    call sync_arch()
    nx = decomp%nx
    ny = decomp%ny

    if (decomp%edge_n) then
      n = 1
      i = iprb_pt(n)
      if (i >= 1 .and. i <= nx) then
        hI = h(i, ny)
        hG = h(i, ny + 1)
        cff = 0.5_rp * (hI + hG)
        hx = sqrt(g / max(cff, 1.0_rp))
        pnI = pn(i, ny)
        pnG = pn(i, ny + 1)
        cx = dtfast * cff * hx * 0.5_rp * (pnI + pnG)
        zx = (0.5_rp + cx) * zeta(i, ny, kstp) + (0.5_rp - cx) * zeta(i, ny + 1, kstp)
        if (cx > cx_lim) then
          zx = zx + (zeta(i, ny, knew) + cx * zeta(i, ny + 1, kstp) - &
               (1.0_rp + cx) * zeta(i, ny, kstp)) * (1.0_rp - cx_lim / cx)**2
        endif
        write(*,'(a,i1,a,i8,a,i4,a,14es26.17)') 'diag: vobc N P', n, '  iic=', iic, &
             '  iif=', iif, '  hx,cff,hI,hG,pnI,pnG,cx,zx,vBry,zBry,vGkstp,vIkstp,vGknew,vmask=', &
             hx, cff, hI, hG, pnI, pnG, cx, zx, vbar_north(i), zeta_north(i), &
             vbar(i, ny + 1, kstp), vbar(i, ny, kstp), vbar(i, ny + 1, knew), vmask(i, ny + 1)
      endif
    endif

    if (decomp%edge_s .and. probe_on) then
      n = 2
      i = iprb_pt(n)
      if (i >= 1 .and. i <= nx) then
        hI = h(i, 1)
        hG = h(i, 0)
        cff = 0.5_rp * (hG + hI)
        hx = sqrt(g / max(cff, 1.0_rp))
        pnI = pn(i, 1)
        pnG = pn(i, 0)
        cx = dtfast * cff * hx * 0.5_rp * (pnG + pnI)
        zx = (0.5_rp + cx) * zeta(i, 1, kstp) + (0.5_rp - cx) * zeta(i, 0, kstp)
        if (cx > cx_lim) then
          zx = zx + (zeta(i, 1, knew) + cx * zeta(i, 0, kstp) - &
               (1.0_rp + cx) * zeta(i, 1, kstp)) * (1.0_rp - cx_lim / cx)**2
        endif
        write(*,'(a,i1,a,i8,a,i4,a,14es26.17)') 'diag: vobc S P', n, '  iic=', iic, &
             '  iif=', iif, '  hx,cff,hI,hG,pnI,pnG,cx,zx,vBry,zBry,vGkstp,vIkstp,vGknew,vmask=', &
             hx, cff, hI, hG, pnI, pnG, cx, zx, vbar_south(i), zeta_south(i), &
             vbar(i, 1, kstp), vbar(i, 2, kstp), vbar(i, 1, knew), vmask(i, 1)
      endif
    endif
    flush(6)
  end subroutine dump_vbar_obc

  subroutine dump_vic(tag)
    ! Restart / halo / pre-RHS v at P1 neighbors i-1:i+2, last interior V
    ! and north-row ghosts. Gated to probe_steps so day runs stay quiet.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, n, nx, ny, slot
    integer(ip) :: js(5), ks(4)
    real(rp) :: vv, vw, ve, vs, vn, vm

    if (.not. probe_on) return
    call sync_arch()
    nx = decomp%nx
    ny = decomp%ny
    js = (/ ny - 2, ny - 1, ny, ny + 1, ny + 2 /)
    ks = (/ 1, 50, 99, 100 /)
    do i = iprb_pt(1) - 1, iprb_pt(1) + 2
      if (i < 1 .or. i > nx) cycle
      do n = 1, 3
        slot = n
        do j = 1, 5
          if (js(j) < decomp%js .or. js(j) > decomp%je) cycle
          do k = 1, 4
            if (ks(k) < 1 .or. ks(k) > decomp%nz) cycle
            vv = v(i, js(j), ks(k), slot)
            vw = 0.0_rp; ve = 0.0_rp; vs = 0.0_rp; vn = 0.0_rp
            vm = vmask(i, js(j))
            if (i - 1 >= decomp%is) vw = v(i - 1, js(j), ks(k), slot)
            if (i + 1 <= decomp%ie) ve = v(i + 1, js(j), ks(k), slot)
            if (js(j) - 1 >= decomp%js) vs = v(i, js(j) - 1, ks(k), slot)
            if (js(j) + 1 <= decomp%je) vn = v(i, js(j) + 1, ks(k), slot)
            write(*,'(a,i6,a,i6,a,i2,a,i4,a,a,a,i8,a,6es26.17)') 'diag: vic  i=', i, &
                 ' j=', js(j), '  slot=', slot, '  k=', ks(k), '  ', trim(tag), &
                 '  iic=', iic, '  v,vW,vE,vS,vN,vmask=', vv, vw, ve, vs, vn, vm
          enddo
        enddo
      enddo
    enddo
    flush(6)
  end subroutine dump_vic

end module geom

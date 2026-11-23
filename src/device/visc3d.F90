module visc3d_mod
  ! Harmonic viscosity along s-surfaces (UV_VIS2). Adds to rufrc/rvfrc
  ! and to Hz*u, Hz*v at nnew (already mass-weighted after step3d_uv1).
  ! cff is per unit area [m2/s2] and rufrc/rvfrc are integrated over the
  ! cell area [m4/s2], so cff enters them times dm_u*dn_u (dm_v*dn_v), as
  ! in UCLA ROMS from ucla-roms#473.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use halo_mod
  use uv_rhs, only: dump_vn3
  use arch_mod, only: sync_arch
  use geom, only: dump_rv3d
  implicit none
  private
  public :: visc3d

contains

  subroutine visc3d()
    integer(ip) :: i, j, k, nx, ny, nz, i_u0, i_u1, j_v0, lim, ml
    real(rp) :: cff
    integer(ip) :: n_stp, n_new
    real(rp) :: dt_loc, rvfrc_p1_0

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_stp = nstp
    n_new = nnew
    dt_loc = dt
    i_u0 = 1
    if (decomp%edge_w) i_u0 = 2
    i_u1 = nx
    j_v0 = 1
    if (decomp%edge_s) j_v0 = 2
    ! u, v(nstp) and Hz are current: no exchange (2026-10-08 audit).
    ! P1 rvfrc probe: host reads only when probes are on.
    rvfrc_p1_0 = 0.0_rp
    i = iprb_pt(1)
    j = jprb_pt(1)
    if (probe_on .and. i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) then
      call sync_arch()
      rvfrc_p1_0 = rvfrc(i, j)
    endif

    if (probe_on) then
      ! Probe steps: per-level path (dump_visc reads the 2-D work arrays).
      do k = 1, nz
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 0, ny
          do i = 0, nx
            cff = 0.5_rp * Hz(i, j, k) * visc2_r(i, j) * ( &
                  dn_r(i, j) * pm(i, j) * ((pn(i, j) + pn(i + 1, j)) * u(i + 1, j, k, n_stp) - &
                                           (pn(i - 1, j) + pn(i, j)) * u(i, j, k, n_stp)) - &
                  dm_r(i, j) * pn(i, j) * ((pm(i, j) + pm(i, j + 1)) * v(i, j + 1, k, n_stp) - &
                                           (pm(i, j - 1) + pm(i, j)) * v(i, j, k, n_stp)))
            wrk2a(i, j) = cff * dn_r(i, j) * dn_r(i, j)
            wrk2b(i, j) = -cff * dm_r(i, j) * dm_r(i, j)
          enddo
        enddo
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny + 1
          do i = 1, nx + 1
            cff = 0.125_rp * (Hz(i - 1, j, k) + Hz(i, j, k) + Hz(i - 1, j - 1, k) + Hz(i, j - 1, k)) * &
                  visc2_p(i, j) * ( &
                  0.25_rp * (pm(i - 1, j) + pm(i, j) + pm(i - 1, j - 1) + pm(i, j - 1)) * dn_p(i, j) * &
                  ((pn(i, j - 1) + pn(i, j)) * v(i, j, k, n_stp) - &
                   (pn(i - 1, j - 1) + pn(i - 1, j)) * v(i - 1, j, k, n_stp)) + &
                  0.25_rp * (pn(i - 1, j) + pn(i, j) + pn(i - 1, j - 1) + pn(i, j - 1)) * dm_p(i, j) * &
                  ((pm(i - 1, j) + pm(i, j)) * u(i, j, k, n_stp) - &
                   (pm(i - 1, j - 1) + pm(i, j - 1)) * u(i, j - 1, k, n_stp))) * pmask(i, j)
            wrk2c(i, j) = cff * dm_p(i, j) * dm_p(i, j)
            wrk2d(i, j) = cff * dn_p(i, j) * dn_p(i, j)
          enddo
        enddo
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny
          do i = i_u0, i_u1
            cff = 0.125_rp * (pm(i - 1, j) + pm(i, j)) * (pn(i - 1, j) + pn(i, j)) * ( &
                  (pn(i - 1, j) + pn(i, j)) * (wrk2a(i, j) - wrk2a(i - 1, j)) + &
                  (pm(i - 1, j) + pm(i, j)) * (wrk2c(i, j + 1) - wrk2c(i, j)))
            rufrc(i, j) = rufrc(i, j) + cff * dm_u(i, j) * dn_u(i, j)
            u(i, j, k, n_new) = u(i, j, k, n_new) + dt_loc * cff
          enddo
        enddo
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = j_v0, ny
          do i = 1, nx
            cff = 0.125_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1)) * ( &
                  (pn(i, j - 1) + pn(i, j)) * (wrk2d(i + 1, j) - wrk2d(i, j)) + &
                  (pm(i, j - 1) + pm(i, j)) * (wrk2b(i, j) - wrk2b(i, j - 1)))
            rvfrc(i, j) = rvfrc(i, j) + cff * dm_v(i, j) * dn_v(i, j)
            v(i, j, k, n_new) = v(i, j, k, n_new) + dt_loc * cff
          enddo
        enddo
        if (k == 1 .or. k == 50 .or. k == 99 .or. k == nz) call dump_visc(k)
      enddo
    else
      ! Flux stages for all levels, then one thread per column adds them
      ! in k order (rufrc/rvfrc sums match the per-level path).
      call ensure_wrk3()
      lim = dry_lim(G_MOM)
      ! zero viscosity within 2 (3 for the stress kernels, read one cell
      ! away) cells: the update is +-0, skip (gpu_state mixact/mixact3)
      ml = mix_lim
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 0, ny
          do i = 0, nx
            if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
            if (ml > 0) then; if (mixact3(i, j) == 0) cycle; end if
            cff = 0.5_rp * Hz(i, j, k) * visc2_r(i, j) * ( &
                  dn_r(i, j) * pm(i, j) * ((pn(i, j) + pn(i + 1, j)) * u(i + 1, j, k, n_stp) - &
                                           (pn(i - 1, j) + pn(i, j)) * u(i, j, k, n_stp)) - &
                  dm_r(i, j) * pn(i, j) * ((pm(i, j) + pm(i, j + 1)) * v(i, j + 1, k, n_stp) - &
                                           (pm(i, j - 1) + pm(i, j)) * v(i, j, k, n_stp)))
            wrk3a(i, j, k) = cff * dn_r(i, j) * dn_r(i, j)
            wrk3b(i, j, k) = -cff * dm_r(i, j) * dm_r(i, j)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny + 1
          do i = 1, nx + 1
            if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
            if (ml > 0) then; if (mixact3(i, j) == 0) cycle; end if
            cff = 0.125_rp * (Hz(i - 1, j, k) + Hz(i, j, k) + Hz(i - 1, j - 1, k) + Hz(i, j - 1, k)) * &
                  visc2_p(i, j) * ( &
                  0.25_rp * (pm(i - 1, j) + pm(i, j) + pm(i - 1, j - 1) + pm(i, j - 1)) * dn_p(i, j) * &
                  ((pn(i, j - 1) + pn(i, j)) * v(i, j, k, n_stp) - &
                   (pn(i - 1, j - 1) + pn(i - 1, j)) * v(i - 1, j, k, n_stp)) + &
                  0.25_rp * (pn(i - 1, j) + pn(i, j) + pn(i - 1, j - 1) + pn(i, j - 1)) * dm_p(i, j) * &
                  ((pm(i - 1, j) + pm(i, j)) * u(i, j, k, n_stp) - &
                   (pm(i - 1, j - 1) + pm(i, j - 1)) * u(i, j - 1, k, n_stp))) * pmask(i, j)
            wrk3c(i, j, k) = cff * dm_p(i, j) * dm_p(i, j)
            wrk3d(i, j, k) = cff * dn_p(i, j) * dn_p(i, j)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = i_u0, i_u1
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
          if (ml > 0) then; if (mixact(i, j) == 0) cycle; end if
          do k = 1, nz
            cff = 0.125_rp * (pm(i - 1, j) + pm(i, j)) * (pn(i - 1, j) + pn(i, j)) * ( &
                  (pn(i - 1, j) + pn(i, j)) * (wrk3a(i, j, k) - wrk3a(i - 1, j, k)) + &
                  (pm(i - 1, j) + pm(i, j)) * (wrk3c(i, j + 1, k) - wrk3c(i, j, k)))
            rufrc(i, j) = rufrc(i, j) + cff * dm_u(i, j) * dn_u(i, j)
            u(i, j, k, n_new) = u(i, j, k, n_new) + dt_loc * cff
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = j_v0, ny
        do i = 1, nx
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
          if (ml > 0) then; if (mixact(i, j) == 0) cycle; end if
          do k = 1, nz
            cff = 0.125_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1)) * ( &
                  (pn(i, j - 1) + pn(i, j)) * (wrk3d(i + 1, j, k) - wrk3d(i, j, k)) + &
                  (pm(i, j - 1) + pm(i, j)) * (wrk3b(i, j, k) - wrk3b(i, j - 1, k)))
            rvfrc(i, j) = rvfrc(i, j) + cff * dm_v(i, j) * dn_v(i, j)
            v(i, j, k, n_new) = v(i, j, k, n_new) + dt_loc * cff
          enddo
        enddo
      enddo
    endif
    call dump_visc_sum()
    call dump_vn3('after visc3d')
    i = iprb_pt(1)
    j = jprb_pt(1)
    cff = 0.0_rp
    if (probe_on .and. i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) then
      call sync_arch()
      cff = rvfrc(i, j) - rvfrc_p1_0
    endif
    call dump_rv3d('after visc3d', cff)

    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_end()
  end subroutine visc3d

  subroutine dump_visc(k)
    ! visc3d V increment at P1 neighbors. wrk2d/wrk2b are VFx/VFe.
    integer(ip), intent(in) :: k
    integer(ip) :: i, j
    real(rp) :: cff, vw, ve, vs, vn

    if (.not. probe_on) return
    call sync_arch()
    do i = iprb_pt(1) - 1, iprb_pt(1) + 2
      if (i < 1 .or. i > decomp%nx) cycle
      do j = jprb_pt(1), jprb_pt(1) + 1
        if (j < 1 .or. j > decomp%ny) cycle
        cff = 0.125_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1)) * ( &
              (pn(i, j - 1) + pn(i, j)) * (wrk2d(i + 1, j) - wrk2d(i, j)) + &
              (pm(i, j - 1) + pm(i, j)) * (wrk2b(i, j) - wrk2b(i, j - 1)))
        vw = 0.0_rp; ve = 0.0_rp; vs = 0.0_rp; vn = 0.0_rp
        if (i - 1 >= decomp%is) vw = v(i - 1, j, k, nstp)
        if (i + 1 <= decomp%ie) ve = v(i + 1, j, k, nstp)
        if (j - 1 >= decomp%js) vs = v(i, j - 1, k, nstp)
        if (j + 1 <= decomp%je) vn = v(i, j + 1, k, nstp)
        write(*,'(a,i6,a,i6,a,i8,a,i4,a,8es26.17)') 'diag: visc  i=', i, &
             ' j=', j, '  after visc3d  iic=', iic, '  k=', k, &
             '  v_stp,v_new,cff,dtcff,VFx,VFxE,VFe,VFeS=', &
             v(i, j, k, nstp), v(i, j, k, nnew), cff, dt * cff, &
             wrk2d(i, j), wrk2d(i + 1, j), wrk2b(i, j), wrk2b(i, j - 1)
        write(*,'(a,i6,a,i6,a,i8,a,i4,a,9es26.17)') 'diag: visc  i=', i, &
             ' j=', j, '  visc nbor  iic=', iic, '  k=', k, &
             '  vW,vE,vS,vN,vmask,pmask,pmaskE,visc2r,visc2p=', &
             vw, ve, vs, vn, vmask(i, j), pmask(i, j), pmask(i + 1, j), &
             visc2_r(i, j), visc2_p(i, j)
        call dump_vfx_psi(i, j, k)
      enddo
    enddo
    flush(6)
  end subroutine dump_visc

  subroutine dump_vfx_psi(i, j, k)
    ! VFx psi-stencil at (i,j): four Hz corners, u(nstp) S/N, v(nstp) W/E.
    integer(ip), intent(in) :: i, j, k
    real(rp) :: HzSW, HzSE, HzNW, HzNE, uS, uN, vW, vE
    real(rp) :: visc2p, pmsk, dnp, dmp, du, dv

    HzSW = Hz(i - 1, j - 1, k)
    HzSE = Hz(i, j - 1, k)
    HzNW = Hz(i - 1, j, k)
    HzNE = Hz(i, j, k)
    uS = u(i, j - 1, k, nstp)
    uN = u(i, j, k, nstp)
    vW = v(i - 1, j, k, nstp)
    vE = v(i, j, k, nstp)
    visc2p = visc2_p(i, j)
    pmsk = pmask(i, j)
    dnp = dn_p(i, j)
    dmp = dm_p(i, j)
    dv = (pn(i, j - 1) + pn(i, j)) * vE - (pn(i - 1, j - 1) + pn(i - 1, j)) * vW
    du = (pm(i - 1, j) + pm(i, j)) * uN - (pm(i - 1, j - 1) + pm(i, j - 1)) * uS
    write(*,'(a,i6,a,i6,a,i8,a,i4,a,5es26.17)') 'diag: vfx  i=', i, &
         ' j=', j, '  VFx Hz  iic=', iic, '  k=', k, &
         '  HzSW,HzSE,HzNW,HzNE,VFx=', HzSW, HzSE, HzNW, HzNE, wrk2d(i, j)
    write(*,'(a,i6,a,i6,a,i8,a,i4,a,10es26.17)') 'diag: vfx  i=', i, &
         ' j=', j, '  VFx uv  iic=', iic, '  k=', k, &
         '  uS,uN,vW,vE,visc2p,pmask,dnp,dmp,du,dv=', &
         uS, uN, vW, vE, visc2p, pmsk, dnp, dmp, du, dv
  end subroutine dump_vfx_psi

  subroutine dump_visc_sum()
    integer(ip) :: i, j, k, nz
    real(rp) :: vsum

    if (.not. probe_on) return
    call sync_arch()
    nz = decomp%nz
    do i = iprb_pt(1) - 1, iprb_pt(1) + 2
      if (i < 1 .or. i > decomp%nx) cycle
      do j = jprb_pt(1), jprb_pt(1) + 1
        if (j < 1 .or. j > decomp%ny) cycle
        vsum = 0.0_rp
        do k = 1, nz
          vsum = vsum + v(i, j, k, nnew)
        enddo
        write(*,'(a,i6,a,i6,a,i8,a,5es26.17)') 'diag: visc  i=', i, &
             ' j=', j, '  after visc3d sum  iic=', iic, &
             '  vsum,v1,v50,v99,v100=', vsum, v(i, j, 1, nnew), &
             v(i, j, 50, nnew), v(i, j, 99, nnew), v(i, j, nz, nnew)
      enddo
    enddo
    flush(6)
  end subroutine dump_visc_sum

end module visc3d_mod

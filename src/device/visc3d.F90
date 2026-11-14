#include "mem_attr.inc"
module visc3d_mod
  ! Harmonic viscosity along s-surfaces (UV_VIS2). Adds to rufrc/rvfrc
  ! and to Hz*u, Hz*v at nnew (already mass-weighted after step3d_uv1).
  ! IMEX_IGW imex_mix_split = 1 (CPU visc3d_mod under IMEX_IGW, c65bb854):
  ! the Laplacian acts on this step's corrected velocity (split_vel), not u^n.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use halo_mod
  use uv_rhs, only: dump_vn3
  use arch_mod, only: sync_arch
  use geom, only: dump_rv3d
  use imex_igw, only: imex_on, imex_mix_split, imex_solver_enabled
  implicit none
  private
  public :: visc3d

  ! imex_mix_split = 1: the velocity the Laplacian acts on (u, v points), the depth means of the
  ! step's increment and the face-thickness sums (split_vel)
  real(rp), WORK_ATTR allocatable :: uk3(:,:,:), vk3(:,:,:), dub(:,:), dvb(:,:)

contains

  subroutine visc3d()
    integer(ip) :: i, j, k, nx, ny, nz, i_u0, i_u1, j_v0, lim, ml
    real(rp) :: cff
    integer(ip) :: n_stp, n_new
    real(rp) :: dt_loc, rvfrc_p1_0
    logical :: spl

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

    ! IMEX_IGW imex_mix_split = 1: the stress from the corrected momentum (split_vel)
    spl = imex_on .and. imex_mix_split == 1 .and. imex_solver_enabled
    if (spl) call split_vel()

    if (probe_on .and. .not. spl) then
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
            rufrc(i, j) = rufrc(i, j) + cff
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
            rvfrc(i, j) = rvfrc(i, j) + cff
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
      if (spl) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 0, ny
          do i = 0, nx
            if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
            if (ml > 0) then; if (mixact3(i, j) == 0) cycle; end if
            cff = 0.5_rp * Hz(i, j, k) * visc2_r(i, j) * ( &
                  dn_r(i, j) * pm(i, j) * ((pn(i, j) + pn(i + 1, j)) * uk3(i + 1, j, k) - &
                                           (pn(i - 1, j) + pn(i, j)) * uk3(i, j, k)) - &
                  dm_r(i, j) * pn(i, j) * ((pm(i, j) + pm(i, j + 1)) * vk3(i, j + 1, k) - &
                                           (pm(i, j - 1) + pm(i, j)) * vk3(i, j, k)))
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
                  ((pn(i, j - 1) + pn(i, j)) * vk3(i, j, k) - &
                   (pn(i - 1, j - 1) + pn(i - 1, j)) * vk3(i - 1, j, k)) + &
                  0.25_rp * (pn(i - 1, j) + pn(i, j) + pn(i - 1, j - 1) + pn(i, j - 1)) * dm_p(i, j) * &
                  ((pm(i - 1, j) + pm(i, j)) * uk3(i, j, k) - &
                   (pm(i - 1, j - 1) + pm(i, j - 1)) * uk3(i, j - 1, k))) * pmask(i, j)
            wrk3c(i, j, k) = cff * dm_p(i, j) * dm_p(i, j)
            wrk3d(i, j, k) = cff * dn_p(i, j) * dn_p(i, j)
          enddo
        enddo
      enddo
      else
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
      endif
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
            rufrc(i, j) = rufrc(i, j) + cff
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
            rvfrc(i, j) = rvfrc(i, j) + cff
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

  subroutine split_vel()
    ! IMEX_IGW imex_mix_split = 1 (CPU visc3d_s_tile under IMEX_IGW, c65bb854 with the land-face mask
    ! of 4cef380a): uk3, vk3 = the corrected momentum u, v(nnew) / Hz_f of this step less the depth
    ! mean of its increment, dub = sum_k (u(nnew) - Hz_f u^n) / sum_k Hz_f (the depth mean of u(nnew)
    ! is the explicit barotropic step that step3d_uv2 replaces, so it stays at u^n), times umask (land
    ! faces carry the explicit rhs in u(nnew), u^n is masked); the physical edges as the stock field
    ! has them: normal faces u^n (0 at walls, the OBC's value at open faces), the tangential ghost row
    ! gamma2 x interior (masked) at a closed edge and u^n at an open one, west/east before south/north
    ! as on the CPU. u, v(nnew) halos: exchanged after the corrector's correction (step_driver).
    ! Points 0..nx+1, 0..ny+1 (the stencil of the flux kernels); the CPU skips u point 0 at a west
    ! edge (v point 0 at a south edge), whose fluxes are not used.
    real(rp), parameter :: gamma2 = 1.0_rp   ! as obc.F90 (free slip, the only value on the GPU)
    integer(ip) :: i, j, k, nx, ny, nz, n_new, n_stp, j0, i0
    real(rp) :: cff, su, sh
    logical :: ow, oe, os, on, ew, ee, es, en
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n   ! scalars for the kernels
    n_new = nnew; n_stp = nstp
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. allocated(uk3)) then
      allocate(uk3(decomp%is:decomp%ie, decomp%js:decomp%je, nz), vk3(decomp%is:decomp%ie, decomp%js:decomp%je, nz))
      allocate(dub(decomp%is:decomp%ie, decomp%js:decomp%je), dvb(decomp%is:decomp%ie, decomp%js:decomp%je))
      uk3 = 0.0_rp; vk3 = 0.0_rp; dub = 0.0_rp; dvb = 0.0_rp
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 0, nx + 1
        su = 0.0_rp; sh = 0.0_rp
        do k = 1, nz
          cff = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
          su = su + u(i, j, k, n_new) - cff * u(i, j, k, n_stp)
          sh = sh + cff
        enddo
        dub(i, j) = su / sh
        su = 0.0_rp; sh = 0.0_rp
        do k = 1, nz
          cff = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
          su = su + v(i, j, k, n_new) - cff * v(i, j, k, n_stp)
          sh = sh + cff
        enddo
        dvb(i, j) = su / sh
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 0, ny + 1
        do i = 0, nx + 1
          uk3(i, j, k) = (u(i, j, k, n_new) / (0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))) - dub(i, j)) * umask(i, j)
          vk3(i, j, k) = (v(i, j, k, n_new) / (0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))) - dvb(i, j)) * vmask(i, j)
        enddo
      enddo
    enddo
    ! west and east edges (CPU: Uk(istr), Vk(istr-1) for j = jstr-1.., jstrV-1..jend+1)
    if (ew .or. ee) then
      j0 = 0; if (es) j0 = 1
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 0, ny + 1
          if (ew) then
            uk3(1, j, k) = u(1, j, k, n_stp)
            if (j >= j0) then
              if (ow) then
                vk3(0, j, k) = v(0, j, k, n_stp)
              else
                vk3(0, j, k) = gamma2 * vk3(1, j, k) * vmask(0, j)
              endif
            endif
          endif
          if (ee) then
            uk3(nx + 1, j, k) = u(nx + 1, j, k, n_stp)
            if (j >= j0) then
              if (oe) then
                vk3(nx + 1, j, k) = v(nx + 1, j, k, n_stp)
              else
                vk3(nx + 1, j, k) = gamma2 * vk3(nx, j, k) * vmask(nx + 1, j)
              endif
            endif
          endif
        enddo
      enddo
    endif
    ! south and north edges, after west and east (the corner faces take the west/east values first)
    if (es .or. en) then
      i0 = 0; if (ew) i0 = 1
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 1, nz
        do i = 0, nx + 1
          if (es) then
            vk3(i, 1, k) = v(i, 1, k, n_stp)
            if (i >= i0) then
              if (os) then
                uk3(i, 0, k) = u(i, 0, k, n_stp)
              else
                uk3(i, 0, k) = gamma2 * uk3(i, 1, k) * umask(i, 0)
              endif
            endif
          endif
          if (en) then
            vk3(i, ny + 1, k) = v(i, ny + 1, k, n_stp)
            if (i >= i0) then
              if (on) then
                uk3(i, ny + 1, k) = u(i, ny + 1, k, n_stp)
              else
                uk3(i, ny + 1, k) = gamma2 * uk3(i, ny, k) * umask(i, ny + 1)
              endif
            endif
          endif
        enddo
      enddo
    endif
  end subroutine split_vel

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

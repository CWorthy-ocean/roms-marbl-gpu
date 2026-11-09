module advect
  ! Horizontal UP3 (UPSTREAM_TS + LAND_CURV, wet-only face T) and
  ! thickness-aware vertical C4 (predictor) / U3 (corrector). Fluxes
  ! only; LF-AM3 and Hz(n+1) updates stay in pre_step3d / step3d_t.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use halo_mod
  use time_state
  use arch_mod, only: sync_arch
  implicit none
  private

  public :: horiz_up3_fluxes
  public :: horiz_c4_fluxes
  public :: vert_c4_fluxes
  public :: vert_u3_fluxes
  public :: tracer_advect_fluxes
  public :: dump_horiz_flux_probe
  public :: fuse_tracers
  public :: pre_tracers_horiz_fused
  public :: pre_tracers_vert_fused
  public :: cor_tracers_fused

contains

  subroutine tracer_advect_fluxes(tidx, itrc, centered)
    integer(ip), intent(in) :: tidx, itrc
    logical,     intent(in) :: centered
    call horiz_up3_fluxes(tidx, itrc)
    if (centered) then
      call vert_c4_fluxes(tidx, itrc)
    else
      call vert_u3_fluxes(tidx, itrc)
    endif
  end subroutine tracer_advect_fluxes

  subroutine horiz_up3_fluxes(tidx, itrc)
    ! UPSTREAM_TS + LAND_CURV, fused per direction so curvature cannot
    ! race the flux store through 2-D work arrays. Land T=0 is kept out
    ! of the centered face value; finished wrk3a/wrk3b are zero on land faces.
    integer(ip), intent(in) :: tidx, itrc
    integer(ip) :: i, j, k, nx, ny, nz, t_slot, itr
    integer(ip) :: west, east, south, north
    real(rp) :: d_l, d_c, d_r, curv_l, curv_r, t_l, t_c, face_t, uflux

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    call ensure_wrk3()
    t_slot = tidx
    itr = itrc
    ! edge codes: 0 interior, 1 physical edge (wall), 2 open-boundary edge,
    ! whose faces get the inflow term (open_edge_codes)
    call open_edge_codes(west, east, south, north)

    call hg_begin()
    call hg_add(F_UMASK)
    call hg_add(F_VMASK)
    call hg_add(F_RMASK)
    call hg_add(F_T, t_slot, itr)
    call hg_add(F_FLXU)
    call hg_add(F_FLXV)
    call hg_end()

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        do k = 1, nz
          t_c = t(i, j, k, t_slot, itr)
          t_l = t(i - 1, j, k, t_slot, itr)
          d_c = apply_mask(t_c - t_l, umask(i, j))
          if (west >= 1 .and. i == 1) then
            d_l = d_c
          else
            d_l = apply_mask(t_l - t(i - 2, j, k, t_slot, itr), umask(i - 1, j))
          endif
          if (east >= 1 .and. i == nx + 1) then
            d_r = d_c
          else
            d_r = apply_mask(t(i + 1, j, k, t_slot, itr) - t_c, umask(i + 1, j))
          endif
          curv_l = (d_c - d_l) * umask(i - 1, j) * umask(i, j)
          curv_r = (d_r - d_c) * umask(i, j) * umask(i + 1, j)
          if (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp) then
            face_t = 0.5_rp * (t_l + t_c)
          elseif (rmask(i - 1, j) > 0.5_rp) then
            face_t = t_l
          elseif (rmask(i, j) > 0.5_rp) then
            face_t = t_c
          else
            face_t = 0.0_rp
          endif
          uflux = FlxU(i, j, k)
          wrk3a(i, j, k) = apply_mask(face_t * uflux - coef_up3 * ( &
               curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp)), &
               umask(i, j))
          if (west == 2 .and. i == 1) wrk3a(i, j, k) = wrk3a(i, j, k) + &
               apply_mask(0.5_rp * (t_l - t_c) * max(uflux, 0.0_rp), umask(i, j))
          if (east == 2 .and. i == nx + 1) wrk3a(i, j, k) = wrk3a(i, j, k) + &
               apply_mask(0.5_rp * (t_c - t_l) * min(uflux, 0.0_rp), umask(i, j))
        enddo
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        do k = 1, nz
          t_c = t(i, j, k, t_slot, itr)
          t_l = t(i, j - 1, k, t_slot, itr)
          d_c = apply_mask(t_c - t_l, vmask(i, j))
          if (south >= 1 .and. j == 1) then
            d_l = d_c
          else
            d_l = apply_mask(t_l - t(i, j - 2, k, t_slot, itr), vmask(i, j - 1))
          endif
          if (north >= 1 .and. j == ny + 1) then
            d_r = d_c
          else
            d_r = apply_mask(t(i, j + 1, k, t_slot, itr) - t_c, vmask(i, j + 1))
          endif
          curv_l = (d_c - d_l) * vmask(i, j - 1) * vmask(i, j)
          curv_r = (d_r - d_c) * vmask(i, j) * vmask(i, j + 1)
          if (rmask(i, j - 1) > 0.5_rp .and. rmask(i, j) > 0.5_rp) then
            face_t = 0.5_rp * (t_l + t_c)
          elseif (rmask(i, j - 1) > 0.5_rp) then
            face_t = t_l
          elseif (rmask(i, j) > 0.5_rp) then
            face_t = t_c
          else
            face_t = 0.0_rp
          endif
          uflux = FlxV(i, j, k)
          wrk3b(i, j, k) = apply_mask(face_t * uflux - coef_up3 * ( &
               curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp)), &
               vmask(i, j))
          if (south == 2 .and. j == 1) wrk3b(i, j, k) = wrk3b(i, j, k) + &
               apply_mask(0.5_rp * (t_l - t_c) * max(uflux, 0.0_rp), vmask(i, j))
          if (north == 2 .and. j == ny + 1) wrk3b(i, j, k) = wrk3b(i, j, k) + &
               apply_mask(0.5_rp * (t_c - t_l) * min(uflux, 0.0_rp), vmask(i, j))
        enddo
      enddo
    enddo

    call hg_begin()
    call hg_add(F_WRK3A)
    call hg_add(F_WRK3B)
    call hg_end()
  end subroutine horiz_up3_fluxes

  subroutine horiz_c4_fluxes(tidx, itrc)
    ! Centered 4th-order horizontal fluxes: CPU compute_horiz_tracer_fluxes.h
    ! without UPSTREAM_TS/AKIMA, as used by the pre_step3d predictor. Face
    ! gradients are masked; at open edges the outermost one is copied out.
    integer(ip), intent(in) :: tidx, itrc
    integer(ip) :: i, j, k, nx, ny, nz, t_slot, itr
    integer(ip) :: west, east, south, north
    real(rp) :: d_l, d_c, d_r, t_l, t_c

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    call ensure_wrk3()
    t_slot = tidx
    itr = itrc
    ! edge codes: 0 interior, 1 physical edge (wall), 2 open-boundary edge,
    ! whose faces get the inflow term (open_edge_codes)
    call open_edge_codes(west, east, south, north)

    call hg_begin()
    call hg_add(F_UMASK)
    call hg_add(F_VMASK)
    call hg_add(F_T, t_slot, itr)
    call hg_add(F_FLXU)
    call hg_add(F_FLXV)
    call hg_end()

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        do k = 1, nz
          t_c = t(i, j, k, t_slot, itr)
          t_l = t(i - 1, j, k, t_slot, itr)
          d_c = apply_mask(t_c - t_l, umask(i, j))
          if (west >= 1 .and. i == 1) then
            d_l = d_c
          else
            d_l = apply_mask(t_l - t(i - 2, j, k, t_slot, itr), umask(i - 1, j))
          endif
          if (east >= 1 .and. i == nx + 1) then
            d_r = d_c
          else
            d_r = apply_mask(t(i + 1, j, k, t_slot, itr) - t_c, umask(i + 1, j))
          endif
          wrk3a(i, j, k) = apply_mask(0.5_rp * (t_c + t_l - (d_r - d_l) / 6.0_rp) * &
               FlxU(i, j, k), umask(i, j))
          if (west == 2 .and. i == 1) wrk3a(i, j, k) = wrk3a(i, j, k) + &
               apply_mask(0.5_rp * (t_l - t_c) * max(FlxU(i, j, k), 0.0_rp), umask(i, j))
          if (east == 2 .and. i == nx + 1) wrk3a(i, j, k) = wrk3a(i, j, k) + &
               apply_mask(0.5_rp * (t_c - t_l) * min(FlxU(i, j, k), 0.0_rp), umask(i, j))
        enddo
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        do k = 1, nz
          t_c = t(i, j, k, t_slot, itr)
          t_l = t(i, j - 1, k, t_slot, itr)
          d_c = apply_mask(t_c - t_l, vmask(i, j))
          if (south >= 1 .and. j == 1) then
            d_l = d_c
          else
            d_l = apply_mask(t_l - t(i, j - 2, k, t_slot, itr), vmask(i, j - 1))
          endif
          if (north >= 1 .and. j == ny + 1) then
            d_r = d_c
          else
            d_r = apply_mask(t(i, j + 1, k, t_slot, itr) - t_c, vmask(i, j + 1))
          endif
          wrk3b(i, j, k) = apply_mask(0.5_rp * (t_c + t_l - (d_r - d_l) / 6.0_rp) * &
               FlxV(i, j, k), vmask(i, j))
          if (south == 2 .and. j == 1) wrk3b(i, j, k) = wrk3b(i, j, k) + &
               apply_mask(0.5_rp * (t_l - t_c) * max(FlxV(i, j, k), 0.0_rp), vmask(i, j))
          if (north == 2 .and. j == ny + 1) wrk3b(i, j, k) = wrk3b(i, j, k) + &
               apply_mask(0.5_rp * (t_c - t_l) * min(FlxV(i, j, k), 0.0_rp), vmask(i, j))
        enddo
      enddo
    enddo

    call hg_begin()
    call hg_add(F_WRK3A)
    call hg_add(F_WRK3B)
    call hg_end()
  end subroutine horiz_c4_fluxes

  subroutine dump_horiz_flux_probe(tag, dtau_in)
    ! Face-by-face UP3 dump at the coastal cell that climbs on GPU.
    ! Call immediately after horiz_up3_fluxes(itemp), before the update.
    character(len=*), intent(in) :: tag
    real(rp),         intent(in) :: dtau_in
    integer(ip), parameter :: iprobe = 154_ip, jprobe = 136_ip
    integer(ip) :: i, j, k, kmax, nz
    real(rp) :: div, dT, absmax, hzv

    if (.not. probe_on) return
    call sync_arch()
    i = iprobe
    j = jprobe
    nz = decomp%nz
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return

    write(*,'(a,a,a,i8,a,2i6,a,3es14.6)') 'diag: flux B  ', trim(tag), &
         '  iic=', iic, '  i,j=', i, j, '  pm,pn,dtau=', pm(i, j), pn(i, j), dtau_in
    write(*,'(a,a,a,9es11.3)') 'diag: flux B  ', trim(tag), &
         '  mask umW,umE,vmS,vmN,rm WCESN=', &
         umask(i, j), umask(i + 1, j), vmask(i, j), vmask(i, j + 1), &
         rmask(i - 1, j), rmask(i, j), rmask(i + 1, j), rmask(i, j - 1), rmask(i, j + 1)
    write(*,'(a,a,a,5es14.6)') 'diag: flux B  ', trim(tag), &
         '  Tnrhs(k=nz) WCESN=', &
         t(i - 1, j, nz, nrhs, itemp), t(i, j, nz, nrhs, itemp), &
         t(i + 1, j, nz, nrhs, itemp), t(i, j - 1, nz, nrhs, itemp), &
         t(i, j + 1, nz, nrhs, itemp)
    write(*,'(a,a,a,5es14.6)') 'diag: flux B  ', trim(tag), &
         '  Tnrhs(k=1) WCESN=', &
         t(i - 1, j, 1, nrhs, itemp), t(i, j, 1, nrhs, itemp), &
         t(i + 1, j, 1, nrhs, itemp), t(i, j - 1, 1, nrhs, itemp), &
         t(i, j + 1, 1, nrhs, itemp)

    absmax = -1.0_rp
    kmax = 1
    do k = 1, nz
      div = pm(i, j) * pn(i, j) * (wrk3a(i + 1, j, k) - wrk3a(i, j, k) + &
                                   wrk3b(i, j + 1, k) - wrk3b(i, j, k))
      if (abs(div) > absmax) then
        absmax = abs(div)
        kmax = k
      endif
    enddo
    write(*,'(a,a,a,i4,es14.6)') 'diag: flux B  ', trim(tag), &
         '  kmax|div|,|div|=', kmax, absmax

    do k = 1, nz
      hzv = max(Hz(i, j, k), 1.0e-30_rp)
      div = pm(i, j) * pn(i, j) * (wrk3a(i + 1, j, k) - wrk3a(i, j, k) + &
                                   wrk3b(i, j + 1, k) - wrk3b(i, j, k))
      dT = -dtau_in * div / hzv
      write(*,'(a,a,a,i4,12es13.5)') 'diag: flux B  ', trim(tag), ' allk', k, &
           t(i, j, k, nrhs, itemp), Hz(i, j, k), Hz_bak(i, j, k), Hz_fwd(i, j, k), &
           wrk3a(i, j, k), wrk3a(i + 1, j, k), wrk3b(i, j, k), wrk3b(i, j + 1, k), &
           FlxU(i, j, k), FlxU(i + 1, j, k), FlxV(i, j, k), FlxV(i, j + 1, k)
      if (k /= 1 .and. k /= nz .and. k /= kmax) cycle
      write(*,'(a,a,a,i4,a,es14.6,a,3es14.6)') 'diag: flux B  ', trim(tag), &
           '  k=', k, '  Tnrhs=', t(i, j, k, nrhs, itemp), &
           '  Hz,Hz_bak,Hz_fwd=', Hz(i, j, k), Hz_bak(i, j, k), Hz_fwd(i, j, k)
      write(*,'(a,a,a,i4,a,4es14.6)') 'diag: flux B  ', trim(tag), &
           '  k=', k, '  FX_w,FX_e,FE_s,FE_n=', &
           wrk3a(i, j, k), wrk3a(i + 1, j, k), wrk3b(i, j, k), wrk3b(i, j + 1, k)
      write(*,'(a,a,a,i4,a,4es14.6)') 'diag: flux B  ', trim(tag), &
           '  k=', k, '  FlxU_w,FlxU_e,FlxV_s,FlxV_n=', &
           FlxU(i, j, k), FlxU(i + 1, j, k), FlxV(i, j, k), FlxV(i, j + 1, k)
      write(*,'(a,a,a,i4,a,3es14.6)') 'diag: flux B  ', trim(tag), &
           '  k=', k, '  div,dT_horiz/Hz,dT/Hz_fwd=', div, dT, &
           -dtau_in * div / max(Hz_fwd(i, j, k), 1.0e-30_rp)
    enddo
    flush(6)
  end subroutine dump_horiz_flux_probe

  subroutine vert_c4_fluxes(tidx, itrc)
    integer(ip), intent(in) :: tidx, itrc
    call vert_fluxes(tidx, itrc, .true.)
  end subroutine vert_c4_fluxes

  subroutine vert_u3_fluxes(tidx, itrc)
    integer(ip), intent(in) :: tidx, itrc
    call vert_fluxes(tidx, itrc, .false.)
  end subroutine vert_u3_fluxes

  subroutine vert_fluxes(tidx, itrc, centered)
    integer(ip), intent(in) :: tidx, itrc
    logical,     intent(in) :: centered
    integer(ip) :: i, j, k, nz
    real(rp) :: t_km1, t_k, t_kp1, t_kp2
    real(rp) :: Hz_km1, Hz_k, Hz_kp1, Hz_kp2
    real(rp) :: tplus, tminus
    logical :: at_bottom, at_surface

    nz = decomp%nz
    call ensure_wrk3()
    call hg_begin()
    call hg_add(F_RMASK)
    call hg_add(F_T, tidx, itrc)
    call hg_add(F_HZ)
    call hg_add(F_WE)
    call hg_end()
    ! Zero col_wc on the device (k = 0 and nz stay zero; a host assignment
    ! would migrate the whole array to the CPU every call).
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 0, nz
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          col_wc(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        do k = 1, nz - 1
          at_bottom  = (k == 1)
          at_surface = (k == nz - 1)
          t_k   = t(i, j, k, tidx, itrc)
          t_kp1 = t(i, j, k + 1, tidx, itrc)
          Hz_k   = Hz(i, j, k)
          Hz_kp1 = Hz(i, j, k + 1)
          if (at_bottom) then
            t_km1  = t_k
            Hz_km1 = Hz_k
          else
            t_km1  = t(i, j, k - 1, tidx, itrc)
            Hz_km1 = Hz(i, j, k - 1)
          endif
          if (at_surface) then
            t_kp2  = t_kp1
            Hz_kp2 = Hz_kp1
          else
            t_kp2  = t(i, j, k + 2, tidx, itrc)
            Hz_kp2 = Hz(i, j, k + 2)
          endif
          tplus = thickness_tplus(t_km1, t_k, t_kp1, Hz_km1, Hz_k, Hz_kp1, at_bottom)
          tminus = thickness_tminus(t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2, &
                                    at_bottom, at_surface)
          if (centered) then
            col_wc(i, j, k) = 0.5_rp * (tplus + tminus) * We(i, j, k)
          else
            col_wc(i, j, k) = tplus * max(We(i, j, k), 0.0_rp) + tminus * min(We(i, j, k), 0.0_rp)
          endif
          col_wc(i, j, k) = apply_mask(col_wc(i, j, k), rmask(i, j))
        enddo
      enddo
    enddo
    call halo_fill_3d_w(col_wc, STAGGER_RHO)
  end subroutine vert_fluxes


  ! ---- Tracer-batched advection: all tracers in one launch --------------
  ! The face fluxes are recomputed inline from t(tidx) with the same
  ! arithmetic as horiz_c4/up3_fluxes and vert_fluxes, so wrk3a/wrk3b/col_wc never
  ! go through memory and each cell's geometry is loaded once for all
  ! tracers. Neighbouring threads read t(tidx) while writing other slots,
  ! so tidx must not be a slot the kernel writes (fuse_tracers checks).

  logical function fuse_tracers(tidx, w1, w2)
    ! Probe steps keep the per-tracer path: their dumps read wrk3a/wrk3b/col_wc.
    integer(ip), intent(in) :: tidx, w1, w2
    fuse_tracers = (.not. probe_on) .and. tidx /= w1 .and. tidx /= w2
  end function fuse_tracers

#ifdef USE_CUDA
  attributes(device) &
#endif
  function c4_fx(i, j, k, ts, itr, west, east, nx) result(fx)
    integer(ip), value :: i, j, k, ts, itr, west, east, nx
    real(rp) :: fx, d_l, d_c, d_r, t_l, t_c
    t_c = t(i, j, k, ts, itr)
    t_l = t(i - 1, j, k, ts, itr)
    d_c = apply_mask(t_c - t_l, umask(i, j))
    if (west >= 1 .and. i == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - t(i - 2, j, k, ts, itr), umask(i - 1, j))
    endif
    if (east >= 1 .and. i == nx + 1) then
      d_r = d_c
    else
      d_r = apply_mask(t(i + 1, j, k, ts, itr) - t_c, umask(i + 1, j))
    endif
    fx = apply_mask(0.5_rp * (t_c + t_l - (d_r - d_l) / 6.0_rp) * &
         FlxU(i, j, k), umask(i, j))
    if (west == 2 .and. i == 1) fx = fx + &
         apply_mask(0.5_rp * (t_l - t_c) * max(FlxU(i, j, k), 0.0_rp), umask(i, j))
    if (east == 2 .and. i == nx + 1) fx = fx + &
         apply_mask(0.5_rp * (t_c - t_l) * min(FlxU(i, j, k), 0.0_rp), umask(i, j))
  end function c4_fx

#ifdef USE_CUDA
  attributes(device) &
#endif
  function c4_fe(i, j, k, ts, itr, south, north, ny) result(fe)
    integer(ip), value :: i, j, k, ts, itr, south, north, ny
    real(rp) :: fe, d_l, d_c, d_r, t_l, t_c
    t_c = t(i, j, k, ts, itr)
    t_l = t(i, j - 1, k, ts, itr)
    d_c = apply_mask(t_c - t_l, vmask(i, j))
    if (south >= 1 .and. j == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - t(i, j - 2, k, ts, itr), vmask(i, j - 1))
    endif
    if (north >= 1 .and. j == ny + 1) then
      d_r = d_c
    else
      d_r = apply_mask(t(i, j + 1, k, ts, itr) - t_c, vmask(i, j + 1))
    endif
    fe = apply_mask(0.5_rp * (t_c + t_l - (d_r - d_l) / 6.0_rp) * &
         FlxV(i, j, k), vmask(i, j))
    if (south == 2 .and. j == 1) fe = fe + &
         apply_mask(0.5_rp * (t_l - t_c) * max(FlxV(i, j, k), 0.0_rp), vmask(i, j))
    if (north == 2 .and. j == ny + 1) fe = fe + &
         apply_mask(0.5_rp * (t_c - t_l) * min(FlxV(i, j, k), 0.0_rp), vmask(i, j))
  end function c4_fe

#ifdef USE_CUDA
  attributes(device) &
#endif
  function up3_fx(i, j, k, ts, itr, west, east, nx) result(fx)
    integer(ip), value :: i, j, k, ts, itr, west, east, nx
    real(rp) :: fx, d_l, d_c, d_r, curv_l, curv_r, t_l, t_c, face_t, uflux
    t_c = t(i, j, k, ts, itr)
    t_l = t(i - 1, j, k, ts, itr)
    d_c = apply_mask(t_c - t_l, umask(i, j))
    if (west >= 1 .and. i == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - t(i - 2, j, k, ts, itr), umask(i - 1, j))
    endif
    if (east >= 1 .and. i == nx + 1) then
      d_r = d_c
    else
      d_r = apply_mask(t(i + 1, j, k, ts, itr) - t_c, umask(i + 1, j))
    endif
    curv_l = (d_c - d_l) * umask(i - 1, j) * umask(i, j)
    curv_r = (d_r - d_c) * umask(i, j) * umask(i + 1, j)
    if (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp) then
      face_t = 0.5_rp * (t_l + t_c)
    elseif (rmask(i - 1, j) > 0.5_rp) then
      face_t = t_l
    elseif (rmask(i, j) > 0.5_rp) then
      face_t = t_c
    else
      face_t = 0.0_rp
    endif
    uflux = FlxU(i, j, k)
    fx = apply_mask(face_t * uflux - coef_up3 * ( &
         curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp)), &
         umask(i, j))
    if (west == 2 .and. i == 1) fx = fx + &
         apply_mask(0.5_rp * (t_l - t_c) * max(uflux, 0.0_rp), umask(i, j))
    if (east == 2 .and. i == nx + 1) fx = fx + &
         apply_mask(0.5_rp * (t_c - t_l) * min(uflux, 0.0_rp), umask(i, j))
  end function up3_fx

#ifdef USE_CUDA
  attributes(device) &
#endif
  function up3_fe(i, j, k, ts, itr, south, north, ny) result(fe)
    integer(ip), value :: i, j, k, ts, itr, south, north, ny
    real(rp) :: fe, d_l, d_c, d_r, curv_l, curv_r, t_l, t_c, face_t, uflux
    t_c = t(i, j, k, ts, itr)
    t_l = t(i, j - 1, k, ts, itr)
    d_c = apply_mask(t_c - t_l, vmask(i, j))
    if (south >= 1 .and. j == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - t(i, j - 2, k, ts, itr), vmask(i, j - 1))
    endif
    if (north >= 1 .and. j == ny + 1) then
      d_r = d_c
    else
      d_r = apply_mask(t(i, j + 1, k, ts, itr) - t_c, vmask(i, j + 1))
    endif
    curv_l = (d_c - d_l) * vmask(i, j - 1) * vmask(i, j)
    curv_r = (d_r - d_c) * vmask(i, j) * vmask(i, j + 1)
    if (rmask(i, j - 1) > 0.5_rp .and. rmask(i, j) > 0.5_rp) then
      face_t = 0.5_rp * (t_l + t_c)
    elseif (rmask(i, j - 1) > 0.5_rp) then
      face_t = t_l
    elseif (rmask(i, j) > 0.5_rp) then
      face_t = t_c
    else
      face_t = 0.0_rp
    endif
    uflux = FlxV(i, j, k)
    fe = apply_mask(face_t * uflux - coef_up3 * ( &
         curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp)), &
         vmask(i, j))
    if (south == 2 .and. j == 1) fe = fe + &
         apply_mask(0.5_rp * (t_l - t_c) * max(uflux, 0.0_rp), vmask(i, j))
    if (north == 2 .and. j == ny + 1) fe = fe + &
         apply_mask(0.5_rp * (t_c - t_l) * min(uflux, 0.0_rp), vmask(i, j))
  end function up3_fe

#ifdef USE_CUDA
  attributes(device) &
#endif
  function v_face(i, j, k, ts, itr, ctr, nz) result(fz)
    ! vert_fluxes' Fz(i,j,k); zero at the bottom (k=0) and top (k=nz).
    integer(ip), value :: i, j, k, ts, itr, ctr, nz
    real(rp) :: fz, t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2
    real(rp) :: tplus, tminus
    logical :: at_bottom, at_surface
    fz = 0.0_rp
    if (k < 1 .or. k > nz - 1) return
    at_bottom  = (k == 1)
    at_surface = (k == nz - 1)
    t_k   = t(i, j, k, ts, itr)
    t_kp1 = t(i, j, k + 1, ts, itr)
    Hz_k   = Hz(i, j, k)
    Hz_kp1 = Hz(i, j, k + 1)
    if (at_bottom) then
      t_km1  = t_k
      Hz_km1 = Hz_k
    else
      t_km1  = t(i, j, k - 1, ts, itr)
      Hz_km1 = Hz(i, j, k - 1)
    endif
    if (at_surface) then
      t_kp2  = t_kp1
      Hz_kp2 = Hz_kp1
    else
      t_kp2  = t(i, j, k + 2, ts, itr)
      Hz_kp2 = Hz(i, j, k + 2)
    endif
    tplus = thickness_tplus(t_km1, t_k, t_kp1, Hz_km1, Hz_k, Hz_kp1, at_bottom)
    tminus = thickness_tminus(t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2, &
                              at_bottom, at_surface)
    if (ctr == 1) then
      fz = 0.5_rp * (tplus + tminus) * We(i, j, k)
    else
      fz = tplus * max(We(i, j, k), 0.0_rp) + tminus * min(We(i, j, k), 0.0_rp)
    endif
    fz = apply_mask(fz, rmask(i, j))
  end function v_face

  subroutine tracer_halos(tidx, horiz, vert)
    ! The halo fills horiz_*_fluxes / vert_fluxes do, for every tracer, cut
    ! to the first step of a run (2026-10-08 audit): the masks are static,
    ! FlxU/FlxV (set_HUV, set_HUV1, step3d_uv2), Hz (set_depth over the
    ! whole array), We (omega) and t(tidx) (pre_step3d's and rho_eos(nnew)'s
    ! exchanges) are current at every call after it.
    integer(ip), intent(in) :: tidx
    logical, intent(in) :: horiz, vert
    integer(ip) :: itrc
    if (.not. is_first_run_step()) return
    if (horiz) then
      call hg_begin()
      call hg_add(F_UMASK)
      call hg_add(F_VMASK)
      call hg_add(F_FLXU)
      call hg_add(F_FLXV)
      call hg_end()
    endif
    call halo_fill_2d(rmask, STAGGER_RHO)
    if (vert) then
      call hg_begin()
      call hg_add(F_HZ)
      call hg_add(F_WE)
      call hg_end()
    endif
    do itrc = 1, nt
      call halo_fill_3d_t(t, STAGGER_RHO, tidx, itrc)
    enddo
  end subroutine tracer_halos

  subroutine pre_tracers_horiz_fused(tidx, n_stp, n_new, indx, dtau, cf_stp, cf_bak)
    ! pre_step3d horizontal C4 predictor for all tracers (see above).
    integer(ip), intent(in) :: tidx, n_stp, n_new, indx
    real(rp), intent(in) :: dtau, cf_stp, cf_bak
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, ts, ns, nn, nb, lim
    integer(ip) :: west, east, south, north
    real(rp) :: dtp, cs, cb, dtpp, hb, hzc

    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = nt
    ts = tidx; ns = n_stp; nn = n_new; nb = indx
    dtp = dtau; cs = cf_stp; cb = cf_bak
    ! edge codes: 0 interior, 1 physical edge (wall), 2 open-boundary edge,
    ! whose faces get the inflow term (open_edge_codes)
    call open_edge_codes(west, east, south, north)
    call tracer_halos(tidx, .true., .false.)
    lim = dry_lim(G_TRC)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
          dtpp = dtp * pm(i, j) * pn(i, j)
          hb = Hz_bak(i, j, k)
          hzc = Hz(i, j, k)
          do itr = 1, ntr
            t(i, j, k, nn, itr) = hb * ( &
                 cs * t(i, j, k, ns, itr) + cb * t(i, j, k, nb, itr)) - &
                 dtpp * (c4_fx(i + 1, j, k, ts, itr, west, east, nx) - &
                         c4_fx(i, j, k, ts, itr, west, east, nx) + &
                         c4_fe(i, j + 1, k, ts, itr, south, north, ny) - &
                         c4_fe(i, j, k, ts, itr, south, north, ny))
            t(i, j, k, nb, itr) = hzc * t(i, j, k, ns, itr)
          enddo
        enddo
      enddo
    enddo
  end subroutine pre_tracers_horiz_fused

  subroutine pre_tracers_vert_fused(tidx, n_new, dtau)
    ! pre_step3d vertical C4 predictor for all tracers (column kernel).
    integer(ip), intent(in) :: tidx, n_new
    real(rp), intent(in) :: dtau
    call tracer_halos(tidx, .false., .true.)
    call vert_column_fused(tidx, n_new, dtau, 1)
  end subroutine pre_tracers_vert_fused

  subroutine vert_column_fused(tidx, n_new, dtp, ctr)
    ! t(:,:,:,n_new,itr) -= dt*pm*pn*(Fz(k) - Fz(k-1)) for every tracer,
    ! one thread per column. The window t_km1..t_kp2 / Hz_km1..Hz_kp2 is
    ! shifted up the column, so each level is loaded once and each face
    ! value (v_face's arithmetic exactly) is computed once and reused as
    ! the lower face of the next level. ctr = 1: centered (predictor);
    ! 0: upstream-biased (corrector).
    ! thickness_tplus/tminus are expanded inline (as calls from this kernel
    ! they were not inlined): their slopes dlt(k-1) = dlt_m(k) and
    ! dlt(k+1) = dlt_p(k) are the same expressions on the same operands, so
    ! each is computed once (as dlt_p) and carried up the column in
    ! registers, with the bottom (dlt_m = 0, hkm = Hz_k) and surface
    ! (dlt_p = 0, hkp2 = Hz_kp1) cases as in the functions.
    integer(ip), intent(in) :: tidx, n_new, ctr
    real(rp), intent(in) :: dtp
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, ts, nn, ctr_l, lim
    real(rp) :: dtpp, fz_lo, fz_k, tplus, tminus, rm, w
    real(rp) :: t_k, t_kp1, t_kp2, Hz_k, Hz_kp1, Hz_kp2
    real(rp) :: dlt_m, dlt_c, dlt_p, hkm, hkp2, tlin

    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = nt
    ts = tidx; nn = n_new; ctr_l = ctr
    lim = dry_lim(G_TRC)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        dtpp = dtp * pm(i, j) * pn(i, j)
        rm = rmask(i, j)
        do itr = 1, ntr
          fz_lo = 0.0_rp
          t_k = t(i, j, 1, ts, itr)
          t_kp1 = t(i, j, 2, ts, itr)
          Hz_k = Hz(i, j, 1)
          Hz_kp1 = Hz(i, j, 2)
          ! k = 1 (bottom): dlt_m = 0, hkm = Hz_k
          dlt_c = (t_kp1 - t_k) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
          dlt_m = 0.0_rp
          hkm = Hz_k
          do k = 1, nz - 1
            if (k == nz - 1) then
              t_kp2 = t_kp1
              Hz_kp2 = Hz_kp1
              hkp2 = Hz_kp1
              dlt_p = 0.0_rp
            else
              t_kp2 = t(i, j, k + 2, ts, itr)
              Hz_kp2 = Hz(i, j, k + 2)
              hkp2 = Hz_kp2
              dlt_p = (t_kp2 - t_kp1) / max(Hz_kp1 + Hz_kp2, 1.0e-30_rp)
            endif
            tlin = (Hz_kp1 * t_k + Hz_k * t_kp1) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
            tplus = tlin - Hz_k * Hz_kp1 * (dlt_c - dlt_m) / max(hkm + Hz_k + Hz_kp1, 1.0e-30_rp)
            tminus = tlin - Hz_k * Hz_kp1 * (dlt_p - dlt_c) / max(Hz_k + Hz_kp1 + hkp2, 1.0e-30_rp)
            w = We(i, j, k)
            if (ctr_l == 1) then
              fz_k = 0.5_rp * (tplus + tminus) * w
            else
              fz_k = tplus * max(w, 0.0_rp) + tminus * min(w, 0.0_rp)
            endif
            fz_k = apply_mask(fz_k, rm)
            t(i, j, k, nn, itr) = t(i, j, k, nn, itr) - dtpp * (fz_k - fz_lo)
            fz_lo = fz_k
            dlt_m = dlt_c; dlt_c = dlt_p; hkm = Hz_k
            t_k = t_kp1; t_kp1 = t_kp2
            Hz_k = Hz_kp1; Hz_kp1 = Hz_kp2
          enddo
          ! Top level: the surface face is zero.
          t(i, j, nz, nn, itr) = t(i, j, nz, nn, itr) - dtpp * (0.0_rp - fz_lo)
        enddo
      enddo
    enddo
  end subroutine vert_column_fused


  subroutine cor_tracers_fused(tidx, n_new, dt_in)
    ! step3d_t horizontal UP3 (all levels) then vertical U3 (column kernel)
    ! for all tracers.
    integer(ip), intent(in) :: tidx, n_new
    real(rp), intent(in) :: dt_in
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, ts, nn, lim, jpr, njp
    integer(ip) :: west, east, south, north
    real(rp) :: dtl, dtpp, dtpp1, fe_lo, fe_mid
    logical :: two, act0, act1

    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = nt
    ts = tidx; nn = n_new; dtl = dt_in
    ! edge codes: 0 interior, 1 physical edge (wall), 2 open-boundary edge,
    ! whose faces get the inflow term (open_edge_codes)
    call open_edge_codes(west, east, south, north)
    call tracer_halos(tidx, .true., .true.)
    lim = dry_lim(G_TRC)
    njp = (ny + 1) / 2
    ! Each thread the points j and j+1 (shared y-face up3_fe(i, j+1) once).
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do k = 1, nz
      do jpr = 0, njp - 1
        do i = 1, nx
          j = 1 + 2 * jpr
          two = (j + 1 <= ny)
          act0 = .true.
          act1 = two
          if (lim > 0) then
            act0 = (colact(i, j) /= 0)
            if (two) act1 = (colact(i, j + 1) /= 0)
          endif
          if (.not. (act0 .or. act1)) cycle
          dtpp = dtl * pm(i, j) * pn(i, j)
          if (act1) dtpp1 = dtl * pm(i, j + 1) * pn(i, j + 1)
          do itr = 1, ntr
            fe_lo = up3_fe(i, j, k, ts, itr, south, north, ny)
            fe_mid = up3_fe(i, j + 1, k, ts, itr, south, north, ny)
            if (act0) then
              t(i, j, k, nn, itr) = t(i, j, k, nn, itr) - dtpp * &
                   (up3_fx(i + 1, j, k, ts, itr, west, east, nx) - &
                    up3_fx(i, j, k, ts, itr, west, east, nx) + &
                    fe_mid - &
                    fe_lo)
            endif
            if (act1) then
              t(i, j + 1, k, nn, itr) = t(i, j + 1, k, nn, itr) - dtpp1 * &
                   (up3_fx(i + 1, j + 1, k, ts, itr, west, east, nx) - &
                    up3_fx(i, j + 1, k, ts, itr, west, east, nx) + &
                    up3_fe(i, j + 2, k, ts, itr, south, north, ny) - &
                    fe_mid)
            endif
          enddo
        enddo
      enddo
    enddo
    ! Vertical U3 pass, one thread per column (reads the stored
    ! horizontally updated t, as the former fused form did via tn).
    call vert_column_fused(tidx, n_new, dt_in, 0)
  end subroutine cor_tracers_fused

end module advect

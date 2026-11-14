module pre_step3d_mod
  ! LF-AM3 predictor. Writes t,u,v(:,:,:,nnew=3) at n+1/2 and stores
  ! Hz*field(n) in slot 3-nstp for the corrector.
  !
  ! Tracer implicit Akt is host Thomas (device CUF left Hz*T). UV
  ! interiors use istrU/jstrV so open-boundary faces stay for u3dbc.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use halo_mod
  use arch_mod, only: sync_arch
  use column_solve, only: implicit_tracer_columns
  use advect, only: horiz_c4_fluxes, vert_c4_fluxes, dump_horiz_flux_probe, &
                    fuse_tracers, pre_tracers_horiz_fused, pre_tracers_vert_fused
  use uv_rhs
  use passive_mod, only: ntp, tp, passive_predictor, passive_t3dbc
  use obc, only: u3dbc, v3dbc, t3dbc
  use rivers_mod, only: river_on, rivers_ts, rivers_uv, RIV_PRE
  use imex_igw, only: imex_on, imex_solve_stage, imex_stage_pred, imex_predictor_transport
  implicit none
  private
  public :: pre_step3d
  public :: ncol_probe, icol_probe, jcol_probe

  integer(ip), parameter :: ncol_probe = nprb_pt
  integer(ip), parameter :: icol_probe(ncol_probe) = iprb_pt
  integer(ip), parameter :: jcol_probe(ncol_probe) = jprb_pt

contains

  subroutine pre_step3d()
    integer(ip) :: i, j, k, itrc, indx, iAkt, nx, ny, nz, lim, limm
    integer(ip) :: n_stp, n_new, i_u0, i_u1, j_v0, ip4, jp4
    real(rp) :: dtau, cf_stp, cf_bak, cff, FlxDiv
    real(rp) :: u_fwd(COL_KMAX)
    real(rp) :: fck, wck, cfk, fcm, wcm, cfm
    real(rp) :: dc0, am3, dcp, uk
    logical :: probe_u, fused_v, keep

    if (imex_on) then
      call pre_step3d_imex()
      return
    endif
    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_stp = nstp
    n_new = nnew
    indx = 3 - nstp
    i_u0 = 1
    if (decomp%edge_w) i_u0 = 2
    i_u1 = nx
    j_v0 = 1
    if (decomp%edge_s) j_v0 = 2
    call lf_am3_weights(is_first_time_step(), dt, dtau, cf_stp, cf_bak)
    call dump_pre_col('pre in', 1, indx)
    if (nz > COL_KMAX) error stop 'pre_step3d: nz exceeds COL_KMAX'

    ! Hz, u/v(nrhs), FlxU/FlxV, We/Wi are current here (exchanged after
    ! their last write): no exchange (2026-10-08 audit).
    call set_bottom_drag()
    lim = dry_lim(G_TRC)
    limm = dry_lim(G_MOM)

#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do k = 1, nz
      do j = 0, ny
        do i = 0, nx
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
          FlxDiv = 0.5_rp * dtau * pm(i, j) * pn(i, j) * ( &
                   FlxU(i + 1, j, k) - FlxU(i, j, k) + &
                   FlxV(i, j + 1, k) - FlxV(i, j, k) + &
                   We(i, j, k) + Wi(i, j, k) - We(i, j, k - 1) - Wi(i, j, k - 1))
          Hz_bak(i, j, k) = Hz(i, j, k) + FlxDiv
          Hz_fwd(i, j, k) = Hz(i, j, k) - FlxDiv
        enddo
      enddo
    enddo

    if (fuse_tracers(nrhs, n_new, indx)) then
      call pre_tracers_horiz_fused(nrhs, n_stp, n_new, indx, dtau, cf_stp, cf_bak)
    else
    do itrc = 1, nt
      call horiz_c4_fluxes(nrhs, itrc)
      if (itrc == itemp) call dump_horiz_flux_probe('pre_step3d', dtau)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          do k = 1, nz
            t(i, j, k, n_new, itrc) = Hz_bak(i, j, k) * ( &
                 cf_stp * t(i, j, k, n_stp, itrc) + cf_bak * t(i, j, k, indx, itrc)) - &
                 dtau * pm(i, j) * pn(i, j) * (wrk3a(i + 1, j, k) - wrk3a(i, j, k) + &
                                               wrk3b(i, j + 1, k) - wrk3b(i, j, k))
            t(i, j, k, indx, itrc) = Hz(i, j, k) * t(i, j, k, n_stp, itrc)
          enddo
        enddo
      enddo
    enddo
    endif
    ! River faces: the horizontal pass masked their flux; add the river
    ! flux (all tracers) before the vertical pass.
    if (river_on) call rivers_ts(RIV_PRE, n_new, dtau, 1_ip, nt)
    call dump_pre_col('pre horiz', 2, indx)

    call dump_rv_pred('after prsgrd')
    call dump_vn3('pre in')
    call add_horiz_uv_rhs_all(.false.)
    call dump_rv_pred('after pred horiz')
    call dump_vn3('after pred horiz')
    call dump_pre_u('horiz', indx, dtau, cf_stp, cf_bak, .false.)
    call add_vert_uv_rhs(1, ny)
    call dump_vvs('pred vert')
    call dump_vvf('pred vert')
    call dump_vn3('after pred vert')
    call dump_pre_u('vert', indx, dtau, cf_stp, cf_bak, .false.)

    ! All tracers' vertical update in one launch, then each tracer's solve.
    fused_v = fuse_tracers(nrhs, n_new, n_new)
    if (fused_v) call pre_tracers_vert_fused(nrhs, n_new, dtau)
    do itrc = 1, nt
      if (.not. fused_v) then
      call vert_c4_fluxes(nrhs, itrc)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          do k = 1, nz
            t(i, j, k, n_new, itrc) = t(i, j, k, n_new, itrc) - dtau * pm(i, j) * pn(i, j) * &
                                     (col_wc(i, j, k) - col_wc(i, j, k - 1))
          enddo
        enddo
      enddo
      endif
      if (itrc == itemp) call dump_pre_col('pre vadv', 3, indx)

      iAkt = min(itrc, 2)
      call implicit_tracer_columns(nx, ny, nz, n_new, itrc, iAkt, dtau, .true.)
      if (itrc == itemp) call dump_pre_col('pre impl', 4, indx)
      ! no exchange here: the group after t3dbc below exchanges t(nnew) and
      ! nothing reads its halo before that (t3dbc reads slot nstp)
    enddo
    if (ntp > 0) call passive_predictor(dtau, cf_stp, cf_bak, n_stp, n_new, indx)

    probe_u = .false.
    if (.not. probe_on) then
    ! AM3 + ru (the former col_dc(1:nz), col_dc(0)) computed level by level
    ! inside the Thomas sweep that consumes it (same expressions; u(indx)
    ! is read before it is overwritten at each level), the back-substituted
    ! velocity of the previous level kept in a register. Probe steps keep the
    ! two-kernel form below for the 'am3' / 'pred AM3' dumps.
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j))
        am3 = 0.5_rp * (Hz_bak(i, j, nz) + Hz_bak(i - 1, j, nz)) * ( &
              cf_stp * u(i, j, nz, n_stp) + cf_bak * u(i, j, nz, indx)) + dc0 * ru(i, j, nz)
        u(i, j, nz, indx) = 0.5_rp * (Hz(i, j, nz) + Hz(i - 1, j, nz)) * u(i, j, nz, n_stp)
        fck = 2.0_rp * dtau * (Akv(i, j, nz - 1) + Akv(i - 1, j, nz - 1)) / &
             (Hz_fwd(i, j, nz) + Hz_fwd(i - 1, j, nz) + Hz_fwd(i, j, nz - 1) + Hz_fwd(i - 1, j, nz - 1))
        wck = dc0 * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i - 1, j, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, nz) + Hz_fwd(i - 1, j, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        dcp = cff * (am3 + dtau * sustr(i, j))
        col_dc(i, j, nz) = dcp
        do k = nz - 1, 2, -1
          am3 = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i - 1, j, k)) * ( &
                cf_stp * u(i, j, k, n_stp) + cf_bak * u(i, j, k, indx)) + dc0 * ru(i, j, k)
          u(i, j, k, indx) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * u(i, j, k, n_stp)
          fcm = 2.0_rp * dtau * (Akv(i, j, k - 1) + Akv(i - 1, j, k - 1)) / &
               (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k) + Hz_fwd(i, j, k - 1) + Hz_fwd(i - 1, j, k - 1))
          wcm = dc0 * 0.5_rp * (Wi(i, j, k - 1) + Wi(i - 1, j, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          dcp = cff * (am3 + dcp * (fck - min(wck, 0.0_rp)))
          col_dc(i, j, k) = dcp
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        am3 = 0.5_rp * (Hz_bak(i, j, 1) + Hz_bak(i - 1, j, 1)) * ( &
              cf_stp * u(i, j, 1, n_stp) + cf_bak * u(i, j, 1, indx)) + dc0 * ru(i, j, 1)
        u(i, j, 1, indx) = 0.5_rp * (Hz(i, j, 1) + Hz(i - 1, j, 1)) * u(i, j, 1, n_stp)
        uk = (am3 + dcp * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz_fwd(i, j, 1) + Hz_fwd(i - 1, j, 1)) + 0.5_rp * dtau * (r_D(i, j) + r_D(i - 1, j)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        u(i, j, 1, n_new) = uk
        do k = 2, nz
          uk = col_dc(i, j, k) + col_cf(i, j, k - 1) * uk
          u(i, j, k, n_new) = uk
        enddo
      enddo
    enddo
    else
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        col_dc(i, j, 0) = dtau * 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j))
        do k = 1, nz
          col_dc(i, j, k) = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i - 1, j, k)) * ( &
                     cf_stp * u(i, j, k, n_stp) + cf_bak * u(i, j, k, indx)) + col_dc(i, j, 0) * ru(i, j, k)
          u(i, j, k, indx) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * u(i, j, k, n_stp)
        enddo
      enddo
    enddo
    ! Column Thomas: one device thread per U-face column (the k-recurrence
    ! stays serial in the thread), after the explicit Hz*AM3 + ru kernel.
    if (probe_on) call sync_arch()
    call dump_pre_u('am3', indx, dtau, cf_stp, cf_bak, .true.)
    call dump_uvisc('in')
    ip4 = iprb_pt(4); jp4 = jprb_pt(4)
    keep = probe_on   ! probes read col_fc/col_wc; otherwise they stay in registers
    probe_u = probe_on .and. ip4 >= i_u0 .and. ip4 <= i_u1 .and. jp4 >= 1 .and. jp4 <= ny
    if (probe_u) then
      ! The 'fwd' probe prints u_new before back-substitution: keep it.
      call sync_arch()
      u_fwd(1:nz) = u(ip4, jp4, 1:nz, n_new)
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        fck = 2.0_rp * dtau * (Akv(i, j, nz - 1) + Akv(i - 1, j, nz - 1)) / &
             (Hz_fwd(i, j, nz) + Hz_fwd(i - 1, j, nz) + Hz_fwd(i, j, nz - 1) + Hz_fwd(i - 1, j, nz - 1))
        wck = col_dc(i, j, 0) * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i - 1, j, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, nz) + Hz_fwd(i - 1, j, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        col_dc(i, j, nz) = cff * (col_dc(i, j, nz) + dtau * sustr(i, j))
        if (keep) then
          col_fc(i, j, nz - 1) = fck
          col_wc(i, j, nz - 1) = wck
        endif
        do k = nz - 1, 2, -1
          fcm = 2.0_rp * dtau * (Akv(i, j, k - 1) + Akv(i - 1, j, k - 1)) / &
               (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k) + Hz_fwd(i, j, k - 1) + Hz_fwd(i - 1, j, k - 1))
          wcm = col_dc(i, j, 0) * 0.5_rp * (Wi(i, j, k - 1) + Wi(i - 1, j, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          col_dc(i, j, k) = cff * (col_dc(i, j, k) + col_dc(i, j, k + 1) * (fck - min(wck, 0.0_rp)))
          if (keep) then
            col_fc(i, j, k - 1) = fcm
            col_wc(i, j, k - 1) = wcm
          endif
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        ! CPU pre_step3d writes AM3 u_new unmasked; umask is applied in set_HUV1.
        u(i, j, 1, n_new) = (col_dc(i, j, 1) + col_dc(i, j, 2) * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz_fwd(i, j, 1) + Hz_fwd(i - 1, j, 1)) + 0.5_rp * dtau * (r_D(i, j) + r_D(i - 1, j)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        do k = 2, nz
          u(i, j, k, n_new) = col_dc(i, j, k) + col_cf(i, j, k - 1) * u(i, j, k - 1, n_new)
        enddo
      enddo
    enddo
    endif
    if (probe_u) then
      call sync_arch()
      call dump_uvisc_at('fwd', ip4, jp4, u_fwd)
      call dump_uvisc_at('back', ip4, jp4)
    endif
    call dump_pre_u('impl', indx, dtau, cf_stp, cf_bak, .true.)

    call dump_vim('pred visc in')
    if (.not. probe_on) then
    ! AM3 + ru (the former col_dc(1:nz), col_dc(0)) computed level by level
    ! inside the Thomas sweep that consumes it (same expressions; u(indx)
    ! is read before it is overwritten at each level), the back-substituted
    ! velocity of the previous level kept in a register. Probe steps keep the
    ! two-kernel form below for the 'am3' / 'pred AM3' dumps.
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * 0.25_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1))
        am3 = 0.5_rp * (Hz_bak(i, j, nz) + Hz_bak(i, j - 1, nz)) * ( &
              cf_stp * v(i, j, nz, n_stp) + cf_bak * v(i, j, nz, indx)) + dc0 * rv(i, j, nz)
        v(i, j, nz, indx) = 0.5_rp * (Hz(i, j, nz) + Hz(i, j - 1, nz)) * v(i, j, nz, n_stp)
        fck = 2.0_rp * dtau * (Akv(i, j, nz - 1) + Akv(i, j - 1, nz - 1)) / &
             (Hz_fwd(i, j, nz) + Hz_fwd(i, j - 1, nz) + Hz_fwd(i, j, nz - 1) + Hz_fwd(i, j - 1, nz - 1))
        wck = dc0 * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i, j - 1, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, nz) + Hz_fwd(i, j - 1, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        dcp = cff * (am3 + dtau * svstr(i, j))
        col_dc(i, j, nz) = dcp
        do k = nz - 1, 2, -1
          am3 = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i, j - 1, k)) * ( &
                cf_stp * v(i, j, k, n_stp) + cf_bak * v(i, j, k, indx)) + dc0 * rv(i, j, k)
          v(i, j, k, indx) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * v(i, j, k, n_stp)
          fcm = 2.0_rp * dtau * (Akv(i, j, k - 1) + Akv(i, j - 1, k - 1)) / &
               (Hz_fwd(i, j, k) + Hz_fwd(i, j - 1, k) + Hz_fwd(i, j, k - 1) + Hz_fwd(i, j - 1, k - 1))
          wcm = dc0 * 0.5_rp * (Wi(i, j, k - 1) + Wi(i, j - 1, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i, j - 1, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          dcp = cff * (am3 + dcp * (fck - min(wck, 0.0_rp)))
          col_dc(i, j, k) = dcp
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        am3 = 0.5_rp * (Hz_bak(i, j, 1) + Hz_bak(i, j - 1, 1)) * ( &
              cf_stp * v(i, j, 1, n_stp) + cf_bak * v(i, j, 1, indx)) + dc0 * rv(i, j, 1)
        v(i, j, 1, indx) = 0.5_rp * (Hz(i, j, 1) + Hz(i, j - 1, 1)) * v(i, j, 1, n_stp)
        uk = (am3 + dcp * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz_fwd(i, j, 1) + Hz_fwd(i, j - 1, 1)) + 0.5_rp * dtau * (r_D(i, j) + r_D(i, j - 1)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        v(i, j, 1, n_new) = uk
        do k = 2, nz
          uk = col_dc(i, j, k) + col_cf(i, j, k - 1) * uk
          v(i, j, k, n_new) = uk
        enddo
      enddo
    enddo
    else
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        col_dc(i, j, 0) = dtau * 0.25_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1))
        do k = 1, nz
          col_dc(i, j, k) = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i, j - 1, k)) * ( &
                     cf_stp * v(i, j, k, n_stp) + cf_bak * v(i, j, k, indx)) + col_dc(i, j, 0) * rv(i, j, k)
          v(i, j, k, indx) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * v(i, j, k, n_stp)
        enddo
      enddo
    enddo
    call dump_vdc('pred AM3')
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        fck = 2.0_rp * dtau * (Akv(i, j, nz - 1) + Akv(i, j - 1, nz - 1)) / &
             (Hz_fwd(i, j, nz) + Hz_fwd(i, j - 1, nz) + Hz_fwd(i, j, nz - 1) + Hz_fwd(i, j - 1, nz - 1))
        wck = col_dc(i, j, 0) * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i, j - 1, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, nz) + Hz_fwd(i, j - 1, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        col_dc(i, j, nz) = cff * (col_dc(i, j, nz) + dtau * svstr(i, j))
        if (keep) then
          col_fc(i, j, nz - 1) = fck
          col_wc(i, j, nz - 1) = wck
        endif
        do k = nz - 1, 2, -1
          fcm = 2.0_rp * dtau * (Akv(i, j, k - 1) + Akv(i, j - 1, k - 1)) / &
               (Hz_fwd(i, j, k) + Hz_fwd(i, j - 1, k) + Hz_fwd(i, j, k - 1) + Hz_fwd(i, j - 1, k - 1))
          wcm = col_dc(i, j, 0) * 0.5_rp * (Wi(i, j, k - 1) + Wi(i, j - 1, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i, j - 1, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          col_dc(i, j, k) = cff * (col_dc(i, j, k) + col_dc(i, j, k + 1) * (fck - min(wck, 0.0_rp)))
          if (keep) then
            col_fc(i, j, k - 1) = fcm
            col_wc(i, j, k - 1) = wcm
          endif
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        v(i, j, 1, n_new) = (col_dc(i, j, 1) + col_dc(i, j, 2) * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz_fwd(i, j, 1) + Hz_fwd(i, j - 1, 1)) + 0.5_rp * dtau * (r_D(i, j) + r_D(i, j - 1)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        do k = 2, nz
          v(i, j, k, n_new) = col_dc(i, j, k) + col_cf(i, j, k - 1) * v(i, j, k - 1, n_new)
        enddo
      enddo
    enddo

    endif
    call dump_vn3('after pred impl')
    if (river_on) call rivers_uv(RIV_PRE)    ! CPU pre_step3d: river faces, then u3dbc/v3dbc
    call u3dbc()
    call v3dbc()
    call dump_vn3('after v3dbc')
    call dump_pre_u('obc', indx, dtau, cf_stp, cf_bak, .false.)
    call t3dbc(itemp)
    call t3dbc(isalt)
    if (ntp > 0) call passive_t3dbc()
    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_add(F_T, nnew, itemp)
    call hg_add(F_T, nnew, isalt)
    call hg_end()
    ! slot 3 of the passives: post the exchange; advanced after visc3d and
    ! completed by passive_corrector (nothing touches the slot in between)
    if (ntp > 0) then
      if (tp_overlap_on(1)) then
        call halo_tp_start(tp, ntp, nnew)
      else
        call halo_fill_3d_tp(tp, ntp, nnew)
      endif
    endif
    call dump_pre_step_t()
  end subroutine pre_step3d

  subroutine pre_step3d_imex()
    ! IMEX_IGW (imex_igw.F90): the CPU's two passes of pre_step3d_tile.
    ! Pass 1, the momentum predictor from the level-n transport (Hz_bak,
    ! Hz_fwd from FlxU, FlxV, We, Wi of set_HUV/omega), then the elliptic
    ! correction of u*, v* (imex_solve_stage), rivers and the open-boundary
    ! values, and the predictor tracer transport Flx_pred into FlxU, FlxV,
    ! We, Wi (imex_predictor_transport); pass 2, the tracer predictor with
    ! Hz_bak, Hz_fwd rebuilt from Flx_pred. The kernels are the stock ones
    ! (non-probe forms) in the new order.
    integer(ip) :: i, j, k, itrc, indx, iAkt, nx, ny, nz, lim, limm
    integer(ip) :: n_stp, n_new, i_u0, i_u1, j_v0
    real(rp) :: dtau, cf_stp, cf_bak, cff
    real(rp) :: fck, wck, cfk, fcm, wcm, cfm
    real(rp) :: dc0, am3, dcp, uk
    logical :: fused_v

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_stp = nstp
    n_new = nnew
    indx = 3 - nstp
    i_u0 = 1
    if (decomp%edge_w) i_u0 = 2
    i_u1 = nx
    j_v0 = 1
    if (decomp%edge_s) j_v0 = 2
    call lf_am3_weights(is_first_time_step(), dt, dtau, cf_stp, cf_bak)
    if (nz > COL_KMAX) error stop 'pre_step3d: nz exceeds COL_KMAX'
    call set_bottom_drag()
    lim = dry_lim(G_TRC)
    limm = dry_lim(G_MOM)

    ! ---- pass 1: momentum predictor
    call hz_bak_fwd(dtau, lim)
    call add_horiz_uv_rhs_all(.false.)
    call add_vert_uv_rhs(1, ny)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j))
        am3 = 0.5_rp * (Hz_bak(i, j, nz) + Hz_bak(i - 1, j, nz)) * ( &
              cf_stp * u(i, j, nz, n_stp) + cf_bak * u(i, j, nz, indx)) + dc0 * ru(i, j, nz)
        u(i, j, nz, indx) = 0.5_rp * (Hz(i, j, nz) + Hz(i - 1, j, nz)) * u(i, j, nz, n_stp)
        fck = 2.0_rp * dtau * (Akv(i, j, nz - 1) + Akv(i - 1, j, nz - 1)) / &
             (Hz_fwd(i, j, nz) + Hz_fwd(i - 1, j, nz) + Hz_fwd(i, j, nz - 1) + Hz_fwd(i - 1, j, nz - 1))
        wck = dc0 * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i - 1, j, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, nz) + Hz_fwd(i - 1, j, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        dcp = cff * (am3 + dtau * sustr(i, j))
        col_dc(i, j, nz) = dcp
        do k = nz - 1, 2, -1
          am3 = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i - 1, j, k)) * ( &
                cf_stp * u(i, j, k, n_stp) + cf_bak * u(i, j, k, indx)) + dc0 * ru(i, j, k)
          u(i, j, k, indx) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * u(i, j, k, n_stp)
          fcm = 2.0_rp * dtau * (Akv(i, j, k - 1) + Akv(i - 1, j, k - 1)) / &
               (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k) + Hz_fwd(i, j, k - 1) + Hz_fwd(i - 1, j, k - 1))
          wcm = dc0 * 0.5_rp * (Wi(i, j, k - 1) + Wi(i - 1, j, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          dcp = cff * (am3 + dcp * (fck - min(wck, 0.0_rp)))
          col_dc(i, j, k) = dcp
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        am3 = 0.5_rp * (Hz_bak(i, j, 1) + Hz_bak(i - 1, j, 1)) * ( &
              cf_stp * u(i, j, 1, n_stp) + cf_bak * u(i, j, 1, indx)) + dc0 * ru(i, j, 1)
        u(i, j, 1, indx) = 0.5_rp * (Hz(i, j, 1) + Hz(i - 1, j, 1)) * u(i, j, 1, n_stp)
        uk = (am3 + dcp * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz_fwd(i, j, 1) + Hz_fwd(i - 1, j, 1)) + 0.5_rp * dtau * (r_D(i, j) + r_D(i - 1, j)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        u(i, j, 1, n_new) = uk
        do k = 2, nz
          uk = col_dc(i, j, k) + col_cf(i, j, k - 1) * uk
          u(i, j, k, n_new) = uk
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (limm > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * 0.25_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1))
        am3 = 0.5_rp * (Hz_bak(i, j, nz) + Hz_bak(i, j - 1, nz)) * ( &
              cf_stp * v(i, j, nz, n_stp) + cf_bak * v(i, j, nz, indx)) + dc0 * rv(i, j, nz)
        v(i, j, nz, indx) = 0.5_rp * (Hz(i, j, nz) + Hz(i, j - 1, nz)) * v(i, j, nz, n_stp)
        fck = 2.0_rp * dtau * (Akv(i, j, nz - 1) + Akv(i, j - 1, nz - 1)) / &
             (Hz_fwd(i, j, nz) + Hz_fwd(i, j - 1, nz) + Hz_fwd(i, j, nz - 1) + Hz_fwd(i, j - 1, nz - 1))
        wck = dc0 * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i, j - 1, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, nz) + Hz_fwd(i, j - 1, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        dcp = cff * (am3 + dtau * svstr(i, j))
        col_dc(i, j, nz) = dcp
        do k = nz - 1, 2, -1
          am3 = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i, j - 1, k)) * ( &
                cf_stp * v(i, j, k, n_stp) + cf_bak * v(i, j, k, indx)) + dc0 * rv(i, j, k)
          v(i, j, k, indx) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * v(i, j, k, n_stp)
          fcm = 2.0_rp * dtau * (Akv(i, j, k - 1) + Akv(i, j - 1, k - 1)) / &
               (Hz_fwd(i, j, k) + Hz_fwd(i, j - 1, k) + Hz_fwd(i, j, k - 1) + Hz_fwd(i, j - 1, k - 1))
          wcm = dc0 * 0.5_rp * (Wi(i, j, k - 1) + Wi(i, j - 1, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i, j - 1, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          dcp = cff * (am3 + dcp * (fck - min(wck, 0.0_rp)))
          col_dc(i, j, k) = dcp
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        am3 = 0.5_rp * (Hz_bak(i, j, 1) + Hz_bak(i, j - 1, 1)) * ( &
              cf_stp * v(i, j, 1, n_stp) + cf_bak * v(i, j, 1, indx)) + dc0 * rv(i, j, 1)
        v(i, j, 1, indx) = 0.5_rp * (Hz(i, j, 1) + Hz(i, j - 1, 1)) * v(i, j, 1, n_stp)
        uk = (am3 + dcp * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz_fwd(i, j, 1) + Hz_fwd(i, j - 1, 1)) + 0.5_rp * dtau * (r_D(i, j) + r_D(i, j - 1)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        v(i, j, 1, n_new) = uk
        do k = 2, nz
          uk = col_dc(i, j, k) + col_cf(i, j, k - 1) * uk
          v(i, j, k, n_new) = uk
        enddo
      enddo
    enddo
    call imex_solve_stage(imex_stage_pred, dtau, cf_bak)   ! elliptic correction of u*, v*
    if (river_on) call rivers_uv(RIV_PRE)
    call u3dbc()
    call v3dbc()
    call imex_predictor_transport()                        ! Flx_pred -> FlxU, FlxV; We, Wi

    ! ---- pass 2: tracer predictor with Flx_pred
    call hz_bak_fwd(dtau, lim)
    if (fuse_tracers(nrhs, n_new, indx)) then
      call pre_tracers_horiz_fused(nrhs, n_stp, n_new, indx, dtau, cf_stp, cf_bak)
    else
    do itrc = 1, nt
      call horiz_c4_fluxes(nrhs, itrc)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          do k = 1, nz
            t(i, j, k, n_new, itrc) = Hz_bak(i, j, k) * ( &
                 cf_stp * t(i, j, k, n_stp, itrc) + cf_bak * t(i, j, k, indx, itrc)) - &
                 dtau * pm(i, j) * pn(i, j) * (wrk3a(i + 1, j, k) - wrk3a(i, j, k) + &
                                               wrk3b(i, j + 1, k) - wrk3b(i, j, k))
            t(i, j, k, indx, itrc) = Hz(i, j, k) * t(i, j, k, n_stp, itrc)
          enddo
        enddo
      enddo
    enddo
    endif
    if (river_on) call rivers_ts(RIV_PRE, n_new, dtau, 1_ip, nt)
    fused_v = fuse_tracers(nrhs, n_new, n_new)
    if (fused_v) call pre_tracers_vert_fused(nrhs, n_new, dtau)
    do itrc = 1, nt
      if (.not. fused_v) then
      call vert_c4_fluxes(nrhs, itrc)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          do k = 1, nz
            t(i, j, k, n_new, itrc) = t(i, j, k, n_new, itrc) - dtau * pm(i, j) * pn(i, j) * &
                                     (col_wc(i, j, k) - col_wc(i, j, k - 1))
          enddo
        enddo
      enddo
      endif
      iAkt = min(itrc, 2)
      call implicit_tracer_columns(nx, ny, nz, n_new, itrc, iAkt, dtau, .true.)
    enddo
    if (ntp > 0) call passive_predictor(dtau, cf_stp, cf_bak, n_stp, n_new, indx)
    call t3dbc(itemp)
    call t3dbc(isalt)
    if (ntp > 0) call passive_t3dbc()
    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_add(F_T, nnew, itemp)
    call hg_add(F_T, nnew, isalt)
    call hg_end()
    if (ntp > 0) then
      if (tp_overlap_on(1)) then
        call halo_tp_start(tp, ntp, nnew)
      else
        call halo_fill_3d_tp(tp, ntp, nnew)
      endif
    endif
  end subroutine pre_step3d_imex

  subroutine hz_bak_fwd(dtau, lim)
    ! Hz_bak, Hz_fwd = Hz +- dtau/2 pm pn div(FlxU, FlxV, We + Wi) (the
    ! stock pre_step3d kernel)
    real(rp), intent(in) :: dtau
    integer(ip), intent(in) :: lim
    integer(ip) :: i, j, k, nx, ny, nz, lm
    real(rp) :: FlxDiv, dt_l
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    lm = lim; dt_l = dtau
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do k = 1, nz
      do j = 0, ny
        do i = 0, nx
          if (lm > 0) then; if (colact(i, j) == 0) cycle; end if
          FlxDiv = 0.5_rp * dt_l * pm(i, j) * pn(i, j) * ( &
                   FlxU(i + 1, j, k) - FlxU(i, j, k) + &
                   FlxV(i, j + 1, k) - FlxV(i, j, k) + &
                   We(i, j, k) + Wi(i, j, k) - We(i, j, k - 1) - Wi(i, j, k - 1))
          Hz_bak(i, j, k) = Hz(i, j, k) + FlxDiv
          Hz_fwd(i, j, k) = Hz(i, j, k) - FlxDiv
        enddo
      enddo
    enddo
  end subroutine hz_bak_fwd

  subroutine set_bottom_drag()
    ! CPU pre_step3d "rd" -> r_D at rho points from bottom-level u,v(nstp):
    ! finite-volume log layer if zob > 0, else rdrg capped by 0.8*Hz/dt.
    ! step3d_uv1 needs r_D(i-1,j) and r_D(i,j-1), so cover 0:nx, 0:ny.
    real(rp), parameter :: vonkar = 0.41_rp
    integer(ip) :: i, j, nx, ny, n_stp
    real(rp) :: cff, zob_loc, rdrg_loc, dt_loc

    nx = decomp%nx
    ny = decomp%ny
    n_stp = nstp
    zob_loc = zob
    rdrg_loc = rdrg
    dt_loc = dt
    ! u, v(nstp) are current (step3d_uv2's exchange): no exchange.

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny
      do i = 0, nx
        if (Hz(i, j, 1) <= 0.0_rp) then
          r_D(i, j) = 0.0_rp
        elseif (zob_loc > 0.0_rp) then
          cff = sqrt(0.333333333333_rp * ( &
                u(i, j, 1, n_stp)**2 + u(i + 1, j, 1, n_stp)**2 + &
                u(i, j, 1, n_stp) * u(i + 1, j, 1, n_stp) + &
                v(i, j, 1, n_stp)**2 + v(i, j + 1, 1, n_stp)**2 + &
                v(i, j, 1, n_stp) * v(i, j + 1, 1, n_stp)))
          r_D(i, j) = cff * (vonkar / log(1.0_rp + 0.5_rp * Hz(i, j, 1) / zob_loc))**2
        else
          r_D(i, j) = min(rdrg_loc, 0.8_rp * Hz(i, j, 1) / dt_loc)
        endif
      enddo
    enddo
  end subroutine set_bottom_drag

  subroutine dump_pre_step_t()
    integer(ip) :: i, j, k, im, jm, km
    real(rp) :: tmax, hmin
    if (.not. probe_on) return
    call sync_arch()
    tmax = 0.0_rp
    hmin = 1.0e30_rp
    im = 0; jm = 0; km = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (Hz_fwd(i, j, k) < hmin) hmin = Hz_fwd(i, j, k)
          if (abs(t(i, j, k, nnew, 1)) > tmax) then
            tmax = abs(t(i, j, k, nnew, 1))
            im = i; jm = j; km = k
          endif
        enddo
      enddo
    enddo
    write(*,'(a,es14.6,a,3i6)') 'pre_step3d: max |t| = ', tmax, ' at i,j,k =', im, jm, km
    if (im > 0) then
      write(*,'(a,3es14.6)') 'pre_step3d: Hz, Hz_fwd, Hz_bak = ', &
           Hz(im, jm, km), Hz_fwd(im, jm, km), Hz_bak(im, jm, km)
      write(*,'(a,4es14.6)') 'pre_step3d: T, h, zeta, D = ', t(im, jm, km, nnew, 1), &
           h(im, jm), zeta(im, jm, knew), h(im, jm) + zeta(im, jm, knew)
    endif
    write(*,'(a,es14.6)') 'pre_step3d: min wet Hz_fwd = ', hmin
    flush(6)
  end subroutine dump_pre_step_t

  subroutine dump_pre_col(tag, stage, indx)
    ! Same "diag: pcol" lines as roms_bugfix smoke_pcol.
    ! stage 1: T(nstp), T(indx), Hz, Akt     2: t(nnew), Hz_bak, Hz_fwd, t(indx)
    !       3: t(nnew), col_wc, We, Wi           4: t(nnew), Hz_fwd, Akt, Hz
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: stage, indx
    integer(ip) :: n, i, j, k
    real(rp) :: a(4)
    if (.not. probe_on) return
    call sync_arch()
    do n = 1, ncol_probe
      i = icol_probe(n)
      j = jcol_probe(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      do k = 1, decomp%nz
        select case (stage)
        case (1)
          a = (/ t(i, j, k, nstp, itemp), t(i, j, k, indx, itemp), Hz(i, j, k), Akt(i, j, k, itemp) /)
        case (2)
          a = (/ t(i, j, k, nnew, itemp), Hz_bak(i, j, k), Hz_fwd(i, j, k), t(i, j, k, indx, itemp) /)
        case (3)
          a = (/ t(i, j, k, nnew, itemp), col_wc(i, j, k), We(i, j, k), Wi(i, j, k) /)
        case default
          a = (/ t(i, j, k, nnew, itemp), Hz_fwd(i, j, k), Akt(i, j, k, itemp), Hz(i, j, k) /)
        end select
        write(*,'(a,a,a,i1,a,i6,i4,4es22.14)') 'diag: pcol ', trim(tag), ' P', n, &
             ' iic,k,a1..a4=', iic, k, a
      enddo
    enddo
    flush(6)
  end subroutine dump_pre_col

  subroutine dump_pre_u(tag, indx, dtau, cf_stp, cf_bak, with_dc)
    ! LF-AM3 u interpolant at P4 (456,56) into slot nnew=3.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: indx
    real(rp), intent(in) :: dtau, cf_stp, cf_bak
    logical, intent(in) :: with_dc
    integer(ip) :: i, j, k, kw, nz
    real(rp) :: hzu, hzb, hzf, dc0, dck
    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    nz = decomp%nz
    dc0 = dtau * 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j))
    do k = 1, nz
      kw = max(k - 1, 0)
      hzu = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
      hzb = 0.5_rp * (Hz_bak(i, j, k) + Hz_bak(i - 1, j, k))
      hzf = 0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k))
      dck = 0.0_rp
      if (with_dc) then
        dc0 = col_dc(i, j, 0)
        dck = col_dc(i, j, k)
      endif
      write(*,'(a,a,a,i8,a,i4,a,20es26.17)') 'diag: pre u P4  ', trim(tag), &
           '  iic=', iic, '  k=', k, &
           '  nstp,nnew,indx,cf_stp,cf_bak,dtau,u_stp,u_bak,u_new,ru,HzU,HzbU,HzfU,DC0,DC,AkvW,AkvE,umask,rDW,rDE=', &
           real(nstp, rp), real(nnew, rp), real(indx, rp), cf_stp, cf_bak, dtau, &
           u(i, j, k, nstp), u(i, j, k, indx), u(i, j, k, nnew), ru(i, j, k), &
           hzu, hzb, hzf, dc0, dck, Akv(i - 1, j, kw), Akv(i, j, kw), umask(i, j), &
           r_D(i - 1, j), r_D(i, j)
    enddo
    flush(6)
  end subroutine dump_pre_u

  subroutine dump_uvisc(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j
    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call dump_uvisc_at(tag, i, j)
  end subroutine dump_uvisc

  subroutine dump_uvisc_at(tag, i, j, u_col)
    ! U-face Thomas intermediates at P4 during pre_step3d impl viscos.
    ! u_col, if given, replaces u(i,j,:,nnew) in the print (pre-solve u).
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: i, j
    real(rp), intent(in), optional :: u_col(:)
    integer(ip) :: k, km, kw, nz, n_new
    real(rp) :: fc, wc, cf, hzf, uk
    if (.not. probe_on) return
    nz = decomp%nz
    n_new = nnew
    do k = 1, nz
      km = max(k - 1, 1)
      kw = max(k - 1, 0)
      if (tag == 'in') then
        fc = 0.0_rp
        wc = 0.0_rp
        cf = 0.0_rp
      else
        fc = col_fc(i, j, km)
        wc = col_wc(i, j, km)
        cf = col_cf(i, j, km)
      endif
      hzf = 0.5_rp * (Hz_fwd(i, j, k) + Hz_fwd(i - 1, j, k))
      if (present(u_col)) then
        uk = u_col(k)
      else
        uk = u(i, j, k, n_new)
      endif
      write(*,'(a,a,a,i8,a,i4,a,15es26.17)') 'diag: uvisc P4  ', trim(tag), &
           '  iic=', iic, '  k=', k, &
           '  FC,WC,CF,DC,HzfU,AkvW,AkvE,WiW,WiE,u_new,rDW,rDE,sustr,DC0=', &
           fc, wc, cf, col_dc(i, j, k), hzf, &
           Akv(i - 1, j, kw), Akv(i, j, kw), Wi(i - 1, j, kw), Wi(i, j, kw), &
           uk, r_D(i - 1, j), r_D(i, j), sustr(i, j), col_dc(i, j, 0)
    enddo
    flush(6)
  end subroutine dump_uvisc_at

end module pre_step3d_mod

module step3d_t_mod
  ! Tracer corrector: UP3 + thickness-aware U3, then implicit Akt/Wi.
  ! On entry t(:,:,:,nnew) is Hz(n)*t(n) stored by pre_step3d.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use halo_mod
  use arch_mod, only: sync_arch
  use column_solve, only: implicit_tracer_columns, implicit_tracer_columns_frc
  use advect, only: horiz_up3_fluxes, vert_u3_fluxes, dump_horiz_flux_probe, &
                    fuse_tracers, cor_tracers_fused
  use obc, only: t3dbc
  use pre_step3d_mod, only: ncol_probe, icol_probe, jcol_probe
  use passive_mod, only: ntp, passive_corrector, tp_subcycled
  use rivers_mod, only: river_on, rivers_ts, RIV_COR
  use cdr_src_mod, only: cdr_vol_on, cdr_apply_ts
  implicit none
  private
  public :: step3d_t

contains

  subroutine step3d_t()
    integer(ip) :: i, j, k, itrc, iAkt, nx, ny, nz, n_new, lim
    real(rp) :: cff, dt_loc
    logical :: fused

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_new = nnew
    dt_loc = dt
    if (nz > COL_KMAX) error stop 'step3d_t: nz exceeds COL_KMAX'
    ! Hz (set_depth) and Wi (omega) are current: no exchange.

    ! Advection of all tracers in one launch when it is safe (see
    ! fuse_tracers); forcing, the implicit solve and the OBC stay per tracer.
    fused = fuse_tracers(nrhs, n_new, n_new)
    lim = dry_lim(G_TRC)
    if (fused) call cor_tracers_fused(nrhs, n_new, dt_loc)
    ! River faces: their flux was masked by the advection; add it before the
    ! forcing and the implicit solve (all tracers here, per tracer below).
    if (fused .and. river_on) call rivers_ts(RIV_COR, n_new, dt_loc, 1_ip, nt)
    ! CDR volume releases carry T and S (after the fused advection; the CPU
    ! adds them between its horizontal and vertical passes, which differ
    ! only in rounding as the vertical pass reads slot nrhs)
    if (fused .and. cdr_vol_on) call cdr_apply_ts(n_new, dt_loc, 1_ip, int(nt, ip))
    do itrc = 1, nt
      if (.not. fused) then
      if (itrc == itemp) call dump_cor_col('cor in', 1)
      call horiz_up3_fluxes(nrhs, itrc)
      if (itrc == itemp) call dump_horiz_flux_probe('step3d_t', dt_loc)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          do k = 1, nz
            t(i, j, k, n_new, itrc) = t(i, j, k, n_new, itrc) - dt_loc * pm(i, j) * pn(i, j) * &
                 (wrk3a(i + 1, j, k) - wrk3a(i, j, k) + wrk3b(i, j + 1, k) - wrk3b(i, j, k))
          enddo
        enddo
      enddo
      if (itrc == itemp) call dump_cor_col('cor horiz', 2)
      if (cdr_vol_on) call cdr_apply_ts(n_new, dt_loc, int(itrc, ip), int(itrc, ip))

      call vert_u3_fluxes(nrhs, itrc)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          do k = 1, nz
            t(i, j, k, n_new, itrc) = t(i, j, k, n_new, itrc) - dt_loc * pm(i, j) * pn(i, j) * &
                 (col_wc(i, j, k) - col_wc(i, j, k - 1))
          enddo
        enddo
      enddo
      if (itrc == itemp) call dump_cor_col('cor vadv', 3)
      if (river_on) call rivers_ts(RIV_COR, n_new, dt_loc, itrc, itrc)
      endif

      if (itrc == itemp) then
        if (have_bulk) then
#ifdef USE_CUDA
          !$cuf kernel do(2) <<<*,*>>>
#endif
          do j = 1, ny
            do i = 1, nx
              if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
              t(i, j, nz, n_new, itemp) = t(i, j, nz, n_new, itemp) + &
                   dt_loc * swflx(i, j) * tair(i, j)
            enddo
          enddo
        else
#ifdef USE_CUDA
          !$cuf kernel do(2) <<<*,*>>>
#endif
          do j = 1, ny
            do i = 1, nx
              if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
              t(i, j, nz, n_new, itemp) = t(i, j, nz, n_new, itemp) + dt_loc * swflx(i, j) * &
                   t(i, j, nz, n_new, itemp) / Hz(i, j, nz)
            enddo
          enddo
        endif
      endif
      if (.not. probe_on) then
        ! Forcing applied inside the implicit solve's forward sweep (same
        ! operations, same order); probe steps keep the separate kernel for
        ! the 'cor forc' dump.
        iAkt = min(itrc, 2)
        call implicit_tracer_columns_frc(nx, ny, nz, n_new, itrc, iAkt, dt_loc)
      else
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
          t(i, j, nz, n_new, itrc) = t(i, j, nz, n_new, itrc) + dt_loc * stflx(i, j, itrc)

          if (itrc == itemp) then
            do k = nz - 1, 1, -1
              cff = srflx(i, j) * swr_frac(i, j, k) - &
                    ghat(i, j, k) * (stflx(i, j, itemp) - srflx(i, j))
              t(i, j, k + 1, n_new, itemp) = t(i, j, k + 1, n_new, itemp) - dt_loc * cff
              t(i, j, k, n_new, itemp) = t(i, j, k, n_new, itemp) + dt_loc * cff
            enddo
          elseif (itrc == isalt) then
            do k = nz - 1, 1, -1
              cff = -dt_loc * ghat(i, j, k) * stflx(i, j, isalt)
              t(i, j, k + 1, n_new, isalt) = t(i, j, k + 1, n_new, isalt) - cff
              t(i, j, k, n_new, isalt) = t(i, j, k, n_new, isalt) + cff
            enddo
          endif
        enddo
      enddo

      if (itrc == itemp) call dump_cor_col('cor forc', 4)
      iAkt = min(itrc, 2)
      call implicit_tracer_columns(nx, ny, nz, n_new, itrc, iAkt, dt_loc, .false.)
      endif
      if (itrc == itemp) call dump_cor_col('cor impl', 5)
      call t3dbc(itrc)
      ! no exchange here: MARBL and t3dmix read t(nnew) at the point only
      ! (t3dmix's stencil is slot nrhs) and rho_eos(nnew) exchanges it
    enddo
    if (ntp > 0 .and. .not. tp_subcycled) call passive_corrector(dt_loc)   ! subcycled: at BGC window ends
  end subroutine step3d_t

  subroutine dump_cor_col(tag, stage)
    ! Same "diag: pcol" lines as the roms_bugfix step3d_t probes.
    ! stage 1: t(nnew), t(nrhs), Hz, Akt    2: t(nnew), FlxU, FlxV, t(nrhs)
    !       3: t(nnew), col_wc, We, Wi          4: t(nnew), ghat, swr_frac, Hz
    !       5: t(nnew), Hz, Akt, Wi
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: stage
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
          a = (/ t(i, j, k, nnew, itemp), t(i, j, k, nrhs, itemp), Hz(i, j, k), Akt(i, j, k, itemp) /)
        case (2)
          a = (/ t(i, j, k, nnew, itemp), FlxU(i, j, k), FlxV(i, j, k), t(i, j, k, nrhs, itemp) /)
        case (3)
          a = (/ t(i, j, k, nnew, itemp), col_wc(i, j, k), We(i, j, k), Wi(i, j, k) /)
        case (4)
          a = (/ t(i, j, k, nnew, itemp), ghat(i, j, k), swr_frac(i, j, k), Hz(i, j, k) /)
        case default
          a = (/ t(i, j, k, nnew, itemp), Hz(i, j, k), Akt(i, j, k, itemp), Wi(i, j, k) /)
        end select
        write(*,'(a,a,a,i1,a,i6,i4,4es22.14)') 'diag: pcol ', trim(tag), ' P', n, &
             ' iic,k,a1..a4=', iic, k, a
      enddo
    enddo
    flush(6)
  end subroutine dump_cor_col

end module step3d_t_mod

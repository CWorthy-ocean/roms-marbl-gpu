#include "mem_attr.inc"
module uv_rhs
  ! UV_COR + CURVGRID metric terms, UPSTREAM_UV (corrector) or centered
  ! (predictor) horizontal advection, and SPLINE_UV vertical advection. Adds into gpu_state ru, rv (already holding the PGF).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use arch_mod, only: sync_arch
  implicit none
  private
  public :: add_horiz_uv_rhs
  public :: add_horiz_uv_rhs_all
  public :: add_vert_uv_rhs
  public :: dump_rv_pred
  public :: dump_vn3
  public :: dump_vim
  public :: dump_vdc
  public :: dump_vvs
  public :: dump_vvf

  ! P1 south-V corrector accumulators (host). Reset at k=1 each step.
  real(rp) :: p1_sum, p1_r1, p1_r50, p1_rnz
  real(rp) :: p1_f1, p1_f2, p1_f3, p1_f4

  ! f/max(pm*pn,1e-30) at rho points (the Coriolis factor of the 3-D kernels),
  ! built once by ensure_fomn: f, pm, pn are static after the grid load.
  real(rp), WORK_ATTR allocatable :: fomn_r(:,:)

contains

  subroutine add_horiz_uv_rhs_all(upstream)
    ! Probe steps keep the per-level path (its dumps read the 2-D work
    ! arrays level by level); otherwise one launch per stage.
    logical, intent(in) :: upstream
    integer(ip) :: k
    if (probe_on) then
      do k = 1, decomp%nz
        call add_horiz_uv_rhs(k, upstream)
      enddo
    else
      call add_horiz_uv_rhs_3d(upstream)
    endif
  end subroutine add_horiz_uv_rhs_all

  subroutine add_horiz_uv_rhs(k, upstream)
    integer(ip), intent(in) :: k
    logical, intent(in) :: upstream
    integer(ip) :: i, j, nx, ny, kk, n_rhs
    integer(ip) :: i_u0, i_u1, i_ufx_lo, i_ux_lo, i_ux_hi, j_ve_lo, j_ve_hi
    integer(ip) :: j_ue_lo, j_ue_hi, i_vx_lo, i_vx_hi, j_v0, j_v1
    logical :: edge_w, edge_e, edge_s, edge_n, up
    real(rp) :: cff, fomn

    nx = decomp%nx
    ny = decomp%ny
    kk = k
    n_rhs = nrhs
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n
    up = upstream
    ! CPU istrU:iend / jstrV:jend — skip open-boundary U/V faces.
    i_u0 = 1
    if (edge_w) i_u0 = 2
    i_u1 = nx + 1
    if (edge_e) i_u1 = nx
    i_ufx_lo = i_u0 - 1
    j_v0 = 1
    if (edge_s) j_v0 = 2
    j_v1 = ny + 1
    if (edge_n) j_v1 = ny

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny
      do i = 0, nx
        fomn = f(i, j) / max(pm(i, j) * pn(i, j), 1.0e-30_rp)
        cff = 0.5_rp * Hz(i, j, kk) * (fomn + 0.5_rp * ( &
             (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs)) * dndx(i, j) - &
             (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs)) * dmde(i, j)))
        wrk2a(i, j) = cff * (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs))
        wrk2d(i, j) = cff * (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        ru(i, j, kk) = ru(i, j, kk) + 0.5_rp * (wrk2a(i, j) + wrk2a(i - 1, j))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, j_v1
      do i = 1, nx
        rv(i, j, kk) = rv(i, j, kk) - 0.5_rp * (wrk2d(i, j) + wrk2d(i, j - 1))
      enddo
    enddo
    if (up) then
      call dump_uvh_probe('after cor', kk)
      call dump_uvc_probe('corr cor', kk)
      call dump_p1_rvh('cor', kk)
      call dump_cor_b(kk)
    else
      call dump_uvc_probe('pred cor', kk)
      call dump_p1_rvh('pred cor', kk)
    endif

    ! CPU: imin=istrU on the west edge, then uxx(istrU-1)=uxx(istrU).
    if (edge_w) then
      i_ux_lo = i_u0
    else
      i_ux_lo = 0
    endif
    if (edge_e) then
      i_ux_hi = nx
    else
      i_ux_hi = nx + 1
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_ux_lo, i_ux_hi
        wrk2e(i, j) = u(i - 1, j, kk, n_rhs) - 2.0_rp * u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs)
        wrk2f(i, j) = FlxU(i - 1, j, kk) - 2.0_rp * FlxU(i, j, kk) + FlxU(i + 1, j, kk)
      enddo
    enddo
    if (edge_w) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 1, ny
        wrk2e(i_ufx_lo, j) = wrk2e(i_u0, j)
        wrk2f(i_ufx_lo, j) = wrk2f(i_u0, j)
      enddo
    endif
    if (edge_e) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 1, ny
        wrk2e(nx + 1, j) = wrk2e(nx, j)
        wrk2f(nx + 1, j) = wrk2f(nx, j)
      enddo
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_ufx_lo, i_u1
        if (up) then
          wrk2a(i, j) = upstream_uv_face(u(i, j, kk, n_rhs), u(i + 1, j, kk, n_rhs), &
             wrk2e(i, j), wrk2e(i + 1, j), wrk2f(i, j), wrk2f(i + 1, j), &
             FlxU(i, j, kk), FlxU(i + 1, j, kk))
        else
          wrk2a(i, j) = centered_uv_face(u(i, j, kk, n_rhs), u(i + 1, j, kk, n_rhs), &
             wrk2e(i, j), wrk2e(i + 1, j), wrk2f(i, j), wrk2f(i + 1, j), &
             FlxU(i, j, kk), FlxU(i + 1, j, kk))
        endif
      enddo
    enddo

    ! CPU: jmin=jstrV on south (then vee(jstrV-1)=vee(jstrV)), jmax=jend on north.
    if (edge_s) then
      j_ve_lo = j_v0
    else
      j_ve_lo = 0
    endif
    if (edge_n) then
      j_ve_hi = j_v1
    else
      j_ve_hi = ny + 1
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_ve_lo, j_ve_hi
      do i = 1, nx
        wrk2e(i, j) = v(i, j - 1, kk, n_rhs) - 2.0_rp * v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs)
        wrk2f(i, j) = FlxV(i, j - 1, kk) - 2.0_rp * FlxV(i, j, kk) + FlxV(i, j + 1, kk)
      enddo
    enddo
    if (edge_s) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 1, nx
        wrk2e(i, j_v0 - 1) = wrk2e(i, j_v0)
        wrk2f(i, j_v0 - 1) = wrk2f(i, j_v0)
      enddo
    endif
    if (edge_n) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 1, nx
        wrk2e(i, ny + 1) = wrk2e(i, ny)
        wrk2f(i, ny + 1) = wrk2f(i, ny)
      enddo
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0 - 1, j_v1
      do i = 1, nx
        if (up) then
          wrk2d(i, j) = upstream_uv_face(v(i, j, kk, n_rhs), v(i, j + 1, kk, n_rhs), &
             wrk2e(i, j), wrk2e(i, j + 1), wrk2f(i, j), wrk2f(i, j + 1), &
             FlxV(i, j, kk), FlxV(i, j + 1, kk))
        else
          wrk2d(i, j) = centered_uv_face(v(i, j, kk, n_rhs), v(i, j + 1, kk, n_rhs), &
             wrk2e(i, j), wrk2e(i, j + 1), wrk2f(i, j), wrk2f(i, j + 1), &
             FlxV(i, j, kk), FlxV(i, j + 1, kk))
        endif
      enddo
    enddo

    if (edge_s) then
      j_ue_lo = 1
    else
      j_ue_lo = 0
    endif
    if (edge_n) then
      j_ue_hi = ny
    else
      j_ue_hi = ny + 1
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_ue_lo, j_ue_hi
      do i = i_u0, i_u1
        wrk2e(i, j) = u(i, j - 1, kk, n_rhs) - 2.0_rp * u(i, j, kk, n_rhs) + u(i, j + 1, kk, n_rhs)
      enddo
    enddo
    if (edge_s) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = i_u0, i_u1
        wrk2e(i, 0) = wrk2e(i, 1)
      enddo
    endif
    if (edge_n) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = i_u0, i_u1
        wrk2e(i, ny + 1) = wrk2e(i, ny)
      enddo
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 0, nx
        wrk2f(i, j) = FlxV(i - 1, j, kk) - 2.0_rp * FlxV(i, j, kk) + FlxV(i + 1, j, kk)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = i_u0, i_u1
        if (up) then
          wrk2b(i, j) = upstream_uv_face(u(i, j - 1, kk, n_rhs), u(i, j, kk, n_rhs), &
             wrk2e(i, j - 1), wrk2e(i, j), wrk2f(i - 1, j), wrk2f(i, j), &
             FlxV(i, j, kk), FlxV(i - 1, j, kk))
        else
          wrk2b(i, j) = centered_uv_face(u(i, j - 1, kk, n_rhs), u(i, j, kk, n_rhs), &
             wrk2e(i, j - 1), wrk2e(i, j), wrk2f(i - 1, j), wrk2f(i, j), &
             FlxV(i, j, kk), FlxV(i - 1, j, kk))
        endif
      enddo
    enddo

    if (edge_w) then
      i_vx_lo = 1
    else
      i_vx_lo = 0
    endif
    if (edge_e) then
      i_vx_hi = nx
    else
      i_vx_hi = nx + 1
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, j_v1
      do i = i_vx_lo, i_vx_hi
        wrk2e(i, j) = v(i - 1, j, kk, n_rhs) - 2.0_rp * v(i, j, kk, n_rhs) + v(i + 1, j, kk, n_rhs)
      enddo
    enddo
    if (edge_w) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = j_v0, j_v1
        wrk2e(0, j) = wrk2e(1, j)
      enddo
    endif
    if (edge_e) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = j_v0, j_v1
        wrk2e(nx + 1, j) = wrk2e(nx, j)
      enddo
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0 - 1, j_v1
      do i = 1, nx + 1
        wrk2f(i, j) = FlxU(i, j - 1, kk) - 2.0_rp * FlxU(i, j, kk) + FlxU(i, j + 1, kk)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, j_v1
      do i = 1, nx + 1
        if (up) then
          wrk2c(i, j) = upstream_uv_face(v(i - 1, j, kk, n_rhs), v(i, j, kk, n_rhs), &
             wrk2e(i - 1, j), wrk2e(i, j), wrk2f(i, j - 1), wrk2f(i, j), &
             FlxU(i, j, kk), FlxU(i, j - 1, kk))
        else
          wrk2c(i, j) = centered_uv_face(v(i - 1, j, kk, n_rhs), v(i, j, kk, n_rhs), &
             wrk2e(i - 1, j), wrk2e(i, j), wrk2f(i, j - 1), wrk2f(i, j), &
             FlxU(i, j, kk), FlxU(i, j - 1, kk))
        endif
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        ru(i, j, kk) = ru(i, j, kk) - wrk2a(i, j) + wrk2a(i - 1, j) - wrk2b(i, j + 1) + wrk2b(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, j_v1
      do i = 1, nx
        rv(i, j, kk) = rv(i, j, kk) - wrk2c(i + 1, j) + wrk2c(i, j) - wrk2d(i, j) + wrk2d(i, j - 1)
      enddo
    enddo
    if (up) then
      call dump_uvh_probe('after adv', kk)
      call dump_vvh_probe('after adv', kk)
      call dump_uva_probe('corr adv', kk)
      call dump_p1_rvh('adv', kk)
      call dump_adv_b(kk)
    else
      call dump_uva_probe('pred adv', kk)
      call dump_p1_rvh('pred adv', kk)
    endif
  end subroutine add_horiz_uv_rhs

  subroutine ensure_fomn()
    ! The same expression the kernels evaluated per point, once per rho point.
    integer(ip) :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    if (allocated(fomn_r)) then
      if (lbound(fomn_r, 1) == is .and. ubound(fomn_r, 1) == ie .and. &
          lbound(fomn_r, 2) == js .and. ubound(fomn_r, 2) == je) return
      deallocate(fomn_r)
    endif
    allocate(fomn_r(is:ie, js:je))
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        fomn_r(i, j) = f(i, j) / max(pm(i, j) * pn(i, j), 1.0e-30_rp)
      enddo
    enddo
  end subroutine ensure_fomn

  subroutine add_horiz_uv_rhs_3d(upstream)
    ! add_horiz_uv_rhs for all levels at once: the same per-point
    ! arithmetic as 3-D kernels on 3-D work arrays (no probes).
    logical, intent(in) :: upstream
    integer(ip) :: i, j, nx, ny, nz, kk, n_rhs
    integer(ip) :: i_u0, i_u1, i_ufx_lo, i_ux_lo, i_ux_hi, j_ve_lo, j_ve_hi
    integer(ip) :: j_ue_lo, j_ue_hi, i_vx_lo, i_vx_hi, j_v0, j_v1
    logical :: edge_w, edge_e, edge_s, edge_n, up
    real(rp) :: cff, fomn
    integer(ip) :: ii, jj, lim
    real(rp) :: c_0, c_m, rr, uxm, ux0, uxp, fxm, fx0, fxp, uem, ue0, uep, fvm_j, fv0_j, fvm_jp, fv0_jp
    real(rp) :: fa_m, fa_0, fb_0, fb_p
    real(rp) :: cfu, u_0, u_e, u_w, u_s, u_n, v_0, v_n, v_s, v_w, v_e
    real(rp) :: g_w, g_0, g_e, g_s, g_se, h_0, h_w, h_n, h_nw, h_s
    integer(ip) :: ipr, jpr, nip, njp
    logical :: two, act0, act1

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    call ensure_wrk3()
    n_rhs = nrhs
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n
    up = upstream
    ! CPU istrU:iend / jstrV:jend — skip open-boundary U/V faces.
    i_u0 = 1
    if (edge_w) i_u0 = 2
    i_u1 = nx + 1
    if (edge_e) i_u1 = nx
    i_ufx_lo = i_u0 - 1
    j_v0 = 1
    if (edge_s) j_v0 = 2
    j_v1 = ny + 1
    if (edge_n) j_v1 = ny

    ! Curvature ranges of the former wrk3e/wrk3f temporaries; the edge copies
    ! (uxx(istrU-1) = uxx(istrU) etc.) become index clamps below.
    if (edge_w) then
      i_ux_lo = i_u0
    else
      i_ux_lo = 0
    endif
    if (edge_e) then
      i_ux_hi = nx
    else
      i_ux_hi = nx + 1
    endif
    if (edge_s) then
      j_ve_lo = j_v0
    else
      j_ve_lo = 0
    endif
    if (edge_n) then
      j_ve_hi = j_v1
    else
      j_ve_hi = ny + 1
    endif
    if (edge_s) then
      j_ue_lo = 1
    else
      j_ue_lo = 0
    endif
    if (edge_n) then
      j_ue_hi = ny
    else
      j_ue_hi = ny + 1
    endif
    if (edge_w) then
      i_vx_lo = 1
    else
      i_vx_lo = 0
    endif
    if (edge_e) then
      i_vx_hi = nx
    else
      i_vx_hi = nx + 1
    endif
    lim = dry_lim(G_UVRHS)
    call ensure_fomn()
    nip = (i_u1 - i_u0 + 2) / 2
    njp = (j_v1 - j_v0 + 2) / 2
    if (up) then
    ! ---- ru (UPSTREAM_UV faces), each thread the points i and i+1; per point the
    ! Coriolis/curvature term of the two neighbouring rho points, the u-by-FlxU faces
    ! i-1 and i, the u-by-FlxV faces j and j+1, each face the same expression as
    ! upstream_uv_face (expanded here: as a call from this kernel it was not inlined),
    ! and the same accumulation order: ru + 0.5*(cor_i + cor_im1) - fa_i + fa_im1 - fb_jp1 + fb_j.
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do kk = 1, nz
      do j = 1, ny
        do ipr = 0, nip - 1
          i = i_u0 + 2 * ipr
          two = (i + 1 <= i_u1)
          act0 = .true.
          act1 = two
          if (lim > 0) then
            act0 = (colact(i, j) /= 0)
            if (two) act1 = (colact(i + 1, j) /= 0)
          endif
          if (.not. (act0 .or. act1)) cycle
          ! ---- point i
          u_0 = u(i, j, kk, n_rhs)
          u_e = u(i + 1, j, kk, n_rhs)
          u_w = u(i - 1, j, kk, n_rhs)
          cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
               (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs)) * dndx(i, j) - &
               (u_0 + u_e) * dmde(i, j)))
          c_0 = cff * (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs))
          cff = 0.5_rp * Hz(i - 1, j, kk) * (fomn_r(i - 1, j) + 0.5_rp * ( &
               (v(i - 1, j, kk, n_rhs) + v(i - 1, j + 1, kk, n_rhs)) * dndx(i - 1, j) - &
               (u_w + u_0) * dmde(i - 1, j)))
          c_m = cff * (v(i - 1, j, kk, n_rhs) + v(i - 1, j + 1, kk, n_rhs))
          ii = max(i - 1, i_ux_lo)
          uxm = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
          fxm = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
          ii = min(max(i, i_ux_lo), i_ux_hi)
          ux0 = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
          fx0 = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
          ii = min(i + 1, i_ux_hi)
          uxp = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
          fxp = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
          g_w = FlxU(i - 1, j, kk)
          g_0 = FlxU(i, j, kk)
          g_e = FlxU(i + 1, j, kk)
          cfu = g_w + g_0 - uv_delta * (fxm + fx0)
          fa_m = 0.25_rp * (cfu * (u_w + u_0) - uv_gamma * ( &
               max(cfu, 0.0_rp) * uxm + min(cfu, 0.0_rp) * ux0))
          cfu = g_0 + g_e - uv_delta * (fx0 + fxp)
          fa_0 = 0.25_rp * (cfu * (u_0 + u_e) - uv_gamma * ( &
               max(cfu, 0.0_rp) * ux0 + min(cfu, 0.0_rp) * uxp))
          u_s = u(i, j - 1, kk, n_rhs)
          u_n = u(i, j + 1, kk, n_rhs)
          jj = max(j - 1, j_ue_lo)
          uem = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
          ue0 = u_s - 2.0_rp * u_0 + u_n
          jj = min(j + 1, j_ue_hi)
          uep = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
          h_w = FlxV(i - 1, j, kk)
          h_nw = FlxV(i - 1, j + 1, kk)
          h_0 = FlxV(i, j, kk)
          h_n = FlxV(i, j + 1, kk)
          fvm_j = FlxV(i - 2, j, kk) - 2.0_rp * h_w + h_0
          fvm_jp = FlxV(i - 2, j + 1, kk) - 2.0_rp * h_nw + h_n
          fv0_j = h_w - 2.0_rp * h_0 + FlxV(i + 1, j, kk)
          fv0_jp = h_nw - 2.0_rp * h_n + FlxV(i + 1, j + 1, kk)
          cfu = h_0 + h_w - uv_delta * (fvm_j + fv0_j)
          fb_0 = 0.25_rp * (cfu * (u_s + u_0) - uv_gamma * ( &
               max(cfu, 0.0_rp) * uem + min(cfu, 0.0_rp) * ue0))
          cfu = h_n + h_nw - uv_delta * (fvm_jp + fv0_jp)
          fb_p = 0.25_rp * (cfu * (u_0 + u_n) - uv_gamma * ( &
               max(cfu, 0.0_rp) * ue0 + min(cfu, 0.0_rp) * uep))
          if (act0) ru(i, j, kk) = ru(i, j, kk) + 0.5_rp * (c_0 + c_m) - fa_0 + fa_m - fb_p + fb_0
          if (act1) then
            ! ---- point i+1. Reused from point i (same expressions, same operands):
            ! Coriolis at rho i (c_m), face fa_m = fa_0 (curvatures max(i,lo) =
            ! min(max(i,lo),hi) and min(max(i+1,lo),hi) = min(i+1,hi) for i_u0 <= i < i_u1),
            ! FlxV x-curvatures at i (fvm_j, fvm_jp = fv0_j, fv0_jp of point i).
            i = i + 1
            c_m = c_0
            fa_m = fa_0
            ux0 = uxp
            fx0 = fxp
            u_w = u_0
            u_0 = u_e
            u_e = u(i + 1, j, kk, n_rhs)
            g_0 = g_e
            g_e = FlxU(i + 1, j, kk)
            cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
                 (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs)) * dndx(i, j) - &
                 (u_0 + u_e) * dmde(i, j)))
            c_0 = cff * (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs))
            ii = min(i + 1, i_ux_hi)
            uxp = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
            fxp = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
            cfu = g_0 + g_e - uv_delta * (fx0 + fxp)
            fa_0 = 0.25_rp * (cfu * (u_0 + u_e) - uv_gamma * ( &
                 max(cfu, 0.0_rp) * ux0 + min(cfu, 0.0_rp) * uxp))
            fvm_j = fv0_j
            fvm_jp = fv0_jp
            h_w = h_0
            h_nw = h_n
            u_s = u(i, j - 1, kk, n_rhs)
            u_n = u(i, j + 1, kk, n_rhs)
            jj = max(j - 1, j_ue_lo)
            uem = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
            ue0 = u_s - 2.0_rp * u_0 + u_n
            jj = min(j + 1, j_ue_hi)
            uep = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
            h_0 = FlxV(i, j, kk)
            h_n = FlxV(i, j + 1, kk)
            fv0_j = h_w - 2.0_rp * h_0 + FlxV(i + 1, j, kk)
            fv0_jp = h_nw - 2.0_rp * h_n + FlxV(i + 1, j + 1, kk)
            cfu = h_0 + h_w - uv_delta * (fvm_j + fv0_j)
            fb_0 = 0.25_rp * (cfu * (u_s + u_0) - uv_gamma * ( &
                 max(cfu, 0.0_rp) * uem + min(cfu, 0.0_rp) * ue0))
            cfu = h_n + h_nw - uv_delta * (fvm_jp + fv0_jp)
            fb_p = 0.25_rp * (cfu * (u_0 + u_n) - uv_gamma * ( &
                 max(cfu, 0.0_rp) * ue0 + min(cfu, 0.0_rp) * uep))
            ru(i, j, kk) = ru(i, j, kk) + 0.5_rp * (c_0 + c_m) - fa_0 + fa_m - fb_p + fb_0
          endif
        enddo
      enddo
    enddo
    ! ---- rv, each thread the points j and j+1: rv - 0.5*(cor_j + cor_jm1) - fc_ip1 + fc_i - fd_j + fd_jm1
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do kk = 1, nz
      do jpr = 0, njp - 1
        do i = 1, nx
          j = j_v0 + 2 * jpr
          two = (j + 1 <= j_v1)
          act0 = .true.
          act1 = two
          if (lim > 0) then
            act0 = (colact(i, j) /= 0)
            if (two) act1 = (colact(i, j + 1) /= 0)
          endif
          if (.not. (act0 .or. act1)) cycle
          ! ---- point j
          v_0 = v(i, j, kk, n_rhs)
          v_n = v(i, j + 1, kk, n_rhs)
          v_s = v(i, j - 1, kk, n_rhs)
          cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
               (v_0 + v_n) * dndx(i, j) - &
               (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs)) * dmde(i, j)))
          c_0 = cff * (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs))
          cff = 0.5_rp * Hz(i, j - 1, kk) * (fomn_r(i, j - 1) + 0.5_rp * ( &
               (v_s + v_0) * dndx(i, j - 1) - &
               (u(i, j - 1, kk, n_rhs) + u(i + 1, j - 1, kk, n_rhs)) * dmde(i, j - 1)))
          c_m = cff * (u(i, j - 1, kk, n_rhs) + u(i + 1, j - 1, kk, n_rhs))
          jj = max(j - 1, j_ve_lo)
          uxm = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
          fxm = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
          jj = min(max(j, j_ve_lo), j_ve_hi)
          ux0 = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
          fx0 = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
          jj = min(j + 1, j_ve_hi)
          uxp = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
          fxp = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
          h_s = FlxV(i, j - 1, kk)
          h_0 = FlxV(i, j, kk)
          h_n = FlxV(i, j + 1, kk)
          cfu = h_s + h_0 - uv_delta * (fxm + fx0)
          fa_m = 0.25_rp * (cfu * (v_s + v_0) - uv_gamma * ( &
               max(cfu, 0.0_rp) * uxm + min(cfu, 0.0_rp) * ux0))
          cfu = h_0 + h_n - uv_delta * (fx0 + fxp)
          fa_0 = 0.25_rp * (cfu * (v_0 + v_n) - uv_gamma * ( &
               max(cfu, 0.0_rp) * ux0 + min(cfu, 0.0_rp) * uxp))
          v_w = v(i - 1, j, kk, n_rhs)
          v_e = v(i + 1, j, kk, n_rhs)
          ii = max(i - 1, i_vx_lo)
          uem = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
          ue0 = v_w - 2.0_rp * v_0 + v_e
          ii = min(i + 1, i_vx_hi)
          uep = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
          g_s = FlxU(i, j - 1, kk)
          g_se = FlxU(i + 1, j - 1, kk)
          g_0 = FlxU(i, j, kk)
          g_e = FlxU(i + 1, j, kk)
          fvm_j = FlxU(i, j - 2, kk) - 2.0_rp * g_s + g_0
          fvm_jp = FlxU(i + 1, j - 2, kk) - 2.0_rp * g_se + g_e
          fv0_j = g_s - 2.0_rp * g_0 + FlxU(i, j + 1, kk)
          fv0_jp = g_se - 2.0_rp * g_e + FlxU(i + 1, j + 1, kk)
          cfu = g_0 + g_s - uv_delta * (fvm_j + fv0_j)
          fb_0 = 0.25_rp * (cfu * (v_w + v_0) - uv_gamma * ( &
               max(cfu, 0.0_rp) * uem + min(cfu, 0.0_rp) * ue0))
          cfu = g_e + g_se - uv_delta * (fvm_jp + fv0_jp)
          fb_p = 0.25_rp * (cfu * (v_0 + v_e) - uv_gamma * ( &
               max(cfu, 0.0_rp) * ue0 + min(cfu, 0.0_rp) * uep))
          if (act0) rv(i, j, kk) = rv(i, j, kk) - 0.5_rp * (c_0 + c_m) - fb_p + fb_0 - fa_0 + fa_m
          if (act1) then
            ! ---- point j+1. Reused from point j (same expressions, same operands):
            ! Coriolis at rho j (c_m), face fa_m = fa_0 (curvatures max(j,lo) =
            ! min(max(j,lo),hi) and min(max(j+1,lo),hi) = min(j+1,hi) for j_v0 <= j < j_v1,
            ! j_ve_hi = j_v1), FlxU y-curvatures at j (fvm_j, fvm_jp = fv0_j, fv0_jp of point j).
            j = j + 1
            c_m = c_0
            fa_m = fa_0
            ux0 = uxp
            fx0 = fxp
            v_s = v_0
            v_0 = v_n
            v_n = v(i, j + 1, kk, n_rhs)
            h_0 = h_n
            h_n = FlxV(i, j + 1, kk)
            cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
                 (v_0 + v_n) * dndx(i, j) - &
                 (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs)) * dmde(i, j)))
            c_0 = cff * (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs))
            jj = min(j + 1, j_ve_hi)
            uxp = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
            fxp = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
            cfu = h_0 + h_n - uv_delta * (fx0 + fxp)
            fa_0 = 0.25_rp * (cfu * (v_0 + v_n) - uv_gamma * ( &
                 max(cfu, 0.0_rp) * ux0 + min(cfu, 0.0_rp) * uxp))
            fvm_j = fv0_j
            fvm_jp = fv0_jp
            g_s = g_0
            g_se = g_e
            v_w = v(i - 1, j, kk, n_rhs)
            v_e = v(i + 1, j, kk, n_rhs)
            ii = max(i - 1, i_vx_lo)
            uem = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
            ue0 = v_w - 2.0_rp * v_0 + v_e
            ii = min(i + 1, i_vx_hi)
            uep = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
            g_0 = FlxU(i, j, kk)
            g_e = FlxU(i + 1, j, kk)
            fv0_j = g_s - 2.0_rp * g_0 + FlxU(i, j + 1, kk)
            fv0_jp = g_se - 2.0_rp * g_e + FlxU(i + 1, j + 1, kk)
            cfu = g_0 + g_s - uv_delta * (fvm_j + fv0_j)
            fb_0 = 0.25_rp * (cfu * (v_w + v_0) - uv_gamma * ( &
                 max(cfu, 0.0_rp) * uem + min(cfu, 0.0_rp) * ue0))
            cfu = g_e + g_se - uv_delta * (fvm_jp + fv0_jp)
            fb_p = 0.25_rp * (cfu * (v_0 + v_e) - uv_gamma * ( &
                 max(cfu, 0.0_rp) * ue0 + min(cfu, 0.0_rp) * uep))
            rv(i, j, kk) = rv(i, j, kk) - 0.5_rp * (c_0 + c_m) - fb_p + fb_0 - fa_0 + fa_m
          endif
        enddo
      enddo
    enddo
    else
    ! ---- the same with centered_uv_face (predictor)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do kk = 1, nz
      do j = 1, ny
        do ipr = 0, nip - 1
          i = i_u0 + 2 * ipr
          two = (i + 1 <= i_u1)
          act0 = .true.
          act1 = two
          if (lim > 0) then
            act0 = (colact(i, j) /= 0)
            if (two) act1 = (colact(i + 1, j) /= 0)
          endif
          if (.not. (act0 .or. act1)) cycle
          ! ---- point i
          u_0 = u(i, j, kk, n_rhs)
          u_e = u(i + 1, j, kk, n_rhs)
          u_w = u(i - 1, j, kk, n_rhs)
          cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
               (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs)) * dndx(i, j) - &
               (u_0 + u_e) * dmde(i, j)))
          c_0 = cff * (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs))
          cff = 0.5_rp * Hz(i - 1, j, kk) * (fomn_r(i - 1, j) + 0.5_rp * ( &
               (v(i - 1, j, kk, n_rhs) + v(i - 1, j + 1, kk, n_rhs)) * dndx(i - 1, j) - &
               (u_w + u_0) * dmde(i - 1, j)))
          c_m = cff * (v(i - 1, j, kk, n_rhs) + v(i - 1, j + 1, kk, n_rhs))
          ii = max(i - 1, i_ux_lo)
          uxm = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
          fxm = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
          ii = min(max(i, i_ux_lo), i_ux_hi)
          ux0 = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
          fx0 = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
          ii = min(i + 1, i_ux_hi)
          uxp = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
          fxp = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
          g_w = FlxU(i - 1, j, kk)
          g_0 = FlxU(i, j, kk)
          g_e = FlxU(i + 1, j, kk)
          fa_m = 0.25_rp * (u_w + u_0 - uv_delta * (uxm + ux0)) * &
               (g_w + g_0 - uv_delta * (fxm + fx0))
          fa_0 = 0.25_rp * (u_0 + u_e - uv_delta * (ux0 + uxp)) * &
               (g_0 + g_e - uv_delta * (fx0 + fxp))
          u_s = u(i, j - 1, kk, n_rhs)
          u_n = u(i, j + 1, kk, n_rhs)
          jj = max(j - 1, j_ue_lo)
          uem = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
          ue0 = u_s - 2.0_rp * u_0 + u_n
          jj = min(j + 1, j_ue_hi)
          uep = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
          h_w = FlxV(i - 1, j, kk)
          h_nw = FlxV(i - 1, j + 1, kk)
          h_0 = FlxV(i, j, kk)
          h_n = FlxV(i, j + 1, kk)
          fvm_j = FlxV(i - 2, j, kk) - 2.0_rp * h_w + h_0
          fvm_jp = FlxV(i - 2, j + 1, kk) - 2.0_rp * h_nw + h_n
          fv0_j = h_w - 2.0_rp * h_0 + FlxV(i + 1, j, kk)
          fv0_jp = h_nw - 2.0_rp * h_n + FlxV(i + 1, j + 1, kk)
          fb_0 = 0.25_rp * (u_s + u_0 - uv_delta * (uem + ue0)) * &
               (h_0 + h_w - uv_delta * (fvm_j + fv0_j))
          fb_p = 0.25_rp * (u_0 + u_n - uv_delta * (ue0 + uep)) * &
               (h_n + h_nw - uv_delta * (fvm_jp + fv0_jp))
          if (act0) ru(i, j, kk) = ru(i, j, kk) + 0.5_rp * (c_0 + c_m) - fa_0 + fa_m - fb_p + fb_0
          if (act1) then
            ! ---- point i+1. Reused from point i (same expressions, same operands):
            ! Coriolis at rho i (c_m), face fa_m = fa_0 (curvatures max(i,lo) =
            ! min(max(i,lo),hi) and min(max(i+1,lo),hi) = min(i+1,hi) for i_u0 <= i < i_u1),
            ! FlxV x-curvatures at i (fvm_j, fvm_jp = fv0_j, fv0_jp of point i).
            i = i + 1
            c_m = c_0
            fa_m = fa_0
            ux0 = uxp
            fx0 = fxp
            u_w = u_0
            u_0 = u_e
            u_e = u(i + 1, j, kk, n_rhs)
            g_0 = g_e
            g_e = FlxU(i + 1, j, kk)
            cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
                 (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs)) * dndx(i, j) - &
                 (u_0 + u_e) * dmde(i, j)))
            c_0 = cff * (v(i, j, kk, n_rhs) + v(i, j + 1, kk, n_rhs))
            ii = min(i + 1, i_ux_hi)
            uxp = u(ii - 1, j, kk, n_rhs) - 2.0_rp * u(ii, j, kk, n_rhs) + u(ii + 1, j, kk, n_rhs)
            fxp = FlxU(ii - 1, j, kk) - 2.0_rp * FlxU(ii, j, kk) + FlxU(ii + 1, j, kk)
            fa_0 = 0.25_rp * (u_0 + u_e - uv_delta * (ux0 + uxp)) * &
                 (g_0 + g_e - uv_delta * (fx0 + fxp))
            fvm_j = fv0_j
            fvm_jp = fv0_jp
            h_w = h_0
            h_nw = h_n
            u_s = u(i, j - 1, kk, n_rhs)
            u_n = u(i, j + 1, kk, n_rhs)
            jj = max(j - 1, j_ue_lo)
            uem = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
            ue0 = u_s - 2.0_rp * u_0 + u_n
            jj = min(j + 1, j_ue_hi)
            uep = u(i, jj - 1, kk, n_rhs) - 2.0_rp * u(i, jj, kk, n_rhs) + u(i, jj + 1, kk, n_rhs)
            h_0 = FlxV(i, j, kk)
            h_n = FlxV(i, j + 1, kk)
            fv0_j = h_w - 2.0_rp * h_0 + FlxV(i + 1, j, kk)
            fv0_jp = h_nw - 2.0_rp * h_n + FlxV(i + 1, j + 1, kk)
            fb_0 = 0.25_rp * (u_s + u_0 - uv_delta * (uem + ue0)) * &
                 (h_0 + h_w - uv_delta * (fvm_j + fv0_j))
            fb_p = 0.25_rp * (u_0 + u_n - uv_delta * (ue0 + uep)) * &
                 (h_n + h_nw - uv_delta * (fvm_jp + fv0_jp))
            ru(i, j, kk) = ru(i, j, kk) + 0.5_rp * (c_0 + c_m) - fa_0 + fa_m - fb_p + fb_0
          endif
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,(32,16,1)>>>
#endif
    do kk = 1, nz
      do jpr = 0, njp - 1
        do i = 1, nx
          j = j_v0 + 2 * jpr
          two = (j + 1 <= j_v1)
          act0 = .true.
          act1 = two
          if (lim > 0) then
            act0 = (colact(i, j) /= 0)
            if (two) act1 = (colact(i, j + 1) /= 0)
          endif
          if (.not. (act0 .or. act1)) cycle
          ! ---- point j
          v_0 = v(i, j, kk, n_rhs)
          v_n = v(i, j + 1, kk, n_rhs)
          v_s = v(i, j - 1, kk, n_rhs)
          cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
               (v_0 + v_n) * dndx(i, j) - &
               (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs)) * dmde(i, j)))
          c_0 = cff * (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs))
          cff = 0.5_rp * Hz(i, j - 1, kk) * (fomn_r(i, j - 1) + 0.5_rp * ( &
               (v_s + v_0) * dndx(i, j - 1) - &
               (u(i, j - 1, kk, n_rhs) + u(i + 1, j - 1, kk, n_rhs)) * dmde(i, j - 1)))
          c_m = cff * (u(i, j - 1, kk, n_rhs) + u(i + 1, j - 1, kk, n_rhs))
          jj = max(j - 1, j_ve_lo)
          uxm = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
          fxm = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
          jj = min(max(j, j_ve_lo), j_ve_hi)
          ux0 = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
          fx0 = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
          jj = min(j + 1, j_ve_hi)
          uxp = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
          fxp = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
          h_s = FlxV(i, j - 1, kk)
          h_0 = FlxV(i, j, kk)
          h_n = FlxV(i, j + 1, kk)
          fa_m = 0.25_rp * (v_s + v_0 - uv_delta * (uxm + ux0)) * &
               (h_s + h_0 - uv_delta * (fxm + fx0))
          fa_0 = 0.25_rp * (v_0 + v_n - uv_delta * (ux0 + uxp)) * &
               (h_0 + h_n - uv_delta * (fx0 + fxp))
          v_w = v(i - 1, j, kk, n_rhs)
          v_e = v(i + 1, j, kk, n_rhs)
          ii = max(i - 1, i_vx_lo)
          uem = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
          ue0 = v_w - 2.0_rp * v_0 + v_e
          ii = min(i + 1, i_vx_hi)
          uep = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
          g_s = FlxU(i, j - 1, kk)
          g_se = FlxU(i + 1, j - 1, kk)
          g_0 = FlxU(i, j, kk)
          g_e = FlxU(i + 1, j, kk)
          fvm_j = FlxU(i, j - 2, kk) - 2.0_rp * g_s + g_0
          fvm_jp = FlxU(i + 1, j - 2, kk) - 2.0_rp * g_se + g_e
          fv0_j = g_s - 2.0_rp * g_0 + FlxU(i, j + 1, kk)
          fv0_jp = g_se - 2.0_rp * g_e + FlxU(i + 1, j + 1, kk)
          fb_0 = 0.25_rp * (v_w + v_0 - uv_delta * (uem + ue0)) * &
               (g_0 + g_s - uv_delta * (fvm_j + fv0_j))
          fb_p = 0.25_rp * (v_0 + v_e - uv_delta * (ue0 + uep)) * &
               (g_e + g_se - uv_delta * (fvm_jp + fv0_jp))
          if (act0) rv(i, j, kk) = rv(i, j, kk) - 0.5_rp * (c_0 + c_m) - fb_p + fb_0 - fa_0 + fa_m
          if (act1) then
            ! ---- point j+1. Reused from point j (same expressions, same operands):
            ! Coriolis at rho j (c_m), face fa_m = fa_0 (curvatures max(j,lo) =
            ! min(max(j,lo),hi) and min(max(j+1,lo),hi) = min(j+1,hi) for j_v0 <= j < j_v1,
            ! j_ve_hi = j_v1), FlxU y-curvatures at j (fvm_j, fvm_jp = fv0_j, fv0_jp of point j).
            j = j + 1
            c_m = c_0
            fa_m = fa_0
            ux0 = uxp
            fx0 = fxp
            v_s = v_0
            v_0 = v_n
            v_n = v(i, j + 1, kk, n_rhs)
            h_0 = h_n
            h_n = FlxV(i, j + 1, kk)
            cff = 0.5_rp * Hz(i, j, kk) * (fomn_r(i, j) + 0.5_rp * ( &
                 (v_0 + v_n) * dndx(i, j) - &
                 (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs)) * dmde(i, j)))
            c_0 = cff * (u(i, j, kk, n_rhs) + u(i + 1, j, kk, n_rhs))
            jj = min(j + 1, j_ve_hi)
            uxp = v(i, jj - 1, kk, n_rhs) - 2.0_rp * v(i, jj, kk, n_rhs) + v(i, jj + 1, kk, n_rhs)
            fxp = FlxV(i, jj - 1, kk) - 2.0_rp * FlxV(i, jj, kk) + FlxV(i, jj + 1, kk)
            fa_0 = 0.25_rp * (v_0 + v_n - uv_delta * (ux0 + uxp)) * &
                 (h_0 + h_n - uv_delta * (fx0 + fxp))
            fvm_j = fv0_j
            fvm_jp = fv0_jp
            g_s = g_0
            g_se = g_e
            v_w = v(i - 1, j, kk, n_rhs)
            v_e = v(i + 1, j, kk, n_rhs)
            ii = max(i - 1, i_vx_lo)
            uem = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
            ue0 = v_w - 2.0_rp * v_0 + v_e
            ii = min(i + 1, i_vx_hi)
            uep = v(ii - 1, j, kk, n_rhs) - 2.0_rp * v(ii, j, kk, n_rhs) + v(ii + 1, j, kk, n_rhs)
            g_0 = FlxU(i, j, kk)
            g_e = FlxU(i + 1, j, kk)
            fv0_j = g_s - 2.0_rp * g_0 + FlxU(i, j + 1, kk)
            fv0_jp = g_se - 2.0_rp * g_e + FlxU(i + 1, j + 1, kk)
            fb_0 = 0.25_rp * (v_w + v_0 - uv_delta * (uem + ue0)) * &
                 (g_0 + g_s - uv_delta * (fvm_j + fv0_j))
            fb_p = 0.25_rp * (v_0 + v_e - uv_delta * (ue0 + uep)) * &
                 (g_e + g_se - uv_delta * (fvm_jp + fv0_jp))
            rv(i, j, kk) = rv(i, j, kk) - 0.5_rp * (c_0 + c_m) - fb_p + fb_0 - fa_0 + fa_m
          endif
        enddo
      enddo
    enddo
    endif
  end subroutine add_horiz_uv_rhs_3d

  subroutine dump_p1_rvh(tag, k)
    ! Split corrector horiz at P1 south V-face (391,432). After cor,
    ! wrk2d is VFe; after adv, wrk2c/wrk2d are VFx/VFe. Accumulators
    ! rebuild the pgf+cor column because later k already have adv.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: k
    integer(ip) :: i, j, nz, n_rhs
    real(rp) :: vfe, vfes, vfxw, vfxe, inc, fomn, fomnS
    logical :: is_cor

    if (.not. probe_on) return
    i = iprb_pt(1)
    j = jprb_pt(1)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    nz = decomp%nz
    n_rhs = nrhs
    is_cor = (index(tag, 'cor') > 0)
    if (k == 1) then
      p1_sum = 0.0_rp
      p1_f1 = 0.0_rp
      p1_f2 = 0.0_rp
      p1_f3 = 0.0_rp
      p1_f4 = 0.0_rp
    endif
    p1_sum = p1_sum + rv(i, j, k)
    if (k == 1) p1_r1 = rv(i, j, k)
    if (k == 50) p1_r50 = rv(i, j, k)
    if (k == nz) p1_rnz = rv(i, j, k)

    if (is_cor) then
      vfe = wrk2d(i, j)
      vfes = wrk2d(i, j - 1)
      inc = -0.5_rp * (vfe + vfes)
      p1_f1 = p1_f1 + vfe
      p1_f2 = p1_f2 + vfes
      call dump_p1_vfes(k, i, j, n_rhs, vfes)
      if (k == 1 .or. k == 50 .or. k == nz) then
        fomn = f(i, j) / max(pm(i, j) * pn(i, j), 1.0e-30_rp)
        fomnS = f(i, j - 1) / max(pm(i, j - 1) * pn(i, j - 1), 1.0e-30_rp)
        write(*,'(a,a,a,i4,a,i8,a,12es26.17)') 'diag: rvh P1  ', trim(tag), ' k=', k, &
             '  iic=', iic, '  VFe,VFeS,inc,rv,Hz,HzS,fomn,fomnS,u,uS,v,vS=', &
             vfe, vfes, inc, rv(i, j, k), Hz(i, j, k), Hz(i, j - 1, k), &
             fomn, fomnS, u(i, j, k, n_rhs), u(i, j - 1, k, n_rhs), &
             v(i, j, k, n_rhs), v(i, j - 1, k, n_rhs)
      endif
      if (k == nz) then
        inc = -0.5_rp * (p1_f1 + p1_f2)
        write(*,'(a,a,a,i8,a,7es26.17)') 'diag: rvh P1  ', trim(tag), '  iic=', iic, &
             '  VFe,VFeS,inc,sum_rv,rv1,rv50,rv100=', &
             p1_f1, p1_f2, inc, p1_sum, p1_r1, p1_r50, p1_rnz
        write(*,'(a,a,i8,a,6es26.17)') 'diag: rv3d P1  after cor', &
             '  iic=', iic, '  sum_rv,rv1,rv50,rv100,rvfrc,extra=', &
             p1_sum, p1_r1, p1_r50, p1_rnz, rvfrc(i, j), inc
      endif
    else
      vfxw = wrk2c(i, j)
      vfxe = wrk2c(i + 1, j)
      vfe = wrk2d(i, j)
      vfes = wrk2d(i, j - 1)
      inc = -vfxe + vfxw - vfe + vfes
      p1_f1 = p1_f1 + vfxw
      p1_f2 = p1_f2 + vfxe
      p1_f3 = p1_f3 + vfe
      p1_f4 = p1_f4 + vfes
      call dump_p1_vfxe(k, i, j, n_rhs, vfxe)
      if (k == 1 .or. k == 50 .or. k == nz) then
        write(*,'(a,a,a,i4,a,i8,a,12es26.17)') 'diag: rvh P1  ', trim(tag), ' k=', k, &
             '  iic=', iic, '  VFxW,VFxE,VFe,VFeS,inc,rv,vW,v,vE,Fu,FuS,FuE=', &
             vfxw, vfxe, vfe, vfes, inc, rv(i, j, k), &
             v(i - 1, j, k, n_rhs), v(i, j, k, n_rhs), v(i + 1, j, k, n_rhs), &
             FlxU(i, j, k), FlxU(i, j - 1, k), FlxU(i + 1, j, k)
      endif
      if (k == nz) then
        inc = -p1_f2 + p1_f1 - p1_f3 + p1_f4
        write(*,'(a,a,a,i8,a,9es26.17)') 'diag: rvh P1  ', trim(tag), '  iic=', iic, &
             '  VFxW,VFxE,VFe,VFeS,inc,sum_rv,rv1,rv50,rv100=', &
             p1_f1, p1_f2, p1_f3, p1_f4, inc, p1_sum, p1_r1, p1_r50, p1_rnz
        write(*,'(a,a,i8,a,6es26.17)') 'diag: rv3d P1  after adv', &
             '  iic=', iic, '  sum_rv,rv1,rv50,rv100,rvfrc,extra=', &
             p1_sum, p1_r1, p1_r50, p1_rnz, rvfrc(i, j), inc
      endif
    endif
    flush(6)
  end subroutine dump_p1_rvh

  subroutine dump_p1_vfes(k, i, j, n_rhs, vfes)
    ! Every-k Coriolis VFe at the south neighbor of P1: rho (391,431).
    integer(ip), intent(in) :: k, i, j, n_rhs
    real(rp), intent(in) :: vfes
    integer(ip) :: js
    real(rp) :: hzs, uu, ue, vv, vn, dx, de, foms, cffs, rebuilt

    js = j - 1
    if (js < 1) return
    hzs = Hz(i, js, k)
    uu = u(i, js, k, n_rhs)
    ue = u(i + 1, js, k, n_rhs)
    vv = v(i, js, k, n_rhs)
    vn = v(i, j, k, n_rhs)
    dx = dndx(i, js)
    de = dmde(i, js)
    foms = f(i, js) / max(pm(i, js) * pn(i, js), 1.0e-30_rp)
    cffs = 0.5_rp * hzs * (foms + 0.5_rp * ((vv + vn) * dx - (uu + ue) * de))
    rebuilt = cffs * (uu + ue)
    write(*,'(a,i4,a,i8,a,11es18.10)') 'diag: vfes P1  k=', k, &
         '  iic=', iic, '  VFeS,rebuilt,Hz,u,uE,v,vN,dndx,dmde,fomn,cff=', &
         vfes, rebuilt, hzs, uu, ue, vv, vn, dx, de, foms, cffs
  end subroutine dump_p1_vfes

  subroutine dump_p1_vfxe(k, i, j, n_rhs, vfxe)
    ! Every-k UP3 VFx at the east U-face of P1 V (391,432): VFx(i+1,j).
    integer(ip), intent(in) :: k, i, j, n_rhs
    real(rp), intent(in) :: vfxe
    integer(ip) :: ie
    real(rp) :: vw, vv, ve, vee, fu, fus, fun, hu, hus, cffs, rebuilt

    ie = i + 1
    if (ie > decomp%nx + 1) return
    vw = v(i - 1, j, k, n_rhs)
    vv = v(i, j, k, n_rhs)
    ve = v(ie, j, k, n_rhs)
    vee = v(ie + 1, j, k, n_rhs)
    fu = FlxU(ie, j, k)
    fus = FlxU(ie, j - 1, k)
    fun = FlxU(ie, j + 1, k)
    hu = fus - 2.0_rp * fu + fun
    hus = FlxU(ie, j - 2, k) - 2.0_rp * fus + fu
    rebuilt = upstream_uv_face(vv, ve, vw - 2.0_rp * vv + ve, &
         vv - 2.0_rp * ve + vee, hus, hu, fu, fus)
    cffs = fu + fus - uv_delta * (hu + hus)
    write(*,'(a,i4,a,i8,a,12es18.10)') 'diag: vfxe P1  k=', k, &
         '  iic=', iic, '  VFxE,rebuilt,vW,v,vE,vEE,Fu,FuS,FuN,Huee,HueeS,cff=', &
         vfxe, rebuilt, vw, vv, ve, vee, fu, fus, fun, hu, hus, cffs
  end subroutine dump_p1_vfxe

  subroutine dump_uvc_probe(tag, k)
    ! Coriolis / metric U-RHS at P1 (318,432). wrk2a is still UFx here.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: k
    integer(ip) :: i, j, n_rhs
    real(rp) :: fomn, fomnW
    if (.not. probe_on) return
    if (.not. (k == 1 .or. k == 50 .or. k == 99 .or. k == 100)) return
    i = iprb_pt(1)
    j = jprb_pt(1)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    n_rhs = nrhs
    fomn = f(i, j) / max(pm(i, j) * pn(i, j), 1.0e-30_rp)
    fomnW = f(i - 1, j) / max(pm(i - 1, j) * pn(i - 1, j), 1.0e-30_rp)
    write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,18es26.17)') 'diag: uvC  i=', i, &
         ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
         '  Hz,HzW,fomn,fomnW,dndx,dndxW,dmde,dmdeW,u,uE,uW,v,vN,vW,vWN,UFx,UFxW,ru=', &
         Hz(i, j, k), Hz(i - 1, j, k), fomn, fomnW, &
         dndx(i, j), dndx(i - 1, j), dmde(i, j), dmde(i - 1, j), &
         u(i, j, k, n_rhs), u(i + 1, j, k, n_rhs), u(i - 1, j, k, n_rhs), &
         v(i, j, k, n_rhs), v(i, j + 1, k, n_rhs), v(i - 1, j, k, n_rhs), &
         v(i - 1, j + 1, k, n_rhs), wrk2a(i, j), wrk2a(i - 1, j), ru(i, j, k)
    flush(6)
  end subroutine dump_uvc_probe

  subroutine dump_uva_probe(tag, k)
    ! Advective U-RHS at P1; wrk2a/wrk2b are UFx/UFe after the flux loops.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: k
    integer(ip) :: i, j, n_rhs
    if (.not. probe_on) return
    if (.not. (k == 1 .or. k == 50 .or. k == 99 .or. k == 100)) return
    i = iprb_pt(1)
    j = jprb_pt(1)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    n_rhs = nrhs
    write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,12es26.17)') 'diag: uvA  i=', i, &
         ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
         '  UFx,UFxW,UFeS,UFeN,uS,u,uN,FvS,FvWS,FvN,FvWN,ru=', &
         wrk2a(i, j), wrk2a(i - 1, j), wrk2b(i, j), wrk2b(i, j + 1), &
         u(i, j - 1, k, n_rhs), u(i, j, k, n_rhs), u(i, j + 1, k, n_rhs), &
         FlxV(i, j, k), FlxV(i - 1, j, k), FlxV(i, j + 1, k), FlxV(i - 1, j + 1, k), &
         ru(i, j, k)
    flush(6)
  end subroutine dump_uva_probe

  subroutine dump_uvh_probe(tag, k)
    ! Halo / raw uxx / UP3 flux / Coriolis at west i=1,2,3 and east i=512.
    ! After cor: wrk2a is the Coriolis UFx. After adv: wrk2a/wrk2b are UFx/UFe.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: k
    integer(ip) :: n, i, j, ii, updated, nx, ny
    real(rp) :: uu(5), um(3), fl(3), xx(3), fxw, fxc, fes, fen, ruv
    logical :: after_cor

    if (.not. probe_on) return
    call sync_arch()
    after_cor = (index(tag, 'cor') > 0)
    nx = decomp%nx
    ny = decomp%ny
    do n = 1, nprb_u
      i = iprb_u(n)
      j = jprb_u(n)
      if (i < 0 .or. i > nx + 1 .or. j < 1 .or. j > ny) cycle
      ! Same faces as CPU istrU:iend (skip west i=1 / east i=nx+1).
      updated = 0
      if (i >= 1 .and. i <= nx + 1) then
        if (.not. ((decomp%edge_w .and. i == 1) .or. (decomp%edge_e .and. i == nx + 1))) &
             updated = 1
      endif
      uu = 0.0_rp; um = -1.0_rp; fl = 0.0_rp; xx = 0.0_rp
      fxw = 0.0_rp; fxc = 0.0_rp; fes = 0.0_rp; fen = 0.0_rp; ruv = 0.0_rp
      do ii = 1, 5
        if (i - 3 + ii >= -1 .and. i - 3 + ii <= nx + 2) &
             uu(ii) = u(i - 3 + ii, j, k, nrhs)
      enddo
      do ii = 1, 3
        if (i - 2 + ii >= 0 .and. i - 2 + ii <= nx + 1) then
          um(ii) = umask(i - 2 + ii, j)
          fl(ii) = FlxU(i - 2 + ii, j, k)
        endif
        if (i - 3 + ii >= -1 .and. i - 1 + ii <= nx + 2) &
             xx(ii) = u(i - 3 + ii, j, k, nrhs) - 2.0_rp * u(i - 2 + ii, j, k, nrhs) + &
                      u(i - 1 + ii, j, k, nrhs)
      enddo
      if (i - 1 >= decomp%is) fxw = wrk2a(i - 1, j)
      if (i >= decomp%is .and. i <= decomp%ie) fxc = wrk2a(i, j)
      if (.not. after_cor) then
        if (i >= decomp%is .and. i <= decomp%ie) fes = wrk2b(i, j)
        if (i >= decomp%is .and. i <= decomp%ie) fen = wrk2b(i, j + 1)
      endif
      ruv = ru(i, j, k)
      write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,i2,a,18es16.8)') 'diag: uvh  i=', i, &
           ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, '  upd=', updated, &
           '  u-2..+2,FlxU-1..+1,uxx-1..+1,fxW,fx,feS,feN,ru=', &
           uu(1), uu(2), uu(3), uu(4), uu(5), fl(1), fl(2), fl(3), &
           xx(1), xx(2), xx(3), fxw, fxc, fes, fen, ruv
      if (.not. after_cor) then
        call dump_ufe_line('ufeS', i, j, j, k, updated)
        call dump_ufe_line('ufeN', i, j, j + 1, k, updated)
      endif
    enddo
    flush(6)
  end subroutine dump_uvh_probe

  subroutine dump_ufe_line(tag, i, jg, jf, k, updated)
    ! UFe stencil at V-face jf for U-point i: u/uee, FlxV(i-2:i+1),
    ! raw Hvxx(i-1:i), the stored UFe, and the UP3 transport cff.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: i, jg, jf, k, updated
    integer(ip) :: ii, nx, ny, n_rhs
    real(rp) :: ul, ur, uee_l, uee_r, fv(-2:1), hv_l, hv_r, ufe, cff

    nx = decomp%nx
    ny = decomp%ny
    n_rhs = nrhs
    ul = 0.0_rp; ur = 0.0_rp; uee_l = 0.0_rp; uee_r = 0.0_rp
    fv = 0.0_rp; hv_l = 0.0_rp; hv_r = 0.0_rp; ufe = 0.0_rp; cff = 0.0_rp
    if (jf - 1 >= decomp%js .and. jf - 1 <= decomp%je) ul = u(i, jf - 1, k, n_rhs)
    if (jf >= decomp%js .and. jf <= decomp%je) ur = u(i, jf, k, n_rhs)
    if (jf - 2 >= decomp%js .and. jf <= decomp%je) &
         uee_l = u(i, jf - 2, k, n_rhs) - 2.0_rp * u(i, jf - 1, k, n_rhs) + u(i, jf, k, n_rhs)
    if (jf - 1 >= decomp%js .and. jf + 1 <= decomp%je) &
         uee_r = u(i, jf - 1, k, n_rhs) - 2.0_rp * u(i, jf, k, n_rhs) + u(i, jf + 1, k, n_rhs)
    do ii = -2, 1
      if (i + ii >= decomp%is .and. i + ii <= decomp%ie .and. &
          jf >= decomp%js .and. jf <= decomp%je) fv(ii) = FlxV(i + ii, jf, k)
    enddo
    hv_l = fv(-2) - 2.0_rp * fv(-1) + fv(0)
    hv_r = fv(-1) - 2.0_rp * fv(0) + fv(1)
    if (i >= decomp%is .and. i <= decomp%ie .and. jf >= decomp%js .and. jf <= decomp%je) &
         ufe = wrk2b(i, jf)
    cff = fv(0) + fv(-1) - uv_delta * (hv_r + hv_l)
    write(*,'(a,a,a,i6,a,i6,a,i8,a,i4,a,i2,a,12es18.10)') 'diag: ', trim(tag), &
         '  i=', i, ' j=', jg, '  after adv  iic=', iic, '  k=', k, '  upd=', updated, &
         '  uL,uR,ueeL,ueeR,Fv-2..+1,Hv-1,Hv,UFe,cff=', &
         ul, ur, uee_l, uee_r, fv(-2), fv(-1), fv(0), fv(1), hv_l, hv_r, ufe, cff
  end subroutine dump_ufe_line

  subroutine dump_vvh_probe(tag, k)
    ! V-face analog of dump_uvh_probe: FlxU / VFx at N/S OBC and first interior.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: k
    integer(ip) :: n, i, j, jj, updated, nx, ny
    real(rp) :: vv(5), vm(3), fl(3), xx(3), fxs, fxc, few, fee, rvv

    if (.not. probe_on) return
    call sync_arch()
    nx = decomp%nx
    ny = decomp%ny
    do n = 1, nprb_v
      i = iprb_v(n)
      j = jprb_v(n)
      if (i < 1 .or. i > nx .or. j < 1 .or. j > ny + 1) cycle
      updated = 0
      if (.not. ((decomp%edge_s .and. j == 1) .or. (decomp%edge_n .and. j == ny + 1))) &
           updated = 1
      vv = 0.0_rp; vm = -1.0_rp; fl = 0.0_rp; xx = 0.0_rp
      fxs = 0.0_rp; fxc = 0.0_rp; few = 0.0_rp; fee = 0.0_rp; rvv = 0.0_rp
      do jj = 1, 5
        if (j - 3 + jj >= 0 .and. j - 3 + jj <= ny + 2) &
             vv(jj) = v(i, j - 3 + jj, k, nrhs)
      enddo
      do jj = 1, 3
        if (j - 2 + jj >= 0 .and. j - 2 + jj <= ny + 1) then
          vm(jj) = vmask(i, j - 2 + jj)
          fl(jj) = FlxU(i, j - 2 + jj, k)
        endif
        if (j - 3 + jj >= 0 .and. j - 1 + jj <= ny + 2) &
             xx(jj) = v(i, j - 3 + jj, k, nrhs) - 2.0_rp * v(i, j - 2 + jj, k, nrhs) + &
                      v(i, j - 1 + jj, k, nrhs)
      enddo
      if (j - 1 >= decomp%js) fxs = wrk2d(i, j - 1)
      if (j >= decomp%js .and. j <= decomp%je) fxc = wrk2d(i, j)
      if (i >= decomp%is .and. i <= decomp%ie .and. j >= decomp%js .and. j <= decomp%je) &
           few = wrk2c(i, j)
      if (i + 1 >= decomp%is .and. i + 1 <= decomp%ie .and. &
          j >= decomp%js .and. j <= decomp%je) fee = wrk2c(i + 1, j)
      rvv = rv(i, j, k)
      write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,i2,a,18es16.8)') 'diag: vvh  i=', i, &
           ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, '  upd=', updated, &
           '  v-2..+2,FlxU-1..+1,vxx-1..+1,feS,fe,fxW,fxE,rv=', &
           vv(1), vv(2), vv(3), vv(4), vv(5), fl(1), fl(2), fl(3), &
           xx(1), xx(2), xx(3), fxs, fxc, few, fee, rvv
      call dump_vfx_line('vfxW', i, j, i, j, k, updated)
      call dump_vfx_line('vfxE', i, j, i + 1, j, k, updated)
    enddo
    flush(6)
  end subroutine dump_vvh_probe

  subroutine dump_vfx_line(tag, ig, jg, i, j, k, updated)
    ! VFx stencil at U-face i for V-point j: v/vxx, FlxU(i, j-2:j+1).
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: ig, jg, i, j, k, updated
    integer(ip) :: jj, n_rhs
    real(rp) :: vl, vr, vee_l, vee_r, fu(-2:1), hu_l, hu_r, vfx, cff

    n_rhs = nrhs
    vl = 0.0_rp; vr = 0.0_rp; vee_l = 0.0_rp; vee_r = 0.0_rp
    fu = 0.0_rp; hu_l = 0.0_rp; hu_r = 0.0_rp; vfx = 0.0_rp; cff = 0.0_rp
    if (i - 1 >= decomp%is .and. i - 1 <= decomp%ie .and. &
        j >= decomp%js .and. j <= decomp%je) vl = v(i - 1, j, k, n_rhs)
    if (i >= decomp%is .and. i <= decomp%ie .and. &
        j >= decomp%js .and. j <= decomp%je) vr = v(i, j, k, n_rhs)
    if (i - 2 >= decomp%is .and. i <= decomp%ie .and. &
        j >= decomp%js .and. j <= decomp%je) &
         vee_l = v(i - 2, j, k, n_rhs) - 2.0_rp * v(i - 1, j, k, n_rhs) + v(i, j, k, n_rhs)
    if (i - 1 >= decomp%is .and. i + 1 <= decomp%ie .and. &
        j >= decomp%js .and. j <= decomp%je) &
         vee_r = v(i - 1, j, k, n_rhs) - 2.0_rp * v(i, j, k, n_rhs) + v(i + 1, j, k, n_rhs)
    do jj = -2, 1
      if (i >= decomp%is .and. i <= decomp%ie .and. &
          j + jj >= decomp%js .and. j + jj <= decomp%je) fu(jj) = FlxU(i, j + jj, k)
    enddo
    hu_l = fu(-2) - 2.0_rp * fu(-1) + fu(0)
    hu_r = fu(-1) - 2.0_rp * fu(0) + fu(1)
    if (i >= decomp%is .and. i <= decomp%ie .and. j >= decomp%js .and. j <= decomp%je) &
         vfx = wrk2c(i, j)
    cff = fu(0) + fu(-1) - uv_delta * (hu_r + hu_l)
    write(*,'(a,a,a,i6,a,i6,a,i8,a,i4,a,i2,a,12es18.10)') 'diag: ', trim(tag), &
         '  i=', ig, ' j=', jg, '  after adv  iic=', iic, '  k=', k, '  upd=', updated, &
         '  vL,vR,veeL,veeR,Fu-2..+1,Hu-1,Hu,VFx,cff=', &
         vl, vr, vee_l, vee_r, fu(-2), fu(-1), fu(0), fu(1), hu_l, hu_r, vfx, cff
  end subroutine dump_vfx_line

  subroutine add_vert_uv_rhs(j0, j1)
    ! Vertical advection (SPLINE_UV) of u and v for rows j0:j1, one device
    ! thread per column (the k-recurrences stay serial in the thread); the
    ! u pass over all rows runs before the v pass, as the col_* scratch is
    ! reused per column. Was called once per row.
    ! The forward sweep stores only what the backward sweep reads (col_cf,
    ! col_wc); DC and the back-substituted FC of the previous level stay in
    ! registers. Probe steps (keep) also store the full col_dc/col_wc/col_cf
    ! columns that dump_vert_b prints. Same operations in the same order.
    integer(ip), intent(in) :: j0, j1
    integer(ip) :: i, k, nx, nz, jj, n_rhs, i_u0, i_u1, j_v0, j_v1, jv0, jv1, jp, lim
    real(rp) :: cff, dc_k, dc_kp, cf_k, wc_km1, wc_k, wc_p, dc_p
    logical :: keep

    nx = decomp%nx
    nz = decomp%nz
    n_rhs = nrhs
    i_u0 = 1
    if (decomp%edge_w) i_u0 = 2
    i_u1 = nx + 1
    if (decomp%edge_e) i_u1 = nx
    lim = dry_lim(G_UVRHS)
    keep = probe_on

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do jj = j0, j1
    do i = i_u0, i_u1
      if (lim > 0) then; if (colact(i, jj) == 0) cycle; end if
      dc_k = 0.5625_rp * (Hz(i, jj, 1) + Hz(i - 1, jj, 1)) - &
             0.0625_rp * (Hz(i + 1, jj, 1) + Hz(i - 2, jj, 1))
      cf_k = 1.0_rp
      wc_km1 = 2.0_rp * u(i, jj, 1, n_rhs)
      if (keep) then
        col_cf(i, jj, 1) = cf_k
        col_wc(i, jj, 0) = wc_km1
      endif
      do k = 1, nz - 1
        dc_kp = 0.5625_rp * (Hz(i, jj, k + 1) + Hz(i - 1, jj, k + 1)) - &
                0.0625_rp * (Hz(i + 1, jj, k + 1) + Hz(i - 2, jj, k + 1))
        cff = 1.0_rp / (2.0_rp * dc_k + dc_kp * (2.0_rp - cf_k))
        cf_k = cff * dc_k
        col_cf(i, jj, k + 1) = cf_k
        wc_k = cff * (3.0_rp * (dc_k * u(i, jj, k + 1, n_rhs) + &
             dc_kp * u(i, jj, k, n_rhs)) - dc_kp * wc_km1)
        col_wc(i, jj, k) = wc_k
        wc_km1 = wc_k
        dc_k = dc_kp
      enddo
      wc_p = (2.0_rp * u(i, jj, nz, n_rhs) - wc_km1) / &
           (1.0_rp - cf_k)
      dc_p = 0.0_rp
      if (keep) then
        col_wc(i, jj, nz) = wc_p
        col_dc(i, jj, nz) = dc_p
      endif
      do k = nz - 1, 1, -1
        wc_k = col_wc(i, jj, k) - col_cf(i, jj, k + 1) * wc_p
        dc_k = wc_k * 0.5_rp * (We(i, jj, k) + We(i - 1, jj, k) - 0.125_rp * ( &
             (We(i + 1, jj, k) - We(i, jj, k)) * umask(i + 1, jj) - &
             (We(i - 1, jj, k) - We(i - 2, jj, k)) * umask(i - 1, jj)))
        ru(i, jj, k + 1) = ru(i, jj, k + 1) - dc_p + dc_k
        if (keep) then
          col_wc(i, jj, k) = wc_k
          col_dc(i, jj, k) = dc_k
        endif
        wc_p = wc_k
        dc_p = dc_k
      enddo
      ru(i, jj, 1) = ru(i, jj, 1) - dc_p
    enddo
    enddo
    jp = jprb_pt(4)
    if (probe_on .and. jp >= j0 .and. jp <= j1) call dump_vert_b('u', jp)

    j_v0 = 1
    if (decomp%edge_s) j_v0 = 2
    j_v1 = decomp%ny + 1
    if (decomp%edge_n) j_v1 = decomp%ny
    jv0 = max(j0, j_v0)
    jv1 = min(j1, j_v1)
    if (jv0 <= jv1) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do jj = jv0, jv1
      do i = 1, nx
        if (lim > 0) then; if (colact(i, jj) == 0) cycle; end if
        dc_k = 0.5625_rp * (Hz(i, jj, 1) + Hz(i, jj - 1, 1)) - &
               0.0625_rp * (Hz(i, jj + 1, 1) + Hz(i, jj - 2, 1))
        cf_k = 1.0_rp
        wc_km1 = 2.0_rp * v(i, jj, 1, n_rhs)
        if (keep) then
          col_cf(i, jj, 1) = cf_k
          col_wc(i, jj, 0) = wc_km1
        endif
        do k = 1, nz - 1
          dc_kp = 0.5625_rp * (Hz(i, jj, k + 1) + Hz(i, jj - 1, k + 1)) - &
                  0.0625_rp * (Hz(i, jj + 1, k + 1) + Hz(i, jj - 2, k + 1))
          cff = 1.0_rp / (2.0_rp * dc_k + dc_kp * (2.0_rp - cf_k))
          cf_k = cff * dc_k
          col_cf(i, jj, k + 1) = cf_k
          wc_k = cff * (3.0_rp * (dc_k * v(i, jj, k + 1, n_rhs) + &
               dc_kp * v(i, jj, k, n_rhs)) - dc_kp * wc_km1)
          col_wc(i, jj, k) = wc_k
          wc_km1 = wc_k
          dc_k = dc_kp
        enddo
        wc_p = (2.0_rp * v(i, jj, nz, n_rhs) - wc_km1) / &
             (1.0_rp - cf_k)
        dc_p = 0.0_rp
        if (keep) then
          col_wc(i, jj, nz) = wc_p
          col_dc(i, jj, nz) = dc_p
        endif
        do k = nz - 1, 1, -1
          wc_k = col_wc(i, jj, k) - col_cf(i, jj, k + 1) * wc_p
          dc_k = wc_k * 0.5_rp * (We(i, jj, k) + We(i, jj - 1, k) - 0.125_rp * ( &
               (We(i, jj + 1, k) - We(i, jj, k)) * vmask(i, jj + 1) - &
               (We(i, jj - 1, k) - We(i, jj - 2, k)) * vmask(i, jj - 1)))
          rv(i, jj, k + 1) = rv(i, jj, k + 1) - dc_p + dc_k
          if (keep) then
            col_wc(i, jj, k) = wc_k
            col_dc(i, jj, k) = dc_k
          endif
          wc_p = wc_k
          dc_p = dc_k
        enddo
        rv(i, jj, 1) = rv(i, jj, 1) - dc_p
      enddo
      enddo
      if (probe_on .and. jp >= jv0 .and. jp <= jv1) call dump_vert_b('v', jp)
    endif
  end subroutine add_vert_uv_rhs

  logical function rhs_b_k(k)
    integer(ip), intent(in) :: k
    rhs_b_k = .true.
  end function rhs_b_k

  subroutine dump_cor_b(k)
    ! Coriolis + curvature increment at family B (154,136). wrk2a is UFx,
    ! wrk2d is VFe.
    integer(ip), intent(in) :: k
    integer(ip) :: i, j, n_rhs
    real(rp) :: fomn, fomnW, fomnS

    if (.not. probe_on) return
    if (.not. rhs_b_k(k)) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    n_rhs = nrhs
    fomn = f(i, j) / max(pm(i, j) * pn(i, j), 1.0e-30_rp)
    fomnW = f(i - 1, j) / max(pm(i - 1, j) * pn(i - 1, j), 1.0e-30_rp)
    fomnS = f(i, j - 1) / max(pm(i, j - 1) * pn(i, j - 1), 1.0e-30_rp)
    write(*,'(a,i8,a,i4,a,18es26.17)') 'diag: cor B u  iic=', iic, '  k=', k, &
         '  Hz,HzW,fomn,fomnW,dndx,dndxW,dmde,dmdeW,u,uE,uW,v,vN,vW,vWN,UFx,UFxW,ru=', &
         Hz(i, j, k), Hz(i - 1, j, k), fomn, fomnW, &
         dndx(i, j), dndx(i - 1, j), dmde(i, j), dmde(i - 1, j), &
         u(i, j, k, n_rhs), u(i + 1, j, k, n_rhs), u(i - 1, j, k, n_rhs), &
         v(i, j, k, n_rhs), v(i, j + 1, k, n_rhs), v(i - 1, j, k, n_rhs), &
         v(i - 1, j + 1, k, n_rhs), wrk2a(i, j), wrk2a(i - 1, j), ru(i, j, k)
    write(*,'(a,i8,a,i4,a,18es26.17)') 'diag: cor B v  iic=', iic, '  k=', k, &
         '  Hz,HzS,fomn,fomnS,dndx,dndxS,dmde,dmdeS,u,uE,uS,uSE,v,vN,vS,VFe,VFeS,rv=', &
         Hz(i, j, k), Hz(i, j - 1, k), fomn, fomnS, &
         dndx(i, j), dndx(i, j - 1), dmde(i, j), dmde(i, j - 1), &
         u(i, j, k, n_rhs), u(i + 1, j, k, n_rhs), u(i, j - 1, k, n_rhs), &
         u(i + 1, j - 1, k, n_rhs), v(i, j, k, n_rhs), v(i, j + 1, k, n_rhs), &
         v(i, j - 1, k, n_rhs), wrk2d(i, j), wrk2d(i, j - 1), rv(i, j, k)
    flush(6)
  end subroutine dump_cor_b

  subroutine dump_adv_b(k)
    ! Horizontal UP3 fluxes at family B. wrk2a/wrk2b are UFx/UFe,
    ! wrk2c/wrk2d are VFx/VFe.
    integer(ip), intent(in) :: k
    integer(ip) :: i, j, n_rhs

    if (.not. probe_on) return
    if (.not. rhs_b_k(k)) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    n_rhs = nrhs
    write(*,'(a,i8,a,i4,a,12es26.17)') 'diag: adv B u  iic=', iic, '  k=', k, &
         '  UFx,UFxW,UFeS,UFeN,uS,u,uN,FvS,FvWS,FvN,FvWN,ru=', &
         wrk2a(i, j), wrk2a(i - 1, j), wrk2b(i, j), wrk2b(i, j + 1), &
         u(i, j - 1, k, n_rhs), u(i, j, k, n_rhs), u(i, j + 1, k, n_rhs), &
         FlxV(i, j, k), FlxV(i - 1, j, k), FlxV(i, j + 1, k), FlxV(i - 1, j + 1, k), &
         ru(i, j, k)
    write(*,'(a,i8,a,i4,a,12es26.17)') 'diag: adv B v  iic=', iic, '  k=', k, &
         '  VFxW,VFxE,VFeS,VFeN,vW,v,vE,FuW,FuWS,FuE,FuES,rv=', &
         wrk2c(i, j), wrk2c(i + 1, j), wrk2d(i, j - 1), wrk2d(i, j), &
         v(i - 1, j, k, n_rhs), v(i, j, k, n_rhs), v(i + 1, j, k, n_rhs), &
         FlxU(i, j, k), FlxU(i, j - 1, k), FlxU(i + 1, j, k), FlxU(i + 1, j - 1, k), &
         rv(i, j, k)
    flush(6)
  end subroutine dump_adv_b

  subroutine dump_vert_b(face, j)
    ! SPLINE_UV column at family B. col_wc is FC, col_dc is W*U / W*V.
    character(len=*), intent(in) :: face
    integer(ip), intent(in) :: j
    integer(ip) :: i, k, nz, n_rhs
    real(rp) :: hzw, wew, vel, ruv, dcm

    if (.not. probe_on) return
    if (j /= jprb_pt(4)) return
    call sync_arch()
    i = iprb_pt(4)
    if (i < 1 .or. i > decomp%nx) return
    nz = decomp%nz
    n_rhs = nrhs
    do k = 1, nz
      dcm = 0.0_rp
      if (k > 1) dcm = col_dc(i, j, k - 1)
      if (trim(face) == 'u') then
        hzw = 0.5625_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) - &
              0.0625_rp * (Hz(i + 1, j, k) + Hz(i - 2, j, k))
        wew = 0.5_rp * (We(i, j, k) + We(i - 1, j, k) - 0.125_rp * ( &
             (We(i + 1, j, k) - We(i, j, k)) * umask(i + 1, j) - &
             (We(i - 1, j, k) - We(i - 2, j, k)) * umask(i - 1, j)))
        vel = u(i, j, k, n_rhs)
        ruv = ru(i, j, k)
      else
        hzw = 0.5625_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) - &
              0.0625_rp * (Hz(i, j + 1, k) + Hz(i, j - 2, k))
        wew = 0.5_rp * (We(i, j, k) + We(i, j - 1, k) - 0.125_rp * ( &
             (We(i, j + 1, k) - We(i, j, k)) * vmask(i, j + 1) - &
             (We(i, j - 1, k) - We(i, j - 2, k)) * vmask(i, j - 1)))
        vel = v(i, j, k, n_rhs)
        ruv = rv(i, j, k)
      endif
      write(*,'(a,a,a,i8,a,i4,a,8es26.17)') 'diag: vert B ', trim(face), &
           '  iic=', iic, '  k=', k, '  ru,u_rhs,HzW,WeW,FC,DC,DCm,FC0=', &
           ruv, vel, hzw, wew, col_wc(i, j, k), col_dc(i, j, k), dcm, col_wc(i, j, 0)
    enddo
    flush(6)
  end subroutine dump_vert_b

  subroutine dump_rv_pred(tag)
    ! Predictor rv at P1 last interior V-face and north OBC face.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, jj, js(2)
    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(1)
    if (i < 1 .or. i > decomp%nx) return
    js = (/ decomp%ny, decomp%ny + 1 /)
    do jj = 1, 2
      j = js(jj)
      do k = 1, decomp%nz
        if (k == 1 .or. k == 50 .or. k == 99 .or. k == 100) then
          write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,es26.17)') 'diag: rvP  i=', i, &
               ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, '  rv=', rv(i, j, k)
        endif
      enddo
    enddo
    flush(6)
  end subroutine dump_rv_pred

  subroutine dump_vn3(tag)
    ! Predictor v slots + rv at P1 neighbors i-1:i+2, last interior V
    ! and north OBC. Distinguishes horiz RHS vs implicit visc vs v3dbc.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k
    if (.not. probe_on) return
    call sync_arch()
    do i = iprb_pt(1) - 1, iprb_pt(1) + 2
      if (i < 1 .or. i > decomp%nx) cycle
      do j = jprb_pt(1), jprb_pt(1) + 1
        if (j < 1 .or. j > decomp%ny + 1) cycle
        do k = 1, decomp%nz
          if (k == 1 .or. k == 50 .or. k == 99 .or. k == 100) then
            write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,8es26.17)') 'diag: vn3  i=', i, &
                 ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
                 '  v_stp,v_rhs,v_new,rv,vmask,v_bak,Hz,HzS=', &
                 v(i, j, k, nstp), v(i, j, k, nrhs), v(i, j, k, nnew), rv(i, j, k), &
                 vmask(i, j), v(i, j, k, 3_ip - nstp), Hz(i, j, k), Hz(i, j - 1, k)
          endif
        enddo
      enddo
    enddo
    flush(6)
  end subroutine dump_vn3

  subroutine dump_vim(tag)
    ! Implicit V-column inputs at (318,432) and (317,432). Call before
    ! the V AM3 kernel so v(indx) is still the bak slot.
    character(len=*), intent(in) :: tag
    integer(ip) :: ii, i, j, k, ibak, ks(4), igs(2)
    if (.not. probe_on) return
    call sync_arch()
    ibak = 3 - nstp
    igs = (/ iprb_pt(1), iprb_pt(1) - 1 /)
    j = jprb_pt(1)
    if (igs(1) < 1 .or. igs(1) > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    ks = (/ 1, 50, 99, 100 /)
    do ii = 1, 2
      i = igs(ii)
      do k = 1, 4
        if (ks(k) < 1 .or. ks(k) > decomp%nz) cycle
        write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,13es26.17)') 'diag: vim  i=', i, &
             ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', ks(k), &
             '  HzB,HzBS,HzF,HzFS,Akv,AkvS,Wi,WiS,rd,rdS,rv,v_stp,v_bak=', &
             Hz_bak(i, j, ks(k)), Hz_bak(i, j - 1, ks(k)), &
             Hz_fwd(i, j, ks(k)), Hz_fwd(i, j - 1, ks(k)), &
             Akv(i, j, ks(k)), Akv(i, j - 1, ks(k)), &
             Wi(i, j, ks(k)), Wi(i, j - 1, ks(k)), &
             r_D(i, j), r_D(i, j - 1), rv(i, j, ks(k)), &
             v(i, j, ks(k), nstp), v(i, j, ks(k), ibak)
      enddo
    enddo
    flush(6)
  end subroutine dump_vim

  subroutine dump_vdc(tag)
    ! col_dc after V AM3+rv, before Thomas elimination.
    character(len=*), intent(in) :: tag
    integer(ip) :: ii, i, j, k, ks(4), igs(2)
    if (.not. probe_on) return
    call sync_arch()
    igs = (/ iprb_pt(1), iprb_pt(1) - 1 /)
    j = jprb_pt(1)
    if (igs(1) < 1 .or. igs(1) > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    ks = (/ 1, 50, 99, 100 /)
    do ii = 1, 2
      i = igs(ii)
      do k = 1, 4
        if (ks(k) < 1 .or. ks(k) > decomp%nz) cycle
        write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,3es26.17)') 'diag: vdc  i=', i, &
             ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', ks(k), &
             '  DC0,DC,rv=', col_dc(i, j, 0), col_dc(i, j, ks(k)), rv(i, j, ks(k))
      enddo
    enddo
    flush(6)
  end subroutine dump_vdc

  subroutine dump_vvs(tag)
    ! SPLINE_UV V-stencil at (318,432)/(317,432): Hz/We/v(nrhs)/vmask at j-2:j+1.
    character(len=*), intent(in) :: tag
    integer(ip) :: ii, i, j, k, n_rhs, ks(4), igs(4)
    if (.not. probe_on) return
    call sync_arch()
    n_rhs = nrhs
    igs = (/ iprb_pt(1) - 1, iprb_pt(1), iprb_pt(1) + 1, iprb_pt(1) + 2 /)
    j = jprb_pt(1)
    if (j < 1 .or. j > decomp%ny) return
    ks = (/ 1, 50, 99, 100 /)
    do ii = 1, 4
      i = igs(ii)
      if (i < 1 .or. i > decomp%nx) cycle
      do k = 1, 4
        if (ks(k) < 1 .or. ks(k) > decomp%nz) cycle
        write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,15es26.17)') 'diag: vvs  i=', i, &
             ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', ks(k), &
             '  Hz2,Hz1,Hz0,HzN,We2,We1,We0,WeN,v2,v1,v0,vN,vm1,vm0,vmN=', &
             Hz(i, j - 2, ks(k)), Hz(i, j - 1, ks(k)), Hz(i, j, ks(k)), Hz(i, j + 1, ks(k)), &
             We(i, j - 2, ks(k)), We(i, j - 1, ks(k)), We(i, j, ks(k)), We(i, j + 1, ks(k)), &
             v(i, j - 2, ks(k), n_rhs), v(i, j - 1, ks(k), n_rhs), &
             v(i, j, ks(k), n_rhs), v(i, j + 1, ks(k), n_rhs), &
             vmask(i, j - 1), vmask(i, j), vmask(i, j + 1)
      enddo
    enddo
    flush(6)
  end subroutine dump_vvs

  subroutine dump_vvf(tag)
    ! Hz/We interpolants, spline FC (col_wc), W*V flux DC (col_dc), rv after vert.
    character(len=*), intent(in) :: tag
    integer(ip) :: ii, i, j, k, kk, ks(4), igs(4)
    real(rp) :: HzW, WeW
    if (.not. probe_on) return
    call sync_arch()
    igs = (/ iprb_pt(1) - 1, iprb_pt(1), iprb_pt(1) + 1, iprb_pt(1) + 2 /)
    j = jprb_pt(1)
    if (j < 1 .or. j > decomp%ny) return
    ks = (/ 1, 50, 99, 100 /)
    do ii = 1, 4
      i = igs(ii)
      if (i < 1 .or. i > decomp%nx) cycle
      do kk = 1, 4
        k = ks(kk)
        if (k < 1 .or. k > decomp%nz) cycle
        HzW = 0.5625_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) - &
              0.0625_rp * (Hz(i, j + 1, k) + Hz(i, j - 2, k))
        WeW = 0.5_rp * (We(i, j, k) + We(i, j - 1, k) - 0.125_rp * ( &
             (We(i, j + 1, k) - We(i, j, k)) * vmask(i, j + 1) - &
             (We(i, j - 1, k) - We(i, j - 2, k)) * vmask(i, j - 1)))
        write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,5es26.17)') 'diag: vvf  i=', i, &
             ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
             '  HzW,WeW,FC,DC,rv=', HzW, WeW, col_wc(i, j, k), col_dc(i, j, k), rv(i, j, k)
      enddo
    enddo
    flush(6)
  end subroutine dump_vvf

end module uv_rhs

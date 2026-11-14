#include "mem_attr.inc"
module bgc_window_mod
  ! Kernels of the BGC subcycling window (bgc_subcycle_mod), mirroring
  ! UCLA ROMS bgc_subcycle_mod (dev 5c8034bf, #428/#430) routine by routine
  ! with the CPU's expressions and operand order (the practice of
  ! passive_column.F90), on the tracer-innermost passive layout
  ! tp(ntr, 0:ntpcol, nz, 3) (gpu_state, tpcol):
  !   stats   per-level Courant numbers, sub-pass criterion, explicit
  !           diffusion and vertical Courant numbers (bgcsub_window_tile's
  !           first loops), row maxima then level maxima;
  !   H1      per-level coefficients of a sub-pass (hcoef): pseudo-volumes,
  !           swept fractions, inner-operator denominators;
  !   H2      inner swept fluxes and the inner operators sx, sy (hlevel,
  !           sweep_x/sweep_y) on the cross-shaped region (gpu_state xc_*):
  !           one thread per (tracer, column, level block);
  !   H3      outer swept fluxes of sx, sy, TS_DIF2 diffusion of the
  !           window-start state (sub-pass 1), the update of the content
  !           per area (hupdate): one thread per (tracer, interior wet
  !           column, level block);
  !   factor  the implicit-diffusion factors with the window-integrated
  !           Akt and the window-end Hz (bgcsub_window_tile's vertical part);
  !   vertical  1-D swept flux (vsweep), surface sources, implicit solve
  !           (vpass): one thread per (tracer, interior wet column);
  !   t3dbc   bgcsub_t3dbc;
  !   red     per-tracer reductions of the diagnostics (fixed order).
  ! A face shared by two cells is evaluated by both of their threads with
  ! identical operands (as up3_kernel), which -Mnofma keeps bitwise equal.
  ! Separate from bgc_subcycle_mod so the explicit kernel launches are
  ! never inlined into their callers (passive_column.F90 header).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private
  public :: WLEV, bgcw_stats, bgcw_hcoef, bgcw_inner, bgcw_outer, bgcw_factor, bgcw_vertical
  public :: bgcw_t3dbc, bgcw_reduce, bgcw_vs, bgcw_edge_r, bgcw_subq, bgcw_edge_tp
  ! Levels per horizontal launch (the scratch of the inner operators and
  ! the coefficients hold WLEV levels).
  integer(ip), parameter :: WLEV = 20
  real(rp), parameter :: r6 = 1.0_rp / 6.0_rp

contains

  ! ------------------------------------------------------------ face flux
  DEVICE_ATTR
  function swf(vv, c, sll, sl, sr, srr, ml, m0, mr, hl, hr) result(ff)
    ! sweep_x/sweep_y for one face between cells L (sl) and R (sr): the
    ! UP3 parabola of the upstream cell integrated over the swept part,
    ! F = V [q_u +- (1-c)((2-c) d_near + (1+c) d_far)/6], with the masked
    ! differences d = (s_R - s_L)*mask (UPSTREAM_TS_LAND_CURV: a masked far
    ! difference is replaced by the near one; hl/hr = the far difference is
    ! inside the valid range, else the near one is used).
    real(rp), intent(in) :: vv, c, sll, sl, sr, srr, ml, m0, mr
    logical, intent(in) :: hl, hr
    real(rp) :: ff, dn, df
    dn = (sr - sl) * m0
    if (vv > 0.0_rp) then
      if (hl) then
        df = (sl - sll) * ml
        if (ml < 0.5_rp) df = dn
      else
        df = dn
      endif
      ff = vv * (sl + (1.0_rp - c) * ((2.0_rp - c) * dn + (1.0_rp + c) * df) * r6)
    else if (vv < 0.0_rp) then
      if (hr) then
        df = (srr - sr) * mr
        if (mr < 0.5_rp) df = dn
      else
        df = dn
      endif
      ff = vv * (sr - (1.0_rp - c) * ((2.0_rp - c) * dn + (1.0_rp + c) * df) * r6)
    else
      ff = 0.0_rp
    endif
  end function swf

  DEVICE_ATTR
  function qwin(l, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk) result(qv)
    ! the vertical pass's q(l) = c(l)/v1(l) of one tracer column at face m:
    ! levels m-1..m+2 from the register window qa..qd, others from the
    ! content cio (flat offset it0 + tk (l-1)) and v1 (g0 + gk (l-1)), with
    ! the same division and rounding as the window's
    integer(ip), intent(in) :: l, m
    real(rp), intent(in) :: qa, qb, qc, qd
    real(pp), intent(in) :: cio(*)
    real(rp), intent(in) :: wv1(*)
    integer(8), intent(in) :: it0, tk, g0, gk
    real(rp) :: qv
    if (l == m) then
      qv = qb
    else if (l == m + 1) then
      qv = qc
    else if (l == m - 1) then
      qv = qa
    else if (l == m + 2) then
      qv = qd
    else
      qv = real(real(real(cio(it0 + tk * int(l - 1, 8)), rp) / wv1(g0 + gk * int(l - 1, 8)), pp), rp)
    endif
  end function qwin

  DEVICE_ATTR
  function vwin(l, m, vb, vc, wv1, g0, gk) result(vv)
    ! the vertical pass's v1(l) at face m: levels m, m+1 from the register
    ! window vb, vc, others from wv1 (flat offset g0 + gk (l-1))
    integer(ip), intent(in) :: l, m
    real(rp), intent(in) :: vb, vc
    real(rp), intent(in) :: wv1(*)
    integer(8), intent(in) :: g0, gk
    real(rp) :: vv
    if (l == m) then
      vv = vb
    else if (l == m + 1) then
      vv = vc
    else
      vv = wv1(g0 + gk * int(l - 1, 8))
    endif
  end function vwin

#ifdef USE_CUDA
  ! ---------------------------------------------------------------- stats
  attributes(global) launch_bounds(128, 8) subroutine stats_kernel(nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, &
      VX, VY, WA, HZS, pm_, pn_, rmask_, pmon_u_, pnom_v_, sp_, tnumax, dtw, cols)
    ! one thread per (column i = 0..nx+1, level k), consecutive threads on
    ! consecutive i: the maxima over the rows j of cface, c1d, cout, -v1min,
    ! crit (rows 0..ny+1, columns 0..nx+1) and of the explicit diffusion
    ! number and the vertical Courant number (interior). Maxima are exact,
    ! so the order of the cells does not matter.
    integer(ip), value :: nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi
    real(rp), value :: tnumax, dtw
    real(rp), device, intent(in) :: VX(is:ie, js:je, nz), VY(is:ie, js:je, nz), WA(is:ie, js:je, 0:nz), HZS(is:ie, js:je, nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), device, intent(in) :: pmon_u_(is:ie, js:je), pnom_v_(is:ie, js:je), sp_(is:ie, js:je)
    real(rp), device, intent(inout) :: cols(7, 0:nx + 1, nz)
    integer(ip) :: tid, i, j, k, ncol
    real(rp) :: cface, c1d, cout, v1min, crit, rv, cff, dvx, dvy, cfc, ccell, smx, wmx
    tid = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    ncol = nx + 2
    if (tid > ncol * nz) return
    i = mod(tid - 1, ncol); k = (tid - 1) / ncol + 1
    cface = 0.0_rp; c1d = 0.0_rp; cout = 0.0_rp; v1min = 1.0_rp; crit = 0.0_rp
    smx = 0.0_rp; wmx = 0.0_rp
    do j = 0, ny + 1
      if (rmask_(i, j) < 0.5_rp) cycle
      rv = pm_(i, j) * pn_(i, j) / HZS(i, j, k)
      cff = 0.0_rp; dvx = 0.0_rp; dvy = 0.0_rp; cfc = 0.0_rp
      if (i + 1 <= ihi) then
        cff = cff + max(VX(i + 1, j, k), 0.0_rp); dvx = dvx + VX(i + 1, j, k)
        cfc = max(cfc, max(VX(i + 1, j, k), 0.0_rp) * rv)
      endif
      if (i > ilo) then
        cff = cff - min(VX(i, j, k), 0.0_rp); dvx = dvx - VX(i, j, k)
        cfc = max(cfc, -min(VX(i, j, k), 0.0_rp) * rv)
      endif
      if (j + 1 <= jhi) then
        cff = cff + max(VY(i, j + 1, k), 0.0_rp); dvy = dvy + VY(i, j + 1, k)
        cfc = max(cfc, max(VY(i, j + 1, k), 0.0_rp) * rv)
      endif
      if (j > jlo) then
        cff = cff - min(VY(i, j, k), 0.0_rp); dvy = dvy - VY(i, j, k)
        cfc = max(cfc, -min(VY(i, j, k), 0.0_rp) * rv)
      endif
      cface = max(cface, cfc)
      c1d = max(c1d, dvx * rv, dvy * rv)
      cout = max(cout, cff * rv)
      if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) v1min = min(v1min, 1.0_rp - (dvx + dvy) * rv)
      ccell = max(cfc, dvx * rv, dvy * rv)
      crit = max(crit, ccell / max(min(1.0_rp, 1.0_rp - (dvx + dvy) * rv), 1.0e-3_rp))
    enddo
    if (i >= 1 .and. i <= nx) then
      ! explicit diffusion number with the largest diffusivity of the BGC
      ! tracers, dmx = max(tnu2) + sponge (= the CPU's maxval of diff2:
      ! rounded addition is monotonic)
      do j = 1, ny
        cff = dtw * pm_(i, j) * pn_(i, j) / HZS(i, j, k) * 0.25_rp * ( &
              ((tnumax + sp_(i + 1, j)) + (tnumax + sp_(i, j))) * pmon_u_(i + 1, j) * (HZS(i + 1, j, k) + HZS(i, j, k)) &
            + ((tnumax + sp_(i, j)) + (tnumax + sp_(i - 1, j))) * pmon_u_(i, j) * (HZS(i, j, k) + HZS(i - 1, j, k)) &
            + ((tnumax + sp_(i, j + 1)) + (tnumax + sp_(i, j))) * pnom_v_(i, j + 1) * (HZS(i, j + 1, k) + HZS(i, j, k)) &
            + ((tnumax + sp_(i, j)) + (tnumax + sp_(i, j - 1))) * pnom_v_(i, j) * (HZS(i, j, k) + HZS(i, j - 1, k)) )
        smx = max(smx, cff * rmask_(i, j))
        if (k <= nz - 1) wmx = max(wmx, abs(WA(i, j, k)) * pm_(i, j) * pn_(i, j) / min(HZS(i, j, k), HZS(i, j, k + 1)) &
                                        * rmask_(i, j))
      enddo
    endif
    cols(1, i, k) = cface; cols(2, i, k) = c1d; cols(3, i, k) = cout; cols(4, i, k) = -v1min
    cols(5, i, k) = crit; cols(6, i, k) = smx; cols(7, i, k) = wmx
  end subroutine stats_kernel

  attributes(global) launch_bounds(128, 8) subroutine stats_level_kernel(nx, nz, cols, lck)
    ! one thread per (quantity, level): the maximum over the columns (exact)
    integer(ip), value :: nx, nz
    real(rp), device, intent(in) :: cols(7, 0:nx + 1, nz)
    real(rp), device, intent(inout) :: lck(7, nz)
    integer(ip) :: tid, q, k, i
    real(rp) :: m
    tid = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (tid > 7 * nz) return
    q = mod(tid - 1, 7) + 1; k = (tid - 1) / 7 + 1
    m = cols(q, 0, k)
    do i = 1, nx + 1
      m = max(m, cols(q, i, k))
    enddo
    lck(q, k) = m
  end subroutine stats_level_kernel

  ! ------------------------------------------------- H2: inner operators
  attributes(global) launch_bounds(128, 4) subroutine inner_kernel(nxc, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, &
      ilo, ihi, jlo, jhi, west, east, south, north, ncl, xcol, xi, xj, xnb, q, sxs, sys, VX, VY, &
      vol, cxf, cyf, dnx, dny, fmk, mk, umask_, vmask_, rmask_)
    ! hlevel's inner part for one (tracer, cross-region column): the inner
    ! y fluxes of q through the faces j, j+1 and sx = (q + Yq)/2 (rows
    ! 1..ny, columns ilo..ihi); the inner x fluxes through i, i+1 and
    ! sy = (q + Xq)/2 (columns 1..nx, rows jlo..jhi). Levels k0..k1, stored
    ! at k - k0 + 1 in the scratch and the coefficients. Sub-pass isub:
    ! levels with isub > mk(k) are skipped (the CPU's cycle).
    integer(ip), value :: nxc, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi
    integer(ip), value :: west, east, south, north, ncl
    integer(ip), device, intent(in) :: xcol(*), xi(*), xj(*), xnb(8, *)
    real(pp), device, intent(in) :: q(ntr, 0:ncl, nz)
    real(rp), device, intent(inout) :: sxs(ntr, 0:ncl, WLEV), sys(ntr, 0:ncl, WLEV)
    real(rp), device, intent(in) :: VX(is:ie, js:je, nz), VY(is:ie, js:je, nz)
    real(rp), device, intent(in) :: vol(is:ie, js:je, WLEV), cxf(is:ie, js:je, WLEV), cyf(is:ie, js:je, WLEV)
    real(rp), device, intent(in) :: dnx(is:ie, js:je, WLEV), dny(is:ie, js:je, WLEV), fmk(nz)
    integer(ip), device, intent(in) :: mk(nz)
    real(rp), device, intent(in) :: umask_(is:ie, js:je), vmask_(is:ie, js:je), rmask_(is:ie, js:je)
    integer(ip) :: n, itr, c, i, j, k, kk, nb(8)
    logical :: dox, doy, wet, hl1, hr1, hl2, hr2, gl1, gr1, gl2, gr2
    real(rp) :: fm, s0, vv, f1, f2
    real(rp) :: vmm, vm0, vm1, vm2, umm, um0, um1, um2
    real(rp) :: ym2, ym1, yp1, yp2, xm2, xm1, xp1, xp2, vy0, vy1, cy0, cy1, dy, vx0, vx1, cx0, cx1, dx, vl
    n = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (n > nxc) return
    itr = threadIdx%x
    c = xcol(n); i = xi(n); j = xj(n)
    nb(1) = xnb(1, n); nb(2) = xnb(2, n); nb(3) = xnb(3, n); nb(4) = xnb(4, n)
    nb(5) = xnb(5, n); nb(6) = xnb(6, n); nb(7) = xnb(7, n); nb(8) = xnb(8, n)
    dox = (j >= 1 .and. j <= ny)
    doy = (i >= 1 .and. i <= nx)
    wet = rmask_(i, j) > 0.5_rp
    ! the column's masks and range tests, once for all levels
    if (dox) then
      vmm = vmask_(i, j - 1); vm0 = vmask_(i, j); vm1 = vmask_(i, j + 1); vm2 = vmask_(i, j + 2)
    endif
    if (doy) then
      umm = umask_(i - 1, j); um0 = umask_(i, j); um1 = umask_(i + 1, j); um2 = umask_(i + 2, j)
    endif
    hl1 = (j - 1 > jlo); hr1 = (j + 1 <= jhi); hl2 = (j > jlo); hr2 = (j + 2 <= jhi)
    gl1 = (i - 1 > ilo); gr1 = (i + 1 <= ihi); gl2 = (i > ilo); gr2 = (i + 2 <= ihi)
    do k = k0, k1
      if (isub > 1) then
        if (isub > mk(k)) cycle
      endif
      kk = k - k0 + 1
      ! every operand of the level is loaded before any arithmetic, so the
      ! loads are in flight together (one memory latency per level)
      fm = fmk(k)
      s0 = real(q(itr, c, k), rp)
      if (wet) then
        vl = vol(i, j, kk)
        if (dox) then
          ym2 = real(q(itr, nb(5), k), rp); ym1 = real(q(itr, nb(6), k), rp)
          yp1 = real(q(itr, nb(7), k), rp); yp2 = real(q(itr, nb(8), k), rp)
          vy0 = VY(i, j, k); vy1 = VY(i, j + 1, k)
          cy0 = cyf(i, j, kk); cy1 = cyf(i, j + 1, kk); dy = dny(i, j, kk)
        endif
        if (doy) then
          xm2 = real(q(itr, nb(1), k), rp); xm1 = real(q(itr, nb(2), k), rp)
          xp1 = real(q(itr, nb(3), k), rp); xp2 = real(q(itr, nb(4), k), rp)
          vx0 = VX(i, j, k); vx1 = VX(i + 1, j, k)
          cx0 = cxf(i, j, kk); cx1 = cxf(i + 1, j, kk); dx = dnx(i, j, kk)
        endif
        if (dox) then
          ! face j (between j-1 and j)
          vv = fm * vy0
          f1 = swf(vv, cy0, ym2, ym1, s0, yp1, vmm, vm0, vm1, hl1, hr1)
          if (south == 2 .and. j == 1 .and. vv > 0.0_rp) f1 = vv * ym1
          ! face j+1
          vv = fm * vy1
          f2 = swf(vv, cy1, ym1, s0, yp1, yp2, vm0, vm1, vm2, hl2, hr2)
          if (north == 2 .and. j + 1 == ny + 1 .and. vv < 0.0_rp) f2 = vv * yp1
          sxs(itr, c, kk) = 0.5_rp * (s0 + (vl * s0 - (f2 - f1)) / dy)
        endif
        if (doy) then
          vv = fm * vx0
          f1 = swf(vv, cx0, xm2, xm1, s0, xp1, umm, um0, um1, gl1, gr1)
          if (west == 2 .and. i == 1 .and. vv > 0.0_rp) f1 = vv * xm1
          vv = fm * vx1
          f2 = swf(vv, cx1, xm1, s0, xp1, xp2, um0, um1, um2, gl2, gr2)
          if (east == 2 .and. i + 1 == nx + 1 .and. vv < 0.0_rp) f2 = vv * xp1
          sys(itr, c, kk) = 0.5_rp * (s0 + (vl * s0 - (f2 - f1)) / dx)
        endif
      else
        if (dox) sxs(itr, c, kk) = s0
        if (doy) sys(itr, c, kk) = s0
      endif
    enddo
  end subroutine inner_kernel

  ! ------------------------------------------------ H3: outer and update
  attributes(global) launch_bounds(128, 4) subroutine outer_kernel(nwet, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, &
      ilo, ihi, jlo, jhi, west, east, south, north, ncl, wi, wj, nbt, q, sxs, sys, cout, VX, VY, HZS, &
      cxf, cyf, fmk, mk, pm_, pn_, umask_, vmask_, rmask_, pmon_u_, pnom_v_, sp_, tnu, dodiff, dtw, dbud, ew, ee, es, en)
    ! hlevel's outer part and hupdate for one (tracer, interior wet column):
    ! outer swept fluxes of sx through the x faces i, i+1 and of sy through
    ! the y faces j, j+1; on sub-pass 1 of a tracer (dodiff = 1) the TS_DIF2
    ! fluxes of the window-start state q over the window (dtw) with the
    ! diffusivity tnu(itr) + sp(i, j); content per area into cout:
    ! HZS q - pm pn div F (sub-pass 1), then cout - pm pn div F (sub-passes
    ! isub > 1, on the levels with isub <= mk(k); q is then the previous
    ! sub-pass's concentration). dbud = 1: the boundary-face fluxes into
    ! ew/ee/es/en (the budget's boundary term), summed over the sub-passes.
    integer(ip), value :: nwet, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi
    integer(ip), value :: west, east, south, north, ncl, dodiff, dbud
    real(rp), value :: dtw
    integer(ip), device, intent(in) :: wi(*), wj(*), nbt(8, *)
    real(pp), device, intent(in) :: q(ntr, 0:ncl, nz)
    real(rp), device, intent(in) :: sxs(ntr, 0:ncl, WLEV), sys(ntr, 0:ncl, WLEV)
    real(pp), device, intent(inout) :: cout(ntr, 0:ncl, nz)
    real(rp), device, intent(in) :: VX(is:ie, js:je, nz), VY(is:ie, js:je, nz), HZS(is:ie, js:je, nz)
    real(rp), device, intent(in) :: cxf(is:ie, js:je, WLEV), cyf(is:ie, js:je, WLEV), fmk(nz)
    integer(ip), device, intent(in) :: mk(nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    real(rp), device, intent(in) :: rmask_(is:ie, js:je), pmon_u_(is:ie, js:je), pnom_v_(is:ie, js:je), sp_(is:ie, js:je)
    real(rp), device, intent(in) :: tnu(*)
    real(rp), device, intent(inout) :: ew(ntr, ny, nz), ee(ntr, ny, nz), es(ntr, nx, nz), en(ntr, nx, nz)
    integer(ip) :: cc, itr, i, j, k, kk, nb(8)
    real(rp) :: fm, pmn, bg, x0, xm2, xm1, xp1, xp2, vv, fw, fe, fs, fn, q0, qxm, qxp, qym, qyp
    real(rp) :: umm, um0, um1, um2, vmm, vm0, vm1, vm2, dfw, dfe, dfs, dfn
    real(rp) :: y0, ym2, ym1, yp1, yp2, vx0, vx1, vy0, vy1, cx0, cx1, cy0, cy1, h0, hw, he, hs, hn, co
    logical :: gl1, gr1, gl2, gr2, hl1, hr1, hl2, hr2
    cc = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (cc > nwet) return
    itr = threadIdx%x
    i = wi(cc); j = wj(cc)
    nb(1) = nbt(1, cc); nb(2) = nbt(2, cc); nb(3) = nbt(3, cc); nb(4) = nbt(4, cc)
    nb(5) = nbt(5, cc); nb(6) = nbt(6, cc); nb(7) = nbt(7, cc); nb(8) = nbt(8, cc)
    pmn = pm_(i, j) * pn_(i, j)
    bg = 0.0_rp
    if (dodiff == 1) bg = tnu(itr)
    ! the column's masks, range tests and the level-independent leading
    ! factors of the diffusion fluxes ((dtw*0.25)*(d2 + d2))*pmon_u, as
    ! the CPU's expression evaluates them, once for all levels
    umm = umask_(i - 1, j); um0 = umask_(i, j); um1 = umask_(i + 1, j); um2 = umask_(i + 2, j)
    vmm = vmask_(i, j - 1); vm0 = vmask_(i, j); vm1 = vmask_(i, j + 1); vm2 = vmask_(i, j + 2)
    gl1 = (i - 1 > ilo); gr1 = (i + 1 <= ihi); gl2 = (i > ilo); gr2 = (i + 2 <= ihi)
    hl1 = (j - 1 > jlo); hr1 = (j + 1 <= jhi); hl2 = (j > jlo); hr2 = (j + 2 <= jhi)
    dfw = 0.0_rp; dfe = 0.0_rp; dfs = 0.0_rp; dfn = 0.0_rp
    if (dodiff == 1) then
      dfw = dtw * 0.25_rp * ((bg + sp_(i, j)) + (bg + sp_(i - 1, j))) * pmon_u_(i, j)
      dfe = dtw * 0.25_rp * ((bg + sp_(i + 1, j)) + (bg + sp_(i, j))) * pmon_u_(i + 1, j)
      dfs = dtw * 0.25_rp * ((bg + sp_(i, j)) + (bg + sp_(i, j - 1))) * pnom_v_(i, j)
      dfn = dtw * 0.25_rp * ((bg + sp_(i, j + 1)) + (bg + sp_(i, j))) * pnom_v_(i, j + 1)
    endif
    do k = k0, k1
      if (isub > 1) then
        if (isub > mk(k)) cycle
      endif
      kk = k - k0 + 1
      ! every operand of the level is loaded before any arithmetic, so the
      ! loads are in flight together (one memory latency per level)
      fm = fmk(k)
      x0 = sxs(itr, cc, kk)
      xm2 = sxs(itr, nb(1), kk); xm1 = sxs(itr, nb(2), kk); xp1 = sxs(itr, nb(3), kk); xp2 = sxs(itr, nb(4), kk)
      y0 = sys(itr, cc, kk)
      ym2 = sys(itr, nb(5), kk); ym1 = sys(itr, nb(6), kk); yp1 = sys(itr, nb(7), kk); yp2 = sys(itr, nb(8), kk)
      vx0 = VX(i, j, k); vx1 = VX(i + 1, j, k); vy0 = VY(i, j, k); vy1 = VY(i, j + 1, k)
      cx0 = cxf(i, j, kk); cx1 = cxf(i + 1, j, kk); cy0 = cyf(i, j, kk); cy1 = cyf(i, j + 1, kk)
      q0 = real(q(itr, cc, k), rp)
      h0 = HZS(i, j, k)
      if (dodiff == 1) then
        qxm = real(q(itr, nb(2), k), rp); qxp = real(q(itr, nb(3), k), rp)
        qym = real(q(itr, nb(6), k), rp); qyp = real(q(itr, nb(7), k), rp)
        hw = HZS(i - 1, j, k); he = HZS(i + 1, j, k); hs = HZS(i, j - 1, k); hn = HZS(i, j + 1, k)
      endif
      if (isub /= 1) co = real(cout(itr, cc, k), rp)
      ! x faces of sx (sweep_x over rows jstr..jend)
      vv = fm * vx0
      fw = swf(vv, cx0, xm2, xm1, x0, xp1, umm, um0, um1, gl1, gr1)
      if (west == 2 .and. i == 1 .and. vv > 0.0_rp) fw = vv * xm1
      vv = fm * vx1
      fe = swf(vv, cx1, xm1, x0, xp1, xp2, um0, um1, um2, gl2, gr2)
      if (east == 2 .and. i + 1 == nx + 1 .and. vv < 0.0_rp) fe = vv * xp1
      ! y faces of sy (sweep_y over columns istr..iend)
      vv = fm * vy0
      fs = swf(vv, cy0, ym2, ym1, y0, yp1, vmm, vm0, vm1, hl1, hr1)
      if (south == 2 .and. j == 1 .and. vv > 0.0_rp) fs = vv * ym1
      vv = fm * vy1
      fn = swf(vv, cy1, ym1, y0, yp1, yp2, vm0, vm1, vm2, hl2, hr2)
      if (north == 2 .and. j + 1 == ny + 1 .and. vv < 0.0_rp) fn = vv * yp1
      if (dodiff == 1) then
        fw = fw - dfw * (h0 + hw) * (q0 - qxm) * um0
        fe = fe - dfe * (he + h0) * (qxp - q0) * um1
        fs = fs - dfs * (h0 + hs) * (q0 - qym) * vm0
        fn = fn - dfn * (hn + h0) * (qyp - q0) * vm1
      endif
      if (isub == 1) then
        cout(itr, cc, k) = real(h0 * q0 - pmn * (fe - fw + fn - fs), pp)
      else
        cout(itr, cc, k) = real(co - pmn * (fe - fw + fn - fs), pp)
      endif
      if (dbud == 1) then
        if (isub == 1) then
          if (west >= 1 .and. i == 1) ew(itr, j, k) = fw * rmask_(i, j)
          if (east >= 1 .and. i == nx) ee(itr, j, k) = fe * rmask_(i, j)
          if (south >= 1 .and. j == 1) es(itr, i, k) = fs * rmask_(i, j)
          if (north >= 1 .and. j == ny) en(itr, i, k) = fn * rmask_(i, j)
        else
          if (west >= 1 .and. i == 1) ew(itr, j, k) = ew(itr, j, k) + fw * rmask_(i, j)
          if (east >= 1 .and. i == nx) ee(itr, j, k) = ee(itr, j, k) + fe * rmask_(i, j)
          if (south >= 1 .and. j == 1) es(itr, i, k) = es(itr, i, k) + fs * rmask_(i, j)
          if (north >= 1 .and. j == ny) en(itr, i, k) = en(itr, i, k) + fn * rmask_(i, j)
        endif
      endif
    enddo
  end subroutine outer_kernel

  ! ------------------------------------------- vertical, sources, solve
  attributes(global) launch_bounds(128, 4) subroutine vertical_kernel(nwet, ntr, nz, is, ie, js, je, ncl, wi, wj, &
      q, cio, qs, WA, HZS, Hz_, pm_, pn_, rmask_, wv1, wfc, wpv, wcf, shadow, sfa, rain, diag, &
      dbef, daft, dsrf, dnmn, dncn)
    ! vpass for one (tracer, interior wet column): content per area cio
    ! (from the horizontal pass) -> concentration, in place. The column's
    ! pseudo-thickness v1 = HZS - pm pn div V comes from factor (wv1, shared
    ! by the tracers); the concentrations q = c/v1 of levels m-1..m+2 ride
    ! along the face loop in registers (qwin), and a walk that reaches
    ! further recomputes q(l) = c(l)/v1(l) from cio, which stays intact until
    ! the back substitution (the same division, so the same value); the
    ! 1-D swept fluxes of the window's vertical transport (vsweep, with dw
    ! formed where it is used); the surface source on the top cell (tracers:
    ! sfa(itr, c); shadow = 1: the rain volume rain(i, j)); the implicit
    ! solve with factor's factors (window-end Hz, window-integrated Akt),
    ! its forward elimination fused into the flux loop (each level's value
    ! is final once its upper face is done; the CPU's operations and order
    ! per value) and stored in qs (slot 3 of tp, free at this point of the
    ! window; the shadow's own array). diag = 1: per column the content before (q = the
    ! window-start state) and after, the surface source, and the new
    ! negatives (tracers) or max |c - 1| (shadow, into dbef).
    integer(ip), value :: nwet, ntr, nz, is, ie, js, je, ncl, shadow, diag
    integer(ip), device, intent(in) :: wi(*), wj(*)
    real(pp), device, intent(in) :: q(ntr, 0:ncl, nz)
    real(pp), device, intent(inout) :: cio(ntr, 0:ncl, nz), qs(ntr, 0:ncl, nz)
    real(rp), device, intent(in) :: WA(is:ie, js:je, 0:nz), HZS(is:ie, js:je, nz), Hz_(is:ie, js:je, nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), device, intent(in) :: wv1(is:ie, js:je, nz)
    real(rp), device, intent(in) :: wfc(is:ie, js:je, nz), wpv(is:ie, js:je, nz), wcf(is:ie, js:je, nz)
    real(rp), device, intent(in) :: sfa(ntr, *), rain(is:ie, js:je)
    real(rp), device, intent(inout) :: dbef(ntr, *), daft(ntr, *), dsrf(ntr, *), dnmn(ntr, *), dncn(ntr, *)
    integer(ip) :: c, itr, i, j, k, m, jj, n, m1
    real(rp) :: pmn, rm, w, rem, s, cl, fl, fprev, bef, aft, nmn, ncn, dev, q0, cc, fw, vj, qj, dwa, dwb, ck
    real(rp) :: pmv, pnv, qa, qb, qc, qd, cb, c2, c3, c4, c5, vb, vc, vd, v4, v5, wcur, wnx, pvc, pvx, fcc, fcx
    integer(8) :: it0, tk, g0, gk
    c = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (c > nwet) return
    itr = threadIdx%x
    i = wi(c); j = wj(c)
    n = nz
    rm = rmask_(i, j)
    pmv = pm_(i, j); pnv = pn_(i, j)
    ! flat offsets of (itr, c, 1) in cio and of (i, j, 1) in wv1, and their level strides (qwin)
    tk = int(ntr, 8) * int(ncl + 1, 8)
    it0 = int(itr, 8) + int(ntr, 8) * int(c, 8)
    gk = int(ie - is + 1, 8) * int(je - js + 1, 8)
    g0 = int(i - is + 1, 8) + int(ie - is + 1, 8) * int(j - js, 8)
    ! the register window at face m = 1: q(0) (never used), q(1), q(2), q(3); the contents c and pseudo-
    ! thicknesses v of levels 1..4 and the transport and factors of face 1 (each loaded one face ahead
    ! of its use, so the loop's loads are in flight while the previous face is computed)
    qa = 0.0_rp
    vb = wv1(i, j, 1); cb = real(cio(itr, c, 1), rp)
    c2 = 0.0_rp; c3 = 0.0_rp; c4 = 0.0_rp; c5 = 0.0_rp; vc = 1.0_rp; vd = 1.0_rp; v4 = 1.0_rp; v5 = 1.0_rp
    if (n >= 2) then
      vc = wv1(i, j, 2); c2 = real(cio(itr, c, 2), rp)
    endif
    if (n >= 3) then
      vd = wv1(i, j, 3); c3 = real(cio(itr, c, 3), rp)
    endif
    if (n >= 4) then
      v4 = wv1(i, j, 4); c4 = real(cio(itr, c, 4), rp)
    endif
    wcur = 0.0_rp; wnx = 0.0_rp; fcc = 0.0_rp; fcx = 0.0_rp
    if (n >= 2) wcur = WA(i, j, 1)
    pvc = wpv(i, j, 1); pvx = pvc
    qb = real(real(cb / vb, pp), rp)
    qc = 0.0_rp; qd = 0.0_rp
    if (n >= 2) qc = real(real(c2 / vc, pp), rp)
    if (n >= 3) qd = real(real(c3 / vd, pp), rp)
    ! vsweep: fl(m) through the w-faces m = 1..n-1 (fl(0) = fl(n) = 0),
    ! applied as c(m) = c(m) - (fl(m) - fl(m-1)) once fl(m) is known; dw(m)
    ! = q(m+1) - q(m), dw(0) = dw(1), dw(n) = dw(n-1)
    fprev = 0.0_rp
    fw = 0.0_rp
    do m = 1, n - 1
      ! the next face's operands
      if (m + 1 <= n - 1) wnx = WA(i, j, m + 1)
      pvx = wpv(i, j, m + 1); fcx = wfc(i, j, m)
      if (m + 4 <= n) then
        v5 = wv1(i, j, m + 4); c5 = real(cio(itr, c, m + 4), rp)
      endif
      w = wcur * pmv * pnv
      if (w > 0.0_rp) then
        rem = w; jj = m; s = 0.0_rp
        do while (jj >= 1)
          vj = vwin(jj, m, vb, vc, wv1, g0, gk); qj = qwin(jj, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
          if (rem >= vj) then
            s = s + vj * qj; rem = rem - vj; jj = jj - 1
            if (rem <= 0.0_rp) exit
          else
            cl = rem / vj
            m1 = min(max(jj, 1_ip), n - 1)
            dwa = qwin(m1 + 1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk) - qwin(m1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
            m1 = min(max(jj - 1, 1_ip), n - 1)
            dwb = qwin(m1 + 1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk) - qwin(m1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
            s = s + rem * (qj + (1.0_rp - cl) * ((2.0_rp - cl) * dwa + (1.0_rp + cl) * dwb) * r6)
            rem = 0.0_rp
            exit
          endif
        enddo
        if (rem > 0.0_rp) s = s + rem * qwin(1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
        fl = s
      else if (w < 0.0_rp) then
        rem = -w; jj = m + 1; s = 0.0_rp
        do while (jj <= n)
          vj = vwin(jj, m, vb, vc, wv1, g0, gk); qj = qwin(jj, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
          if (rem >= vj) then
            s = s + vj * qj; rem = rem - vj; jj = jj + 1
            if (rem <= 0.0_rp) exit
          else
            cl = rem / vj
            m1 = min(max(jj - 1, 1_ip), n - 1)
            dwa = qwin(m1 + 1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk) - qwin(m1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
            m1 = min(max(jj, 1_ip), n - 1)
            dwb = qwin(m1 + 1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk) - qwin(m1, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
            s = s + rem * (qj - (1.0_rp - cl) * ((2.0_rp - cl) * dwa + (1.0_rp + cl) * dwb) * r6)
            rem = 0.0_rp
            exit
          endif
        enddo
        if (rem > 0.0_rp) s = s + rem * qwin(n, m, qa, qb, qc, qd, cio, wv1, it0, tk, g0, gk)
        fl = -s
      else
        fl = 0.0_rp
      endif
      cc = cb - (fl - fprev)
      fprev = fl
      ! forward elimination of level m (implicit vertical mixing)
      if (m == 1) then
        fw = pvc * cc
      else
        fw = pvc * (cc + fcc * fw)
      endif
      qs(itr, c, m) = real(fw, pp)
      ! slide the register window to face m + 1
      qa = qb; qb = qc; qc = qd
      if (m + 3 <= n) qd = real(real(c4 / v4, pp), rp)
      cb = c2; c2 = c3; c3 = c4; c4 = c5
      vb = vc; vc = vd; vd = v4; v4 = v5
      wcur = wnx; pvc = pvx; fcc = fcx
    enddo
    cc = cb - (0.0_rp - fprev)
    ! sources
    if (shadow == 0) then
      cc = cc + sfa(itr, c)
    else
      cc = cc + rain(i, j)
    endif
    ! top level and back substitution (window-integrated Akt, window-end Hz)
    ck = (cc + wfc(i, j, nz - 1) * fw) / wpv(i, j, nz) * rm
    cio(itr, c, nz) = real(ck, pp)
    do k = nz - 1, 1, -1
      ck = (real(qs(itr, c, k), rp) + wcf(i, j, k) * ck) * rm
      cio(itr, c, k) = real(ck, pp)
    enddo
    if (diag == 1) then
      pmn = pm_(i, j) * pn_(i, j)
      if (shadow == 0) then
        bef = 0.0_rp; aft = 0.0_rp; nmn = 0.0_rp; ncn = 0.0_rp
        do k = 1, nz
          q0 = real(q(itr, c, k), rp)
          ck = real(cio(itr, c, k), rp)
          bef = bef + HZS(i, j, k) * q0 / pmn * rm
          aft = aft + Hz_(i, j, k) * ck / pmn * rm
          if (q0 >= 0.0_rp .and. ck < 0.0_rp) then        ! negatives made by this window
            nmn = min(nmn, ck); ncn = ncn + 1.0_rp
          endif
        enddo
        dbef(itr, c) = bef; daft(itr, c) = aft; dnmn(itr, c) = nmn; dncn(itr, c) = ncn
        dsrf(itr, c) = sfa(itr, c) / pmn * rm
      else
        dev = 0.0_rp
        do k = 1, nz
          dev = max(dev, abs(real(cio(itr, c, k), rp) - 1.0_rp))
        enddo
        dbef(itr, c) = dev
      endif
    endif
  end subroutine vertical_kernel

  ! ----------------------------------------------------------- reductions
  attributes(global) launch_bounds(256, 2) subroutine red_kernel(op, ntr, m, a, out)
    ! out(itr) = sum (op 0), max (1) or min (2) of a(itr, 1:m); one block of
    ! 256 threads per tracer, a fixed order
    integer(ip), value :: op, ntr, m
    real(rp), device, intent(in) :: a(ntr, *)
    real(rp), device, intent(inout) :: out(*)
    real(rp), shared :: buf(256)
    integer(ip) :: itr, tid, n, s
    real(rp) :: acc
    itr = blockIdx%x; tid = threadIdx%x
    acc = 0.0_rp
    if (op == 1) acc = -huge(1.0_rp)
    if (op == 2) acc = huge(1.0_rp)
    do n = tid, m, 256
      if (op == 0) then
        acc = acc + a(itr, n)
      else if (op == 1) then
        acc = max(acc, a(itr, n))
      else
        acc = min(acc, a(itr, n))
      endif
    enddo
    buf(tid) = acc
    call syncthreads()
    s = 128
    do while (s >= 1)
      if (tid <= s) then
        if (op == 0) then
          buf(tid) = buf(tid) + buf(tid + s)
        else if (op == 1) then
          buf(tid) = max(buf(tid), buf(tid + s))
        else
          buf(tid) = min(buf(tid), buf(tid + s))
        endif
      endif
      call syncthreads()
      s = s / 2
    enddo
    if (tid == 1) out(itr) = buf(1)
  end subroutine red_kernel
#endif

  ! ============================================================ launchers
  subroutine bgcw_stats(VX, VY, WA, HZS, tnumax, dtw, rows, lck)
    ! column maxima (rows: (7, 0:nx+1, nz)), then lck(1:7, k): cface, c1d, cout, -v1min, crit,
    ! explicit diffusion number, vertical Courant number of level k
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), WA(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz), HZS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), intent(in) :: tnumax, dtw
    real(rp), WORK_ATTR intent(inout) :: rows(7, 0:decomp%nx + 1, decomp%nz), lck(7, decomp%nz)
    integer(ip) :: nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, nt
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call window_ranges(ilo, ihi, jlo, jhi)
#ifdef USE_CUDA
    nt = (nx + 2) * nz
    call stats_kernel<<<(nt + 127) / 128, 128>>>(nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, &
         VX, VY, WA, HZS, pm, pn, rmask, pmon_u, pnom_v, diff2_sp, tnumax, dtw, rows)
    call stats_level_kernel<<<(7 * nz + 127) / 128, 128>>>(nx, nz, rows, lck)
#else
    error stop 'bgcw_stats: CUDA build only'
#endif
  end subroutine bgcw_stats

  subroutine bgcw_hcoef(k0, k1, isub, mk, VX, VY, VS, fmk, vol, cxf, cyf, dnx, dny)
    ! hcoef for levels k0..k1 of sub-pass isub (VS: the sub-pass's
    ! pseudo-thickness; HZS on sub-pass 1), stored at k - k0 + 1; levels
    ! with isub > mk(k) are skipped
    integer(ip), intent(in) :: k0, k1, isub
    integer(ip), WORK_ATTR intent(in) :: mk(decomp%nz)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), fmk(decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: vol(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), cxf(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), cyf(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), dnx(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), dny(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV)
    integer(ip) :: i, j, k, kk, nx, ny, ilo, ihi, jlo, jhi
    real(rp) :: fm, vv, v0
    nx = decomp%nx; ny = decomp%ny
    call window_ranges(ilo, ihi, jlo, jhi)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = k0, k1
      do j = jlo, jhi
        do i = ilo, ihi
         if (isub == 1 .or. isub <= mk(k)) then
          kk = k - k0 + 1
          fm = fmk(k)
          v0 = VS(i, j, k) / (pm(i, j) * pn(i, j))
          vol(i, j, kk) = v0
          if (i >= 1 .and. i <= nx + 1) then
            vv = fm * VX(i, j, k)
            if (vv > 0.0_rp) then
              cxf(i, j, kk) = vv / (VS(i - 1, j, k) / (pm(i - 1, j) * pn(i - 1, j)))
            else if (vv < 0.0_rp) then
              cxf(i, j, kk) = -vv / v0
            else
              cxf(i, j, kk) = 0.0_rp
            endif
          endif
          if (j >= 1 .and. j <= ny + 1) then
            vv = fm * VY(i, j, k)
            if (vv > 0.0_rp) then
              cyf(i, j, kk) = vv / (VS(i, j - 1, k) / (pm(i, j - 1) * pn(i, j - 1)))
            else if (vv < 0.0_rp) then
              cyf(i, j, kk) = -vv / v0
            else
              cyf(i, j, kk) = 0.0_rp
            endif
          endif
          if (j >= 1 .and. j <= ny) dny(i, j, kk) = v0 - fm * (VY(i, j + 1, k) - VY(i, j, k))
          if (i >= 1 .and. i <= nx) dnx(i, j, kk) = v0 - fm * (VX(i + 1, j, k) - VX(i, j, k))
         endif
        enddo
      enddo
    enddo
  end subroutine bgcw_hcoef

  ! ------------------------------------------------ sub-passes isub > 1
  subroutine bgcw_vs(isub, klo, khi, mk, fmk, VX, VY, HZS, VS)
    ! the sub-pass's pseudo-thickness on the interior cells of the levels
    ! with isub <= mk(k) (bgcsub_window_tile): VS = HZS - (isub-1) fm pm pn
    ! div V; the halo and the edge rows follow (exchange, bgcw_edge_r)
    integer(ip), intent(in) :: isub, klo, khi
    integer(ip), WORK_ATTR intent(in) :: mk(decomp%nz)
    real(rp), WORK_ATTR intent(in) :: fmk(decomp%nz)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), HZS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: VS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    integer(ip) :: i, j, k, nx, ny
    nx = decomp%nx; ny = decomp%ny
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = klo, khi
      do j = 1, ny
        do i = 1, nx
          if (isub <= mk(k)) VS(i, j, k) = HZS(i, j, k) - real(isub - 1, rp) * fmk(k) * pm(i, j) * pn(i, j) * &
                                           (VX(i + 1, j, k) - VX(i, j, k) + VY(i, j + 1, k) - VY(i, j, k))
        enddo
      enddo
    enddo
  end subroutine bgcw_vs

  subroutine bgcw_edge_r(klo, khi, neb, ebi, ebj, src, dst)
    ! edge_copy of a 3-D field: the physical-edge rows (cells ebi, ebj),
    ! levels klo..khi, from src
    integer(ip), intent(in) :: klo, khi, neb
    integer(ip), WORK_ATTR intent(in) :: ebi(*), ebj(*)
    real(rp), WORK_ATTR intent(in) :: src(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: dst(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    integer(ip) :: k, p
    if (neb <= 0) return
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do k = klo, khi
      do p = 1, neb
        dst(ebi(p), ebj(p), k) = src(ebi(p), ebj(p), k)
      enddo
    enddo
  end subroutine bgcw_edge_r

  subroutine bgcw_subq(ntr, isub, klo, khi, mk, cin, VS, q)
    ! sub-pass isub > 1: the concentrations q = content / VS of the
    ! interior wet columns on the levels with isub <= mk(k) (tp layout, one
    ! slot each; the CPU's t(:,:,:,3) = t(:,:,:,nnew)/VS)
    integer(ip), intent(in) :: ntr, isub, klo, khi
    integer(ip), WORK_ATTR intent(in) :: mk(decomp%nz)
    real(pp), WORK_ATTR intent(in) :: cin(ntr, 0:ntpcol, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: VS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(pp), WORK_ATTR intent(inout) :: q(ntr, 0:ntpcol, decomp%nz)
    integer(ip) :: k, c, m, nw
    nw = nwet
    if (nw <= 0) return
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = klo, khi
      do c = 1, nw
        do m = 1, ntr
          if (isub <= mk(k)) q(m, c, k) = real(real(cin(m, c, k), rp) / VS(wet_i(c), wet_j(c), k), pp)
        enddo
      enddo
    enddo
  end subroutine bgcw_subq

  subroutine bgcw_edge_tp(ntr, klo, khi, neb, ebi, ebj, src, dst)
    ! edge_copy in the tp layout: the stored columns of the physical-edge
    ! rows (cells ebi, ebj) and the shared land column 0, levels klo..khi,
    ! from src (one slot each)
    integer(ip), intent(in) :: ntr, klo, khi, neb
    integer(ip), WORK_ATTR intent(in) :: ebi(*), ebj(*)
    real(pp), WORK_ATTR intent(in) :: src(ntr, 0:ntpcol, decomp%nz)
    real(pp), WORK_ATTR intent(inout) :: dst(ntr, 0:ntpcol, decomp%nz)
    integer(ip) :: k, p, m, c
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = klo, khi
      do p = 0, neb
        do m = 1, ntr
          c = 0
          if (p > 0) c = tpcol(ebi(max(p, 1_ip)), ebj(max(p, 1_ip)))
          if (p == 0 .or. c > 0) dst(m, c, k) = src(m, c, k)
        enddo
      enddo
    enddo
  end subroutine bgcw_edge_tp

  subroutine bgcw_inner(ntr, k0, k1, isub, mk, q, sxs, sys, VX, VY, vol, cxf, cyf, dnx, dny, fmk)
    ! H2 for ntr tracers of q (tp layout, one slot), levels k0..k1 of
    ! sub-pass isub (levels with isub > mk(k) skipped)
    integer(ip), intent(in) :: ntr, k0, k1, isub
    integer(ip), WORK_ATTR intent(in) :: mk(decomp%nz)
    real(pp), WORK_ATTR intent(in) :: q(*)
    real(rp), WORK_ATTR intent(inout) :: sxs(*), sys(*)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: vol(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), cxf(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), cyf(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), dnx(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), dny(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV)
    real(rp), WORK_ATTR intent(in) :: fmk(decomp%nz)
    integer(ip) :: nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, west, east, south, north
#ifdef USE_CUDA
    integer(ip) :: ncb
    type(dim3) :: grid, block
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call window_ranges(ilo, ihi, jlo, jhi)
    call open_edge_codes(west, east, south, north)
    if (nxc <= 0) return
#ifdef USE_CUDA
    if (ntr > 128) error stop 'bgcw_inner: ntr > 128 (kernel block size)'
    ncb = max(1_ip, 128_ip / ntr)
    block = dim3(ntr, ncb, 1)
    grid = dim3((nxc + ncb - 1) / ncb, 1, 1)
    call inner_kernel<<<grid, block>>>(nxc, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, &
         west, east, south, north, ntpcol, xc_col, xc_i, xc_j, xc_nb, q, sxs, sys, VX, VY, &
         vol, cxf, cyf, dnx, dny, fmk, mk, umask, vmask, rmask)
#else
    error stop 'bgcw_inner: CUDA build only'
#endif
  end subroutine bgcw_inner

  subroutine bgcw_outer(ntr, k0, k1, isub, mk, q, sxs, sys, cout, VX, VY, HZS, cxf, cyf, fmk, tnu, dodiff, dtw, &
                        dbud, ew, ee, es, en)
    ! H3 for ntr tracers: content per area of q into cout (tp layout),
    ! levels k0..k1 of sub-pass isub (levels with isub > mk(k) skipped)
    integer(ip), intent(in) :: ntr, k0, k1, isub, dodiff, dbud
    integer(ip), WORK_ATTR intent(in) :: mk(decomp%nz)
    real(pp), WORK_ATTR intent(in) :: q(*)
    real(pp), WORK_ATTR intent(inout) :: cout(*)
    real(rp), WORK_ATTR intent(in) :: sxs(*), sys(*), tnu(*)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), HZS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: cxf(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), cyf(decomp%is:decomp%ie, decomp%js:decomp%je, WLEV), fmk(decomp%nz)
    real(rp), intent(in) :: dtw
    real(rp), WORK_ATTR intent(inout) :: ew(*), ee(*), es(*), en(*)
    integer(ip) :: nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, west, east, south, north
#ifdef USE_CUDA
    integer(ip) :: ncb
    type(dim3) :: grid, block
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call window_ranges(ilo, ihi, jlo, jhi)
    call open_edge_codes(west, east, south, north)
    if (nwet <= 0) return
#ifdef USE_CUDA
    ncb = max(1_ip, 128_ip / ntr)
    block = dim3(ntr, ncb, 1)
    grid = dim3((nwet + ncb - 1) / ncb, 1, 1)
    call outer_kernel<<<grid, block>>>(nwet, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, &
         west, east, south, north, ntpcol, wet_i, wet_j, tpnb, q, sxs, sys, cout, VX, VY, HZS, &
         cxf, cyf, fmk, mk, pm, pn, umask, vmask, rmask, pmon_u, pnom_v, diff2_sp, tnu, dodiff, dtw, dbud, ew, ee, es, en)
#else
    error stop 'bgcw_outer: CUDA build only'
#endif
  end subroutine bgcw_outer

  subroutine bgcw_factor(VX, VY, HZS, KA, wv1, wfc, wpv, wcf)
    ! per interior wet column, the CPU's expressions: the pseudo-thickness
    ! left by the horizontal pass, v1 = HZS - pm pn div V (wv1); the
    ! window's implicit-diffusion factors fcr(k) = 2 KA/(Hz(k)+Hz(k+1)),
    ! pivots pivr, upper factors cfr, and topr (the divisor of the top
    ! level) in wpv(nz)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: HZS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: KA(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: wv1(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: wfc(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), wpv(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), wcf(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    integer(ip) :: c, i, j, k, nz
    real(rp) :: cff
    nz = decomp%nz
    if (nwet <= 0) return
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*>>>
#endif
    do c = 1, nwet
      i = wet_i(c); j = wet_j(c)
      do k = 1, nz
        wv1(i, j, k) = HZS(i, j, k) - pm(i, j) * pn(i, j) * (VX(i + 1, j, k) - VX(i, j, k) + VY(i, j + 1, k) - VY(i, j, k))
      enddo
      do k = 1, nz - 1
        wfc(i, j, k) = 2.0_rp * KA(i, j, k) / (Hz(i, j, k) + Hz(i, j, k + 1))
      enddo
      cff = 1.0_rp / (Hz(i, j, 1) + wfc(i, j, 1))
      wpv(i, j, 1) = cff; wcf(i, j, 1) = cff * wfc(i, j, 1)
      do k = 2, nz - 1
        cff = 1.0_rp / (Hz(i, j, k) + wfc(i, j, k) + wfc(i, j, k - 1) - wcf(i, j, k - 1) * wfc(i, j, k - 1))
        wpv(i, j, k) = cff; wcf(i, j, k) = cff * wfc(i, j, k)
      enddo
      wpv(i, j, nz) = Hz(i, j, nz) + wfc(i, j, nz - 1) - wcf(i, j, nz - 1) * wfc(i, j, nz - 1)
    enddo
  end subroutine bgcw_factor

  subroutine bgcw_vertical(ntr, q, cio, qs, WA, HZS, wv1, wfc, wpv, wcf, shadow, sfa, rain, diag, &
                           dbef, daft, dsrf, dnmn, dncn)
    ! vertical pass for ntr tracers (tp layout): q = window-start state,
    ! cio = content per area in, concentration out; qs = scratch (slot 3)
    integer(ip), intent(in) :: ntr, shadow, diag
    real(pp), WORK_ATTR intent(in) :: q(*)
    real(pp), WORK_ATTR intent(inout) :: cio(*), qs(*)
    real(rp), WORK_ATTR intent(in) :: WA(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: HZS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: wv1(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: wfc(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: wpv(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: wcf(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: sfa(*), rain(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: dbef(*), daft(*), dsrf(*), dnmn(*), dncn(*)
    integer(ip) :: nz, is, ie, js, je
#ifdef USE_CUDA
    integer(ip) :: ncb
    type(dim3) :: grid, block
#endif
    nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    if (nwet <= 0) return
#ifdef USE_CUDA
    ncb = max(1_ip, 128_ip / ntr)
    block = dim3(ntr, ncb, 1)
    grid = dim3((nwet + ncb - 1) / ncb, 1, 1)
    call vertical_kernel<<<grid, block>>>(nwet, ntr, nz, is, ie, js, je, ntpcol, wet_i, wet_j, &
         q, cio, qs, WA, HZS, Hz, pm, pn, rmask, wv1, wfc, wpv, wcf, shadow, sfa, rain, diag, &
         dbef, daft, dsrf, dnmn, dncn)
#else
    error stop 'bgcw_vertical: CUDA build only'
#endif
  end subroutine bgcw_vertical

  subroutine bgcw_reduce(op, ntr, m, a, out)
    integer(ip), intent(in) :: op, ntr, m
    real(rp), WORK_ATTR intent(in) :: a(*)
    real(rp), WORK_ATTR intent(inout) :: out(*)
#ifdef USE_CUDA
    if (m <= 0) return
    call red_kernel<<<ntr, 256>>>(op, ntr, m, a, out)
#else
    error stop 'bgcw_reduce: CUDA build only'
#endif
  end subroutine bgcw_reduce

  ! ------------------------------------------------- window boundaries
  subroutine bgcw_t3dbc(tp, ntr, ns, nn, VX, VY, HZS, mbry, nbw, tbw, tbe, tbs, tbn)
    ! bgcsub_t3dbc for all tracers: Orlanski radiation with the window
    ! Courant numbers of the summed boundary-face transports (clipped to 1),
    ! the window-start state (slot ns) on the right side, nudging toward the
    ! window-end boundary data (mbry; else 0, as passive_t3dbc); gradient
    ! conditions on closed sides; corners as the mean of the two adjacent
    ! boundary points. Slot nn is written.
    integer(ip), intent(in) :: ntr, ns, nn
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), HZS(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    logical, intent(in) :: mbry
    integer(ip), intent(in) :: nbw
    real(rp), WORK_ATTR intent(in) :: tbw(decomp%js:decomp%je, decomp%nz, nbw), tbe(decomp%js:decomp%je, decomp%nz, nbw)
    real(rp), WORK_ATTR intent(in) :: tbs(decomp%is:decomp%ie, decomp%nz, nbw), tbn(decomp%is:decomp%ie, decomp%nz, nbw)
    integer(ip) :: i, j, k, itr, nx, ny, nz, mb
    real(rp) :: cx, cy, cext, tn, text, g0c, g0p, g1c, g1p, w
    logical :: sw, se, nw, ne
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    mb = merge(1_ip, 0_ip, mbry)

    if (decomp%edge_w) then
      if (obc_west) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(0, j), k, ns), rp) - real(tp(itr, tpcol(0, j - 1), k, ns), rp)) * vmask(0, j)
              g0p = (real(tp(itr, tpcol(0, j + 1), k, ns), rp) - real(tp(itr, tpcol(0, j), k, ns), rp)) * vmask(0, j + 1)
              g1c = (real(tp(itr, tpcol(1, j), k, ns), rp) - real(tp(itr, tpcol(1, j - 1), k, ns), rp)) * vmask(1, j)
              g1p = (real(tp(itr, tpcol(1, j + 1), k, ns), rp) - real(tp(itr, tpcol(1, j), k, ns), rp)) * vmask(1, j + 1)
              cx = -VX(1, j, k) * pm(0, j) / (0.5_rp * (HZS(0, j, k) + HZS(1, j, k)) * dn_u(1, j))
              cy = 0.5_rp * (VY(0, j, k) / (0.5_rp * (HZS(0, j, k) + HZS(0, j - 1, k)) * dm_v(0, j)) &
                   + VY(0, j + 1, k) / (0.5_rp * (HZS(0, j + 1, k) + HZS(0, j, k)) * dm_v(0, j + 1))) * pn(0, j)
              cy = max(-1.0_rp, min(1.0_rp, cy))
              if (cx > 0.0_rp) then
                cext = 0.0_rp; cx = min(cx, 1.0_rp)
              else
                cext = min(-cx, 1.0_rp); cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(itr, tpcol(0, j), k, ns), rp) &
                   - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                   + cx * (real(tp(itr, tpcol(1, j), k, ns), rp) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
              text = 0.0_rp
              if (mb == 1) text = tbw(j, k, itr)
              tn = (1.0_rp - cext) * tn + cext * text
              tp(itr, tpcol(0, j), k, nn) = real(tn * rmask(0, j), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(0, j), k, nn) = real(real(tp(itr, tpcol(1, j), k, nn), rp) * rmask(0, j), pp)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_e) then
      if (obc_east) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(nx, j), k, ns), rp) - real(tp(itr, tpcol(nx, j - 1), k, ns), rp)) * vmask(nx, j)
              g0p = (real(tp(itr, tpcol(nx, j + 1), k, ns), rp) - real(tp(itr, tpcol(nx, j), k, ns), rp)) * vmask(nx, j + 1)
              g1c = (real(tp(itr, tpcol(nx + 1, j), k, ns), rp) - real(tp(itr, tpcol(nx + 1, j - 1), k, ns), rp)) &
                    * vmask(nx + 1, j)
              g1p = (real(tp(itr, tpcol(nx + 1, j + 1), k, ns), rp) - real(tp(itr, tpcol(nx + 1, j), k, ns), rp)) &
                    * vmask(nx + 1, j + 1)
              cx = VX(nx + 1, j, k) * pm(nx + 1, j) / (0.5_rp * (HZS(nx + 1, j, k) + HZS(nx, j, k)) * dn_u(nx + 1, j))
              cy = 0.5_rp * (VY(nx + 1, j, k) / (0.5_rp * (HZS(nx + 1, j, k) + HZS(nx + 1, j - 1, k)) * dm_v(nx + 1, j)) &
                   + VY(nx + 1, j + 1, k) / (0.5_rp * (HZS(nx + 1, j + 1, k) + HZS(nx + 1, j, k)) * dm_v(nx + 1, j + 1))) &
                   * pn(nx + 1, j)
              cy = max(-1.0_rp, min(1.0_rp, cy))
              if (cx > 0.0_rp) then
                cext = 0.0_rp; cx = min(cx, 1.0_rp)
              else
                cext = min(-cx, 1.0_rp); cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(itr, tpcol(nx + 1, j), k, ns), rp) &
                   - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                   + cx * (real(tp(itr, tpcol(nx, j), k, ns), rp) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mb == 1) text = tbe(j, k, itr)
              tn = (1.0_rp - cext) * tn + cext * text
              tp(itr, tpcol(nx + 1, j), k, nn) = real(tn * rmask(nx + 1, j), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do j = 1, ny
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(nx + 1, j), k, nn) = real(real(tp(itr, tpcol(nx, j), k, nn), rp) * rmask(nx + 1, j), pp)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_s) then
      if (obc_south) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(i, 1), k, ns), rp) - real(tp(itr, tpcol(i - 1, 1), k, ns), rp)) * umask(i, 1)
              g0p = (real(tp(itr, tpcol(i + 1, 1), k, ns), rp) - real(tp(itr, tpcol(i, 1), k, ns), rp)) * umask(i + 1, 1)
              g1c = (real(tp(itr, tpcol(i, 0), k, ns), rp) - real(tp(itr, tpcol(i - 1, 0), k, ns), rp)) * umask(i, 0)
              g1p = (real(tp(itr, tpcol(i + 1, 0), k, ns), rp) - real(tp(itr, tpcol(i, 0), k, ns), rp)) * umask(i + 1, 0)
              cy = -VY(i, 1, k) * pn(i, 0) / (0.5_rp * (HZS(i, 0, k) + HZS(i, 1, k)) * dm_v(i, 1))
              cx = 0.5_rp * (VX(i, 0, k) / (0.5_rp * (HZS(i, 0, k) + HZS(i - 1, 0, k)) * dn_u(i, 0)) &
                   + VX(i + 1, 0, k) / (0.5_rp * (HZS(i + 1, 0, k) + HZS(i, 0, k)) * dn_u(i + 1, 0))) * pm(i, 0)
              cx = max(-1.0_rp, min(1.0_rp, cx))
              if (cy > 0.0_rp) then
                cext = 0.0_rp; cy = min(cy, 1.0_rp)
              else
                cext = min(-cy, 1.0_rp); cy = 0.0_rp
              endif
              ! g1* is the boundary row (jstr-1), g0* the first interior row
              tn = (1.0_rp - cy) * (real(tp(itr, tpcol(i, 0), k, ns), rp) &
                   - max(cx, 0.0_rp) * g1c - min(cx, 0.0_rp) * g1p) &
                   + cy * (real(tp(itr, tpcol(i, 1), k, ns), rp) - max(cx, 0.0_rp) * g0c - min(cx, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mb == 1) text = tbs(i, k, itr)
              tn = (1.0_rp - cext) * tn + cext * text
              tp(itr, tpcol(i, 0), k, nn) = real(tn * rmask(i, 0), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(i, 0), k, nn) = real(real(tp(itr, tpcol(i, 1), k, nn), rp) * rmask(i, 0), pp)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_n) then
      if (obc_north) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              g0c = (real(tp(itr, tpcol(i, ny), k, ns), rp) - real(tp(itr, tpcol(i - 1, ny), k, ns), rp)) * umask(i, ny)
              g0p = (real(tp(itr, tpcol(i + 1, ny), k, ns), rp) - real(tp(itr, tpcol(i, ny), k, ns), rp)) * umask(i + 1, ny)
              g1c = (real(tp(itr, tpcol(i, ny + 1), k, ns), rp) - real(tp(itr, tpcol(i - 1, ny + 1), k, ns), rp)) &
                    * umask(i, ny + 1)
              g1p = (real(tp(itr, tpcol(i + 1, ny + 1), k, ns), rp) - real(tp(itr, tpcol(i, ny + 1), k, ns), rp)) &
                    * umask(i + 1, ny + 1)
              cy = VY(i, ny + 1, k) * pn(i, ny + 1) / (0.5_rp * (HZS(i, ny + 1, k) + HZS(i, ny, k)) * dm_v(i, ny + 1))
              cx = 0.5_rp * (VX(i, ny + 1, k) / (0.5_rp * (HZS(i, ny + 1, k) + HZS(i - 1, ny + 1, k)) * dn_u(i, ny + 1)) &
                   + VX(i + 1, ny + 1, k) / (0.5_rp * (HZS(i + 1, ny + 1, k) + HZS(i, ny + 1, k)) * dn_u(i + 1, ny + 1))) &
                   * pm(i, ny + 1)
              cx = max(-1.0_rp, min(1.0_rp, cx))
              if (cy > 0.0_rp) then
                cext = 0.0_rp; cy = min(cy, 1.0_rp)
              else
                cext = min(-cy, 1.0_rp); cy = 0.0_rp
              endif
              tn = (1.0_rp - cy) * (real(tp(itr, tpcol(i, ny + 1), k, ns), rp) &
                   - max(cx, 0.0_rp) * g1c - min(cx, 0.0_rp) * g1p) &
                   + cy * (real(tp(itr, tpcol(i, ny), k, ns), rp) - max(cx, 0.0_rp) * g0c - min(cx, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mb == 1) text = tbn(i, k, itr)
              tn = (1.0_rp - cext) * tn + cext * text
              tp(itr, tpcol(i, ny + 1), k, nn) = real(tn * rmask(i, ny + 1), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do i = 1, nx
          do k = 1, nz
            do itr = 1, ntr
              tp(itr, tpcol(i, ny + 1), k, nn) = real(real(tp(itr, tpcol(i, ny), k, nn), rp) * rmask(i, ny + 1), pp)
            enddo
          enddo
        enddo
      endif
    endif

    ! corners (both sides physical edges): (rmask(1) t(1) + rmask(2) t(2))/w
    sw = decomp%edge_s .and. decomp%edge_w; se = decomp%edge_s .and. decomp%edge_e
    nw = decomp%edge_n .and. decomp%edge_w; ne = decomp%edge_n .and. decomp%edge_e
    if (.not. (sw .or. se .or. nw .or. ne)) return
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do k = 1, nz
      do itr = 1, ntr
        if (sw) then
          w = rmask(1, 0) + rmask(0, 1)
          if (w > 0.0_rp) then
            tp(itr, tpcol(0, 0), k, nn) = real((rmask(1, 0) * real(tp(itr, tpcol(1, 0), k, nn), rp) &
                 + rmask(0, 1) * real(tp(itr, tpcol(0, 1), k, nn), rp)) / w, pp)
          else
            tp(itr, tpcol(0, 0), k, nn) = 0.0_pp
          endif
        endif
        if (se) then
          w = rmask(nx, 0) + rmask(nx + 1, 1)
          if (w > 0.0_rp) then
            tp(itr, tpcol(nx + 1, 0), k, nn) = real((rmask(nx, 0) * real(tp(itr, tpcol(nx, 0), k, nn), rp) &
                 + rmask(nx + 1, 1) * real(tp(itr, tpcol(nx + 1, 1), k, nn), rp)) / w, pp)
          else
            tp(itr, tpcol(nx + 1, 0), k, nn) = 0.0_pp
          endif
        endif
        if (nw) then
          w = rmask(1, ny + 1) + rmask(0, ny)
          if (w > 0.0_rp) then
            tp(itr, tpcol(0, ny + 1), k, nn) = real((rmask(1, ny + 1) * real(tp(itr, tpcol(1, ny + 1), k, nn), rp) &
                 + rmask(0, ny) * real(tp(itr, tpcol(0, ny), k, nn), rp)) / w, pp)
          else
            tp(itr, tpcol(0, ny + 1), k, nn) = 0.0_pp
          endif
        endif
        if (ne) then
          w = rmask(nx, ny + 1) + rmask(nx + 1, ny)
          if (w > 0.0_rp) then
            tp(itr, tpcol(nx + 1, ny + 1), k, nn) = real((rmask(nx, ny + 1) * real(tp(itr, tpcol(nx, ny + 1), k, nn), rp) &
                 + rmask(nx + 1, ny) * real(tp(itr, tpcol(nx + 1, ny), k, nn), rp)) / w, pp)
          else
            tp(itr, tpcol(nx + 1, ny + 1), k, nn) = 0.0_pp
          endif
        endif
      enddo
    enddo
  end subroutine bgcw_t3dbc

end module bgc_window_mod

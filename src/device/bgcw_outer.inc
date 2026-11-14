  ! H3, outer fluxes and update
  ! Included twice by bgc_window.F90: as outer_kernel (BGCW_SRC undefined: every
  ! column but those of the list rl, bit for bit the kernel without
  ! sources) and as outer_kernel_src (BGCW_SRC: only the columns of rl, with the
  ! sources), so the source code costs the other columns no registers.
  attributes(global) launch_bounds(128, 4) subroutine BGCW_KNAME(nwet, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, &
      ilo, ihi, jlo, jhi, west, east, south, north, ncl, wi, wj, nbt, q, sxs, sys, cout, VX, VY, HZS, &
      cxf, cyf, fmk, mk, pm_, pn_, umask_, vmask_, rmask_, pmon_u_, pnom_v_, sp_, tnu, dodiff, dtw, dbud, ew, ee, es, en, &
      nrl, rl, shd, rbo, wfu, wfv, rua, rcl, drv)
    ! hlevel's outer part and hupdate for one (tracer, interior wet column):
    ! outer swept fluxes of sx through the x faces i, i+1 and of sy through
    ! the y faces j, j+1; on sub-pass 1 of a tracer (dodiff = 1) the TS_DIF2
    ! fluxes of the window-start state q over the window (dtw) with the
    ! diffusivity tnu(itr) + sp(i, j); content per area into cout:
    ! HZS q - pm pn div F (sub-pass 1), then cout - pm pn div F (sub-passes
    ! isub > 1, on the levels with isub <= mk(k); q is then the previous
    ! sub-pass's concentration). dbud = 1: the boundary-face fluxes into
    ! ew/ee/es/en (the budget's boundary term), summed over the sub-passes,
    ! and the river faces' fluxes into the cell into drv(itr, w) (the
    ! budget's river term). River faces (rbo(cc), bits as inner_kernel)
    ! carry the window's river transport (rivf) before the diffusion is
    ! added (hupdate after river_faces): the columns of the list rl(1:nrl)
    ! in the BGCW_SRC variant, which the other one skips.
    integer(ip), value :: nwet, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi
    integer(ip), value :: west, east, south, north, ncl, dodiff, dbud, shd, nrl
    integer(ip), device, intent(in) :: rbo(*), rl(*), wfu(is:ie, js:je), wfv(is:ie, js:je)
    real(rp), device, intent(in) :: rua(ntr, *), rcl(*)
    real(rp), device, intent(inout) :: drv(ntr, *)
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
#ifdef BGCW_SRC
    integer(ip) :: rb, w, m
#endif
    real(rp) :: fm, pmn, bg, x0, xm2, xm1, xp1, xp2, vv, fw, fe, fs, fn, q0, qxm, qxp, qym, qyp
    real(rp) :: umm, um0, um1, um2, vmm, vm0, vm1, vm2, dfw, dfe, dfs, dfn
    real(rp) :: y0, ym2, ym1, yp1, yp2, vx0, vx1, vy0, vy1, cx0, cx1, cy0, cy1, h0, hw, he, hs, hn, co
    logical :: gl1, gr1, gl2, gr2, hl1, hr1, hl2, hr2
#ifdef BGCW_SRC
    m = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (m > nrl) return
    cc = rl(m)
#else
    cc = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (cc > nwet) return
    if (nrl > 0) then
      if (rbo(cc) /= 0) return
    endif
#endif
    itr = threadIdx%x
    i = wi(cc); j = wj(cc)
    nb(1) = nbt(1, cc); nb(2) = nbt(2, cc); nb(3) = nbt(3, cc); nb(4) = nbt(4, cc)
    nb(5) = nbt(5, cc); nb(6) = nbt(6, cc); nb(7) = nbt(7, cc); nb(8) = nbt(8, cc)
#ifdef BGCW_SRC
    rb = rbo(cc)
#endif
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
#ifdef BGCW_SRC
      if (rb /= 0) then
        if (iand(rb, 1) /= 0) then
          w = wfu(i, j); fw = rivf(fm, rua(itr, w), vx0, rcl(w), shd)
        endif
        if (iand(rb, 2) /= 0) then
          w = wfu(i + 1, j); fe = rivf(fm, rua(itr, w), vx1, rcl(w), shd)
        endif
        if (iand(rb, 4) /= 0) then
          w = wfv(i, j); fs = rivf(fm, rua(itr, w), vy0, rcl(w), shd)
        endif
        if (iand(rb, 8) /= 0) then
          w = wfv(i, j + 1); fn = rivf(fm, rua(itr, w), vy1, rcl(w), shd)
        endif
      endif
#endif
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
#ifdef BGCW_SRC
        if (rb /= 0) then
          if (iand(rb, 1) /= 0) then
            w = wfu(i, j); drv(itr, w) = drv(itr, w) + fw
          endif
          if (iand(rb, 2) /= 0) then
            w = wfu(i + 1, j); drv(itr, w) = drv(itr, w) - fe
          endif
          if (iand(rb, 4) /= 0) then
            w = wfv(i, j); drv(itr, w) = drv(itr, w) + fs
          endif
          if (iand(rb, 8) /= 0) then
            w = wfv(i, j + 1); drv(itr, w) = drv(itr, w) - fn
          endif
        endif
#endif
      endif
    enddo
  end subroutine BGCW_KNAME

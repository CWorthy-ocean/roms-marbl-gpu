  ! H2, inner operators
  ! Included twice by bgc_window.F90: as inner_kernel (BGCW_SRC undefined: every
  ! column but those of the list rl, bit for bit the kernel without
  ! sources) and as inner_kernel_src (BGCW_SRC: only the columns of rl, with the
  ! sources), so the source code costs the other columns no registers.
  attributes(global) launch_bounds(128, 4) subroutine BGCW_KNAME(nxc, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, &
      ilo, ihi, jlo, jhi, west, east, south, north, ncl, xcol, xi, xj, xnb, q, sxs, sys, VX, VY, &
      vol, cxf, cyf, dnx, dny, fmk, mk, umask_, vmask_, rmask_, nrl, rl, shd, rbx, wfu, wfv, rua, rcl)
    ! hlevel's inner part for one (tracer, cross-region column): the inner
    ! y fluxes of q through the faces j, j+1 and sx = (q + Yq)/2 (rows
    ! 1..ny, columns ilo..ihi); the inner x fluxes through i, i+1 and
    ! sy = (q + Xq)/2 (columns 1..nx, rows jlo..jhi). Levels k0..k1, stored
    ! at k - k0 + 1 in the scratch and the coefficients. Sub-pass isub:
    ! levels with isub > mk(k) are skipped (the CPU's cycle). River faces
    ! (rbx(n): bits 1, 2 = u faces i, i+1; 4, 8 = v faces j, j+1; window
    ! face index from wfu/wfv) carry the window's river transport (rivf)
    ! instead, after the inflow rule (river_faces after sweep_x/sweep_y):
    ! the columns of the list rl(1:nrl) in the BGCW_SRC variant, which the
    ! other one skips.
    integer(ip), value :: nxc, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi
    integer(ip), value :: west, east, south, north, ncl, shd, nrl
    integer(ip), device, intent(in) :: xcol(*), xi(*), xj(*), xnb(8, *), rbx(*), rl(*)
    integer(ip), device, intent(in) :: wfu(is:ie, js:je), wfv(is:ie, js:je)
    real(rp), device, intent(in) :: rua(ntr, *), rcl(*)
    real(pp), device, intent(in) :: q(ntr, 0:ncl, nz)
    real(rp), device, intent(inout) :: sxs(ntr, 0:ncl, WLEV), sys(ntr, 0:ncl, WLEV)
    real(rp), device, intent(in) :: VX(is:ie, js:je, nz), VY(is:ie, js:je, nz)
    real(rp), device, intent(in) :: vol(is:ie, js:je, WLEV), cxf(is:ie, js:je, WLEV), cyf(is:ie, js:je, WLEV)
    real(rp), device, intent(in) :: dnx(is:ie, js:je, WLEV), dny(is:ie, js:je, WLEV), fmk(nz)
    integer(ip), device, intent(in) :: mk(nz)
    real(rp), device, intent(in) :: umask_(is:ie, js:je), vmask_(is:ie, js:je), rmask_(is:ie, js:je)
    integer(ip) :: n, itr, c, i, j, k, kk, nb(8)
#ifdef BGCW_SRC
    integer(ip) :: rb, w, m
#endif
    logical :: dox, doy, wet, hl1, hr1, hl2, hr2, gl1, gr1, gl2, gr2
    real(rp) :: fm, s0, vv, f1, f2
    real(rp) :: vmm, vm0, vm1, vm2, umm, um0, um1, um2
    real(rp) :: ym2, ym1, yp1, yp2, xm2, xm1, xp1, xp2, vy0, vy1, cy0, cy1, dy, vx0, vx1, cx0, cx1, dx, vl
#ifdef BGCW_SRC
    m = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (m > nrl) return
    n = rl(m)
#else
    n = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (n > nxc) return
    if (nrl > 0) then
      if (rbx(n) /= 0) return
    endif
#endif
    itr = threadIdx%x
    c = xcol(n); i = xi(n); j = xj(n)
    nb(1) = xnb(1, n); nb(2) = xnb(2, n); nb(3) = xnb(3, n); nb(4) = xnb(4, n)
    nb(5) = xnb(5, n); nb(6) = xnb(6, n); nb(7) = xnb(7, n); nb(8) = xnb(8, n)
#ifdef BGCW_SRC
    rb = rbx(n)
#endif
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
#ifdef BGCW_SRC
          if (rb /= 0) then
            if (iand(rb, 4) /= 0) then
              w = wfv(i, j); f1 = rivf(fm, rua(itr, w), vy0, rcl(w), shd)
            endif
            if (iand(rb, 8) /= 0) then
              w = wfv(i, j + 1); f2 = rivf(fm, rua(itr, w), vy1, rcl(w), shd)
            endif
          endif
#endif
          sxs(itr, c, kk) = 0.5_rp * (s0 + (vl * s0 - (f2 - f1)) / dy)
        endif
        if (doy) then
          vv = fm * vx0
          f1 = swf(vv, cx0, xm2, xm1, s0, xp1, umm, um0, um1, gl1, gr1)
          if (west == 2 .and. i == 1 .and. vv > 0.0_rp) f1 = vv * xm1
          vv = fm * vx1
          f2 = swf(vv, cx1, xm1, s0, xp1, xp2, um0, um1, um2, gl2, gr2)
          if (east == 2 .and. i + 1 == nx + 1 .and. vv < 0.0_rp) f2 = vv * xp1
#ifdef BGCW_SRC
          if (rb /= 0) then
            if (iand(rb, 1) /= 0) then
              w = wfu(i, j); f1 = rivf(fm, rua(itr, w), vx0, rcl(w), shd)
            endif
            if (iand(rb, 2) /= 0) then
              w = wfu(i + 1, j); f2 = rivf(fm, rua(itr, w), vx1, rcl(w), shd)
            endif
          endif
#endif
          sys(itr, c, kk) = 0.5_rp * (s0 + (vl * s0 - (f2 - f1)) / dx)
        endif
      else
        if (dox) sxs(itr, c, kk) = s0
        if (doy) sys(itr, c, kk) = s0
      endif
    enddo
  end subroutine BGCW_KNAME

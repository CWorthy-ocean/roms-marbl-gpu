  ! vertical pass, sources, implicit solve
  ! Included twice by bgc_window.F90: as vertical_kernel (BGCW_SRC undefined: every
  ! column but those of the list rl, bit for bit the kernel without
  ! sources) and as vertical_kernel_src (BGCW_SRC: only the columns of rl, with the
  ! sources), so the source code costs the other columns no registers.
  attributes(global) launch_bounds(128, 4) subroutine BGCW_KNAME(nwet, ntr, nz, is, ie, js, je, ncl, wi, wj, &
      q, cio, qs, WA, HZS, Hz_, pm_, pn_, rmask_, wv1, wfc, wpv, wcf, shadow, sfa, rain, diag, &
      dbef, daft, dsrf, dnmn, dncn, nx, ny, nqd, nrl, rl, hsm, wq0, wq1, ia, id, cpa, c3f)
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
    ! negatives (tracers) or max |c - 1| (shadow, into dbef). Point sources
    ! (vsrc: 3D CDR forcing of tracers ia/id from c3f, the release contents
    ! cpa of the column's profiles wq0..wq1) at each level after its swept
    ! flux and (top) the surface source, as vpass: the columns of the list
    ! rl(1:nrl) in the BGCW_SRC variant, which the other one skips (hsm).
    integer(ip), value :: nwet, ntr, nz, is, ie, js, je, ncl, shadow, diag, nx, ny, nqd, ia, id, nrl
    integer(ip), device, intent(in) :: wq0(*), wq1(*), rl(*), hsm(*)
    real(rp), device, intent(in) :: cpa(ntr, nqd, nz), c3f(nx, ny, nz, 2)
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
#ifdef BGCW_SRC
    integer(ip) :: p0, p1, hs
#endif
    real(rp) :: pmn, rm, w, rem, s, cl, fl, fprev, bef, aft, nmn, ncn, dev, q0, cc, fw, vj, qj, dwa, dwb, ck
    real(rp) :: pmv, pnv, qa, qb, qc, qd, cb, c2, c3, c4, c5, vb, vc, vd, v4, v5, wcur, wnx, pvc, pvx, fcc, fcx
    integer(8) :: it0, tk, g0, gk
#ifdef BGCW_SRC
    m = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (m > nrl) return
    c = rl(m)
#else
    c = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (c > nwet) return
    if (nrl > 0) then
      if (hsm(c) /= 0) return
    endif
#endif
    itr = threadIdx%x
    i = wi(c); j = wj(c)
    n = nz
    rm = rmask_(i, j)
    pmv = pm_(i, j); pnv = pn_(i, j)
#ifdef BGCW_SRC
    p0 = wq0(c); p1 = wq1(c)
    hs = 0
    if (p1 >= p0 .or. itr == ia .or. itr == id) hs = 1
#endif
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
#ifdef BGCW_SRC
      if (hs == 1) cc = vsrc(cc, m, itr, i, j, pmv * pnv, p0, p1, ia, id, ntr, nqd, nz, nx, ny, cpa, c3f)
#endif
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
#ifdef BGCW_SRC
    if (hs == 1) cc = vsrc(cc, nz, itr, i, j, pmv * pnv, p0, p1, ia, id, ntr, nqd, nz, nx, ny, cpa, c3f)
#endif
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
  end subroutine BGCW_KNAME

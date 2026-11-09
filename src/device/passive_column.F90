#include "mem_attr.inc"
module passive_column_mod
  ! Passive-tracer transport kernels for the tracer-innermost layout
  ! tp(ntr, 0:ntpcol, nz, 3) over the stored columns: one thread per
  ! (tracer, wet column, block of HLEV levels) for the horizontal steps
  ! (separate C4 predictor and UP3 corrector kernels), one thread per
  ! (tracer, cell) for the lateral diffusion, one thread per (tracer, wet
  ! column) for the vertical advection + implicit solve. Thread blocks are
  ! (ntr, cells), so consecutive threads hold consecutive tracers of a cell
  ! and every tracer load of a warp is one contiguous line. The time slots
  ! are passed to the kernels as separate arrays. The arithmetic per element
  ! is that of the former per-cell kernels (pc4/pup3 faces,
  ! vert_column_fused, the Thomas solve), so the FP64 build is bit-identical
  ! to them.
  ! Separate from passive_mod so the explicit kernel launches are never
  ! inlined into their callers (nvfortran 25.9 segfaulted when they were).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use gpu_state, only: nwet, wet_i, wet_j
  use formulas
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private
  public :: passive_horiz_run, passive_t3dmix_run, passive_column_run
  ! Levels per thread in the horizontal kernels: the column quantities
  ! (stencil columns, mask tests, edges) are formed once per HLEV levels.
  ! The kernels' launch_bounds assume blocks of at most 128 threads (ntr <= 128).
  integer(ip), parameter :: HLEV = 20

contains

  ! ------------------------------------------- column factorization
  subroutine passive_factor(dtau, use_fwd)
    ! Thomas factors of the implicit vertical-diffusion matrix for every
    ! column (col_fc = 1/diag, col_cf = upper factor, col_wc = fc +
    ! max(wc,0)), the salinity diffusivity; same arithmetic as
    ! implicit_tracer_columns. Shared by all passive tracers.
    real(rp), intent(in) :: dtau
    logical, intent(in) :: use_fwd
    integer(ip) :: i, j, k, nx, ny, nz, iAkt, lim
    real(rp) :: cff, dc0, hz_k, hz_kp, fck, wck, cfk, fcm, wcm, cfm
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    iAkt = min(2_ip, int(size(Akt, 4), ip))
    lim = dry_lim(G_TRC)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * pm(i, j) * pn(i, j)
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, 1); hz_kp = Hz_fwd(i, j, 2)
        else
          hz_k = Hz(i, j, 1); hz_kp = Hz(i, j, 2)
        endif
        fck = 2.0_rp * dtau * Akt(i, j, 1, iAkt) / (hz_kp + hz_k)
        wck = dc0 * Wi(i, j, 1)
        cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp))
        cfk = cff * (fck - min(wck, 0.0_rp))
        col_fc(i, j, 1) = cff
        col_cf(i, j, 1) = cfk
        col_wc(i, j, 1) = fck + max(wck, 0.0_rp)
        do k = 2, nz - 1
          fcm = fck; wcm = wck; cfm = cfk
          if (use_fwd) then
            hz_k = Hz_fwd(i, j, k); hz_kp = Hz_fwd(i, j, k + 1)
          else
            hz_k = Hz(i, j, k); hz_kp = Hz(i, j, k + 1)
          endif
          fck = 2.0_rp * dtau * Akt(i, j, k, iAkt) / (hz_k + hz_kp)
          wck = dc0 * Wi(i, j, k)
          cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp) + &
                          fcm - min(wcm, 0.0_rp) - &
                          cfm * (fcm + max(wcm, 0.0_rp)))
          cfk = cff * (fck - min(wck, 0.0_rp))
          col_fc(i, j, k) = cff
          col_cf(i, j, k) = cfk
          col_wc(i, j, k) = fck + max(wck, 0.0_rp)
        enddo
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, nz)
        else
          hz_k = Hz(i, j, nz)
        endif
        ! top: 1 / (hz + fc(nz-1) - min(wc(nz-1),0) - cf(nz-1)*(fc+max(wc,0))(nz-1))
        col_fc(i, j, nz) = 1.0_rp / (hz_k + fck - min(wck, 0.0_rp) - cfk * (fck + max(wck, 0.0_rp)))
        col_wc(i, j, nz) = fck + max(wck, 0.0_rp)
      enddo
    enddo
  end subroutine passive_factor

  ! --------------------------------------- horizontal face divergences
  ! One level of one (tracer, wet column): the stencil values are loaded by
  ! the caller (all loads of a level ahead of its arithmetic), the mask
  ! tests are formed once per column (bu*/bv*/br* = mask > 0.5, the test of
  ! apply_mask). Same expressions as pc4_fx/pc4_fe and pup3_fx/pup3_fe.
  DEVICE_ATTR
  function selm(val, m) result(out)
    ! apply_mask with the test done: val where the mask is > 0.5, else 0
    real(rp), intent(in) :: val
    logical, intent(in) :: m
    real(rp) :: out
    out = 0.0_rp
    if (m) out = val
  end function selm

  DEVICE_ATTR
  subroutine c4_div(t0, xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2, wedge, eedge, sedge, nedge, &
                    bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, qxw, qxe, qys, qyn)
    ! The four curvature quotients of the C4 faces of pc4_fx/pc4_fe
    ! ((drw - dlw) / 6 and (dre - dcw) / 6 per direction), formed together
    ! so that all stencil loads of a level are issued before the first
    ! division; c4_faces completes the faces with the same expressions.
    real(rp), intent(in) :: t0, xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2
    logical, intent(in) :: wedge, eedge, sedge, nedge, bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3
    real(rp), intent(out) :: qxw, qxe, qys, qyn
    real(rp) :: dcw, dlw, drw, dre, dcs, dls, drs, drn
    dcw = selm(t0 - xm1, bu1)
    if (wedge) then
      dlw = dcw
    else
      dlw = selm(xm1 - xm2, bu0)
    endif
    drw = selm(xp1 - t0, bu2)
    if (eedge) then
      dre = drw
    else
      dre = selm(xp2 - xp1, bu3)
    endif
    dcs = selm(t0 - ym1, bv1)
    if (sedge) then
      dls = dcs
    else
      dls = selm(ym1 - ym2, bv0)
    endif
    drs = selm(yp1 - t0, bv2)
    if (nedge) then
      drn = drs
    else
      drn = selm(yp2 - yp1, bv3)
    endif
    qxw = (drw - dlw) / 6.0_rp
    qxe = (dre - dcw) / 6.0_rp
    qys = (drs - dls) / 6.0_rp
    qyn = (drn - dcs) / 6.0_rp
  end subroutine c4_div

  DEVICE_ATTR
  subroutine c4_faces(t0, xm1, xp1, ym1, yp1, bu1, bu2, bv1, bv2, qxw, qxe, qys, qyn, fu1, fu2, fv1, fv2, &
                      wedge, eedge, sedge, nedge, usc, hdiv)
    ! fxe - fxw + fen - fes with the C4 faces; usc: the UPSCALING term on
    ! the faces at the physical edges
    real(rp), intent(in) :: t0, xm1, xp1, ym1, yp1, qxw, qxe, qys, qyn, fu1, fu2, fv1, fv2
    logical, intent(in) :: bu1, bu2, bv1, bv2, wedge, eedge, sedge, nedge, usc
    real(rp), intent(out) :: hdiv
    real(rp) :: fxw, fxe, fes, fen
    fxw = selm(0.5_rp * (t0 + xm1 - qxw) * fu1, bu1)
    fxe = selm(0.5_rp * (xp1 + t0 - qxe) * fu2, bu2)
    fes = selm(0.5_rp * (t0 + ym1 - qys) * fv1, bv1)
    fen = selm(0.5_rp * (yp1 + t0 - qyn) * fv2, bv2)
    if (usc) then
      if (wedge) fxw = fxw + selm(0.5_rp * (xm1 - t0) * max(fu1, 0.0_rp), bu1)
      if (eedge) fxe = fxe + selm(0.5_rp * (xp1 - t0) * min(fu2, 0.0_rp), bu2)
      if (sedge) fes = fes + selm(0.5_rp * (ym1 - t0) * max(fv1, 0.0_rp), bv1)
      if (nedge) fen = fen + selm(0.5_rp * (yp1 - t0) * min(fv2, 0.0_rp), bv2)
    endif
    hdiv = fxe - fxw + fen - fes
  end subroutine c4_faces

  DEVICE_ATTR
  subroutine up3_div(t0, xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2, wedge, eedge, sedge, nedge, &
                     um0, um1, um2, um3, vm0, vm1, vm2, vm3, bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, &
                     brw, brc, bre, brs, brn, fu1, fu2, fv1, fv2, usc, hdiv)
    ! fxe - fxw + fen - fes with the UP3 faces; usc: the UPSCALING term on
    ! the faces at the physical edges
    real(rp), intent(in) :: t0, xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2, fu1, fu2, fv1, fv2
    real(rp), intent(in) :: um0, um1, um2, um3, vm0, vm1, vm2, vm3
    logical, intent(in) :: wedge, eedge, sedge, nedge, bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, usc
    logical, intent(in) :: brw, brc, bre, brs, brn
    real(rp), intent(out) :: hdiv
    real(rp) :: dcw, dlw, drw, dre, fxw, fxe, fes, fen, face_t
    dcw = selm(t0 - xm1, bu1)
    if (wedge) then
      dlw = dcw
    else
      dlw = selm(xm1 - xm2, bu0)
    endif
    drw = selm(xp1 - t0, bu2)
    if (brw .and. brc) then
      face_t = 0.5_rp * (xm1 + t0)
    elseif (brw) then
      face_t = xm1
    elseif (brc) then
      face_t = t0
    else
      face_t = 0.0_rp
    endif
    fxw = selm(face_t * fu1 - coef_up3 * ( &
          (dcw - dlw) * um0 * um1 * max(fu1, 0.0_rp) + (drw - dcw) * um1 * um2 * min(fu1, 0.0_rp)), bu1)
    if (eedge) then
      dre = drw
    else
      dre = selm(xp2 - xp1, bu3)
    endif
    if (brc .and. bre) then
      face_t = 0.5_rp * (t0 + xp1)
    elseif (brc) then
      face_t = t0
    elseif (bre) then
      face_t = xp1
    else
      face_t = 0.0_rp
    endif
    fxe = selm(face_t * fu2 - coef_up3 * ( &
          (drw - dcw) * um1 * um2 * max(fu2, 0.0_rp) + (dre - drw) * um2 * um3 * min(fu2, 0.0_rp)), bu2)
    dcw = selm(t0 - ym1, bv1)
    if (sedge) then
      dlw = dcw
    else
      dlw = selm(ym1 - ym2, bv0)
    endif
    drw = selm(yp1 - t0, bv2)
    if (brs .and. brc) then
      face_t = 0.5_rp * (ym1 + t0)
    elseif (brs) then
      face_t = ym1
    elseif (brc) then
      face_t = t0
    else
      face_t = 0.0_rp
    endif
    fes = selm(face_t * fv1 - coef_up3 * ( &
          (dcw - dlw) * vm0 * vm1 * max(fv1, 0.0_rp) + (drw - dcw) * vm1 * vm2 * min(fv1, 0.0_rp)), bv1)
    if (nedge) then
      dre = drw
    else
      dre = selm(yp2 - yp1, bv3)
    endif
    if (brc .and. brn) then
      face_t = 0.5_rp * (t0 + yp1)
    elseif (brc) then
      face_t = t0
    elseif (brn) then
      face_t = yp1
    else
      face_t = 0.0_rp
    endif
    fen = selm(face_t * fv2 - coef_up3 * ( &
          (drw - dcw) * vm1 * vm2 * max(fv2, 0.0_rp) + (dre - drw) * vm2 * vm3 * min(fv2, 0.0_rp)), bv2)
    if (usc) then
      if (wedge) fxw = fxw + selm(0.5_rp * (xm1 - t0) * max(fu1, 0.0_rp), bu1)
      if (eedge) fxe = fxe + selm(0.5_rp * (xp1 - t0) * min(fu2, 0.0_rp), bu2)
      if (sedge) fes = fes + selm(0.5_rp * (ym1 - t0) * max(fv1, 0.0_rp), bv1)
      if (nedge) fen = fen + selm(0.5_rp * (yp1 - t0) * min(fv2, 0.0_rp), bv2)
    endif
    hdiv = fxe - fxw + fen - fes
  end subroutine up3_div

  DEVICE_ATTR
  subroutine stencil_cols(ci, i, nx, ncl, nwet, nbt_, nb8)
    ! Columns of the x and y stencil neighbours of wet column ci, all valid
    ! columns (0 = the zero column). x neighbours: wet cells of a row are
    ! consecutive columns, and a neighbour's value is used only when the
    ! cells up to it are wet (the faces are masked otherwise), so ci -/+ d
    ! is exact whenever it matters and a finite tracer value when it does
    ! not. Halo neighbours (outside 1:nx) and the y neighbours come from the
    ! table.
    integer(ip), intent(in) :: ci, i, nx, ncl, nwet
    integer(ip), intent(in) :: nbt_(8, nwet)
    integer(ip), intent(out) :: nb8(8)
    if (i - 2 >= 1) then
      nb8(1) = max(ci - 2, 0_ip)
    else
      nb8(1) = nbt_(1, ci)
    endif
    if (i - 1 >= 1) then
      nb8(2) = ci - 1
    else
      nb8(2) = nbt_(2, ci)
    endif
    if (i + 1 <= nx) then
      nb8(3) = min(ci + 1, ncl)
    else
      nb8(3) = nbt_(3, ci)
    endif
    if (i + 2 <= nx) then
      nb8(4) = min(ci + 2, ncl)
    else
      nb8(4) = nbt_(4, ci)
    endif
    nb8(5) = nbt_(5, ci); nb8(6) = nbt_(6, ci); nb8(7) = nbt_(7, ci); nb8(8) = nbt_(8, ci)
  end subroutine stencil_cols

  ! ------------- horizontal steps, one thread per (tracer, wet column, block of levels)
  DEVICE_ATTR
  subroutine c4_col(itr, ci, i, j, k0, k1, cs, cb, dtp, ntr, nx, ny, is, ie, js, je, nz, &
                    west, east, south, north, tps, tpo, tpn, tpk, nbt_, ncl, nwet, Hz_, Hzb_, pm_, pn_, &
                    FlxU_, FlxV_, umask_, vmask_)
    ! LF-AM3 predictor at levels k0:k1 with C4 faces from tps (slot ts),
    ! old field tpo (slot ns), saved Hz-weighted into tpk (slot nb, read
    ! first), result into tpn (slot nn). The column quantities (stencil
    ! columns, mask tests, edges, dtp*pm*pn) are formed once for all levels;
    ! the operands needed after the divisions are loaded one level ahead.
    integer(ip), intent(in) :: itr, ci, i, j, k0, k1, ntr, nx, ny, is, ie, js, je, nz
    integer(ip), intent(in) :: west, east, south, north
    real(rp), intent(in) :: cs, cb, dtp
    integer(ip), intent(in) :: ncl, nwet, nbt_(8, nwet)
    real(pp), intent(in) :: tps(ntr, 0:ncl, nz), tpo(ntr, 0:ncl, nz)
    real(pp), intent(inout) :: tpn(ntr, 0:ncl, nz), tpk(ntr, 0:ncl, nz)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), Hzb_(is:ie, js:je, nz), FlxU_(is:ie, js:je, nz), FlxV_(is:ie, js:je, nz)
    real(rp), intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    real(rp) :: dtpp, t0, hdiv, told, tbk, hb, hzc, fu1, fu2, fv1, fv2
    real(rp) :: xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2, qxw, qxe, qys, qyn
    logical :: wedge, eedge, sedge, nedge, bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, usc
    integer(ip) :: nb8(8), k
    call stencil_cols(ci, i, nx, ncl, nwet, nbt_, nb8)
    dtpp = dtp * pm_(i, j) * pn_(i, j)
    bu0 = umask_(i - 1, j) > 0.5_rp; bu1 = umask_(i, j) > 0.5_rp
    bu2 = umask_(i + 1, j) > 0.5_rp; bu3 = umask_(i + 2, j) > 0.5_rp
    bv0 = vmask_(i, j - 1) > 0.5_rp; bv1 = vmask_(i, j) > 0.5_rp
    bv2 = vmask_(i, j + 1) > 0.5_rp; bv3 = vmask_(i, j + 2) > 0.5_rp
    wedge = (west >= 1 .and. i == 1)
    eedge = (east >= 1 .and. i == nx)
    sedge = (south >= 1 .and. j == 1)
    nedge = (north >= 1 .and. j == ny)
    usc = (west == 2 .or. east == 2 .or. south == 2 .or. north == 2)
    k = k0
    told = real(tpo(itr, ci, k), rp); tbk = real(tpk(itr, ci, k), rp)
    fu1 = FlxU_(i, j, k); fu2 = FlxU_(i + 1, j, k)
    fv1 = FlxV_(i, j, k); fv2 = FlxV_(i, j + 1, k)
    hb = Hzb_(i, j, k); hzc = Hz_(i, j, k)
    do k = k0, k1
      t0 = real(tps(itr, ci, k), rp)
      xm2 = real(tps(itr, nb8(1), k), rp); xm1 = real(tps(itr, nb8(2), k), rp)
      xp1 = real(tps(itr, nb8(3), k), rp); xp2 = real(tps(itr, nb8(4), k), rp)
      ym2 = real(tps(itr, nb8(5), k), rp); ym1 = real(tps(itr, nb8(6), k), rp)
      yp1 = real(tps(itr, nb8(7), k), rp); yp2 = real(tps(itr, nb8(8), k), rp)
      call c4_div(t0, xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2, wedge, eedge, sedge, nedge, &
                  bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, qxw, qxe, qys, qyn)
      call c4_faces(t0, xm1, xp1, ym1, yp1, bu1, bu2, bv1, bv2, qxw, qxe, qys, qyn, fu1, fu2, fv1, fv2, &
                    wedge, eedge, sedge, nedge, usc, hdiv)
      tpn(itr, ci, k) = real(hb * (cs * told + cb * tbk) - dtpp * hdiv, pp)
      tpk(itr, ci, k) = real(hzc * told, pp)
      if (k < k1) then
        told = real(tpo(itr, ci, k + 1), rp); tbk = real(tpk(itr, ci, k + 1), rp)
        fu1 = FlxU_(i, j, k + 1); fu2 = FlxU_(i + 1, j, k + 1)
        fv1 = FlxV_(i, j, k + 1); fv2 = FlxV_(i, j + 1, k + 1)
        hb = Hzb_(i, j, k + 1); hzc = Hz_(i, j, k + 1)
      endif
    enddo
  end subroutine c4_col

  DEVICE_ATTR
  subroutine up3_col(itr, ci, i, j, k0, k1, dtp, ntr, nx, ny, is, ie, js, je, nz, &
                     west, east, south, north, tps, tpn, nbt_, ncl, nwet, pm_, pn_, FlxU_, FlxV_, &
                     umask_, vmask_, rmask_)
    ! Corrector at levels k0:k1: UP3 faces from tps (slot ts) applied to tpn
    ! (slot nn). The 3-D arrays are addressed through flat 64-bit offsets
    ! (column part once, one stride per level).
    integer(ip), intent(in) :: itr, ci, i, j, k0, k1, ntr, nx, ny, is, ie, js, je, nz
    integer(ip), intent(in) :: west, east, south, north
    real(rp), intent(in) :: dtp
    integer(ip), intent(in) :: ncl, nwet, nbt_(8, nwet)
    real(pp), intent(in) :: tps(*)
    real(pp), intent(inout) :: tpn(*)
    real(rp), intent(in) :: FlxU_(*), FlxV_(*)
    real(rp), intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp) :: dtpp, t0, hdiv, tnn, fu1, fu2, fv1, fv2
    real(rp) :: xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2
    real(rp) :: um0, um1, um2, um3, vm0, vm1, vm2, vm3
    logical :: wedge, eedge, sedge, nedge, bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, usc
    logical :: brw, brc, bre, brs, brn
    integer(ip) :: nb8(8), k
    integer(8) :: tk, ts0, o1, o2, o3, o4, o5, o6, o7, o8, gk, g0, nxa
    call stencil_cols(ci, i, nx, ncl, nwet, nbt_, nb8)
    dtpp = dtp * pm_(i, j) * pn_(i, j)
    um0 = umask_(i - 1, j); um1 = umask_(i, j); um2 = umask_(i + 1, j); um3 = umask_(i + 2, j)
    vm0 = vmask_(i, j - 1); vm1 = vmask_(i, j); vm2 = vmask_(i, j + 1); vm3 = vmask_(i, j + 2)
    bu0 = um0 > 0.5_rp; bu1 = um1 > 0.5_rp; bu2 = um2 > 0.5_rp; bu3 = um3 > 0.5_rp
    bv0 = vm0 > 0.5_rp; bv1 = vm1 > 0.5_rp; bv2 = vm2 > 0.5_rp; bv3 = vm3 > 0.5_rp
    brw = rmask_(i - 1, j) > 0.5_rp; brc = rmask_(i, j) > 0.5_rp; bre = rmask_(i + 1, j) > 0.5_rp
    brs = rmask_(i, j - 1) > 0.5_rp; brn = rmask_(i, j + 1) > 0.5_rp
    wedge = (west >= 1 .and. i == 1)
    eedge = (east >= 1 .and. i == nx)
    sedge = (south >= 1 .and. j == 1)
    nedge = (north >= 1 .and. j == ny)
    usc = (west == 2 .or. east == 2 .or. south == 2 .or. north == 2)
    ! tp(itr, c, k) = tps(itr + ntr*c + ntr*(ncl+1)*(k-1)); F(i, j, k) = F(g0 + nxa*nya*(k-1))
    tk = int(ntr, 8) * int(ncl + 1, 8)
    ts0 = int(itr, 8) + int(ntr, 8) * int(ci, 8) + tk * int(k0 - 1, 8)
    o1 = int(ntr, 8) * int(nb8(1) - ci, 8); o2 = int(ntr, 8) * int(nb8(2) - ci, 8)
    o3 = int(ntr, 8) * int(nb8(3) - ci, 8); o4 = int(ntr, 8) * int(nb8(4) - ci, 8)
    o5 = int(ntr, 8) * int(nb8(5) - ci, 8); o6 = int(ntr, 8) * int(nb8(6) - ci, 8)
    o7 = int(ntr, 8) * int(nb8(7) - ci, 8); o8 = int(ntr, 8) * int(nb8(8) - ci, 8)
    nxa = int(ie - is + 1, 8)
    gk = nxa * int(je - js + 1, 8)
    g0 = int(i - is + 1, 8) + nxa * int(j - js, 8) + gk * int(k0 - 1, 8)
    do k = k0, k1
      t0 = real(tps(ts0), rp)
      xm2 = real(tps(ts0 + o1), rp); xm1 = real(tps(ts0 + o2), rp)
      xp1 = real(tps(ts0 + o3), rp); xp2 = real(tps(ts0 + o4), rp)
      ym2 = real(tps(ts0 + o5), rp); ym1 = real(tps(ts0 + o6), rp)
      yp1 = real(tps(ts0 + o7), rp); yp2 = real(tps(ts0 + o8), rp)
      tnn = real(tpn(ts0), rp)
      fu1 = FlxU_(g0); fu2 = FlxU_(g0 + 1)
      fv1 = FlxV_(g0); fv2 = FlxV_(g0 + nxa)
      call up3_div(t0, xm2, xm1, xp1, xp2, ym2, ym1, yp1, yp2, wedge, eedge, sedge, nedge, &
                   um0, um1, um2, um3, vm0, vm1, vm2, vm3, bu0, bu1, bu2, bu3, bv0, bv1, bv2, bv3, &
                   brw, brc, bre, brs, brn, fu1, fu2, fv1, fv2, usc, hdiv)
      tpn(ts0) = real(tnn - dtpp * hdiv, pp)
      ts0 = ts0 + tk
      g0 = g0 + gk
    enddo
  end subroutine up3_col

  ! ------------------------------ lateral diffusion, one thread per (tracer, cell)
  DEVICE_ATTR
  subroutine t3dmix_point(itr, ci, i, j, k, nr, nn, dtl, ntr, nx, is, ie, js, je, nz, nslot, &
                          tp_, nbt_, ncl, nwet, Hz_, pm_, pn_, pmon_u_, pnom_v_, umask_, vmask_, sp_, tnu_)
    ! Harmonic diffusion along s-surfaces, faces computed inline (same
    ! expressions as the former passive_t3dmix). The diffusivity of tracer
    ! itr is tnu_(itr) + sp_(i, j), its background tnu2 plus the sponge ramp:
    ! the CPU's diff2(i, j, itrc), summed in the same order (mixing.F90 then
    ! set_nudgcof).
    integer(ip), intent(in) :: itr, ci, i, j, k, nr, nn, ntr, nx, is, ie, js, je, nz, nslot
    real(rp), intent(in) :: dtl
    integer(ip), intent(in) :: ncl, nwet, nbt_(8, nwet)
    real(pp), intent(inout) :: tp_(ntr, 0:ncl, nz, nslot)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), pm_(is:ie, js:je), pn_(is:ie, js:je)
    real(rp), intent(in) :: pmon_u_(is:ie, js:je), pnom_v_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    real(rp), intent(in) :: sp_(is:ie, js:je), tnu_(ntr)
    real(rp) :: fxw, fxe, fes, fen, bg
    bg = tnu_(itr)
    fxw = apply_mask(0.25_rp * ((bg + sp_(i, j)) + (bg + sp_(i - 1, j))) * &
         pmon_u_(i, j) * (Hz_(i, j, k) + Hz_(i - 1, j, k)) * &
         (real(tp_(itr, ci, k, nr), rp) - real(tp_(itr, nbt_(2, ci), k, nr), rp)), umask_(i, j))
    fxe = apply_mask(0.25_rp * ((bg + sp_(i + 1, j)) + (bg + sp_(i, j))) * &
         pmon_u_(i + 1, j) * (Hz_(i + 1, j, k) + Hz_(i, j, k)) * &
         (real(tp_(itr, nbt_(3, ci), k, nr), rp) - real(tp_(itr, ci, k, nr), rp)), umask_(i + 1, j))
    fes = apply_mask(0.25_rp * ((bg + sp_(i, j)) + (bg + sp_(i, j - 1))) * &
         pnom_v_(i, j) * (Hz_(i, j, k) + Hz_(i, j - 1, k)) * &
         (real(tp_(itr, ci, k, nr), rp) - real(tp_(itr, nbt_(6, ci), k, nr), rp)), vmask_(i, j))
    fen = apply_mask(0.25_rp * ((bg + sp_(i, j + 1)) + (bg + sp_(i, j))) * &
         pnom_v_(i, j + 1) * (Hz_(i, j + 1, k) + Hz_(i, j, k)) * &
         (real(tp_(itr, nbt_(7, ci), k, nr), rp) - real(tp_(itr, ci, k, nr), rp)), vmask_(i, j + 1))
    tp_(itr, ci, k, nn) = real(real(tp_(itr, ci, k, nn), rp) + dtl * pm_(i, j) * pn_(i, j) * &
         (fxe - fxw + fen - fes) / max(Hz_(i, j, k), 1.0e-30_rp), pp)
  end subroutine t3dmix_point

  ! ------------------- vertical advection + implicit solve, one thread per (tracer, column)
  DEVICE_ATTR
  subroutine column_col(itr, ci, i, j, ctr, dtp, ntr, is, ie, js, je, nz, nts, isl, mir, &
                        tps, tpn, ncl, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_, sfo, psf_)
    ! The vertical flux divergence (vert_column_fused arithmetic, sliding
    ! window over k) feeds the Thomas forward sweep level by level, with
    ! the intermediate dc kept in place in tpn (slot nn); then the back
    ! substitution. tps = slot ts (the advected field, not written).
    ! ctr = 1: centred (predictor), 0: upwind-weighted (corrector). mir = 1:
    ! tracer 1 receives salinity's surface flux and the KPP nonlocal term
    ! between the vertical advection and the solve. sfo = 1: every other
    ! tracer receives its surface flux psf_(itr, ci) (CDR_LITE; 0 for most
    ! tracers) on the top layer at the same point, dt*stflx as the CPU's
    ! step3d_t. The face values are
    ! thickness_tplus/thickness_tminus with their shared differences carried
    ! between levels: dlt_m(k) = dlt(k-1) and dlt(k) = dlt_p(k-1) (same
    ! operands in the same order). The next level's tpn value is loaded
    ! before this level's store.
    integer(ip), intent(in) :: itr, ci, i, j, ctr, ntr, is, ie, js, je, nz, nts, isl, mir, ncl
    real(rp), intent(in) :: dtp
    real(pp), intent(in) :: tps(ntr, 0:ncl, nz)
    real(pp), intent(inout) :: tpn(ntr, 0:ncl, nz)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), We_(is:ie, js:je, 0:nz)
    real(rp), intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), intent(in) :: stflx_(is:ie, js:je, nts), ghat_(is:ie, js:je, 0:nz)
    real(rp), intent(in) :: fc_(is:ie, js:je, 0:nz), cf_(is:ie, js:je, 0:nz), wc_(is:ie, js:je, 0:nz)
    integer(ip), intent(in) :: sfo
    real(rp), intent(in) :: psf_(ntr, *)
    integer(ip) :: k
    real(rp) :: dtpp, rm, w, sfx, cffk, cffm, tt
    real(rp) :: Hz_km1, Hz_k, Hz_kp1, Hz_kp2, fck, wcm, cfk
    real(rp) :: fz_lo, fz_k, tplus, tminus, rhs, dck, tnk
    real(rp) :: t_km1, t_k, t_kp1, t_kp2
    real(rp) :: dlt, dlt_m, dlt_p, tlin, hkm, hkp2
    logical :: at_bottom, at_surface
    dtpp = dtp * pm_(i, j) * pn_(i, j)
    rm = rmask_(i, j)
    sfx = 0.0_rp
    if (sfo == 1) sfx = psf_(itr, ci)
    if (mir == 1 .and. itr == 1) sfx = stflx_(i, j, isl)
    Hz_k = Hz_(i, j, 1)
    Hz_kp1 = Hz_(i, j, 2)
    Hz_km1 = Hz_k
    Hz_kp2 = Hz_kp1
    w = 0.0_rp
    t_k = real(tps(itr, ci, 1), rp)
    t_kp1 = real(tps(itr, ci, 2), rp)
    t_km1 = t_k
    t_kp2 = t_kp1
    fz_lo = 0.0_rp
    dck = 0.0_rp
    wcm = 0.0_rp
    dlt = 0.0_rp
    dlt_p = 0.0_rp
    tnk = real(tpn(itr, ci, 1), rp)
    ! ---- bottom to surface: vertical fluxes and forward elimination
    do k = 1, nz
      if (k >= 2 .and. k <= nz - 2) then
        ! interior level: the general branch of every test below
        Hz_kp2 = Hz_(i, j, k + 2)
        w = We_(i, j, k)
        fck = fc_(i, j, k)
        t_kp2 = real(tps(itr, ci, k + 2), rp)
        dlt_m = dlt
        dlt = dlt_p
        hkm = Hz_km1
        tlin = (Hz_kp1 * t_k + Hz_k * t_kp1) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
        tplus = tlin - Hz_k * Hz_kp1 * (dlt - dlt_m) / max(hkm + Hz_k + Hz_kp1, 1.0e-30_rp)
        hkp2 = Hz_kp2
        dlt_p = (t_kp2 - t_kp1) / max(Hz_kp1 + Hz_kp2, 1.0e-30_rp)
        tminus = tlin - Hz_k * Hz_kp1 * (dlt_p - dlt) / max(Hz_k + Hz_kp1 + hkp2, 1.0e-30_rp)
        if (ctr == 1) then
          fz_k = 0.5_rp * (tplus + tminus) * w
        else
          fz_k = tplus * max(w, 0.0_rp) + tminus * min(w, 0.0_rp)
        endif
        fz_k = apply_mask(fz_k, rm)
        rhs = real(tnk - dtpp * (fz_k - fz_lo), pp)
        fz_lo = fz_k
        t_km1 = t_k; t_k = t_kp1; t_kp1 = t_kp2
        if (mir == 1 .and. itr == 1) then
          cffk = -dtp * ghat_(i, j, k) * sfx
          rhs = real(rhs + cffk, pp)
          cffm = -dtp * ghat_(i, j, k - 1) * sfx
          rhs = real(rhs - cffm, pp)
        endif
        tnk = real(tpn(itr, ci, k + 1), rp)
        dck = fck * (rhs + dck * wcm)
        tpn(itr, ci, k) = real(dck, pp)
        wcm = wc_(i, j, k)
        Hz_km1 = Hz_k; Hz_k = Hz_kp1; Hz_kp1 = Hz_kp2
        cycle
      endif
      at_bottom = (k == 1)
      at_surface = (k == nz - 1)
      if (k < nz) then
        if (at_surface) then
          Hz_kp2 = Hz_kp1
        else
          Hz_kp2 = Hz_(i, j, k + 2)
        endif
        w = We_(i, j, k)
      endif
      fck = fc_(i, j, k)
      if (k < nz) then
        if (at_surface) then
          t_kp2 = t_kp1
        else
          t_kp2 = real(tps(itr, ci, k + 2), rp)
        endif
        if (at_bottom) then
          dlt = (t_kp1 - t_k) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
          dlt_m = 0.0_rp
          hkm = Hz_k
        else
          dlt_m = dlt
          dlt = dlt_p
          hkm = Hz_km1
        endif
        tlin = (Hz_kp1 * t_k + Hz_k * t_kp1) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
        tplus = tlin - Hz_k * Hz_kp1 * (dlt - dlt_m) / max(hkm + Hz_k + Hz_kp1, 1.0e-30_rp)
        if (at_surface) then
          hkp2 = Hz_kp1
          dlt_p = 0.0_rp
        else
          hkp2 = Hz_kp2
          dlt_p = (t_kp2 - t_kp1) / max(Hz_kp1 + Hz_kp2, 1.0e-30_rp)
        endif
        tminus = tlin - Hz_k * Hz_kp1 * (dlt_p - dlt) / max(Hz_k + Hz_kp1 + hkp2, 1.0e-30_rp)
        if (ctr == 1) then
          fz_k = 0.5_rp * (tplus + tminus) * w
        else
          fz_k = tplus * max(w, 0.0_rp) + tminus * min(w, 0.0_rp)
        endif
        fz_k = apply_mask(fz_k, rm)
      else
        fz_k = 0.0_rp
      endif
      rhs = real(tnk - dtpp * (fz_k - fz_lo), pp)
      fz_lo = fz_k
      t_km1 = t_k; t_k = t_kp1; t_kp1 = t_kp2
      if (mir == 1 .and. itr == 1) then
        ! salinity forcing of the mirror tracer, in the original order
        if (k == nz) rhs = real(rhs + dtp * sfx, pp)
        if (k < nz) then
          cffk = -dtp * ghat_(i, j, k) * sfx
          rhs = real(rhs + cffk, pp)
        endif
        if (k > 1) then
          cffm = -dtp * ghat_(i, j, k - 1) * sfx
          rhs = real(rhs - cffm, pp)
        endif
      else if (sfo == 1) then
        if (k == nz) rhs = real(rhs + dtp * sfx, pp)
      endif
      if (k < nz) tnk = real(tpn(itr, ci, k + 1), rp)
      if (k == 1) then
        dck = fck * rhs
        tpn(itr, ci, k) = real(dck, pp)
      else if (k < nz) then
        dck = fck * (rhs + dck * wcm)
        tpn(itr, ci, k) = real(dck, pp)
      else
        tt = apply_mask((rhs + dck * wcm) * fck, rm)
        tpn(itr, ci, k) = real(tt, pp)
        dck = tt
      endif
      if (k < nz) then
        wcm = wc_(i, j, k)
        Hz_km1 = Hz_k; Hz_k = Hz_kp1; Hz_kp1 = Hz_kp2
      endif
    enddo
    ! ---- surface to bottom: back substitution (dck holds the level above)
    if (nz > 1) tnk = real(tpn(itr, ci, nz - 1), rp)
    do k = nz - 1, 1, -1
      cfk = cf_(i, j, k)
      tt = apply_mask(tnk + cfk * dck, rm)
      if (k > 1) tnk = real(tpn(itr, ci, k - 1), rp)
      tpn(itr, ci, k) = real(tt, pp)
      dck = tt
    enddo
  end subroutine column_col

#ifdef USE_CUDA
  ! Every kernel of this module carries launch_bounds: with nvfortran 25.9 a
  ! launch_bounds anywhere in the module also changes the register limit of
  ! the kernels without one (t3dmix: (128, 10) keeps its former 48 registers
  ! and code). Measured best: C4 (128, 4), UP3 (128, 5), column (128, 7).
  attributes(global) launch_bounds(128, 4) subroutine c4_kernel(nwet, nlev, wet_i, wet_j, cs, cb, dtp, ntr, nx, ny, nz, is, ie, js, je, &
                                          west, east, south, north, tps, tpo, tpn, tpk, nbt_, ncl, &
                                          Hz_, Hzb_, pm_, pn_, FlxU_, FlxV_, umask_, vmask_)
    ! block (ntr, cells), grid (cells, level blocks of nlev)
    integer(ip), value :: nwet, nlev, ntr, nx, ny, nz, is, ie, js, je, west, east, south, north, ncl
    real(rp), value :: cs, cb, dtp
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip), device, intent(in) :: nbt_(8, nwet)
    real(pp), device, intent(in) :: tps(ntr, 0:ncl, nz), tpo(ntr, 0:ncl, nz)
    real(pp), device, intent(inout) :: tpn(ntr, 0:ncl, nz), tpk(ntr, 0:ncl, nz)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), Hzb_(is:ie, js:je, nz), FlxU_(is:ie, js:je, nz), FlxV_(is:ie, js:je, nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    integer(ip) :: cc, itr, i, j, k0, k1
    cc = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (cc > nwet) return
    itr = threadIdx%x
    k0 = (blockIdx%y - 1) * nlev + 1
    k1 = min(k0 + nlev - 1, nz)
    i = wet_i(cc); j = wet_j(cc)
    call c4_col(itr, cc, i, j, k0, k1, cs, cb, dtp, ntr, nx, ny, is, ie, js, je, nz, &
                west, east, south, north, tps, tpo, tpn, tpk, nbt_, ncl, nwet, Hz_, Hzb_, pm_, pn_, &
                FlxU_, FlxV_, umask_, vmask_)
  end subroutine c4_kernel

  attributes(global) launch_bounds(128, 5) subroutine up3_kernel(nwet, nlev, wet_i, wet_j, dtp, ntr, nx, ny, nz, is, ie, js, je, &
                                           west, east, south, north, tps, tpn, nbt_, ncl, &
                                           pm_, pn_, FlxU_, FlxV_, umask_, vmask_, rmask_)
    integer(ip), value :: nwet, nlev, ntr, nx, ny, nz, is, ie, js, je, west, east, south, north, ncl
    real(rp), value :: dtp
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip), device, intent(in) :: nbt_(8, nwet)
    real(pp), device, intent(in) :: tps(ntr, 0:ncl, nz)
    real(pp), device, intent(inout) :: tpn(ntr, 0:ncl, nz)
    real(rp), device, intent(in) :: FlxU_(is:ie, js:je, nz), FlxV_(is:ie, js:je, nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je), rmask_(is:ie, js:je)
    integer(ip) :: cc, itr, i, j, k0, k1
    cc = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (cc > nwet) return
    itr = threadIdx%x
    k0 = (blockIdx%y - 1) * nlev + 1
    k1 = min(k0 + nlev - 1, nz)
    i = wet_i(cc); j = wet_j(cc)
    call up3_col(itr, cc, i, j, k0, k1, dtp, ntr, nx, ny, is, ie, js, je, nz, &
                 west, east, south, north, tps, tpn, nbt_, ncl, nwet, pm_, pn_, FlxU_, FlxV_, &
                 umask_, vmask_, rmask_)
  end subroutine up3_kernel

  attributes(global) launch_bounds(128, 10) subroutine t3dmix_kernel(nel, nwet, nmx, mixc_, wet_i, wet_j, nr, nn, dtl, ntr, nx, ny, nz, is, ie, js, je, nslot, &
                                              tp_, nbt_, ncl, Hz_, pm_, pn_, pmon_u_, pnom_v_, umask_, vmask_, sp_, tnu_)
    integer(ip), value :: nel, nr, nn, ntr, nx, ny, nz, is, ie, js, je, nslot
    integer(ip), value :: nwet, nmx
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*), mixc_(*)
    integer(ip) :: cc
    real(rp), value :: dtl
    integer(ip), value :: ncl
    integer(ip), device, intent(in) :: nbt_(8, nwet)
    real(pp), device, intent(inout) :: tp_(ntr, 0:ncl, nz, nslot)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), pm_(is:ie, js:je), pn_(is:ie, js:je)
    real(rp), device, intent(in) :: pmon_u_(is:ie, js:je), pnom_v_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    real(rp), device, intent(in) :: sp_(is:ie, js:je), tnu_(ntr)
    integer(ip) :: tid, c, itr, i, j, k
    tid = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (tid > nel) return
    itr = mod(tid - 1, ntr) + 1; c = (tid - 1) / ntr
    cc = mixc_(mod(c, nmx) + 1); k = c / nmx + 1
    i = wet_i(cc); j = wet_j(cc)
    call t3dmix_point(itr, cc, i, j, k, nr, nn, dtl, ntr, nx, is, ie, js, je, nz, nslot, &
                      tp_, nbt_, ncl, nwet, Hz_, pm_, pn_, pmon_u_, pnom_v_, umask_, vmask_, sp_, tnu_)
  end subroutine t3dmix_kernel

  attributes(global) launch_bounds(128, 7) subroutine column_kernel(nwet, wet_i, wet_j, ctr, dtp, ntr, nz, is, ie, js, je, nts, isl, mir, &
                                              tps, tpn, ncl, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_, sfo, psf_)
    ! block (ntr, columns), grid (columns)
    integer(ip), value :: nwet, ctr, ntr, nz, is, ie, js, je, nts, isl, mir, ncl
    real(rp), value :: dtp
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    real(pp), device, intent(in) :: tps(ntr, 0:ncl, nz)
    real(pp), device, intent(inout) :: tpn(ntr, 0:ncl, nz)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), We_(is:ie, js:je, 0:nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), device, intent(in) :: stflx_(is:ie, js:je, nts), ghat_(is:ie, js:je, 0:nz)
    real(rp), device, intent(in) :: fc_(is:ie, js:je, 0:nz), cf_(is:ie, js:je, 0:nz), wc_(is:ie, js:je, 0:nz)
    integer(ip), value :: sfo
    real(rp), device, intent(in) :: psf_(ntr, *)
    integer(ip) :: cc, itr, i, j
    cc = (blockIdx%x - 1) * blockDim%y + threadIdx%y
    if (cc > nwet) return
    itr = threadIdx%x
    i = wet_i(cc); j = wet_j(cc)
    call column_col(itr, cc, i, j, ctr, dtp, ntr, is, ie, js, je, nz, nts, isl, mir, &
                    tps, tpn, ncl, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_, sfo, psf_)
  end subroutine column_kernel
#endif

  ! ------------------------------------------------------------ launchers
  subroutine passive_horiz_run(tp, ntr, mode, ts, ns, nn, nb, cs, cb, dtp)
    ! mode 1: LF-AM3 predictor with C4 faces from slot ts, old field in slot
    ! ns, saved Hz-weighted into slot nb, result into slot nn (ts and ns may
    ! be the same slot; nn and nb differ from both). mode 2: corrector, UP3
    ! faces from slot ts applied to slot nn (nn /= ts). Each slot is passed
    ! as its own array (sequence association, no copy).
    integer(ip), intent(in) :: ntr, mode, ts, ns, nn, nb
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), intent(in) :: cs, cb, dtp
    integer(ip) :: nx, ny, nz, is, ie, js, je, west, east, south, north
#ifdef USE_CUDA
    integer(ip) :: ncb
    type(dim3) :: grid, block
#else
    integer(ip) :: itr, c
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    ! edge codes: 0 interior, 1 physical edge, 2 edge with the UPSCALING term
    west  = merge(1_ip, 0_ip, decomp%edge_w) + merge(1_ip, 0_ip, decomp%edge_w .and. uscl_obc)
    east  = merge(1_ip, 0_ip, decomp%edge_e) + merge(1_ip, 0_ip, decomp%edge_e .and. uscl_obc)
    south = merge(1_ip, 0_ip, decomp%edge_s) + merge(1_ip, 0_ip, decomp%edge_s .and. uscl_obc)
    north = merge(1_ip, 0_ip, decomp%edge_n) + merge(1_ip, 0_ip, decomp%edge_n .and. uscl_obc)
    if (nwet <= 0) return
#ifdef USE_CUDA
    if (ntr > 128) error stop 'passive_horiz_run: ntr > 128 (kernel block size)'
    ncb = max(1_ip, 128_ip / ntr)
    block = dim3(ntr, ncb, 1)
    grid = dim3((nwet + ncb - 1) / ncb, (nz + HLEV - 1) / HLEV, 1)
    if (mode == 1) then
      call c4_kernel<<<grid, block>>>(nwet, HLEV, wet_i, wet_j, cs, cb, dtp, ntr, nx, ny, nz, is, ie, js, je, &
           west, east, south, north, tp(1, 0, 1, ts), tp(1, 0, 1, ns), tp(1, 0, 1, nn), tp(1, 0, 1, nb), &
           tpnb, ntpcol, Hz, Hz_bak, pm, pn, FlxU, FlxV, umask, vmask)
    else
      call up3_kernel<<<grid, block>>>(nwet, HLEV, wet_i, wet_j, dtp, ntr, nx, ny, nz, is, ie, js, je, &
           west, east, south, north, tp(1, 0, 1, ts), tp(1, 0, 1, nn), &
           tpnb, ntpcol, pm, pn, FlxU, FlxV, umask, vmask, rmask)
    endif
#else
    do c = 1, nwet; do itr = 1, ntr
      if (mode == 1) then
        call c4_col(itr, c, wet_i(c), wet_j(c), 1_ip, nz, cs, cb, dtp, ntr, nx, ny, is, ie, js, je, nz, &
                    west, east, south, north, tp(1, 0, 1, ts), tp(1, 0, 1, ns), tp(1, 0, 1, nn), tp(1, 0, 1, nb), &
                    tpnb, ntpcol, nwet, Hz, Hz_bak, pm, pn, FlxU, FlxV, umask, vmask)
      else
        call up3_col(itr, c, wet_i(c), wet_j(c), 1_ip, nz, dtp, ntr, nx, ny, is, ie, js, je, nz, &
                     west, east, south, north, tp(1, 0, 1, ts), tp(1, 0, 1, nn), &
                     tpnb, ntpcol, nwet, pm, pn, FlxU, FlxV, umask, vmask, rmask)
      endif
    enddo; enddo
#endif
  end subroutine passive_horiz_run

  subroutine passive_t3dmix_run(tp, ntr, nr, nn, dtl, tnu)
    ! tnu(ntr): each tracer's background tnu2 (passive_mod tnu2_tp).
    integer(ip), intent(in) :: ntr, nr, nn
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), intent(in) :: dtl
    real(rp), WORK_ATTR intent(in) :: tnu(ntr)
    integer(ip) :: nx, ny, nz, is, ie, js, je, nel
#ifndef USE_CUDA
    integer(ip) :: itr, i, j, k
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    ! only the wet cells with a nonzero diffusivity within 2 cells (mixc):
    ! elsewhere the update is +-0 (gpu_state build_mix_map)
    nel = ntr * nmix * nz
    if (nel <= 0) return
#ifdef USE_CUDA
    call t3dmix_kernel<<<(nel + 127) / 128, 128>>>(nel, nwet, nmix, mixc, wet_i, wet_j, nr, nn, dtl, ntr, nx, ny, nz, is, ie, js, je, 3_ip, &
         tp, tpnb, ntpcol, Hz, pm, pn, pmon_u, pnom_v, umask, vmask, diff2_sp, tnu)
#else
    do k = 1, nz; do j = 1, ny; do i = 1, nx; do itr = 1, ntr
      if (tpcol(i, j) == 0) cycle
      call t3dmix_point(itr, tpcol(i, j), i, j, k, nr, nn, dtl, ntr, nx, is, ie, js, je, nz, 3_ip, &
                        tp, tpnb, ntpcol, nwet, Hz, pm, pn, pmon_u, pnom_v, umask, vmask, diff2_sp, tnu)
    enddo; enddo; enddo; enddo
#endif
  end subroutine passive_t3dmix_run

  subroutine passive_column_run(tp, ntr, ts, nn, dtp, ctr, use_fwd, mirror, sflx_on, psf)
    ! Factorization, then one thread per (tracer, column); slot ts is read
    ! (the advected field), slot nn (/= ts) updated in place. sflx_on: add
    ! the surface fluxes psf(itr, wet column) (the corrector's CDR_LITE).
    integer(ip), intent(in) :: ntr, ts, nn, ctr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), intent(in) :: dtp
    logical, intent(in) :: use_fwd, mirror, sflx_on
    real(rp), WORK_ATTR contiguous, intent(in) :: psf(:,:)
    integer(ip) :: nz, mir, is, ie, js, je, nts, isl, sfo
#ifdef USE_CUDA
    integer(ip) :: ncb
    type(dim3) :: grid, block
#else
    integer(ip) :: itr, c
#endif
    nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nts = int(size(stflx, 3), ip); isl = int(isalt, ip)
    mir = merge(1_ip, 0_ip, mirror)
    sfo = merge(1_ip, 0_ip, sflx_on)
    if (sfo == 1 .and. size(psf, 1) /= ntr) error stop 'passive_column_run: surface flux array shape'
    call passive_factor(dtp, use_fwd)
    if (nwet <= 0) return
#ifdef USE_CUDA
    if (ntr > 128) error stop 'passive_column_run: ntr > 128 (kernel block size)'
    ncb = max(1_ip, 128_ip / ntr)
    block = dim3(ntr, ncb, 1)
    grid = dim3((nwet + ncb - 1) / ncb, 1, 1)
    call column_kernel<<<grid, block>>>(nwet, wet_i, wet_j, ctr, dtp, ntr, nz, is, ie, js, je, nts, isl, mir, &
         tp(1, 0, 1, ts), tp(1, 0, 1, nn), ntpcol, Hz, We, pm, pn, rmask, stflx, ghat, col_fc, col_cf, col_wc, &
         sfo, psf)
#else
    do c = 1, nwet; do itr = 1, ntr
      call column_col(itr, c, wet_i(c), wet_j(c), ctr, dtp, ntr, is, ie, js, je, nz, nts, isl, mir, &
                      tp(1, 0, 1, ts), tp(1, 0, 1, nn), ntpcol, Hz, We, pm, pn, rmask, stflx, ghat, col_fc, col_cf, col_wc, &
                      sfo, psf)
    enddo; enddo
#endif
  end subroutine passive_column_run

end module passive_column_mod

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
  public :: bgcw_t3dbc, bgcw_reduce, bgcw_vs, bgcw_edge_r, bgcw_subq, bgcw_edge_tp, bgcw_rcol, bgcw_rbits
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
  function rivf(fm, ra, vel, rc, shd) result(ff)
    ! river_faces: the window's river tracer transport through a river face
    ! of level k, distributed over the column like the summed face
    ! transport, fm*RUA*V/rcol (shadow, shd = 1: the transported volume
    ! fm*V; no transport through the column: 0)
    real(rp), intent(in) :: fm, ra, vel, rc
    integer(ip), intent(in) :: shd
    real(rp) :: ff
    if (shd == 1 .or. rc == 0.0_rp) then
      if (shd == 1) then
        ff = fm * vel
      else
        ff = 0.0_rp
      endif
    else
      ff = fm * ra * vel / rc
    endif
  end function rivf

  DEVICE_ATTR
  function vsrc(cc0, k, itr, i, j, pmn, q0, q1, ia, id, ntr, nqd, nz, nx, ny, cpa, c3) result(cc)
    ! vpass's point sources of level k after the vertical swept flux (and
    ! the surface flux): 3D CDR forcing of ALK (ia) / DIC (id), then the
    ! releases of the cell's profiles q0..q1 in order, cc + (pm pn)*src
    real(rp), intent(in) :: cc0, pmn
    integer(ip), intent(in) :: k, itr, i, j, q0, q1, ia, id, ntr, nqd, nz, nx, ny
    real(rp), intent(in) :: cpa(ntr, nqd, nz), c3(nx, ny, nz, 2)
    real(rp) :: cc
    integer(ip) :: q
    cc = cc0
    if (itr == ia) cc = cc + pmn * c3(i, j, k, 1)
    if (itr == id) cc = cc + pmn * c3(i, j, k, 2)
    do q = q0, q1
      cc = cc + pmn * cpa(itr, q, k)
    enddo
  end function vsrc

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
#define BGCW_KNAME inner_kernel
#include "bgcw_inner.inc"
#undef BGCW_KNAME
#define BGCW_KNAME inner_kernel_src
#define BGCW_SRC
#include "bgcw_inner.inc"
#undef BGCW_SRC
#undef BGCW_KNAME

  ! ------------------------------------------------ H3: outer and update
#define BGCW_KNAME outer_kernel
#include "bgcw_outer.inc"
#undef BGCW_KNAME
#define BGCW_KNAME outer_kernel_src
#define BGCW_SRC
#include "bgcw_outer.inc"
#undef BGCW_SRC
#undef BGCW_KNAME

  ! ------------------------------------------- vertical, sources, solve
#define BGCW_KNAME vertical_kernel
#include "bgcw_vertical.inc"
#undef BGCW_KNAME
#define BGCW_KNAME vertical_kernel_src
#define BGCW_SRC
#include "bgcw_vertical.inc"
#undef BGCW_SRC
#undef BGCW_KNAME

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

  subroutine bgcw_inner(ntr, k0, k1, isub, mk, q, sxs, sys, VX, VY, vol, cxf, cyf, dnx, dny, fmk, &
                        shd, rbx, nrl, rl, wfu, wfv, rua, rcl)
    ! H2 for ntr tracers of q (tp layout, one slot), levels k0..k1 of
    ! sub-pass isub (levels with isub > mk(k) skipped); the columns with
    ! river faces (rl(1:nrl), bits rbx) by inner_kernel_src, with the
    ! window's river transport rua (shd = 1: shadow)
    integer(ip), intent(in) :: ntr, k0, k1, isub, shd, nrl
    integer(ip), WORK_ATTR intent(in) :: rbx(*), rl(*), wfu(*), wfv(*)
    real(rp), WORK_ATTR intent(in) :: rua(*), rcl(*)
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
         vol, cxf, cyf, dnx, dny, fmk, mk, umask, vmask, rmask, nrl, rl, shd, rbx, wfu, wfv, rua, rcl)
    if (nrl > 0) then
      grid = dim3((nrl + ncb - 1) / ncb, 1, 1)
      call inner_kernel_src<<<grid, block>>>(nxc, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, &
           west, east, south, north, ntpcol, xc_col, xc_i, xc_j, xc_nb, q, sxs, sys, VX, VY, &
           vol, cxf, cyf, dnx, dny, fmk, mk, umask, vmask, rmask, nrl, rl, shd, rbx, wfu, wfv, rua, rcl)
    endif
#else
    error stop 'bgcw_inner: CUDA build only'
#endif
  end subroutine bgcw_inner

  subroutine bgcw_outer(ntr, k0, k1, isub, mk, q, sxs, sys, cout, VX, VY, HZS, cxf, cyf, fmk, tnu, dodiff, dtw, &
                        dbud, ew, ee, es, en, shd, rbo, nrl, rl, wfu, wfv, rua, rcl, drv)
    ! H3 for ntr tracers: content per area of q into cout (tp layout),
    ! levels k0..k1 of sub-pass isub (levels with isub > mk(k) skipped);
    ! the columns with river faces (rl(1:nrl), bits rbo) by
    ! outer_kernel_src, their river fluxes into drv when dbud = 1
    integer(ip), intent(in) :: ntr, k0, k1, isub, dodiff, dbud, shd, nrl
    integer(ip), WORK_ATTR intent(in) :: rbo(*), rl(*), wfu(*), wfv(*)
    real(rp), WORK_ATTR intent(in) :: rua(*), rcl(*)
    real(rp), WORK_ATTR intent(inout) :: drv(*)
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
         cxf, cyf, fmk, mk, pm, pn, umask, vmask, rmask, pmon_u, pnom_v, diff2_sp, tnu, dodiff, dtw, dbud, ew, ee, es, en, &
         nrl, rl, shd, rbo, wfu, wfv, rua, rcl, drv)
    if (nrl > 0) then
      grid = dim3((nrl + ncb - 1) / ncb, 1, 1)
      call outer_kernel_src<<<grid, block>>>(nwet, k0, k1, isub, ntr, nx, ny, nz, is, ie, js, je, ilo, ihi, jlo, jhi, &
           west, east, south, north, ntpcol, wet_i, wet_j, tpnb, q, sxs, sys, cout, VX, VY, HZS, &
           cxf, cyf, fmk, mk, pm, pn, umask, vmask, rmask, pmon_u, pnom_v, diff2_sp, tnu, dodiff, dtw, dbud, ew, ee, es, en, &
           nrl, rl, shd, rbo, wfu, wfv, rua, rcl, drv)
    endif
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
                           dbef, daft, dsrf, dnmn, dncn, nqd, nrl, rl, hsm, wq0, wq1, ia, id, cpa, c3)
    ! vertical pass for ntr tracers (tp layout): q = window-start state,
    ! cio = content per area in, concentration out; qs = scratch (slot 3);
    ! the columns with point sources (rl(1:nrl), hsm = 1) by
    ! vertical_kernel_src: cpa(ntr, nqd, nz) of the profiles wq0..wq1 of
    ! each column, 3D CDR forcing c3 of tracers ia (ALK) / id (DIC)
    integer(ip), intent(in) :: ntr, shadow, diag, nqd, ia, id, nrl
    integer(ip), WORK_ATTR intent(in) :: wq0(*), wq1(*), rl(*), hsm(*)
    real(rp), WORK_ATTR intent(in) :: cpa(*), c3(*)
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
         dbef, daft, dsrf, dnmn, dncn, decomp%nx, decomp%ny, nqd, nrl, rl, hsm, wq0, wq1, ia, id, cpa, c3)
    if (nrl > 0) then
      grid = dim3((nrl + ncb - 1) / ncb, 1, 1)
      call vertical_kernel_src<<<grid, block>>>(nwet, ntr, nz, is, ie, js, je, ntpcol, wet_i, wet_j, &
           q, cio, qs, WA, HZS, Hz, pm, pn, rmask, wv1, wfc, wpv, wcf, shadow, sfa, rain, diag, &
           dbef, daft, dsrf, dnmn, dncn, decomp%nx, decomp%ny, nqd, nrl, rl, hsm, wq0, wq1, ia, id, cpa, c3)
    endif
#else
    error stop 'bgcw_vertical: CUDA build only'
#endif
  end subroutine bgcw_vertical

  ! ------------------------------------------------------- river faces
  subroutine bgcw_rcol(nw, wi, wj, wt, VX, VY, rcl)
    ! rcolu/rcolv: the window's summed transport through each river face
    ! over the column (the CPU's sum(VX(i,j,:)), in level order)
    integer(ip), intent(in) :: nw
    integer(ip), WORK_ATTR intent(in) :: wi(*), wj(*), wt(*)
    real(rp), WORK_ATTR intent(in) :: VX(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz), VY(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: rcl(*)
    integer(ip) :: w, k, nz, i, j
    real(rp) :: a
    if (nw <= 0) return
    nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*>>>
#endif
    do w = 1, nw
      i = wi(w); j = wj(w)
      a = 0.0_rp
      if (wt(w) == 1) then
        do k = 1, nz
          a = a + VX(i, j, k)
        enddo
      else
        do k = 1, nz
          a = a + VY(i, j, k)
        enddo
      endif
      rcl(w) = a
    enddo
  end subroutine bgcw_rcol

  subroutine bgcw_rbits(wfu, wfv, rbx, rbo)
    ! the river-face bits of every cross-region column (rbx, for inner)
    ! and every interior wet column (rbo, for outer): 1, 2 = u faces i,
    ! i+1 (rows 1..ny for the outer, jlo..jhi with i in 1..nx for the
    ! inner's x faces), 4, 8 = v faces j, j+1 (columns 1..nx; ilo..ihi with
    ! j in 1..ny for the inner's y faces)
    integer(ip), WORK_ATTR intent(in) :: wfu(decomp%is:decomp%ie, decomp%js:decomp%je), wfv(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip), WORK_ATTR intent(inout) :: rbx(*), rbo(*)
    integer(ip) :: n, i, j, b, nx, ny
    nx = decomp%nx; ny = decomp%ny
    if (nxc > 0) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do n = 1, nxc
        i = xc_i(n); j = xc_j(n)
        b = 0
        if (i >= 1 .and. i <= nx) then
          if (wfu(i, j) > 0) b = b + 1
          if (wfu(i + 1, j) > 0) b = b + 2
        endif
        if (j >= 1 .and. j <= ny) then
          if (wfv(i, j) > 0) b = b + 4
          if (wfv(i, j + 1) > 0) b = b + 8
        endif
        rbx(n) = b
      enddo
    endif
    if (nwet > 0) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do n = 1, nwet
        i = wet_i(n); j = wet_j(n)
        b = 0
        if (wfu(i, j) > 0) b = b + 1
        if (wfu(i + 1, j) > 0) b = b + 2
        if (wfv(i, j) > 0) b = b + 4
        if (wfv(i, j + 1) > 0) b = b + 8
        rbo(n) = b
      enddo
    endif
  end subroutine bgcw_rbits

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

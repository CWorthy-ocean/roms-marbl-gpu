#include "cuf_launch.inc"
module obc
  ! Flather M2 (normal) + Chapman zeta; Orlanski M3/T on open edges.
  ! Closed wall when obc_* is false. Periodic edges are not physical.
  ! All edge and corner updates run as device kernels on work_stream: the
  ! host only branches on edge flags, so no managed field is touched from
  ! the host (host loops here used to migrate whole pages every fast step).
  ! The g0/g1 edge-gradient work arrays of the CPU code are recomputed
  ! inline as the same differences, so the arithmetic is unchanged.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use formulas, only: g, apply_mask
  use pflx_mod, only: ub_tune, ub_w, ub_e, ub_s, ub_n
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private

  real(rp), parameter :: cx_lim = 0.292893218813452_rp
  real(rp), parameter :: eps = 1.0e-33_rp
  ! CPU gamma2 (GAMMA2_SETTINGS): +1 free slip, -1 no slip, used by the
  ! closed-wall tangential copies. Only free slip is supported, as in the
  ! pmask of geom (and of the CPU, which ignores gamma2 there).
  real(rp), parameter :: gamma2 = 1.0_rp

  public :: zetabc
  public :: u2dbc
  public :: v2dbc
  public :: uv2dbc
  public :: uv2dbc_corners
  public :: u3dbc
  public :: v3dbc
  public :: t3dbc

contains

  subroutine zetabc()
    ! All four zeta_new edges in one launch (thread m = row or column
    ! index). Corners are set by the copy kernel in step2d (host_zeta_obc).
    integer(ip) :: m, nx, ny, ks, mmax
    real(rp) :: cx, dtf
    logical :: ew, ee, es, en, ow, oe, os, on
    nx = decomp%nx; ny = decomp%ny
    ks = kstp; dtf = dtfast
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. (ew .or. ee .or. es .or. en)) return
    mmax = max(nx, ny) + 1
    CUF_DO1
    do m = 0, mmax
      if (m <= ny + 1) then
        if (ew) then
          if (ow) then
            cx = dtf * pm(1, m) * sqrt(g * max(h(1, m), 1.0_rp))
            zeta_new(0, m) = apply_mask((1.0_rp - cx) * zeta(0, m, ks) + cx * zeta(1, m, ks), rmask(0, m))
          else
            zeta_new(0, m) = apply_mask(zeta_new(1, m), rmask(0, m))
          endif
        endif
        if (ee) then
          if (oe) then
            cx = dtf * pm(nx, m) * sqrt(g * max(h(nx, m), 1.0_rp))
            zeta_new(nx + 1, m) = apply_mask((1.0_rp - cx) * zeta(nx + 1, m, ks) + cx * zeta(nx, m, ks), rmask(nx + 1, m))
          else
            zeta_new(nx + 1, m) = apply_mask(zeta_new(nx, m), rmask(nx + 1, m))
          endif
        endif
      endif
      if (m <= nx + 1) then
        if (es) then
          if (os) then
            cx = dtf * pn(m, 1) * sqrt(g * max(h(m, 1), 1.0_rp))
            zeta_new(m, 0) = apply_mask((1.0_rp - cx) * zeta(m, 0, ks) + cx * zeta(m, 1, ks), rmask(m, 0))
          else
            zeta_new(m, 0) = apply_mask(zeta_new(m, 1), rmask(m, 0))
          endif
        endif
        if (en) then
          if (on) then
            cx = dtf * pn(m, ny) * sqrt(g * max(h(m, ny), 1.0_rp))
            zeta_new(m, ny + 1) = apply_mask((1.0_rp - cx) * zeta(m, ny + 1, ks) + cx * zeta(m, ny, ks), rmask(m, ny + 1))
          else
            zeta_new(m, ny + 1) = apply_mask(zeta_new(m, ny), rmask(m, ny + 1))
          endif
        endif
      endif
    enddo
  end subroutine zetabc

  subroutine u2dbc()
    ! Normal (W/E Flather) and tangential (S/N upstream radiation or
    ! Neumann) ubar edges in one launch; corners in uv2dbc_corners.
    integer(ip) :: m, nx, ny, i0, ks, kn, mmax
    real(rp) :: cff, hx, cx, zx, cy, cext, dtf
    real(rp) :: g0m, g0c, g1m, g1c
    logical :: ew, ee, es, en, ow, oe, os, on
    nx = decomp%nx; ny = decomp%ny
    ks = kstp; kn = knew; dtf = dtfast
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. (ew .or. ee .or. es .or. en)) return
    i0 = 1
    if (ew) i0 = 2
    mmax = max(nx, ny) + 1
    CUF_DO1
    do m = 0, mmax
      if (m >= 1 .and. m <= ny) then
        if (ew) then
          if (ow) then
            cff = 0.5_rp * (h(0, m) + h(1, m))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pm(0, m) + pm(1, m))
            zx = (0.5_rp + cx) * zeta(1, m, ks) + (0.5_rp - cx) * zeta(0, m, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(1, m, kn) + cx * zeta(0, m, ks) - (1.0_rp + cx) * zeta(1, m, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            ubar(1, m, kn) = 0.5_rp * ((1.0_rp - cx) * ubar(1, m, ks) + cx * ubar(2, m, ks) + &
                 ubar_west(m) - hx * (zx - zeta_west(m)))
            ubar(1, m, kn) = apply_mask(ubar(1, m, kn), umask(1, m))
          else
            ubar(1, m, kn) = 0.0_rp
          endif
        endif
        if (ee) then
          if (oe) then
            cff = 0.5_rp * (h(nx, m) + h(nx + 1, m))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pm(nx, m) + pm(nx + 1, m))
            zx = (0.5_rp + cx) * zeta(nx, m, ks) + (0.5_rp - cx) * zeta(nx + 1, m, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(nx, m, kn) + cx * zeta(nx + 1, m, ks) - (1.0_rp + cx) * zeta(nx, m, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            ubar(nx + 1, m, kn) = 0.5_rp * ((1.0_rp - cx) * ubar(nx + 1, m, ks) + cx * ubar(nx, m, ks) + &
                 ubar_east(m) + hx * (zx - zeta_east(m)))
            ubar(nx + 1, m, kn) = apply_mask(ubar(nx + 1, m, kn), umask(nx + 1, m))
          else
            ubar(nx + 1, m, kn) = 0.0_rp
          endif
        endif
      endif
      ! Tangent (CPU u2dbc_tile): upstream radiation, inflow nudged to
      ! ubar_south/ubar_north; closed edges take the wall copy.
      ! g0(i) = ubar(i+1,edge) - ubar(i,edge), g1 likewise one row inside.
      if (es) then
        if (os) then
          if (m >= i0 .and. m <= nx) then
            g0m = ubar(m, 0, ks) - ubar(m - 1, 0, ks)
            g0c = ubar(m + 1, 0, ks) - ubar(m, 0, ks)
            g1m = ubar(m, 1, ks) - ubar(m - 1, 1, ks)
            g1c = ubar(m + 1, 1, ks) - ubar(m, 1, ks)
            cx = -0.125_rp * dtf * (vbar(m, 1, ks) + vbar(m - 1, 1, ks)) * &
                 (pn(m, 0) + pn(m - 1, 0) + pn(m, 1) + pn(m - 1, 1))
            cy = 0.125_rp * dtf * (ubar(m, 0, ks) + ubar(m, 1, ks)) * &
                 (pm(m, 0) + pm(m - 1, 0) + pm(m, 1) + pm(m - 1, 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            ubar(m, 0, kn) = (1.0_rp - cx) * (ubar(m, 0, ks) &
                 - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c) &
                 + cx * (ubar(m, 1, ks) - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c)
            ubar(m, 0, kn) = ((1.0_rp - cext) * ubar(m, 0, kn) + cext * ubar_south(m)) * umask(m, 0)
          endif
        elseif (m >= i0 .and. m <= nx) then
          ! Closed wall (CPU u2dbc, i = istr..iendR). The end faces m = 1
          ! (west edge) and nx+1 (east edge) read normal faces this launch
          ! writes, so uv2dbc_corners sets them.
          ubar(m, 0, kn) = gamma2 * ubar(m, 1, kn) * umask(m, 0)
        endif
      endif
      if (en) then
        if (on) then
          if (m >= i0 .and. m <= nx) then
            g0m = ubar(m, ny, ks) - ubar(m - 1, ny, ks)
            g0c = ubar(m + 1, ny, ks) - ubar(m, ny, ks)
            g1m = ubar(m, ny + 1, ks) - ubar(m - 1, ny + 1, ks)
            g1c = ubar(m + 1, ny + 1, ks) - ubar(m, ny + 1, ks)
            cx = 0.125_rp * dtf * (vbar(m, ny + 1, ks) + vbar(m - 1, ny + 1, ks)) * &
                 (pn(m, ny) + pn(m - 1, ny) + pn(m, ny + 1) + pn(m - 1, ny + 1))
            cy = 0.125_rp * dtf * (ubar(m, ny, ks) + ubar(m, ny + 1, ks)) * &
                 (pm(m, ny) + pm(m - 1, ny) + pm(m, ny + 1) + pm(m - 1, ny + 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            ubar(m, ny + 1, kn) = (1.0_rp - cx) * (ubar(m, ny + 1, ks) &
                 - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c) &
                 + cx * (ubar(m, ny, ks) - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c)
            ubar(m, ny + 1, kn) = ((1.0_rp - cext) * ubar(m, ny + 1, kn) + &
                 cext * ubar_north(m)) * umask(m, ny + 1)
          endif
        elseif (m >= i0 .and. m <= nx) then
          ubar(m, ny + 1, kn) = gamma2 * ubar(m, ny, kn) * umask(m, ny + 1)
        endif
      endif
    enddo
  end subroutine u2dbc

  subroutine v2dbc()
    ! Transpose of u2dbc: normal S/N Flather, tangential W/E, one launch.
    integer(ip) :: m, nx, ny, j0, ks, kn, mmax
    real(rp) :: cff, hx, cx, zx, cy, cext, dtf
    real(rp) :: g0m, g0c, g1m, g1c
    logical :: ew, ee, es, en, ow, oe, os, on
    nx = decomp%nx; ny = decomp%ny
    ks = kstp; kn = knew; dtf = dtfast
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. (ew .or. ee .or. es .or. en)) return
    j0 = 1
    if (es) j0 = 2
    mmax = max(nx, ny) + 1
    CUF_DO1
    do m = 0, mmax
      if (m >= 1 .and. m <= nx) then
        if (es) then
          if (os) then
            cff = 0.5_rp * (h(m, 0) + h(m, 1))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pn(m, 0) + pn(m, 1))
            zx = (0.5_rp + cx) * zeta(m, 1, ks) + (0.5_rp - cx) * zeta(m, 0, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(m, 1, kn) + cx * zeta(m, 0, ks) - (1.0_rp + cx) * zeta(m, 1, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            vbar(m, 1, kn) = 0.5_rp * ((1.0_rp - cx) * vbar(m, 1, ks) + cx * vbar(m, 2, ks) + &
                 vbar_south(m) - hx * (zx - zeta_south(m)))
            vbar(m, 1, kn) = apply_mask(vbar(m, 1, kn), vmask(m, 1))
          else
            vbar(m, 1, kn) = 0.0_rp
          endif
        endif
        if (en) then
          if (on) then
            cff = 0.5_rp * (h(m, ny) + h(m, ny + 1))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pn(m, ny) + pn(m, ny + 1))
            zx = (0.5_rp + cx) * zeta(m, ny, ks) + (0.5_rp - cx) * zeta(m, ny + 1, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(m, ny, kn) + cx * zeta(m, ny + 1, ks) - (1.0_rp + cx) * zeta(m, ny, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            vbar(m, ny + 1, kn) = 0.5_rp * ((1.0_rp - cx) * vbar(m, ny + 1, ks) + cx * vbar(m, ny, ks) + &
                 vbar_north(m) + hx * (zx - zeta_north(m)))
            vbar(m, ny + 1, kn) = apply_mask(vbar(m, ny + 1, kn), vmask(m, ny + 1))
          else
            vbar(m, ny + 1, kn) = 0.0_rp
          endif
        endif
      endif
      ! Tangent (CPU v2dbc_tile), transpose of u2dbc.
      if (ew) then
        if (ow) then
          if (m >= j0 .and. m <= ny) then
            g0m = vbar(0, m, ks) - vbar(0, m - 1, ks)
            g0c = vbar(0, m + 1, ks) - vbar(0, m, ks)
            g1m = vbar(1, m, ks) - vbar(1, m - 1, ks)
            g1c = vbar(1, m + 1, ks) - vbar(1, m, ks)
            cx = -0.125_rp * dtf * (ubar(1, m, ks) + ubar(1, m - 1, ks)) * &
                 (pm(0, m) + pm(0, m - 1) + pm(1, m) + pm(1, m - 1))
            cy = 0.125_rp * dtf * (vbar(0, m, ks) + vbar(1, m, ks)) * &
                 (pn(0, m) + pn(0, m - 1) + pn(1, m) + pn(1, m - 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            vbar(0, m, kn) = (1.0_rp - cx) * (vbar(0, m, ks) &
                 - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c) &
                 + cx * (vbar(1, m, ks) - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c)
            vbar(0, m, kn) = ((1.0_rp - cext) * vbar(0, m, kn) + cext * vbar_west(m)) * vmask(0, m)
          endif
        elseif (m >= j0 .and. m <= ny) then
          ! Closed wall (CPU v2dbc, j = jstr..jendR). The end faces m = 1
          ! (south edge) and ny+1 (north edge) read normal faces this launch
          ! writes, so uv2dbc_corners sets them.
          vbar(0, m, kn) = gamma2 * vbar(1, m, kn) * vmask(0, m)
        endif
      endif
      if (ee) then
        if (oe) then
          if (m >= j0 .and. m <= ny) then
            g0m = vbar(nx, m, ks) - vbar(nx, m - 1, ks)
            g0c = vbar(nx, m + 1, ks) - vbar(nx, m, ks)
            g1m = vbar(nx + 1, m, ks) - vbar(nx + 1, m - 1, ks)
            g1c = vbar(nx + 1, m + 1, ks) - vbar(nx + 1, m, ks)
            cx = 0.125_rp * dtf * (ubar(nx + 1, m, ks) + ubar(nx + 1, m - 1, ks)) * &
                 (pm(nx, m) + pm(nx, m - 1) + pm(nx + 1, m) + pm(nx + 1, m - 1))
            cy = 0.125_rp * dtf * (vbar(nx, m, ks) + vbar(nx + 1, m, ks)) * &
                 (pn(nx, m) + pn(nx, m - 1) + pn(nx + 1, m) + pn(nx + 1, m - 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            vbar(nx + 1, m, kn) = (1.0_rp - cx) * (vbar(nx + 1, m, ks) &
                 - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c) &
                 + cx * (vbar(nx, m, ks) - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c)
            vbar(nx + 1, m, kn) = ((1.0_rp - cext) * vbar(nx + 1, m, kn) + &
                 cext * vbar_east(m)) * vmask(nx + 1, m)
          endif
        elseif (m >= j0 .and. m <= ny) then
          vbar(nx + 1, m, kn) = gamma2 * vbar(nx, m, kn) * vmask(nx + 1, m)
        endif
      endif
    enddo
  end subroutine v2dbc

  subroutine uv2dbc()
    ! u2dbc and v2dbc in one launch: thread m runs the ubar body, then the
    ! vbar body. They write different arrays (ubar / vbar at knew) and each
    ! reads the other only at kstp, so the results are those of the two
    ! separate launches.
    integer(ip) :: m, nx, ny, i0, j0, ks, kn, mmax
    real(rp) :: cff, hx, cx, zx, cy, cext, dtf
    real(rp) :: g0m, g0c, g1m, g1c
    logical :: ew, ee, es, en, ow, oe, os, on
    nx = decomp%nx; ny = decomp%ny
    ks = kstp; kn = knew; dtf = dtfast
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. (ew .or. ee .or. es .or. en)) return
    i0 = 1
    if (ew) i0 = 2
    j0 = 1
    if (es) j0 = 2
    mmax = max(nx, ny) + 1
    CUF_DO1
    do m = 0, mmax
      if (m >= 1 .and. m <= ny) then
        if (ew) then
          if (ow) then
            cff = 0.5_rp * (h(0, m) + h(1, m))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pm(0, m) + pm(1, m))
            zx = (0.5_rp + cx) * zeta(1, m, ks) + (0.5_rp - cx) * zeta(0, m, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(1, m, kn) + cx * zeta(0, m, ks) - (1.0_rp + cx) * zeta(1, m, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            ubar(1, m, kn) = 0.5_rp * ((1.0_rp - cx) * ubar(1, m, ks) + cx * ubar(2, m, ks) + &
                 ubar_west(m) - hx * (zx - zeta_west(m)))
            ubar(1, m, kn) = apply_mask(ubar(1, m, kn), umask(1, m))
          else
            ubar(1, m, kn) = 0.0_rp
          endif
        endif
        if (ee) then
          if (oe) then
            cff = 0.5_rp * (h(nx, m) + h(nx + 1, m))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pm(nx, m) + pm(nx + 1, m))
            zx = (0.5_rp + cx) * zeta(nx, m, ks) + (0.5_rp - cx) * zeta(nx + 1, m, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(nx, m, kn) + cx * zeta(nx + 1, m, ks) - (1.0_rp + cx) * zeta(nx, m, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            ubar(nx + 1, m, kn) = 0.5_rp * ((1.0_rp - cx) * ubar(nx + 1, m, ks) + cx * ubar(nx, m, ks) + &
                 ubar_east(m) + hx * (zx - zeta_east(m)))
            ubar(nx + 1, m, kn) = apply_mask(ubar(nx + 1, m, kn), umask(nx + 1, m))
          else
            ubar(nx + 1, m, kn) = 0.0_rp
          endif
        endif
      endif
      ! Tangent (CPU u2dbc_tile): upstream radiation, inflow nudged to
      ! ubar_south/ubar_north; closed edges take the wall copy.
      ! g0(i) = ubar(i+1,edge) - ubar(i,edge), g1 likewise one row inside.
      if (es) then
        if (os) then
          if (m >= i0 .and. m <= nx) then
            g0m = ubar(m, 0, ks) - ubar(m - 1, 0, ks)
            g0c = ubar(m + 1, 0, ks) - ubar(m, 0, ks)
            g1m = ubar(m, 1, ks) - ubar(m - 1, 1, ks)
            g1c = ubar(m + 1, 1, ks) - ubar(m, 1, ks)
            cx = -0.125_rp * dtf * (vbar(m, 1, ks) + vbar(m - 1, 1, ks)) * &
                 (pn(m, 0) + pn(m - 1, 0) + pn(m, 1) + pn(m - 1, 1))
            cy = 0.125_rp * dtf * (ubar(m, 0, ks) + ubar(m, 1, ks)) * &
                 (pm(m, 0) + pm(m - 1, 0) + pm(m, 1) + pm(m - 1, 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            ubar(m, 0, kn) = (1.0_rp - cx) * (ubar(m, 0, ks) &
                 - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c) &
                 + cx * (ubar(m, 1, ks) - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c)
            ubar(m, 0, kn) = ((1.0_rp - cext) * ubar(m, 0, kn) + cext * ubar_south(m)) * umask(m, 0)
          endif
        elseif (m >= i0 .and. m <= nx) then
          ! Closed wall (CPU u2dbc, i = istr..iendR). The end faces m = 1
          ! (west edge) and nx+1 (east edge) read normal faces this launch
          ! writes, so uv2dbc_corners sets them.
          ubar(m, 0, kn) = gamma2 * ubar(m, 1, kn) * umask(m, 0)
        endif
      endif
      if (en) then
        if (on) then
          if (m >= i0 .and. m <= nx) then
            g0m = ubar(m, ny, ks) - ubar(m - 1, ny, ks)
            g0c = ubar(m + 1, ny, ks) - ubar(m, ny, ks)
            g1m = ubar(m, ny + 1, ks) - ubar(m - 1, ny + 1, ks)
            g1c = ubar(m + 1, ny + 1, ks) - ubar(m, ny + 1, ks)
            cx = 0.125_rp * dtf * (vbar(m, ny + 1, ks) + vbar(m - 1, ny + 1, ks)) * &
                 (pn(m, ny) + pn(m - 1, ny) + pn(m, ny + 1) + pn(m - 1, ny + 1))
            cy = 0.125_rp * dtf * (ubar(m, ny, ks) + ubar(m, ny + 1, ks)) * &
                 (pm(m, ny) + pm(m - 1, ny) + pm(m, ny + 1) + pm(m - 1, ny + 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            ubar(m, ny + 1, kn) = (1.0_rp - cx) * (ubar(m, ny + 1, ks) &
                 - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c) &
                 + cx * (ubar(m, ny, ks) - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c)
            ubar(m, ny + 1, kn) = ((1.0_rp - cext) * ubar(m, ny + 1, kn) + &
                 cext * ubar_north(m)) * umask(m, ny + 1)
          endif
        elseif (m >= i0 .and. m <= nx) then
          ubar(m, ny + 1, kn) = gamma2 * ubar(m, ny, kn) * umask(m, ny + 1)
        endif
      endif
      if (m >= 1 .and. m <= nx) then
        if (es) then
          if (os) then
            cff = 0.5_rp * (h(m, 0) + h(m, 1))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pn(m, 0) + pn(m, 1))
            zx = (0.5_rp + cx) * zeta(m, 1, ks) + (0.5_rp - cx) * zeta(m, 0, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(m, 1, kn) + cx * zeta(m, 0, ks) - (1.0_rp + cx) * zeta(m, 1, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            vbar(m, 1, kn) = 0.5_rp * ((1.0_rp - cx) * vbar(m, 1, ks) + cx * vbar(m, 2, ks) + &
                 vbar_south(m) - hx * (zx - zeta_south(m)))
            vbar(m, 1, kn) = apply_mask(vbar(m, 1, kn), vmask(m, 1))
          else
            vbar(m, 1, kn) = 0.0_rp
          endif
        endif
        if (en) then
          if (on) then
            cff = 0.5_rp * (h(m, ny) + h(m, ny + 1))
            hx = sqrt(g / max(cff, 1.0_rp))
            cx = dtf * cff * hx * 0.5_rp * (pn(m, ny) + pn(m, ny + 1))
            zx = (0.5_rp + cx) * zeta(m, ny, ks) + (0.5_rp - cx) * zeta(m, ny + 1, ks)
            if (cx > cx_lim) then
              zx = zx + (zeta(m, ny, kn) + cx * zeta(m, ny + 1, ks) - (1.0_rp + cx) * zeta(m, ny, ks)) * &
                   (1.0_rp - cx_lim / cx)**2
            endif
            vbar(m, ny + 1, kn) = 0.5_rp * ((1.0_rp - cx) * vbar(m, ny + 1, ks) + cx * vbar(m, ny, ks) + &
                 vbar_north(m) + hx * (zx - zeta_north(m)))
            vbar(m, ny + 1, kn) = apply_mask(vbar(m, ny + 1, kn), vmask(m, ny + 1))
          else
            vbar(m, ny + 1, kn) = 0.0_rp
          endif
        endif
      endif
      ! Tangent (CPU v2dbc_tile), transpose of u2dbc.
      if (ew) then
        if (ow) then
          if (m >= j0 .and. m <= ny) then
            g0m = vbar(0, m, ks) - vbar(0, m - 1, ks)
            g0c = vbar(0, m + 1, ks) - vbar(0, m, ks)
            g1m = vbar(1, m, ks) - vbar(1, m - 1, ks)
            g1c = vbar(1, m + 1, ks) - vbar(1, m, ks)
            cx = -0.125_rp * dtf * (ubar(1, m, ks) + ubar(1, m - 1, ks)) * &
                 (pm(0, m) + pm(0, m - 1) + pm(1, m) + pm(1, m - 1))
            cy = 0.125_rp * dtf * (vbar(0, m, ks) + vbar(1, m, ks)) * &
                 (pn(0, m) + pn(0, m - 1) + pn(1, m) + pn(1, m - 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            vbar(0, m, kn) = (1.0_rp - cx) * (vbar(0, m, ks) &
                 - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c) &
                 + cx * (vbar(1, m, ks) - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c)
            vbar(0, m, kn) = ((1.0_rp - cext) * vbar(0, m, kn) + cext * vbar_west(m)) * vmask(0, m)
          endif
        elseif (m >= j0 .and. m <= ny) then
          ! Closed wall (CPU v2dbc, j = jstr..jendR). The end faces m = 1
          ! (south edge) and ny+1 (north edge) read normal faces this launch
          ! writes, so uv2dbc_corners sets them.
          vbar(0, m, kn) = gamma2 * vbar(1, m, kn) * vmask(0, m)
        endif
      endif
      if (ee) then
        if (oe) then
          if (m >= j0 .and. m <= ny) then
            g0m = vbar(nx, m, ks) - vbar(nx, m - 1, ks)
            g0c = vbar(nx, m + 1, ks) - vbar(nx, m, ks)
            g1m = vbar(nx + 1, m, ks) - vbar(nx + 1, m - 1, ks)
            g1c = vbar(nx + 1, m + 1, ks) - vbar(nx + 1, m, ks)
            cx = 0.125_rp * dtf * (ubar(nx + 1, m, ks) + ubar(nx + 1, m - 1, ks)) * &
                 (pm(nx, m) + pm(nx, m - 1) + pm(nx + 1, m) + pm(nx + 1, m - 1))
            cy = 0.125_rp * dtf * (vbar(nx, m, ks) + vbar(nx + 1, m, ks)) * &
                 (pn(nx, m) + pn(nx, m - 1) + pn(nx + 1, m) + pn(nx + 1, m - 1))
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            vbar(nx + 1, m, kn) = (1.0_rp - cx) * (vbar(nx + 1, m, ks) &
                 - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c) &
                 + cx * (vbar(nx, m, ks) - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c)
            vbar(nx + 1, m, kn) = ((1.0_rp - cext) * vbar(nx + 1, m, kn) + &
                 cext * vbar_east(m)) * vmask(nx + 1, m)
          endif
        elseif (m >= j0 .and. m <= ny) then
          vbar(nx + 1, m, kn) = gamma2 * vbar(nx, m, kn) * vmask(nx + 1, m)
        endif
      endif
    enddo
  end subroutine uv2dbc

  subroutine uv2dbc_corners()
    ! Corner cells of ubar/vbar(knew), one serial launch after uv2dbc.
    ! (1) The end faces of a closed-wall copy (CPU u2dbc i = istr, iendR;
    ! v2dbc j = jstr, jendR) read the normal face uv2dbc wrote at the
    ! other edge; set here, so no thread of uv2dbc reads a cell another
    ! thread writes. (2) Corner means only between two open edges (CPU
    ! "#if defined OBC_SOUTH && defined OBC_WEST" etc.); next to a closed
    ! edge the corner keeps the wall value, or is left as it is (CPU).
    integer(ip) :: n, nx, ny, kn
    logical :: ew, ee, es, en, ow, oe, os, on
    logical :: sw, se, nw, ne
    nx = decomp%nx; ny = decomp%ny
    kn = knew
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. ((es .or. en) .and. (ew .or. ee))) return
    sw = es .and. ew .and. os .and. ow; se = es .and. ee .and. os .and. oe
    nw = en .and. ew .and. on .and. ow; ne = en .and. ee .and. on .and. oe
    CUF_SERIAL
    do n = 1, 1
      if (es .and. .not. os) then
        if (ew) ubar(1, 0, kn) = gamma2 * ubar(1, 1, kn) * umask(1, 0)
        if (ee) ubar(nx + 1, 0, kn) = gamma2 * ubar(nx + 1, 1, kn) * umask(nx + 1, 0)
      endif
      if (en .and. .not. on) then
        if (ew) ubar(1, ny + 1, kn) = gamma2 * ubar(1, ny, kn) * umask(1, ny + 1)
        if (ee) ubar(nx + 1, ny + 1, kn) = gamma2 * ubar(nx + 1, ny, kn) * umask(nx + 1, ny + 1)
      endif
      if (sw) ubar(1, 0, kn) = 0.5_rp * (ubar(2, 0, kn) + ubar(1, 1, kn))
      if (se) ubar(nx + 1, 0, kn) = 0.5_rp * (ubar(nx, 0, kn) + ubar(nx + 1, 1, kn))
      if (nw) ubar(1, ny + 1, kn) = 0.5_rp * (ubar(2, ny + 1, kn) + ubar(1, ny, kn))
      if (ne) ubar(nx + 1, ny + 1, kn) = 0.5_rp * (ubar(nx, ny + 1, kn) + ubar(nx + 1, ny, kn))
      if (ew .and. .not. ow) then
        if (es) vbar(0, 1, kn) = gamma2 * vbar(1, 1, kn) * vmask(0, 1)
        if (en) vbar(0, ny + 1, kn) = gamma2 * vbar(1, ny + 1, kn) * vmask(0, ny + 1)
      endif
      if (ee .and. .not. oe) then
        if (es) vbar(nx + 1, 1, kn) = gamma2 * vbar(nx, 1, kn) * vmask(nx + 1, 1)
        if (en) vbar(nx + 1, ny + 1, kn) = gamma2 * vbar(nx, ny + 1, kn) * vmask(nx + 1, ny + 1)
      endif
      if (sw) vbar(0, 1, kn) = 0.5_rp * (vbar(0, 2, kn) + vbar(1, 1, kn))
      if (se) vbar(nx + 1, 1, kn) = 0.5_rp * (vbar(nx + 1, 2, kn) + vbar(nx, 1, kn))
      if (nw) vbar(0, ny + 1, kn) = 0.5_rp * (vbar(0, ny, kn) + vbar(1, ny + 1, kn))
      if (ne) vbar(nx + 1, ny + 1, kn) = 0.5_rp * (vbar(nx + 1, ny, kn) + vbar(nx, ny + 1, kn))
    enddo
  end subroutine uv2dbc_corners

  real(rp) function dtfwd()
    if (nnew == 3) then
      dtfwd = 0.5_rp * dt
    else
      dtfwd = dt
    endif
  end function dtfwd

  subroutine u3dbc()
    ! CPU u3dbc_tile: normal u by Orlanski radiation with tangential
    ! upstream term (pmask gradients) and inflow nudging; tangential u on
    ! S/N by upstream radiation nudged to u_south/u_north; closed edges as
    ! the CPU (normal 0, wall copy); corner means between two open edges.
    integer(ip) :: i, j, k, nx, ny, nz, i0, ns, nn, nr, iu1
    real(rp) :: dft, dfx, dfy, cff, cx, cy, cext, dtw, ub
    logical :: ubt
    real(rp) :: g0c, g0p, g1c, g1p, g0m, g1m
    logical :: sw, se, nw, ne
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ns = nstp; nn = nnew; nr = nrhs; ub = ubind; ubt = ub_tune
    dtw = dtfwd()
    i0 = 1
    if (decomp%edge_w) i0 = 2
    iu1 = nx                        ! CPU iendR
    if (decomp%edge_e) iu1 = nx + 1

    ! Normal W/E: g0(j) = (u(edge,j) - u(edge,j-1)) * pmask(edge,j), g1
    ! likewise one face inside; the stencil uses g(j) and g(j+1).
    if (decomp%edge_w .and. obc_west) then
      CUF_DO2
      do k = 1, nz
        do j = 1, ny
          g0c = (u(1, j, k, ns) - u(1, j - 1, k, ns)) * pmask(1, j)
          g0p = (u(1, j + 1, k, ns) - u(1, j, k, ns)) * pmask(1, j + 1)
          g1c = (u(2, j, k, ns) - u(2, j - 1, k, ns)) * pmask(2, j)
          g1p = (u(2, j + 1, k, ns) - u(2, j, k, ns)) * pmask(2, j + 1)
          dft = u(2, j, k, ns) - u(2, j, k, nn)
          dfx = u(2, j, k, nn) - u(3, j, k, nn)
          if (dft * (g1c + g1p) > 0.0_rp) then
            dfy = g1c
          else
            dfy = g1p
          endif
          cff = max(dfx * dfx + dfy * dfy, eps)
          cy = min(cff, max(dft * dfy, -cff))
          cx = dft * dfx
          if (cx < 0.0_rp) then
            cx = 0.0_rp; cy = 0.0_rp
            if (u_west(j, k) > 0.0_rp) then
              cext = u_west(j, k)
            else
              cext = ub
            endif
            cext = cext * dtw * 0.5_rp * (pm(0, j) + pm(1, j))
          else
            cext = 0.0_rp
          endif
          u(1, j, k, nn) = (cff * u(1, j, k, ns) + cx * u(2, j, k, nn) &
               - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) / (cff + cx)
          if (ubt) cext = max(cext, min(ub_w(j), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          u(1, j, k, nn) = ((1.0_rp - cext) * u(1, j, k, nn) + cext * u_west(j, k)) * umask(1, j)
        enddo
      enddo
    elseif (decomp%edge_w) then
      CUF_DO2
      do k = 1, nz
        do j = 1, ny
          u(1, j, k, nn) = 0.0_rp
        enddo
      enddo
    endif

    if (decomp%edge_e .and. obc_east) then
      CUF_DO2
      do k = 1, nz
        do j = 1, ny
          g0c = (u(nx, j, k, ns) - u(nx, j - 1, k, ns)) * pmask(nx, j)
          g0p = (u(nx, j + 1, k, ns) - u(nx, j, k, ns)) * pmask(nx, j + 1)
          g1c = (u(nx + 1, j, k, ns) - u(nx + 1, j - 1, k, ns)) * pmask(nx + 1, j)
          g1p = (u(nx + 1, j + 1, k, ns) - u(nx + 1, j, k, ns)) * pmask(nx + 1, j + 1)
          dft = u(nx, j, k, ns) - u(nx, j, k, nn)
          dfx = u(nx, j, k, nn) - u(nx - 1, j, k, nn)
          if (dft * (g0c + g0p) > 0.0_rp) then
            dfy = g0c
          else
            dfy = g0p
          endif
          cff = max(dfx * dfx + dfy * dfy, eps)
          cy = min(cff, max(dft * dfy, -cff))
          cx = dft * dfx
          if (cx < 0.0_rp) then
            cx = 0.0_rp; cy = 0.0_rp
            if (u_east(j, k) < 0.0_rp) then
              cext = -u_east(j, k)
            else
              cext = ub
            endif
            cext = cext * dtw * 0.5_rp * (pm(nx, j) + pm(nx + 1, j))
          else
            cext = 0.0_rp
          endif
          u(nx + 1, j, k, nn) = (cff * u(nx + 1, j, k, ns) + cx * u(nx, j, k, nn) &
               - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) / (cff + cx)
          if (ubt) cext = max(cext, min(ub_e(j), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          u(nx + 1, j, k, nn) = ((1.0_rp - cext) * u(nx + 1, j, k, nn) + &
               cext * u_east(j, k)) * umask(nx + 1, j)
        enddo
      enddo
    elseif (decomp%edge_e) then
      CUF_DO2
      do k = 1, nz
        do j = 1, ny
          u(nx + 1, j, k, nn) = 0.0_rp
        enddo
      enddo
    endif

    ! Tangential S/N: g0(i) = u(i+1,edge) - u(i,edge), g1 one row inside;
    ! the stencil uses g(i-1) and g(i).
    if (decomp%edge_s .and. obc_south) then
      CUF_DO2
      do k = 1, nz
        do i = i0, nx
          g0m = u(i, 0, k, ns) - u(i - 1, 0, k, ns)
          g0c = u(i + 1, 0, k, ns) - u(i, 0, k, ns)
          g1m = u(i, 1, k, ns) - u(i - 1, 1, k, ns)
          g1c = u(i + 1, 1, k, ns) - u(i, 1, k, ns)
          cx = -0.125_rp * dtw * (v(i, 1, k, nr) + v(i - 1, 1, k, nr)) * &
               (pn(i, 0) + pn(i - 1, 0) + pn(i, 1) + pn(i - 1, 1))
          cy = 0.125_rp * dtw * (u(i, 0, k, nr) + u(i, 1, k, nr)) * &
               (pm(i, 0) + pm(i - 1, 0) + pm(i, 1) + pm(i - 1, 1))
          if (cx > 0.0_rp) then
            cext = 0.0_rp
          else
            cext = -cx; cx = 0.0_rp
          endif
          u(i, 0, k, nn) = (1.0_rp - cx) * (u(i, 0, k, ns) &
               - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c) &
               + cx * (u(i, 1, k, ns) - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c)
          if (ubt) cext = max(cext, min(ub_s(i), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          u(i, 0, k, nn) = ((1.0_rp - cext) * u(i, 0, k, nn) + cext * u_south(i, k)) * umask(i, 0)
        enddo
      enddo
    elseif (decomp%edge_s) then
      CUF_DO2
      do k = 1, nz
        do i = 1, iu1
          u(i, 0, k, nn) = gamma2 * u(i, 1, k, nn) * umask(i, 0)
        enddo
      enddo
    endif

    if (decomp%edge_n .and. obc_north) then
      CUF_DO2
      do k = 1, nz
        do i = i0, nx
          g0m = u(i, ny, k, ns) - u(i - 1, ny, k, ns)
          g0c = u(i + 1, ny, k, ns) - u(i, ny, k, ns)
          g1m = u(i, ny + 1, k, ns) - u(i - 1, ny + 1, k, ns)
          g1c = u(i + 1, ny + 1, k, ns) - u(i, ny + 1, k, ns)
          cx = 0.125_rp * dtw * (v(i, ny + 1, k, nr) + v(i - 1, ny + 1, k, nr)) * &
               (pn(i, ny + 1) + pn(i - 1, ny + 1) + pn(i, ny) + pn(i - 1, ny))
          cy = 0.125_rp * dtw * (u(i, ny, k, nr) + u(i, ny + 1, k, nr)) * &
               (pm(i, ny + 1) + pm(i - 1, ny + 1) + pm(i, ny) + pm(i - 1, ny))
          if (cx > 0.0_rp) then
            cext = 0.0_rp
          else
            cext = -cx; cx = 0.0_rp
          endif
          u(i, ny + 1, k, nn) = (1.0_rp - cx) * (u(i, ny + 1, k, ns) &
               - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c) &
               + cx * (u(i, ny, k, ns) - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c)
          if (ubt) cext = max(cext, min(ub_n(i), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          u(i, ny + 1, k, nn) = ((1.0_rp - cext) * u(i, ny + 1, k, nn) + &
               cext * u_north(i, k)) * umask(i, ny + 1)
        enddo
      enddo
    elseif (decomp%edge_n) then
      CUF_DO2
      do k = 1, nz
        do i = 1, iu1
          u(i, ny + 1, k, nn) = gamma2 * u(i, ny, k, nn) * umask(i, ny + 1)
        enddo
      enddo
    endif

    ! Corner means only between two open edges (CPU u3dbc_tile).
    sw = decomp%edge_s .and. decomp%edge_w .and. obc_south .and. obc_west
    se = decomp%edge_s .and. decomp%edge_e .and. obc_south .and. obc_east
    nw = decomp%edge_n .and. decomp%edge_w .and. obc_north .and. obc_west
    ne = decomp%edge_n .and. decomp%edge_e .and. obc_north .and. obc_east
    if (.not. (sw .or. se .or. nw .or. ne)) return
    CUF_DO1
    do k = 1, nz
      if (sw) u(1, 0, k, nn) = 0.5_rp * (u(2, 0, k, nn) + u(1, 1, k, nn))
      if (se) u(nx + 1, 0, k, nn) = 0.5_rp * (u(nx, 0, k, nn) + u(nx + 1, 1, k, nn))
      if (nw) u(1, ny + 1, k, nn) = 0.5_rp * (u(2, ny + 1, k, nn) + u(1, ny, k, nn))
      if (ne) u(nx + 1, ny + 1, k, nn) = 0.5_rp * (u(nx, ny + 1, k, nn) + u(nx + 1, ny, k, nn))
    enddo
  end subroutine u3dbc

  subroutine v3dbc()
    ! CPU v3dbc_tile: transpose of u3dbc (normal v on S/N, tangential v
    ! on W/E nudged to v_west/v_east).
    integer, parameter :: npi = 4, npk = 4
    integer(ip) :: i, j, k, nx, ny, nz, j0, ns, nn, nr, jv1
    integer(ip) :: ip0, kk, ii
    real(rp) :: dft, dfx, dfy, cff, cx, cy, cext, dtw, ub
    logical :: ubt
    real(rp) :: g0c, g0p, g1c, g1p, g0m, g1m
    real(rp) :: pv(16, npi, npk)
    integer(ip) :: kprb(npk)
    logical :: sw, se, nw, ne, probe_n
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ns = nstp; nn = nnew; nr = nrhs; ub = ubind; ubt = ub_tune
    dtw = dtfwd()
    j0 = 1
    if (decomp%edge_s) j0 = 2
    jv1 = ny                        ! CPU jendR
    if (decomp%edge_n) jv1 = ny + 1

    if (decomp%edge_s .and. obc_south) then
      CUF_DO2
      do k = 1, nz
        do i = 1, nx
          g1c = (v(i, 2, k, ns) - v(i - 1, 2, k, ns)) * pmask(i, 2)
          g1p = (v(i + 1, 2, k, ns) - v(i, 2, k, ns)) * pmask(i + 1, 2)
          g0c = (v(i, 1, k, ns) - v(i - 1, 1, k, ns)) * pmask(i, 1)
          g0p = (v(i + 1, 1, k, ns) - v(i, 1, k, ns)) * pmask(i + 1, 1)
          dft = v(i, 2, k, ns) - v(i, 2, k, nn)
          dfx = v(i, 2, k, nn) - v(i, 3, k, nn)
          if (dft * (g1c + g1p) > 0.0_rp) then
            dfy = g1c
          else
            dfy = g1p
          endif
          cff = max(dfx * dfx + dfy * dfy, eps)
          cy = min(cff, max(dft * dfy, -cff))
          cx = dft * dfx
          if (cx < 0.0_rp) then
            cx = 0.0_rp; cy = 0.0_rp
            if (v_south(i, k) > 0.0_rp) then
              cext = v_south(i, k)
            else
              cext = ub
            endif
            cext = cext * dtw * 0.5_rp * (pn(i, 0) + pn(i, 1))
          else
            cext = 0.0_rp
          endif
          v(i, 1, k, nn) = (cff * v(i, 1, k, ns) + cx * v(i, 2, k, nn) &
               - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) / (cff + cx)
          if (ubt) cext = max(cext, min(ub_s(i), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          v(i, 1, k, nn) = ((1.0_rp - cext) * v(i, 1, k, nn) + cext * v_south(i, k)) * vmask(i, 1)
        enddo
      enddo
    elseif (decomp%edge_s) then
      CUF_DO2
      do k = 1, nz
        do i = 1, nx
          v(i, 1, k, nn) = 0.0_rp
        enddo
      enddo
    endif

    if (decomp%edge_n .and. obc_north) then
      ! North Orlanski probe ("diag: v3b"): the intermediates are taken on
      ! the host from the pre-update fields, the result after the kernel.
      probe_n = probe_on
      if (probe_n) then
        call sync_arch()
        kprb = (/ 1, 50, 99, 100 /)
        ip0 = iprb_pt(1) - 1
        pv = 0.0_rp
        do kk = 1, npk
          k = kprb(kk)
          if (k > nz) cycle
          do ii = 1, npi
            i = ip0 + ii - 1
            if (i < 1 .or. i > nx) cycle
            g0c = (v(i, ny, k, ns) - v(i - 1, ny, k, ns)) * pmask(i, ny)
            g0p = (v(i + 1, ny, k, ns) - v(i, ny, k, ns)) * pmask(i + 1, ny)
            g1c = (v(i, ny + 1, k, ns) - v(i - 1, ny + 1, k, ns)) * pmask(i, ny + 1)
            g1p = (v(i + 1, ny + 1, k, ns) - v(i, ny + 1, k, ns)) * pmask(i + 1, ny + 1)
            dft = v(i, ny, k, ns) - v(i, ny, k, nn)
            dfx = v(i, ny, k, nn) - v(i, ny - 1, k, nn)
            if (dft * (g0c + g0p) > 0.0_rp) then
              dfy = g0c
            else
              dfy = g0p
            endif
            cff = max(dfx * dfx + dfy * dfy, eps)
            cy = min(cff, max(dft * dfy, -cff))
            cx = dft * dfx
            if (cx < 0.0_rp) then
              cx = 0.0_rp; cy = 0.0_rp
              if (v_north(i, k) < 0.0_rp) then
                cext = -v_north(i, k)
              else
                cext = ub
              endif
              cext = cext * dtw * 0.5_rp * (pn(i, ny) + pn(i, ny + 1))
            else
              cext = 0.0_rp
            endif
            pv(:, ii, kk) = (/ v(i, ny + 1, k, ns), v(i, ny + 1, k, nn), v(i, ny, k, nn), &
                 v(i, ny - 1, k, nn), g0c, g1c, g0p, g1p, dft, dfx, dfy, cff, cx, cy, &
                 cext, 1.0_rp /)
          enddo
        enddo
      endif
      CUF_DO2
      do k = 1, nz
        do i = 1, nx
          g0c = (v(i, ny, k, ns) - v(i - 1, ny, k, ns)) * pmask(i, ny)
          g0p = (v(i + 1, ny, k, ns) - v(i, ny, k, ns)) * pmask(i + 1, ny)
          g1c = (v(i, ny + 1, k, ns) - v(i - 1, ny + 1, k, ns)) * pmask(i, ny + 1)
          g1p = (v(i + 1, ny + 1, k, ns) - v(i, ny + 1, k, ns)) * pmask(i + 1, ny + 1)
          dft = v(i, ny, k, ns) - v(i, ny, k, nn)
          dfx = v(i, ny, k, nn) - v(i, ny - 1, k, nn)
          if (dft * (g0c + g0p) > 0.0_rp) then
            dfy = g0c
          else
            dfy = g0p
          endif
          cff = max(dfx * dfx + dfy * dfy, eps)
          cy = min(cff, max(dft * dfy, -cff))
          cx = dft * dfx
          if (cx < 0.0_rp) then
            cx = 0.0_rp; cy = 0.0_rp
            if (v_north(i, k) < 0.0_rp) then
              cext = -v_north(i, k)
            else
              cext = ub
            endif
            cext = cext * dtw * 0.5_rp * (pn(i, ny) + pn(i, ny + 1))
          else
            cext = 0.0_rp
          endif
          v(i, ny + 1, k, nn) = (cff * v(i, ny + 1, k, ns) + cx * v(i, ny, k, nn) &
               - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) / (cff + cx)
          if (ubt) cext = max(cext, min(ub_n(i), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          v(i, ny + 1, k, nn) = ((1.0_rp - cext) * v(i, ny + 1, k, nn) + &
               cext * v_north(i, k)) * vmask(i, ny + 1)
        enddo
      enddo
      if (probe_n) then
        call sync_arch()
        do kk = 1, npk
          k = kprb(kk)
          do ii = 1, npi
            i = ip0 + ii - 1
            if (pv(16, ii, kk) /= 1.0_rp) cycle
            write(*,'(a,i6,a,i6,a,i8,a,i4,a,18es26.17)') 'diag: v3b  i=', i, &
                 ' j=', ny + 1, '  north orl  iic=', iic, '  k=', k, &
                 '  vn_stp,vin,vint,vinb,g0,g1,g0p,g1p,dft,dfx,dfy,cff,cx,cy,cext,vbry,vmask,vout=', &
                 pv(1:15, ii, kk), v_north(i, k), vmask(i, ny + 1), v(i, ny + 1, k, nn)
          enddo
        enddo
        flush(6)
      endif
    elseif (decomp%edge_n) then
      CUF_DO2
      do k = 1, nz
        do i = 1, nx
          v(i, ny + 1, k, nn) = 0.0_rp
        enddo
      enddo
    endif

    if (decomp%edge_w .and. obc_west) then
      CUF_DO2
      do k = 1, nz
        do j = j0, ny
          g0m = v(0, j, k, ns) - v(0, j - 1, k, ns)
          g0c = v(0, j + 1, k, ns) - v(0, j, k, ns)
          g1m = v(1, j, k, ns) - v(1, j - 1, k, ns)
          g1c = v(1, j + 1, k, ns) - v(1, j, k, ns)
          cx = -0.125_rp * dtw * (u(1, j, k, nr) + u(1, j - 1, k, nr)) * &
               (pm(0, j) + pm(0, j - 1) + pm(1, j) + pm(1, j - 1))
          cy = 0.125_rp * dtw * (v(0, j, k, nr) + v(1, j, k, nr)) * &
               (pn(0, j) + pn(0, j - 1) + pn(1, j) + pn(1, j - 1))
          if (cx > 0.0_rp) then
            cext = 0.0_rp
          else
            cext = -cx; cx = 0.0_rp
          endif
          v(0, j, k, nn) = (1.0_rp - cx) * (v(0, j, k, ns) &
               - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c) &
               + cx * (v(1, j, k, ns) - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c)
          if (ubt) cext = max(cext, min(ub_w(j), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          v(0, j, k, nn) = ((1.0_rp - cext) * v(0, j, k, nn) + cext * v_west(j, k)) * vmask(0, j)
        enddo
      enddo
    elseif (decomp%edge_w) then
      CUF_DO2
      do k = 1, nz
        do j = 1, jv1
          v(0, j, k, nn) = gamma2 * v(1, j, k, nn) * vmask(0, j)
        enddo
      enddo
    endif

    if (decomp%edge_e .and. obc_east) then
      CUF_DO2
      do k = 1, nz
        do j = j0, ny
          g0m = v(nx, j, k, ns) - v(nx, j - 1, k, ns)
          g0c = v(nx, j + 1, k, ns) - v(nx, j, k, ns)
          g1m = v(nx + 1, j, k, ns) - v(nx + 1, j - 1, k, ns)
          g1c = v(nx + 1, j + 1, k, ns) - v(nx + 1, j, k, ns)
          cx = 0.125_rp * dtw * (u(nx + 1, j, k, nr) + u(nx + 1, j - 1, k, nr)) * &
               (pm(nx + 1, j) + pm(nx + 1, j - 1) + pm(nx, j) + pm(nx, j - 1))
          cy = 0.125_rp * dtw * (v(nx, j, k, nr) + v(nx + 1, j, k, nr)) * &
               (pn(nx + 1, j) + pn(nx + 1, j - 1) + pn(nx, j) + pn(nx, j - 1))
          if (cx > 0.0_rp) then
            cext = 0.0_rp
          else
            cext = -cx; cx = 0.0_rp
          endif
          v(nx + 1, j, k, nn) = (1.0_rp - cx) * (v(nx + 1, j, k, ns) &
               - max(cy, 0.0_rp) * g1m - min(cy, 0.0_rp) * g1c) &
               + cx * (v(nx, j, k, ns) - max(cy, 0.0_rp) * g0m - min(cy, 0.0_rp) * g0c)
          if (ubt) cext = max(cext, min(ub_e(j), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
          v(nx + 1, j, k, nn) = ((1.0_rp - cext) * v(nx + 1, j, k, nn) + &
               cext * v_east(j, k)) * vmask(nx + 1, j)
        enddo
      enddo
    elseif (decomp%edge_e) then
      CUF_DO2
      do k = 1, nz
        do j = 1, jv1
          v(nx + 1, j, k, nn) = gamma2 * v(nx, j, k, nn) * vmask(nx + 1, j)
        enddo
      enddo
    endif

    ! Corner means only between two open edges (CPU v3dbc_tile).
    sw = decomp%edge_s .and. decomp%edge_w .and. obc_south .and. obc_west
    se = decomp%edge_s .and. decomp%edge_e .and. obc_south .and. obc_east
    nw = decomp%edge_n .and. decomp%edge_w .and. obc_north .and. obc_west
    ne = decomp%edge_n .and. decomp%edge_e .and. obc_north .and. obc_east
    if (.not. (sw .or. se .or. nw .or. ne)) return
    CUF_DO1
    do k = 1, nz
      if (sw) v(0, 1, k, nn) = 0.5_rp * (v(0, 2, k, nn) + v(1, 1, k, nn))
      if (se) v(nx + 1, 1, k, nn) = 0.5_rp * (v(nx + 1, 2, k, nn) + v(nx, 1, k, nn))
      if (nw) v(0, ny + 1, k, nn) = 0.5_rp * (v(0, ny, k, nn) + v(1, ny + 1, k, nn))
      if (ne) v(nx + 1, ny + 1, k, nn) = 0.5_rp * (v(nx + 1, ny, k, nn) + v(nx, ny + 1, k, nn))
    enddo
  end subroutine v3dbc

  subroutine t3dbc(itrc)
    ! CPU t3dbc_tile (OBC_TORLANSKI): upstream radiation with tangential
    ! upstream advection, nudged toward t_<edge> on inflow, then corners.
    ! Outflow uses the interior at nstp on W/S and at nnew on E/N (as CPU).
    ! g0(j) = (t(edge,j) - t(edge,j-1)) * mask(edge,j), g1 one row inside;
    ! the stencil uses g(j) and g(j+1).
    integer(ip), intent(in) :: itrc
    integer(ip) :: i, j, k, nx, ny, nz, ns, nn, nr, it
    real(rp) :: cx, cy, cext, dtw, cff
    real(rp) :: g0c, g0p, g1c, g1p
    logical :: sw, se, nw, ne, ubt
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ns = nstp; nn = nnew; nr = nrhs; it = itrc; ubt = ub_tune
    dtw = dtfwd()

    if (decomp%edge_w) then
      if (obc_west) then
        CUF_DO2
        do k = 1, nz
          do j = 1, ny
            g0c = (t(0, j, k, ns, it) - t(0, j - 1, k, ns, it)) * vmask(0, j)
            g0p = (t(0, j + 1, k, ns, it) - t(0, j, k, ns, it)) * vmask(0, j + 1)
            g1c = (t(1, j, k, ns, it) - t(1, j - 1, k, ns, it)) * vmask(1, j)
            g1p = (t(1, j + 1, k, ns, it) - t(1, j, k, ns, it)) * vmask(1, j + 1)
            cx = -dtw * u(1, j, k, nr) * pm(0, j)
            cy = 0.5_rp * dtw * (v(0, j, k, nr) + v(0, j + 1, k, nr)) * pn(0, j)
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            t(0, j, k, nn, it) = (1.0_rp - cx) * (t(0, j, k, ns, it) &
                 - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                 + cx * (t(1, j, k, ns, it) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
            if (ubt) cext = max(cext, min(ub_w(j), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
            t(0, j, k, nn, it) = ((1.0_rp - cext) * t(0, j, k, nn, it) + &
                 cext * t_west(j, k, it)) * rmask(0, j)
          enddo
        enddo
      else
        ! Closed (CPU t3dbc_tile, j = jstr..jend): gradient, masked.
        CUF_DO2
        do k = 1, nz
          do j = 1, ny
            t(0, j, k, nn, it) = t(1, j, k, nn, it) * rmask(0, j)
          enddo
        enddo
      endif
    endif

    if (decomp%edge_e) then
      if (obc_east) then
        CUF_DO2
        do k = 1, nz
          do j = 1, ny
            g0c = (t(nx, j, k, ns, it) - t(nx, j - 1, k, ns, it)) * vmask(nx, j)
            g0p = (t(nx, j + 1, k, ns, it) - t(nx, j, k, ns, it)) * vmask(nx, j + 1)
            g1c = (t(nx + 1, j, k, ns, it) - t(nx + 1, j - 1, k, ns, it)) * vmask(nx + 1, j)
            g1p = (t(nx + 1, j + 1, k, ns, it) - t(nx + 1, j, k, ns, it)) * vmask(nx + 1, j + 1)
            cx = dtw * u(nx + 1, j, k, nr) * pm(nx + 1, j)
            cy = 0.5_rp * dtw * (v(nx + 1, j, k, nr) + v(nx + 1, j + 1, k, nr)) * pn(nx + 1, j)
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            t(nx + 1, j, k, nn, it) = (1.0_rp - cx) * (t(nx + 1, j, k, ns, it) &
                 - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                 + cx * (t(nx, j, k, ns, it) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
            if (ubt) cext = max(cext, min(ub_e(j), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
            t(nx + 1, j, k, nn, it) = ((1.0_rp - cext) * t(nx + 1, j, k, nn, it) + &
                 cext * t_east(j, k, it)) * rmask(nx + 1, j)
          enddo
        enddo
      else
        CUF_DO2
        do k = 1, nz
          do j = 1, ny
            t(nx + 1, j, k, nn, it) = t(nx, j, k, nn, it) * rmask(nx + 1, j)
          enddo
        enddo
      endif
    endif

    if (decomp%edge_s) then
      if (obc_south) then
        CUF_DO2
        do k = 1, nz
          do i = 1, nx
            g0c = (t(i, 0, k, ns, it) - t(i - 1, 0, k, ns, it)) * umask(i, 0)
            g0p = (t(i + 1, 0, k, ns, it) - t(i, 0, k, ns, it)) * umask(i + 1, 0)
            g1c = (t(i, 1, k, ns, it) - t(i - 1, 1, k, ns, it)) * umask(i, 1)
            g1p = (t(i + 1, 1, k, ns, it) - t(i, 1, k, ns, it)) * umask(i + 1, 1)
            cx = -dtw * v(i, 1, k, nr) * pn(i, 0)
            cy = 0.5_rp * dtw * (u(i, 0, k, nr) + u(i + 1, 0, k, nr)) * pm(i, 0)
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            t(i, 0, k, nn, it) = (1.0_rp - cx) * (t(i, 0, k, ns, it) &
                 - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                 + cx * (t(i, 1, k, ns, it) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
            if (ubt) cext = max(cext, min(ub_s(i), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
            t(i, 0, k, nn, it) = ((1.0_rp - cext) * t(i, 0, k, nn, it) + &
                 cext * t_south(i, k, it)) * rmask(i, 0)
          enddo
        enddo
      else
        ! Closed (CPU, i = istr..iend); the corners are set below.
        CUF_DO2
        do k = 1, nz
          do i = 1, nx
            t(i, 0, k, nn, it) = t(i, 1, k, nn, it) * rmask(i, 0)
          enddo
        enddo
      endif
    endif

    if (decomp%edge_n) then
      if (obc_north) then
        CUF_DO2
        do k = 1, nz
          do i = 1, nx
            g0c = (t(i, ny, k, ns, it) - t(i - 1, ny, k, ns, it)) * umask(i, ny)
            g0p = (t(i + 1, ny, k, ns, it) - t(i, ny, k, ns, it)) * umask(i + 1, ny)
            g1c = (t(i, ny + 1, k, ns, it) - t(i - 1, ny + 1, k, ns, it)) * umask(i, ny + 1)
            g1p = (t(i + 1, ny + 1, k, ns, it) - t(i, ny + 1, k, ns, it)) * umask(i + 1, ny + 1)
            cx = dtw * v(i, ny + 1, k, nr) * pn(i, ny + 1)
            cy = 0.5_rp * dtw * (u(i, ny + 1, k, nr) + u(i + 1, ny + 1, k, nr)) * pm(i, ny + 1)
            if (cx > 0.0_rp) then
              cext = 0.0_rp
            else
              cext = -cx; cx = 0.0_rp
            endif
            t(i, ny + 1, k, nn, it) = (1.0_rp - cx) * (t(i, ny + 1, k, ns, it) &
                 - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                 + cx * (t(i, ny, k, ns, it) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
            if (ubt) cext = max(cext, min(ub_n(i), 1.0_rp))   ! SPONGE_TUNE (CPU ub_tune)
            t(i, ny + 1, k, nn, it) = ((1.0_rp - cext) * t(i, ny + 1, k, nn, it) + &
                 cext * t_north(i, k, it)) * rmask(i, ny + 1)
          enddo
        enddo
      else
        CUF_DO2
        do k = 1, nz
          do i = 1, nx
            t(i, ny + 1, k, nn, it) = t(i, ny, k, nn, it) * rmask(i, ny + 1)
          enddo
        enddo
      endif
    endif

    ! Corners: mask-weighted mean of the two edge neighbours (CPU corner rule,
    ! applied at every physical corner, open or closed).
    sw = decomp%edge_s .and. decomp%edge_w; se = decomp%edge_s .and. decomp%edge_e
    nw = decomp%edge_n .and. decomp%edge_w; ne = decomp%edge_n .and. decomp%edge_e
    if (.not. (sw .or. se .or. nw .or. ne)) return
    CUF_DO1
    do k = 1, nz
      if (sw) then
        cff = rmask(1, 0) + rmask(0, 1)
        if (cff > 0.0_rp) then
          cff = 1.0_rp / cff
          t(0, 0, k, nn, it) = cff * (rmask(1, 0) * t(1, 0, k, nn, it) + &
               rmask(0, 1) * t(0, 1, k, nn, it))
        else
          t(0, 0, k, nn, it) = 0.0_rp
        endif
      endif
      if (se) then
        cff = rmask(nx, 0) + rmask(nx + 1, 1)
        if (cff > 0.0_rp) then
          cff = 1.0_rp / cff
          t(nx + 1, 0, k, nn, it) = cff * (rmask(nx, 0) * t(nx, 0, k, nn, it) + &
               rmask(nx + 1, 1) * t(nx + 1, 1, k, nn, it))
        else
          t(nx + 1, 0, k, nn, it) = 0.0_rp
        endif
      endif
      if (nw) then
        cff = rmask(1, ny + 1) + rmask(0, ny)
        if (cff > 0.0_rp) then
          cff = 1.0_rp / cff
          t(0, ny + 1, k, nn, it) = cff * (rmask(1, ny + 1) * t(1, ny + 1, k, nn, it) + &
               rmask(0, ny) * t(0, ny, k, nn, it))
        else
          t(0, ny + 1, k, nn, it) = 0.0_rp
        endif
      endif
      if (ne) then
        cff = rmask(nx, ny + 1) + rmask(nx + 1, ny)
        if (cff > 0.0_rp) then
          cff = 1.0_rp / cff
          t(nx + 1, ny + 1, k, nn, it) = cff * (rmask(nx, ny + 1) * t(nx, ny + 1, k, nn, it) + &
               rmask(nx + 1, ny) * t(nx + 1, ny, k, nn, it))
        else
          t(nx + 1, ny + 1, k, nn, it) = 0.0_rp
        endif
      endif
    enddo
  end subroutine t3dbc

end module obc

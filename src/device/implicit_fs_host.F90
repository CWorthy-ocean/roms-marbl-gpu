module fs_host
  ! Implicit free surface (implicit_fs.F90), the host part of the multigrid
  ! preconditioner: the global coarse levels, held
  ! redundantly on every compute rank as on the CPU (implicit_fs_mod.F90 of
  ! ucla-roms, mg_* routines): the allgather of the
  ! coarsest distributed level into one global grid, its further 2x2
  ! aggregation while nx ny > 600 and min(nx, ny) >= 4, the smoother on the
  ! global levels, and the banded Cholesky solve of the coarsest. The
  ! arithmetic (loop order, expressions) is the CPU's.
  !
  ! Plain host code, compiled on its own at -O1 (Makefile): the device
  ! module passes this rank's block of the coarsest distributed level
  ! (packed i fastest) and gets its block of the result back.
  !
  ! Local level numbering: 1 = the gathered global level (the CPU's
  ! mg_ld+1), ng = the coarsest (the CPU's mg_lc).
  use comm_mod, only: comm_allgather_i4, comm_allgatherv_r8
  use fs_chol, only: chol_factor, chol_solve
  implicit none
  private

  public :: fsh_layout, fsh_setup, fsh_vcycle, fsh_print, fsh_get_layout, fsh_factor_from, fsh_direct_vec
  public :: fsh_levels, fsh_set_faces
  public :: fsh_colord

  ! fs_gpu_opt 16 (set by fs_mg): the factor and solve by fs_chol (implicit_fs_chol.F90: column
  ! order, compiled at -O2 -Mnovect; the same operations in the same order, bit-identical)
  logical :: fsh_colord = .false.
  real(8), allocatable :: sv(:)

  integer, parameter :: maxl = 20
  integer :: ng = 0, np = 1, rank = 0, nu = 2, gtot = 0, nc = 0, bw = 0
  logical :: jacobi = .false., per_x = .false., per_y = .false.
  integer :: lnx(maxl) = 0, lny(maxl) = 0, loff(maxl) = 0
  integer, allocatable :: gcnt(:), gdsp(:), gi0(:), gj0(:), gnx(:), gny(:), ccnt(:), cdsp(:)
  real(8), allocatable :: px(:), pb(:), pr(:), pe(:), pcw(:), pcs(:), pdi(:)
  real(8), allocatable :: gsend(:), grecv(:), ab(:,:), sol(:), tmp(:,:)
  real(8), parameter :: om = 0.8_8
  ! coarse face factors of the global levels (fs_mg_faces; gfx(ic, l), l = 2..ng; 0.5 = 'half')
  real(8), allocatable :: gfx(:,:), gfy(:,:)

contains

  subroutine fsh_layout(np_in, rank_in, inode, jnode, nxd, nyd, nu_in, jac, perx, pery, ng_out)
    ! The gather layout of the coarsest distributed level (each rank's
    ! (inode, jnode, nx, ny)), the global levels' sizes and pools
    ! (mg_build_layout from 'global layout' on).
    integer, intent(in) :: np_in, rank_in, inode, jnode, nxd, nyd, nu_in
    logical, intent(in) :: jac, perx, pery
    integer, intent(out) :: ng_out
    integer :: info(4, np_in), k, i, l, tot, nxg, nyg
    np = np_in; rank = rank_in; nu = nu_in; jacobi = jac; per_x = perx; per_y = pery
    call comm_allgather_i4(4, [inode, jnode, nxd, nyd], info)
    allocate(gcnt(np), gdsp(np), gi0(np), gj0(np), gnx(np), gny(np), ccnt(np), cdsp(np))
    nxg = 0; nyg = 0
    do k = 1, np
      if (info(2, k) == 0) nxg = nxg + info(3, k)
      if (info(1, k) == 0) nyg = nyg + info(4, k)
    enddo
    tot = 0
    do k = 1, np
      gnx(k) = info(3, k); gny(k) = info(4, k)
      gi0(k) = 0; gj0(k) = 0
      do i = 1, np
        if (info(2, i) == info(2, k) .and. info(1, i) < info(1, k)) gi0(k) = gi0(k) + info(3, i)
        if (info(1, i) == info(1, k) .and. info(2, i) < info(2, k)) gj0(k) = gj0(k) + info(4, i)
      enddo
      gcnt(k) = info(3, k) * info(4, k)
      gdsp(k) = tot; tot = tot + gcnt(k)
    enddo
    gtot = tot
    tot = 0
    do k = 1, np
      ccnt(k) = 3 * gnx(k) * gny(k) + gnx(k) + gny(k)
      cdsp(k) = tot; tot = tot + ccnt(k)
    enddo
    l = 1
    lnx(l) = nxg; lny(l) = nyg
    do
      if (lnx(l) * lny(l) <= 600 .or. min(lnx(l), lny(l)) < 4 .or. l + 1 >= maxl) exit
      l = l + 1
      lnx(l) = (lnx(l - 1) + 1) / 2; lny(l) = (lny(l - 1) + 1) / 2
    enddo
    ng = l
    tot = 0
    do l = 1, ng
      loff(l) = tot + 1
      tot = tot + (lnx(l) + 2) * (lny(l) + 2)
    enddo
    allocate(px(tot), pb(tot), pr(tot), pe(tot), pcw(tot), pcs(tot), pdi(tot))
    px = 0; pb = 0; pr = 0; pe = 0; pcw = 0; pcs = 0; pdi = 0
    allocate(gsend(3 * nxd * nyd + nxd + nyd + 2), grecv(3 * gtot + 2 * np * (1 + maxval(gnx) + maxval(gny))))
    nc = lnx(ng) * lny(ng)
    if (per_y) then
      bw = nc - 1
    else
      bw = lnx(ng)
    endif
    bw = min(bw, nc - 1)
    allocate(ab(0:bw, nc), sol(nc), sv(0:bw + 1))
    allocate(tmp(maxval(lnx(1:ng)), maxval(lny(1:ng))))
    ng_out = ng
  end subroutine fsh_layout

  subroutine fsh_print(l0)
    ! the global levels' sizes (rank 0 prints)
    integer, intent(in) :: l0
    integer :: l
    do l = 1, ng
      write(*,'(1x,A,I2,A,I5,A,I5,A,L1)') 'IMPLICIT_FS mg level', l0 + l - 1, ' nx', lnx(l), ' ny', lny(l), &
           ' global ', .true.
    enddo
    write(*,'(1x,A,I6,A,I5)') 'IMPLICIT_FS mg coarsest unknowns', nc, ' bandwidth', bw
  end subroutine fsh_print

  subroutine fsh_setup(nxd, nyd, send)
    ! send: this rank's packet of the coarsest distributed level,
    ! e(1:nx,1:ny), cw(1:nx+1,1:ny), cs(1:nx,1:ny+1) (mg_gather_coef), then
    ! the global coarsening and the factor (mg_setup from the gather on)
    integer, intent(in) :: nxd, nyd
    real(8), intent(in) :: send(*)
    integer :: n, l
    n = 3 * nxd * nyd + nxd + nyd
    gsend(1:n) = send(1:n)
    call comm_allgatherv_r8(n, gsend, grecv, ccnt, cdsp)
    call gather_coef(lnx(1), lny(1), pe(loff(1)), pcw(loff(1)), pcs(loff(1)), pdi(loff(1)))
    do l = 2, ng
      call coarsen(lnx(l - 1), lny(l - 1), pe(loff(l - 1)), pcw(loff(l - 1)), pcs(loff(l - 1)), &
                   lnx(l), lny(l), gfx(1:lnx(l) + 1, l), gfy(1:lny(l) + 1, l), &
                   pe(loff(l)), pcw(loff(l)), pcs(loff(l)), pdi(loff(l)))
    enddo
    call factor(lnx(ng), lny(ng), pe(loff(ng)), pcw(loff(ng)), pcs(loff(ng)))
  end subroutine fsh_setup

  subroutine gather_coef(gx, gy, ge, gcw, gcs, gdi)
    integer, intent(in) :: gx, gy
    real(8), dimension(0:gx + 1, 0:gy + 1), intent(inout) :: ge, gcw, gcs, gdi
    integer :: i, j, k, n, i0, j0, mx, my
    ge = 0; gcw = 0; gcs = 0
    do k = 1, np
      i0 = gi0(k); j0 = gj0(k); mx = gnx(k); my = gny(k); n = cdsp(k)
      do j = 1, my
        do i = 1, mx
          n = n + 1; ge(i0 + i, j0 + j) = grecv(n)
        enddo
      enddo
      do j = 1, my
        do i = 1, mx + 1
          n = n + 1; gcw(i0 + i, j0 + j) = grecv(n)
        enddo
      enddo
      do j = 1, my + 1
        do i = 1, mx
          n = n + 1; gcs(i0 + i, j0 + j) = grecv(n)
        enddo
      enddo
    enddo
    call diag(gx, gy, ge, gcw, gcs, gdi)
  end subroutine gather_coef

  subroutine coarsen(nxf, nyf, ef, cwf, csf, nxc, nyc, fx, fy, e, cw, cs, di)
    ! fx, fy: factors of the coarse faces (fs_mg_faces; CPU 36a863a8 mg_coarsen)
    integer, intent(in) :: nxf, nyf, nxc, nyc
    real(8), intent(in) :: fx(nxc + 1), fy(nyc + 1)
    real(8), dimension(0:nxf + 1, 0:nyf + 1), intent(in) :: ef, cwf, csf
    real(8), dimension(0:nxc + 1, 0:nyc + 1), intent(inout) :: e, cw, cs, di
    integer :: i, j, ic, jc, fi, fj
    e = 0; cw = 0; cs = 0; di = 0
    do j = 1, nyf
      jc = (j + 1) / 2
      do i = 1, nxf
        ic = (i + 1) / 2
        e(ic, jc) = e(ic, jc) + ef(i, j)
      enddo
    enddo
    do j = 1, nyf
      jc = (j + 1) / 2
      do ic = 1, nxc + 1
        fi = min(2 * ic - 1, nxf + 1)
        cw(ic, jc) = cw(ic, jc) + fx(ic) * cwf(fi, j)
      enddo
    enddo
    do jc = 1, nyc + 1
      fj = min(2 * jc - 1, nyf + 1)
      do i = 1, nxf
        ic = (i + 1) / 2
        cs(ic, jc) = cs(ic, jc) + fy(jc) * csf(i, fj)
      enddo
    enddo
    call diag(nxc, nyc, e, cw, cs, di)
  end subroutine coarsen

  subroutine diag(nx, ny, e, cw, cs, di)
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: e, cw, cs
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(inout) :: di
    integer :: i, j
    di = 0
    do j = 1, ny
      do i = 1, nx
        if (e(i, j) > 0._8) di(i, j) = 1._8 / (e(i, j) + cw(i, j) + cw(i + 1, j) + cs(i, j) + cs(i, j + 1))
      enddo
    enddo
  end subroutine diag

  subroutine factor(nx, ny, e, cw, cs)
    ! banded Cholesky (mg_factor)
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: e, cw, cs
    integer :: i, j, k, kk, n, b, c, r, q, nr
    real(8) :: s, t
    n = nc; b = bw
    if (fsh_colord) then
      call chol_factor(nx, ny, e, cw, cs, per_x, per_y, n, b, ab, sv)
      return
    endif
    ab = 0
    do j = 1, ny
      do i = 1, nx
        k = i + nx * (j - 1)
        if (e(i, j) > 0._8) then
          ab(0, k) = e(i, j) + cw(i, j) + cw(i + 1, j) + cs(i, j) + cs(i, j + 1)
        else
          ab(0, k) = 1._8
          cycle
        endif
        if (i > 1) call addl(k, k - 1, -cw(i, j))
        if (j > 1) call addl(k, k - nx, -cs(i, j))
        if (per_x) then
          if (i == nx .and. nx > 2) call addl(k, k - nx + 1, -cw(1, j))
        endif
        if (per_y) then
          if (j == ny .and. ny > 2) call addl(k, i, -cs(i, 1))
        endif
      enddo
    enddo
    do k = 1, n
      do kk = 1, min(b, n - k)
        i = mod(k + kk - 1, nx) + 1; j = (k + kk - 1) / nx + 1
        if (e(i, j) <= 0._8) ab(kk, k) = 0._8
      enddo
      i = mod(k - 1, nx) + 1; j = (k - 1) / nx + 1
      if (e(i, j) <= 0._8) ab(1:min(b, n - k), k) = 0._8
    enddo
    do c = 1, n
      s = ab(0, c)
      do kk = max(1, c - b), c - 1
        s = s - ab(c - kk, kk)**2
      enddo
      ab(0, c) = sqrt(s)
      do r = c + 1, min(n, c + b)
        s = ab(r - c, c)
        do kk = max(1, r - b), c - 1
          s = s - ab(r - kk, kk) * ab(c - kk, kk)
        enddo
        ab(r - c, c) = s / ab(0, c)
      enddo
    enddo
  contains
    subroutine addl(row, col, v)
      integer, intent(in) :: row, col
      real(8), intent(in) :: v
      integer :: rr, cc
      rr = max(row, col); cc = min(row, col)
      if (rr - cc <= b) ab(rr - cc, cc) = ab(rr - cc, cc) + v
    end subroutine addl
  end subroutine factor

  subroutine direct(nx, ny, b, x)
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: b
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(inout) :: x
    integer :: i, j, k, kk, n
    real(8) :: s
    n = nc
    do j = 1, ny
      do i = 1, nx
        sol(i + nx * (j - 1)) = b(i, j)
      enddo
    enddo
    if (fsh_colord) then
      call chol_solve(n, bw, ab, sol)
    else
      do k = 1, n
        s = sol(k)
        do kk = max(1, k - bw), k - 1
          s = s - ab(k - kk, kk) * sol(kk)
        enddo
        sol(k) = s / ab(0, k)
      enddo
      do k = n, 1, -1
        s = sol(k)
        do kk = k + 1, min(n, k + bw)
          s = s - ab(kk - k, k) * sol(kk)
        enddo
        sol(k) = s / ab(0, k)
      enddo
    endif
    x = 0
    do j = 1, ny
      do i = 1, nx
        x(i, j) = sol(i + nx * (j - 1))
      enddo
    enddo
  end subroutine direct

  subroutine fsh_vcycle(nxd, nyd, bblk, xblk)
    ! bblk: this rank's rhs block of the coarsest distributed level (i
    ! fastest); xblk: its block of the global result (mg_gather_vec, the
    ! global levels of mg_vcycle, mg_scatter_vec)
    integer, intent(in) :: nxd, nyd
    real(8), intent(in) :: bblk(*)
    real(8), intent(out) :: xblk(*)
    integer :: l
    call gather_vec(nxd * nyd, bblk, lnx(1), lny(1), pb(loff(1)))
    do l = 1, ng - 1
      call level_down(l)
      call restrict(lnx(l), lny(l), pr(loff(l)), lnx(l + 1), lny(l + 1), pb(loff(l + 1)))
    enddo
    call direct(lnx(ng), lny(ng), pb(loff(ng)), px(loff(ng)))
    do l = ng - 1, 1, -1
      call prolong(lnx(l + 1), lny(l + 1), px(loff(l + 1)), lnx(l), lny(l), px(loff(l)), pe(loff(l)))
      call level_up(l)
    enddo
    call scatter_vec(lnx(1), lny(1), px(loff(1)), nxd, nyd, xblk)
  end subroutine fsh_vcycle

  subroutine gather_vec(n, b, gx, gy, gb)
    integer, intent(in) :: n, gx, gy
    real(8), intent(in) :: b(*)
    real(8), dimension(0:gx + 1, 0:gy + 1), intent(inout) :: gb
    integer :: i, j, k, m
    gsend(1:n) = b(1:n)
    call comm_allgatherv_r8(n, gsend, grecv, gcnt, gdsp)
    gb = 0
    do k = 1, np
      m = gdsp(k)
      do j = 1, gny(k)
        do i = 1, gnx(k)
          m = m + 1; gb(gi0(k) + i, gj0(k) + j) = grecv(m)
        enddo
      enddo
    enddo
  end subroutine gather_vec

  subroutine scatter_vec(gx, gy, gxv, nx, ny, x)
    integer, intent(in) :: gx, gy, nx, ny
    real(8), dimension(0:gx + 1, 0:gy + 1), intent(in) :: gxv
    real(8), intent(out) :: x(nx, ny)
    integer :: i, j, k
    k = rank + 1
    do j = 1, ny
      do i = 1, nx
        x(i, j) = gxv(gi0(k) + i, gj0(k) + j)
      enddo
    enddo
  end subroutine scatter_vec

  subroutine level_down(l)
    integer, intent(in) :: l
    integer :: k, o
    o = loff(l)
    call zero(lnx(l), lny(l), px(o))
    do k = 1, nu
      call halo_glob(lnx(l), lny(l), px(o))
      call smooth(lnx(l), lny(l), px(o), pb(o), pcw(o), pcs(o), pdi(o), 1)
    enddo
    call halo_glob(lnx(l), lny(l), px(o))
    call resid(lnx(l), lny(l), px(o), pb(o), pe(o), pcw(o), pcs(o), pr(o))
  end subroutine level_down

  subroutine level_up(l)
    integer, intent(in) :: l
    integer :: k, o
    o = loff(l)
    do k = 1, nu
      call halo_glob(lnx(l), lny(l), px(o))
      call smooth(lnx(l), lny(l), px(o), pb(o), pcw(o), pcs(o), pdi(o), 2)
    enddo
  end subroutine level_up

  subroutine zero(nx, ny, x)
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(inout) :: x
    x = 0
  end subroutine zero

  subroutine halo_glob(nx, ny, x)
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(inout) :: x
    integer :: i, j
    if (per_x) then
      do j = 1, ny
        x(0, j) = x(nx, j); x(nx + 1, j) = x(1, j)
      enddo
    else
      x(0, :) = 0; x(nx + 1, :) = 0
    endif
    if (per_y) then
      do i = 1, nx
        x(i, 0) = x(i, ny); x(i, ny + 1) = x(i, 1)
      enddo
    else
      x(:, 0) = 0; x(:, ny + 1) = 0
    endif
  end subroutine halo_glob

  subroutine smooth(nx, ny, x, b, cw, cs, di, mode)
    integer, intent(in) :: nx, ny, mode
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(inout) :: x
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: b, cw, cs, di
    integer :: i, j, c, col, i0
    if (jacobi) then
      do j = 1, ny
        do i = 1, nx
          tmp(i, j) = di(i, j) * (b(i, j) + cw(i, j) * x(i - 1, j) + cw(i + 1, j) * x(i + 1, j) &
               + cs(i, j) * x(i, j - 1) + cs(i, j + 1) * x(i, j + 1))
        enddo
      enddo
      do j = 1, ny
        do i = 1, nx
          x(i, j) = (1._8 - om) * x(i, j) + om * tmp(i, j)
        enddo
      enddo
      return
    endif
    do c = 1, 2
      col = mod(c + mode, 2)
      do j = 1, ny
        i0 = 1 + mod(j + col, 2)
        do i = i0, nx, 2
          x(i, j) = di(i, j) * (b(i, j) + cw(i, j) * x(i - 1, j) + cw(i + 1, j) * x(i + 1, j) &
               + cs(i, j) * x(i, j - 1) + cs(i, j + 1) * x(i, j + 1))
        enddo
      enddo
    enddo
  end subroutine smooth

  subroutine resid(nx, ny, x, b, e, cw, cs, r)
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: x, b, e, cw, cs
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(inout) :: r
    integer :: i, j
    do j = 1, ny
      do i = 1, nx
        r(i, j) = 0._8
        if (e(i, j) > 0._8) r(i, j) = b(i, j) - (e(i, j) * x(i, j) + cw(i, j) * (x(i, j) - x(i - 1, j)) &
             + cw(i + 1, j) * (x(i, j) - x(i + 1, j)) + cs(i, j) * (x(i, j) - x(i, j - 1)) &
             + cs(i, j + 1) * (x(i, j) - x(i, j + 1)))
      enddo
    enddo
  end subroutine resid

  subroutine restrict(nxf, nyf, rf, nxc, nyc, bc)
    integer, intent(in) :: nxf, nyf, nxc, nyc
    real(8), dimension(0:nxf + 1, 0:nyf + 1), intent(in) :: rf
    real(8), dimension(0:nxc + 1, 0:nyc + 1), intent(inout) :: bc
    integer :: i, j
    bc = 0
    do j = 1, nyf
      do i = 1, nxf
        bc((i + 1) / 2, (j + 1) / 2) = bc((i + 1) / 2, (j + 1) / 2) + rf(i, j)
      enddo
    enddo
  end subroutine restrict

  subroutine prolong(nxc, nyc, xc, nxf, nyf, xf, ef)
    integer, intent(in) :: nxc, nyc, nxf, nyf
    real(8), dimension(0:nxc + 1, 0:nyc + 1), intent(in) :: xc
    real(8), dimension(0:nxf + 1, 0:nyf + 1), intent(inout) :: xf
    real(8), dimension(0:nxf + 1, 0:nyf + 1), intent(in) :: ef
    integer :: i, j
    do j = 1, nyf
      do i = 1, nxf
        if (ef(i, j) > 0._8) xf(i, j) = xf(i, j) + xc((i + 1) / 2, (j + 1) / 2)
      enddo
    enddo
  end subroutine prolong

  subroutine fsh_get_layout(ng_o, lnx_o, lny_o, gi0_o, gj0_o, gnx_o, gny_o, gcnt_o, gdsp_o, ccnt_o, cdsp_o)
    ! the layout for the device-held global levels (fs_gpu_coarse = 'device')
    integer, intent(out) :: ng_o, lnx_o(maxl), lny_o(maxl)
    integer, intent(out) :: gi0_o(np), gj0_o(np), gnx_o(np), gny_o(np), gcnt_o(np), gdsp_o(np), ccnt_o(np), cdsp_o(np)
    ng_o = ng; lnx_o = lnx; lny_o = lny
    gi0_o = gi0; gj0_o = gj0; gnx_o = gnx; gny_o = gny; gcnt_o = gcnt; gdsp_o = gdsp; ccnt_o = ccnt; cdsp_o = cdsp
  end subroutine fsh_get_layout

  subroutine fsh_levels(ng_o, lnx_o, lny_o)
    ! the global levels' sizes (1 = the gathered level)
    integer, intent(out) :: ng_o, lnx_o(maxl), lny_o(maxl)
    ng_o = ng; lnx_o = lnx; lny_o = lny
  end subroutine fsh_levels

  subroutine fsh_set_faces(n1, fx, fy)
    ! the coarse face factors of the global levels (fx(ic, l), l = 2..ng; fs_mg.F90 mg_faces)
    integer, intent(in) :: n1
    real(8), intent(in) :: fx(n1, maxl), fy(n1, maxl)
    if (allocated(gfx)) deallocate(gfx, gfy)
    allocate(gfx(n1, maxl), gfy(n1, maxl))
    gfx = fx; gfy = fy
  end subroutine fsh_set_faces

  subroutine fsh_factor_from(nx, ny, e, cw, cs)
    ! the coarsest level's factor from coefficients formed on the device
    integer, intent(in) :: nx, ny
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: e, cw, cs
    call factor(nx, ny, e, cw, cs)
  end subroutine fsh_factor_from

  subroutine fsh_direct_vec(n, b, x)
    ! the coarsest level's direct solve on packed vectors (i fastest)
    integer, intent(in) :: n
    real(8), intent(in) :: b(n)
    real(8), intent(out) :: x(n)
    integer :: k, kk
    real(8) :: s
    sol(1:n) = b(1:n)
    if (fsh_colord) then
      call chol_solve(n, bw, ab, sol)
    else
      do k = 1, n
        s = sol(k)
        do kk = max(1, k - bw), k - 1
          s = s - ab(k - kk, kk) * sol(kk)
        enddo
        sol(k) = s / ab(0, k)
      enddo
      do k = n, 1, -1
        s = sol(k)
        do kk = k + 1, min(n, k + bw)
          s = s - ab(kk - k, k) * sol(kk)
        enddo
        sol(k) = s / ab(0, k)
      enddo
    endif
    x(1:n) = sol(1:n)
  end subroutine fsh_direct_vec

end module fs_host

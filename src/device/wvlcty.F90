#include "mem_attr.inc"
module wvlcty_mod
  ! True vertical velocity at rho points (CPU wvlcty_tile), for output only:
  ! the S-coordinate vertical velocity from the bottom-up integral of the
  ! mass-flux divergence times pm*pn (without the moving-interface term),
  ! plus the water of the CDR volume releases accumulated bottom-up as omega
  ! adds it to Wi (CPU wvlcty_tile since ucla-roms#TBD, which also adds pipes;
  ! the GPU has none), interpolated to the rho levels, plus the projection
  ! of u, v (slot ns, time n+1 at the end of a step: nnew, the CPU's choice)
  ! along the slopes of the S surfaces; at the physical edges the gradient
  ! condition (the edge row/column and corners copy their neighbour).
  ! Computed on the interior 1:nx, 1:ny of each tile, the edges as above.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use cdr_src_mod, only: cdr_vol_on, cdr_cmap, cd_vol, cq_w, cc_p0, cq_rel
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: wvlcty_compute

  real(rp), WORK_ATTR allocatable :: wrk(:,:,:)

contains

  subroutine wvlcty_compute(wv, slot)
    real(rp), WORK_ATTR intent(inout) :: wv(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    integer(ip), intent(in) :: slot
    integer(ip) :: i, j, k, nx, ny, nz, ns, w, e, s, n, cvo, cc, q, r
    real(rp) :: wxl, wxr, wes, wen, cfv
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ns = slot
    cvo = merge(1_ip, 0_ip, cdr_vol_on)
    if (nz < 3) error stop 'wvlcty_compute: nz < 3'
    if (.not. allocated(wrk)) allocate(wrk(decomp%is:decomp%ie, decomp%js:decomp%je, 0:nz))
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = 1, ny
      do i = 1, nx
        wrk(i, j, 0) = 0.0_rp
        do k = 1, nz
          wrk(i, j, k) = wrk(i, j, k - 1) - pm(i, j) * pn(i, j) * ( &
               FlxU(i + 1, j, k) - FlxU(i, j, k) &
               + FlxV(i, j + 1, k) - FlxV(i, j, k))
        enddo
        ! CDR volume releases (CPU wvlcty_tile), in release order
        if (cvo == 1) then
          cc = cdr_cmap(i, j)
          if (cc > 0) then
            do q = cc_p0(cc), cc_p0(cc + 1) - 1
              r = cq_rel(q)
              cfv = 0.0_rp
              do k = 1, nz
                cfv = cfv + cd_vol(r) * cq_w(k, q)
                wrk(i, j, k) = wrk(i, j, k) + pm(i, j) * pn(i, j) * cfv
              enddo
            enddo
          endif
        endif
        wv(i, j, nz) = +0.375_rp * wrk(i, j, nz) + 0.75_rp * wrk(i, j, nz - 1) &
                       - 0.125_rp * wrk(i, j, nz - 2)
        do k = nz - 1, 2, -1
          wv(i, j, k) = +0.5625_rp * (wrk(i, j, k) + wrk(i, j, k - 1)) &
                        - 0.0625_rp * (wrk(i, j, k + 1) + wrk(i, j, k - 2))
        enddo
        wv(i, j, 1) = -0.125_rp * wrk(i, j, 2) + 0.75_rp * wrk(i, j, 1) &
                      + 0.375_rp * wrk(i, j, 0)
        do k = 1, nz
          wxl = u(i, j, k, ns) * (z_r(i, j, k) - z_r(i - 1, j, k)) * (pm(i, j) + pm(i - 1, j))
          wxr = u(i + 1, j, k, ns) * (z_r(i + 1, j, k) - z_r(i, j, k)) * (pm(i + 1, j) + pm(i, j))
          wes = v(i, j, k, ns) * (z_r(i, j, k) - z_r(i, j - 1, k)) * (pn(i, j) + pn(i, j - 1))
          wen = v(i, j + 1, k, ns) * (z_r(i, j + 1, k) - z_r(i, j, k)) * (pn(i, j + 1) + pn(i, j))
          wv(i, j, k) = wv(i, j, k) + 0.25_rp * (wxl + wxr + wes + wen)
        enddo
      enddo
    enddo
    w = merge(1, 0, decomp%edge_w); e = merge(1, 0, decomp%edge_e)
    s = merge(1, 0, decomp%edge_s); n = merge(1, 0, decomp%edge_n)
    if (w + e + s + n == 0) return
    ! the gradient condition: edge columns first, then edge rows, then corners
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = 1, ny
        if (w == 1) wv(0, j, k) = wv(1, j, k)
        if (e == 1) wv(nx + 1, j, k) = wv(nx, j, k)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do i = 1, nx
        if (s == 1) wv(i, 0, k) = wv(i, 1, k)
        if (n == 1) wv(i, ny + 1, k) = wv(i, ny, k)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      if (w == 1 .and. s == 1) wv(0, 0, k) = wv(1, 1, k)
      if (w == 1 .and. n == 1) wv(0, ny + 1, k) = wv(1, ny, k)
      if (e == 1 .and. s == 1) wv(nx + 1, 0, k) = wv(nx, 1, k)
      if (e == 1 .and. n == 1) wv(nx + 1, ny + 1, k) = wv(nx, ny, k)
    enddo
  end subroutine wvlcty_compute

end module wvlcty_mod

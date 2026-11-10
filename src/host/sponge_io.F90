#include "mem_attr.inc"
module sponge_io
  ! Child side of the nesting pressure fluxes: UCLA ROMS sponge_tune.F90
  ! (SPONGE_TUNE cpp key + &sponge_tune_settings ub_tune). After every
  ! step's outputs (the CPU's adjust_orlanski), the inflow nudging
  ! coefficient ub_<side> of each open side is moved toward matching the
  ! parent's baroclinic energy flux normal to the side (up_west, up_east,
  ! vp_south, vp_north of the boundary files, interpolated in time at the
  ! CPU's time, t(n+1) + dt/2) with the child's own flux one point inside
  ! (up/vp of pflx_mod at u point 2 / nx-1, v point 2 / ny-1):
  !   ub_south/west += alpha (pf - cf), ub_north/east -= alpha (pf - cf),
  !   alpha = 0.5 ub_max dt / sponge_timescale, ub clipped to [-1, 2],
  ! with |pf| > 100 m^4 s^-3 taken as 0 (and pf_north = 0 where the north
  ! boundary row is land). The 3-D Orlanski boundary conditions (u3dbc,
  ! v3dbc, t3dbc, the passive tracers) nudge with max(cext, min(ub, 1)).
  ! ub_<side> are restart fields (ub_west, ub_east: eta_rho; ub_south,
  ! ub_north: xi_rho; the CPU's names).
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use io_server, only: io_stage8_line
  use bry_io, only: set_pflx_bry
  use pflx_mod, only: ub_tune, sponge_timescale, up, vp, ub_w, ub_e, ub_s, ub_n, pf_w, pf_e, pf_s, pf_n
  use arch_mod, only: sync_arch
  use comm_mod, only: comm_rank
  implicit none
  private
  public :: adjust_orlanski, def_rst_ub, stage_rst_ub, put_rst_ub, load_rst_ub

  real(rp), parameter :: ub_mx = 2.0_rp, ub_mn = -1.0_rp   ! the CPU's clipping range
  character(len=8), parameter :: ub_name(4) = (/'ub_west ', 'ub_east ', 'ub_south', 'ub_north'/)

contains

  subroutine adjust_orlanski()
    ! The CPU's adjust_orlanski (set_pflx, comp_cflx, the ub update), after
    ! the step. All on the device (one small kernel per open side; the host
    ! touching up/vp would migrate their managed pages every step).
    integer(ip) :: i, j, nx, ny
    real(rp) :: tneed, a, pf
    if (.not. ub_tune) return
    nx = decomp%nx; ny = decomp%ny
    ! the CPU's set_frc_data time here: tdays = t(n+1), frc_time '1/2 fwd'
    tneed = (start_time + dt * real(iic - ntstart, rp) + dt) * sec2day + 0.5_rp * dt * sec2day
    call set_pflx_bry(tneed)
    a = 0.5_rp * ub_mx * dt / sponge_timescale
    if (decomp%edge_s .and. obc_south) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 1, nx
        pf = pf_s(i)
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        ub_s(i) = ub_s(i) + a * (pf - vp(i, 2))
        ub_s(i) = max(ub_mn, ub_s(i))
        ub_s(i) = min(ub_mx, ub_s(i))
      enddo
    endif
    if (decomp%edge_n .and. obc_north) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 1, nx
        pf = pf_n(i)
        if (rmask(i, ny + 1) < 1.0_rp) pf = 0.0_rp
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        ub_n(i) = ub_n(i) - a * (pf - vp(i, ny - 1))
        ub_n(i) = max(ub_mn, ub_n(i))
        ub_n(i) = min(ub_mx, ub_n(i))
      enddo
    endif
    if (decomp%edge_w .and. obc_west) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 1, ny
        pf = pf_w(j)
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        ub_w(j) = ub_w(j) + a * (pf - up(2, j))
        ub_w(j) = max(ub_mn, ub_w(j))
        ub_w(j) = min(ub_mx, ub_w(j))
      enddo
    endif
    if (decomp%edge_e .and. obc_east) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 1, ny
        pf = pf_e(j)
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        ub_e(j) = ub_e(j) - a * (pf - up(nx - 1, j))
        ub_e(j) = max(ub_mn, ub_e(j))
        ub_e(j) = min(ub_mx, ub_e(j))
      enddo
    endif
  end subroutine adjust_orlanski

  ! ------------------------------------------------------------ restart
  subroutine line_range(side, lo, hi, fpos)
    ! This tile's part of side `side`'s restart line: the interior points
    ! 1..n of an edge tile, plus the end points 0 / n+1 (corners, 0) on the
    ! first / last tile along the edge, so that every file point is written
    ! by one tile. fpos: the file index of point lo. hi < lo: no part.
    integer, intent(in) :: side
    integer, intent(out) :: lo, hi, fpos
    logical :: on
    select case (side)
    case (1); on = decomp%edge_w
    case (2); on = decomp%edge_e
    case (3); on = decomp%edge_s
    case default; on = decomp%edge_n
    end select
    lo = 1; hi = 0; fpos = 1
    if (.not. on) return
    if (side <= 2) then
      lo = merge(0, 1, decomp%edge_s); hi = merge(int(decomp%ny) + 1, int(decomp%ny), decomp%edge_n)
      fpos = int(decomp%j0) + lo
    else
      lo = merge(0, 1, decomp%edge_w); hi = merge(int(decomp%nx) + 1, int(decomp%nx), decomp%edge_e)
      fpos = int(decomp%i0) + lo
    endif
  end subroutine line_range

  subroutine def_rst_ub(ncid, d_xi, d_eta, d_t)
    ! The CPU's restart definitions (basic_output, SPONGE_TUNE + ub_tune).
    integer, intent(in) :: ncid, d_xi, d_eta, d_t
    integer :: id
    if (.not. ub_tune) return
    id = nc_def_var(ncid, 'ub_west', (/d_eta, d_t/))
    call nc_put_att_text(ncid, 'ub_west', 'long_name', 'west bc tuning coeff')
    id = nc_def_var(ncid, 'ub_east', (/d_eta, d_t/))
    call nc_put_att_text(ncid, 'ub_east', 'long_name', 'east bc tuning coeff')
    id = nc_def_var(ncid, 'ub_north', (/d_xi, d_t/))
    call nc_put_att_text(ncid, 'ub_north', 'long_name', 'north bc tuning coeff')
    id = nc_def_var(ncid, 'ub_south', (/d_xi, d_t/))
    call nc_put_att_text(ncid, 'ub_south', 'long_name', 'south bc tuning coeff')
  end subroutine def_rst_ub

  subroutine stage_rst_ub()
    ! io_server path: every tile stages the four lines (empty off the edge).
    integer :: lo, hi, fp
    if (.not. ub_tune) return
    call line_range(1, lo, hi, fp)
    call io_stage8_line('ub_west', ub_w, int(decomp%js), int(decomp%je), lo, hi, fp)
    call line_range(2, lo, hi, fp)
    call io_stage8_line('ub_east', ub_e, int(decomp%js), int(decomp%je), lo, hi, fp)
    call line_range(3, lo, hi, fp)
    call io_stage8_line('ub_south', ub_s, int(decomp%is), int(decomp%ie), lo, hi, fp)
    call line_range(4, lo, hi, fp)
    call io_stage8_line('ub_north', ub_n, int(decomp%is), int(decomp%ie), lo, hi, fp)
  end subroutine stage_rst_ub

  subroutine put_rst_ub(ncid, rec)
    ! Serial-writer path (this tile's turn, the file open).
    integer, intent(in) :: ncid, rec
    if (.not. ub_tune) return
    call put1(1, ub_w, int(decomp%js), int(decomp%je))
    call put1(2, ub_e, int(decomp%js), int(decomp%je))
    call put1(3, ub_s, int(decomp%is), int(decomp%ie))
    call put1(4, ub_n, int(decomp%is), int(decomp%ie))
  contains
    subroutine put1(side, a, lb, ub)
      integer, intent(in) :: side, lb, ub
      real(rp), WORK_ATTR intent(in) :: a(lb:ub)
      integer :: lo, hi, fp
      real(8), allocatable :: tmp(:,:)
      call line_range(side, lo, hi, fp)
      if (hi < lo) return
      allocate(tmp(hi - lo + 1, 1))
      tmp(:, 1) = real(a(lo:hi), 8)
      call nc_put_2d_rec_d(ncid, trim(ub_name(side)), tmp, hi - lo + 1, 1, rec, fp, 1)
      deallocate(tmp)
    end subroutine put1
  end subroutine put_rst_ub

  subroutine load_rst_ub(ncid, rec)
    ! The CPU's get_init_ub: ub_<side> of the current record on the edge
    ! tiles' interior points; absent: 0 (with the CPU's warning).
    integer, intent(in) :: ncid, rec
    if (.not. ub_tune) return
    call sync_arch()
    call get1(1, ub_w, int(decomp%js), int(decomp%je), decomp%edge_w, int(decomp%ny), int(decomp%j0))
    call get1(2, ub_e, int(decomp%js), int(decomp%je), decomp%edge_e, int(decomp%ny), int(decomp%j0))
    call get1(3, ub_s, int(decomp%is), int(decomp%ie), decomp%edge_s, int(decomp%nx), int(decomp%i0))
    call get1(4, ub_n, int(decomp%is), int(decomp%ie), decomp%edge_n, int(decomp%nx), int(decomp%i0))
  contains
    subroutine get1(side, a, lb, ub, on, n, o)
      integer, intent(in) :: side, lb, ub, n, o
      logical, intent(in) :: on
      real(rp), WORK_ATTR intent(inout) :: a(lb:ub)
      real(rp), allocatable :: tmp(:,:)
      if (.not. nc_has_var(ncid, trim(ub_name(side)))) then
        if (comm_rank() == 0) write(*,'(3a)') ' --WARNING: ', trim(ub_name(side)), &
             ' not in initial file, setting to zero.'
        a = 0.0_rp
        return
      endif
      if (.not. on) return
      allocate(tmp(n, 1))
      call nc_read_2d(ncid, trim(ub_name(side)), tmp, n, 1, rec, o + 1, 1)
      a(1:n) = tmp(:, 1)
      deallocate(tmp)
    end subroutine get1
  end subroutine load_rst_ub

end module sponge_io

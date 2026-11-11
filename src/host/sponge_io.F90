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
  use pflx_mod, only: ub_tune, sponge_timescale, up, vp, ub_w, ub_e, ub_s, ub_n, pf_w, pf_e, pf_s, pf_n, &
                      wrt_sponge, sponge_avg
  use arch_mod, only: sync_arch
  use comm_mod, only: comm_rank
  implicit none
  private
  public :: adjust_orlanski, def_rst_ub, stage_rst_ub, put_rst_ub, load_rst_ub, init_spn_arrays
  ! for sponge_out (the _spn file): the child flux of the step, the averages
  ! (or the step's values without sponge_avg) and the average counter
  public :: cf_w, cf_e, cf_s, cf_n, cfa_w, cfa_e, cfa_s, cfa_n, pfa_w, pfa_e, pfa_s, pfa_n
  public :: uba_w, uba_e, uba_s, uba_n, navg_spn
  real(rp), WORK_ATTR allocatable :: cf_w(:), cf_e(:), cf_s(:), cf_n(:)
  real(rp), WORK_ATTR allocatable :: cfa_w(:), cfa_e(:), cfa_s(:), cfa_n(:)
  real(rp), WORK_ATTR allocatable :: pfa_w(:), pfa_e(:), pfa_s(:), pfa_n(:)
  real(rp), WORK_ATTR allocatable :: uba_w(:), uba_e(:), uba_s(:), uba_n(:)
  real(rp) :: navg_spn = 0.0_rp

  real(rp), parameter :: ub_mx = 2.0_rp, ub_mn = -1.0_rp   ! the CPU's clipping range
  character(len=8), parameter :: ub_name(4) = (/'ub_west ', 'ub_east ', 'ub_south', 'ub_north'/)

contains

  subroutine init_spn_arrays()
    ! After alloc_state; zero (the corners and the off-edge parts stay 0).
    integer(ip) :: is, ie, js, je
    if (.not. ub_tune .or. allocated(cf_w)) return
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    allocate(cf_w(js:je), cf_e(js:je), cf_s(is:ie), cf_n(is:ie))
    allocate(cfa_w(js:je), cfa_e(js:je), cfa_s(is:ie), cfa_n(is:ie))
    allocate(pfa_w(js:je), pfa_e(js:je), pfa_s(is:ie), pfa_n(is:ie))
    allocate(uba_w(js:je), uba_e(js:je), uba_s(is:ie), uba_n(is:ie))
    cf_w = 0; cf_e = 0; cf_s = 0; cf_n = 0
    cfa_w = 0; cfa_e = 0; cfa_s = 0; cfa_n = 0
    pfa_w = 0; pfa_e = 0; pfa_s = 0; pfa_n = 0
    uba_w = 0; uba_e = 0; uba_s = 0; uba_n = 0
  end subroutine init_spn_arrays

  subroutine adjust_orlanski()
    ! The CPU's adjust_orlanski (set_pflx, comp_cflx, the ub update) and
    ! calc_spn_avg, before the step's outputs (ucla-roms sponge-tune-pio-fixes
    ! order). All on the device (one small kernel per open side; the host
    ! touching up/vp would migrate their managed pages every step). The
    ! clamped parent flux and the child flux are kept in pf_<side> /
    ! cf_<side>; with wrt_sponge their running means (sponge_avg) or the
    ! step's values (coef 1) go to cfa/pfa/uba_<side> for sponge_out.
    integer(ip) :: i, j, nx, ny
    real(rp) :: tneed, a, pf, cf, c
    logical :: av
    if (.not. ub_tune) return
    nx = decomp%nx; ny = decomp%ny
    ! the CPU's set_frc_data time here: tdays = t(n+1), frc_time '1/2 fwd'
    tneed = (start_time + dt * real(iic - ntstart, rp) + dt) * sec2day + 0.5_rp * dt * sec2day
    call set_pflx_bry(tneed)
    a = 0.5_rp * ub_mx * dt / sponge_timescale
    ! the CPU's calc_spn_avg counter (reset by the output)
    navg_spn = navg_spn + 1.0_rp
    c = 1.0_rp / navg_spn
    if (.not. sponge_avg) c = 1.0_rp
    av = wrt_sponge
    if (decomp%edge_s .and. obc_south) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 1, nx
        pf = pf_s(i)
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        cf = vp(i, 2)
        pf_s(i) = pf
        cf_s(i) = cf
        ub_s(i) = ub_s(i) + a * (pf - cf)
        ub_s(i) = max(ub_mn, ub_s(i))
        ub_s(i) = min(ub_mx, ub_s(i))
        if (av) then
          cfa_s(i) = cfa_s(i) * (1.0_rp - c) + cf * c
          pfa_s(i) = pfa_s(i) * (1.0_rp - c) + pf * c
          uba_s(i) = uba_s(i) * (1.0_rp - c) + ub_s(i) * c
        endif
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
        cf = vp(i, ny - 1)
        pf_n(i) = pf
        cf_n(i) = cf
        ub_n(i) = ub_n(i) - a * (pf - cf)
        ub_n(i) = max(ub_mn, ub_n(i))
        ub_n(i) = min(ub_mx, ub_n(i))
        if (av) then
          cfa_n(i) = cfa_n(i) * (1.0_rp - c) + cf * c
          pfa_n(i) = pfa_n(i) * (1.0_rp - c) + pf * c
          uba_n(i) = uba_n(i) * (1.0_rp - c) + ub_n(i) * c
        endif
      enddo
    endif
    if (decomp%edge_w .and. obc_west) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 1, ny
        pf = pf_w(j)
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        cf = up(2, j)
        pf_w(j) = pf
        cf_w(j) = cf
        ub_w(j) = ub_w(j) + a * (pf - cf)
        ub_w(j) = max(ub_mn, ub_w(j))
        ub_w(j) = min(ub_mx, ub_w(j))
        if (av) then
          cfa_w(j) = cfa_w(j) * (1.0_rp - c) + cf * c
          pfa_w(j) = pfa_w(j) * (1.0_rp - c) + pf * c
          uba_w(j) = uba_w(j) * (1.0_rp - c) + ub_w(j) * c
        endif
      enddo
    endif
    if (decomp%edge_e .and. obc_east) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 1, ny
        pf = pf_e(j)
        if (abs(pf) > 100.0_rp) pf = 0.0_rp
        cf = up(nx - 1, j)
        pf_e(j) = pf
        cf_e(j) = cf
        ub_e(j) = ub_e(j) - a * (pf - cf)
        ub_e(j) = max(ub_mn, ub_e(j))
        ub_e(j) = min(ub_mx, ub_e(j))
        if (av) then
          cfa_e(j) = cfa_e(j) * (1.0_rp - c) + cf * c
          pfa_e(j) = pfa_e(j) * (1.0_rp - c) + pf * c
          uba_e(j) = uba_e(j) * (1.0_rp - c) + ub_e(j) * c
        endif
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

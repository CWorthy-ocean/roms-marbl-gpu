#include "mem_attr.inc"
module sponge_out
  ! The sponge-tuning output (&sponge_tune_settings wrt_sponge): ucla-roms
  ! sponge_tune write_sp_tune / create_sp_tune_file as fixed on its branch
  ! sponge-tune-pio-fixes. Every output_period_sponge (counted in steps of
  ! dt after the step's tuning), a record of <output_root>_spn.<date>.nc
  ! (nrpf_sponge records per file, the date of the first record) with
  ! ocean_time (time n+1, seconds) and, for every open side, cf_<side>
  ! (the child's flux one point inside), pf_<side> (the parent's flux from
  ! the boundary file, clamped as used) and ub_<side> (the coefficient):
  ! their means since the last record (sponge_avg) or the values of the
  ! output step (sponge_avg off), along xi_rho (south, north) or eta_rho
  ! (east, west), float32 as on the CPU, corners 0. The per-tile lines are
  ! summed over the ranks; compute rank 0 writes.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use his_io, only: output_fname, refdatestr
  use comm_mod, only: comm_rank, comm_reduce_sum
  use arch_mod, only: sync_arch
  use pflx_mod, only: ub_tune, wrt_sponge, nrpf_sponge, output_period_sponge
  use sponge_io, only: cfa_w, cfa_e, cfa_s, cfa_n, pfa_w, pfa_e, pfa_s, pfa_n, uba_w, uba_e, uba_s, uba_n, navg_spn
  implicit none
  private
  public :: set_spn_root, init_spn_output, wrt_spn

  character(len=512) :: root = './output', fname = ''
  real(rp) :: output_time = 0.0_rp
  integer :: record = 0, gx = 0, gy = 0
  logical :: on = .false.
  character(len=5), parameter :: sname(4) = (/'south', 'north', 'east ', 'west '/)
  character(len=5), parameter :: slabel(4) = (/'South', 'North', 'East ', 'West '/)

contains

  subroutine set_spn_root(r)
    character(len=*), intent(in) :: r
    root = trim(r)
  end subroutine set_spn_root

  subroutine init_spn_output()
    on = ub_tune .and. wrt_sponge
    if (.not. on) return
    if (output_period_sponge <= 0.0_rp) error stop 'sponge_out: output_period_sponge must be > 0'
    if (nrpf_sponge < 1) error stop 'sponge_out: nrpf_sponge must be >= 1'
    gx = int(decomp%gnx) + 2; gy = int(decomp%gny) + 2
    record = nrpf_sponge; output_time = 0.0_rp
  end subroutine init_spn_output

  logical function side_open(s)
    integer, intent(in) :: s
    select case (s)
    case (1); side_open = obc_south
    case (2); side_open = obc_north
    case (3); side_open = obc_east
    case default; side_open = obc_west
    end select
  end function side_open

  subroutine wrt_spn()
    ! After adjust_orlanski, every step (the CPU's output test in it).
    integer :: s, ntot, off, ncid, v
    real(8), allocatable :: line(:), gline(:)
    real(rp) :: twrite
    if (.not. on) return
    output_time = output_time + dt
    if (output_time < output_period_sponge) return
    output_time = 0.0_rp
    twrite = start_time + dt * real(iic - ntstart + 1, rp)
    if (record == nrpf_sponge) then
      record = 0
      call output_fname(root, 'spn', twrite, fname)
      if (comm_rank() == 0) call create_spn_file(trim(fname))
    endif
    record = record + 1
    ! lines of every side, (cf, pf, ub) x (south, north: gx; east, west: gy)
    ntot = 3 * (2 * gx + 2 * gy)
    allocate(line(ntot), gline(ntot))
    line = 0.0d0
    call sync_arch()
    off = 0
    call put_line(1, decomp%edge_s, cfa_s, pfa_s, uba_s, int(decomp%is), int(decomp%ie), gx, int(decomp%i0), &
                  decomp%edge_w, decomp%edge_e, int(decomp%nx))
    call put_line(2, decomp%edge_n, cfa_n, pfa_n, uba_n, int(decomp%is), int(decomp%ie), gx, int(decomp%i0), &
                  decomp%edge_w, decomp%edge_e, int(decomp%nx))
    call put_line(3, decomp%edge_e, cfa_e, pfa_e, uba_e, int(decomp%js), int(decomp%je), gy, int(decomp%j0), &
                  decomp%edge_s, decomp%edge_n, int(decomp%ny))
    call put_line(4, decomp%edge_w, cfa_w, pfa_w, uba_w, int(decomp%js), int(decomp%je), gy, int(decomp%j0), &
                  decomp%edge_s, decomp%edge_n, int(decomp%ny))
    call comm_reduce_sum(ntot, line, gline)
    navg_spn = 0.0_rp
    if (comm_rank() == 0) then
      call nc_open_write(trim(fname), ncid)
      call nc_put_1d_rec(ncid, 'ocean_time', twrite, record)
      off = 0
      do s = 1, 4
        if (.not. side_open(s)) then
          off = off + 3 * merge(gx, gy, s <= 2)
          cycle
        endif
        do v = 1, 3
          call put_rec(ncid, trim(vname(v, s)), gline(off + 1:off + merge(gx, gy, s <= 2)))
          off = off + merge(gx, gy, s <= 2)
        enddo
      enddo
      call nc_close(ncid)
      write(*,'(7x,a,1x,f11.4,2x,a,i7,1x,a,i4)') 'orlanski_tune :: wrote output, tdays =', twrite * sec2day, &
           'step =', iic, 'rec =', record
    endif
    deallocate(line, gline)
  contains
    subroutine put_line(s, edge, cf, pf, ub, lb, ubd, n, o, first, last, nl)
      ! this tile's part (interior points, plus the corner on the first /
      ! last tile along the edge) of side s's three lines
      integer, intent(in) :: s, lb, ubd, n, o, nl
      logical, intent(in) :: edge, first, last
      real(rp), WORK_ATTR intent(in) :: cf(lb:ubd), pf(lb:ubd), ub(lb:ubd)
      integer :: l, lo, hi, base
      base = off
      off = off + 3 * n
      if (.not. (edge .and. side_open(s))) return
      lo = merge(0, 1, first); hi = merge(nl + 1, nl, last)
      do l = lo, hi
        line(base + o + l) = cf(l)
        line(base + n + o + l) = pf(l)
        line(base + 2 * n + o + l) = ub(l)
      enddo
    end subroutine put_line
  end subroutine wrt_spn

  character(len=16) function vname(v, s)
    integer, intent(in) :: v, s
    character(len=3), parameter :: pre(3) = (/'cf_', 'pf_', 'ub_'/)
    vname = pre(v)//trim(sname(s))
  end function vname

  subroutine put_rec(ncid, name, x)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(8), intent(in) :: x(:)
    real(4), allocatable :: f(:,:)
    allocate(f(size(x), 1))
    f(:, 1) = real(x, 4)
    call nc_put_2d_rec_f(ncid, name, f, size(x), 1, record, 1, 1)
    deallocate(f)
  end subroutine put_rec

  subroutine create_spn_file(path)
    ! The CPU's create_sp_tune_file: ocean_time and, for every open side,
    ! cf_/pf_/ub_<side> with its long_name and units.
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_x, d_y, id, s, d
    call nc_create(path, ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_x = nc_def_dim(ncid, 'xi_rho', gx)
    d_y = nc_def_dim(ncid, 'eta_rho', gy)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    do s = 1, 4
      if (.not. side_open(s)) cycle
      d = merge(d_x, d_y, s <= 2)
      id = nc_def_var_f(ncid, trim(vname(1, s)), (/d, d_t/))
      call nc_put_att_text(ncid, trim(vname(1, s)), 'long_name', trim(slabel(s))//' boundary child flux')
      call nc_put_att_text(ncid, trim(vname(1, s)), 'units', 'W/m')
      id = nc_def_var_f(ncid, trim(vname(2, s)), (/d, d_t/))
      call nc_put_att_text(ncid, trim(vname(2, s)), 'long_name', trim(slabel(s))//' boundary parent flux')
      call nc_put_att_text(ncid, trim(vname(2, s)), 'units', 'W/m')
      id = nc_def_var_f(ncid, trim(vname(3, s)), (/d, d_t/))
      call nc_put_att_text(ncid, trim(vname(3, s)), 'long_name', trim(slabel(s))//' boundary binding velocity')
      call nc_put_att_text(ncid, trim(vname(3, s)), 'units', 'm/s')
    enddo
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu sponge tuning')
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created sponge-tuning file: ', trim(path)
  end subroutine create_spn_file

end module sponge_out

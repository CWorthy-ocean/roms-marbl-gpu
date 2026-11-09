#include "mem_attr.inc"
module upscale_out
  ! Upscaling output (CPU upscale_output.F90, UPSCALING), for runs that write
  ! the open-boundary transects a parent domain's CDR forcing is made from.
  ! The open-boundary inflow term 0.5*(t_out - t_in)*inflow of the
  ! horizontal tracer fluxes, once the CPU's UPSCALING term, applies in every
  ! run on the open sides (gpu_state open_edge_codes). With do_upscale:
  !  - every step (after step3d_t) the outward advective flux of ALK, DIC,
  !    their ALT_CO2 twins and the CDR_LITE tracers through each open
  !    boundary face (the corrector's UP3 faces with the term, recomputed
  !    from the same operands), and the running mean of ALK - ALK_ALT_CO2,
  !    DIC - DIC_ALT_CO2, each CDR_LITE flux and the layer thickness of the
  !    boundary cells (corners included), as the CPU's calc_average:
  !    avg*(1 - 1/n) + x/n;
  !  - records as the CPU's wrt_upscale: one at the first step, then every
  !    output_period_uscl, nrpf_uscl per file, at the time of the step's
  !    start; main's final call adds the last step's fluxes once more and
  !    writes when the period is due. Files <root>_uscl.<date>.nc
  !    (his_io output_fname; double), written by compute rank 0 from a
  !    sum over the ranks of the boundary lines.
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use formulas, only: coef_up3
  use comm_mod, only: comm_rank, comm_reduce_sum
  use passive_mod, only: ntp, tp, passive_name, ncdr, cdr0
  use marbl_gpu, only: marbl_on
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: read_upscale_namelist, set_upscale_root, init_upscale_output, wrt_upscale, wrt_upscale_final

  integer  :: nrpf_uscl = 48
  real(rp) :: output_period_uscl = 100.0_rp
  logical  :: do_upscale = .false.
  namelist /upscale_settings/ nrpf_uscl, output_period_uscl, do_upscale

  logical :: on = .false.
  character(len=512) :: root = './output', fname = ''
  integer :: record = 0, nfile = 0, navg = 0, nsel = 0, nv = 0
  real(rp) :: otime = 0.0_rp
  ! sides in the CPU's file order
  integer, parameter :: SN = 1, SS = 2, SE = 3, SW = 4
  character(len=5), parameter :: sname(4) = (/'north', 'south', 'east ', 'west '/)
  character(len=8), parameter :: sadj(4) = (/'northern', 'southern', 'eastern ', 'western '/)
  logical :: side_on(4) = .false., side_mine(4) = .false.
  integer :: nline(4) = 0, boff(4) = 0, nbuf = 0, ncbuf = 0
  ! selected passive tracers: ALK, ALK_ALT_CO2, DIC, DIC_ALT_CO2, CDR_LITE
  integer(ip), WORK_ATTR allocatable :: sel(:)
  ! outward fluxes rX(along, k, sel) and means aX(along, k, v): v = 1 ALK
  ! additionality, 2 DIC additionality, 2+m CDR_LITE tracer m, nv layer thickness
  real(rp), WORK_ATTR allocatable :: rw(:,:,:), re(:,:,:), rs(:,:,:), rn(:,:,:)
  real(rp), WORK_ATTR allocatable :: aw(:,:,:), ae(:,:,:), as(:,:,:), an(:,:,:)
  real(8), allocatable :: hbuf(:), gbuf(:)

contains

  subroutine read_upscale_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    rewind(unit)
    read(unit, nml=upscale_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'UPSCALE_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_upscale_namelist'
    endif
    if (ios < 0) do_upscale = .false.
  end subroutine read_upscale_namelist

  subroutine set_upscale_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_upscale_root

  integer(ip) function pidx(name) result(itp)
    character(len=*), intent(in) :: name
    integer(ip) :: m
    itp = 0
    do m = 1, ntp
      if (trim(passive_name(m)) == trim(name)) itp = m
    enddo
    if (itp == 0) then
      write(*,'(3a)') 'init_upscale_output: upscaling needs the MARBL tracer ', trim(name), ' (not in this run)'
      error stop 'init_upscale_output: missing MARBL tracer'
    endif
  end function pidx

  subroutine init_upscale_output()
    integer :: is, ie, js, je, nz, m, s
    on = .false.
    if (.not. do_upscale) return
    if (.not. marbl_on) error stop 'init_upscale_output: upscaling needs MARBL'
    if (nrpf_uscl < 1) error stop 'init_upscale_output: nrpf_uscl must be >= 1'
    nsel = 4 + int(ncdr)
    nv = 3 + int(ncdr)
    allocate(sel(nsel))
    sel(1) = pidx('ALK'); sel(2) = pidx('ALK_ALT_CO2'); sel(3) = pidx('DIC'); sel(4) = pidx('DIC_ALT_CO2')
    do m = 1, int(ncdr)
      sel(4 + m) = cdr0 + m
    enddo
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    allocate(rw(js:je, nz, nsel), re(js:je, nz, nsel), rs(is:ie, nz, nsel), rn(is:ie, nz, nsel))
    allocate(aw(js:je, nz, nv), ae(js:je, nz, nv), as(is:ie, nz, nv), an(is:ie, nz, nv))
    call zero_all()
    side_on = (/obc_north, obc_south, obc_east, obc_west/)
    side_mine = (/decomp%edge_n, decomp%edge_s, decomp%edge_e, decomp%edge_w/)
    nline(SN) = int(decomp%gnx) + 2; nline(SS) = nline(SN)
    nline(SE) = int(decomp%gny) + 2; nline(SW) = nline(SE)
    nbuf = 0
    do s = 1, 4
      boff(s) = nbuf
      if (side_on(s)) nbuf = nbuf + nv * nline(s) * nz
    enddo
    ncbuf = 2 * sum(nline, mask=side_on)
    allocate(hbuf(max(nbuf, ncbuf)), gbuf(max(nbuf, ncbuf)))
    ! the CPU's start: a record at the first step
    record = nrpf_uscl; nfile = 0; navg = 0; otime = output_period_uscl
    on = .true.
    if (comm_rank() == 0) write(*,'(a,es12.4,a,i0,a,4l2)') 'upscale output: period ', output_period_uscl, &
         ' s, ', nrpf_uscl, ' records per file, sides N S E W', side_on
  end subroutine init_upscale_output

  subroutine zero_all()
    integer :: i, j, k, q, is, ie, js, je, nz, ns, nvv
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; ns = nsel; nvv = nv
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = js, je
        do q = 1, ns
          rw(j, k, q) = 0.0_rp; re(j, k, q) = 0.0_rp
        enddo
        do q = 1, nvv
          aw(j, k, q) = 0.0_rp; ae(j, k, q) = 0.0_rp
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do i = is, ie
        do q = 1, ns
          rs(i, k, q) = 0.0_rp; rn(i, k, q) = 0.0_rp
        enddo
        do q = 1, nvv
          as(i, k, q) = 0.0_rp; an(i, k, q) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine zero_all

  subroutine capture()
    ! The outward flux through the boundary faces of the interior edge
    ! cells: the corrector's UP3 face of up3_div (passive_column) from slot
    ! nrhs, with the UPSCALING term; west/south -F, east/north F.
    integer :: i, j, k, q, nx, ny, nz, ns, nr
    integer(ip) :: itr, c0, cm, cp
    real(rp) :: t0, xm, xp, fu, m0, m1, m2, m3, dc, dl, dr, de, ft, f
    logical :: b1, b2, bo, bc
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ns = nsel; nr = nrhs
    if (decomp%edge_w .and. obc_west) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do q = 1, ns
        do k = 1, nz
          do j = 1, ny
            itr = sel(q)
            c0 = tpcol(1, j); cm = tpcol(0, j); cp = tpcol(2, j)
            t0 = real(tp(itr, c0, k, nr), rp); xm = real(tp(itr, cm, k, nr), rp); xp = real(tp(itr, cp, k, nr), rp)
            fu = FlxU(1, j, k)
            m0 = umask(0, j); m1 = umask(1, j); m2 = umask(2, j)
            b1 = m1 > 0.5_rp; b2 = m2 > 0.5_rp
            bo = rmask(0, j) > 0.5_rp; bc = rmask(1, j) > 0.5_rp
            dc = 0.0_rp
            if (b1) dc = t0 - xm
            dl = dc
            dr = 0.0_rp
            if (b2) dr = xp - t0
            if (bo .and. bc) then
              ft = 0.5_rp * (xm + t0)
            elseif (bo) then
              ft = xm
            elseif (bc) then
              ft = t0
            else
              ft = 0.0_rp
            endif
            f = 0.0_rp
            if (b1) f = ft * fu - coef_up3 * ((dc - dl) * m0 * m1 * max(fu, 0.0_rp) + (dr - dc) * m1 * m2 * min(fu, 0.0_rp))
            if (b1) f = f + 0.5_rp * (xm - t0) * max(fu, 0.0_rp)    ! the open-boundary inflow term
            rw(j, k, q) = -f
          enddo
        enddo
      enddo
    endif
    if (decomp%edge_e .and. obc_east) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do q = 1, ns
        do k = 1, nz
          do j = 1, ny
            itr = sel(q)
            c0 = tpcol(nx, j); cm = tpcol(nx - 1, j); cp = tpcol(nx + 1, j)
            t0 = real(tp(itr, c0, k, nr), rp); xm = real(tp(itr, cm, k, nr), rp); xp = real(tp(itr, cp, k, nr), rp)
            fu = FlxU(nx + 1, j, k)
            m1 = umask(nx, j); m2 = umask(nx + 1, j); m3 = umask(nx + 2, j)
            b1 = m1 > 0.5_rp; b2 = m2 > 0.5_rp
            bc = rmask(nx, j) > 0.5_rp; bo = rmask(nx + 1, j) > 0.5_rp
            dc = 0.0_rp
            if (b1) dc = t0 - xm
            dr = 0.0_rp
            if (b2) dr = xp - t0
            de = dr
            if (bc .and. bo) then
              ft = 0.5_rp * (t0 + xp)
            elseif (bc) then
              ft = t0
            elseif (bo) then
              ft = xp
            else
              ft = 0.0_rp
            endif
            f = 0.0_rp
            if (b2) f = ft * fu - coef_up3 * ((dr - dc) * m1 * m2 * max(fu, 0.0_rp) + (de - dr) * m2 * m3 * min(fu, 0.0_rp))
            if (b2) f = f + 0.5_rp * (xp - t0) * min(fu, 0.0_rp)    ! the open-boundary inflow term
            re(j, k, q) = f
          enddo
        enddo
      enddo
    endif
    if (decomp%edge_s .and. obc_south) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do q = 1, ns
        do k = 1, nz
          do i = 1, nx
            itr = sel(q)
            c0 = tpcol(i, 1); cm = tpcol(i, 0); cp = tpcol(i, 2)
            t0 = real(tp(itr, c0, k, nr), rp); xm = real(tp(itr, cm, k, nr), rp); xp = real(tp(itr, cp, k, nr), rp)
            fu = FlxV(i, 1, k)
            m0 = vmask(i, 0); m1 = vmask(i, 1); m2 = vmask(i, 2)
            b1 = m1 > 0.5_rp; b2 = m2 > 0.5_rp
            bo = rmask(i, 0) > 0.5_rp; bc = rmask(i, 1) > 0.5_rp
            dc = 0.0_rp
            if (b1) dc = t0 - xm
            dl = dc
            dr = 0.0_rp
            if (b2) dr = xp - t0
            if (bo .and. bc) then
              ft = 0.5_rp * (xm + t0)
            elseif (bo) then
              ft = xm
            elseif (bc) then
              ft = t0
            else
              ft = 0.0_rp
            endif
            f = 0.0_rp
            if (b1) f = ft * fu - coef_up3 * ((dc - dl) * m0 * m1 * max(fu, 0.0_rp) + (dr - dc) * m1 * m2 * min(fu, 0.0_rp))
            if (b1) f = f + 0.5_rp * (xm - t0) * max(fu, 0.0_rp)    ! the open-boundary inflow term
            rs(i, k, q) = -f
          enddo
        enddo
      enddo
    endif
    if (decomp%edge_n .and. obc_north) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do q = 1, ns
        do k = 1, nz
          do i = 1, nx
            itr = sel(q)
            c0 = tpcol(i, ny); cm = tpcol(i, ny - 1); cp = tpcol(i, ny + 1)
            t0 = real(tp(itr, c0, k, nr), rp); xm = real(tp(itr, cm, k, nr), rp); xp = real(tp(itr, cp, k, nr), rp)
            fu = FlxV(i, ny + 1, k)
            m1 = vmask(i, ny); m2 = vmask(i, ny + 1); m3 = vmask(i, ny + 2)
            b1 = m1 > 0.5_rp; b2 = m2 > 0.5_rp
            bc = rmask(i, ny) > 0.5_rp; bo = rmask(i, ny + 1) > 0.5_rp
            dc = 0.0_rp
            if (b1) dc = t0 - xm
            dr = 0.0_rp
            if (b2) dr = xp - t0
            de = dr
            if (bc .and. bo) then
              ft = 0.5_rp * (t0 + xp)
            elseif (bc) then
              ft = t0
            elseif (bo) then
              ft = xp
            else
              ft = 0.0_rp
            endif
            f = 0.0_rp
            if (b2) f = ft * fu - coef_up3 * ((dr - dc) * m1 * m2 * max(fu, 0.0_rp) + (de - dr) * m2 * m3 * min(fu, 0.0_rp))
            if (b2) f = f + 0.5_rp * (xp - t0) * min(fu, 0.0_rp)    ! the open-boundary inflow term
            rn(i, k, q) = f
          enddo
        enddo
      enddo
    endif
  end subroutine capture

  subroutine average()
    ! The CPU's calc_average over the tile's boundary points (corners
    ! included): avg*(1 - coef) + x*coef, coef = 1/navg.
    integer :: i, j, k, m, nx, ny, nz, nc, j0, j1, i0, i1
    real(rp) :: coef, omc
    navg = navg + 1
    coef = 1.0_rp / real(navg, rp)
    omc = 1.0_rp - coef
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; nc = int(ncdr)
    j0 = merge(0, 1, decomp%edge_s); j1 = merge(ny + 1, ny, decomp%edge_n)
    i0 = merge(0, 1, decomp%edge_w); i1 = merge(nx + 1, nx, decomp%edge_e)
    if (decomp%edge_w .and. obc_west) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = j0, j1
          aw(j, k, 1) = aw(j, k, 1) * omc + (rw(j, k, 1) - rw(j, k, 2)) * coef
          aw(j, k, 2) = aw(j, k, 2) * omc + (rw(j, k, 3) - rw(j, k, 4)) * coef
          do m = 1, nc
            aw(j, k, 2 + m) = aw(j, k, 2 + m) * omc + rw(j, k, 4 + m) * coef
          enddo
          aw(j, k, nc + 3) = aw(j, k, nc + 3) * omc + Hz(0, j, k) * coef
        enddo
      enddo
    endif
    if (decomp%edge_e .and. obc_east) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = j0, j1
          ae(j, k, 1) = ae(j, k, 1) * omc + (re(j, k, 1) - re(j, k, 2)) * coef
          ae(j, k, 2) = ae(j, k, 2) * omc + (re(j, k, 3) - re(j, k, 4)) * coef
          do m = 1, nc
            ae(j, k, 2 + m) = ae(j, k, 2 + m) * omc + re(j, k, 4 + m) * coef
          enddo
          ae(j, k, nc + 3) = ae(j, k, nc + 3) * omc + Hz(nx + 1, j, k) * coef
        enddo
      enddo
    endif
    if (decomp%edge_s .and. obc_south) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do i = i0, i1
          as(i, k, 1) = as(i, k, 1) * omc + (rs(i, k, 1) - rs(i, k, 2)) * coef
          as(i, k, 2) = as(i, k, 2) * omc + (rs(i, k, 3) - rs(i, k, 4)) * coef
          do m = 1, nc
            as(i, k, 2 + m) = as(i, k, 2 + m) * omc + rs(i, k, 4 + m) * coef
          enddo
          as(i, k, nc + 3) = as(i, k, nc + 3) * omc + Hz(i, 0, k) * coef
        enddo
      enddo
    endif
    if (decomp%edge_n .and. obc_north) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do i = i0, i1
          an(i, k, 1) = an(i, k, 1) * omc + (rn(i, k, 1) - rn(i, k, 2)) * coef
          an(i, k, 2) = an(i, k, 2) * omc + (rn(i, k, 3) - rn(i, k, 4)) * coef
          do m = 1, nc
            an(i, k, 2 + m) = an(i, k, 2 + m) * omc + rn(i, k, 4 + m) * coef
          enddo
          an(i, k, nc + 3) = an(i, k, nc + 3) * omc + Hz(i, ny + 1, k) * coef
        enddo
      enddo
    endif
  end subroutine average

  subroutine wrt_upscale()
    ! After step3d_t (the CPU calls it at the end of its tracer loop): this
    ! step's fluxes into the means, a record when due, at time n.
    if (.not. on) return
    call capture()
    call advance(start_time + dt * real(iic - ntstart, rp))
  end subroutine wrt_upscale

  subroutine wrt_upscale_final()
    ! main's final call: the last fluxes once more, at the end time.
    if (.not. on) return
    call advance(start_time + dt * real(iic - ntstart + 1, rp))
  end subroutine wrt_upscale_final

  subroutine advance(tnow)
    real(rp), intent(in) :: tnow
    call average()
    otime = otime + dt
    if (record == nrpf_uscl .and. otime >= output_period_uscl) then
      nfile = nfile + 1
      record = 0
      call output_fname(root, 'uscl', tnow, fname)
      call create_file(fname)
    endif
    if (otime >= output_period_uscl) then
      record = record + 1
      call write_record(tnow)
      otime = 0.0_rp
      navg = 0
    endif
  end subroutine advance

  subroutine write_record(tnow)
    ! The means summed over the ranks into the global lines (each boundary
    ! point is on one tile), written by rank 0; then zeroed.
    real(rp), intent(in) :: tnow
    integer :: s, v, ncid
    character(len=64) :: vn
    call sync_arch()
    hbuf(1:nbuf) = 0.0d0
    if (side_on(SN) .and. side_mine(SN)) call put_x(an, boff(SN))
    if (side_on(SS) .and. side_mine(SS)) call put_x(as, boff(SS))
    if (side_on(SE) .and. side_mine(SE)) call put_y(ae, boff(SE))
    if (side_on(SW) .and. side_mine(SW)) call put_y(aw, boff(SW))
    call comm_reduce_sum(nbuf, hbuf, gbuf)
    if (comm_rank() == 0) then
      call nc_open_write(trim(fname), ncid)
      call nc_put_1d_rec(ncid, 'ocean_time', tnow, record)
      do s = 1, 4
        if (.not. side_on(s)) cycle
        do v = 1, nv
          vn = var_name(v, s)
          call nc_put_2d_rec_d(ncid, trim(vn), gbuf(boff(s) + (v - 1) * nline(s) * decomp%nz + 1), &
                               nline(s), int(decomp%nz), record)
        enddo
      enddo
      call nc_close(ncid)
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote uscl t = ', tnow, '  rec = ', record, '  file ', nfile
    endif
    call zero_means()
  contains
    subroutine put_x(a, off)
      real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%nz, nv)
      integer, intent(in) :: off
      integer :: i, k, vv, i0, i1, g
      i0 = merge(0, 1, decomp%edge_w); i1 = merge(int(decomp%nx) + 1, int(decomp%nx), decomp%edge_e)
      do vv = 1, nv
        do k = 1, decomp%nz
          do i = i0, i1
            g = int(decomp%i0) + i          ! xi_rho index (1 = the western ring)
            hbuf(off + (vv - 1) * nline(SN) * decomp%nz + (k - 1) * nline(SN) + g) = a(i, k, vv)
          enddo
        enddo
      enddo
    end subroutine put_x
    subroutine put_y(a, off)
      real(rp), WORK_ATTR intent(in) :: a(decomp%js:decomp%je, decomp%nz, nv)
      integer, intent(in) :: off
      integer :: j, k, vv, j0, j1, g
      j0 = merge(0, 1, decomp%edge_s); j1 = merge(int(decomp%ny) + 1, int(decomp%ny), decomp%edge_n)
      do vv = 1, nv
        do k = 1, decomp%nz
          do j = j0, j1
            g = int(decomp%j0) + j          ! eta_rho index (1 = the southern ring)
            hbuf(off + (vv - 1) * nline(SE) * decomp%nz + (k - 1) * nline(SE) + g) = a(j, k, vv)
          enddo
        enddo
      enddo
    end subroutine put_y
  end subroutine write_record

  subroutine zero_means()
    integer :: i, j, k, q, is, ie, js, je, nz, nvv
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; nvv = nv
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = js, je
        do q = 1, nvv
          aw(j, k, q) = 0.0_rp; ae(j, k, q) = 0.0_rp
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do i = is, ie
        do q = 1, nvv
          as(i, k, q) = 0.0_rp; an(i, k, q) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine zero_means

  function var_name(v, s) result(vn)
    integer, intent(in) :: v, s
    character(len=64) :: vn
    if (v == 1) then
      vn = 'ALK_add_'//trim(sname(s))
    elseif (v == 2) then
      vn = 'DIC_add_'//trim(sname(s))
    elseif (v == nv) then
      vn = 'h_'//trim(sname(s))
    else
      vn = trim(passive_name(cdr0 + v - 2))//'_add_'//trim(sname(s))
    endif
  end function var_name

  subroutine create_file(path)
    ! The CPU's dimensions, variables (double), attributes and order (sides
    ! north, south, east, west), then the boundary coordinates.
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, id, n, s, v, dl, c0, nz
    character(len=64) :: vn
    nz = decomp%nz
    ! coordinates first: every rank takes part in the sum
    call sync_arch()
    hbuf(1:ncbuf) = 0.0d0
    c0 = 0
    do s = 1, 4
      if (.not. side_on(s)) cycle
      if (side_mine(s)) call put_c(s, c0)
      c0 = c0 + 2 * nline(s)
    enddo
    call comm_reduce_sum(ncbuf, hbuf, gbuf)
    if (comm_rank() /= 0) return
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', nline(SN))
    d_xiu = nc_def_dim(ncid, 'xi_u', nline(SN) - 1)
    d_eta = nc_def_dim(ncid, 'eta_rho', nline(SE))
    d_etav = nc_def_dim(ncid, 'eta_v', nline(SE) - 1)
    d_s = nc_def_dim(ncid, 's_rho', nz)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    do s = 1, 4
      if (.not. side_on(s)) cycle
      dl = merge(d_xi, d_eta, s == SN .or. s == SS)
      do v = 1, nv
        vn = var_name(v, s)
        id = nc_def_var(ncid, trim(vn), (/dl, d_s, d_t/))
        if (v == 1) then
          call nc_put_att_text(ncid, trim(vn), 'long_name', 'ALK additionality on '//trim(sadj(s))//' boundary')
          call nc_put_att_text(ncid, trim(vn), 'units', 'meq/m^3s-1')
        elseif (v == 2) then
          call nc_put_att_text(ncid, trim(vn), 'long_name', 'DIC additionality on '//trim(sadj(s))//' boundary')
          call nc_put_att_text(ncid, trim(vn), 'units', 'mmol/m^3s-1')
        elseif (v == nv) then
          call nc_put_att_text(ncid, trim(vn), 'long_name', 'Layer thickness on '//trim(sadj(s))//' boundary')
          call nc_put_att_text(ncid, trim(vn), 'units', 'meters')
        else
          call nc_put_att_text(ncid, trim(vn), 'long_name', &
               trim(passive_name(cdr0 + v - 2))//' additionality on '//trim(sadj(s))//' boundary')
          call nc_put_att_text(ncid, trim(vn), 'units', 'mMol m-3s-1')
        endif
      enddo
      vn = 'lat_'//trim(sname(s))
      id = nc_def_var(ncid, trim(vn), (/dl/))
      call nc_put_att_text(ncid, trim(vn), 'long_name', 'Latitudes of tracer points on '//trim(sadj(s))//' boundary')
      call nc_put_att_text(ncid, trim(vn), 'units', 'degrees')
      vn = 'lon_'//trim(sname(s))
      id = nc_def_var(ncid, trim(vn), (/dl/))
      call nc_put_att_text(ncid, trim(vn), 'long_name', 'Longitudes of tracer points on '//trim(sadj(s))//' boundary')
      call nc_put_att_text(ncid, trim(vn), 'units', 'degrees')
    enddo
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu upscaling output')
    call nc_enddef(ncid)
    c0 = 0
    do s = 1, 4
      if (.not. side_on(s)) cycle
      call nc_put_1d(ncid, 'lat_'//trim(sname(s)), gbuf(c0 + 1:c0 + nline(s)))
      call nc_put_1d(ncid, 'lon_'//trim(sname(s)), gbuf(c0 + nline(s) + 1:c0 + 2 * nline(s)))
      c0 = c0 + 2 * nline(s)
    enddo
    call nc_close(ncid)
    write(*,'(a,a)') 'created uscl file: ', trim(path)
  contains
    subroutine put_c(sd, off)
      ! lat/lon of the boundary rho points (ring row/column, corners on
      ! the corner tiles), as the CPU's latr(i0:i1, j1) etc.
      integer, intent(in) :: sd, off
      integer :: i, j, g, i0, i1, j0, j1
      i0 = merge(0, 1, decomp%edge_w); i1 = merge(int(decomp%nx) + 1, int(decomp%nx), decomp%edge_e)
      j0 = merge(0, 1, decomp%edge_s); j1 = merge(int(decomp%ny) + 1, int(decomp%ny), decomp%edge_n)
      if (sd == SN .or. sd == SS) then
        j = merge(int(decomp%ny) + 1, 0, sd == SN)
        do i = i0, i1
          g = int(decomp%i0) + i
          hbuf(off + g) = latr(i, j)
          hbuf(off + nline(sd) + g) = lonr(i, j)
        enddo
      else
        i = merge(int(decomp%nx) + 1, 0, sd == SE)
        do j = j0, j1
          g = int(decomp%j0) + j
          hbuf(off + g) = latr(i, j)
          hbuf(off + nline(sd) + g) = lonr(i, j)
        enddo
      endif
    end subroutine put_c
  end subroutine create_file

end module upscale_out

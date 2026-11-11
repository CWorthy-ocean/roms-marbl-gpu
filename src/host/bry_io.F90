#include "mem_attr.inc"
#ifdef USE_CUDA
! Rounded multiply: keeps c*b from contracting into an FMA on the device,
! so the device interpolation matches the (no-FMA) host arithmetic bit
! for bit.
#define BMUL(a, b) __dmul_rn(a, b)
#else
#define BMUL(a, b) ((a) * (b))
#endif
module bry_io
  ! Boundary forcing: time-interpolate zeta/u/v/T/S on open edges.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use frc_io, only: case_frcfiles, case_nfrc, cyc_bracket, frc_file_ncid, cached_times, frc_run_ncid_path
  use passive_mod, only: ntp, passive_bry, passive_name, tb_west, tb_east, tb_south, tb_north
  use nc_io
  use arch_mod, only: sync_arch
  use pflx_mod, only: ub_tune, pf_w, pf_e, pf_s, pf_n
#ifdef USE_CUDA
  use cudafor
  use cudadevice, only: __dmul_rn
#endif
  implicit none
  private

  ! Records are catalogued across every forcing file holding the
  ! variable (monthly boundary files), sorted by time.
  type :: series1d
    character(len=64)  :: vname = ''
    character(len=64)  :: tname = 'bry_time'
    integer :: n = 0, nrec = 0, it1 = 1, it2 = 2
    logical :: ready = .false., warned = .false.
    real(rp) :: cycle = -1.0_rp, t1e = 0.0_rp, t2e = 0.0_rp
    real(rp), allocatable :: times(:)
    character(len=512), allocatable :: paths(:)
    integer, allocatable :: irec(:)
    real(rp), WORK_ATTR allocatable :: buf(:,:)   ! (n,2), managed: read on the device
    integer :: f0 = 1   ! file start of this tile's sub-range along the edge
  end type series1d

  type :: series_z
    character(len=64)  :: vname = ''
    character(len=64)  :: tname = 'bry_time'
    integer :: n = 0, nz = 0, nrec = 0, it1 = 1, it2 = 2
    logical :: ready = .false., warned = .false.
    real(rp) :: cycle = -1.0_rp, t1e = 0.0_rp, t2e = 0.0_rp
    real(rp), allocatable :: times(:)
    character(len=512), allocatable :: paths(:)
    integer, allocatable :: irec(:)
    real(rp), WORK_ATTR allocatable :: buf(:,:,:) ! (n,nz,2), managed
    integer :: f0 = 1   ! file start of this tile's sub-range along the edge
  end type series_z

  type(series1d) :: ser_zw, ser_ubw, ser_vbw, ser_ze, ser_ube, ser_vbe
  type(series1d) :: ser_zs, ser_ubs, ser_vbs, ser_zn, ser_ubn, ser_vbn
  type(series_z) :: ser_uw, ser_vw, ser_tw, ser_sw, ser_ue, ser_ve, ser_te, ser_se
  type(series_z) :: ser_us, ser_vs, ser_ts, ser_ss, ser_un, ser_vn, ser_tn, ser_sn
  ! MARBL tracers (passive_mod MARBL mode): one series per tracer and side
  type(series_z), allocatable :: ser_tbw(:), ser_tbe(:), ser_tbs(:), ser_tbn(:)
  ! SPONGE_TUNE: the parent's baroclinic energy flux normal to each side
  ! (extracted by the parent as up_<side>/vp_<side>)
  type(series1d) :: ser_upw, ser_upe, ser_vps, ser_vpn

  public :: init_bry
  public :: set_bry_all
  public :: set_pflx_bry
  public :: dump_bry_north
  public :: write_toy_bry

contains

  subroutine init_bry()
    integer(ip) :: m, nmiss
    have_bry = .false.
    obc_west = .false.; obc_east = .false.; obc_south = .false.; obc_north = .false.
    if (case_nfrc < 1) return
    call bind1(ser_zw, 'zeta_west', 'j'); call bind1(ser_ubw, 'ubar_west', 'j'); call bind1(ser_vbw, 'vbar_west', 'j')
    call bind1(ser_ze, 'zeta_east', 'j'); call bind1(ser_ube, 'ubar_east', 'j'); call bind1(ser_vbe, 'vbar_east', 'j')
    call bind1(ser_zs, 'zeta_south', 'i'); call bind1(ser_ubs, 'ubar_south', 'i'); call bind1(ser_vbs, 'vbar_south', 'i')
    call bind1(ser_zn, 'zeta_north', 'i'); call bind1(ser_ubn, 'ubar_north', 'i'); call bind1(ser_vbn, 'vbar_north', 'i')
    call bindz(ser_uw, 'u_west', 'j'); call bindz(ser_vw, 'v_west', 'j')
    call bindz(ser_tw, 'temp_west', 'j'); call bindz(ser_sw, 'salt_west', 'j')
    call bindz(ser_ue, 'u_east', 'j'); call bindz(ser_ve, 'v_east', 'j')
    call bindz(ser_te, 'temp_east', 'j'); call bindz(ser_se, 'salt_east', 'j')
    call bindz(ser_us, 'u_south', 'i'); call bindz(ser_vs, 'v_south', 'i')
    call bindz(ser_ts, 'temp_south', 'i'); call bindz(ser_ss, 'salt_south', 'i')
    call bindz(ser_un, 'u_north', 'i'); call bindz(ser_vn, 'v_north', 'i')
    call bindz(ser_tn, 'temp_north', 'i'); call bindz(ser_sn, 'salt_north', 'i')
    obc_west  = ser_zw%ready .or. ser_ubw%ready
    obc_east  = ser_ze%ready .or. ser_ube%ready
    obc_south = ser_zs%ready .or. ser_vbs%ready
    obc_north = ser_zn%ready .or. ser_vbn%ready
    have_bry = obc_west .or. obc_east .or. obc_south .or. obc_north
    if (ub_tune) then
      call bind1(ser_upw, 'up_west', 'j'); call bind1(ser_upe, 'up_east', 'j')
      call bind1(ser_vps, 'vp_south', 'i'); call bind1(ser_vpn, 'vp_north', 'i')
      write(*,'(a,4l2)') 'sponge_tune: parent up_west, up_east, vp_south, vp_north found:', &
           ser_upw%ready, ser_upe%ready, ser_vps%ready, ser_vpn%ready
    endif
    if (have_bry) write(*,'(a,4l2)') 'OBC W E S N =', obc_west, obc_east, obc_south, obc_north
    if (have_bry .and. passive_bry .and. ntp > 0) then
      allocate(ser_tbw(ntp), ser_tbe(ntp), ser_tbs(ntp), ser_tbn(ntp))
      nmiss = 0
      do m = 1, ntp
        call bindz(ser_tbw(m), trim(passive_name(m))//'_west', 'j')
        call bindz(ser_tbe(m), trim(passive_name(m))//'_east', 'j')
        call bindz(ser_tbs(m), trim(passive_name(m))//'_south', 'i')
        call bindz(ser_tbn(m), trim(passive_name(m))//'_north', 'i')
        if (.not. (ser_tbw(m)%ready .or. ser_tbe(m)%ready .or. ser_tbs(m)%ready .or. ser_tbn(m)%ready)) &
             nmiss = nmiss + 1
      enddo
      write(*,'(a,i0,a,i0,a)') 'passive (MARBL, CDR_LITE) boundary data bound for ', ntp - nmiss, ' of ', ntp, &
           ' tracers (missing ones use 0 on inflow)'
    endif
  end subroutine init_bry

  subroutine set_bry_all()
    real(rp) :: tneed
    integer :: nz, is, js
    integer(ip) :: m
    if (.not. have_bry) return
    tneed = frc_modtime()
    call fill1j(ser_zw, tneed, zeta_west)
    call fill1j(ser_ubw, tneed, ubar_west)
    call fill1j(ser_vbw, tneed, vbar_west)
    call fill1j(ser_ze, tneed, zeta_east)
    call fill1j(ser_ube, tneed, ubar_east)
    call fill1j(ser_vbe, tneed, vbar_east)
    call fill1i(ser_zs, tneed, zeta_south)
    call fill1i(ser_ubs, tneed, ubar_south)
    call fill1i(ser_vbs, tneed, vbar_south)
    call fill1i(ser_zn, tneed, zeta_north)
    call fill1i(ser_ubn, tneed, ubar_north)
    call fill1i(ser_vbn, tneed, vbar_north)
    nz = decomp%nz
    is = decomp%is
    js = decomp%js
    call fillzj(ser_uw, tneed, u_west(js, 1), nz)
    call fillzj(ser_vw, tneed, v_west(js, 1), nz)
    call fillzj(ser_ue, tneed, u_east(js, 1), nz)
    call fillzj(ser_ve, tneed, v_east(js, 1), nz)
    call fillzi(ser_us, tneed, u_south(is, 1), nz)
    call fillzi(ser_vs, tneed, v_south(is, 1), nz)
    call fillzi(ser_un, tneed, u_north(is, 1), nz)
    call fillzi(ser_vn, tneed, v_north(is, 1), nz)
    call fillzj(ser_tw, tneed, t_west(js, 1, itemp), nz)
    call fillzj(ser_sw, tneed, t_west(js, 1, isalt), nz)
    call fillzj(ser_te, tneed, t_east(js, 1, itemp), nz)
    call fillzj(ser_se, tneed, t_east(js, 1, isalt), nz)
    call fillzi(ser_ts, tneed, t_south(is, 1, itemp), nz)
    call fillzi(ser_ss, tneed, t_south(is, 1, isalt), nz)
    call fillzi(ser_tn, tneed, t_north(is, 1, itemp), nz)
    call fillzi(ser_sn, tneed, t_north(is, 1, isalt), nz)
    if (passive_bry .and. allocated(ser_tbw)) then
      do m = 1, ntp
        call fillzj(ser_tbw(m), tneed, tb_west(js, 1, m), nz)
        call fillzj(ser_tbe(m), tneed, tb_east(js, 1, m), nz)
        call fillzi(ser_tbs(m), tneed, tb_south(is, 1, m), nz)
        call fillzi(ser_tbn(m), tneed, tb_north(is, 1, m), nz)
      enddo
    endif
    call dump_bry_north('after clim', tneed)
  end subroutine set_bry_all

  subroutine set_pflx_bry(tneed)
    ! The CPU's set_pflx: the parent fluxes at time tneed (days), on the
    ! device into pf_<side>; a side without data keeps 0 (the CPU's missing
    ! forcing).
    real(rp), intent(in) :: tneed
    call fill1j(ser_upw, tneed, pf_w)
    call fill1j(ser_upe, tneed, pf_e)
    call fill1i(ser_vps, tneed, pf_s)
    call fill1i(ser_vpn, tneed, pf_n)
  end subroutine set_pflx_bry

  subroutine tile_window(axis, n, f0)
    ! An edge series of global length gn+2 / gn+1 / gn (rho with boundary
    ! points, u/v, interior rho) is read as this tile's n+2 / n+1 / n points
    ! starting at the tile's global offset; any other length is left whole.
    character(len=*), intent(in)    :: axis
    integer,          intent(inout) :: n
    integer,          intent(out)   :: f0
    integer :: g, l, o
    if (axis == 'j') then
      g = int(decomp%gny); l = int(decomp%ny); o = int(decomp%j0)
    else
      g = int(decomp%gnx); l = int(decomp%nx); o = int(decomp%i0)
    endif
    f0 = 1
    if (n == g + 2 .or. n == g + 1 .or. n == g) then
      n = n - g + l
      f0 = o
    endif
  end subroutine tile_window

  subroutine bind1(s, vname, axis)
    type(series1d),   intent(inout) :: s
    character(len=*), intent(in)    :: vname, axis
    integer :: n1, n2
    s%vname = trim(vname)
    call build_catalog(vname, s%tname, s%times, s%paths, s%irec, s%nrec, n1, n2, s%cycle)
    s%ready = (s%nrec > 0)
    if (.not. s%ready) return
    s%n = n1
    call tile_window(axis, s%n, s%f0)
    allocate(s%buf(s%n, 2))
    if (s%nrec >= 2) then
      ! No bracket yet: the first adv1 (the first set_bry_all) reads it.
      s%it1 = 0; s%it2 = 0
    else
      s%it1 = 1; s%it2 = 1
      call read_rec1(s, s%it1, 1)
      call read_rec1(s, s%it2, 2)
    endif
  end subroutine bind1

  subroutine bindz(s, vname, axis)
    type(series_z),   intent(inout) :: s
    character(len=*), intent(in)    :: vname, axis
    integer :: n1, n2
    s%vname = trim(vname)
    call build_catalog(vname, s%tname, s%times, s%paths, s%irec, s%nrec, n1, n2, s%cycle)
    s%ready = (s%nrec > 0)
    if (.not. s%ready) return
    s%n = n1; s%nz = n2
    call tile_window(axis, s%n, s%f0)
    allocate(s%buf(s%n, n2, 2))
    if (s%nrec >= 2) then
      ! No bracket yet: the first advz (the first set_bry_all) reads it.
      s%it1 = 0; s%it2 = 0
    else
      s%it1 = 1; s%it2 = 1
      call read_recz(s, s%it1, 1)
      call read_recz(s, s%it2, 2)
    endif
  end subroutine bindz

  subroutine build_catalog(vname, tname, times, paths, irec, ntot, n1, n2, cycle)
    character(len=*), intent(in) :: vname, tname
    real(rp), allocatable, intent(inout) :: times(:)
    character(len=512), allocatable, intent(inout) :: paths(:)
    integer, allocatable, intent(inout) :: irec(:)
    integer, intent(out) :: ntot, n1, n2
    real(rp), intent(out) :: cycle
    integer :: pass, i, k, j, ncid, a1, a2, a3, nrec, isw
    real(rp), allocatable :: tloc(:)
    real(rp) :: tsw, cyc
    character(len=512) :: psw
    logical :: lcyc

    ntot = 0; n1 = 0; n2 = 0; cycle = -1.0_rp
    do pass = 1, 2
      if (pass == 2) then
        if (ntot < 1) return
        allocate(times(ntot), paths(ntot), irec(ntot))
        ntot = 0
      endif
      do i = 1, case_nfrc
        ncid = frc_file_ncid(i)   ! cached handle, closed by close_frc_cache
        if (nc_has_var(ncid, vname)) then
          call nc_var_shape(ncid, vname, a1, a2, a3, nrec)
          if (nc_has_var(ncid, tname)) nrec = nc_var_nrecs(ncid, tname)
          nrec = max(1, nrec)
          if (pass == 1) then
            if (n1 == 0) then
              n1 = a1; n2 = a2
              if (nc_has_var(ncid, tname)) then
                call nc_get_att_real(ncid, tname, 'cycle_length', cyc, lcyc)
                if (lcyc .and. cyc > 0.0_rp) cycle = cyc
              endif
            endif
            ntot = ntot + nrec
          else
            allocate(tloc(nrec))
            tloc = 0.0_rp
            if (nc_has_var(ncid, tname)) then
              if (.not. cached_times(i, tname, tloc)) call nc_get_1d(ncid, tname, tloc)
            endif
            do k = 1, nrec
              ntot = ntot + 1
              times(ntot) = tloc(k)
              paths(ntot) = case_frcfiles(i)
              irec(ntot) = k
            enddo
            deallocate(tloc)
          endif
        endif
      enddo
    enddo

    do i = 2, ntot
      tsw = times(i); psw = paths(i); isw = irec(i)
      j = i - 1
      do while (j >= 1)
        if (times(j) <= tsw) exit
        times(j + 1) = times(j); paths(j + 1) = paths(j); irec(j + 1) = irec(j)
        j = j - 1
      enddo
      times(j + 1) = tsw; paths(j + 1) = psw; irec(j + 1) = isw
    enddo
  end subroutine build_catalog

  subroutine copy_upper1(s)
    ! slot 1 = slot 2 on the device (the record already read).
    type(series1d), intent(inout) :: s
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaMemcpy(s%buf(:,1), s%buf(:,2), s%n, cudaMemcpyDeviceToDevice)
    if (ierr /= 0) error stop 'copy_upper1: cudaMemcpy'
#else
    s%buf(:,1) = s%buf(:,2)
#endif
  end subroutine copy_upper1

  subroutine copy_upperz(s)
    ! slot 1 = slot 2 on the device (the record already read).
    type(series_z), intent(inout) :: s
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaMemcpy(s%buf(:,:,1), s%buf(:,:,2), s%n * s%nz, cudaMemcpyDeviceToDevice)
    if (ierr /= 0) error stop 'copy_upperz: cudaMemcpy'
#else
    s%buf(:,:,1) = s%buf(:,:,2)
#endif
  end subroutine copy_upperz

  subroutine read_rec1(s, idx, slot)
    type(series1d), intent(inout) :: s
    integer,        intent(in)    :: idx, slot
    integer :: ncid
    real(rp), allocatable :: tmp(:,:)
#ifdef USE_CUDA
    integer :: ierr
#endif
    allocate(tmp(s%n, 1))
    ncid = frc_run_ncid_path(s%paths(idx))   ! kept open for the run
    if (ncid < 0) then
      call nc_open_read(s%paths(idx), ncid)
      call nc_read_2d(ncid, s%vname, tmp, s%n, 1, s%irec(idx), s%f0, 1)
      call nc_close(ncid)
    else
      call nc_read_2d(ncid, s%vname, tmp, s%n, 1, s%irec(idx), s%f0, 1)
    endif
#ifdef USE_CUDA
    ierr = cudaMemcpy(s%buf(:,slot), tmp(:,1), s%n, cudaMemcpyHostToDevice)
    if (ierr /= 0) error stop 'read_rec1: cudaMemcpy'
#else
    s%buf(:,slot) = tmp(:,1)
#endif
    deallocate(tmp)
  end subroutine read_rec1

  subroutine read_recz(s, idx, slot)
    type(series_z), intent(inout) :: s
    integer,        intent(in)    :: idx, slot
    integer :: ncid
    real(rp), allocatable :: tmp(:,:)
#ifdef USE_CUDA
    integer :: ierr
#endif
    allocate(tmp(s%n, s%nz))
    ncid = frc_run_ncid_path(s%paths(idx))   ! kept open for the run
    if (ncid < 0) then
      call nc_open_read(s%paths(idx), ncid)
      call nc_read_2d(ncid, s%vname, tmp, s%n, s%nz, s%irec(idx), s%f0, 1)
      call nc_close(ncid)
    else
      call nc_read_2d(ncid, s%vname, tmp, s%n, s%nz, s%irec(idx), s%f0, 1)
    endif
#ifdef USE_CUDA
    ierr = cudaMemcpy(s%buf(:,:,slot), tmp, s%n * s%nz, cudaMemcpyHostToDevice)
    if (ierr /= 0) error stop 'read_recz: cudaMemcpy'
#else
    s%buf(:,:,slot) = tmp
#endif
    deallocate(tmp)
  end subroutine read_recz

  ! Time interpolation runs on the device (the two records live in
  ! managed buffers), writing exactly the index range the host version
  ! wrote. Only the weights are computed on the host.
  subroutine fill1j(s, tneed, dest)
    type(series1d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%js:decomp%je)
    real(rp) :: c1, c2
    integer :: lo, cnt
    if (.not. s%ready) return
    call adv1(s, tneed)
    call wgt1(s, tneed, c1, c2)
    if (s%n == decomp%ny + 2) then
      lo = 0; cnt = s%n
    elseif (s%n == decomp%ny + 1) then
      lo = 1; cnt = s%n
    elseif (s%n == decomp%ny) then
      lo = 1; cnt = s%n
    else
      lo = decomp%js; cnt = min(s%n, decomp%je - decomp%js + 1)
    endif
    call interp1_dev(s%buf, s%n, c1, c2, dest, decomp%js, decomp%je, lo, cnt)
  end subroutine fill1j

  subroutine fill1i(s, tneed, dest)
    type(series1d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie)
    real(rp) :: c1, c2
    integer :: lo, cnt
    if (.not. s%ready) return
    call adv1(s, tneed)
    call wgt1(s, tneed, c1, c2)
    if (s%n == decomp%nx + 2) then
      lo = 0; cnt = s%n
    elseif (s%n == decomp%nx + 1) then
      lo = 1; cnt = s%n
    elseif (s%n == decomp%nx) then
      lo = 1; cnt = s%n
    else
      lo = decomp%is; cnt = min(s%n, decomp%ie - decomp%is + 1)
    endif
    call interp1_dev(s%buf, s%n, c1, c2, dest, decomp%is, decomp%ie, lo, cnt)
  end subroutine fill1i

  subroutine interp1_dev(b, n, c1, c2, dest, d0, d1, lo, cnt)
    integer,  intent(in) :: n, d0, d1, lo, cnt
    real(rp), WORK_ATTR intent(in) :: b(n, 2)
    real(rp), intent(in) :: c1, c2
    real(rp), WORK_ATTR intent(inout) :: dest(d0:d1)
    integer :: i
    real(rp) :: x
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*>>>
#endif
    do i = 1, cnt
      x = BMUL(c1, b(i, 1)) + BMUL(c2, b(i, 2))
      if (x /= x) x = 0.0_rp
      dest(lo + i - 1) = x
    enddo
  end subroutine interp1_dev

  subroutine fillzj(s, tneed, dest, nzd)
    ! dest is the first element of a (js:je, nzd) slab.
    type(series_z), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    integer,        intent(in)    :: nzd
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%js:decomp%je, nzd)
    real(rp) :: c1, c2
    integer :: nz, lo, cnt
    if (.not. s%ready) return
    call advz(s, tneed)
    call wgtz(s, tneed, c1, c2)
    nz = min(nzd, s%nz)
    if (s%n == decomp%ny + 2) then
      lo = 0; cnt = s%n
    elseif (s%n == decomp%ny + 1) then
      lo = 1; cnt = s%n
    elseif (s%n == decomp%ny) then
      lo = 1; cnt = s%n
    else
      lo = decomp%js; cnt = 0
    endif
    call interpz_dev(s%buf, s%n, s%nz, c1, c2, dest, decomp%js, decomp%je, nzd, nz, lo, cnt)
  end subroutine fillzj

  subroutine fillzi(s, tneed, dest, nzd)
    type(series_z), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    integer,        intent(in)    :: nzd
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, nzd)
    real(rp) :: c1, c2
    integer :: nz, lo, cnt
    if (.not. s%ready) return
    call advz(s, tneed)
    call wgtz(s, tneed, c1, c2)
    nz = min(nzd, s%nz)
    if (s%n == decomp%nx + 2) then
      lo = 0; cnt = s%n
    elseif (s%n == decomp%nx + 1) then
      lo = 1; cnt = s%n
    elseif (s%n == decomp%nx) then
      lo = 1; cnt = s%n
    else
      lo = decomp%is; cnt = 0
    endif
    call interpz_dev(s%buf, s%n, s%nz, c1, c2, dest, decomp%is, decomp%ie, nzd, nz, lo, cnt)
  end subroutine fillzi

  subroutine interpz_dev(b, n, nzs, c1, c2, dest, d0, d1, nzd, nz, lo, cnt)
    ! dest(:, 1:nz) = 0, then the lo:lo+cnt-1 range gets the interpolant
    ! (NaN -> 0), as the host loops did.
    integer,  intent(in) :: n, nzs, d0, d1, nzd, nz, lo, cnt
    real(rp), WORK_ATTR intent(in) :: b(n, nzs, 2)
    real(rp), intent(in) :: c1, c2
    real(rp), WORK_ATTR intent(inout) :: dest(d0:d1, nzd)
    integer :: j, k
    real(rp) :: x
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do k = 1, nz
      do j = d0, d1
        x = 0.0_rp
        if (j >= lo .and. j <= lo + cnt - 1) then
          x = BMUL(c1, b(j - lo + 1, k, 1)) + BMUL(c2, b(j - lo + 1, k, 2))
          if (x /= x) x = 0.0_rp
        endif
        dest(j, k) = x
      enddo
    enddo
  end subroutine interpz_dev

  subroutine adv1(s, tneed)
    type(series1d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    integer :: rec, i, i1, i2, old2
    if (s%nrec < 2) return
    if (s%cycle > 0.0_rp) then
      call cyc_bracket(s%times, s%nrec, s%cycle, tneed, i1, i2, s%t1e, s%t2e)
      if (i1 /= s%it1) then
        old2 = s%it2
        s%it1 = i1
        if (i1 == old2) then
          call copy_upper1(s)
        else
          call read_rec1(s, i1, 1)
        endif
      endif
      if (i2 /= s%it2) then
        s%it2 = i2
        call read_rec1(s, i2, 2)
      endif
      return
    endif
    if (s%it1 >= 1) then
      if (s%times(s%it2) >= tneed .and. s%times(s%it1) <= tneed) return
    endif
    rec = 1
    do i = 1, s%nrec - 1
      if (s%times(i) <= tneed) rec = i
      if (s%times(i) <= tneed .and. s%times(i + 1) >= tneed) then
        rec = i; exit
      endif
    enddo
    rec = max(1, min(rec, s%nrec - 1))
    if (rec == s%it1 .and. rec + 1 == s%it2) return
    old2 = s%it2
    s%it1 = rec; s%it2 = rec + 1
    ! moving on by one record: the old upper bracket becomes the lower one
    if (s%it1 == old2) then
      call copy_upper1(s)
    else
      call read_rec1(s, s%it1, 1)
    endif
    call read_rec1(s, s%it2, 2)
  end subroutine adv1

  subroutine advz(s, tneed)
    type(series_z), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    integer :: rec, i, i1, i2, old2
    if (s%nrec < 2) return
    if (s%cycle > 0.0_rp) then
      call cyc_bracket(s%times, s%nrec, s%cycle, tneed, i1, i2, s%t1e, s%t2e)
      if (i1 /= s%it1) then
        old2 = s%it2
        s%it1 = i1
        if (i1 == old2) then
          call copy_upperz(s)
        else
          call read_recz(s, i1, 1)
        endif
      endif
      if (i2 /= s%it2) then
        s%it2 = i2
        call read_recz(s, i2, 2)
      endif
      return
    endif
    if (s%it1 >= 1) then
      if (s%times(s%it2) >= tneed .and. s%times(s%it1) <= tneed) return
    endif
    rec = 1
    do i = 1, s%nrec - 1
      if (s%times(i) <= tneed) rec = i
      if (s%times(i) <= tneed .and. s%times(i + 1) >= tneed) then
        rec = i; exit
      endif
    enddo
    rec = max(1, min(rec, s%nrec - 1))
    if (rec == s%it1 .and. rec + 1 == s%it2) return
    old2 = s%it2
    s%it1 = rec; s%it2 = rec + 1
    ! moving on by one record: the old upper bracket becomes the lower one
    if (s%it1 == old2) then
      call copy_upperz(s)
    else
      call read_recz(s, s%it1, 1)
    endif
    call read_recz(s, s%it2, 2)
  end subroutine advz

  subroutine wgt1(s, tneed, c1, c2)
    type(series1d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp),       intent(out)   :: c1, c2
    if (s%cycle > 0.0_rp) then
      call time_weights(s%nrec, s%t1e, s%t2e, tneed, s%vname, s%warned, c1, c2)
    else
      call time_weights(s%nrec, s%times(s%it1), s%times(s%it2), tneed, s%vname, s%warned, c1, c2)
    endif
  end subroutine wgt1

  subroutine wgtz(s, tneed, c1, c2)
    type(series_z), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp),       intent(out)   :: c1, c2
    if (s%cycle > 0.0_rp) then
      call time_weights(s%nrec, s%t1e, s%t2e, tneed, s%vname, s%warned, c1, c2)
    else
      call time_weights(s%nrec, s%times(s%it1), s%times(s%it2), tneed, s%vname, s%warned, c1, c2)
    endif
  end subroutine wgtz

  subroutine time_weights(nrec, t1, t2, tneed, vname, warned, c1, c2)
    ! Linear in [t1,t2]; held at the nearest record outside the data.
    integer,          intent(in)    :: nrec
    real(rp),         intent(in)    :: t1, t2, tneed
    character(len=*), intent(in)    :: vname
    logical,          intent(inout) :: warned
    real(rp),         intent(out)   :: c1, c2
    real(rp) :: dtm
    c1 = 1.0_rp; c2 = 0.0_rp
    if (nrec < 2) return
    dtm = t2 - t1
    if (abs(dtm) < 1.0e-12_rp) return
    c2 = min(1.0_rp, max(0.0_rp, (tneed - t1) / dtm))
    c1 = 1.0_rp - c2
    if ((tneed < t1 .or. tneed > t2) .and. .not. warned) then
      write(*,'(a,a,a,es14.6,a,2es14.6)') 'WARNING bry: ', trim(vname), ' time', tneed, &
           ' outside data', t1, t2
      warned = .true.
    endif
  end subroutine time_weights

  subroutine dump_bry_north(tag, tneed)
    ! Clim interpolation of zeta_north / vbar_north at P1, before tides.
    character(len=*), intent(in) :: tag
    real(rp),         intent(in) :: tneed
    integer(ip) :: i, n, it1, it2, k
    real(rp) :: c1, c2, t1, t2, zb1, zb2, vb1, vb2

    if (.not. probe_on .and. iic /= ntstart) return
    call sync_arch()
    i = iprb_pt(1)
    n = ser_zn%n
    if (i < decomp%is .or. i > decomp%ie) return
    t1 = 0.0_rp; t2 = 0.0_rp; c1 = 1.0_rp; c2 = 0.0_rp
    zb1 = 0.0_rp; zb2 = 0.0_rp; vb1 = 0.0_rp; vb2 = 0.0_rp
    it1 = 1; it2 = 2
    if (ser_zn%ready) then
      it1 = ser_zn%it1; it2 = ser_zn%it2
      t1 = ser_zn%times(it1); t2 = ser_zn%times(it2)
      call wgt1(ser_zn, tneed, c1, c2)
      if (n == decomp%nx + 2 .and. i + 1 >= 1 .and. i + 1 <= n) then
        zb1 = ser_zn%buf(i + 1, 1); zb2 = ser_zn%buf(i + 1, 2)
      elseif (i >= 1 .and. i <= n) then
        zb1 = ser_zn%buf(i, 1); zb2 = ser_zn%buf(i, 2)
      endif
    endif
    if (ser_vbn%ready) then
      if (ser_vbn%n == decomp%nx + 2 .and. i + 1 >= 1 .and. i + 1 <= ser_vbn%n) then
        vb1 = ser_vbn%buf(i + 1, 1); vb2 = ser_vbn%buf(i + 1, 2)
      elseif (i >= 1 .and. i <= ser_vbn%n) then
        vb1 = ser_vbn%buf(i, 1); vb2 = ser_vbn%buf(i, 2)
      endif
    endif
    write(*,'(a,i1,a,a,a,i8,a,2i6,a,11es26.17)') 'diag: bryN P', 1, '  ', trim(tag), &
         '  iic=', iic, '  nmap,i=', n, i, '  tneed,t1,t2,c1,c2,zB1,zB2,vB1,vB2,zBry,vBry=', &
         tneed, t1, t2, c1, c2, zb1, zb2, vb1, vb2, zeta_north(i), vbar_north(i)
    do k = 1, decomp%nz
      if (k == 1 .or. k == 50 .or. k == 99 .or. k == 100) then
        write(*,'(a,i1,a,a,a,i8,a,i4,a,es26.17)') 'diag: v3bry N P', 1, '  ', trim(tag), &
             '  iic=', iic, '  k=', k, '  vN=', v_north(i, k)
      endif
    enddo
    flush(6)
  end subroutine dump_bry_north

  subroutine write_toy_bry(path, nxi, neta, nz, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta, nz
    real(rp),         intent(in) :: t0, t1
    integer :: ncid, d_xi, d_eta, d_s, d_t, id
    real(rp), allocatable :: z2(:,:), u2(:,:), tz(:,:,:)
    allocate(z2(neta, 2), u2(nxi, 2), tz(neta, nz, 2))
    z2 = 0.0_rp
    u2 = 0.0_rp
    tz = 15.1_rp
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'bry_time', 2)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_s = nc_def_dim(ncid, 's_rho', nz)
    id = nc_def_var(ncid, 'bry_time', (/d_t/))
    id = nc_def_var(ncid, 'zeta_west', (/d_eta, d_t/))
    id = nc_def_var(ncid, 'zeta_east', (/d_eta, d_t/))
    id = nc_def_var(ncid, 'zeta_south', (/d_xi, d_t/))
    id = nc_def_var(ncid, 'zeta_north', (/d_xi, d_t/))
    id = nc_def_var(ncid, 'ubar_west', (/d_eta, d_t/))
    id = nc_def_var(ncid, 'ubar_east', (/d_eta, d_t/))
    id = nc_def_var(ncid, 'ubar_south', (/d_xi, d_t/))
    id = nc_def_var(ncid, 'ubar_north', (/d_xi, d_t/))
    id = nc_def_var(ncid, 'vbar_west', (/d_eta, d_t/))
    id = nc_def_var(ncid, 'vbar_east', (/d_eta, d_t/))
    id = nc_def_var(ncid, 'vbar_south', (/d_xi, d_t/))
    id = nc_def_var(ncid, 'vbar_north', (/d_xi, d_t/))
    id = nc_def_var(ncid, 'u_west', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'u_east', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'u_south', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'u_north', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'v_west', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'v_east', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'v_south', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'v_north', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'temp_west', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'temp_east', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'temp_south', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'temp_north', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'salt_west', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'salt_east', (/d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'salt_south', (/d_xi, d_s, d_t/))
    id = nc_def_var(ncid, 'salt_north', (/d_xi, d_s, d_t/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'bry_time', (/ t0, t1 /))
    z2 = 0.02_rp
    call nc_put_2d(ncid, 'zeta_west', z2)
    call nc_put_2d(ncid, 'zeta_east', z2)
    call nc_put_2d(ncid, 'zeta_south', u2)
    call nc_put_2d(ncid, 'zeta_north', u2)
    z2 = 0.05_rp
    call nc_put_2d(ncid, 'ubar_west', z2)
    call nc_put_2d(ncid, 'ubar_east', z2)
    call nc_put_2d(ncid, 'ubar_south', u2)
    call nc_put_2d(ncid, 'ubar_north', u2)
    z2 = 0.0_rp
    call nc_put_2d(ncid, 'vbar_west', z2)
    call nc_put_2d(ncid, 'vbar_east', z2)
    call nc_put_2d(ncid, 'vbar_south', u2)
    call nc_put_2d(ncid, 'vbar_north', u2)
    tz = 0.05_rp
    call nc_put_3d(ncid, 'u_west', tz)
    call nc_put_3d(ncid, 'u_east', tz)
    deallocate(tz)
    allocate(tz(nxi, nz, 2)); tz = 0.05_rp
    call nc_put_3d(ncid, 'u_south', tz)
    call nc_put_3d(ncid, 'u_north', tz)
    deallocate(tz)
    allocate(tz(neta, nz, 2)); tz = 0.0_rp
    call nc_put_3d(ncid, 'v_west', tz)
    call nc_put_3d(ncid, 'v_east', tz)
    deallocate(tz)
    allocate(tz(nxi, nz, 2)); tz = 0.0_rp
    call nc_put_3d(ncid, 'v_south', tz)
    call nc_put_3d(ncid, 'v_north', tz)
    deallocate(tz)
    allocate(tz(neta, nz, 2)); tz = 15.1_rp
    call nc_put_3d(ncid, 'temp_west', tz)
    call nc_put_3d(ncid, 'temp_east', tz)
    tz = 35.0_rp
    call nc_put_3d(ncid, 'salt_west', tz)
    call nc_put_3d(ncid, 'salt_east', tz)
    deallocate(tz)
    allocate(tz(nxi, nz, 2)); tz = 15.1_rp
    call nc_put_3d(ncid, 'temp_south', tz)
    call nc_put_3d(ncid, 'temp_north', tz)
    tz = 35.0_rp
    call nc_put_3d(ncid, 'salt_south', tz)
    call nc_put_3d(ncid, 'salt_north', tz)
    call nc_close(ncid)
    deallocate(z2, u2, tz)
  end subroutine write_toy_bry

end module bry_io

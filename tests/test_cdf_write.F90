program test_cdf_write
  ! cdf_direct's direct writes, read back through nf90_get_var bit for bit:
  ! a CDF5 file defined by netCDF under <file>.hdr (no fill, 4-KiB aligned
  ! sections), its header copied with cdf_copy_header, then 3 records
  ! written by 1-4 simulated writers the way io_server writes them
  ! (cdf_plan's page ownership, heads of shared pages handed to the page's
  ! writer, read-modify-write of partly covered pages, cdf_pack_*,
  ! cdf_write, cdf_set_numrecs, cdf_close_w, cdf_truncate). Variables:
  ! float, double and int; real(4) sources into double variables and
  ! real(8) sources into float ones (netCDF's conversions); fixed-size
  ! variables written with record 2 only; a record variable never written
  ! (reads 0: no fill); odd lengths (no length a multiple of 4 KiB); a
  ! geometry of many small variables per page (chains of heads); a header
  ! longer than the first 256-KiB read; O_DIRECT and buffered descriptors.
  ! Checks: values, format (CDF5), record count, exact file length, the
  ! scratch header removed, the Lustre layout against `lfs getstripe`.
  ! Also times cdf_pack_r4 / cdf_pack_r8 and one writer's cdf_write.
  ! usage: test_cdf_write <scratch directory on Lustre>   (exit status 1 on a failure)
  use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use netcdf
  use cdf_direct
  implicit none
  character(len=512) :: dir
  integer :: nfail = 0, ncheck = 0
  integer, parameter :: NV = 14, NREC = 3
  ! variable table: name, source kind (4 real(4), 8 real(8), 1 integer), file type, rank, record?, when written
  ! (0 every record, 2 with record 2 only, -1 never)
  character(len=16) :: vname(NV)
  integer :: vsrc(NV), vtype(NV), vrank(NV), vwhen(NV)
  logical :: visrec(NV)
  integer :: nx, ny, nz
  integer, parameter :: NA = 6, NS = 3
  integer :: nadd = 0

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) dir = '.'
  call set_table()
  call run_case('w_d1.nc', 37, 23, 5, 1, .true., 1, .false.)
  call run_case('w_d2.nc', 37, 23, 5, 2, .true., 1, .false.)
  call run_case('w_d3.nc', 37, 23, 5, 3, .true., 1, .false.)
  call run_case('w_d4.nc', 37, 23, 5, 4, .true., 1, .false.)
  call run_case('w_d4b.nc', 37, 23, 5, 4, .true., 2, .false.)
  call run_case('w_b3.nc', 37, 23, 5, 3, .false., 1, .false.)
  call run_case('w_small4.nc', 5, 3, 2, 4, .true., 1, .false.)
  call run_case('w_small3.nc', 3, 2, 1, 3, .true., 2, .false.)
  call run_case('w_bighdr.nc', 37, 23, 5, 3, .true., 1, .true.)
  call run_case('w_big2.nc', 301, 207, 11, 4, .true., 1, .false.)
  call speed()
  write(*,'(a,i0,a,i0)') 'test_cdf_write: checks ', ncheck, '  failures ', nfail
  if (nfail > 0) then
    write(*,'(a)') 'test_cdf_write: FAIL'
    error stop 1
  endif
  write(*,'(a)') 'test_cdf_write: PASS'

contains

  subroutine chk(ok, what)
    logical, intent(in) :: ok
    character(len=*), intent(in) :: what
    ncheck = ncheck + 1
    if (.not. ok) then
      nfail = nfail + 1
      write(*,'(2a)') 'FAIL ', what
    endif
  end subroutine chk

  subroutine nc(ierr, what)
    integer, intent(in) :: ierr
    character(len=*), intent(in) :: what
    if (ierr /= nf90_noerr) then
      write(*,'(4a)') 'netCDF error in ', what, ': ', trim(nf90_strerror(ierr))
      error stop 2
    endif
  end subroutine nc

  subroutine set_table()
    nadd = 0
    call addv('ocean_time', 8, nf90_double, 0, .true., 0)
    call addv('time_step', 1, nf90_int, 1, .true., 0)
    call addv('f3', 4, nf90_float, 3, .true., 0)
    call addv('x3', 8, nf90_double, 3, .false., 2)     ! fixed-size, written with record 2
    call addv('d2', 8, nf90_double, 2, .true., 0)
    call addv('s1', 8, nf90_double, 1, .true., 0)
    call addv('i2', 1, nf90_int, 2, .true., 0)
    call addv('d3', 8, nf90_double, 3, .true., 0)
    call addv('y3', 8, nf90_float, 3, .false., 2)      ! fixed-size float from real(8)
    call addv('g2', 8, nf90_float, 2, .true., 0)       ! real(8) -> float
    call addv('h3', 4, nf90_double, 3, .true., 0)      ! real(4) -> double
    call addv('skip', 4, nf90_float, 2, .true., -1)    ! never written: reads 0
    call addv('e2', 4, nf90_float, 2, .true., 0)
    call addv('z1', 8, nf90_double, 1, .false., 2)
  end subroutine set_table

  subroutine addv(n, s, t, r, isr, w)
    character(len=*), intent(in) :: n
    integer, intent(in) :: s, t, r, w
    logical, intent(in) :: isr
    nadd = nadd + 1
    vname(nadd) = n; vsrc(nadd) = s; vtype(nadd) = t; vrank(nadd) = r; visrec(nadd) = isr; vwhen(nadd) = w
  end subroutine addv

  integer(int64) function nelem(iv)
    integer, intent(in) :: iv
    select case (vrank(iv))
    case (0); nelem = 1
    case (1)
      nelem = NS
      if (vname(iv) == 'time_step') nelem = NA
    case (2); nelem = int(nx, int64) * ny
    case default; nelem = int(nx, int64) * ny * nz
    end select
  end function nelem

  real(real64) function gval(iv, rec, e)
    ! deterministic values with NaN, -0, subnormals and extremes for the real(8) sources
    integer, intent(in) :: iv, rec
    integer(int64), intent(in) :: e
    integer :: m
    m = int(mod(e, 97_int64))
    gval = sin(0.37d0 * real(e, 8) + 1.1d0 * iv) * 10.0d0**(mod(e, 7_int64) - 3) + rec
    if (vname(iv) == 'ocean_time') gval = 86400.0d0 * rec + 0.125d0
    if (m == 5) gval = ieee_value(1.0d0, ieee_quiet_nan)
    if (m == 6) gval = -0.0d0
    if (vtype(iv) == nf90_double) then
      if (m == 7) gval = 1.0d-310
      if (m == 8) gval = -huge(1.0d0) * 0.75d0
    else
      if (m == 7) gval = 1.0d-30
      if (m == 8) gval = -3.0d38
      if (m == 9) gval = 1.0d-300            ! rounds to 0 in float
    endif
  end function gval

  real(real32) function gval4(iv, rec, e)
    integer, intent(in) :: iv, rec
    integer(int64), intent(in) :: e
    integer :: m
    m = int(mod(e, 89_int64))
    gval4 = real(sin(0.29d0 * real(e, 8) + 0.7d0 * iv) * 10.0d0**(mod(e, 5_int64) - 2) + rec, real32)
    if (m == 3) gval4 = ieee_value(1.0, ieee_quiet_nan)
    if (m == 4) gval4 = -0.0
    ! subnormals only where the bits are copied: a conversion of one follows the process's flush-to-zero
    ! setting (nvfortran's default), in the expectation as in cdf_pack, and was seen to differ between the two
    if (m == 5) gval4 = merge(1.0e-41, 1.0e-30, vtype(iv) == nf90_float)
    if (m == 6) gval4 = huge(1.0)
  end function gval4

  integer function gint(iv, rec, e)
    integer, intent(in) :: iv, rec
    integer(int64), intent(in) :: e
    gint = int(mod(e * 7919_int64 + iv * 131 + rec, 200001_int64)) - 100000
    if (mod(e, 11_int64) == 3) gint = -huge(1)
    if (vname(iv) == 'time_step') gint = merge(rec * 10, rec, e == 0)
  end function gint

  subroutine pack_var(iv, rec, ftype, b, boff)
    ! the whole variable iv of record rec, as a writer packs it, into b at byte boff
    integer, intent(in) :: iv, rec, ftype
    type(cdf_buf_t), intent(inout) :: b
    integer(int64), intent(in) :: boff
    integer(int64) :: n, e
    real(real64), allocatable :: a8(:)
    real(real32), allocatable :: a4(:)
    integer(int32), allocatable :: ai(:)
    n = nelem(iv)
    select case (vsrc(iv))
    case (8)
      allocate(a8(n))
      do e = 1, n
        a8(e) = gval(iv, rec, e - 1)
      enddo
      call cdf_pack_r8(a8, n, ftype, b, boff)
    case (4)
      allocate(a4(n))
      do e = 1, n
        a4(e) = gval4(iv, rec, e - 1)
      enddo
      call cdf_pack_r4(a4, n, ftype, b, boff)
    case default
      allocate(ai(n))
      do e = 1, n
        ai(e) = gint(iv, rec, e - 1)
      enddo
      call cdf_pack_i4(ai, n, b, boff)
    end select
  end subroutine pack_var

  subroutine run_case(fname, gx, gy, gz, nown, direct, pattern, bighdr)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: gx, gy, gz, nown, pattern
    logical, intent(in) :: direct, bighdr
    character(len=1024) :: path
    character(len=256) :: msg
    integer :: ncid, d_t, d_x, d_y, d_z, d_a, d_s, vid, iv, w, rec, old, cmin, nsg, i, j, s, t
    integer(int64) :: smin, a, fsz, csz
    type(cdf_file_t) :: hf, wf(4)
    logical :: laid, ex, lok
    integer :: sidx(NV), sown(NV), wseg(NV)
    integer(int64) :: soff(NV), snb(NV), wp1(NV), wp2(NV), fp, base, nbuf, cov, hb
    type(cdf_buf_t) :: pb, hbuf(NV), tb
    character(len=4000) :: big
    nx = gx; ny = gy; nz = gz
    path = trim(dir)//'/'//fname
    ! define with netCDF under the scratch name (as nc_io's nc_create / nc_enddef / nc_close)
    call nc(nf90_create(trim(path)//'.hdr', ior(nf90_clobber, nf90_64bit_data), ncid), 'create')
    call nc(nf90_set_fill(ncid, nf90_nofill, old), 'nofill')
    call nc(nf90_def_dim(ncid, 'time', nf90_unlimited, d_t), 'dim')
    call nc(nf90_def_dim(ncid, 'xi', nx, d_x), 'dim')
    call nc(nf90_def_dim(ncid, 'eta', ny, d_y), 'dim')
    call nc(nf90_def_dim(ncid, 's', nz, d_z), 'dim')
    call nc(nf90_def_dim(ncid, 'auxil', NA, d_a), 'dim')
    call nc(nf90_def_dim(ncid, 'nscal', NS, d_s), 'dim')
    do iv = 1, NV
      select case (vrank(iv))
      case (0)
        call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_t/), vid), 'def')
      case (1)
        if (vname(iv) == 'time_step') then
          call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_a, d_t/), vid), 'def')
        elseif (visrec(iv)) then
          call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_s, d_t/), vid), 'def')
        else
          call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_s/), vid), 'def')
        endif
      case (2)
        call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_x, d_y, d_t/), vid), 'def')
      case default
        if (visrec(iv)) then
          call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_x, d_y, d_z, d_t/), vid), 'def')
        else
          call nc(nf90_def_var(ncid, trim(vname(iv)), vtype(iv), (/d_x, d_y, d_z/), vid), 'def')
        endif
      end select
      call nc(nf90_put_att(ncid, vid, 'long_name', 'variable '//trim(vname(iv))), 'att')
    enddo
    call nc(nf90_put_att(ncid, nf90_global, 'title', 'test_cdf_write '//trim(fname)), 'att')
    if (bighdr) then
      big = repeat('header longer than the first read. ', 110)
      do i = 1, 80                                    ! about 300 KB of attributes
        write(msg, '(a,i0)') 'filler_', i
        call nc(nf90_put_att(ncid, nf90_global, trim(msg), big), 'att')
      enddo
    endif
    call nc(nf90_enddef(ncid, h_minfree=0, v_align=4096, v_minfree=0, r_align=4096), 'enddef')
    call nc(nf90_close(ncid), 'close')
    ! the header into the real file
    lok = cdf_copy_header(trim(path)//'.hdr', trim(path), direct, hf, cmin, smin, laid, msg)
    call chk(lok, trim(fname)//' cdf_copy_header: '//trim(msg))
    if (.not. lok) return
    inquire(file=trim(path)//'.hdr', exist=ex)
    call chk(.not. ex, trim(fname)//' scratch header removed')
    inquire(file=trim(path), size=fsz)
    call chk(fsz == hf%hdr_end, trim(fname)//' length after the header copy')
    call chk(mod(hf%hdr_end, 4096_int64) == 0 .and. mod(hf%begin_rec, 4096_int64) == 0, &
             trim(fname)//' aligned header end and record section')
    call layout_check(path, laid, cmin, smin, fname)
    ! the writers open it
    do w = 1, nown
      lok = cdf_open_w(trim(path), wf(w), direct, msg)
      call chk(lok, trim(fname)//' cdf_open_w: '//trim(msg))
      if (.not. lok) return
    enddo
    a = wf(1)%align
    do rec = 1, NREC
      ! segments of this record (the writers' view): every variable written now, its owner
      nsg = 0
      do iv = 1, NV
        if (vwhen(iv) < 0) cycle
        if (vwhen(iv) == 2 .and. rec /= 2) cycle
        i = cdf_var_index(wf(1), trim(vname(iv)))
        call chk(i > 0, trim(fname)//' variable in the header '//trim(vname(iv)))
        nsg = nsg + 1
        sidx(nsg) = iv
        soff(nsg) = wf(1)%v(i)%begin
        if (wf(1)%v(i)%isrec) soff(nsg) = soff(nsg) + int(rec - 1, int64) * wf(1)%recsize
        snb(nsg) = wf(1)%v(i)%vraw
        if (vname(iv) == 'ocean_time' .or. vname(iv) == 'time_step') then
          sown(nsg) = 0
        elseif (pattern == 1) then
          sown(nsg) = mod(iv, nown)
        else
          sown(nsg) = nown - 1                           ! all but the small ones on the last writer
          if (mod(iv, 3) == 0) sown(nsg) = mod(iv, nown)
        endif
      enddo
      call sort_segs(nsg, soff, snb, sown, sidx)
      lok = cdf_plan(nsg, soff, snb, a, wseg, wp1, wp2)
      call chk(lok, trim(fname)//' cdf_plan')
      if (.not. lok) return
      ! heads: a segment that does not write its first page hands those bytes to the page's writer
      do s = 1, nsg
        if (wp1(s) == soff(s) / a) cycle
        hb = min(snb(s), (soff(s) / a + 1) * a - soff(s))
        lok = cdf_buf_reserve(tb, ((snb(s) + 8) / 4096 + 1) * 4096)
        call pack_var(sidx(s), rec, file_type(wf(1), sidx(s)), tb, 0_int64)
        lok = cdf_buf_reserve(hbuf(s), 4096_int64)
        hbuf(s)%u8(1:hb) = tb%u8(1:hb)
      enddo
      ! each writer writes the pages of its segments
      do w = 0, nown - 1
        do s = 1, nsg
          if (sown(s) /= w .or. wp1(s) > wp2(s)) cycle
          fp = soff(s) / a
          base = fp * a
          nbuf = (wp2(s) - fp + 1) * a
          lok = cdf_buf_reserve(pb, nbuf)
          ! read-modify-write of a partly covered first or last page
          if (wp1(s) == fp .and. soff(s) > base) lok = cdf_read_page(wf(w + 1), pb, 0_int64, base, a)
          cov = min(soff(s) + snb(s), (wp2(s) + 1) * a) - max(soff(s), wp2(s) * a)
          do t = s + 1, nsg
            if (wseg(t) /= s) cycle
            cov = cov + min(snb(t), (soff(t) / a + 1) * a - soff(t))
          enddo
          if (cov < a .and. .not. (wp2(s) == fp .and. wp1(s) == fp .and. soff(s) > base)) &
               lok = cdf_read_page(wf(w + 1), pb, (wp2(s) - fp) * a, wp2(s) * a, a)
          call pack_var(sidx(s), rec, file_type(wf(1), sidx(s)), pb, soff(s) - base)
          do t = s + 1, nsg
            if (wseg(t) /= s) cycle
            hb = min(snb(t), (soff(t) / a + 1) * a - soff(t))
            pb%u8(soff(t) - base + 1:soff(t) - base + hb) = hbuf(t)%u8(1:hb)
          enddo
          lok = cdf_write(wf(w + 1), pb, (wp1(s) - fp) * a, wp1(s) * a, (wp2(s) - wp1(s) + 1) * a, 8192_int64)
          call chk(lok, trim(fname)//' cdf_write')
        enddo
      enddo
      lok = cdf_set_numrecs(wf(1), int(rec, int64))
      call chk(lok, trim(fname)//' cdf_set_numrecs')
    enddo
    csz = cdf_calcsize(wf(1))
    do w = 1, nown
      lok = cdf_close_w(wf(w), .true.)
      call chk(lok, trim(fname)//' cdf_close_w')
    enddo
    lok = cdf_truncate(trim(path), csz)
    call chk(lok, trim(fname)//' cdf_truncate')
    inquire(file=trim(path), size=fsz)
    call chk(fsz == csz, trim(fname)//' exact length')
    call verify(path, fname)
    call cdf_buf_free(pb)
    call cdf_buf_free(tb)
    do j = 1, NV
      call cdf_buf_free(hbuf(j))
    enddo
    write(*,'(3a,i0,a,l1,a,i0,a,l1,a,i0,a,f6.1,a)') 'case ', trim(fname), ': writers ', nown, ' direct ', direct, &
         ' pattern ', pattern, '  layout read ', laid, ' (', cmin, ' x ', real(smin) / 1048576.0, ' MiB)'
  end subroutine run_case

  integer function file_type(f, iv)
    type(cdf_file_t), intent(in) :: f
    integer, intent(in) :: iv
    file_type = f%v(cdf_var_index(f, trim(vname(iv))))%xtype
  end function file_type

  subroutine sort_segs(n, off, nb, own, idx)
    integer, intent(in) :: n
    integer(int64), intent(inout) :: off(n), nb(n)
    integer, intent(inout) :: own(n), idx(n)
    integer :: i, j, ti, tw
    integer(int64) :: to, tn
    do i = 2, n
      to = off(i); tn = nb(i); tw = own(i); ti = idx(i)
      j = i - 1
      do while (j >= 1)
        if (off(j) <= to) exit
        off(j + 1) = off(j); nb(j + 1) = nb(j); own(j + 1) = own(j); idx(j + 1) = idx(j)
        j = j - 1
      enddo
      off(j + 1) = to; nb(j + 1) = tn; own(j + 1) = tw; idx(j + 1) = ti
    enddo
  end subroutine sort_segs

  subroutine verify(path, fname)
    character(len=*), intent(in) :: path, fname
    integer :: ncid, fmt, nrf, dimid, iv, vid, rec, xt, nd
    integer(int64) :: n, e, nbad
    real(real64), allocatable :: r8(:)
    real(real32), allocatable :: r4(:)
    integer(int32), allocatable :: ri(:)
    integer :: start(4), cnt(4)
    call nc(nf90_open(trim(path), nf90_nowrite, ncid), 'open '//trim(fname))
    call nc(nf90_inquire(ncid, formatNum=fmt, unlimitedDimId=dimid), 'inquire')
    call chk(fmt == nf90_format_64bit_data, trim(fname)//' format CDF5')
    call nc(nf90_inquire_dimension(ncid, dimid, len=nrf), 'nrec')
    call chk(nrf == NREC, trim(fname)//' record count')
    do iv = 1, NV
      call nc(nf90_inq_varid(ncid, trim(vname(iv)), vid), 'varid')
      call nc(nf90_inquire_variable(ncid, vid, xtype=xt, ndims=nd), 'inqvar')
      n = nelem(iv)
      do rec = 1, NREC
        if (.not. visrec(iv) .and. rec > 1) exit
        start = 1; cnt = 1
        if (visrec(iv)) then
          cnt(1:nd - 1) = shape_of(iv, nd - 1)
          start(nd) = rec
        else
          cnt(1:nd) = shape_of(iv, nd)
        endif
        nbad = 0
        select case (xt)
        case (nf90_double)
          allocate(r8(n))
          call nc(nf90_get_var(ncid, vid, r8, start=start(1:nd), count=cnt(1:nd)), 'get '//trim(vname(iv)))
          do e = 1, n
            if (transfer(r8(e), 1_int64) /= transfer(expect8(iv, rec, e - 1), 1_int64)) nbad = nbad + 1
          enddo
          deallocate(r8)
        case (nf90_float)
          allocate(r4(n))
          call nc(nf90_get_var(ncid, vid, r4, start=start(1:nd), count=cnt(1:nd)), 'get '//trim(vname(iv)))
          do e = 1, n
            if (transfer(r4(e), 1_int32) /= transfer(expect4(iv, rec, e - 1), 1_int32)) nbad = nbad + 1
          enddo
          deallocate(r4)
        case default
          allocate(ri(n))
          call nc(nf90_get_var(ncid, vid, ri, start=start(1:nd), count=cnt(1:nd)), 'get '//trim(vname(iv)))
          do e = 1, n
            if (ri(e) /= gint(iv, rec, e - 1)) nbad = nbad + 1
          enddo
          deallocate(ri)
        end select
        call chk(nbad == 0, trim(fname)//' values of '//trim(vname(iv))//' rec '//achar(48 + rec))
        if (nbad > 0) write(*,'(a,i0,a,i0)') '   wrong values: ', nbad, ' of ', n
      enddo
    enddo
    call nc(nf90_close(ncid), 'close')
  end subroutine verify

  function shape_of(iv, m) result(c)
    integer, intent(in) :: iv, m
    integer :: c(m)
    c = 1
    select case (vrank(iv))
    case (1)
      c(1) = int(nelem(iv))
    case (2)
      c(1) = nx; c(2) = ny
    case (3)
      c(1) = nx; c(2) = ny; c(3) = nz
    end select
  end function shape_of

  real(real64) function expect8(iv, rec, e)
    ! a double variable's value: 0 where never written, the real(4) source widened, or the real(8) source
    integer, intent(in) :: iv, rec
    integer(int64), intent(in) :: e
    integer :: r
    r = rec
    if (.not. visrec(iv)) r = 2
    if (vwhen(iv) < 0) then
      expect8 = 0.0d0
    elseif (vsrc(iv) == 4) then
      expect8 = real(gval4(iv, r, e), real64)
    else
      expect8 = gval(iv, r, e)
    endif
  end function expect8

  real(real32) function expect4(iv, rec, e)
    integer, intent(in) :: iv, rec
    integer(int64), intent(in) :: e
    integer :: r
    r = rec
    if (.not. visrec(iv)) r = 2
    if (vwhen(iv) < 0) then
      expect4 = 0.0
    elseif (vsrc(iv) == 8) then
      expect4 = real(gval(iv, r, e), real32)
    else
      expect4 = gval4(iv, r, e)
    endif
  end function expect4

  subroutine layout_check(path, laid, cmin, smin, fname)
    ! cdf_layout_fd's stripe count and size against `lfs getstripe` (when lfs runs here)
    character(len=*), intent(in) :: path, fname
    logical, intent(in) :: laid
    integer, intent(in) :: cmin
    integer(int64), intent(in) :: smin
    character(len=1024) :: tmp
    integer :: u, ios, ec, c, ncomp
    integer(int64) :: sz
    tmp = trim(path)//'.lfs.txt'
    call execute_command_line('lfs getstripe --component-count '//trim(path)//' > '//trim(tmp)//' 2>&1 && '// &
         'lfs getstripe -c '//trim(path)//' >> '//trim(tmp)//' 2>&1 && lfs getstripe -S '//trim(path)//' >> '// &
         trim(tmp)//' 2>&1', exitstat=ec)
    if (ec /= 0) then
      write(*,'(3a,l1)') 'layout ', trim(fname), ': lfs getstripe not usable here; cdf_layout_fd found a layout: ', laid
      return
    endif
    open(newunit=u, file=trim(tmp), status='old', action='read', iostat=ios)
    if (ios /= 0) return
    read(u, *, iostat=ios) ncomp
    if (ios == 0) read(u, *, iostat=ios) c
    if (ios == 0) read(u, *, iostat=ios) sz
    close(u)
    if (ios /= 0) then
      write(*,'(3a)') 'layout ', trim(fname), ': lfs output not read'
      return
    endif
    if (ncomp > 0) then
      ! composite (progressive) layout: lfs -c / -S print one component; cdf_layout_fd takes the first record's
      write(*,'(3a,i0,a,i0,a,i0,a,i0)') 'layout ', trim(fname), ': composite, ', ncomp, ' components; first record on ', &
           cmin, ' x ', smin, '; lfs (last instantiated) ', c, ' x ', sz
      call chk(laid .and. cmin >= 1 .and. smin >= 65536, trim(fname)//' composite layout read')
      return
    endif
    call chk(laid .and. cmin == c .and. smin == sz, trim(fname)//' layout = lfs getstripe')
    write(*,'(3a,i0,a,i0,a,i0,a,i0)') 'layout ', trim(fname), ': lfs ', c, ' x ', sz, '  cdf_layout_fd ', cmin, &
         ' x ', smin
  end subroutine layout_check

  subroutine speed()
    ! pack and write rates (information, not checks)
    integer(int64), parameter :: N4 = 67108864_int64, N8 = 33554432_int64
    real(real32), allocatable :: a4(:)
    real(real64), allocatable :: a8(:)
    type(cdf_buf_t) :: b
    type(cdf_file_t) :: hf, wf
    integer(int64) :: e, c0, c1, cr, smin
    integer :: cmin
    logical :: lok, laid
    character(len=256) :: msg
    character(len=1024) :: path
    integer :: ncid, d, vid, old
    allocate(a4(N4), a8(N8))
    do e = 1, N4
      a4(e) = real(e, real32) * 0.5
    enddo
    do e = 1, N8
      a8(e) = real(e, real64) * 0.25d0
    enddo
    lok = cdf_buf_reserve(b, N4 * 4 + 4096)
    call system_clock(c0, cr)
    call cdf_pack_r4(a4, N4, CDF_FLOAT, b, 4_int64)
    call system_clock(c1)
    write(*,'(a,f7.2,a)') 'speed: cdf_pack_r4 float  ', real(N4 * 4, 8) / 1.0d9 / max(real(c1 - c0, 8) / cr, 1.0d-9), &
         ' GB/s (first touch of the buffer)'
    call system_clock(c0, cr)
    call cdf_pack_r4(a4, N4, CDF_FLOAT, b, 4_int64)
    call system_clock(c1)
    write(*,'(a,f7.2,a)') 'speed: cdf_pack_r4 float  ', real(N4 * 4, 8) / 1.0d9 / max(real(c1 - c0, 8) / cr, 1.0d-9), ' GB/s'
    call system_clock(c0, cr)
    call cdf_pack_r4(a4, N4, CDF_FLOAT, b, 0_int64)
    call system_clock(c1)
    write(*,'(a,f7.2,a)') 'speed: cdf_pack_r4 float  ', real(N4 * 4, 8) / 1.0d9 / max(real(c1 - c0, 8) / cr, 1.0d-9), &
         ' GB/s (8-byte aligned destination)'
    call system_clock(c0)
    call cdf_pack_r8(a8, N8, CDF_DOUBLE, b, 4_int64)
    call system_clock(c1)
    write(*,'(a,f7.2,a)') 'speed: cdf_pack_r8 double ', real(N8 * 8, 8) / 1.0d9 / max(real(c1 - c0, 8) / cr, 1.0d-9), ' GB/s'
    call system_clock(c0)
    call cdf_pack_r8(a8, N8, CDF_FLOAT, b, 4_int64)
    call system_clock(c1)
    write(*,'(a,f7.2,a)') 'speed: cdf_pack_r8 float  ', real(N8 * 8, 8) / 1.0d9 / max(real(c1 - c0, 8) / cr, 1.0d-9), &
         ' GB/s (of the real(8) source)'
    ! one writer, one 256-MiB O_DIRECT write in 64-MiB pieces
    path = trim(dir)//'/w_speed.nc'
    call nc(nf90_create(trim(path)//'.hdr', ior(nf90_clobber, nf90_64bit_data), ncid), 'create')
    call nc(nf90_set_fill(ncid, nf90_nofill, old), 'nofill')
    call nc(nf90_def_dim(ncid, 'n', int(N4), d), 'dim')
    call nc(nf90_def_var(ncid, 'v', nf90_float, (/d/), vid), 'def')
    call nc(nf90_enddef(ncid, h_minfree=0, v_align=4096, v_minfree=0, r_align=4096), 'enddef')
    call nc(nf90_close(ncid), 'close')
    lok = cdf_copy_header(trim(path)//'.hdr', trim(path), .true., hf, cmin, smin, laid, msg)
    lok = lok .and. cdf_open_w(trim(path), wf, .true., msg)
    if (.not. lok) then
      write(*,'(2a)') 'speed: no write test: ', trim(msg)
      return
    endif
    call cdf_pack_r4(a4, N4, CDF_FLOAT, b, 0_int64)
    call system_clock(c0)
    lok = cdf_write(wf, b, 0_int64, wf%hdr_end, N4 * 4, 67108864_int64)
    call system_clock(c1)
    call chk(lok, 'speed write')
    write(*,'(a,f7.2,a)') 'speed: cdf_write one writer ', real(N4 * 4, 8) / 1.0d9 / max(real(c1 - c0, 8) / cr, 1.0d-9), &
         ' GB/s'
    lok = cdf_close_w(wf, .true.)
    call verify_speed(path, a4)
    call cdf_buf_free(b)
  end subroutine speed

  subroutine verify_speed(path, a4)
    character(len=*), intent(in) :: path
    real(real32), intent(in) :: a4(:)
    integer :: ncid, vid
    real(real32), allocatable :: r4(:)
    allocate(r4(size(a4)))
    call nc(nf90_open(trim(path), nf90_nowrite, ncid), 'open speed')
    call nc(nf90_inq_varid(ncid, 'v', vid), 'varid')
    call nc(nf90_get_var(ncid, vid, r4), 'get speed')
    call nc(nf90_close(ncid), 'close')
    call chk(all(r4 == a4), 'speed file values')
  end subroutine verify_speed

end program test_cdf_write

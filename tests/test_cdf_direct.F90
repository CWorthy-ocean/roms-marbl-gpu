program test_cdf_direct
  ! cdf_direct against nf90_get_var, bit for bit, on files this program
  ! writes: CDF-1, CDF-2, CDF-5 with fixed and record variables of every
  ! type cdf_direct converts (double, float, int, short, byte), odd lengths
  ! (no length a multiple of the 4-KiB alignment), NaN / -0 / subnormal /
  ! huge values, a header longer than the first read, other header and
  ! variable alignments (nf90_enddef), a file with one short record variable
  ! (records packed, not padded), a truncated copy, a variable that is not
  ! in the file, types left to netCDF, and a netCDF-4 file (refused).
  ! usage: test_cdf_direct <scratch directory>   (exit status 1 on a failure)
  use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, real64
  use netcdf
  use cdf_direct
  implicit none
  character(len=512) :: dir
  integer :: nfail = 0, ncheck = 0
  integer, parameter :: NX = 37, NY = 23, NZ = 5, NR = 3

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) dir = '.'
  if (dir == '--file') then
    call check_file()
    stop
  endif
  call run_format('cdf1.nc', nf90_classic_model, 0, .false.)
  call run_format('cdf2.nc', nf90_64bit_offset, 0, .false.)
  call run_format('cdf5.nc', nf90_64bit_data, 0, .false.)
  call run_format('cdf5_big_header.nc', nf90_64bit_data, 1, .false.)
  call run_format('cdf2_align.nc', nf90_64bit_offset, 2, .false.)
  call run_format('cdf1_onerec.nc', nf90_classic_model, 3, .false.)
  call run_format('cdf5_trunc.nc', nf90_64bit_data, 0, .true.)
  call refuse_nc4('nc4.nc')
  write(*,'(a,i0,a,i0)') 'test_cdf_direct: checks ', ncheck, '  failures ', nfail
  if (nfail > 0) then
    write(*,'(a)') 'test_cdf_direct: FAIL'
    error stop 1
  endif
  write(*,'(a)') 'test_cdf_direct: PASS'

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

  real(real64) function val(i, j, k, r, iv)
    ! test values: special ones at fixed places, else non-trivial bits
    integer, intent(in) :: i, j, k, r, iv
    real(real64) :: z
    integer :: h
    h = mod(i * 7919 + j * 104729 + k * 1299709 + r * 15485863 + iv * 32452843, 1000003)
    z = 0.0_real64
    select case (mod(h, 97))
    case (0); val = ieee_nan()
    case (1); val = -z
    case (2); val = 4.9e-324_real64 * (1 + mod(h, 5))
    case (3); val = -1.7e308_real64
    case (4); val = 1.0e-40_real64
    case default
      val = (h - 500000) * 1.2345678901234567e-3_real64 + 1.0_real64 / (h + 1)
    end select
  end function val

  real(real64) function ieee_nan()
    integer(int64), parameter :: bits = int(z'7FF8000000000123', int64)
    ieee_nan = transfer(bits, 1.0_real64)
  end function ieee_nan

  subroutine make_file(path, cmode, variant)
    character(len=*), intent(in) :: path
    integer, intent(in) :: cmode, variant
    integer :: ncid, dx, dy, dz, dt, vd2, vd3, vr2, vr3, vf3, vi2, vs2, vb2, vus, vsx, dt2, r, k, j, i
    real(real64) :: a3(NX, NY, NZ), a2(NX, NY)
    real(real32) :: f3(NX, NY, NZ)
    integer :: i2(NX, NY)
    integer(int16) :: s2(NX, NY)
    integer(int8) :: b2(NX, NY)
    character(len=:), allocatable :: big
    call nc(nf90_create(path, ior(nf90_clobber, cmode), ncid), 'create '//path)
    call nc(nf90_put_att(ncid, nf90_global, 'title', 'cdf_direct test'), 'att')
    call nc(nf90_put_att(ncid, nf90_global, 'ints', (/1, 2, 3/)), 'att')
    call nc(nf90_put_att(ncid, nf90_global, 'dbl', 3.25_real64), 'att')
    call nc(nf90_put_att(ncid, nf90_global, 'odd', 'abcde'), 'att')
    if (variant == 1) then
      allocate(character(len=400001) :: big)
      big = repeat('x', 400001)
      call nc(nf90_put_att(ncid, nf90_global, 'long_text', big), 'att big')
    endif
    call nc(nf90_def_dim(ncid, 'xi_rho', NX, dx), 'dim')
    call nc(nf90_def_dim(ncid, 'eta_rho', NY, dy), 'dim')
    call nc(nf90_def_dim(ncid, 's_rho', NZ, dz), 'dim')
    call nc(nf90_def_dim(ncid, 'time', nf90_unlimited, dt), 'dim')
    if (variant == 3) then
      ! one record variable of shorts (37 x 1 values: 74 bytes, padded 76): records packed
      call nc(nf90_def_dim(ncid, 'one', 1, dt2), 'dim')
      call nc(nf90_def_var(ncid, 'sx', nf90_short, (/dx, dt2, dt/), vsx), 'def sx')
      call nc(nf90_def_var(ncid, 'd2', nf90_double, (/dx, dy/), vd2), 'def d2')
      call nc(nf90_enddef(ncid), 'enddef')
      call nc(nf90_put_var(ncid, vd2, reshape((/(val(i, 1, 1, 0, 1), i = 1, NX * NY)/), (/NX, NY/))), 'put d2')
      do r = 1, NR
        do i = 1, NX
          s2(i, 1) = int(mod(i * 1013 + r * 7, 65536) - 32768, int16)
        enddo
        call nc(nf90_put_var(ncid, vsx, s2(:, 1:1), start=(/1, 1, r/), count=(/NX, 1, 1/)), 'put sx')
      enddo
      call nc(nf90_close(ncid), 'close')
      return
    endif
    call nc(nf90_def_var(ncid, 'd2', nf90_double, (/dx, dy/), vd2), 'def')
    call nc(nf90_put_att(ncid, vd2, 'units', 'm'), 'att')
    call nc(nf90_def_var(ncid, 'd3', nf90_double, (/dx, dy, dz/), vd3), 'def')
    call nc(nf90_def_var(ncid, 'r2', nf90_double, (/dx, dy, dt/), vr2), 'def')
    call nc(nf90_put_att(ncid, vr2, 'long_name', 'record 2-D double'), 'att')
    call nc(nf90_put_att(ncid, vr2, 'fv', (/-1.0_real32, 2.0_real32/)), 'att')
    call nc(nf90_def_var(ncid, 'r3', nf90_double, (/dx, dy, dz, dt/), vr3), 'def')
    call nc(nf90_def_var(ncid, 'f3', nf90_float, (/dx, dy, dz, dt/), vf3), 'def')
    call nc(nf90_def_var(ncid, 'i2', nf90_int, (/dx, dy, dt/), vi2), 'def')
    call nc(nf90_def_var(ncid, 's2', nf90_short, (/dx, dy, dt/), vs2), 'def')
    call nc(nf90_put_att(ncid, vs2, 'scale_factor', 0.5_real64), 'att')
    call nc(nf90_def_var(ncid, 'b2', nf90_byte, (/dx, dy, dt/), vb2), 'def')
    if (cmode == nf90_64bit_data) then
      call nc(nf90_def_var(ncid, 'us2', nf90_ushort, (/dx, dy, dt/), vus), 'def')
    endif
    if (variant == 2) then
      call nc(nf90_enddef(ncid, h_minfree=1001, v_align=4, v_minfree=13, r_align=4), 'enddef align')
    else
      call nc(nf90_enddef(ncid), 'enddef')
    endif
    do k = 1, NZ
      do j = 1, NY
        do i = 1, NX
          a3(i, j, k) = val(i, j, k, 0, 2)
        enddo
      enddo
    enddo
    call nc(nf90_put_var(ncid, vd3, a3), 'put d3')
    call nc(nf90_put_var(ncid, vd2, a3(:, :, 2)), 'put d2')
    do r = 1, NR
      do k = 1, NZ
        do j = 1, NY
          do i = 1, NX
            a3(i, j, k) = val(i, j, k, r, 3)
            ! float: finite values and the specials a float holds
            f3(i, j, k) = real(val(i, j, k, r, 4) * 1.0e-3_real64, real32)
            if (mod(i + j + k + r, 53) == 0) f3(i, j, k) = transfer(int(z'7FC00007', int32), 1.0_real32)
            if (mod(i + j + k + r, 59) == 0) f3(i, j, k) = transfer(int(z'00000003', int32), 1.0_real32)
            if (mod(i + 2 * j + k + r, 61) == 0) f3(i, j, k) = transfer(int(z'7F800001', int32), 1.0_real32)
          enddo
        enddo
      enddo
      a2 = a3(:, :, 1) * 3.0_real64
      do j = 1, NY
        do i = 1, NX
          i2(i, j) = (i * 2654435 + j * 40503 + r * 977) * merge(1, -1, mod(i + j, 2) == 0)
          s2(i, j) = int(mod(i * 331 + j * 7 + r * 13, 65536) - 32768, int16)
          b2(i, j) = int(mod(i * 5 + j * 3 + r, 256) - 128, int8)
        enddo
      enddo
      call nc(nf90_put_var(ncid, vr2, a2, start=(/1, 1, r/), count=(/NX, NY, 1/)), 'put r2')
      call nc(nf90_put_var(ncid, vr3, a3, start=(/1, 1, 1, r/), count=(/NX, NY, NZ, 1/)), 'put r3')
      call nc(nf90_put_var(ncid, vf3, f3, start=(/1, 1, 1, r/), count=(/NX, NY, NZ, 1/)), 'put f3')
      call nc(nf90_put_var(ncid, vi2, i2, start=(/1, 1, r/), count=(/NX, NY, 1/)), 'put i2')
      call nc(nf90_put_var(ncid, vs2, s2, start=(/1, 1, r/), count=(/NX, NY, 1/)), 'put s2')
      call nc(nf90_put_var(ncid, vb2, b2, start=(/1, 1, r/), count=(/NX, NY, 1/)), 'put b2')
      if (cmode == nf90_64bit_data) then
        call nc(nf90_put_var(ncid, vus, mod(abs(i2), 60000), start=(/1, 1, r/), count=(/NX, NY, 1/)), 'put us2')
      endif
    enddo
    call nc(nf90_close(ncid), 'close')
  end subroutine make_file

  subroutine truncate_copy(src, dst, cut)
    ! dst = src without its last cut bytes (stream I/O)
    character(len=*), intent(in) :: src, dst
    integer, intent(in) :: cut
    integer :: u, n
    integer(int8), allocatable :: bytes(:)
    open(newunit=u, file=src, access='stream', form='unformatted', status='old')
    inquire(unit=u, size=n)
    allocate(bytes(n))
    read(u) bytes
    close(u)
    open(newunit=u, file=dst, access='stream', form='unformatted', status='replace')
    write(u) bytes(1:n - cut)
    close(u)
  end subroutine truncate_copy

  subroutine run_format(name, cmode, variant, trunc)
    character(len=*), intent(in) :: name
    integer, intent(in) :: cmode, variant
    logical, intent(in) :: trunc
    character(len=600) :: path
    character(len=256) :: msg
    type(cdf_file_t) :: f
    type(cdf_buf_t) :: b
    integer :: ncid, r, nloc, nrej, ierr
    character(len=8), parameter :: names(9) = (/'d2      ', 'd3      ', 'r2      ', 'r3      ', 'f3      ', &
                                                'i2      ', 's2      ', 'b2      ', 'us2     '/)
    logical :: ok
    path = trim(dir)//'/'//name
    if (trunc) then
      call make_file(trim(path)//'.full', cmode, variant)
      call truncate_copy(trim(path)//'.full', trim(path), 30000)
    else
      call make_file(trim(path), cmode, variant)
    endif
    ok = cdf_open(trim(path), f, msg)
    call chk(ok, trim(name)//': cdf_open ('//trim(msg)//')')
    if (.not. ok) return
    nloc = 0; nrej = 0
    ierr = nf90_open(trim(path), nf90_nowrite, ncid)
    if (ierr /= nf90_noerr) then
      ! netCDF itself refuses the truncated file: then only cdf_locate's answers are checked
      call chk(trunc, trim(name)//': nf90_open '//trim(nf90_strerror(ierr)))
      write(*,'(3a)') trim(name), ': netCDF refuses it: ', trim(nf90_strerror(ierr))
      call chk(located(f, 'r2', 3, (/1, 1, NR/), (/NX, NY, 1/)), trim(name)//': r2 last record located')
      call chk(.not. located(f, 'r3', 4, (/1, 1, 1, NR/), (/NX, NY, NZ, 1/)), trim(name)//': r3 last record refused')
      call cdf_close(f)
      return
    endif
    if (variant == 3) then
      do r = 1, NR
        call one_var(ncid, f, b, 'sx', 3, (/1, 1, r, 0/), (/NX, 1, 1, 0/), name, nloc, nrej)
      enddo
      call one_var(ncid, f, b, 'd2', 2, (/1, 1, 0, 0/), (/NX, NY, 0, 0/), name, nloc, nrej)
    else
      call one_var(ncid, f, b, 'd2', 2, (/1, 1, 0, 0/), (/NX, NY, 0, 0/), name, nloc, nrej)
      call one_var(ncid, f, b, 'd3', 3, (/1, 1, 1, 0/), (/NX, NY, NZ, 0/), name, nloc, nrej)
      call one_var(ncid, f, b, 'd3', 3, (/1, 1, 1, 0/), (/NX, NY, NZ - 2, 0/), name, nloc, nrej)
      do r = 1, NR
        call one_var(ncid, f, b, 'r2', 3, (/1, 1, r, 0/), (/NX, NY, 1, 0/), name, nloc, nrej)
        call one_var(ncid, f, b, 'r3', 4, (/1, 1, 1, r/), (/NX, NY, NZ, 1/), name, nloc, nrej)
        call one_var(ncid, f, b, 'f3', 4, (/1, 1, 1, r/), (/NX, NY, NZ, 1/), name, nloc, nrej)
        call one_var(ncid, f, b, 'i2', 3, (/1, 1, r, 0/), (/NX, NY, 1, 0/), name, nloc, nrej)
        call one_var(ncid, f, b, 's2', 3, (/1, 1, r, 0/), (/NX, NY, 1, 0/), name, nloc, nrej)
        call one_var(ncid, f, b, 'b2', 3, (/1, 1, r, 0/), (/NX, NY, 1, 0/), name, nloc, nrej)
      enddo
      ! left to netCDF: a missing variable, a type not converted, a slab that is not contiguous,
      ! a record past the end, a count over two records
      call chk(.not. located(f, 'nothere', 3, (/1, 1, 1/), (/NX, NY, 1/)), trim(name)//': missing variable refused')
      call chk(.not. located(f, 'us2', 3, (/1, 1, 1/), (/NX, NY, 1/)), trim(name)//': ushort refused')
      call chk(.not. located(f, 'r3', 4, (/2, 1, 1, 1/), (/NX - 1, NY, NZ, 1/)), trim(name)//': strided slab refused')
      call chk(.not. located(f, 'r3', 4, (/1, 1, 1, NR + 1/), (/NX, NY, NZ, 1/)), trim(name)//': record past end refused')
      call chk(.not. located(f, 'r2', 3, (/1, 1, 1/), (/NX, NY, 2/)), trim(name)//': two records refused')
      call chk(.not. located(f, 'r2', 2, (/1, 1/), (/NX, NY/)), trim(name)//': rank mismatch refused')
    endif
    if (trunc) then
      call chk(nrej > 0 .and. nloc > 0, trim(name)//': truncated file: some variables refused, others read')
    else
      call chk(nrej == 0, trim(name)//': every variable located')
    endif
    write(*,'(2a,i0,a,i0,a)') trim(name), ': ', nloc, ' slabs read directly, ', nrej, ' left to netCDF'
    call nc(nf90_close(ncid), 'close')
    call cdf_buf_free(b)
    call cdf_close(f)
  end subroutine run_format

  logical function located(f, vname, m, start, cnt)
    type(cdf_file_t), intent(in) :: f
    character(len=*), intent(in) :: vname
    integer, intent(in) :: m, start(m), cnt(m)
    integer(int64) :: off, nbytes
    integer :: xt, gx, gy
    located = cdf_locate(f, vname, m, start, cnt, off, nbytes, xt, gx, gy)
  end function located

  subroutine one_var(ncid, f, b, vname, m, start4, cnt4, fname, nloc, nrej)
    ! one slab: cdf_locate/read/unpack (whole and windows, two piece sizes) vs nf90_get_var
    integer, intent(in) :: ncid, m, start4(4), cnt4(4)
    type(cdf_file_t), intent(in) :: f
    type(cdf_buf_t), intent(inout) :: b
    character(len=*), intent(in) :: vname, fname
    integer, intent(inout) :: nloc, nrej
    integer(int64) :: off, nbytes, dshift
    integer :: xt, gx, gy, varid, xtype, n, n3, i0, j0, n1, n2, i, j, k, ip, ierr
    real(real64), allocatable :: ref(:), got(:), w(:)
    integer, allocatable :: iref(:)
    logical :: ok, same
    integer(int64), parameter :: pieces(2) = (/4096_int64, 268435456_int64/)
    if (.not. cdf_locate(f, vname, m, start4(1:m), cnt4(1:m), off, nbytes, xt, gx, gy)) then
      nrej = nrej + 1
      return
    endif
    nloc = nloc + 1
    n = 1
    do i = 1, m
      n = n * cnt4(i)
    enddo
    n3 = 1
    if (m >= 3 .and. cnt4(3) > 1) n3 = cnt4(3)
    call chk(gx == cnt4(1), trim(fname)//' '//vname//': gx')
    allocate(ref(n), got(n))
    call nc(nf90_inq_varid(ncid, vname, varid), 'varid '//vname)
    call nc(nf90_inquire_variable(ncid, varid, xtype=xtype), 'inq '//vname)
    call chk(xtype == xt, trim(fname)//' '//vname//': type')
    if (xtype == nf90_int .or. xtype == nf90_short .or. xtype == nf90_byte) then
      allocate(iref(n))
      ierr = nf90_get_var(ncid, varid, iref, start=start4(1:m), count=cnt4(1:m))
      ref = real(iref, real64)
    else
      ierr = nf90_get_var(ncid, varid, ref, start=start4(1:m), count=cnt4(1:m))
    endif
    call chk(ierr == nf90_noerr, trim(fname)//' '//vname//': nf90_get_var '//trim(nf90_strerror(ierr)))
    do ip = 1, 2
      ok = cdf_read(f, off, nbytes, b, dshift, pieces(ip))
      call chk(ok, trim(fname)//' '//vname//': cdf_read')
      if (.not. ok) return
      call cdf_unpack(b, dshift, xt, gx, n / (gx * n3), 1, 1, gx, n / (gx * n3), n3, got)
      same = all(transfer(got, 0_int64, n) == transfer(ref, 0_int64, n))
      call chk(same, trim(fname)//' '//vname//': whole slab bits')
      if (.not. same) then
        do i = 1, n
          if (transfer(got(i), 0_int64) /= transfer(ref(i), 0_int64)) then
            write(*,'(a,i0,a,es24.16,a,es24.16,a,i0,a,i0)') '  first difference at ', i, ': netCDF ', ref(i), &
                 '  direct ', got(i), '  dshift ', dshift, '  off ', off
            exit
          endif
        enddo
      endif
    enddo
    ! windows (odd offsets and sizes), as par_read packs them
    gy = n / (gx * n3)
    do k = 1, 3
      i0 = 1 + mod(5 * k, gx); j0 = 1 + mod(3 * k, gy)
      n1 = max(1, min(gx - i0 + 1, 7 * k)); n2 = max(1, min(gy - j0 + 1, 4 * k + 1))
      allocate(w(n1 * n2 * n3))
      call cdf_unpack(b, dshift, xt, gx, gy, i0, j0, n1, n2, n3, w)
      same = .true.
      do ip = 1, n3
        do j = 1, n2
          do i = 1, n1
            if (transfer(w(((ip - 1) * n2 + (j - 1)) * n1 + i), 0_int64) /= &
                transfer(ref(((ip - 1) * gy + (j0 + j - 2)) * gx + i0 + i - 1), 0_int64)) same = .false.
          enddo
        enddo
      enddo
      call chk(same, trim(fname)//' '//vname//': window bits')
      deallocate(w)
    enddo
  end subroutine one_var

  subroutine check_file()
    ! test_cdf_direct --file <path> <var> ...: the last record (or the whole
    ! variable) of each named variable, direct against nf90_get_var, with the
    ! read and unpack times (a check of a real file, e.g. a model restart)
    character(len=1024) :: path, vname
    character(len=256) :: msg
    type(cdf_file_t) :: f
    type(cdf_buf_t) :: b
    integer :: ncid, varid, ndims, dimids(8), dl(8), m, k, ia, xt, gx, gy, n, n3, nrec
    integer :: start(8), cnt(8), c0, c1, c2, rate
    integer(int64) :: off, nbytes, dshift
    real(real64), allocatable :: ref(:), got(:)
    logical :: ok
    call get_command_argument(2, path)
    ok = cdf_open(trim(path), f, msg)
    write(*,'(3a,l1,3a,i0,a,i0,a,i0,a,i0)') 'file ', trim(path), ': cdf_open ', ok, ' ', trim(msg), &
         '  version ', f%version, '  vars ', f%nvars, '  numrecs ', f%numrecs, '  recsize ', f%recsize
    if (.not. ok) return
    call nc(nf90_open(trim(path), nf90_nowrite, ncid), 'open')
    do ia = 3, command_argument_count()
      call get_command_argument(ia, vname)
      call nc(nf90_inq_varid(ncid, trim(vname), varid), 'varid '//trim(vname))
      call nc(nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids), 'inq')
      do k = 1, ndims
        call nc(nf90_inquire_dimension(ncid, dimids(k), len=dl(k)), 'dim')
      enddo
      call nc(nf90_inquire(ncid, unlimitedDimId=k), 'unlimited')
      nrec = 0
      if (k > 0) call nc(nf90_inquire_dimension(ncid, k, len=nrec), 'numrecs')
      m = ndims; start(1:m) = 1; cnt(1:m) = dl(1:m)
      if (f%v(findv(f, trim(vname)))%isrec) then
        start(m) = nrec; cnt(m) = 1
      endif
      call system_clock(c0, rate)
      ok = cdf_locate(f, trim(vname), m, start(1:m), cnt(1:m), off, nbytes, xt, gx, gy)
      call chk(ok, trim(vname)//': located')
      if (.not. ok) cycle
      ok = cdf_read(f, off, nbytes, b, dshift, 268435456_int64)
      call chk(ok, trim(vname)//': read')
      call system_clock(c1)
      n = int(nbytes / (f%v(findv(f, trim(vname)))%xsz))
      n3 = n / (gx * gy)
      allocate(got(n), ref(n))
      call cdf_unpack(b, dshift, xt, gx, gy, 1, 1, gx, gy, n3, got)
      call system_clock(c2)
      call nc(nf90_get_var(ncid, varid, ref, start=start(1:m), count=cnt(1:m)), 'get')
      call chk(all(transfer(got, 0_int64, n) == transfer(ref, 0_int64, n)), trim(vname)//': bits')
      write(*,'(2a,i0,a,f8.3,a,f8.3,a,f8.3,a)') trim(vname), ': ', nbytes, ' bytes  read ', real(c1 - c0) / rate, &
           ' s  unpack ', real(c2 - c1) / rate, ' s  (', real(nbytes, real64) / max(1, c2 - c1) * rate / 1.0e9, &
           ' GB/s)'
      deallocate(got, ref)
    enddo
    write(*,'(a,i0,a,i0)') 'check_file: checks ', ncheck, '  failures ', nfail
  end subroutine check_file

  integer function findv(f, vname)
    type(cdf_file_t), intent(in) :: f
    character(len=*), intent(in) :: vname
    integer :: i
    findv = 0
    do i = 1, f%nvars
      if (f%v(i)%name == vname) findv = i
    enddo
  end function findv

  subroutine refuse_nc4(name)
    character(len=*), intent(in) :: name
    character(len=600) :: path
    character(len=256) :: msg
    type(cdf_file_t) :: f
    logical :: ok
    path = trim(dir)//'/'//name
    call make_file(trim(path), nf90_netcdf4, 0)
    ok = cdf_open(trim(path), f, msg)
    call chk(.not. ok, trim(name)//': netCDF-4 refused')
    write(*,'(3a)') trim(name), ': refused: ', trim(msg)
    call cdf_close(f)
  end subroutine refuse_nc4

end program test_cdf_direct

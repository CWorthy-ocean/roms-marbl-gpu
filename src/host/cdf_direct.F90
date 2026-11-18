module cdf_direct
  ! Direct (O_DIRECT) reads of classic-format netCDF files: CDF-1 (classic),
  ! CDF-2 (64-bit offset) and CDF-5 (64-bit data).
  !
  ! The Lustre client's buffered read path gives a node about 1.3 GB/s on a
  ! large file, however many ranks read; O_DIRECT reads bypass the page cache
  ! and its read-ahead and reach 12-17 GB/s on two Derecho GPU nodes
  ! (GPU_IO_BENCH2.md). netCDF cannot open a file with O_DIRECT, so this
  ! module reads the bytes itself:
  !   - cdf_open parses the file's header (the format is documented and
  !     small: dimensions, attributes, variables with their type, shape and
  !     byte offset) and opens the file a second time with O_DIRECT;
  !   - cdf_locate turns a netCDF hyperslab (start, count) of one variable
  !     into a contiguous byte range, or says it cannot (then the caller
  !     reads that variable through netCDF);
  !   - cdf_read reads the page-aligned range around it into an aligned
  !     buffer (pread in pieces);
  !   - cdf_unpack converts a window of it from big-endian to host order and
  !     to real(8), as nf90_get_var into a real(8) array (or into an integer
  !     array and then real(), as par_read's read_full does for integers)
  !     would give, bit for bit.
  ! ssize_t is taken as C long (LP64 Linux; nvfortran 25.9 has no c_ssize_t).
  ! Only plain POSIX calls (open, pread, lseek, close, posix_memalign, free)
  ! through ISO_C_BINDING; no new library. Linux only (O_DIRECT's value is
  ! architecture-specific); elsewhere cdf_open says it is not available.
  ! Compiled on its own (Makefile, like frc_reader.F90): plain host code with
  ! external C calls, kept out of nvfortran's main command.
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
  implicit none
  private
  public :: cdf_file_t, cdf_buf_t, cdf_open, cdf_close, cdf_locate, cdf_read, cdf_unpack, cdf_buf_free
  public :: cdf_progress_if

  ! netCDF external types this module converts; others are left to netCDF
  integer, parameter, public :: CDF_BYTE = 1, CDF_CHAR = 2, CDF_SHORT = 3, CDF_INT = 4, CDF_FLOAT = 5, &
                                CDF_DOUBLE = 6
  integer, parameter :: MAXD = 8            ! dimensions of a variable handled here (more: netCDF reads it)
  integer, parameter :: MAXNAME = 256       ! NC_MAX_NAME

  type :: cdf_var_t
    character(len=MAXNAME) :: name = ''
    integer :: xtype = 0, ndims = 0, xsz = 0
    integer(int64) :: dimf(MAXD) = 1        ! lengths in Fortran order (record dimension last: numrecs)
    logical :: isrec = .false., ok = .false.
    integer(int64) :: begin = 0, vlen = 0   ! vlen: bytes per record (or whole), padded as netCDF does
    integer(int64) :: vraw = 0              ! the same unpadded
  end type cdf_var_t

  type :: cdf_file_t
    integer :: version = 0                  ! 1, 2 or 5
    integer(c_int) :: fd = -1               ! the O_DIRECT descriptor
    integer(int64) :: numrecs = 0, recsize = 0, fsize = 0, align = 4096
    integer :: nvars = 0
    type(cdf_var_t), allocatable :: v(:)
  end type cdf_file_t

  type :: cdf_buf_t
    type(c_ptr) :: p = c_null_ptr
    integer(int64) :: cap = 0
    integer(int8), pointer :: u8(:) => null()
    integer(int32), pointer :: u32(:) => null()
  end type cdf_buf_t

  abstract interface
    subroutine cdf_progress_if()
    end subroutine cdf_progress_if
  end interface

  integer(c_int), parameter :: O_RDONLY = 0, SEEK_END = 2, EINTR = 4
#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
  integer(c_int), parameter :: O_DIRECT = 16384             ! 0o40000
  logical, parameter :: HAVE_DIRECT = .true.
#elif defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
  integer(c_int), parameter :: O_DIRECT = 65536             ! 0o200000
  logical, parameter :: HAVE_DIRECT = .true.
#elif defined(__linux__) && defined(__powerpc64__)
  integer(c_int), parameter :: O_DIRECT = 131072            ! 0o400000
  logical, parameter :: HAVE_DIRECT = .true.
#else
  integer(c_int), parameter :: O_DIRECT = 0
  logical, parameter :: HAVE_DIRECT = .false.
#endif

  interface
    integer(c_int) function c_open(path, flags) bind(C, name='open')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value :: flags
    end function c_open
    integer(c_long) function c_pread(fd, buf, n, off) bind(C, name='pread')
      import :: c_int, c_ptr, c_size_t, c_long, c_int64_t
      integer(c_int), value :: fd
      type(c_ptr), value :: buf
      integer(c_size_t), value :: n
      integer(c_int64_t), value :: off
    end function c_pread
    integer(c_int64_t) function c_lseek(fd, off, whence) bind(C, name='lseek')
      import :: c_int, c_int64_t
      integer(c_int), value :: fd, whence
      integer(c_int64_t), value :: off
    end function c_lseek
    integer(c_int) function c_close(fd) bind(C, name='close')
      import :: c_int
      integer(c_int), value :: fd
    end function c_close
    integer(c_int) function c_posix_memalign(p, alignment, n) bind(C, name='posix_memalign')
      import :: c_int, c_ptr, c_size_t
      type(c_ptr), intent(out) :: p
      integer(c_size_t), value :: alignment, n
    end function c_posix_memalign
    subroutine c_free(p) bind(C, name='free')
      import :: c_ptr
      type(c_ptr), value :: p
    end subroutine c_free
    integer(c_int) function c_getpagesize() bind(C, name='getpagesize')
      import :: c_int
    end function c_getpagesize
#if defined(__linux__)
    type(c_ptr) function c_errno_location() bind(C, name='__errno_location')
      import :: c_ptr
    end function c_errno_location
#endif
  end interface

contains

  integer function errno()
#if defined(__linux__)
    integer(c_int), pointer :: e
    call c_f_pointer(c_errno_location(), e)
    errno = e
#else
    errno = 0
#endif
  end function errno

  integer function xsize(xtype)
    ! bytes per value of a netCDF external type (0: not a classic/CDF-5 type)
    integer, intent(in) :: xtype
    select case (xtype)
    case (1, 2, 7);  xsize = 1               ! byte, char, ubyte
    case (3, 8);     xsize = 2               ! short, ushort
    case (4, 5, 9);  xsize = 4               ! int, float, uint
    case (6, 10, 11); xsize = 8              ! double, int64, uint64
    case default;    xsize = 0
    end select
  end function xsize

  logical function cdf_open(path, f, msg) result(ok)
    ! Parse the header of a CDF-1/2/5 file and open it with O_DIRECT. On
    ! .false. msg says why (not a classic file, an unsupported header, no
    ! O_DIRECT); nothing is left open.
    character(len=*), intent(in) :: path
    type(cdf_file_t), intent(inout) :: f
    character(len=*), intent(out) :: msg
    integer(c_int) :: fd0
    integer(int64) :: nh, got
    integer(c_long) :: k
    integer(int8), allocatable, target :: h(:)
    logical :: more
    ok = .false.
    msg = ''
    call cdf_close(f)
    if (.not. HAVE_DIRECT) then
      msg = 'O_DIRECT is not available on this platform'
      return
    endif
    fd0 = c_open(trim(path)//c_null_char, O_RDONLY)
    if (fd0 < 0) then
      write(msg, '(a,i0)') 'open failed, errno ', errno()
      return
    endif
    f%fsize = c_lseek(fd0, 0_c_int64_t, SEEK_END)
    if (f%fsize < 8) then
      msg = 'file too short for a netCDF header'
      fd0 = c_close(fd0)
      return
    endif
    nh = min(f%fsize, 262144_int64)
    do
      allocate(h(nh))
      got = 0
      do while (got < nh)
        k = c_pread(fd0, c_loc(h(got + 1)), int(nh - got, c_size_t), int(got, c_int64_t))
        if (k <= 0) exit
        got = got + k
      enddo
      if (got < nh) then
        write(msg, '(a,i0)') 'header read failed, errno ', errno()
        deallocate(h)
        fd0 = c_close(fd0)
        return
      endif
      ok = parse_header(h, nh, f, msg, more)
      deallocate(h)
      if (ok .or. .not. more) exit
      if (nh >= f%fsize .or. nh >= 268435456_int64) then
        msg = 'header longer than the file or than 256 MiB'
        exit
      endif
      nh = min(f%fsize, 4 * nh)
    enddo
    fd0 = c_close(fd0)
    if (.not. ok) return
    f%align = max(4096_int64, int(c_getpagesize(), int64))
    f%fd = c_open(trim(path)//c_null_char, O_RDONLY + O_DIRECT)
    if (f%fd < 0) then
      write(msg, '(a,i0)') 'open with O_DIRECT failed, errno ', errno()
      ok = .false.
      f%fd = -1
    endif
  end function cdf_open

  subroutine cdf_close(f)
    type(cdf_file_t), intent(inout) :: f
    integer(c_int) :: ierr
    if (f%fd >= 0) ierr = c_close(f%fd)
    f%fd = -1
    if (allocated(f%v)) deallocate(f%v)
    f%nvars = 0
    f%version = 0
  end subroutine cdf_close

  logical function parse_header(h, nh, f, msg, more) result(ok)
    ! The classic header (NetCDF Users Guide, "File Format Specification";
    ! CDF-5: PnetCDF's): magic, numrecs, dimension list, global attributes,
    ! variable list. Big-endian; CDF-5 has 64-bit counts, lengths and dimids,
    ! CDF-2 and CDF-5 64-bit begins. more = the buffer ended first.
    integer(int8), intent(in) :: h(:)
    integer(int64), intent(in) :: nh
    type(cdf_file_t), intent(inout) :: f
    character(len=*), intent(inout) :: msg
    logical, intent(out) :: more
    integer(int64) :: p, tag, n, i, j, nlen, nd, recdim, vsize, prod, prevend, ndim
    integer(int64), allocatable :: dlen(:), dimid(:)
    integer :: ns, nb, xt, nrv, d
    logical :: ovf
    ok = .false.; more = .false.; ovf = .false.
    p = 0
    if (h(1) /= 67_int8 .or. h(2) /= 68_int8 .or. h(3) /= 70_int8) then
      msg = 'not a classic netCDF file (no CDF magic; netCDF-4/HDF5?)'
      return
    endif
    f%version = int(h(4))
    select case (f%version)
    case (1); ns = 4; nb = 4
    case (2); ns = 4; nb = 8
    case (5); ns = 8; nb = 8
    case default
      write(msg, '(a,i0)') 'unknown CDF version ', f%version
      return
    end select
    p = 4
    f%numrecs = getn(ns)
    if (ns == 4 .and. f%numrecs == 4294967295_int64) then
      msg = 'streaming numrecs'
      return
    endif
    ! dimensions
    tag = getn(4); n = getn(ns)
    if (ovf) goto 900
    if (.not. (tag == 0 .and. n == 0) .and. tag /= 10) then
      msg = 'bad dimension list'; return
    endif
    if (n < 0 .or. n > f%fsize / 8) then
      msg = 'bad dimension count'; return
    endif
    allocate(dlen(0:max(n, 1_int64) - 1))
    dlen = 0
    ndim = n
    recdim = -1
    do i = 0, n - 1
      nlen = getn(ns)
      if (ovf) goto 900
      if (nlen < 0 .or. nlen > MAXNAME) then
        msg = 'bad dimension name'; return
      endif
      p = p + pad4(nlen)
      dlen(i) = getn(ns)
      if (ovf) goto 900
      if (dlen(i) == 0) recdim = i
    enddo
    if (.not. skip_atts()) goto 800
    ! variables
    tag = getn(4); n = getn(ns)
    if (ovf) goto 900
    if (.not. (tag == 0 .and. n == 0) .and. tag /= 11) then
      msg = 'bad variable list'; return
    endif
    if (n < 0 .or. n > f%fsize / 16) then
      msg = 'bad variable count'; return
    endif
    if (allocated(f%v)) deallocate(f%v)
    allocate(f%v(max(n, 1_int64)))
    f%nvars = int(n)
    do i = 1, n
      nlen = getn(ns)
      if (ovf) goto 900
      if (nlen < 0 .or. nlen > MAXNAME) then
        msg = 'bad variable name'; return
      endif
      if (p + nlen > nh) goto 900
      f%v(i)%name = ''
      do j = 1, nlen
        f%v(i)%name(j:j) = achar(iand(int(h(p + j)), 255))
      enddo
      p = p + pad4(nlen)
      nd = getn(ns)
      if (ovf) goto 900
      if (nd < 0 .or. nd > 1024) then
        msg = 'bad variable rank'; return
      endif
      if (allocated(dimid)) deallocate(dimid)
      allocate(dimid(max(nd, 1_int64)))
      do j = 1, nd
        dimid(j) = getn(ns)
      enddo
      if (ovf) goto 900
      if (.not. skip_atts()) goto 800
      xt = int(getn(4))
      vsize = getn(ns)
      f%v(i)%begin = getn(nb)
      if (ovf) goto 900
      if (xsize(xt) == 0) then
        msg = 'bad variable type'; return
      endif
      f%v(i)%xtype = xt
      f%v(i)%xsz = xsize(xt)
      f%v(i)%ndims = int(nd)
      do j = 1, nd
        if (dimid(j) < 0 .or. dimid(j) >= ndim) then
          msg = 'bad dimension id'; return
        endif
      enddo
      f%v(i)%isrec = (nd > 0 .and. dimid(1) == recdim .and. recdim >= 0)
      ! netCDF's NC_var_shape: product of the non-record lengths, padded to 4 bytes
      prod = 1
      do j = 1, nd
        if (j == 1 .and. f%v(i)%isrec) cycle
        prod = prod * max(dlen(dimid(j)), 1_int64)
      enddo
      f%v(i)%vraw = prod * f%v(i)%xsz
      f%v(i)%vlen = f%v(i)%vraw
      if (mod(f%v(i)%vlen, 4_int64) /= 0) f%v(i)%vlen = f%v(i)%vlen + 4 - mod(f%v(i)%vlen, 4_int64)
      ! vsize as written must agree (CDF-1/2 store 2^32 - 1 for a larger variable)
      if (vsize /= f%v(i)%vlen .and. .not. (ns == 4 .and. vsize == 4294967295_int64)) then
        write(msg, '(3a,i0,a,i0)') 'variable ', trim(f%v(i)%name), ': vsize ', vsize, ' not ', f%v(i)%vlen
        return
      endif
      f%v(i)%ok = (nd <= MAXD .and. (xt == CDF_BYTE .or. xt == CDF_SHORT .or. xt == CDF_INT .or. &
                   xt == CDF_FLOAT .or. xt == CDF_DOUBLE))
      if (nd <= MAXD) then
        do d = 1, int(nd)
          f%v(i)%dimf(d) = dlen(dimid(nd + 1 - d))
        enddo
        if (f%v(i)%isrec) f%v(i)%dimf(nd) = f%numrecs
      endif
    enddo
    ! record size as netCDF computes it (NC_computeshapes), and the layout
    ! check: record variables follow each other, each vlen bytes long
    f%recsize = 0; nrv = 0; prevend = -1
    do i = 1, f%nvars
      if (.not. f%v(i)%isrec) cycle
      nrv = nrv + 1
      if (nrv > 1 .and. f%v(i)%begin /= prevend) then
        write(msg, '(3a)') 'record variable ', trim(f%v(i)%name), ' not where netCDF puts it'
        return
      endif
      prevend = f%v(i)%begin + f%v(i)%vlen
      f%recsize = f%recsize + f%v(i)%vlen
    enddo
    if (nrv == 1) then           ! one record variable: records are packed, not padded
      do i = 1, f%nvars
        if (f%v(i)%isrec) f%recsize = f%v(i)%vraw
      enddo
    endif
    ok = .true.
    return
800 continue
    if (ovf) goto 900
    return
900 continue
    more = .true.
    msg = 'header longer than the buffer'
  contains
    integer(int64) function getn(sz)
      ! big-endian unsigned 4- or 8-byte integer at p (0-based), advancing p
      integer, intent(in) :: sz
      integer :: q
      getn = 0
      if (p + sz > nh) then
        ovf = .true.
        p = p + sz
        return
      endif
      do q = 1, sz
        getn = ior(ishft(getn, 8), int(iand(int(h(p + q)), 255), int64))
      enddo
      p = p + sz
    end function getn
    integer(int64) function pad4(m)
      integer(int64), intent(in) :: m
      pad4 = ((m + 3) / 4) * 4
    end function pad4
    logical function skip_atts()
      ! an attribute list: tag, count, then name, type, count, values padded to 4
      integer(int64) :: t, na, a, ln, nv
      integer :: ty
      skip_atts = .false.
      t = getn(4); na = getn(ns)
      if (ovf) return
      if (.not. (t == 0 .and. na == 0) .and. t /= 12) then
        msg = 'bad attribute list'; return
      endif
      if (na < 0 .or. na > f%fsize / 8) then
        msg = 'bad attribute count'; return
      endif
      do a = 1, na
        ln = getn(ns)
        if (ovf) return
        if (ln < 0 .or. ln > MAXNAME) then
          msg = 'bad attribute name'; return
        endif
        p = p + pad4(ln)
        ty = int(getn(4))
        nv = getn(ns)
        if (ovf) return
        if (xsize(ty) == 0 .or. nv < 0 .or. nv > f%fsize) then
          msg = 'bad attribute'; return
        endif
        p = p + pad4(nv * xsize(ty))
        if (p > nh) then
          ovf = .true.; return
        endif
      enddo
      skip_atts = .true.
    end function skip_atts
  end function parse_header

  logical function cdf_locate(f, name, m, start, cnt, off, nbytes, xtype, gx, gy) result(ok)
    ! The bytes of the hyperslab start(1:m), cnt(1:m) (Fortran order, as for
    ! nf90_get_var) of variable name, if they are one contiguous range of the
    ! file that this module converts: off, nbytes, the type, and the first
    ! two lengths of the variable (gx, gy). .false.: read it through netCDF.
    type(cdf_file_t), intent(in) :: f
    character(len=*), intent(in) :: name
    integer, intent(in) :: m, start(m), cnt(m)
    integer(int64), intent(out) :: off, nbytes
    integer, intent(out) :: xtype, gx, gy
    integer :: iv, i, d, pl, nd
    integer(int64) :: stride, e0, ext
    ok = .false.
    off = 0; nbytes = 0; xtype = 0; gx = 0; gy = 0
    iv = 0
    do i = 1, f%nvars
      if (f%v(i)%name == name) then
        iv = i
        exit
      endif
    enddo
    if (iv == 0) return
    if (.not. f%v(iv)%ok) return
    nd = f%v(iv)%ndims
    if (m /= nd .or. nd < 1) return
    do d = 1, nd
      ext = f%v(iv)%dimf(d)
      if (start(d) < 1 .or. cnt(d) < 1 .or. int(start(d), int64) + cnt(d) - 1 > ext) return
    enddo
    ! contiguous: every dimension faster than the last one with a count > 1
    ! is whole, and the record dimension (if any) has a count of 1
    pl = 1
    do d = 1, nd
      if (cnt(d) > 1) pl = d
    enddo
    do d = 1, pl - 1
      if (start(d) /= 1 .or. int(cnt(d), int64) /= f%v(iv)%dimf(d)) return
    enddo
    if (f%v(iv)%isrec .and. cnt(nd) /= 1) return
    e0 = 0; stride = 1
    do d = 1, nd
      if (f%v(iv)%isrec .and. d == nd) exit
      e0 = e0 + (start(d) - 1) * stride
      stride = stride * f%v(iv)%dimf(d)
    enddo
    off = f%v(iv)%begin + e0 * f%v(iv)%xsz
    if (f%v(iv)%isrec) off = off + int(start(nd) - 1, int64) * f%recsize
    nbytes = f%v(iv)%xsz
    do d = 1, nd
      nbytes = nbytes * cnt(d)
    enddo
    if (off < 0 .or. off + nbytes > f%fsize) return          ! truncated file
    if (f%v(iv)%xsz >= 4 .and. mod(off, 4_int64) /= 0) return
    xtype = f%v(iv)%xtype
    gx = int(f%v(iv)%dimf(1))
    gy = 1
    if (nd >= 2) gy = int(f%v(iv)%dimf(2))
    ok = .true.
  end function cdf_locate

  logical function cdf_read(f, off, nbytes, b, dshift, piece, progress) result(ok)
    ! Bytes off .. off+nbytes-1 of the file into b, read with O_DIRECT as the
    ! enclosing aligned range in pieces of at most piece bytes; the data start
    ! at byte dshift (0-based) of b. progress (optional) is called after
    ! every piece (the caller's MPI progress). .false.: a read failed.
    type(cdf_file_t), intent(in) :: f
    integer(int64), intent(in) :: off, nbytes, piece
    type(cdf_buf_t), intent(inout) :: b
    integer(int64), intent(out) :: dshift
    procedure(cdf_progress_if), optional :: progress
    integer(int64) :: a, aoff, need, alen, done, req, pc
    integer(c_long) :: k
    ok = .false.
    a = f%align
    aoff = (off / a) * a
    dshift = off - aoff
    need = off + nbytes - aoff
    alen = ((need + a - 1) / a) * a
    pc = max(a, (piece / a) * a)
    if (f%fd < 0) return
    if (.not. buf_reserve(b, alen, a)) return
    done = 0
    do while (done < need)
      req = min(pc, alen - done)
      k = c_pread(f%fd, c_loc(b%u8(done + 1)), int(req, c_size_t), int(aoff + done, c_int64_t))
      if (k < 0) then
        if (errno() == EINTR) cycle
        return
      endif
      if (k == 0) exit                                   ! end of file
      done = done + k
      if (k < req) then
        if (done >= need) exit                           ! short read at the end of the file
        if (mod(done, a) /= 0) return                    ! cannot continue aligned
      endif
      if (present(progress)) call progress()
    enddo
    ok = (done >= need)
  end function cdf_read

  logical function buf_reserve(b, n, a) result(ok)
    type(cdf_buf_t), intent(inout) :: b
    integer(int64), intent(in) :: n, a
    integer(c_int) :: ierr
    ok = .true.
    if (b%cap >= n .and. c_associated(b%p)) return
    call cdf_buf_free(b)
    ! 2 MiB alignment (at least the page): huge-page friendly, multiple of any O_DIRECT rule
    ierr = c_posix_memalign(b%p, int(max(a, 2097152_int64), c_size_t), int(n, c_size_t))
    if (ierr /= 0) then
      b%p = c_null_ptr
      ok = .false.
      return
    endif
    b%cap = n
    call c_f_pointer(b%p, b%u8, [n])
    call c_f_pointer(b%p, b%u32, [n / 4])
  end function buf_reserve

  subroutine cdf_buf_free(b)
    type(cdf_buf_t), intent(inout) :: b
    if (c_associated(b%p)) call c_free(b%p)
    b%p = c_null_ptr
    b%cap = 0
    nullify(b%u8, b%u32)
  end subroutine cdf_buf_free

  elemental integer(int32) function bswap32(x)
    integer(int32), intent(in) :: x
    bswap32 = ior(ior(ishft(x, 24), iand(ishft(x, 8), int(z'00FF0000', int32))), &
                  ior(iand(ishft(x, -8), int(z'0000FF00', int32)), iand(ishft(x, -24), int(z'000000FF', int32))))
  end function bswap32

  elemental integer(int64) function bswap64(x)
    integer(int64), intent(in) :: x
    integer(int64), parameter :: M16 = int(z'0000FFFF0000FFFF', int64), M8 = int(z'00FF00FF00FF00FF', int64)
    integer(int64) :: y
    y = ior(ishft(x, 32), ishft(x, -32))
    y = ior(ishft(iand(y, M16), 16), iand(ishft(y, -16), M16))
    bswap64 = ior(ishft(iand(y, M8), 8), iand(ishft(y, -8), M8))
  end function bswap64

  subroutine cdf_unpack(b, dshift, xtype, gx, gy, i0, j0, n1, n2, n3, out)
    ! Window (i0..i0+n1-1, j0..j0+n2-1, 1..n3) of the field (gx, gy, n3) whose
    ! big-endian bytes start at byte dshift of b, into out(n1*n2*n3) as
    ! real(8) in host order (Fortran order), converted as netCDF converts.
    ! Per row: the bytes are copied as 32-bit words (the data are only 4-byte
    ! aligned in the file), then swapped in place; simple loops that the
    ! compiler vectorises (an element-wise swap-and-assemble loop ran at
    ! 0.7-0.9 GB/s).
    type(cdf_buf_t), intent(in) :: b
    integer(int64), intent(in) :: dshift
    integer, intent(in) :: xtype, gx, gy, i0, j0, n1, n2, n3
    real(real64), intent(out), target :: out(*)
    integer(int64) :: e0, boff, w, o, ntot
    integer :: i, j, kk, v
    integer(int64), pointer :: o64(:)
    integer(int32), pointer :: o32(:)
    real(real32), pointer :: t4(:)
    integer(int32), allocatable, target :: t32(:)
    ntot = int(n1, int64) * n2 * n3
    if (ntot <= 0) return
    if (xtype == CDF_DOUBLE) then
      call c_f_pointer(c_loc(out(1)), o64, [ntot])
      call c_f_pointer(c_loc(out(1)), o32, [2 * ntot])
    else
      allocate(t32(n1))
      call c_f_pointer(c_loc(t32(1)), t4, [n1])
    endif
    do kk = 1, n3
      do j = 1, n2
        e0 = (int(kk - 1, int64) * gy + (j0 + j - 2)) * gx + (i0 - 1)
        o = (int(kk - 1, int64) * n2 + (j - 1)) * n1
        select case (xtype)
        case (CDF_DOUBLE)
          w = (dshift + e0 * 8) / 4
          o32(2 * o + 1:2 * o + 2 * n1) = b%u32(w + 1:w + 2 * n1)
          o64(o + 1:o + n1) = bswap64(o64(o + 1:o + n1))
        case (CDF_FLOAT)
          w = (dshift + e0 * 4) / 4
          t32(1:n1) = bswap32(b%u32(w + 1:w + n1))
          out(o + 1:o + n1) = real(t4(1:n1), real64)
        case (CDF_INT)
          w = (dshift + e0 * 4) / 4
          t32(1:n1) = bswap32(b%u32(w + 1:w + n1))
          out(o + 1:o + n1) = real(t32(1:n1), real64)
        case (CDF_SHORT)
          ! unsigned 16 bits, then the sign (nvfortran 25.9 -O2 dropped the sign
          ! extension of int(int8 -> int32) inside ior(ishft(...)))
          boff = dshift + e0 * 2
          do i = 1, n1
            v = ior(ishft(iand(int(b%u8(boff + 2 * i - 1), int32), 255), 8), iand(int(b%u8(boff + 2 * i), int32), 255))
            if (v >= 32768) v = v - 65536
            out(o + i) = real(v, real64)
          enddo
        case (CDF_BYTE)
          boff = dshift + e0
          do i = 1, n1
            out(o + i) = real(int(b%u8(boff + i), int32), real64)
          enddo
        case default
          error stop 'cdf_unpack: type not converted here (cdf_locate lets netCDF read it)'
        end select
      enddo
    enddo
    if (allocated(t32)) deallocate(t32)
  end subroutine cdf_unpack

end module cdf_direct

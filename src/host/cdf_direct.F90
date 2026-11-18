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
  !
  ! Direct writes (the GPU writers' CDF5 output, io_server): a file whose
  ! header netCDF has written is
  !   - created from that header by cdf_copy_header (the header pages copied
  !     with O_DIRECT into a new file; its Lustre layout read back by
  !     cdf_layout_fd);
  !   - opened by every writer with cdf_open_w (O_RDWR + O_DIRECT, the header
  !     read with O_DIRECT: no buffered byte, no stat, no lseek);
  !   - described as segments (byte ranges of variables, one owner each);
  !     cdf_plan gives every 4-KiB page one writing segment, so that every
  !     page is written whole by one writer (pages are shared only at
  !     variable boundaries; the bytes a page's writer does not hold reach it
  !     from their owner);
  !   - written with cdf_pack_* (host order -> big-endian, converted to the
  !     file's type as netCDF converts) and cdf_write (aligned pwrite in
  !     pieces; a watchdog line on stderr while one has not returned after
  !     cdf_watchdog_set's seconds);
  !   - completed by cdf_set_numrecs (the first header page rewritten),
  !     cdf_close_w and cdf_truncate (the length netCDF computes).
  ! ssize_t is taken as C long (LP64 Linux; nvfortran 25.9 has no c_ssize_t).
  ! Only plain POSIX calls (open, pread, pwrite, lseek, close, fsync,
  ! truncate, unlink, rename, fgetxattr, gethostname, write, signal, alarm,
  ! posix_memalign, free) through ISO_C_BINDING; no new library. Linux only
  ! (O_DIRECT's value is architecture-specific); elsewhere cdf_open says it
  ! is not available.
  ! Compiled on its own (Makefile, like frc_reader.F90): plain host code with
  ! external C calls, kept out of nvfortran's main command.
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
  implicit none
  private
  public :: cdf_file_t, cdf_buf_t, cdf_open, cdf_close, cdf_locate, cdf_read, cdf_unpack, cdf_buf_free
  public :: cdf_progress_if
  public :: cdf_open_w, cdf_close_w, cdf_var_index, cdf_calcsize, cdf_copy_header, cdf_set_numrecs, cdf_write, &
            cdf_read_page, cdf_pack_r4, cdf_pack_r8, cdf_pack_i4, cdf_plan, cdf_layout_fd, cdf_truncate, &
            cdf_unlink, cdf_rename, cdf_buf_reserve, cdf_watchdog_set, cdf_hostname

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
    integer(int64) :: hdr_end = 0           ! first byte after the header (the lowest variable begin)
    integer(int64) :: begin_rec = -1        ! start of the record section (-1: no record variable)
    integer :: nvars = 0
    type(cdf_var_t), allocatable :: v(:)
    integer(int8), allocatable :: page0(:)  ! cdf_open_w: the first header page (numrecs is rewritten in it)
    character(len=1024) :: path = ''
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
  ! Linux open flags (the same on x86-64, aarch64 and ppc64), SIGALRM
  integer(c_int), parameter :: O_WRONLY = 1, O_RDWR = 2, O_CREAT = 64, O_EXCL = 128, SIGALRM = 14
  ! Lustre layout magics (lov_user_md v1 / v3, composite layout lov_comp_md_v1)
  integer(int64), parameter :: LOV_V1 = int(z'0BD10BD0', int64), LOV_V3 = int(z'0BD30BD0', int64), &
                               LOV_COMP = int(z'0BD60BD0', int64)
  ! watchdog of the direct writes: while a pwrite has not returned after
  ! wd_secs seconds (0: off), the SIGALRM handler writes wd_msg to stderr
  ! (and again every wd_secs); it does nothing else
  integer(c_int), save :: wd_secs = 60
  character(kind=c_char), target, save :: wd_msg(700)
  integer(c_size_t), save :: wd_len = 0
  logical, save :: wd_installed = .false.
  character(len=64), save :: host_name = ''
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
    integer(c_int) function c_open3(path, flags, mode) bind(C, name='open64')
      ! open with a mode (O_CREAT): open64 is open on LP64 glibc; a second binding label for the 3-argument form
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value :: flags, mode
    end function c_open3
    integer(c_long) function c_pwrite(fd, buf, n, off) bind(C, name='pwrite')
      import :: c_int, c_ptr, c_size_t, c_long, c_int64_t
      integer(c_int), value :: fd
      type(c_ptr), value :: buf
      integer(c_size_t), value :: n
      integer(c_int64_t), value :: off
    end function c_pwrite
    integer(c_int) function c_fsync(fd) bind(C, name='fsync')
      import :: c_int
      integer(c_int), value :: fd
    end function c_fsync
    integer(c_int) function c_truncate(path, n) bind(C, name='truncate')
      import :: c_int, c_char, c_int64_t
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int64_t), value :: n
    end function c_truncate
    integer(c_int) function c_unlink(path) bind(C, name='unlink')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: path(*)
    end function c_unlink
    integer(c_int) function c_rename(a, b) bind(C, name='rename')
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: a(*), b(*)
    end function c_rename
    integer(c_long) function c_fgetxattr(fd, name, val, n) bind(C, name='fgetxattr')
      import :: c_int, c_char, c_ptr, c_size_t, c_long
      integer(c_int), value :: fd
      character(kind=c_char), intent(in) :: name(*)
      type(c_ptr), value :: val
      integer(c_size_t), value :: n
    end function c_fgetxattr
    integer(c_int) function c_gethostname(name, n) bind(C, name='gethostname')
      import :: c_int, c_char, c_size_t
      character(kind=c_char) :: name(*)
      integer(c_size_t), value :: n
    end function c_gethostname
    integer(c_long) function c_write(fd, buf, n) bind(C, name='write')
      import :: c_int, c_ptr, c_size_t, c_long
      integer(c_int), value :: fd
      type(c_ptr), value :: buf
      integer(c_size_t), value :: n
    end function c_write
    integer(c_int) function c_alarm(secs) bind(C, name='alarm')
      import :: c_int
      integer(c_int), value :: secs
    end function c_alarm
    type(c_funptr) function c_signal(sig, handler) bind(C, name='signal')
      import :: c_int, c_funptr
      integer(c_int), value :: sig
      type(c_funptr), value :: handler
    end function c_signal
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
    if (allocated(f%page0)) deallocate(f%page0)
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
    ! where the header ends (the lowest begin; without variables, the parsed length) and the record section starts
    f%hdr_end = p
    f%begin_rec = -1
    do i = 1, f%nvars
      if (i == 1) f%hdr_end = f%v(i)%begin
      f%hdr_end = min(f%hdr_end, f%v(i)%begin)
      if (f%v(i)%isrec) then
        if (f%begin_rec < 0) f%begin_rec = f%v(i)%begin
        f%begin_rec = min(f%begin_rec, f%v(i)%begin)
      endif
    enddo
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

  ! ============================== direct writes ==============================

  subroutine cdf_hostname(name)
    character(len=*), intent(out) :: name
    character(kind=c_char) :: h(256)
    integer :: i
    name = ''
    h = c_null_char
    if (c_gethostname(h, 255_c_size_t) /= 0) return
    do i = 1, min(len(name), 255)
      if (h(i) == c_null_char) exit
      name(i:i) = h(i)
    enddo
  end subroutine cdf_hostname

  subroutine cdf_watchdog_set(secs)
    ! seconds before a pwrite that has not returned is reported (0: no watchdog)
    integer, intent(in) :: secs
    wd_secs = int(max(0, secs), c_int)
  end subroutine cdf_watchdog_set

  subroutine wd_handler(sig) bind(C)
    ! SIGALRM: the armed text to stderr, then again after wd_secs (write and alarm only)
    integer(c_int), value :: sig
    integer(c_long) :: k
    integer(c_int) :: r
    if (wd_len > 0) k = c_write(2_c_int, c_loc(wd_msg), wd_len)
    if (wd_secs > 0) r = c_alarm(wd_secs)
  end subroutine wd_handler

  subroutine wd_arm(text)
    character(len=*), intent(in) :: text
    type(c_funptr) :: old
    integer :: i, n
    integer(c_int) :: r
    if (wd_secs <= 0) return
    if (.not. wd_installed) then
      old = c_signal(SIGALRM, c_funloc(wd_handler))
      wd_installed = .true.
    endif
    n = min(len_trim(text), size(wd_msg) - 1)
    wd_len = 0                     ! the handler never sees a half-written text
    do i = 1, n
      wd_msg(i) = text(i:i)
    enddo
    wd_msg(n + 1) = c_new_line
    wd_len = int(n + 1, c_size_t)
    r = c_alarm(wd_secs)
  end subroutine wd_arm

  subroutine wd_disarm()
    integer(c_int) :: r
    if (wd_secs <= 0) return
    r = c_alarm(0_c_int)
    wd_len = 0
  end subroutine wd_disarm

  real(8) function wtime()
    integer(int64) :: c, r
    call system_clock(c, r)
    wtime = real(c, 8) / real(max(r, 1_int64), 8)
  end function wtime

  logical function cdf_buf_reserve(b, n) result(ok)
    ! an aligned buffer of at least n bytes (its contents are lost when it grows)
    type(cdf_buf_t), intent(inout) :: b
    integer(int64), intent(in) :: n
    ok = buf_reserve(b, max(n, 4096_int64), max(4096_int64, int(c_getpagesize(), int64)))
  end function cdf_buf_reserve

  logical function hdr_from_fd(fd, f, b, nread, msg) result(ok)
    ! Read the header of the open file fd into the aligned buffer b (from
    ! 256 KiB, growing) and parse it into f; nread: bytes read (the rest of
    ! b zero). Works on an O_DIRECT descriptor (aligned sizes and buffer).
    integer(c_int), intent(in) :: fd
    type(cdf_file_t), intent(inout) :: f
    type(cdf_buf_t), intent(inout) :: b
    integer(int64), intent(out) :: nread
    character(len=*), intent(inout) :: msg
    integer(int64) :: nh, got, a
    integer(c_long) :: k
    logical :: more
    ok = .false.
    nread = 0
    a = max(4096_int64, int(c_getpagesize(), int64))
    nh = 262144_int64
    do
      if (.not. buf_reserve(b, nh, a)) then
        msg = 'no memory for the header'
        return
      endif
      got = 0
      do while (got < nh)
        k = c_pread(fd, c_loc(b%u8(got + 1)), int(nh - got, c_size_t), int(got, c_int64_t))
        if (k < 0) then
          if (errno() == EINTR) cycle
          write(msg, '(a,i0)') 'header read failed, errno ', errno()
          return
        endif
        if (k == 0) exit
        got = got + k
      enddo
      if (got < nh) b%u8(got + 1:nh) = 0_int8
      nread = got
      ok = parse_header(b%u8(1:nh), nh, f, msg, more)
      if (ok .or. .not. more) return
      if (got < nh .or. nh >= 268435456_int64) then
        msg = 'header longer than the file or than 256 MiB'
        return
      endif
      nh = 4 * nh
    enddo
  end function hdr_from_fd

  logical function cdf_open_w(path, f, direct, msg) result(ok)
    ! Open path (a classic file another rank created with cdf_copy_header)
    ! for direct writes: O_RDWR (+ O_DIRECT when direct), the header read
    ! through that descriptor and parsed; f%page0 keeps the first page.
    ! Refused unless the header and the record section start on page
    ! boundaries. No buffered I/O, no stat, no lseek.
    character(len=*), intent(in) :: path
    type(cdf_file_t), intent(inout) :: f
    logical, intent(in) :: direct
    character(len=*), intent(out) :: msg
    type(cdf_buf_t) :: b
    integer(int64) :: a, nread
    integer(c_int) :: flags, ierr
    ok = .false.
    msg = ''
    call cdf_close(f)
    if (direct .and. .not. HAVE_DIRECT) then
      msg = 'O_DIRECT is not available on this platform'
      return
    endif
    a = max(4096_int64, int(c_getpagesize(), int64))
    f%align = a
    flags = O_RDWR
    if (direct) flags = flags + O_DIRECT
    f%fd = c_open(trim(path)//c_null_char, flags)
    if (f%fd < 0) then
      write(msg, '(a,i0)') 'open for writing failed, errno ', errno()
      f%fd = -1
      return
    endif
    f%path = path
    f%fsize = huge(1_int64) / 4         ! unknown (no stat): only the parser's sanity bounds use it
    ok = hdr_from_fd(f%fd, f, b, nread, msg)
    if (ok) then
      if (mod(f%hdr_end, a) /= 0 .or. (f%begin_rec >= 0 .and. mod(max(f%begin_rec, 0_int64), a) /= 0)) then
        write(msg, '(a,i0,a,i0,a,i0)') 'header end ', f%hdr_end, ' or record section ', f%begin_rec, &
             ' not on a page boundary of ', a
        ok = .false.
      endif
    endif
    if (ok) then
      allocate(f%page0(a))
      f%page0 = b%u8(1:a)
    else
      ierr = c_close(f%fd)
      f%fd = -1
    endif
    call cdf_buf_free(b)
  end function cdf_open_w

  logical function cdf_close_w(f, dosync) result(ok)
    ! fsync (dosync) and close a file opened by cdf_open_w
    type(cdf_file_t), intent(inout) :: f
    logical, intent(in) :: dosync
    ok = .true.
    if (f%fd >= 0) then
      if (dosync) ok = (c_fsync(f%fd) == 0)
      if (c_close(f%fd) /= 0) ok = .false.
    endif
    f%fd = -1
    call cdf_close(f)
  end function cdf_close_w

  integer function cdf_var_index(f, name) result(iv)
    ! index of variable name in f (0: not there)
    type(cdf_file_t), intent(in) :: f
    character(len=*), intent(in) :: name
    integer :: i
    iv = 0
    do i = 1, f%nvars
      if (f%v(i)%name == name) then
        iv = i
        return
      endif
    enddo
  end function cdf_var_index

  integer(int64) function cdf_calcsize(f) result(n)
    ! the length netCDF gives the file (NC_calcsize): the record section
    ! after numrecs records, or the end of the last fixed-size variable
    type(cdf_file_t), intent(in) :: f
    integer :: i
    if (f%begin_rec >= 0) then
      n = f%begin_rec + f%numrecs * f%recsize
      return
    endif
    n = f%hdr_end
    do i = 1, f%nvars
      n = max(n, f%v(i)%begin + f%v(i)%vlen)
    enddo
  end function cdf_calcsize

  logical function cdf_set_numrecs(f, n) result(ok)
    ! the record count n into the first header page (O_DIRECT, the whole page)
    type(cdf_file_t), intent(inout) :: f
    integer(int64), intent(in) :: n
    type(cdf_buf_t) :: b
    integer(int64) :: v, t
    integer :: q, ns
    ok = .false.
    if (.not. allocated(f%page0)) return
    ns = 4
    if (f%version == 5) ns = 8
    v = n
    do q = ns, 1, -1
      t = iand(v, 255_int64)
      if (t > 127) t = t - 256
      f%page0(4 + q) = int(t, int8)
      v = ishft(v, -8)
    enddo
    if (.not. buf_reserve(b, f%align, f%align)) return
    b%u8(1:f%align) = f%page0
    ok = cdf_write(f, b, 0_int64, 0_int64, f%align, f%align)
    call cdf_buf_free(b)
    if (ok) f%numrecs = n
  end function cdf_set_numrecs

  logical function cdf_write(f, b, boff, off, nbytes, piece) result(ok)
    ! Bytes boff .. boff+nbytes-1 (0-based) of b to the file at off, in
    ! pwrite pieces of at most piece bytes. With O_DIRECT, off, boff and
    ! nbytes are multiples of f%align. While a pwrite has not returned after
    ! the watchdog's seconds, a line names it (file, offset, node); one that
    ! took longer is reported when it returns.
    type(cdf_file_t), intent(in) :: f
    type(cdf_buf_t), intent(in) :: b
    integer(int64), intent(in) :: boff, off, nbytes, piece
    integer(int64) :: done, req, pc
    integer(c_long) :: k
    real(8) :: t0, dt
    character(len=700) :: line
    ok = .false.
    if (f%fd < 0) return
    if (len_trim(host_name) == 0) call cdf_hostname(host_name)
    pc = max(f%align, (piece / f%align) * f%align)
    done = 0
    do while (done < nbytes)
      req = min(pc, nbytes - done)
      if (wd_secs > 0) then
        write(line, '(a,i0,a,i0,5a,i0,a)') 'cdf_direct: WATCHDOG pwrite of ', req, ' bytes at offset ', off + done, &
             ' of ', trim(f%path), ' on ', trim(host_name), ' has not returned after ', wd_secs, ' s'
        call wd_arm(line)
      endif
      t0 = wtime()
      k = c_pwrite(f%fd, c_loc(b%u8(boff + done + 1)), int(req, c_size_t), int(off + done, c_int64_t))
      call wd_disarm()
      dt = wtime() - t0
      if (wd_secs > 0 .and. dt > real(wd_secs, 8)) then
        write(*, '(a,i0,a,i0,5a,f9.1,a)') 'cdf_direct: WARNING pwrite of ', req, ' bytes at offset ', off + done, &
             ' of ', trim(f%path), ' on ', trim(host_name), ' took ', dt, ' s'
        flush(6)
      endif
      if (k < 0) then
        if (errno() == EINTR) cycle
        write(*, '(a,i0,a,i0,3a,i0)') 'cdf_direct: pwrite of ', req, ' bytes at offset ', off + done, ' of ', &
             trim(f%path), ' failed, errno ', errno()
        flush(6)
        return
      endif
      if (k == 0) return
      done = done + k
    enddo
    ok = .true.
  end function cdf_write

  logical function cdf_read_page(f, b, boff, off, n) result(ok)
    ! n bytes (aligned) at file offset off into b at boff, with the
    ! descriptor of cdf_open_w (O_DIRECT); past the end of the file: zeros
    type(cdf_file_t), intent(in) :: f
    type(cdf_buf_t), intent(inout) :: b
    integer(int64), intent(in) :: boff, off, n
    integer(int64) :: got
    integer(c_long) :: k
    ok = .false.
    if (f%fd < 0) return
    got = 0
    do while (got < n)
      k = c_pread(f%fd, c_loc(b%u8(boff + got + 1)), int(n - got, c_size_t), int(off + got, c_int64_t))
      if (k < 0) then
        if (errno() == EINTR) cycle
        return
      endif
      if (k == 0) exit
      got = got + k
    enddo
    if (got < n) b%u8(boff + got + 1:boff + n) = 0_int8
    ok = .true.
  end function cdf_read_page

  logical function cdf_truncate(path, n) result(ok)
    character(len=*), intent(in) :: path
    integer(int64), intent(in) :: n
    ok = (c_truncate(trim(path)//c_null_char, int(n, c_int64_t)) == 0)
  end function cdf_truncate

  logical function cdf_unlink(path) result(ok)
    character(len=*), intent(in) :: path
    ok = (c_unlink(trim(path)//c_null_char) == 0)
  end function cdf_unlink

  logical function cdf_rename(a, b) result(ok)
    character(len=*), intent(in) :: a, b
    ok = (c_rename(trim(a)//c_null_char, trim(b)//c_null_char) == 0)
  end function cdf_rename

  logical function cdf_layout_fd(fd, off0, off1, cmin, smin) result(found)
    ! The Lustre layout of the open file fd over bytes off0 .. off1-1: the
    ! smallest stripe count and stripe size of the layout components that
    ! cover them (a plain layout: its stripe count and size). A count of -1
    ! (every OST) is 65535; components with count 0 (the default) are
    ! skipped. .false.: no Lustre layout (another file system) or one this
    ! does not read. From the xattr lustre.lov (little-endian lov_user_md
    ! v1/v3, or lov_comp_md_v1: 32-byte header, 48-byte entries).
    integer(c_int), intent(in) :: fd
    integer(int64), intent(in) :: off0, off1
    integer, intent(out) :: cmin
    integer(int64), intent(out) :: smin
    integer(int8), allocatable, target :: x(:)
    integer(c_long) :: k
    integer(int64) :: magic, nent, st, en, eo, m2, c
    integer :: e
    found = .false.
    cmin = 0
    smin = 0
    allocate(x(65536))
    k = c_fgetxattr(fd, 'lustre.lov'//c_null_char, c_loc(x), 65536_c_size_t)
    if (k < 32) return
    magic = le(0_int64, 4)
    if (magic == LOV_V1 .or. magic == LOV_V3) then
      smin = le(24_int64, 4)
      c = le(28_int64, 2)
      if (c == 0) return
      cmin = int(c)
      found = .true.
    elseif (magic == LOV_COMP) then
      nent = le(14_int64, 2)
      cmin = huge(1)
      smin = huge(1_int64)
      do e = 0, int(nent) - 1
        if (32 + 48 * (e + 1) > k) exit
        st = le(32_int64 + 48 * e + 8, 8)
        en = le(32_int64 + 48 * e + 16, 8)
        eo = le(32_int64 + 48 * e + 24, 4)
        if (en < 0) en = huge(1_int64)                 ! EOF
        if (en <= off0 .or. st >= off1) cycle
        if (eo < 0 .or. eo + 32 > k) cycle
        m2 = le(eo, 4)
        if (m2 /= LOV_V1 .and. m2 /= LOV_V3) cycle
        c = le(eo + 28, 2)
        if (c == 0) cycle
        cmin = min(cmin, int(c))
        smin = min(smin, le(eo + 24, 4))
        found = .true.
      enddo
      if (.not. found) then
        cmin = 0
        smin = 0
      endif
    endif
  contains
    integer(int64) function le(o, nb)
      ! little-endian nb bytes at offset o (0-based); 8 bytes all 0xff give -1
      integer(int64), intent(in) :: o
      integer, intent(in) :: nb
      integer :: q
      le = 0
      do q = nb, 1, -1
        le = ior(ishft(le, 8), iand(int(x(o + q), int64), 255_int64))
      enddo
    end function le
  end function cdf_layout_fd

  logical function cdf_copy_header(hpath, path, direct, f, cmin, smin, laid, msg) result(ok)
    ! The header netCDF wrote into the scratch file hpath (nothing else:
    ! no variable data) becomes the new file path: path is removed if it
    ! exists, created (with O_DIRECT when direct; in the directory's Lustre
    ! layout), the header pages written with O_DIRECT, and hpath removed.
    ! f: the parsed header (no descriptor). laid: a Lustre layout was read,
    ! cmin / smin over the first record (cdf_layout_fd), or over the
    ! fixed-size data when there is no record variable.
    character(len=*), intent(in) :: hpath, path
    logical, intent(in) :: direct
    type(cdf_file_t), intent(inout) :: f
    integer, intent(out) :: cmin
    integer(int64), intent(out) :: smin
    logical, intent(out) :: laid
    character(len=*), intent(out) :: msg
    type(cdf_buf_t) :: b
    type(cdf_file_t) :: w
    integer(c_int) :: fd, flags, ierr
    integer(int64) :: a, alen, nread, r0, r1
    logical :: lok
    ok = .false.
    laid = .false.
    cmin = 0
    smin = 0
    msg = ''
    call cdf_close(f)
    a = max(4096_int64, int(c_getpagesize(), int64))
    fd = c_open(trim(hpath)//c_null_char, O_RDONLY)
    if (fd < 0) then
      write(msg, '(3a,i0)') 'open ', trim(hpath), ' failed, errno ', errno()
      return
    endif
    f%fsize = huge(1_int64) / 4
    lok = hdr_from_fd(fd, f, b, nread, msg)
    ierr = c_close(fd)
    if (.not. lok) then
      call cdf_buf_free(b)
      return
    endif
    alen = ((f%hdr_end + a - 1) / a) * a
    if (alen > b%cap) then
      msg = 'header end past the bytes read'
      call cdf_buf_free(b)
      return
    endif
    if (alen > nread) b%u8(nread + 1:alen) = 0_int8
    lok = cdf_unlink(path)                      ! clobber an earlier file of that name
    flags = O_WRONLY + O_CREAT + O_EXCL
    if (direct .and. HAVE_DIRECT) flags = flags + O_DIRECT
    fd = c_open3(trim(path)//c_null_char, flags, 420_c_int)    ! mode 0644
    if (fd < 0) then
      write(msg, '(3a,i0)') 'create ', trim(path), ' failed, errno ', errno()
      call cdf_buf_free(b)
      return
    endif
    w%fd = fd
    w%align = a
    w%path = path
    lok = cdf_write(w, b, 0_int64, 0_int64, alen, alen)
    call cdf_buf_free(b)
    if (.not. lok) then
      write(msg, '(3a)') 'header write to ', trim(path), ' failed'
      ierr = c_close(fd)
      return
    endif
    if (f%begin_rec >= 0) then
      r0 = f%begin_rec
      r1 = f%begin_rec + max(f%recsize, 1_int64)
    else
      r0 = f%hdr_end
      r1 = max(cdf_calcsize(f), r0 + 1)
    endif
    laid = cdf_layout_fd(fd, r0, r1, cmin, smin)
    ierr = c_close(fd)
    w%fd = -1
    if (ierr /= 0) then
      write(msg, '(3a)') 'close of ', trim(path), ' failed'
      return
    endif
    lok = cdf_unlink(hpath)
    ok = .true.
  end function cdf_copy_header

  logical function cdf_plan(n, off, nb, a, wseg, wp1, wp2) result(ok)
    ! Segments 1..n of one write (byte ranges off(i) .. off(i)+nb(i)-1,
    ! nb(i) > 0, sorted by offset, not overlapping): every a-byte page with
    ! segment bytes is written by one segment, the one that holds the
    ! page's first segment byte. wseg(i): the segment that writes the page
    ! of segment i's first byte (i itself, or the earlier segment whose last
    ! page it is). wp1(i) .. wp2(i): the pages segment i writes (wp1 > wp2:
    ! none, a small segment inside another's page). .false.: not sorted,
    ! overlapping or empty segments.
    integer, intent(in) :: n
    integer(int64), intent(in) :: off(n), nb(n), a
    integer, intent(out) :: wseg(n)
    integer(int64), intent(out) :: wp1(n), wp2(n)
    integer :: i, lastw
    integer(int64) :: fp, lp
    logical :: first
    ok = .false.
    lastw = 0
    do i = 1, n
      if (nb(i) <= 0 .or. off(i) < 0) return
      if (i > 1) then
        if (off(i) < off(i - 1) + nb(i - 1)) return
      endif
      fp = off(i) / a
      lp = (off(i) + nb(i) - 1) / a
      first = .true.
      if (i > 1) first = (off(i - 1) + nb(i - 1) <= fp * a)   ! the previous segment ends before this page
      if (first) then
        wseg(i) = i
        wp1(i) = fp
      else
        wseg(i) = lastw
        wp1(i) = fp + 1
      endif
      wp2(i) = lp
      if (wp1(i) <= wp2(i)) then
        lastw = i
      else
        lastw = wseg(i)
      endif
    enddo
    ok = .true.
  end function cdf_plan

  subroutine swap32_copy(s, d)
    integer(int32), intent(in), contiguous :: s(:)
    integer(int32), intent(inout), contiguous :: d(:)
    integer(int64) :: i
    do i = 1, size(s, kind=int64)
      d(i) = bswap32(s(i))
    enddo
  end subroutine swap32_copy

  subroutine swap64_copy(s, d)
    ! 64-bit values given as pairs of 32-bit words (host order, little-endian
    ! words) -> big-endian bytes as 32-bit words: the halves swap places
    integer(int32), intent(in), contiguous :: s(:)
    integer(int32), intent(inout), contiguous :: d(:)
    integer(int64) :: i
    do i = 1, size(s, kind=int64) / 2
      d(2 * i - 1) = bswap32(s(2 * i))
      d(2 * i) = bswap32(s(2 * i - 1))
    enddo
  end subroutine swap64_copy

  subroutine cdf_pack_r4(src, n, xtype, b, boff)
    ! n real(4) values into b at byte boff (a multiple of 4) as the file's
    ! type xtype (float, or double as netCDF converts), big-endian
    integer(int64), intent(in) :: n, boff
    real(real32), intent(in), target :: src(*)
    integer, intent(in) :: xtype
    type(cdf_buf_t), intent(inout) :: b
    integer(int32), pointer :: s32(:)
    real(real64), allocatable, target :: t8(:)
    integer(int64), parameter :: CH = 1048576
    integer(int64) :: w, i0, m
    if (n <= 0) return
    w = boff / 4
    select case (xtype)
    case (CDF_FLOAT)
      call c_f_pointer(c_loc(src(1)), s32, [n])
      call swap32_copy(s32(1:n), b%u32(w + 1:w + n))
    case (CDF_DOUBLE)
      allocate(t8(min(n, CH)))
      call c_f_pointer(c_loc(t8(1)), s32, [2 * size(t8, kind=int64)])
      do i0 = 1, n, CH
        m = min(CH, n - i0 + 1)
        t8(1:m) = real(src(i0:i0 + m - 1), real64)
        call swap64_copy(s32(1:2 * m), b%u32(w + 2 * (i0 - 1) + 1:w + 2 * (i0 - 1) + 2 * m))
      enddo
    case default
      error stop 'cdf_pack_r4: file type not float or double'
    end select
  end subroutine cdf_pack_r4

  subroutine cdf_pack_r8(src, n, xtype, b, boff)
    ! n real(8) values into b at byte boff (a multiple of 4) as the file's
    ! type xtype (double, or float rounded as netCDF converts), big-endian
    integer(int64), intent(in) :: n, boff
    real(real64), intent(in), target :: src(*)
    integer, intent(in) :: xtype
    type(cdf_buf_t), intent(inout) :: b
    integer(int32), pointer :: s32(:)
    real(real32), allocatable, target :: t4(:)
    integer(int64), parameter :: CH = 1048576
    integer(int64) :: w, i0, m
    if (n <= 0) return
    w = boff / 4
    select case (xtype)
    case (CDF_DOUBLE)
      call c_f_pointer(c_loc(src(1)), s32, [2 * n])
      call swap64_copy(s32(1:2 * n), b%u32(w + 1:w + 2 * n))
    case (CDF_FLOAT)
      allocate(t4(min(n, CH)))
      call c_f_pointer(c_loc(t4(1)), s32, [size(t4, kind=int64)])
      do i0 = 1, n, CH
        m = min(CH, n - i0 + 1)
        t4(1:m) = real(src(i0:i0 + m - 1), real32)
        call swap32_copy(s32(1:m), b%u32(w + i0:w + i0 + m - 1))
      enddo
    case default
      error stop 'cdf_pack_r8: file type not float or double'
    end select
  end subroutine cdf_pack_r8

  subroutine cdf_pack_i4(src, n, b, boff)
    ! n default integers into b at byte boff (a multiple of 4) as netCDF int, big-endian
    integer(int64), intent(in) :: n, boff
    integer(int32), intent(in), target :: src(*)
    type(cdf_buf_t), intent(inout) :: b
    integer(int32), pointer :: s32(:)
    integer(int64) :: w
    if (n <= 0) return
    w = boff / 4
    call c_f_pointer(c_loc(src(1)), s32, [n])
    call swap32_copy(s32(1:n), b%u32(w + 1:w + n))
  end subroutine cdf_pack_i4

end module cdf_direct

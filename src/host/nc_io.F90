module nc_io
  ! Serial NetCDF-Fortran helpers. Host memory only. PIO later.
  !
  ! Output files are CDF5: nc_create defines a file as CDF5 (nf90_64bit_data),
  ! without fill, under the scratch name <path>.hdr; nc_enddef aligns the
  ! fixed-size and the record sections on 4-KiB pages; nc_close then
  !   - copies the header pages into a new file <path> with O_DIRECT and
  !     removes <path>.hdr when no data were put (the files the writer ranks
  !     fill: no buffered byte ever reaches them, cdf_direct), and reports
  !     a Lustre layout too narrow for a large record (ROMS_GPU_IO_WIDE_GB);
  !   - renames <path>.hdr to <path> when data were put (a serial file).
  ! Serial writes open the files with a 64-MiB buffer hint. Input files may
  ! be classic or netCDF-4 (read through netCDF).
  use kinds
  use netcdf
  use cdf_direct, only: cdf_file_t, cdf_copy_header, cdf_rename
  implicit none
  private

  integer, parameter :: MAXCR = 64                  ! files being defined at once
  integer :: cr_ncid(MAXCR) = -1
  logical :: cr_data(MAXCR) = .false.
  character(len=1024) :: cr_path(MAXCR) = ''
  integer, parameter :: SERIAL_BUF = 67108864       ! classic-file buffer hint for serial writes (64 MiB)

  public :: nc_io_direct

  public :: iomsg_text
  public :: nc_open_read
  public :: nc_open_write
  public :: nc_create
  public :: nc_sync
  public :: nc_def_var_f
  public :: nc_put_2d_rec_f
  public :: nc_put_2d_rec_d, nc_put_3d_rec_d
  public :: nc_put_3d_rec_f
  public :: nc_close
  public :: nc_dim_len
  public :: nc_has_dim
  public :: nc_has_var
  public :: nc_var_nrecs
  public :: nc_read_2d
  public :: nc_read_3d
  public :: nc_def_dim
  public :: nc_def_var
  public :: nc_def_var_i, nc_put_ints_rec
  public :: nc_put_vec_rec_d, nc_get_vec_rec_d
  public :: nc_enddef
  public :: nc_put_2d
  public :: nc_put_3d
  public :: nc_put_1d
  public :: nc_put_1d_rec
  public :: nc_put_2d_rec
  public :: nc_put_3d_rec
  public :: nc_put_att_text
  public :: nc_get_1d
  public :: nc_get_ints
  public :: nc_get_att_text
  public :: nc_get_att_real
  public :: nc_var_shape
  public :: nc_forcing_layout
  public :: nc_read_forcing_2d
  public :: nc_check

contains

  function iomsg_text(msg) result(s)
    ! An iomsg as text: nvfortran fills the rest of the buffer with NULs and
    ! starts some messages with a newline, which make the log binary to grep.
    character(len=*), intent(in) :: msg
    character(len=len(msg)) :: s
    integer :: i
    s = msg
    do i = 1, len(s)
      if (iachar(s(i:i)) < 32) s(i:i) = ' '
    enddo
    s = adjustl(s)
  end function iomsg_text

  subroutine nc_check(ierr, where)
    integer, intent(in) :: ierr
    character(len=*), intent(in) :: where
    if (ierr /= nf90_noerr) then
      write(*,*) 'netcdf error in ', trim(where), ': ', trim(nf90_strerror(ierr))
      flush(6)
      error stop 'nc_io'
    endif
  end subroutine nc_check

  subroutine nc_open_read(path, ncid)
    character(len=*), intent(in)  :: path
    integer,          intent(out) :: ncid
    call nc_check(nf90_open(trim(path), nf90_nowrite, ncid), 'open '//trim(path))
  end subroutine nc_open_read

  subroutine nc_open_write(path, ncid)
    character(len=*), intent(in)  :: path
    integer,          intent(out) :: ncid
    integer :: bufr
    bufr = SERIAL_BUF          ! classic files: a large buffer (0.1 GB/s per writer without it)
    call nc_check(nf90_open(trim(path), nf90_write, ncid, chunksize=bufr), 'openw '//trim(path))
  end subroutine nc_open_write

  logical function nc_io_direct()
    ! ROMS_GPU_IO_DIRECT: 1 (default) O_DIRECT writes of CDF5 output; 0 the same byte ranges buffered
    character(len=16) :: val
    integer :: stat
    nc_io_direct = .true.
    call get_environment_variable('ROMS_GPU_IO_DIRECT', val, status=stat)
    if (stat == 0) nc_io_direct = (trim(adjustl(val)) /= '0')
  end function nc_io_direct

  integer function cr_slot(ncid)
    ! the nc_create table entry of ncid (0: not a CDF5 file being created)
    integer, intent(in) :: ncid
    integer :: i
    cr_slot = 0
    do i = 1, MAXCR
      if (cr_ncid(i) == ncid .and. ncid >= 0) then
        cr_slot = i
        return
      endif
    enddo
  end function cr_slot

  subroutine mark_data(ncid)
    ! data put into a file still under its scratch name: nc_close renames it
    integer, intent(in) :: ncid
    integer :: i
    i = cr_slot(ncid)
    if (i > 0) cr_data(i) = .true.
  end subroutine mark_data

  subroutine nc_create(path, ncid)
    ! A CDF5 file without fill under the scratch name <path>.hdr (nc_close
    ! gives it its name).
    character(len=*), intent(in)  :: path
    integer,          intent(out) :: ncid
    integer :: i, old
    call nc_check(nf90_create(trim(path)//'.hdr', ior(nf90_clobber, nf90_64bit_data), ncid), &
                  'create '//trim(path)//'.hdr')
    call nc_check(nf90_set_fill(ncid, nf90_nofill, old), 'set_fill '//trim(path))
    do i = 1, MAXCR
      if (cr_ncid(i) < 0) then
        cr_ncid(i) = ncid
        cr_data(i) = .false.
        cr_path(i) = path
        return
      endif
    enddo
    error stop 'nc_create: too many files being defined at once (raise MAXCR)'
  end subroutine nc_create

  subroutine nc_close(ncid)
    integer, intent(in) :: ncid
    integer :: i, cmin
    integer(8) :: smin
    logical :: laid
    character(len=256) :: msg
    character(len=1024) :: path
    type(cdf_file_t) :: f
    real(8) :: wide_gb
    i = cr_slot(ncid)
    call nc_check(nf90_close(ncid), 'close')
    if (i == 0) return
    path = cr_path(i)
    cr_ncid(i) = -1
    if (cr_data(i)) then
      ! data were put (a serial file): the scratch file becomes the file
      if (.not. cdf_rename(trim(path)//'.hdr', trim(path))) then
        write(*,'(3a)') 'nc_close: rename of ', trim(path), '.hdr failed'
        error stop 'nc_close: rename'
      endif
      return
    endif
    ! header only (the writer ranks fill it): copy the header pages with O_DIRECT into a new file
    if (.not. cdf_copy_header(trim(path)//'.hdr', trim(path), nc_io_direct(), f, cmin, smin, laid, msg)) then
      write(*,'(4a)') 'nc_close: ', trim(path), ': ', trim(msg)
      error stop 'nc_close: CDF5 header copy'
    endif
    wide_gb = wide_threshold_gb()
    if (laid .and. f%begin_rec >= 0 .and. real(f%recsize, 8) >= wide_gb * 2.0d0**30) then
      if (cmin < 16 .or. smin < 4194304_8) then
        write(*,'(4a,f8.2,a,i0,a,f6.1,a)') 'nc_io: WARNING ', trim(path), ': Lustre layout too narrow for direct ', &
             'writes of its ', real(f%recsize, 8) / 2.0d0**30, '-GiB records: ', cmin, ' stripes of ', &
             real(smin, 8) / 1048576.0d0, ' MiB (use lfs setstripe -c 48 -S 16M on the output directory; ' // &
             'tools/submit_gpu.sh does)'
      endif
    endif
  end subroutine nc_close

  real(8) function wide_threshold_gb()
    ! ROMS_GPU_IO_WIDE_GB: record size (GiB) from which a CDF5 output file
    ! should lie on a wide Lustre stripe (default 4; tools/gpu_layout.py)
    character(len=32) :: val
    integer :: stat
    wide_threshold_gb = 4.0d0
    call get_environment_variable('ROMS_GPU_IO_WIDE_GB', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) wide_threshold_gb
      if (stat /= 0) wide_threshold_gb = 4.0d0
    endif
  end function wide_threshold_gb

  integer function nc_dim_len(ncid, name) result(n)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: dimid
    call nc_check(nf90_inq_dimid(ncid, trim(name), dimid), 'inq_dim '//trim(name))
    call nc_check(nf90_inquire_dimension(ncid, dimid, len=n), 'dimlen '//trim(name))
  end function nc_dim_len

  logical function nc_has_dim(ncid, name) result(ok)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: dimid, ierr
    ierr = nf90_inq_dimid(ncid, trim(name), dimid)
    ok = (ierr == nf90_noerr)
  end function nc_has_dim

  logical function nc_has_var(ncid, name) result(ok)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: varid, ierr
    ierr = nf90_inq_varid(ncid, trim(name), varid)
    ok = (ierr == nf90_noerr)
  end function nc_has_var

  integer function nc_var_nrecs(ncid, name) result(n)
    ! Length of the last dimension of a variable (time on UCLA files).
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: varid, ndims, dimids(nf90_max_var_dims)
    n = 1
    if (.not. nc_has_var(ncid, name)) return
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids), &
                  'ndims '//trim(name))
    if (ndims >= 1) then
      call nc_check(nf90_inquire_dimension(ncid, dimids(ndims), len=n), &
                    'var dim '//trim(name))
    endif
    if (n < 1) n = 1
  end function nc_var_nrecs

  subroutine nc_read_2d(ncid, name, dest, nxi, neta, rec, i0f, j0f)
    ! dest is the model slice that receives file(i0f:i0f+nxi-1, j0f:j0f+neta-1)
    ! (i0f, j0f default to 1: the file origin; a tile passes its global offset).
    integer,          intent(in)    :: ncid, nxi, neta
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(nxi, neta)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: varid, ndims, xtype, i0, j0
    integer, allocatable :: start(:), cnt(:), ibuf(:,:)
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f

    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, xtype=xtype, ndims=ndims), 'ndims '//trim(name))
    if (present(rec) .and. ndims >= 3) then
      allocate(start(3), cnt(3))
      start = (/i0, j0, rec/)
      cnt   = (/nxi, neta, 1/)
    elseif (present(rec) .and. ndims == 2 .and. neta == 1) then
      allocate(start(2), cnt(2))
      start = (/i0, rec/)
      cnt   = (/nxi, 1/)
    else
      allocate(start(2), cnt(2))
      start = (/i0, j0/)
      cnt   = (/nxi, neta/)
    endif
    if (xtype == nf90_int .or. xtype == nf90_short .or. xtype == nf90_byte) then
      allocate(ibuf(nxi, neta))
      call nc_check(nf90_get_var(ncid, varid, ibuf, start=start, count=cnt), 'get '//trim(name))
      dest = real(ibuf, rp)
      deallocate(ibuf)
    else
      call nc_check(nf90_get_var(ncid, varid, dest, start=start, count=cnt), 'get '//trim(name))
    endif
    deallocate(start, cnt)
  end subroutine nc_read_2d

  subroutine nc_read_3d(ncid, name, dest, nxi, neta, nz, rec, i0f, j0f)
    integer,          intent(in)    :: ncid, nxi, neta, nz
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(nxi, neta, nz)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: varid, ndims, i0, j0
    integer, allocatable :: start(:), cnt(:)
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f

    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims), 'ndims '//trim(name))
    if (present(rec) .and. ndims >= 4) then
      allocate(start(4), cnt(4))
      start = (/i0, j0, 1, rec/)
      cnt   = (/nxi, neta, nz, 1/)
    else
      allocate(start(3), cnt(3))
      start = (/i0, j0, 1/)
      cnt   = (/nxi, neta, nz/)
    endif
    call nc_check(nf90_get_var(ncid, varid, dest, start=start, count=cnt), 'get '//trim(name))
    deallocate(start, cnt)
  end subroutine nc_read_3d

  integer function nc_def_dim(ncid, name, n) result(dimid)
    integer,          intent(in) :: ncid, n
    character(len=*), intent(in) :: name
    integer :: nlen
    nlen = n
    if (nlen <= 0) nlen = nf90_unlimited
    call nc_check(nf90_def_dim(ncid, trim(name), nlen, dimid), 'def_dim '//trim(name))
  end function nc_def_dim

  integer function nc_def_var(ncid, name, dimids) result(varid)
    integer,          intent(in) :: ncid, dimids(:)
    character(len=*), intent(in) :: name
    call nc_check(nf90_def_var(ncid, trim(name), nf90_double, dimids, varid), 'def_var '//trim(name))
  end function nc_def_var

  integer function nc_def_var_i(ncid, name, dimids) result(varid)
    ! Integer variable (time_step in restart files).
    integer,          intent(in) :: ncid, dimids(:)
    character(len=*), intent(in) :: name
    call nc_check(nf90_def_var(ncid, trim(name), nf90_int, dimids, varid), 'def_var '//trim(name))
  end function nc_def_var_i

  integer function nc_def_var_f(ncid, name, dimids) result(varid)
    ! Single-precision variable (history output).
    integer,          intent(in) :: ncid, dimids(:)
    character(len=*), intent(in) :: name
    call nc_check(nf90_def_var(ncid, trim(name), nf90_float, dimids, varid), 'def_var '//trim(name))
  end function nc_def_var_f

  subroutine nc_sync(ncid)
    integer, intent(in) :: ncid
    call nc_check(nf90_sync(ncid), 'sync')
  end subroutine nc_sync

  subroutine nc_put_2d_rec_d(ncid, name, arr, n1, n2, rec, i0f, j0f)
    ! As nc_put_2d_rec_f for a contiguous (n1, n2) double block.
    integer,          intent(in) :: ncid, n1, n2, rec
    character(len=*), intent(in) :: name
    real(8),          intent(in) :: arr(n1, n2)
    integer,          intent(in), optional :: i0f, j0f
    integer :: varid, i0, j0, nd
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=nd), 'ndims '//trim(name))
    if (nd == 2) then
      ! a line along one edge, (n, time) (the sponge-tuning restart fields):
      ! the block is (n1, 1) at file position i0
      call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, rec/), &
                    count=(/n1 * n2, 1/)), 'putrec '//trim(name))
      return
    endif
    call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, rec/), &
                  count=(/n1, n2, 1/)), 'putrec '//trim(name))
  end subroutine nc_put_2d_rec_d

  subroutine nc_put_3d_rec_d(ncid, name, arr, n1, n2, n3, rec, i0f, j0f)
    integer,          intent(in) :: ncid, n1, n2, n3, rec
    character(len=*), intent(in) :: name
    real(8),          intent(in) :: arr(n1, n2, n3)
    integer,          intent(in), optional :: i0f, j0f
    integer :: varid, i0, j0
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    if (nc_timeless(ncid, varid, 3)) then     ! a variable without the record dimension (rec ignored)
      call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, 1/), count=(/n1, n2, n3/)), 'put '//trim(name))
      return
    endif
    call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, 1, rec/), &
                  count=(/n1, n2, n3, 1/)), 'putrec '//trim(name))
  end subroutine nc_put_3d_rec_d

  subroutine nc_put_2d_rec_f(ncid, name, arr, n1, n2, rec, i0f, j0f)
    ! arr is the first element of a contiguous (n1, n2) float32 block,
    ! written at file position (i0f, j0f) (default the origin).
    integer,          intent(in) :: ncid, n1, n2, rec
    character(len=*), intent(in) :: name
    real(4),          intent(in) :: arr(n1, n2)
    integer,          intent(in), optional :: i0f, j0f
    integer :: varid, i0, j0, nd
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=nd), 'ndims '//trim(name))
    if (nd == 2) then   ! a line (n, time), see nc_put_2d_rec_d
      call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, rec/), &
                    count=(/n1 * n2, 1/)), 'putrec '//trim(name))
      return
    endif
    call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, rec/), &
                  count=(/n1, n2, 1/)), 'putrec '//trim(name))
  end subroutine nc_put_2d_rec_f

  subroutine nc_put_3d_rec_f(ncid, name, arr, n1, n2, n3, rec, i0f, j0f)
    integer,          intent(in) :: ncid, n1, n2, n3, rec
    integer,          intent(in), optional :: i0f, j0f
    integer :: i0, j0
    character(len=*), intent(in) :: name
    real(4),          intent(in) :: arr(n1, n2, n3)
    integer :: varid
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    if (nc_timeless(ncid, varid, 3)) then     ! a variable without the record dimension (rec ignored)
      call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, 1/), count=(/n1, n2, n3/)), 'put '//trim(name))
      return
    endif
    call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, 1, rec/), &
                  count=(/n1, n2, n3, 1/)), 'putrec '//trim(name))
  end subroutine nc_put_3d_rec_f

  subroutine nc_enddef(ncid)
    ! A CDF5 file of nc_create: the fixed-size and the record sections start
    ! on 4-KiB pages (no page holds header and data: cdf_direct's writers
    ! write whole pages); no free space in the header (it is never redefined).
    integer, intent(in) :: ncid
    if (cr_slot(ncid) > 0) then
      call nc_check(nf90_enddef(ncid, h_minfree=0, v_align=4096, v_minfree=0, r_align=4096), 'enddef')
    else
      call nc_check(nf90_enddef(ncid), 'enddef')
    endif
  end subroutine nc_enddef

  subroutine nc_put_2d(ncid, name, arr)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: arr(:, :)
    integer :: varid
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, arr), 'put '//trim(name))
  end subroutine nc_put_2d

  subroutine nc_put_3d(ncid, name, arr)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: arr(:, :, :)
    integer :: varid
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, arr), 'put '//trim(name))
  end subroutine nc_put_3d

  subroutine nc_put_1d(ncid, name, arr)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: arr(:)
    integer :: varid
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, arr), 'put '//trim(name))
  end subroutine nc_put_1d

  subroutine nc_put_1d_rec(ncid, name, val, rec)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: val
    integer :: varid
    real(rp) :: buf(1)
    buf(1) = val
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, buf, start=(/rec/), count=(/1/)), &
                  'putrec '//trim(name))
  end subroutine nc_put_1d_rec

  subroutine nc_put_ints_rec(ncid, name, vals, rec)
    ! Integer vector vals(:) into record rec of a (n, time) variable.
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    integer,          intent(in) :: vals(:)
    integer :: varid
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, vals, start=(/1, rec/), count=(/size(vals), 1/)), &
                  'putrec '//trim(name))
  end subroutine nc_put_ints_rec

  subroutine nc_put_vec_rec_d(ncid, name, vals, rec, i0f)
    ! Double vector vals(:) into record rec of a (n, time) variable (restart scalars), at
    ! position i0f (default 1).
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(8),          intent(in) :: vals(:)
    integer,          intent(in), optional :: i0f
    integer :: varid, i0
    i0 = 1
    if (present(i0f)) i0 = i0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, vals, start=(/i0, rec/), count=(/size(vals), 1/)), &
                  'putrec '//trim(name))
  end subroutine nc_put_vec_rec_d

  subroutine nc_get_vec_rec_d(ncid, name, vals, rec)
    ! Double vector of record rec of a (n, time) variable (restart scalars).
    integer,          intent(in)  :: ncid, rec
    character(len=*), intent(in)  :: name
    real(8),          intent(out) :: vals(:)
    integer :: varid
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_get_var(ncid, varid, vals, start=(/1, rec/), count=(/size(vals), 1/)), &
                  'getrec '//trim(name))
  end subroutine nc_get_vec_rec_d

  subroutine nc_put_2d_rec(ncid, name, arr, rec, i0f, j0f)
    integer,          intent(in) :: ncid, rec
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: arr(:, :)
    integer,          intent(in), optional :: i0f, j0f
    integer :: varid, i0, j0
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, rec/), &
                  count=(/size(arr,1), size(arr,2), 1/)), 'putrec '//trim(name))
  end subroutine nc_put_2d_rec

  subroutine nc_put_3d_rec(ncid, name, arr, rec, i0f, j0f)
    integer,          intent(in) :: ncid, rec
    integer,          intent(in), optional :: i0f, j0f
    integer :: i0, j0
    character(len=*), intent(in) :: name
    real(rp),         intent(in) :: arr(:, :, :)
    integer :: varid
    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    call mark_data(ncid)
    if (nc_timeless(ncid, varid, 3)) then     ! a variable without the record dimension (rec ignored)
      call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, 1/), &
                    count=(/size(arr,1), size(arr,2), size(arr,3)/)), 'put '//trim(name))
      return
    endif
    call nc_check(nf90_put_var(ncid, varid, arr, start=(/i0, j0, 1, rec/), &
                  count=(/size(arr,1), size(arr,2), size(arr,3), 1/)), &
                  'putrec '//trim(name))
  end subroutine nc_put_3d_rec

  logical function nc_timeless(ncid, varid, nd)
    ! the variable has nd dimensions (no record dimension): IMEX_IGW's time-less restart state
    ! (CPU 73a63afb), written through the record routines with rec ignored
    integer, intent(in) :: ncid, varid, nd
    integer :: ndims
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims), 'ndims')
    nc_timeless = (ndims == nd)
  end function nc_timeless

  subroutine nc_put_att_text(ncid, name, att, val)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: name, att, val
    integer :: varid
    if (len_trim(name) == 0) then
      varid = nf90_global
    else
      call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'inq '//trim(name))
    endif
    call nc_check(nf90_put_att(ncid, varid, trim(att), trim(val)), &
                  'att '//trim(att))
  end subroutine nc_put_att_text

  subroutine nc_get_att_real(ncid, name, att, val, found)
    ! Numeric attribute of a variable (e.g. cycle_length of a time axis).
    integer,          intent(in)  :: ncid
    character(len=*), intent(in)  :: name, att
    real(rp),         intent(out) :: val
    logical,          intent(out) :: found
    integer :: varid, ierr
    val = 0.0_rp
    found = .false.
    ierr = nf90_inq_varid(ncid, trim(name), varid)
    if (ierr /= nf90_noerr) return
    ierr = nf90_get_att(ncid, varid, trim(att), val)
    found = (ierr == nf90_noerr)
    if (.not. found) val = 0.0_rp
  end subroutine nc_get_att_real

  subroutine nc_get_att_text(ncid, name, att, val)
    integer,          intent(in)  :: ncid
    character(len=*), intent(in)  :: name, att
    character(len=*), intent(out) :: val
    integer :: varid, ierr
    val = ''
    if (len_trim(name) == 0) then
      varid = nf90_global
    else
      ierr = nf90_inq_varid(ncid, trim(name), varid)
      if (ierr /= nf90_noerr) return
    endif
    ierr = nf90_get_att(ncid, varid, trim(att), val)
    if (ierr /= nf90_noerr) val = ''
  end subroutine nc_get_att_text

  subroutine nc_get_ints(ncid, name, vals, rec)
    ! Integer vector. Optional rec indexes the last dimension (time).
    integer,          intent(in)  :: ncid
    character(len=*), intent(in)  :: name
    integer,          intent(out) :: vals(:)
    integer,          intent(in), optional :: rec
    integer :: varid, ndims, n1, ntake
    integer :: dimids(nf90_max_var_dims)
    integer, allocatable :: start(:), cnt(:)
    vals = 0
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids), &
                  'ndims '//trim(name))
    if (present(rec) .and. ndims >= 2) then
      call nc_check(nf90_inquire_dimension(ncid, dimids(1), len=n1), 'dim1 '//trim(name))
      ntake = min(size(vals), n1)
      allocate(start(ndims), cnt(ndims))
      start = 1
      cnt = 1
      start(ndims) = rec
      cnt(1) = ntake
      call nc_check(nf90_get_var(ncid, varid, vals(1:ntake), start=start, count=cnt), &
                    'get '//trim(name))
      deallocate(start, cnt)
    else
      ntake = size(vals)
      call nc_check(nf90_get_var(ncid, varid, vals), 'get '//trim(name))
    endif
  end subroutine nc_get_ints

  subroutine nc_get_1d(ncid, name, arr, rec)
    integer,          intent(in)  :: ncid
    character(len=*), intent(in)  :: name
    real(rp),         intent(out) :: arr(:)
    integer,          intent(in), optional :: rec
    integer :: varid, ndims
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims), 'ndims '//trim(name))
    if (present(rec) .and. ndims >= 1 .and. size(arr) == 1) then
      call nc_check(nf90_get_var(ncid, varid, arr, start=(/rec/), count=(/1/)), 'get '//trim(name))
    else
      call nc_check(nf90_get_var(ncid, varid, arr), 'get '//trim(name))
    endif
  end subroutine nc_get_1d

  subroutine nc_var_shape(ncid, name, n1, n2, n3, nrec)
    integer,          intent(in)  :: ncid
    character(len=*), intent(in)  :: name
    integer,          intent(out) :: n1, n2, n3, nrec
    integer :: varid, ndims, dimids(nf90_max_var_dims), k, len
    n1 = 1; n2 = 1; n3 = 1; nrec = 1
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids), &
                  'shape '//trim(name))
    if (ndims < 1) return
    call nc_check(nf90_inquire_dimension(ncid, dimids(ndims), len=nrec), 'nrec '//trim(name))
    do k = 1, ndims - 1
      call nc_check(nf90_inquire_dimension(ncid, dimids(k), len=len), 'dim '//trim(name))
      if (k == 1) n1 = len
      if (k == 2) n2 = len
      if (k == 3) n3 = len
    enddo
    if (ndims == 1) then
      n1 = nrec
      nrec = 1
    endif
  end subroutine nc_var_shape

  subroutine nc_forcing_layout(ncid, vname, tname, nxi, neta, nrec, time_axis)
    ! MiniPac bulk is (time, eta, xi). Toy / some flux files are (xi, eta, time).
    ! time_axis: 1 = first file dim, 3 = last file dim, 0 = no time dim.
    integer,          intent(in)  :: ncid
    character(len=*), intent(in)  :: vname, tname
    integer,          intent(out) :: nxi, neta, nrec, time_axis
    integer :: varid, ndims, dimids(nf90_max_var_dims), dlen(3), k, nt
    nxi = 1; neta = 1; nrec = 1; time_axis = 0
    dlen = 1
    call nc_check(nf90_inq_varid(ncid, trim(vname), varid), 'var '//trim(vname))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids), &
                  'layout '//trim(vname))
    do k = 1, min(ndims, 3)
      call nc_check(nf90_inquire_dimension(ncid, dimids(k), len=dlen(k)), &
                    'dim '//trim(vname))
    enddo
    nt = 0
    if (nc_has_var(ncid, tname)) nt = nc_var_nrecs(ncid, tname)
    if (ndims >= 3 .and. nt > 0 .and. dlen(1) == nt) then
      time_axis = 1
      nrec = dlen(1)
      neta = dlen(2)
      nxi = dlen(3)
    elseif (ndims >= 3 .and. nt > 0 .and. dlen(ndims) == nt) then
      time_axis = 3
      nxi = dlen(1)
      neta = dlen(2)
      nrec = dlen(ndims)
    elseif (ndims >= 3) then
      time_axis = 3
      nxi = dlen(1)
      neta = dlen(2)
      nrec = dlen(3)
    elseif (ndims == 2) then
      nxi = dlen(1)
      neta = dlen(2)
      nrec = 1
    else
      nxi = dlen(1)
      nrec = 1
    endif
  end subroutine nc_forcing_layout

  subroutine nc_read_forcing_2d(ncid, name, dest, nxi, neta, rec, time_axis, i0f, j0f)
    integer,          intent(in)    :: ncid, nxi, neta, rec, time_axis
    integer,          intent(in), optional :: i0f, j0f
    integer :: i0, j0
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(nxi, neta)
    integer :: varid, ndims, i, j
    integer, allocatable :: start(:), cnt(:)
    real(rp), allocatable :: tmp3(:,:,:)

    i0 = 1; j0 = 1
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    call nc_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call nc_check(nf90_inquire_variable(ncid, varid, ndims=ndims), 'ndims '//trim(name))
    if (ndims >= 3 .and. time_axis == 1) then
      ! CDL (time, eta, xi) -> dest(xi, eta)
      allocate(start(3), cnt(3), tmp3(1, neta, nxi))
      start = (/rec, j0, i0/)
      cnt   = (/1, neta, nxi/)
      call nc_check(nf90_get_var(ncid, varid, tmp3, start=start, count=cnt), &
                    'get frc '//trim(name))
      do j = 1, neta
        do i = 1, nxi
          dest(i, j) = tmp3(1, j, i)
        enddo
      enddo
      deallocate(tmp3)
    else
      if (ndims >= 3) then
        allocate(start(3), cnt(3))
        start = (/i0, j0, rec/)
        cnt   = (/nxi, neta, 1/)
      else
        allocate(start(2), cnt(2))
        start = (/i0, j0/)
        cnt   = (/nxi, neta/)
      endif
      call nc_check(nf90_get_var(ncid, varid, dest, start=start, count=cnt), &
                    'get frc '//trim(name))
    endif
    deallocate(start, cnt)
  end subroutine nc_read_forcing_2d

end module nc_io

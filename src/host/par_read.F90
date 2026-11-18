#include "mem_attr.inc"
module par_read
  ! Collective, batched reads of tile windows from one NetCDF file.
  !
  ! A tile window read straight from a contiguous (classic) file costs one
  ! short row read per (j, k), and the rate collapses as tiles shrink. Here
  ! the compute ranks instead read whole variable records contiguously, one
  ! variable per rank per batch, and redistribute each tile's window over
  ! MPI. The bytes that land in a tile are exactly the bytes the window read
  ! would have produced.
  !
  ! The bytes of a whole variable record come from one of three readers
  ! (ROMS_GPU_PAR_READ): 'direct' (the default) reads a classic-format file
  ! (CDF-1, CDF-2, CDF-5) with O_DIRECT through cdf_direct, which is several
  ! times faster than buffered reads on Lustre (flush_direct); netCDF-4 files
  ! and anything cdf_direct does not recognise fall back to 'whole', the
  ! whole-variable reads through netCDF; 'mpiio' reads the windows
  ! collectively through parallel netCDF (netCDF-4 files only).
  !
  ! Usage (collective; every compute rank runs the same call sequence):
  !   call pr_begin()                 ! phase 1: pr_read_* only record requests
  !   ... pr_read_2d / pr_read_3d ...
  !   call pr_flush(ncid)             ! parallel reads + redistribution
  !   call pr_take()                  ! phase 2: pr_read_* hand the data out
  !   ... the same pr_read_2d / pr_read_3d sequence ...
  !   call pr_end()
  use kinds
  use netcdf
  use nc_io, only: nc_check
  use comm_mod, only: comm_rank, comm_size, comm_compute
  use run_timers, only: wall_time
  use decomp_mod, only: decomp
  use gpu_state, only: tpcol, ntpcol
  use cdf_direct, only: cdf_file_t, cdf_buf_t, cdf_open, cdf_close, cdf_locate, cdf_read, cdf_unpack, &
                        cdf_buf_free
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private
  public :: pr_begin, pr_flush, pr_take, pr_end, pr_taking
  public :: pr_read_2d, pr_read_3d
  public :: pr_store_2d, pr_store_3d, pr_store_3d_pt

  integer, parameter :: MAXQ = 512
  type :: ent_t
    character(len=64) :: name = ''
    integer :: n1 = 0, n2 = 0, n3 = 1, i0f = 1, j0f = 1, rec = 0
    real(rp), allocatable :: buf(:)
  end type ent_t
  type :: sb_t
    real(rp), allocatable :: b(:)
  end type sb_t
  type(ent_t), save :: q(MAXQ)
  integer :: nq = 0, inext = 0, mode = 0   ! mode 0 idle, 1 queueing, 2 taking
  ! ROMS_GPU_PAR_READ = 'mpiio': every rank reads its windows collectively
  ! through parallel NetCDF (MPI-IO) instead of the whole-variable scheme.
  ! 'direct' (default): whole variables with O_DIRECT from classic files
  ! (flush_direct), else as 'whole'. 'whole': whole variables through netCDF.
  logical :: use_mpiio = .false., use_direct = .true., mode_checked = .false.
  ! ROMS_GPU_PAR_READ_VERIFY=1: every direct read is read again through
  ! netCDF and compared bit for bit (a check; doubles the reads).
  logical :: dr_verify = .false.
  ! O_DIRECT pread size (ROMS_GPU_DIRECT_PIECE_MB, default 1024): on Derecho's Lustre the
  ! Pacific restart read took 22.7 s with 64 MiB, 11.7-13.5 with 256, 8.5-9.5 with 1024 and
  ! 9.8-10.0 with 2048 (one pread per 1.4-GB variable; the MPI progress between pieces is lost)
  integer(8) :: dr_piece = 1073741824_8
#ifdef USE_MPI
  ! requests of the previous batch, progressed between the pieces of a read
  integer, allocatable :: prog_req(:)
  integer :: prog_n = 0
#endif
  ! device staging for the stores (grown to the largest window)
#ifdef USE_CUDA
  real(rp), device, allocatable :: dstage(:)
#else
  real(rp), allocatable :: dstage(:)
#endif
  integer :: dcap = 0
  real(8) :: t_read = 0.0d0, t_comm = 0.0d0

contains

  subroutine pr_begin()
    if (mode /= 0) error stop 'pr_begin: a read set is already open'
    mode = 1
    nq = 0
  end subroutine pr_begin

  subroutine pr_take()
    if (mode /= 1) error stop 'pr_take: pr_flush not done'
    mode = 2
    inext = 0
  end subroutine pr_take

  subroutine pr_end()
    integer :: k
    if (mode == 2 .and. inext /= nq) error stop 'pr_end: not every queued variable was taken'
    do k = 1, nq
      if (allocated(q(k)%buf)) deallocate(q(k)%buf)
      q(k) = ent_t()
    enddo
    nq = 0
    inext = 0
    mode = 0
    if (allocated(dstage)) deallocate(dstage)
    dcap = 0
  end subroutine pr_end

  logical function pr_taking()
    pr_taking = (mode == 2)
  end function pr_taking

  subroutine enqueue(name, n1, n2, n3, rec, i0f, j0f)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3, rec, i0f, j0f
    if (nq >= MAXQ) error stop 'par_read: too many queued variables (raise MAXQ)'
    nq = nq + 1
    q(nq)%name = name
    q(nq)%n1 = n1; q(nq)%n2 = n2; q(nq)%n3 = n3
    q(nq)%rec = rec; q(nq)%i0f = i0f; q(nq)%j0f = j0f
  end subroutine enqueue

  integer function take_next(name, n1, n2, n3) result(k)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3
    inext = inext + 1
    k = inext
    if (k > nq) error stop 'par_read: more variables taken than were queued'
    if (trim(q(k)%name) /= trim(name) .or. q(k)%n1 /= n1 .or. q(k)%n2 /= n2 .or. q(k)%n3 /= n3) then
      write(*,'(5a)') 'par_read: phase-2 call ', trim(name), ' does not match queued ', trim(q(k)%name)
      error stop 'par_read: the two phases must issue the same sequence of reads'
    endif
    if (.not. allocated(q(k)%buf)) error stop 'par_read: variable has no data (pr_flush not run?)'
  end function take_next

  subroutine pr_read_2d(ncid, name, dest, n1, n2, rec, i0f, j0f)
    ! Window (i0f..i0f+n1-1, j0f..j0f+n2-1) of a 2-D (or 2-D + time) variable.
    integer,          intent(in)    :: ncid, n1, n2
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(n1, n2)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, i0, j0, k, i, j
    r = 0; i0 = 1; j0 = 1
    if (present(rec)) r = rec
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, 1, r, i0, j0)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, 1)
      do j = 1, n2
        do i = 1, n1
          dest(i, j) = q(k)%buf((j - 1) * n1 + i)
        enddo
      enddo
      deallocate(q(k)%buf)
    else
      error stop 'pr_read_2d: call pr_begin first'
    endif
  end subroutine pr_read_2d

  subroutine pr_read_3d(ncid, name, dest, n1, n2, n3, rec, i0f, j0f)
    integer,          intent(in)    :: ncid, n1, n2, n3
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(n1, n2, n3)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, i0, j0, k, i, j, kk
    r = 0; i0 = 1; j0 = 1
    if (present(rec)) r = rec
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, n3, r, i0, j0)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, n3)
      do kk = 1, n3
        do j = 1, n2
          do i = 1, n1
            dest(i, j, kk) = q(k)%buf(((kk - 1) * n2 + (j - 1)) * n1 + i)
          enddo
        enddo
      enddo
      deallocate(q(k)%buf)
    else
      error stop 'pr_read_3d: call pr_begin first'
    endif
  end subroutine pr_read_3d

  subroutine read_full(ncid, e, fbuf, gx, gy)
    ! One whole record of a variable, contiguous: fbuf(gx, gy, e%n3).
    integer, intent(in) :: ncid
    type(ent_t), intent(in) :: e
    real(rp), allocatable, intent(out) :: fbuf(:)
    integer, intent(out) :: gx, gy
    integer :: varid, ndims, xtype, dimids(nf90_max_var_dims), dlen(nf90_max_var_dims), k
    integer, allocatable :: start(:), cnt(:), ibuf(:)
    call nc_check(nf90_inq_varid(ncid, trim(e%name), varid), 'var '//trim(e%name))
    call nc_check(nf90_inquire_variable(ncid, varid, xtype=xtype, ndims=ndims, dimids=dimids), &
                  'ndims '//trim(e%name))
    dlen = 1
    do k = 1, ndims
      call nc_check(nf90_inquire_dimension(ncid, dimids(k), len=dlen(k)), 'dim '//trim(e%name))
    enddo
    gx = dlen(1); gy = dlen(2)
    if (e%n3 == 1) then
      if (e%rec > 0 .and. ndims >= 3) then
        allocate(start(3), cnt(3)); start = (/1, 1, e%rec/); cnt = (/gx, gy, 1/)
      else
        allocate(start(2), cnt(2)); start = (/1, 1/); cnt = (/gx, gy/)
      endif
    else
      if (dlen(3) < e%n3) error stop 'par_read: variable has fewer levels than requested'
      if (e%rec > 0 .and. ndims >= 4) then
        allocate(start(4), cnt(4)); start = (/1, 1, 1, e%rec/); cnt = (/gx, gy, e%n3, 1/)
      else
        allocate(start(3), cnt(3)); start = (/1, 1, 1/); cnt = (/gx, gy, e%n3/)
      endif
    endif
    if (e%i0f + e%n1 - 1 > gx .or. e%j0f + e%n2 - 1 > gy) error stop 'par_read: window outside the file extent'
    allocate(fbuf(gx * gy * e%n3))
    if (xtype == nf90_int .or. xtype == nf90_short .or. xtype == nf90_byte) then
      allocate(ibuf(gx * gy * e%n3))
      call nc_check(nf90_get_var(ncid, varid, ibuf, start=start, count=cnt), 'get '//trim(e%name))
      fbuf = real(ibuf, rp)
      deallocate(ibuf)
    else
      call nc_check(nf90_get_var(ncid, varid, fbuf, start=start, count=cnt), 'get '//trim(e%name))
    endif
    deallocate(start, cnt)
  end subroutine read_full

  subroutine pack_window(fbuf, gx, gy, n3, i0f, j0f, n1, n2, out)
    integer, intent(in) :: gx, gy, n3, i0f, j0f, n1, n2
    real(rp), intent(in) :: fbuf(gx, gy, n3)
    real(rp), intent(out) :: out(n1, n2, n3)
    integer :: k
    do k = 1, n3
      out(:, :, k) = fbuf(i0f:i0f + n1 - 1, j0f:j0f + n2 - 1, k)
    enddo
  end subroutine pack_window

  subroutine pr_flush(ncid)
    ! Collective: batch b hands variable b*nc + rank + 1 to this rank, which
    ! reads it whole and sends every tile its window; all ranks receive
    ! their windows of the batch's variables.
    integer, intent(in) :: ncid
    integer :: nc, me, nb, b, iv, jv, t, n, k, cur, prv
    integer :: gx(2), gy(2)
    real(8) :: c0, c1, c2
    type(sb_t) :: fbuf(2)
#ifdef USE_MPI
    integer :: comm, ierr, src
    integer, allocatable :: win(:,:,:), mine(:,:), rreq(:), sreq(:,:), nqs(:)
    type(sb_t), allocatable :: sb(:,:)
#endif
    if (mode /= 1) error stop 'pr_flush: not in the queueing phase'
    nc = comm_size(); me = comm_rank()
    if (.not. mode_checked) call read_mode(me)
#ifdef USE_MPI
    if (use_mpiio .and. nc > 1) then
      if (flush_mpiio(ncid)) return
      if (me == 0) write(*,'(a)') 'par_read: parallel (MPI-IO) open not available for this file; using whole-variable reads'
    endif
    comm = comm_compute()
    allocate(nqs(0:nc - 1))
    call MPI_Allgather(nq, 1, MPI_INTEGER, nqs, 1, MPI_INTEGER, comm, ierr)
    if (any(nqs /= nq)) error stop 'par_read: ranks queued different numbers of variables'
    deallocate(nqs)
    if (nq == 0) return
    allocate(mine(4, nq), win(4, nq, 0:nc - 1))
    do k = 1, nq
      mine(:, k) = (/q(k)%n1, q(k)%n2, q(k)%i0f, q(k)%j0f/)
    enddo
    call MPI_Allgather(mine, 4 * nq, MPI_INTEGER, win, 4 * nq, MPI_INTEGER, comm, ierr)
    deallocate(mine)
    if (use_direct .and. .not. use_mpiio) then
      if (flush_direct(ncid, win)) then
        deallocate(win)
        return
      endif
    endif
    nb = (nq + nc - 1) / nc
    ! Every window is received into its own buffer, so post all receives
    ! now; then read one variable per batch and send its windows without
    ! waiting, keeping one batch of send buffers in flight behind the reads.
    allocate(rreq(nq))
    do jv = 1, nq
      n = q(jv)%n1 * q(jv)%n2 * q(jv)%n3
      allocate(q(jv)%buf(n))
      src = jv - 1 - ((jv - 1) / nc) * nc
      call MPI_Irecv(q(jv)%buf, n, MPI_REAL8, src, jv, comm, rreq(jv), ierr)
    enddo
    allocate(sreq(0:nc - 1, 2), sb(0:nc - 1, 2))
    sreq = MPI_REQUEST_NULL
    do b = 0, nb - 1
      iv = b * nc + me + 1
      cur = mod(b, 2) + 1
      prv = 3 - cur
      c0 = wall_time()
      if (iv <= nq) call read_full(ncid, q(iv), fbuf(cur)%b, gx(cur), gy(cur))
      c1 = wall_time()
      t_read = t_read + (c1 - c0)
      if (iv <= nq) then
        do t = 0, nc - 1
          n = win(1, iv, t) * win(2, iv, t) * q(iv)%n3
          allocate(sb(t, cur)%b(n))
          call pack_window(fbuf(cur)%b, gx(cur), gy(cur), q(iv)%n3, win(3, iv, t), win(4, iv, t), &
                           win(1, iv, t), win(2, iv, t), sb(t, cur)%b)
          call MPI_Isend(sb(t, cur)%b, n, MPI_REAL8, t, iv, comm, sreq(t, cur), ierr)
        enddo
      endif
      ! the previous batch's sends must be done before its buffers go
      call MPI_Waitall(nc, sreq(:, prv), MPI_STATUSES_IGNORE, ierr)
      sreq(:, prv) = MPI_REQUEST_NULL
      do t = 0, nc - 1
        if (allocated(sb(t, prv)%b)) deallocate(sb(t, prv)%b)
      enddo
      if (allocated(fbuf(prv)%b)) deallocate(fbuf(prv)%b)
      c2 = wall_time()
      t_comm = t_comm + (c2 - c1)
    enddo
    c1 = wall_time()
    do cur = 1, 2
      call MPI_Waitall(nc, sreq(:, cur), MPI_STATUSES_IGNORE, ierr)
      do t = 0, nc - 1
        if (allocated(sb(t, cur)%b)) deallocate(sb(t, cur)%b)
      enddo
      if (allocated(fbuf(cur)%b)) deallocate(fbuf(cur)%b)
    enddo
    call MPI_Waitall(nq, rreq, MPI_STATUSES_IGNORE, ierr)
    c2 = wall_time()
    t_comm = t_comm + (c2 - c1)
    deallocate(rreq, sreq, sb, win)
    write(*,'(a,i0,a,i0,a,f7.2,a,f7.2,a)') 'par_read: rank ', me, '  vars ', nq, '  whole-variable reads ', t_read, &
         ' s  redistribution ', t_comm, ' s'
    t_read = 0.0d0; t_comm = 0.0d0
#else
    do iv = 1, nq
      call read_full(ncid, q(iv), fbuf(1)%b, gx(1), gy(1))
      n = q(iv)%n1 * q(iv)%n2 * q(iv)%n3
      allocate(q(iv)%buf(n))
      call pack_window(fbuf(1)%b, gx(1), gy(1), q(iv)%n3, q(iv)%i0f, q(iv)%j0f, q(iv)%n1, q(iv)%n2, q(iv)%buf)
      deallocate(fbuf(1)%b)
    enddo
#endif
  end subroutine pr_flush

  subroutine read_mode(me)
    ! ROMS_GPU_PAR_READ (direct | whole | mpiio), ROMS_GPU_PAR_READ_VERIFY,
    ! ROMS_GPU_DIRECT_PIECE_MB; read once.
    integer, intent(in) :: me
    character(len=16) :: val
    integer :: stat, ln, mb, ios
    call get_environment_variable('ROMS_GPU_PAR_READ', val, length=ln, status=stat)
    use_mpiio = .false.; use_direct = .true.
    if (stat == 0 .and. ln > 0) then
      select case (trim(val))
      case ('direct')
      case ('whole')
        use_direct = .false.
      case ('mpiio')
        use_mpiio = .true.; use_direct = .false.
      case default
        if (me == 0) write(*,'(3a)') 'par_read: ROMS_GPU_PAR_READ=', trim(val), &
             ' not known (direct, whole, mpiio); using direct'
      end select
    endif
    call get_environment_variable('ROMS_GPU_PAR_READ_VERIFY', val, length=ln, status=stat)
    dr_verify = (stat == 0 .and. ln > 0 .and. val(1:1) /= '0')
    call get_environment_variable('ROMS_GPU_DIRECT_PIECE_MB', val, length=ln, status=stat)
    if (stat == 0 .and. ln > 0) then
      read(val, *, iostat=ios) mb
      if (ios == 0 .and. mb >= 1) dr_piece = int(mb, 8) * 1048576_8
    endif
    mode_checked = .true.
  end subroutine read_mode

#ifdef USE_MPI
  logical function direct_slab(ncid, e, m, start, cnt) result(ok)
    ! start/count of e's record as read_full builds them (mirrors it), with
    ! the variable's netCDF type; .false. if netCDF does not know the
    ! variable or read_full would stop (then read_full reports it).
    integer, intent(in) :: ncid
    type(ent_t), intent(in) :: e
    integer, intent(out) :: m, start(4), cnt(4)
    integer :: varid, ndims, dimids(nf90_max_var_dims), dlen(nf90_max_var_dims), k, gx, gy
    ok = .false.
    m = 0; start = 1; cnt = 1
    if (nf90_inq_varid(ncid, trim(e%name), varid) /= nf90_noerr) return
    if (nf90_inquire_variable(ncid, varid, ndims=ndims, dimids=dimids) /= nf90_noerr) return
    if (ndims < 2) return
    dlen = 1
    do k = 1, ndims
      if (nf90_inquire_dimension(ncid, dimids(k), len=dlen(k)) /= nf90_noerr) return
    enddo
    gx = dlen(1); gy = dlen(2)
    if (e%n3 == 1) then
      if (e%rec > 0 .and. ndims >= 3) then
        m = 3; start(1:3) = (/1, 1, e%rec/); cnt(1:3) = (/gx, gy, 1/)
      else
        m = 2; start(1:2) = (/1, 1/); cnt(1:2) = (/gx, gy/)
      endif
    else
      if (ndims < 3) return
      if (dlen(3) < e%n3) return
      if (e%rec > 0 .and. ndims >= 4) then
        m = 4; start = (/1, 1, 1, e%rec/); cnt = (/gx, gy, e%n3, 1/)
      else
        m = 3; start(1:3) = (/1, 1, 1/); cnt(1:3) = (/gx, gy, e%n3/)
      endif
    endif
    ! read_full passes m values; netCDF then needs m == ndims, which cdf_locate checks
    if (e%i0f + e%n1 - 1 > gx .or. e%j0f + e%n2 - 1 > gy) return
    ok = .true.
  end function direct_slab

  subroutine pr_progress()
    ! MPI progress for the previous batch's messages during a read
    integer :: ierr
    logical :: done
    if (prog_n > 0) call MPI_Testall(prog_n, prog_req, done, MPI_STATUSES_IGNORE, ierr)
  end subroutine pr_progress

  logical function flush_direct(ncid, win) result(ok)
    ! Collective. As the whole-variable scheme, but every variable record is
    ! read from the file with O_DIRECT (cdf_direct) and converted while its
    ! tiles' windows are packed. Variables are dealt to the ranks largest
    ! first (round robin, so a batch holds similar sizes); in batch b each
    ! rank reads its variable while the messages of batch b-1 progress, then
    ! packs and sends. .false. (on every rank): the file is not one cdf_direct
    ! reads; nothing was read and the caller uses the whole-variable path.
    integer, intent(in) :: ncid
    integer, intent(in) :: win(:, :, 0:)
    integer :: comm, ierr, nc, me, nb, b, p, k, iv, jv, t, n, cur, prv, m, xtype, gx, gy, gx2, gy2, ln
    integer :: start(4), cnt(4), nrq(2), ndir, nnc, nbad, nver, i
    integer, allocatable :: order(:), owner(:), bvars(:, :), req(:, :)
    integer(8), allocatable :: key(:)
    integer(8) :: off, nbytes, dshift, smax, pos, nread
    logical :: lok, gok, okv
    character(len=1024) :: path
    character(len=256) :: msg
    real(8) :: c0, c1, c2, c3, t_rd, t_pk, t_wt, t_all
    type(cdf_file_t) :: f
    type(cdf_buf_t) :: rb
    type(sb_t) :: sbuf(2)
    real(rp), allocatable :: fbuf(:), vbuf(:)
    ok = .false.
    comm = comm_compute(); nc = comm_size(); me = comm_rank()
    t_all = wall_time()
    call nc_check(nf90_inq_path(ncid, ln, path), 'inq_path')
    lok = cdf_open(trim(path), f, msg)
    call MPI_Allreduce(lok, gok, 1, MPI_LOGICAL, MPI_LAND, comm, ierr)
    if (.not. gok) then
      if (me == 0) then
        if (lok) msg = 'not available on another rank'
        write(*,'(5a)') 'par_read: no direct reads of ', trim(path), ' (', trim(msg), '); whole-variable reads'
      endif
      call cdf_close(f)
      return
    endif
    ok = .true.
    if (me == 0) write(*,'(3a,i0,a,i0,a)') 'par_read: direct (O_DIRECT) reads of ', trim(path), ' (CDF-', f%version, &
         ', pieces of ', dr_piece / 1048576_8, ' MiB)'
    ! deal the variables, largest first (key: the sum of the windows, the same on every rank)
    allocate(key(nq), order(nq), owner(nq))
    do k = 1, nq
      key(k) = 0
      do t = 0, nc - 1
        key(k) = key(k) + int(win(1, k, t), 8) * win(2, k, t) * q(k)%n3
      enddo
      order(k) = k
    enddo
    do k = 2, nq                          ! stable insertion sort, descending
      iv = order(k); p = k - 1
      do while (p >= 1)
        if (key(order(p)) >= key(iv)) exit
        order(p + 1) = order(p); p = p - 1
      enddo
      order(p + 1) = iv
    enddo
    nb = (nq + nc - 1) / nc
    allocate(bvars(0:nc - 1, 0:nb - 1))
    bvars = 0
    smax = 1
    do p = 1, nq
      k = order(p)
      owner(k) = mod(p - 1, nc)
      bvars(owner(k), (p - 1) / nc) = k
      if (owner(k) == me) smax = max(smax, key(k))
    enddo
    allocate(sbuf(1)%b(smax), sbuf(2)%b(smax))
    allocate(req(2 * nc, 2), prog_req(2 * nc))
    req = MPI_REQUEST_NULL; nrq = 0
    do jv = 1, nq
      allocate(q(jv)%buf(q(jv)%n1 * q(jv)%n2 * q(jv)%n3))
    enddo
    t_rd = 0; t_pk = 0; t_wt = 0; nread = 0; ndir = 0; nnc = 0; nbad = 0; nver = 0
    do b = 0, nb - 1
      cur = mod(b, 2) + 1; prv = 3 - cur
      ! receives of this batch first, so its sends find them posted
      nrq(cur) = 0
      do t = 0, nc - 1
        jv = bvars(t, b)
        if (jv == 0) cycle
        nrq(cur) = nrq(cur) + 1
        call MPI_Irecv(q(jv)%buf, q(jv)%n1 * q(jv)%n2 * q(jv)%n3, MPI_REAL8, t, jv, comm, req(nrq(cur), cur), ierr)
      enddo
      iv = bvars(me, b)
      c0 = wall_time()
      okv = .false.
      prog_n = nrq(prv)
      if (prog_n > 0) prog_req(1:prog_n) = req(1:prog_n, prv)
      if (iv > 0) then
        if (direct_slab(ncid, q(iv), m, start, cnt)) then
          if (cdf_locate(f, trim(q(iv)%name), m, start(1:m), cnt(1:m), off, nbytes, xtype, gx, gy)) then
            okv = cdf_read(f, off, nbytes, rb, dshift, dr_piece, pr_progress)
            if (okv) then
              nread = nread + nbytes
              ndir = ndir + 1
            endif
          endif
        endif
        if (.not. okv) then
          ! not readable here (a type, layout or file end cdf_direct leaves to netCDF, or a failed read)
          call read_full(ncid, q(iv), fbuf, gx, gy)
          nnc = nnc + 1
        endif
      endif
      if (prog_n > 0) req(1:prog_n, prv) = prog_req(1:prog_n)
      prog_n = 0
      c1 = wall_time()
      t_rd = t_rd + (c1 - c0)
      if (iv > 0 .and. okv .and. dr_verify) then
        call read_full(ncid, q(iv), fbuf, gx2, gy2)
        n = gx * gy * q(iv)%n3
        allocate(vbuf(n))
        call cdf_unpack(rb, dshift, xtype, gx, gy, 1, 1, gx, gy, q(iv)%n3, vbuf)
        k = 0
        do i = 1, n
          if (transfer(vbuf(i), 0_8) /= transfer(fbuf(i), 0_8)) k = k + 1
        enddo
        if (k > 0 .or. gx2 /= gx .or. gy2 /= gy) then
          write(*,'(a,i0,3a,i0,a,i0)') 'par_read verify: rank ', me, ' ', trim(q(iv)%name), ' DIFFERS in ', k, &
               ' of ', n
          nbad = nbad + 1
        endif
        nver = nver + 1
        deallocate(vbuf)
        c1 = wall_time()
      endif
      ! pack and send this batch's windows
      if (iv > 0) then
        pos = 0
        do t = 0, nc - 1
          if (win(3, iv, t) + win(1, iv, t) - 1 > gx .or. win(4, iv, t) + win(2, iv, t) - 1 > gy) &
               error stop 'par_read: window outside the file extent'
          n = win(1, iv, t) * win(2, iv, t) * q(iv)%n3
          if (okv) then
            call cdf_unpack(rb, dshift, xtype, gx, gy, win(3, iv, t), win(4, iv, t), win(1, iv, t), win(2, iv, t), &
                            q(iv)%n3, sbuf(cur)%b(pos + 1))
          else
            call pack_window(fbuf, gx, gy, q(iv)%n3, win(3, iv, t), win(4, iv, t), win(1, iv, t), win(2, iv, t), &
                             sbuf(cur)%b(pos + 1))
          endif
          nrq(cur) = nrq(cur) + 1
          call MPI_Isend(sbuf(cur)%b(pos + 1), n, MPI_REAL8, t, iv, comm, req(nrq(cur), cur), ierr)
          pos = pos + n
        enddo
      endif
      if (allocated(fbuf)) deallocate(fbuf)
      c2 = wall_time()
      t_pk = t_pk + (c2 - c1)
      ! the previous batch is done before its send buffer is used again
      if (nrq(prv) > 0) call MPI_Waitall(nrq(prv), req(:, prv), MPI_STATUSES_IGNORE, ierr)
      req(:, prv) = MPI_REQUEST_NULL; nrq(prv) = 0
      c3 = wall_time()
      t_wt = t_wt + (c3 - c2)
    enddo
    c2 = wall_time()
    do cur = 1, 2
      if (nrq(cur) > 0) call MPI_Waitall(nrq(cur), req(:, cur), MPI_STATUSES_IGNORE, ierr)
    enddo
    c3 = wall_time()
    t_wt = t_wt + (c3 - c2)
    call cdf_close(f)
    call cdf_buf_free(rb)
    deallocate(sbuf(1)%b, sbuf(2)%b, req, key, order, owner, bvars)
    if (allocated(prog_req)) deallocate(prog_req)
    write(*,'(a,i0,a,i0,a,f7.2,a,f8.3,a,i0,a,i0,a,f7.2,a,f7.2,a,f7.2,a)') 'par_read: rank ', me, '  vars ', nq, &
         '  direct reads ', t_rd, ' s (', real(nread, 8) / 1.0d9, ' GB, ', ndir, ' direct, ', nnc, &
         ' netCDF)  unpack ', t_pk, ' s  redistribution ', t_wt, ' s  total ', c3 - t_all, ' s'
    if (dr_verify) write(*,'(a,i0,a,i0,a,i0,a)') 'par_read verify: rank ', me, '  ', nver, ' variables checked, ', &
         nbad, ' differ'
  end function flush_direct

  logical function flush_mpiio(ncid) result(ok)
    ! Collective window reads: the file is opened once more in parallel mode
    ! on the compute communicator and every queued window is read by its
    ! rank with collective access (same variable order on every rank).
    integer, intent(in) :: ncid
    integer :: pncid, varid, ndims, k, n, ierr, nvars, v
    real(8) :: c0, c1
    character(len=1024) :: path
    integer, allocatable :: start(:), cnt(:)
    call nc_check(nf90_inq_path(ncid, n, path), 'inq_path')
    c0 = wall_time()
    ok = .false.
    ierr = nf90_open(trim(path), ior(nf90_nowrite, nf90_mpiio), pncid, comm=comm_compute(), info=MPI_INFO_NULL)
    if (ierr /= nf90_noerr) return
    call nc_check(nf90_inquire(pncid, nvariables=nvars), 'inquire')
    do v = 1, nvars
      ierr = nf90_var_par_access(pncid, v, nf90_collective)
      if (ierr /= nf90_noerr) then
        ierr = nf90_close(pncid)
        return
      endif
    enddo
    ok = .true.
    do k = 1, nq
      n = q(k)%n1 * q(k)%n2 * q(k)%n3
      allocate(q(k)%buf(n))
      call nc_check(nf90_inq_varid(pncid, trim(q(k)%name), varid), 'var '//trim(q(k)%name))
      call nc_check(nf90_inquire_variable(pncid, varid, ndims=ndims), 'ndims '//trim(q(k)%name))
      if (q(k)%n3 == 1) then
        if (q(k)%rec > 0 .and. ndims >= 3) then
          allocate(start(3), cnt(3)); start = (/q(k)%i0f, q(k)%j0f, q(k)%rec/); cnt = (/q(k)%n1, q(k)%n2, 1/)
        else
          allocate(start(2), cnt(2)); start = (/q(k)%i0f, q(k)%j0f/); cnt = (/q(k)%n1, q(k)%n2/)
        endif
      else
        if (q(k)%rec > 0 .and. ndims >= 4) then
          allocate(start(4), cnt(4)); start = (/q(k)%i0f, q(k)%j0f, 1, q(k)%rec/); cnt = (/q(k)%n1, q(k)%n2, q(k)%n3, 1/)
        else
          allocate(start(3), cnt(3)); start = (/q(k)%i0f, q(k)%j0f, 1/); cnt = (/q(k)%n1, q(k)%n2, q(k)%n3/)
        endif
      endif
      call nc_check(nf90_get_var(pncid, varid, q(k)%buf, start=start, count=cnt), 'get '//trim(q(k)%name))
      deallocate(start, cnt)
    enddo
    call nc_check(nf90_close(pncid), 'close parallel')
    c1 = wall_time()
    write(*,'(a,i0,a,i0,a,f7.2,a)') 'par_read: rank ', comm_rank(), '  vars ', nq, '  collective MPI-IO window reads ', &
         (c1 - c0), ' s'
  end function flush_mpiio
#endif

  subroutine stage_to_device(k, n)
    ! The taken window k (host) into the device staging buffer.
    integer, intent(in) :: k, n
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (n > dcap) then
      if (allocated(dstage)) deallocate(dstage)
      allocate(dstage(n))
      dcap = n
    endif
#ifdef USE_CUDA
    ierr = cudaMemcpy(dstage, q(k)%buf, n, cudaMemcpyHostToDevice)
    if (ierr /= 0) error stop 'par_read: cudaMemcpy to the staging buffer failed'
#else
    dstage(1:n) = q(k)%buf(1:n)
#endif
    deallocate(q(k)%buf)
  end subroutine stage_to_device

  subroutine pr_store_2d(ncid, name, a, i0, j0, n1, n2, rec, i0f, j0f)
    ! Window (i0f.., j0f..) of the file into a(i0:i0+n1-1, j0:j0+n2-1) on the
    ! device (a is the first element of a (is:ie, js:je) field).
    integer,          intent(in) :: ncid, i0, j0, n1, n2
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, fi, fj, k, i, j, ia, ja
    r = 0; fi = 1; fj = 1
    if (present(rec)) r = rec
    if (present(i0f)) fi = i0f
    if (present(j0f)) fj = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, 1, r, fi, fj)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, 1)
      call stage_to_device(k, n1 * n2)
      ia = i0; ja = j0
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, n2
        do i = 1, n1
          a(ia + i - 1, ja + j - 1) = dstage((j - 1) * n1 + i)
        enddo
      enddo
#ifdef USE_CUDA
      if (cudaDeviceSynchronize() /= 0) error stop 'pr_store_2d: kernel failed'
#endif
    else
      error stop 'pr_store_2d: call pr_begin first'
    endif
  end subroutine pr_store_2d

  subroutine pr_store_3d(ncid, name, a, i0, j0, n1, n2, n3, rec, i0f, j0f)
    integer,          intent(in) :: ncid, i0, j0, n1, n2, n3
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, n3)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, fi, fj, k, i, j, kk, ia, ja
    r = 0; fi = 1; fj = 1
    if (present(rec)) r = rec
    if (present(i0f)) fi = i0f
    if (present(j0f)) fj = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, n3, r, fi, fj)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, n3)
      call stage_to_device(k, n1 * n2 * n3)
      ia = i0; ja = j0
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do kk = 1, n3
        do j = 1, n2
          do i = 1, n1
            a(ia + i - 1, ja + j - 1, kk) = dstage(((kk - 1) * n2 + (j - 1)) * n1 + i)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      if (cudaDeviceSynchronize() /= 0) error stop 'pr_store_3d: kernel failed'
#endif
    else
      error stop 'pr_store_3d: call pr_begin first'
    endif
  end subroutine pr_store_3d

  subroutine pr_store_3d_pt(ncid, name, a, ntr, itr, i0, j0, n1, n2, n3, rec, i0f, j0f)
    ! Tracer itr of one slot of the compressed passive array (ntr,
    ! 0:ntpcol, n3): the window is scattered through tpcol; unstored
    ! (land) points are left out, their column stays zero.
    integer,          intent(in) :: ncid, ntr, itr, i0, j0, n1, n2, n3
    character(len=*), intent(in) :: name
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, n3)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, fi, fj, k, i, j, kk, ia, ja, it, cc
    r = 0; fi = 1; fj = 1
    if (present(rec)) r = rec
    if (present(i0f)) fi = i0f
    if (present(j0f)) fj = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, n3, r, fi, fj)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, n3)
      call stage_to_device(k, n1 * n2 * n3)
      ia = i0; ja = j0; it = itr
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do kk = 1, n3
        do j = 1, n2
          do i = 1, n1
            cc = tpcol(ia + i - 1, ja + j - 1)
            if (cc > 0) a(it, cc, kk) = real(dstage(((kk - 1) * n2 + (j - 1)) * n1 + i), pp)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      if (cudaDeviceSynchronize() /= 0) error stop 'pr_store_3d_pt: kernel failed'
#endif
    else
      error stop 'pr_store_3d_pt: call pr_begin first'
    endif
  end subroutine pr_store_3d_pt

end module par_read

module frc_reader
  ! Forcing reader ranks (ROMS_GPU_FRC_READER = 1; comm_mod makes the last
  ! rank of every node a reader). A reader warms its node's page cache: it
  ! reads forcing and boundary records ahead of the model, each record whole
  ! and in file order, so that the compute ranks' own reads of it, later,
  ! come from memory instead of waiting on Lustre (a record read in pieces
  ! by several ranks, as frc_prefetch does, costs a full OST round trip per
  ! piece). The compute ranks' reads are unchanged, so the results are too.
  !
  ! Compute rank 0 posts the records (frc_reader_ahead, from frc_io and
  ! bry_io): for every series, when its upper bracket moves to record i2,
  ! the next RA_AHEAD records of its catalog. A post is (file, time axis,
  ! record); repeats are dropped. Every reader gets every post and reads,
  ! for each variable of the file whose last (slowest) dimension is that
  ! time axis' dimension, in variable order, its slab at that record. It
  ! has no GPU, no model state and no other role; it polls for posts with
  ! a short sleep, so it uses no core while idle. frc_reader_stop (compute
  ! ranks, end of run) tells the readers to quit.
  ! ROMS_GPU_FRC_TIMING=1: a FRCWARM line per record read.
  ! Compiled on its own like comm.F90 (external MPI and netCDF calls).
  use iso_c_binding, only: c_int
  use comm_mod, only: comm_rank, comm_world, comm_world_rank, comm_reader_count, comm_reader_world, &
                      is_reader_rank, is_io_rank
  use netcdf, only: nf90_open, nf90_close, nf90_nowrite, nf90_noerr, nf90_inq_varid, nf90_inquire, &
                    nf90_inquire_variable, nf90_inquire_dimension, nf90_get_var, nf90_double, &
                    nf90_short, nf90_byte, nf90_char, nf90_max_var_dims, nf90_strerror
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private

  integer, parameter :: RA_AHEAD = 2        ! records posted ahead of the upper bracket
  integer, parameter :: TAG_RDR = 7700
  integer, parameter :: PLEN = 512, TLEN = 64
  integer, parameter :: MSGLEN = 1 + 12 + TLEN + PLEN
  integer, parameter :: NSBUF = 64          ! send buffers in flight (compute rank 0)
  integer, parameter :: NRING = 128         ! recent posts remembered (repeats dropped)
  integer, parameter :: MAXF = 64           ! files kept open on a reader

  ! compute rank 0
  logical :: posting = .false., checked = .false.
  character(len=MSGLEN), save :: sbuf(NSBUF)
  integer, save :: sreq(NSBUF)
  integer :: snext = 1, nposts = 0
  character(len=PLEN), save :: ring_p(NRING)
  character(len=TLEN), save :: ring_t(NRING)
  integer, save :: ring_r(NRING) = -1
  integer :: rnext = 1

  interface
    integer(c_int) function c_usleep(us) bind(C, name='usleep')
      import :: c_int
      integer(c_int), value :: us
    end function c_usleep
  end interface

  public :: frc_reader_on, frc_reader_ahead, frc_reader_post, frc_reader_stop, frc_reader_run

contains

  logical function frc_reader_on()
    ! True on compute rank 0 when the job has forcing readers: the only rank
    ! that posts. False everywhere else, so the callers' hooks cost one test.
    if (.not. checked) then
      checked = .true.
      posting = (comm_reader_count() > 0 .and. comm_rank() == 0 .and. .not. is_io_rank() .and. .not. is_reader_rank())
#ifdef USE_MPI
      if (posting) sreq = MPI_REQUEST_NULL
#endif
    endif
    frc_reader_on = posting
  end function frc_reader_on

  subroutine frc_reader_ahead(paths, irec, ntot, cyclic, it2, tname, mark)
    ! A series' catalog (paths, irec of records 1..ntot, in time order) with
    ! upper bracket it2: post records it2+1 .. it2+RA_AHEAD (wrapping for a
    ! climatology), once per bracket (mark holds the it2 last posted).
    character(len=*), intent(in)    :: paths(:)
    integer,          intent(in)    :: irec(:)
    integer,          intent(in)    :: ntot, it2
    logical,          intent(in)    :: cyclic
    character(len=*), intent(in)    :: tname
    integer,          intent(inout) :: mark
    integer :: k, idx
    if (.not. frc_reader_on()) return
    if (ntot < 2 .or. it2 < 1 .or. it2 == mark) return
    mark = it2
    do k = 1, RA_AHEAD
      idx = it2 + k
      if (cyclic) then
        idx = mod(idx - 1, ntot) + 1
      elseif (idx > ntot) then
        exit
      endif
      call frc_reader_post(paths(idx), tname, irec(idx))
    enddo
  end subroutine frc_reader_ahead

  subroutine frc_reader_post(path, tname, rec)
    ! Post record rec (1-based, along the dimension of the time axis tname)
    ! of file path to every reader, unless it was posted recently.
    character(len=*), intent(in) :: path, tname
    integer,          intent(in) :: rec
#ifdef USE_MPI
    integer :: i, ierr
    character(len=TLEN) :: tt
    character(len=PLEN) :: pp
    if (.not. frc_reader_on()) return
    do i = 1, NRING
      if (ring_r(i) /= rec) cycle
      if (ring_t(i) /= tname) cycle
      if (ring_p(i) == path) return
    enddo
    ring_r(rnext) = rec; ring_t(rnext) = tname; ring_p(rnext) = path
    rnext = mod(rnext, NRING) + 1
    do i = 0, comm_reader_count() - 1
      call MPI_Wait(sreq(snext), MPI_STATUS_IGNORE, ierr)   ! long done: the buffer is reused NSBUF posts later
      tt = tname
      pp = path
      write(sbuf(snext), '(a1,i12,a,a)') 'R', rec, tt, pp
      call MPI_Isend(sbuf(snext), MSGLEN, MPI_CHARACTER, comm_reader_world(i), TAG_RDR, comm_world(), &
                     sreq(snext), ierr)
      snext = mod(snext, NSBUF) + 1
    enddo
    nposts = nposts + 1
#endif
  end subroutine frc_reader_post

  subroutine frc_reader_stop()
    ! End of run, every compute rank (rank 0 acts): tell the readers to quit.
#ifdef USE_MPI
    integer :: i, ierr
    character(len=MSGLEN) :: q
    if (.not. frc_reader_on()) return
    call MPI_Waitall(NSBUF, sreq, MPI_STATUSES_IGNORE, ierr)
    q = 'Q'
    do i = 0, comm_reader_count() - 1
      call MPI_Send(q, MSGLEN, MPI_CHARACTER, comm_reader_world(i), TAG_RDR, comm_world(), ierr)
    enddo
    write(*,'(a,i0,a,i0,a)') 'forcing reader: ', nposts, ' records posted to ', comm_reader_count(), ' reader rank(s)'
    posting = .false.
#endif
  end subroutine frc_reader_stop

  subroutine frc_reader_run()
    ! A reader rank's loop, until the quit post.
#ifdef USE_MPI
    integer :: ierr, stat(MPI_STATUS_SIZE), rec, ios, nfile, i, nvar, nreq
    logical :: flag, timing
    character(len=MSGLEN) :: msg
    character(len=PLEN) :: path, fpath(MAXF)
    character(len=TLEN) :: tname
    integer :: fid(MAXF)
    real(8) :: t0, t1, ttot, tmax, mb, mbtot
    character(len=8) :: val
    integer(c_int) :: rc
    call get_environment_variable('ROMS_GPU_FRC_TIMING', val, status=ios)
    timing = (ios == 0 .and. len_trim(val) > 0 .and. val(1:1) == '1')
    nfile = 0
    nreq = 0
    ttot = 0.0d0; tmax = 0.0d0; mbtot = 0.0d0
    write(*,'(a,i0)') 'forcing reader: ready on world rank ', comm_world_rank()
    flush(6)
    do
      call MPI_Iprobe(MPI_ANY_SOURCE, TAG_RDR, comm_world(), flag, stat, ierr)
      if (.not. flag) then
        rc = c_usleep(1000_c_int)
        cycle
      endif
      call MPI_Recv(msg, MSGLEN, MPI_CHARACTER, stat(MPI_SOURCE), TAG_RDR, comm_world(), stat, ierr)
      if (msg(1:1) == 'Q') exit
      read(msg, '(1x,i12,a64,a512)', iostat=ios) rec, tname, path
      if (ios /= 0) cycle
      t0 = MPI_Wtime()
      call warm_record(path, tname, rec, nvar, mb)
      t1 = MPI_Wtime()
      nreq = nreq + 1
      ttot = ttot + (t1 - t0)
      tmax = max(tmax, t1 - t0)
      mbtot = mbtot + mb
      if (timing) write(*,'(a,i0,a,i0,a,i0,a,f9.3,a,f9.3,2a)') 'FRCWARM wrank ', comm_world_rank(), ' rec ', rec, &
           ' vars ', nvar, ' MB ', mb, ' ms ', (t1 - t0) * 1.0d3, '  ', trim(path)
    enddo
    do i = 1, nfile
      ios = nf90_close(fid(i))
    enddo
    write(*,'(a,i0,a,i0,a,f10.1,a,f8.3,a,f9.3,a)') 'forcing reader: world rank ', comm_world_rank(), ' read ', nreq, &
         ' records, ', mbtot, ' MB in ', ttot, ' s (slowest ', tmax * 1.0d3, ' ms)'
    flush(6)
#endif
  contains

    subroutine warm_record(p, tn, r, nv, mbread)
      ! Read the slab at record r of every variable of file p whose last
      ! dimension is the dimension of time axis tn (the unlimited dimension
      ! when tn is not a variable of the file), in variable order. Errors
      ! only skip the read: the compute ranks read the data themselves.
      character(len=*), intent(in)  :: p, tn
      integer,          intent(in)  :: r
      integer,          intent(out) :: nv
      real(8),          intent(out) :: mbread
      integer :: ncid, st, v, nvars, ndims, xtype, tdim, tvar, k, n, dl
      integer :: dimids(nf90_max_var_dims), start(nf90_max_var_dims), cnt(nf90_max_var_dims)
      real(4), allocatable, save :: b4(:)
      real(8), allocatable, save :: b8(:)
      nv = 0
      mbread = 0.0d0
      ncid = -1
      do k = 1, nfile
        if (fpath(k) == p) then
          ncid = fid(k)
          exit
        endif
      enddo
      if (ncid < 0) then
        st = nf90_open(trim(p), nf90_nowrite, ncid)
        if (st /= nf90_noerr) then
          write(*,'(4a)') 'forcing reader: cannot open ', trim(p), ': ', trim(nf90_strerror(st))
          return
        endif
        if (nfile == MAXF) then
          st = nf90_close(fid(1))
          fid(1:MAXF - 1) = fid(2:MAXF); fpath(1:MAXF - 1) = fpath(2:MAXF)
          nfile = MAXF - 1
        endif
        nfile = nfile + 1
        fid(nfile) = ncid; fpath(nfile) = p
      endif
      tdim = -1
      if (nf90_inq_varid(ncid, trim(tn), tvar) == nf90_noerr) then
        st = nf90_inquire_variable(ncid, tvar, ndims=ndims, dimids=dimids)
        if (st == nf90_noerr .and. ndims >= 1) tdim = dimids(ndims)
      endif
      if (tdim < 0) then
        st = nf90_inquire(ncid, unlimitedDimId=tdim)
        if (st /= nf90_noerr) tdim = -1
      endif
      if (tdim < 0) return
      if (nf90_inquire(ncid, nVariables=nvars) /= nf90_noerr) return
      do v = 1, nvars
        if (nf90_inquire_variable(ncid, v, xtype=xtype, ndims=ndims, dimids=dimids) /= nf90_noerr) cycle
        if (ndims < 1) cycle
        if (dimids(ndims) /= tdim .or. xtype == nf90_char) cycle
        n = 1
        do k = 1, ndims - 1
          if (nf90_inquire_dimension(ncid, dimids(k), len=dl) /= nf90_noerr) dl = 0
          cnt(k) = dl
          start(k) = 1
          n = n * dl
        enddo
        cnt(ndims) = 1
        start(ndims) = r
        if (n < 1) cycle
        if (xtype == nf90_double) then
          if (allocated(b8)) then
            if (size(b8) < n) deallocate(b8)
          endif
          if (.not. allocated(b8)) allocate(b8(n))
          st = nf90_get_var(ncid, v, b8(1:n), start=start(1:ndims), count=cnt(1:ndims))
          mbread = mbread + 8.0d0 * n / 1.0d6
        else
          if (allocated(b4)) then
            if (size(b4) < n) deallocate(b4)
          endif
          if (.not. allocated(b4)) allocate(b4(n))
          st = nf90_get_var(ncid, v, b4(1:n), start=start(1:ndims), count=cnt(1:ndims))
          if (xtype == nf90_short) then
            mbread = mbread + 2.0d0 * n / 1.0d6
          elseif (xtype == nf90_byte) then
            mbread = mbread + 1.0d0 * n / 1.0d6
          else
            mbread = mbread + 4.0d0 * n / 1.0d6
          endif
        endif
        if (st == nf90_noerr) nv = nv + 1
      enddo
    end subroutine warm_record

  end subroutine frc_reader_run

end module frc_reader

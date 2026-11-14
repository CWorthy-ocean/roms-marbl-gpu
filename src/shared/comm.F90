module comm_mod
  ! MPI wrapper for halo deliver and the optional I/O server.
  ! World = compute ranks plus writer (I/O server) ranks. Roles, two ways:
  !  - ROMS_GPU_NWRITERS = w (writers per node): on every node the last w
  !    ranks are writers and the others compute ranks, so a job with a
  !    uniform ranks-per-node count (SLURM --ntasks-per-node) works; the
  !    compute ranks must number np_xi x np_eta or the run stops naming the
  !    layout it expected. w = 0: no writers.
  !  - unset (as before): when the job has >= 2 ranks and ROMS_GPU_IO_SERVER
  !    is not '0', every world rank after the first np_xi x np_eta is a
  !    writer (ROMS_GPU_IO_SERVERS overrides their number).
  ! Compute ranks are numbered in world order (node by node), so the tile
  ! decomposition is the same either way; writer_world and compute_of_world
  ! translate between writer, compute and world ranks.
  ! Forcing readers (ROMS_GPU_FRC_READER = 1, src/host/frc_reader.F90): the
  ! last rank of every node reads the forcing records ahead into the node's
  ! page cache. They are taken out first, and the rules above apply to the
  ! other ranks, so the launch adds one rank to every node. All or none:
  ! when a node has no rank to spare (the other ranks would not give
  ! np_xi x np_eta compute ranks, and with the default layout at least one
  ! writer), there are no readers and the run says so.
  ! Compute ranks exchange halos on compute_comm (comm_rank/comm_size
  ! refer to it); the server only receives staged output records over
  ! the world communicator (see io_server). One compute rank is a no-op
  ! for halos (NBR_SELF / NBR_NONE stay in halo_mod). Real neighbor ranks
  ! (>= 0) go through host-staged MPI_Sendrecv of the same pack buffers.
  ! CUDA graphs stay off when comm_size > 1 because deliver is host MPI.
  use kinds
  use iso_c_binding, only: c_ptr, c_f_pointer
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private

  integer(ip) :: my_rank = 0
  integer(ip) :: nproc = 1
  integer :: world_rank = 0, world_size = 1
  integer :: world_comm = 0, compute_comm = 0
  ! GPU-aware MPI (Cray MPICH + libmpi_gtl_cuda): device buffers go straight
  ! to MPI_Sendrecv when MPICH_GPU_SUPPORT_ENABLED=1 is set for the job.
  logical :: comm_gpu_direct = .false.
  integer :: io_rank_w = -1          ! world rank of the first I/O server rank, -1 = none
  integer :: io_nserv = 0            ! number of I/O server (writer) ranks
  logical :: io_rank_flag = .false.
  integer :: node_rank = 0, node_size = 1     ! rank and rank count on this node
  integer, allocatable :: writer_world(:)     ! (0:io_nserv-1) world rank of each writer
  integer, allocatable :: compute_of_world(:) ! (0:world_size-1) compute rank, -1 on writers and readers
  logical :: rd_flag = .false.                ! this rank is a forcing reader
  integer :: rd_n = 0                         ! number of forcing reader ranks (0 or one per node)
  integer :: ncompute_w = 1                   ! number of compute ranks
  integer, allocatable :: reader_world(:)     ! (0:rd_n-1) world rank of each reader
  logical :: ready = .false.
  ! comm_fs_sum (the implicit free surface's PCG reductions, fs_gpu_opt 512): when all compute ranks
  ! (1, 2 or 4) share one node, the sums go through an MPI-3 shared-memory window instead of
  ! MPI_Allreduce. Per rank one column of SHM_SEG doubles: two slots of 8 values (1:8, 9:16) and the
  ! sequence counter (17). shm_cmp (ROMS_GPU_FS_SHM_CMP=1): every sum is also made by MPI_Allreduce
  ! and the bit differences counted (shm_ndiff), the shared-memory result is used.
  integer, parameter :: SHM_SEG = 32
  logical :: shm_on = .false., shm_cmp = .false.
  integer :: shm_win = 0, shm_np = 0, shm_me = 0
  integer(8) :: shm_q = 0, shm_ncall = 0, shm_ndiff = 0
  real(8), pointer, volatile :: shm(:,:) => null()

  public :: init_comm
  public :: finalize_comm
  public :: comm_abort
  public :: comm_rank
  public :: comm_size
  public :: comm_exchange2
  public :: comm_sendrecv_raw
  public :: comm_sendrecv_dev
  public :: comm_post_dev, comm_wait_dev
  public :: comm_ring_dev8, comm_ring_h8, comm_ipc_exchange, comm_ipc_allgather
  public :: comm_min_real
  public :: comm_reduce_maxsum
  public :: comm_reduce_sum
  public :: comm_allreduce_sum, comm_allreduce_max, comm_allreduce_min
  public :: comm_shm_init, comm_fs_sum, comm_fs_sum_stats
  public :: comm_allgather_i4, comm_allgatherv_r8, comm_allgatherv_dev, comm_allgatherv_h8
  public :: comm_gpu_direct
  public :: comm_exchange3
  public :: have_io_server
  public :: is_io_rank
  public :: io_server_rank
  public :: io_server_count, io_server_index, comm_writers
  public :: comm_writer_world, comm_compute_index, comm_node_rank
  public :: is_reader_rank, comm_reader_count, comm_reader_world, comm_ncompute
  public :: comm_world
  public :: comm_world_rank
  public :: comm_world_size
  public :: compute_barrier
  public :: comm_compute

contains

  subroutine init_comm()
#ifdef USE_MPI
    integer :: ierr, color, stat, r, n, nw, nwpn, node_comm, mine, nrd, nrdpn, below
    logical :: flag
    character(len=8) :: val
    integer, allocatable :: role(:)
    logical, allocatable :: isrd(:)
    call MPI_Initialized(flag, ierr)
    if (.not. flag) then
      call MPI_Init(ierr)
      if (ierr /= MPI_SUCCESS) error stop 'init_comm: MPI_Init'
    endif
    world_comm = MPI_COMM_WORLD
    call MPI_Comm_rank(world_comm, world_rank, ierr)
    call MPI_Comm_size(world_comm, world_size, ierr)
    call MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, node_comm, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'init_comm: MPI_Comm_split_type'
    call MPI_Comm_rank(node_comm, node_rank, ierr)
    call MPI_Comm_size(node_comm, node_size, ierr)
    call MPI_Comm_free(node_comm, ierr)
    io_rank_w = -1
    io_nserv = 0
    io_rank_flag = .false.
    nwpn = -1
    val = ''
    call get_environment_variable('ROMS_GPU_NWRITERS', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) nwpn
      if (stat /= 0 .or. nwpn < 0) then
        if (world_rank == 0) write(*,'(3a)') 'init_comm: ROMS_GPU_NWRITERS = "', trim(val), &
             '" is not a number of writers per node (0 or more)'
        error stop 'init_comm: bad ROMS_GPU_NWRITERS'
      endif
    endif
    ! forcing readers: the last rank of every node, when every node can spare one
    allocate(isrd(0:world_size - 1))
    call setup_readers(nwpn, isrd)
    rd_flag = isrd(world_rank)
    nrd = count(isrd)
    nrdpn = merge(1, 0, nrd > 0)
    below = count(.not. isrd(0:world_rank - 1))   ! non-reader ranks before this one
    deallocate(isrd)
    if (nwpn >= 0) then
      io_rank_flag = (.not. rd_flag) .and. (node_rank >= node_size - nrdpn - nwpn)
      call check_node_layout(nwpn, nrdpn)
    elseif (world_size - nrd >= 2) then
      call get_environment_variable('ROMS_GPU_IO_SERVER', val, status=stat)
      if (.not. (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '0')) then
        ! ROMS_GPU_IO_SERVERS = number of writer ranks (default 1); they are
        ! the last world ranks (readers aside) and share the output files
        ! (parallel NetCDF when more than one).
        ! Rule: the compute ranks are np_xi x np_eta from the namelist and
        ! every extra launched rank is a writer. ROMS_GPU_IO_SERVERS overrides.
        io_nserv = 1
        n = namelist_tiles()
        if (n >= 1 .and. world_size - nrd > n) io_nserv = world_size - nrd - n
        call get_environment_variable('ROMS_GPU_IO_SERVERS', val, status=stat)
        if (stat == 0 .and. len_trim(val) > 0) then
          read(val, *, iostat=stat) n
          if (stat == 0 .and. n >= 1 .and. n < world_size - nrd) io_nserv = n
        endif
        io_rank_w = world_size - nrd - io_nserv   ! in the numbering of the non-reader ranks
      endif
      io_rank_flag = (.not. rd_flag) .and. (below >= io_rank_w .and. io_rank_w >= 0)
    endif
    ! the role of every world rank (0 compute, 1 writer, 2 reader): writer,
    ! reader and compute rank tables
    allocate(role(0:world_size - 1), compute_of_world(0:world_size - 1))
    mine = merge(1, 0, io_rank_flag)
    if (rd_flag) mine = 2
    call MPI_Allgather(mine, 1, MPI_INTEGER, role, 1, MPI_INTEGER, world_comm, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'init_comm: MPI_Allgather roles'
    io_nserv = count(role == 1)
    rd_n = count(role == 2)
    allocate(writer_world(0:max(io_nserv, 1) - 1), reader_world(0:max(rd_n, 1) - 1))
    writer_world = -1
    reader_world = -1
    n = 0
    nw = 0
    nrd = 0
    do r = 0, world_size - 1
      if (role(r) == 1) then
        writer_world(nw) = r
        nw = nw + 1
        compute_of_world(r) = -1
      elseif (role(r) == 2) then
        reader_world(nrd) = r
        nrd = nrd + 1
        compute_of_world(r) = -1
      else
        compute_of_world(r) = n
        n = n + 1
      endif
    enddo
    ncompute_w = n
    deallocate(role)
    io_rank_w = -1
    if (io_nserv > 0) io_rank_w = writer_world(0)
    if (io_rank_w >= 0 .or. rd_n > 0) then
      color = merge(1, 0, io_rank_flag)
      if (rd_flag) color = 2
      call MPI_Comm_split(world_comm, color, world_rank, compute_comm, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'init_comm: MPI_Comm_split'
    else
      compute_comm = world_comm
    endif
    call MPI_Comm_rank(compute_comm, r, ierr)
    call MPI_Comm_size(compute_comm, n, ierr)
    my_rank = int(r, ip)
    nproc = int(n, ip)
    call get_environment_variable('MPICH_GPU_SUPPORT_ENABLED', val, status=stat)
    comm_gpu_direct = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '1')
    if (my_rank == 0 .and. .not. io_rank_flag .and. .not. rd_flag) &
         write(*,'(a,l1)') 'comm: GPU-direct halo exchange (MPICH_GPU_SUPPORT_ENABLED) = ', comm_gpu_direct
#endif
    ready = .true.
  end subroutine init_comm

  subroutine check_node_layout(nwpn, nrdpn)
    ! ROMS_GPU_NWRITERS = nwpn: every node needs at least one compute rank,
    ! and the compute ranks must number np_xi x np_eta. Every rank runs the
    ! same check on the same gathered numbers, so all stop together.
    ! nrdpn: forcing readers per node (0 or 1), the last rank of each node.
    integer, intent(in) :: nwpn, nrdpn
#ifdef USE_MPI
    integer :: ierr, nodes, ncomp, ntile, r, lo, hi
    integer, allocatable :: nr(:), ns(:)
    character(len=24) :: sz
    allocate(nr(0:world_size - 1), ns(0:world_size - 1))
    call MPI_Allgather(node_rank, 1, MPI_INTEGER, nr, 1, MPI_INTEGER, world_comm, ierr)
    call MPI_Allgather(node_size, 1, MPI_INTEGER, ns, 1, MPI_INTEGER, world_comm, ierr)
    nodes = count(nr == 0)
    lo = minval(ns); hi = maxval(ns)
    ncomp = 0
    do r = 0, world_size - 1
      if (nr(r) < ns(r) - nwpn - nrdpn) ncomp = ncomp + 1
    enddo
    ntile = namelist_tiles()
    if (lo == hi) then
      write(sz, '(i0)') lo
    else
      write(sz, '(i0,a,i0)') lo, '-', hi
    endif
    if (world_rank == 0) write(*,'(a,i0,a,i0,a,a,a,i0,a,i0,a)') 'comm: ', nodes, ' node(s), ', world_size, &
         ' ranks (', trim(sz), ' per node), ', ncomp, ' compute + ', world_size - ncomp - nodes * nrdpn, &
         ' writer ranks (ROMS_GPU_NWRITERS, writers per node)'
    if (lo <= nwpn + nrdpn .or. (ntile >= 1 .and. ncomp /= ntile)) then
      if (world_rank == 0) then
        write(*,'(a,i0,a,i0,a,a,a)') 'init_comm: ROMS_GPU_NWRITERS = ', nwpn, ' writers per node, but the job has ', &
             nodes, ' node(s) with ', trim(sz), ' ranks each'
        if (lo <= nwpn + nrdpn) write(*,'(a)') 'init_comm: a node has no rank left for computing'
        if (ntile >= 1 .and. ncomp /= ntile) then
          write(*,'(a,i0,a,i0,a)') 'init_comm: that leaves ', ncomp, ' compute ranks; the namelist has np_xi x np_eta = ', &
               ntile, ' tiles'
          if (mod(ntile, nodes) == 0) write(*,'(a,i0,a,i0,a,i0,a,i0,a)') 'init_comm: expected ', nodes, &
               ' node(s) x (', ntile / nodes, ' + ', nwpn + nrdpn, ') = ', nodes * (ntile / nodes + nwpn + nrdpn), ' ranks'
        endif
        flush(6)
      endif
      error stop 'init_comm: rank layout does not match ROMS_GPU_NWRITERS and the namelist tiles'
    endif
    deallocate(nr, ns)
#endif
  end subroutine check_node_layout

  subroutine setup_readers(nwpn, isrd)
    ! ROMS_GPU_FRC_READER = 1: isrd(r) is true for the last rank of every
    ! node, when every node can spare it (see the module header); 0 or unset:
    ! no readers. Every rank decides on the same gathered numbers.
    integer, intent(in)  :: nwpn
    logical, intent(out) :: isrd(0:world_size - 1)
#ifdef USE_MPI
    integer :: ierr, stat, nodes, ntile, lo, r, ncomp, want
    integer, allocatable :: nr(:), ns(:)
    character(len=8) :: val
    character(len=160) :: why
    logical :: srv_off
    isrd = .false.
    want = 0
    val = ''
    call get_environment_variable('ROMS_GPU_FRC_READER', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) want
      if (stat /= 0 .or. want < 0 .or. want > 1) then
        if (world_rank == 0) write(*,'(3a)') 'init_comm: ROMS_GPU_FRC_READER = "', trim(val), &
             '" is not 0 or 1 (forcing reader ranks per node)'
        error stop 'init_comm: bad ROMS_GPU_FRC_READER'
      endif
    endif
    if (want == 0) return
    allocate(nr(0:world_size - 1), ns(0:world_size - 1))
    call MPI_Allgather(node_rank, 1, MPI_INTEGER, nr, 1, MPI_INTEGER, world_comm, ierr)
    call MPI_Allgather(node_size, 1, MPI_INTEGER, ns, 1, MPI_INTEGER, world_comm, ierr)
    nodes = count(nr == 0)
    lo = minval(ns)
    ntile = namelist_tiles()
    why = ''
    if (ntile < 1) then
      why = 'np_xi x np_eta is not known (namelist not found)'
    elseif (lo < 2) then
      why = 'a node has a single rank'
    elseif (nwpn >= 0) then
      ncomp = 0
      do r = 0, world_size - 1
        if (nr(r) < ns(r) - nwpn - 1) ncomp = ncomp + 1
      enddo
      if (lo <= nwpn + 1 .or. ncomp /= ntile) write(why, '(a,i0,a,i0,a)') 'with one reader per node the ranks give ', &
           ncomp, ' compute ranks for ', ntile, ' tiles (ROMS_GPU_NWRITERS)'
    else
      call get_environment_variable('ROMS_GPU_IO_SERVER', val, status=stat)
      srv_off = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '0')
      if (srv_off .and. world_size - nodes /= ntile) then
        write(why, '(a,i0,a,i0,a)') 'without them ', world_size - nodes, ' ranks are left for ', ntile, ' tiles'
      elseif (.not. srv_off .and. world_size - nodes < ntile + 1) then
        write(why, '(a,i0,a,i0,a)') 'without them ', world_size - nodes, ' ranks are left for ', ntile, &
             ' tiles and at least one writer'
      endif
    endif
    if (len_trim(why) > 0) then
      if (world_rank == 0) write(*,'(a,i0,a,a,a)') 'comm: ROMS_GPU_FRC_READER = 1, but the job has no spare rank on its ', &
           nodes, ' node(s) (', trim(why), '): no forcing reader (launch one more rank on every node)'
    else
      do r = 0, world_size - 1
        isrd(r) = (nr(r) == ns(r) - 1)
      enddo
      if (world_rank == 0) write(*,'(a,i0,a)') 'comm: ', nodes, &
           ' forcing reader rank(s) (ROMS_GPU_FRC_READER = 1): the last rank of every node'
    endif
    deallocate(nr, ns)
#else
    isrd = .false.
#endif
  end subroutine setup_readers

  integer function namelist_tiles() result(ntile)
    ! np_xi * np_eta from the namelist named on the command line (argument
    ! 2 of 'run' / 'load'), by a plain text scan; 0 when unavailable.
    character(len=512) :: path, line
    character(len=16) :: mode
    integer :: unit, ios, npx, npy, k
    ntile = 0
    if (command_argument_count() < 2) return
    call get_command_argument(1, mode)
    if (trim(mode) /= 'run' .and. trim(mode) /= 'load') return
    call get_command_argument(2, path)
    open(newunit=unit, file=trim(path), status='old', action='read', iostat=ios)
    if (ios /= 0) return
    npx = 0; npy = 0
    do
      read(unit, '(a)', iostat=ios) line
      if (ios /= 0) exit
      k = index(line, '!')
      if (k > 0) line = line(1:k - 1)
      call grab(line, 'np_xi', npx)
      call grab(line, 'np_eta', npy)
    enddo
    close(unit)
    if (npx >= 1 .and. npy >= 1) ntile = npx * npy
  contains
    subroutine grab(l, key, v)
      character(len=*), intent(in) :: l, key
      integer, intent(inout) :: v
      integer :: i, j, ios2
      i = index(l, key)
      if (i == 0) return
      if (i > 1) then
        if (l(i-1:i-1) /= ' ' .and. l(i-1:i-1) /= ',') return
      endif
      j = i + len(key)
      if (j > len(l)) return
      if (l(j:j) /= ' ' .and. l(j:j) /= '=') return
      j = index(l(i:), '=')
      if (j == 0) return
      read(l(i + j:), *, iostat=ios2) v
      if (ios2 /= 0) v = 0
    end subroutine grab
  end function namelist_tiles

  subroutine finalize_comm()
#ifdef USE_MPI
    integer :: ierr
    logical :: flag
    call MPI_Finalized(flag, ierr)
    if (.not. flag) call MPI_Finalize(ierr)
#endif
    ready = .false.
    my_rank = 0
    nproc = 1
    io_rank_w = -1
    io_rank_flag = .false.
  end subroutine finalize_comm

  subroutine comm_abort(code)
    ! Stops every rank, compute and writer, after an error on this one (an
    ! error stop on one rank alone may leave the others waiting).
    integer, intent(in) :: code
#ifdef USE_MPI
    integer :: ierr
    flush(6)
    call MPI_Abort(MPI_COMM_WORLD, code, ierr)
#endif
    error stop code
  end subroutine comm_abort

  integer function comm_world_size()
    comm_world_size = world_size
  end function comm_world_size

  integer function comm_compute()
    ! The communicator of the compute ranks (world when there is no server).
    comm_compute = compute_comm
  end function comm_compute

  subroutine compute_barrier()
    ! Barrier over the compute ranks (no-op on one rank).
#ifdef USE_MPI
    integer :: ierr
    if (comm_size() > 1) call MPI_Barrier(compute_comm, ierr)
#endif
  end subroutine compute_barrier

  integer(ip) function comm_rank()
    comm_rank = my_rank
  end function comm_rank

  integer(ip) function comm_size()
    comm_size = nproc
  end function comm_size

  logical function have_io_server()
    have_io_server = (io_rank_w >= 0)
  end function have_io_server

  logical function is_io_rank()
    is_io_rank = io_rank_flag
  end function is_io_rank

  logical function is_reader_rank()
    ! This rank is a forcing reader (src/host/frc_reader.F90).
    is_reader_rank = rd_flag
  end function is_reader_rank

  integer function comm_reader_count()
    comm_reader_count = rd_n
  end function comm_reader_count

  integer function comm_reader_world(i)
    ! world rank of forcing reader i (0 .. comm_reader_count()-1)
    integer, intent(in) :: i
    comm_reader_world = reader_world(i)
  end function comm_reader_world

  integer function comm_ncompute()
    ! number of compute ranks (world minus writers and readers)
    comm_ncompute = ncompute_w
  end function comm_ncompute

  integer function io_server_rank()
    io_server_rank = io_rank_w
  end function io_server_rank

  integer function io_server_count()
    io_server_count = io_nserv
  end function io_server_count

  integer function io_server_index()
    ! 0 .. nserv-1 on a writer rank (its rank among the writers), -1 elsewhere
    io_server_index = -1
    if (io_rank_flag) io_server_index = int(my_rank)
  end function io_server_index

  integer function comm_writer_world(w)
    ! world rank of writer w (0 .. nserv-1)
    integer, intent(in) :: w
    comm_writer_world = writer_world(w)
  end function comm_writer_world

  integer function comm_compute_index(r)
    ! compute rank of world rank r (-1 for a writer or reader)
    integer, intent(in) :: r
    comm_compute_index = r
    if (allocated(compute_of_world)) comm_compute_index = compute_of_world(r)
  end function comm_compute_index

  integer function comm_node_rank()
    ! this rank's rank on its node (compute ranks come first on every node)
    comm_node_rank = node_rank
  end function comm_node_rank

  integer function comm_writers()
    ! The writer ranks' own communicator (the split gives it to colour 1;
    ! on a reader rank, the readers' communicator).
    comm_writers = compute_comm
  end function comm_writers

  integer function comm_world()
    comm_world = world_comm
  end function comm_world

  integer function comm_world_rank()
    comm_world_rank = world_rank
  end function comm_world_rank

  integer function mpi_dest(nbr)
    integer(ip), intent(in) :: nbr
#ifdef USE_MPI
    if (nbr >= 0) then
      mpi_dest = int(nbr)
    else
      mpi_dest = MPI_PROC_NULL
    endif
#else
    if (nbr >= 0) error stop 'comm: MPI neighbor but this build has no USE_MPI'
    mpi_dest = -1
#endif
  end function mpi_dest

  subroutine comm_sendrecv_raw(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi)
    ! The two neighbor exchanges on contiguous host buffers of n values
    ! (no allocation or reshape; the caller stages device data into pinned
    ! memory). A missing neighbor is MPI_PROC_NULL: its receive buffer is
    ! left untouched and the caller does not read it.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer,     intent(in) :: n
    real(rp),    intent(in)    :: s_lo(*), s_hi(*)
    real(rp),    intent(inout) :: r_lo(*), r_hi(*)
#ifdef USE_MPI
    integer :: dest_lo, dest_hi, ierr, req(4), stats(MPI_STATUS_SIZE, 4)
    if (.not. ready) call init_comm()
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    ! Both directions in flight at once: receives first, then sends, one wait.
    call MPI_Irecv(r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 301,     compute_comm, req(1), ierr)
    call MPI_Irecv(r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 302, compute_comm, req(2), ierr)
    call MPI_Isend(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 301,     compute_comm, req(3), ierr)
    call MPI_Isend(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 302, compute_comm, req(4), ierr)
    call MPI_Waitall(4, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_sendrecv_raw: waitall'
#else
    if (nbr_lo >= 0 .or. nbr_hi >= 0) error stop 'comm_sendrecv_raw: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_sendrecv_raw

  subroutine comm_sendrecv_dev(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi)
    ! As comm_sendrecv_raw on DEVICE buffers (GPU-aware MPI); the caller has
    ! synchronized the stream that packed them.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer,     intent(in) :: n
#ifdef USE_CUDA
    real(rp), device, intent(in)    :: s_lo(*), s_hi(*)
    real(rp), device, intent(inout) :: r_lo(*), r_hi(*)
#else
    real(rp), intent(in)    :: s_lo(*), s_hi(*)
    real(rp), intent(inout) :: r_lo(*), r_hi(*)
#endif
#ifdef USE_MPI
    integer :: dest_lo, dest_hi, ierr, req(4), stats(MPI_STATUS_SIZE, 4)
    if (.not. ready) call init_comm()
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    ! Both directions in flight at once: receives first, then sends, one wait.
    call MPI_Irecv(r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 311,     compute_comm, req(1), ierr)
    call MPI_Irecv(r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 312, compute_comm, req(2), ierr)
    call MPI_Isend(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 311,     compute_comm, req(3), ierr)
    call MPI_Isend(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 312, compute_comm, req(4), ierr)
    call MPI_Waitall(4, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_sendrecv_dev: waitall'
#else
    if (nbr_lo >= 0 .or. nbr_hi >= 0) error stop 'comm_sendrecv_dev: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_sendrecv_dev

  subroutine comm_ring_dev8(nbr, nw, ns, sw, se, ss, sn, rw, re, rs, rn)
    ! One-round exchange of the four edge rows of a real(8) field with the
    ! four neighbours on DEVICE buffers (GPU-aware MPI): nbr = (west, east,
    ! south, north), nw (ns) values in the west/east (south/north) buffers.
    ! A non-MPI neighbour (NBR_NONE, NBR_SELF) is MPI_PROC_NULL here (the
    ! caller handles the self-wrap). The caller has synchronized the stream
    ! that packed the send buffers. Tags 321-324 (direction of travel),
    ! distinct from the other exchanges. (The implicit free surface's
    ! one-cell rings without corners, fs_data ring; IMEX's one-cell
    ! exchange of the stage solution, imex_igw's pf_ring1_8.)
    integer(ip), intent(in) :: nbr(4)
    integer,     intent(in) :: nw, ns
#ifdef USE_CUDA
    real(8), device, intent(in)    :: sw(*), se(*), ss(*), sn(*)
    real(8), device, intent(inout) :: rw(*), re(*), rs(*), rn(*)
#else
    real(8), intent(in)    :: sw(*), se(*), ss(*), sn(*)
    real(8), intent(inout) :: rw(*), re(*), rs(*), rn(*)
#endif
#ifdef USE_MPI
    integer :: d(4), ierr, req(8), stats(MPI_STATUS_SIZE, 8)
    if (.not. ready) call init_comm()
    d = [mpi_dest(nbr(1)), mpi_dest(nbr(2)), mpi_dest(nbr(3)), mpi_dest(nbr(4))]
    call MPI_Irecv(re, nw, MPI_DOUBLE_PRECISION, d(2), 321, compute_comm, req(1), ierr)   ! travelling west, from the east
    call MPI_Irecv(rw, nw, MPI_DOUBLE_PRECISION, d(1), 322, compute_comm, req(2), ierr)   ! travelling east, from the west
    call MPI_Irecv(rn, ns, MPI_DOUBLE_PRECISION, d(4), 323, compute_comm, req(3), ierr)   ! travelling south, from the north
    call MPI_Irecv(rs, ns, MPI_DOUBLE_PRECISION, d(3), 324, compute_comm, req(4), ierr)   ! travelling north, from the south
    call MPI_Isend(sw, nw, MPI_DOUBLE_PRECISION, d(1), 321, compute_comm, req(5), ierr)
    call MPI_Isend(se, nw, MPI_DOUBLE_PRECISION, d(2), 322, compute_comm, req(6), ierr)
    call MPI_Isend(ss, ns, MPI_DOUBLE_PRECISION, d(3), 323, compute_comm, req(7), ierr)
    call MPI_Isend(sn, ns, MPI_DOUBLE_PRECISION, d(4), 324, compute_comm, req(8), ierr)
    call MPI_Waitall(8, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_ring_dev8: waitall'
#else
    if (any(nbr >= 0)) error stop 'comm_ring_dev8: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_ring_dev8

  subroutine comm_ring_h8(nbr, nw, ns, sw, se, ss, sn, rw, re, rs, rn)
    ! comm_ring_dev8 on HOST buffers (the implicit free surface's rings through mapped pinned
    ! memory, fs_gpu_opt 1): same messages, tags and completion.
    integer(ip), intent(in) :: nbr(4)
    integer,     intent(in) :: nw, ns
    real(8), intent(in)    :: sw(*), se(*), ss(*), sn(*)
    real(8), intent(inout) :: rw(*), re(*), rs(*), rn(*)
#ifdef USE_MPI
    integer :: d(4), ierr, req(8), stats(MPI_STATUS_SIZE, 8)
    if (.not. ready) call init_comm()
    d = [mpi_dest(nbr(1)), mpi_dest(nbr(2)), mpi_dest(nbr(3)), mpi_dest(nbr(4))]
    call MPI_Irecv(re, nw, MPI_DOUBLE_PRECISION, d(2), 321, compute_comm, req(1), ierr)
    call MPI_Irecv(rw, nw, MPI_DOUBLE_PRECISION, d(1), 322, compute_comm, req(2), ierr)
    call MPI_Irecv(rn, ns, MPI_DOUBLE_PRECISION, d(4), 323, compute_comm, req(3), ierr)
    call MPI_Irecv(rs, ns, MPI_DOUBLE_PRECISION, d(3), 324, compute_comm, req(4), ierr)
    call MPI_Isend(sw, nw, MPI_DOUBLE_PRECISION, d(1), 321, compute_comm, req(5), ierr)
    call MPI_Isend(se, nw, MPI_DOUBLE_PRECISION, d(2), 322, compute_comm, req(6), ierr)
    call MPI_Isend(ss, ns, MPI_DOUBLE_PRECISION, d(3), 323, compute_comm, req(7), ierr)
    call MPI_Isend(sn, ns, MPI_DOUBLE_PRECISION, d(4), 324, compute_comm, req(8), ierr)
    call MPI_Waitall(8, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_ring_h8: waitall'
#else
    if (any(nbr >= 0)) error stop 'comm_ring_h8: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_ring_h8

  subroutine comm_post_dev(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi, tag_lo, tag_hi, req)
    ! Non-blocking form of comm_sendrecv_dev: posts the four requests with
    ! the caller's tags (distinct from the blocking exchanges, which use
    ! 301/302 and 311/312, so an exchange in flight cannot be matched by
    ! one posted later) and returns; comm_wait_dev completes them. The
    ! buffers must stay untouched until then.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer,     intent(in) :: n, tag_lo, tag_hi
#ifdef USE_CUDA
    real(rp), device, intent(in)    :: s_lo(*), s_hi(*)
    real(rp), device, intent(inout) :: r_lo(*), r_hi(*)
#else
    real(rp), intent(in)    :: s_lo(*), s_hi(*)
    real(rp), intent(inout) :: r_lo(*), r_hi(*)
#endif
    integer, intent(out) :: req(4)
#ifdef USE_MPI
    integer :: dest_lo, dest_hi, ierr
    if (.not. ready) call init_comm()
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    call MPI_Irecv(r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, tag_lo, compute_comm, req(1), ierr)
    call MPI_Irecv(r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, tag_hi, compute_comm, req(2), ierr)
    call MPI_Isend(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, tag_lo, compute_comm, req(3), ierr)
    call MPI_Isend(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, tag_hi, compute_comm, req(4), ierr)
#else
    req = 0
    if (nbr_lo >= 0 .or. nbr_hi >= 0) error stop 'comm_post_dev: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_post_dev

  real(rp) function comm_min_real(x)
    ! Minimum of x over the compute ranks (x itself without MPI).
    real(rp), intent(in) :: x
#ifdef USE_MPI
    integer :: ierr
    real(rp) :: y
    if (.not. ready) call init_comm()
    call MPI_Allreduce(x, y, 1, MPI_DOUBLE_PRECISION, MPI_MIN, compute_comm, ierr)
    comm_min_real = y
#else
    comm_min_real = x
#endif
  end function comm_min_real

  subroutine comm_reduce_maxsum(n, x, xmax, xsum)
    ! Max and sum of x(1:n) over the compute ranks, on compute rank 0
    ! (x itself without MPI).
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xmax(n), xsum(n)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Reduce(x, xmax, n, MPI_DOUBLE_PRECISION, MPI_MAX, 0, compute_comm, ierr)
    call MPI_Reduce(x, xsum, n, MPI_DOUBLE_PRECISION, MPI_SUM, 0, compute_comm, ierr)
#else
    xmax = x; xsum = x
#endif
  end subroutine comm_reduce_maxsum

  subroutine comm_reduce_sum(n, x, xsum)
    ! Sum of x(1:n) over the compute ranks, on compute rank 0 (x itself
    ! without MPI).
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xsum(n)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Reduce(x, xsum, n, MPI_DOUBLE_PRECISION, MPI_SUM, 0, compute_comm, ierr)
#else
    xsum = x
#endif
  end subroutine comm_reduce_sum

  subroutine comm_allreduce_sum(n, x, xsum)
    ! Sum of x(1:n) over the compute ranks, on every compute rank.
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xsum(n)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Allreduce(x, xsum, n, MPI_DOUBLE_PRECISION, MPI_SUM, compute_comm, ierr)
#else
    xsum = x
#endif
  end subroutine comm_allreduce_sum

  subroutine comm_shm_init(on)
    ! Collective over the compute ranks (fs_gpu_opt 512): set up comm_fs_sum's shared-memory window.
    ! on: in use on every rank (1 rank; or 2 or 4 ranks, all on one node, window made on every rank);
    ! otherwise comm_fs_sum is MPI_Allreduce. The decision is the same on every rank.
    logical, intent(out) :: on
#ifdef USE_MPI
    integer :: ncomm, nsz, ierr, du, ok(1), gok(1)
    integer(kind=MPI_ADDRESS_KIND) :: sz, base
    type(c_ptr) :: cp
    character(len=8) :: val
    integer :: st
    if (.not. ready) call init_comm()
    on = .false.
    val = ''
    call get_environment_variable('ROMS_GPU_FS_SHM_CMP', val, status=st)
    shm_cmp = (st == 0 .and. val(1:1) == '1')
    if (nproc == 1) then
      shm_np = 1; shm_me = 0; shm_on = .true.; on = .true.
      return
    endif
    if (nproc /= 2 .and. nproc /= 4) return
    call MPI_Comm_split_type(compute_comm, MPI_COMM_TYPE_SHARED, int(my_rank), MPI_INFO_NULL, ncomm, ierr)
    ok(1) = 0
    if (ierr == MPI_SUCCESS) then
      call MPI_Comm_size(ncomm, nsz, ierr)
      if (nsz == nproc) ok(1) = 1
    endif
    call MPI_Allreduce(ok, gok, 1, MPI_INTEGER, MPI_MIN, compute_comm, ierr)
    if (gok(1) == 0) return
    ! one segment, allocated by rank 0 of the node communicator (= compute rank 0: key my_rank)
    sz = 0
    if (my_rank == 0) sz = int(SHM_SEG, MPI_ADDRESS_KIND) * 8 * int(nproc, MPI_ADDRESS_KIND)
    call MPI_Win_allocate_shared(sz, 8, MPI_INFO_NULL, ncomm, base, shm_win, ierr)
    ok(1) = merge(1, 0, ierr == MPI_SUCCESS)
    if (ok(1) == 1) then
      call MPI_Win_shared_query(shm_win, 0, sz, du, base, ierr)
      if (ierr /= MPI_SUCCESS .or. sz < int(SHM_SEG, MPI_ADDRESS_KIND) * 8 * int(nproc, MPI_ADDRESS_KIND)) ok(1) = 0
    endif
    call MPI_Allreduce(ok, gok, 1, MPI_INTEGER, MPI_MIN, compute_comm, ierr)
    if (gok(1) == 0) return
    cp = transfer(base, cp)
    call c_f_pointer(cp, shm, [SHM_SEG, int(nproc)])
    call MPI_Win_lock_all(MPI_MODE_NOCHECK, shm_win, ierr)
    shm(:, my_rank + 1) = 0.0d0
    call MPI_Win_sync(shm_win, ierr)
    call MPI_Barrier(compute_comm, ierr)
    shm_np = int(nproc); shm_me = int(my_rank); shm_q = 0; shm_on = .true.; on = .true.
#else
    shm_np = 1; shm_on = .true.; on = .true.
#endif
  end subroutine comm_shm_init

  subroutine comm_fs_sum(n, x, xsum)
    ! Sum of x(1:n) (n <= 8) over the compute ranks, on every compute rank. With the shared-memory
    ! window (comm_shm_init): this rank's values into slot mod(q, 2) of its column, its counter
    ! raised to q, a spin until every counter has reached q, then the pairwise sum in compute-rank
    ! order, x0 + x1 or (x0 + x1) + (x2 + x3), on every rank. Two slots suffice: a rank writes
    ! slot mod(q, 2) again only in reduction q + 2, after every other rank has raised its counter
    ! to q + 1, i.e. has finished reading reduction q. Otherwise MPI_Allreduce.
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xsum(n)
#ifdef USE_MPI
    integer :: k, o, ierr
    real(8) :: q, ref(8)
    if (.not. shm_on) then
      call comm_allreduce_sum(n, x, xsum)
      return
    endif
    shm_ncall = shm_ncall + 1
    if (shm_np == 1) then
      xsum(1:n) = x(1:n)
    else
      shm_q = shm_q + 1; q = real(shm_q, 8); o = 8 * int(mod(shm_q, 2_8))
      shm(o + 1:o + n, shm_me + 1) = x(1:n)
      call MPI_Win_sync(shm_win, ierr)
      shm(17, shm_me + 1) = q
      do k = 1, shm_np
        do while (shm(17, k) < q)
          call MPI_Win_sync(shm_win, ierr)     ! (memory barrier; shm is volatile as well)
        enddo
      enddo
      call MPI_Win_sync(shm_win, ierr)
      if (shm_np == 2) then
        xsum(1:n) = shm(o + 1:o + n, 1) + shm(o + 1:o + n, 2)
      else
        xsum(1:n) = (shm(o + 1:o + n, 1) + shm(o + 1:o + n, 2)) + (shm(o + 1:o + n, 3) + shm(o + 1:o + n, 4))
      endif
    endif
    if (shm_cmp) then
      call MPI_Allreduce(x, ref, n, MPI_DOUBLE_PRECISION, MPI_SUM, compute_comm, ierr)
      do k = 1, n
        if (transfer(ref(k), 0_8) /= transfer(xsum(k), 0_8)) shm_ndiff = shm_ndiff + 1
      enddo
    endif
#else
    xsum = x
#endif
  end subroutine comm_fs_sum

  subroutine comm_fs_sum_stats(on, cmp, ncall, ndiff)
    ! comm_fs_sum: in use, compare mode, calls through the window and bit differences against
    ! MPI_Allreduce (this rank)
    logical, intent(out) :: on, cmp
    integer(8), intent(out) :: ncall, ndiff
    on = shm_on; cmp = shm_cmp; ncall = shm_ncall; ndiff = shm_ndiff
  end subroutine comm_fs_sum_stats

  subroutine comm_allreduce_max(n, x, xmax)
    ! Max of x(1:n) over the compute ranks, on every compute rank.
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xmax(n)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Allreduce(x, xmax, n, MPI_DOUBLE_PRECISION, MPI_MAX, compute_comm, ierr)
#else
    xmax = x
#endif
  end subroutine comm_allreduce_max

  subroutine comm_allreduce_min(n, x, xmin)
    ! Min of x(1:n) over the compute ranks, on every compute rank.
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xmin(n)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Allreduce(x, xmin, n, MPI_DOUBLE_PRECISION, MPI_MIN, compute_comm, ierr)
#else
    xmin = x
#endif
  end subroutine comm_allreduce_min

  subroutine comm_allgather_i4(n, x, xall)
    ! Every compute rank's x(1:n) into xall(1:n, rank+1), on every rank
    ! (implicit free surface: the multigrid gather layout).
    integer, intent(in) :: n
    integer, intent(in) :: x(n)
    integer, intent(out) :: xall(n, *)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Allgather(x, n, MPI_INTEGER, xall, n, MPI_INTEGER, compute_comm, ierr)
#else
    xall(1:n, 1) = x(1:n)
#endif
  end subroutine comm_allgather_i4

  subroutine comm_allgatherv_r8(n, x, xall, cnt, dsp)
    ! Host MPI_Allgatherv of real(8) on the compute ranks (implicit free
    ! surface: the multigrid coarse gather). cnt, dsp per compute rank.
    integer, intent(in) :: n
    real(8), intent(in) :: x(*)
    real(8), intent(inout) :: xall(*)
    integer, intent(in) :: cnt(*), dsp(*)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Allgatherv(x, n, MPI_DOUBLE_PRECISION, xall, cnt, dsp, MPI_DOUBLE_PRECISION, &
                        compute_comm, ierr)
#else
    xall(1:n) = x(1:n)
#endif
  end subroutine comm_allgatherv_r8

  subroutine comm_allgatherv_dev(n, x, xall, cnt, dsp)
    ! Allgatherv of real(8) DEVICE buffers by point-to-point messages to and
    ! from every other compute rank (GPU-aware MPI, as the halo exchanges);
    ! this rank's own block is not copied (the caller places it). cnt, dsp
    ! per compute rank (dsp 0-based). The caller has synchronized the stream
    ! that wrote x. Tags 331 (implicit free surface: the multigrid gather).
    integer, intent(in) :: n
#ifdef USE_CUDA
    real(8), device, intent(in) :: x(*)
    real(8), device, intent(inout) :: xall(*)
#else
    real(8), intent(in) :: x(*)
    real(8), intent(inout) :: xall(*)
#endif
    integer, intent(in) :: cnt(*), dsp(*)
#ifdef USE_MPI
    integer :: ierr, k, nr, me, np
    integer, allocatable :: req(:), stats(:,:)
    if (.not. ready) call init_comm()
    me = int(my_rank); np = int(nproc)
    if (np <= 1) return
    allocate(req(2 * np), stats(MPI_STATUS_SIZE, 2 * np))
    nr = 0
    do k = 0, np - 1
      if (k == me) cycle
      nr = nr + 1
      call irecv_dev8(xall(dsp(k + 1) + 1), cnt(k + 1), k, 331, req(nr))
    enddo
    do k = 0, np - 1
      if (k == me) cycle
      nr = nr + 1
      call MPI_Isend(x, n, MPI_DOUBLE_PRECISION, k, 331, compute_comm, req(nr), ierr)
    enddo
    call MPI_Waitall(nr, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_allgatherv_dev: waitall'
    deallocate(req, stats)
#endif
  end subroutine comm_allgatherv_dev

  subroutine comm_ipc_exchange(nb, hb, hf, rhb, rhf, same)
    ! The implicit free surface's CUDA IPC rings (fs_ipc): send this rank's two 64-byte IPC
    ! handles and its host name to each MPI neighbour nb(d) (d = W, E, S, N; < 0: none) and
    ! receive theirs; same(d): neighbour d runs on this rank's node. (MPI here, not in the device
    ! modules: external MPI calls in the main compile command stop cross-file inlining.)
    integer, intent(in) :: nb(4)
    character(len=64), intent(in) :: hb, hf
    character(len=64), intent(out) :: rhb(4), rhf(4)
    logical, intent(out) :: same(4)
#ifdef USE_MPI
    character(len=MPI_MAX_PROCESSOR_NAME) :: me_host, nb_host(4)
    integer :: d, od, ierr, nr, ln, req(24), stats(MPI_STATUS_SIZE, 24)
    if (.not. ready) call init_comm()
    me_host = ' '
    call MPI_Get_processor_name(me_host, ln, ierr)
    if (ln < len(me_host)) me_host(ln + 1:) = ' '
    nb_host = ' '; rhb = ' '; rhf = ' '
    nr = 0
    do d = 1, 4
      if (nb(d) < 0) cycle
      nr = nr + 1; call MPI_Irecv(nb_host(d), MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, nb(d), 600 + d, compute_comm, req(nr), ierr)
      nr = nr + 1; call MPI_Irecv(rhb(d), 64, MPI_CHARACTER, nb(d), 610 + d, compute_comm, req(nr), ierr)
      nr = nr + 1; call MPI_Irecv(rhf(d), 64, MPI_CHARACTER, nb(d), 620 + d, compute_comm, req(nr), ierr)
    enddo
    do d = 1, 4
      if (nb(d) < 0) cycle
      od = merge(2, 1, d == 1)          ! the neighbour's direction toward this rank
      if (d == 3) od = 4
      if (d == 4) od = 3
      nr = nr + 1; call MPI_Isend(me_host, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, nb(d), 600 + od, compute_comm, req(nr), ierr)
      nr = nr + 1; call MPI_Isend(hb, 64, MPI_CHARACTER, nb(d), 610 + od, compute_comm, req(nr), ierr)
      nr = nr + 1; call MPI_Isend(hf, 64, MPI_CHARACTER, nb(d), 620 + od, compute_comm, req(nr), ierr)
    enddo
    call MPI_Waitall(nr, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_ipc_exchange: waitall'
    do d = 1, 4
      same(d) = (nb(d) >= 0 .and. nb_host(d) == me_host)
    enddo
#else
    rhb = ' '; rhf = ' '; same = .false.
#endif
  end subroutine comm_ipc_exchange

  subroutine comm_ipc_allgather(hb, hf, rhb, rhf, same)
    ! fs_ipc's gather: every compute rank's two 64-byte IPC handles (rhb(:, rank+1)); same: all
    ! compute ranks on this rank's node (host names equal)
    character(len=64), intent(in) :: hb, hf
    character(len=64), intent(out) :: rhb(*), rhf(*)
    logical, intent(out) :: same
#ifdef USE_MPI
    character(len=MPI_MAX_PROCESSOR_NAME) :: me_host
    character(len=MPI_MAX_PROCESSOR_NAME), allocatable :: hosts(:)
    integer :: ierr, ln, k
    if (.not. ready) call init_comm()
    me_host = ' '
    call MPI_Get_processor_name(me_host, ln, ierr)
    if (ln < len(me_host)) me_host(ln + 1:) = ' '
    allocate(hosts(nproc))
    call MPI_Allgather(me_host, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, hosts, MPI_MAX_PROCESSOR_NAME, &
         MPI_CHARACTER, compute_comm, ierr)
    call MPI_Allgather(hb, 64, MPI_CHARACTER, rhb, 64, MPI_CHARACTER, compute_comm, ierr)
    call MPI_Allgather(hf, 64, MPI_CHARACTER, rhf, 64, MPI_CHARACTER, compute_comm, ierr)
    same = .true.
    do k = 1, int(nproc)
      if (hosts(k) /= me_host) same = .false.
    enddo
    deallocate(hosts)
#else
    rhb(1) = hb; rhf(1) = hf; same = .true.
#endif
  end subroutine comm_ipc_allgather

  subroutine comm_allgatherv_h8(xall, cnt, dsp)
    ! comm_allgatherv_dev on one HOST buffer: this rank's block is already in place at
    ! xall(dsp(me+1)+1 ...); the other blocks are received into theirs (tag 331).
    real(8), intent(inout) :: xall(*)
    integer, intent(in) :: cnt(*), dsp(*)
#ifdef USE_MPI
    integer :: ierr, k, nr, me, np
    integer, allocatable :: req(:), stats(:,:)
    if (.not. ready) call init_comm()
    me = int(my_rank); np = int(nproc)
    if (np <= 1) return
    allocate(req(2 * np), stats(MPI_STATUS_SIZE, 2 * np))
    nr = 0
    do k = 0, np - 1
      if (k == me) cycle
      nr = nr + 1
      call MPI_Irecv(xall(dsp(k + 1) + 1), cnt(k + 1), MPI_DOUBLE_PRECISION, k, 331, compute_comm, req(nr), ierr)
    enddo
    do k = 0, np - 1
      if (k == me) cycle
      nr = nr + 1
      call MPI_Isend(xall(dsp(me + 1) + 1), cnt(me + 1), MPI_DOUBLE_PRECISION, k, 331, compute_comm, req(nr), ierr)
    enddo
    call MPI_Waitall(nr, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_allgatherv_h8: waitall'
    deallocate(req, stats)
#endif
  end subroutine comm_allgatherv_h8

  subroutine irecv_dev8(buf, n, src, tag, req)
    ! MPI_Irecv into a device buffer that starts inside a larger array (the
    ! explicit interface passes the element's device address)
    integer, intent(in) :: n, src, tag
#ifdef USE_CUDA
    real(8), device :: buf(*)
#else
    real(8) :: buf(*)
#endif
    integer, intent(out) :: req
#ifdef USE_MPI
    integer :: ierr
    call MPI_Irecv(buf, n, MPI_DOUBLE_PRECISION, src, tag, compute_comm, req, ierr)
#else
    req = 0
#endif
  end subroutine irecv_dev8

  subroutine comm_wait_dev(req)
    integer, intent(inout) :: req(4)
#ifdef USE_MPI
    integer :: ierr, stats(MPI_STATUS_SIZE, 4)
    call MPI_Waitall(4, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_wait_dev: waitall'
#endif
  end subroutine comm_wait_dev

  subroutine comm_exchange2(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), intent(in)    :: send_lo(:,:), send_hi(:,:)
    real(rp), intent(out)   :: recv_lo(:,:), recv_hi(:,:)
#ifdef USE_MPI
    integer :: n, dest_lo, dest_hi, ierr, stat(MPI_STATUS_SIZE)
    real(rp), allocatable :: s_lo(:), s_hi(:), r_lo(:), r_hi(:)
    if (.not. ready) call init_comm()
    n = size(send_lo)
    if (size(send_hi) /= n .or. size(recv_lo) /= n .or. size(recv_hi) /= n) &
         error stop 'comm_exchange2: buffer size mismatch'
    allocate(s_lo(n), s_hi(n), r_lo(n), r_hi(n))
    s_lo = reshape(send_lo, (/n/))
    s_hi = reshape(send_hi, (/n/))
    r_lo = 0.0_rp
    r_hi = 0.0_rp
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    call MPI_Sendrecv(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 101, &
                      r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 101, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange2: west/east slo'
    call MPI_Sendrecv(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 102, &
                      r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 102, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange2: east/west shi'
    recv_lo = reshape(r_lo, shape(recv_lo))
    recv_hi = reshape(r_hi, shape(recv_hi))
    deallocate(s_lo, s_hi, r_lo, r_hi)
#else
    recv_lo = 0.0_rp
    recv_hi = 0.0_rp
    if (nbr_lo >= 0 .or. nbr_hi >= 0) &
         error stop 'comm_exchange2: real MPI neighbor but this build has no USE_MPI'
#endif
  end subroutine comm_exchange2

  subroutine comm_exchange3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), intent(in)    :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), intent(out)   :: recv_lo(:,:,:), recv_hi(:,:,:)
#ifdef USE_MPI
    integer :: n, dest_lo, dest_hi, ierr, stat(MPI_STATUS_SIZE)
    real(rp), allocatable :: s_lo(:), s_hi(:), r_lo(:), r_hi(:)
    if (.not. ready) call init_comm()
    n = size(send_lo)
    if (size(send_hi) /= n .or. size(recv_lo) /= n .or. size(recv_hi) /= n) &
         error stop 'comm_exchange3: buffer size mismatch'
    allocate(s_lo(n), s_hi(n), r_lo(n), r_hi(n))
    s_lo = reshape(send_lo, (/n/))
    s_hi = reshape(send_hi, (/n/))
    r_lo = 0.0_rp
    r_hi = 0.0_rp
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    call MPI_Sendrecv(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 201, &
                      r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 201, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange3: slo'
    call MPI_Sendrecv(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 202, &
                      r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 202, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange3: shi'
    recv_lo = reshape(r_lo, shape(recv_lo))
    recv_hi = reshape(r_hi, shape(recv_hi))
    deallocate(s_lo, s_hi, r_lo, r_hi)
#else
    recv_lo = 0.0_rp
    recv_hi = 0.0_rp
    if (nbr_lo >= 0 .or. nbr_hi >= 0) &
         error stop 'comm_exchange3: real MPI neighbor but this build has no USE_MPI'
#endif
  end subroutine comm_exchange3

end module comm_mod

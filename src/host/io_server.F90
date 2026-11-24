#include "mem_attr.inc"
module io_server
  ! Asynchronous history output.
  !
  ! Compute rank: each output field is gathered on the GPU (on the work
  ! stream, so it is ordered after the step that produced it) into a
  ! float32 device buffer and copied asynchronously into one of two
  ! pinned host record buffers (double records: two pageable ones; four
  ! staging sets, two per type, so a history and a restart record never
  ! wait for each other's buffer); a CUDA event marks the record complete.
  ! io_poll (called every step) sends completed records to the I/O
  ! server rank with non-blocking MPI, or, without a server, writes them
  ! itself. The host never reads managed memory here, so no page
  ! migrations happen and the next step starts at once.
  !
  ! I/O rank: io_server_run loops on commands (OPEN file, PUT record,
  ! QUIT) and writes complete records in completion order (ready_q).
  ! While it writes one, io_progress (between variables, between pwrite
  ! pieces, at the group's closing barrier) takes in the commands that
  ! have arrived, so the compute ranks' sends of later records complete
  ! during the write (ROMS_GPU_IO_PROGRESS=0: only between writes). File
  ! creation (metadata) stays on the compute rank; the server only opens,
  ! appends records and closes.
  ! Commands from one compute rank arrive in order, but those of different
  ! ranks interleave freely: a fast rank's OPEN of file n+1 can arrive
  ! while a slow rank's part of the last record of file n is still on its
  ! way. The server therefore keys every OPEN by (kind, file number) and
  ! switches files only when a record of the next file is written.
  !
  ! Writer groups: the writer ranks are split into groups of
  ! ROMS_GPU_IO_GROUP writers (default: one group of all writers, at most
  ! MAXW; writer_groups). Writers are numbered round-robin over the
  ! nodes (comm_mod), so a group spans the nodes that hold writers. File
  ! nfile of kind fkind belongs to group mod(nfile-1 + fkind-1, ngroups)
  ! (CDR_LITE files: offset 1 with fewer than 3 groups; file_group),
  ! so with several groups a restart and the history file of the same
  ! number, and consecutive history files, are written by different groups
  ! at the same time. Records are dispatched in staging order on every
  ! compute rank (seq), so each group's writers complete its records in one
  ! order, as their group calls need.
  !
  ! Output is CDF5 (nc_io): the writers
  ! write the variable data of the shared file themselves with O_DIRECT
  ! (cdf_direct), not through netCDF: compute rank 0 creates the file
  ! (header only, nc_io); every writer of the group opens it at its first
  ! record (open_for: no netCDF handle, no collective call). Per record
  ! (write_direct) each writer assembles its variables, packs them
  ! big-endian into page-aligned buffers and writes whole 4-KiB pages; a
  ! page shared by two variables is written by the owner of its first
  ! variable, the other owners send it their bytes (heads, one small
  ! message per shared page); then a barrier of the group and writer 0
  ! rewrites the record count in the header. Nothing but these O_DIRECT
  ! calls touches the file while it is written (AGENTS.md: buffered bytes,
  ! netCDF handles, truncates or group locks next to lockless direct writes
  ! slowed or hung the benchmark). At a file switch and at quit the writers
  ! fsync and close, and writer 0 sets the length netCDF computes. Default:
  ! one group of all writers (ROMS_GPU_IO_GROUP overrides); a shared file
  ! written with O_DIRECT on a wide Lustre stripe takes 18-22 GB/s from 16
  ! writers on 2 nodes, two files at once the same in total
  ! (GPU_IO_STEP4_PLAN.md). ROMS_GPU_IO_DIRECT=0: the same byte ranges
  ! through buffered descriptors; ROMS_GPU_IO_PIECE_MB: pwrite size (1024);
  ! ROMS_GPU_IO_WATCHDOG: seconds before a pwrite that has not returned is
  ! reported on stderr (60; 0 off).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use nc_io
  use comm_mod
  use run_timers, only: wall_time
  use cdf_direct, only: cdf_file_t, cdf_buf_t, cdf_open_w, cdf_close_w, cdf_var_index, cdf_calcsize, &
                        cdf_set_numrecs, cdf_write, cdf_read_page, cdf_pack_r4, cdf_pack_r8, cdf_pack_i4, cdf_plan, &
                        cdf_truncate, cdf_buf_reserve, cdf_buf_free, cdf_buf_copy, cdf_watchdog_set, CDF_INT, &
                        CDF_FLOAT, CDF_DOUBLE
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private

  ! IO_KIND_CDR: the CDR_LITE output file (_cdrtrc, double; its records
  ! may carry avg_begin_time / avg_end_time). IO_KIND_CDRGAS: the CDR
  ! gas-exchange output (_cdrgas, double; records also carry the forcing
  ! time of ddic_dco2_time / ddic_dalk_time).
  ! IO_KIND_CDROUT: the CDR output (cdr_output, _cdr, double).
  integer, parameter, public :: IO_KIND_HIS = 1, IO_KIND_RST = 2, IO_KIND_CDR = 3, IO_KIND_CDRGAS = 4
  integer, parameter, public :: IO_KIND_CDROUT = 5
  ! IO_KIND_AVG / IO_KIND_BGCAVG: averaged history (avg_out) and BGC averages (bgc_out), double.
  integer, parameter, public :: IO_KIND_AVG = 6, IO_KIND_BGCAVG = 7
  ! IO_KIND_RND: the random output (random_out, _rnd; surface passive tracers).
  integer, parameter, public :: IO_KIND_RND = 8
  ! IO_KIND_BGC, IO_KIND_BGCDIA, IO_KIND_BGCDIAAVG: BGC tracers, MARBL diagnostics
  ! and their averages (bgc_out, double).
  integer, parameter, public :: IO_KIND_BGC = 9, IO_KIND_BGCDIA = 10, IO_KIND_BGCDIAAVG = 11
  integer, parameter :: NKIND = 11
  integer, parameter :: CMD_OPEN = 1, CMD_PUT = 2, CMD_QUIT = 3
  integer, parameter :: TAG_CMD = 900, TAG_STR = 901, TAG_TIME = 902, TAG_DATA = 910
  integer, parameter :: MAXV = 256, NAMELEN = 32, PATHLEN = 512   ! variables per record: up to ~240 tracers
  integer, parameter :: MAXW = 64                                 ! max writer ranks per group
  integer, parameter :: NHDR = 8 + 6 * MAXV
  ! Staged records in flight: sets 1-2 hold float records (host buffers
  ! hpin1, hpin2), sets 3-4 double records (hpin8_1, hpin8_2), so a
  ! history record and a restart record never wait for each other's buffer.
  integer, parameter :: NSET = 4

  type rec_t
    logical :: used = .false., sent = .false.
    integer :: fkind = 0, rec = 0, nfile = 0, nvars = 0, nfloat = 0
    integer :: seq = 0                       ! staging order (dispatch in this order)
    integer :: tstep = 0                     ! model step (iic) of a restart record, for time_step
    logical :: dbl = .false.                 ! real(8) record (restart)
    real(rp) :: time = 0.0_rp
    logical :: has_tavg = .false.            ! avg_begin_time, avg_end_time in tavg
    real(rp) :: tavg(2) = 0.0_rp
    logical :: has_tforce = .false.          ! ddic_dco2_time, ddic_dalk_time = tforce [days]
    real(rp) :: tforce = 0.0_rp
    real(8) :: tmsg(4) = 0.0d0               ! time, tavg and tforce, sent together
    character(len=PATHLEN) :: open_path = ''
    character(len=NAMELEN) :: names(MAXV) = ''
    integer :: n1(MAXV) = 0, n2(MAXV) = 0, n3(MAXV) = 0, nrank(MAXV) = 0, off(MAXV) = 0
    integer :: i0f(MAXV) = 1, j0f(MAXV) = 1   ! file start of this tile's slab (global offsets)
    integer :: nreq = 0
    integer :: reqs(MAXV + 5 * MAXW) = 0
    ! header messages are non-blocking too (a blocking header send to a
    ! writer busy in a collective deadlocked the writers); their buffers
    ! live here until wait_sends
    integer :: hdr(NHDR) = 0, ohdr(NHDR) = 0
    character(len=NAMELEN * MAXV) :: namebuf = ''
  end type rec_t

  type(rec_t) :: pend(NSET)
  real :: t_ext = 0.0, t_asm = 0.0, t_put = 0.0   ! writer timing of the last record (write_direct)
  integer :: cur = 0                 ! set being filled
  integer :: stage_seq = 0           ! records staged so far (compute side)
  integer :: wgroup_size = 0, wgroups = 1   ! writers per group, number of groups (both sides)
  integer :: me_g = 0, my_group = 0, gcomm = 0   ! server side: index in my group, my group, its communicator
  integer :: hcap = 0, dcap = 0
  integer :: hcap8 = 0, dcap8 = 0
  ! Server side: every tile's part of a record is received into its own
  ! buffer; when all parts of a record are in, each variable is assembled
  ! into one global array and written once at full extent (no tile waits
  ! behind another tile's write, no partial-chunk rewrites). Several record
  ! slots so records can arrive while earlier ones are being written. A step
  ! where every output falls due (restart, averages, _bgc, _bgc_avg, _bgc_dia,
  ! _bgc_dia_avg, _cdrtrc, _cdrgas average and end bracket: nine records) must fit: with several
  ! writers an overrun stops the run. A slot's buffers grow to the largest
  ! record it has held, so idle slots cost little.
  integer, parameter :: NSLOT = 12                 ! records assembled at once (per-tile buffers)
  integer :: nslot_use = NSLOT                     ! ROMS_GPU_IO_NSLOT (1..NSLOT): fewer, to test the overrun path
  type :: tile_buf_t
    real(4), allocatable :: b(:)
    real(8), allocatable :: b8(:)
  end type tile_buf_t
  type(tile_buf_t), allocatable :: tbuf(:,:)        ! (0:ncompute-1, NSLOT)
  type(rec_t), allocatable :: tmeta(:,:)            ! (0:ncompute-1, NSLOT)
  logical, allocatable :: tfill(:,:)                ! (0:ncompute-1, NSLOT)
  integer :: slot_key(3, NSLOT) = -1                ! (fkind, nfile, rec) per slot
  integer :: slot_n(NSLOT) = 0                      ! parts received per slot
  integer :: nserv = 1, me_w = 0                     ! writer ranks; this writer's index (server side)
  real(4), allocatable :: gbuf(:)                   ! one assembled variable
  real(8), allocatable :: gbuf8(:)                  ! one assembled double variable
  logical :: quit_sent = .false.   ! CMD_QUIT sent to the I/O rank (io_finalize may run twice)
#ifdef USE_CUDA
  real(4), pinned, allocatable :: hpin1(:), hpin2(:)
  real(4), device, allocatable :: dstage(:)
  real(8), allocatable :: hpin8_1(:), hpin8_2(:)   ! pageable: restarts are rare, pinning 2 sets cost ~10 s at init
  real(8), device, allocatable :: dstage8(:)
  type(cudaEvent) :: ev(NSET)
#else
  real(4), allocatable :: hpin1(:), hpin2(:), dstage(:)
  real(8), allocatable :: hpin8_1(:), hpin8_2(:), dstage8(:)
#endif
  ! Compute-side fallback (no server): current file per kind.
  character(len=PATHLEN) :: fname_sync(NKIND) = ''
  ! Server side.
  real(4), allocatable :: rbuf(:)
  integer :: nfile_open(NKIND) = -1                ! file number open per kind (-1: none)
  ! Announced files not yet opened: (kind, nfile) and path. A kind has at
  ! most a few files announced ahead of its writes (each compute rank holds
  ! at most NSET staged records).
  integer, parameter :: NANN = 24
  integer :: ann_key(2, NANN) = -1
  character(len=PATHLEN) :: ann_path(NANN) = ''
  ! Server side, CDF5 output (write_direct): the open file per kind, its
  ! path, the aligned page buffer, the heads of shared pages (sent and
  ! received), the options and the timing of the last record.
  integer, parameter :: MAXSEG = MAXV + 8
  logical :: io_direct = .true.
  type(cdf_file_t) :: dfile(NKIND)
  character(len=PATHLEN) :: opath(NKIND) = ''
  type(cdf_buf_t) :: pbuf, hsend(MAXSEG), hrecv(MAXSEG)
  integer(8) :: io_piece = 1073741824_8
  real :: t_sync = 0.0, t_pack = 0.0, t_wr = 0.0
  integer(8) :: d_bytes = 0
  integer :: d_rmw = 0, d_heads = 0

  public :: io_init
  public :: io_stage_begin
  public :: io_init8, io_stage8_begin, io_stage8_2d, io_stage8_3d, io_stage8_3d_pt, io_stage8_end
  public :: io_stage8_line
  public :: io_stage8_vec
  public :: io_stage_2d
  public :: io_stage_3d
  public :: io_stage_3d_p, io_stage_3d_pt
  public :: io_stage_omega
  public :: io_stage_end
  public :: io_poll
  public :: io_flush
  public :: io_finalize
  public :: io_server_run

  ! Server side, the main loop (io_server_run) and what io_progress takes in
  ! while a record is written: complete records queued in completion order,
  ! quits counted, receive time per slot. ROMS_GPU_IO_PROGRESS=0: commands
  ! only between writes (the former behaviour; a test setting).
  integer :: srv_comm = 0, srv_ncompute = 0, srv_nquit = 0
  integer :: ready_q(NSLOT) = 0, nready = 0
  real :: trecv(NSLOT) = 0.0, t_prog = 0.0
  logical :: in_progress = .false., srv_progress = .true.
  integer :: n_early = 0
contains

  subroutine io_init(cap_record, cap_field)
    ! cap_record: floats per staged record; cap_field: floats in the
    ! largest single field.
    integer, intent(in) :: cap_record, cap_field
#ifdef USE_CUDA
    integer :: ierr, i
#endif
    if (hcap > 0) return
    hcap = max(cap_record, 1)
    dcap = max(cap_field, 1)
    ! Set 2 is pinned only when a record finds set 1 still in flight
    ! (io_stage_begin): pinning a set costs seconds per rank at init.
    allocate(hpin1(hcap), dstage(dcap))
#ifdef USE_CUDA
    do i = 1, NSET
      ierr = cudaEventCreateWithFlags(ev(i), cudaEventDisableTiming)
      if (ierr /= 0) error stop 'io_init: cudaEventCreate'
    enddo
#endif
    if (have_io_server()) then
      write(*,'(a,i0,a,f6.2,a)') 'io: asynchronous output to server rank ', &
           io_server_rank(), '  (staging ', real(hcap) * 4.0 / 2.0**30, ' GB pinned, a second set when needed)'
    else
      write(*,'(a)') 'io: no server rank; staged records are written by the compute rank'
    endif
  end subroutine io_init

  subroutine io_stage_begin(fkind, rec, nfile, time, open_path, dbl, tstep, tavg, tforce)
    integer, intent(in) :: fkind, rec, nfile
    real(rp), intent(in) :: time
    character(len=*), intent(in) :: open_path
    logical, intent(in), optional :: dbl
    integer, intent(in), optional :: tstep
    real(rp), intent(in), optional :: tavg(2), tforce
    integer :: idx
    logical :: is_dbl
    if (hcap == 0) error stop 'io_stage_begin: io_init not called'
    is_dbl = .false.
    if (present(dbl)) is_dbl = dbl
    ! The older of the record type's two sets (an unused one first): float
    ! records in sets 1-2, double records in sets 3-4 (their own buffers,
    ! so a restart record still sending does not hold up the next history
    ! record, nor the other way round).
    if (is_dbl) then
      idx = 3
      if (pend(4)%seq < pend(3)%seq) idx = 4
    else
      idx = 1
      if (pend(2)%seq < pend(1)%seq) idx = 2
      if (idx == 2 .and. .not. allocated(hpin2)) then
        ! Reuse set 1 when it is already free (the usual case for daily
        ! output); otherwise pin set 2 now.
        if (set_free(1)) then
          idx = 1
        else
          allocate(hpin2(hcap))
        endif
      endif
    endif
    ! The set must be free: dispatched (with every older record, in staging
    ! order) and its sends complete.
    if (pend(idx)%used .and. .not. pend(idx)%sent) call dispatch_through(pend(idx)%seq, .true.)
    call wait_sends(idx)
    pend(idx) = rec_t()
    pend(idx)%used = .true.
    stage_seq = stage_seq + 1
    pend(idx)%seq = stage_seq
    pend(idx)%fkind = fkind
    pend(idx)%rec = rec
    pend(idx)%nfile = nfile
    pend(idx)%time = time
    pend(idx)%open_path = open_path
    if (present(tstep)) pend(idx)%tstep = tstep
    if (present(tavg)) then
      pend(idx)%has_tavg = .true.
      pend(idx)%tavg = tavg
    endif
    if (present(tforce)) then
      pend(idx)%has_tforce = .true.
      pend(idx)%tforce = tforce
    endif
    cur = idx
  end subroutine io_stage_begin

  subroutine add_var(name, n1, n2, n3, nrank, off, di, dj)
    ! di, dj: the slab's local start minus the local index that maps to file
    ! index decomp%i0 / decomp%j0 (0 for a slab starting at rho point 0 or
    ! u/v point 1; 1 for a rho slab starting at point 1 inside a tiled grid).
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3, nrank
    integer, intent(out) :: off
    integer, intent(in), optional :: di, dj
    integer :: iv, n
    type(rec_t), pointer :: r
    r => null()
    iv = pend(cur)%nvars + 1
    if (iv > MAXV) error stop 'io_stage: too many variables (raise MAXV)'
    if (len_trim(name) > NAMELEN) error stop 'io_stage: variable name longer than NAMELEN'
    n = n1 * n2 * n3
    off = pend(cur)%nfloat + 1
    if (pend(cur)%dbl) then
      if (off + n - 1 > hcap8) error stop 'io_stage8: record exceeds the double staging capacity'
      if (n > dcap8) error stop 'io_stage8: field exceeds the double device staging capacity'
    else
      if (off + n - 1 > hcap) error stop 'io_stage: record exceeds staging capacity'
      if (n > dcap) error stop 'io_stage: field exceeds device staging capacity'
    endif
    pend(cur)%nvars = iv
    pend(cur)%names(iv) = name
    pend(cur)%n1(iv) = n1
    pend(cur)%n2(iv) = n2
    pend(cur)%n3(iv) = n3
    pend(cur)%nrank(iv) = nrank
    pend(cur)%off(iv) = off
    pend(cur)%i0f(iv) = int(decomp%i0)
    pend(cur)%j0f(iv) = int(decomp%j0)
    if (present(di)) pend(cur)%i0f(iv) = pend(cur)%i0f(iv) + di
    if (present(dj)) pend(cur)%j0f(iv) = pend(cur)%j0f(iv) + dj
    ! A periodic direction has no boundary row in the file: interior rho point
    ! 1 (di = 1) is file index decomp%i0, like u point 1 (di = 0).
    if (decomp%periodic_x .and. present(di)) pend(cur)%i0f(iv) = int(decomp%i0) + max(di - 1, 0)
    if (decomp%periodic_y .and. present(dj)) pend(cur)%j0f(iv) = int(decomp%j0) + max(dj - 1, 0)
    pend(cur)%nfloat = pend(cur)%nfloat + n
  end subroutine add_var

  subroutine copy_to_host(off, n)
    ! dstage(1:n) -> pinned set `cur` at off, asynchronously on work_stream.
    integer, intent(in) :: off, n
#ifdef USE_CUDA
    integer :: ierr
    if (mod(cur, 2) == 1) then
      ierr = cudaMemcpyAsync(hpin1(off:off+n-1), dstage(1:n), n, cudaMemcpyDeviceToHost, work_stream)
    else
      ierr = cudaMemcpyAsync(hpin2(off:off+n-1), dstage(1:n), n, cudaMemcpyDeviceToHost, work_stream)
    endif
    if (ierr /= 0) then
      write(*,*) 'io_stage: cudaMemcpyAsync failed', ierr
      error stop 'io_stage'
    endif
#else
    if (mod(cur, 2) == 1) then
      hpin1(off:off+n-1) = dstage(1:n)
    else
      hpin2(off:off+n-1) = dstage(1:n)
    endif
#endif
  end subroutine copy_to_host

  subroutine io_stage_2d(name, a, i0, i1, j0, j1, di, dj)
    ! a is the first element of a (is:ie, js:je) slab (sequence association).
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer, intent(in) :: i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    integer :: i, j, ni, nj, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    call add_var(name, ni, nj, 1, 2, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        dstage((j - j0) * ni + (i - i0) + 1) = real(a(i, j), 4)
      enddo
    enddo
    call copy_to_host(off, ni * nj)
  end subroutine io_stage_2d

  subroutine io_stage_3d(name, a, k0, k1, i0, i1, j0, j1, di, dj)
    ! a is the first element of a (is:ie, js:je, k0:k1) slab.
    character(len=*), intent(in) :: name
    integer, intent(in) :: k0, k1, i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var(name, ni, nj, nk, 3, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(i, j, k), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * nk)
  end subroutine io_stage_3d

  subroutine io_stage_3d_pt(name, a, ntr, itr, k0, k1, i0, i1, j0, j1, di, dj)
    ! Tracer itr of the passive array slot (ntr, 0:ntpcol, k0:k1), gathered
    ! to the (i, j, k) window through tpcol.
    character(len=*), intent(in) :: name
    integer, intent(in) :: ntr, itr, k0, k1, i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    ! Compressed passive layout (gpu_state tpcol): unstored points read column 0 (zero).
    real(pp), WORK_ATTR intent(in) :: a(ntr, 0:ntpcol, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var(name, ni, nj, nk, 3, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(itr, tpcol(i, j), k), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * nk)
  end subroutine io_stage_3d_pt

  subroutine io_stage_3d_p(name, a, k0, k1, i0, i1, j0, j1, di, dj)
    ! As io_stage_3d for a real(pp) (passive tracer) slab.
    character(len=*), intent(in) :: name
    integer, intent(in) :: k0, k1, i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    real(pp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var(name, ni, nj, nk, 3, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(i, j, k), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * nk)
  end subroutine io_stage_3d_p

  subroutine io_stage_omega(name, i0, i1, j0, j1, di, dj)
    ! omega = pm*pn*(We+Wi) on w-levels 0:nz (as the former host put_omega).
    character(len=*), intent(in) :: name
    integer, intent(in) :: i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    integer :: i, j, k, ni, nj, nz, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nz = decomp%nz
    call add_var(name, ni, nj, nz + 1, 3, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = 0, nz
      do j = j0, j1
        do i = i0, i1
          dstage(k * ni * nj + (j - j0) * ni + (i - i0) + 1) = &
               real(pm(i, j) * pn(i, j) * (We(i, j, k) + Wi(i, j, k)), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * (nz + 1))
  end subroutine io_stage_omega

  subroutine io_stage_end()
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaEventRecord(ev(cur), work_stream)
    if (ierr /= 0) error stop 'io_stage_end: cudaEventRecord'
#endif
    pend(cur)%sent = .false.
  end subroutine io_stage_end

  logical function set_ready(idx)
    integer, intent(in) :: idx
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaEventQuery(ev(idx))
    if (ierr == 0) then
      set_ready = .true.
    elseif (ierr == cudaErrorNotReady) then
      set_ready = .false.
    else
      write(*,*) 'io_poll: cudaEventQuery failed', ierr
      error stop 'io_poll'
    endif
#else
    set_ready = .true.
#endif
  end function set_ready

  subroutine io_poll()
    ! Called every step: hand over records whose GPU copies have landed,
    ! oldest first; a record waits for every older one (same send order on
    ! every compute rank, whatever set each record landed in).
    integer :: i
    do
      i = oldest_pending()
      if (i == 0) exit
      if (.not. set_ready(i)) exit
      call dispatch(i, .false.)
    enddo
  end subroutine io_poll

  subroutine io_flush()
    integer :: i
    call dispatch_through(huge(1), .true.)
    do i = 1, NSET
      call wait_sends(i)
    enddo
  end subroutine io_flush

  integer function oldest_pending() result(idx)
    ! The staged, not yet dispatched record with the smallest seq (0: none).
    integer :: i
    idx = 0
    do i = 1, NSET
      if (pend(i)%used .and. .not. pend(i)%sent) then
        if (idx == 0) then
          idx = i
        elseif (pend(i)%seq < pend(idx)%seq) then
          idx = i
        endif
      endif
    enddo
  end function oldest_pending

  subroutine dispatch_through(seq, block)
    ! Dispatch, in staging order, every pending record with seq <= seq.
    integer, intent(in) :: seq
    logical, intent(in) :: block
    integer :: i
    do
      i = oldest_pending()
      if (i == 0) exit
      if (pend(i)%seq > seq) exit
      call dispatch(i, block)
    enddo
  end subroutine dispatch_through

  subroutine writer_groups(nw, gs, ng)
    ! Writers per group and number of groups for nw writers (the same on
    ! the compute and the writer ranks): ROMS_GPU_IO_GROUP, default one
    ! group of all writers (at most MAXW: direct writes of one shared file
    ! take every writer's bandwidth, two files at once as much in total as
    ! one). Lowered until it divides nw.
    integer, intent(in) :: nw
    integer, intent(out) :: gs, ng
    character(len=16) :: val
    integer :: stat, gdef
    gdef = min(max(nw, 1), MAXW)
    gs = gdef
    call get_environment_variable('ROMS_GPU_IO_GROUP', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) gs
      if (stat /= 0) gs = gdef
    endif
    gs = max(1, min(gs, max(nw, 1)))
    do while (mod(max(nw, 1), gs) /= 0)
      gs = gs - 1
    enddo
    ng = max(nw, 1) / gs
  end subroutine writer_groups

  integer function file_group(fkind, nfile)
    ! Writer group of file nfile of kind fkind. Each kind starts in its own
    ! group where there are enough; with two groups the CDR_LITE files share
    ! the restarts' (rare) group, away from the history: on one group,
    ! history and CDR_LITE records together outran the writer.
    integer, intent(in) :: fkind, nfile
    integer :: koff
    koff = fkind - 1
    if (fkind >= IO_KIND_CDR .and. wgroups < 3) koff = 1
    file_group = mod(max(nfile, 1) - 1 + koff, wgroups)
  end function file_group

  subroutine io_finalize()
#ifdef USE_MPI
    integer :: hdr(NHDR), ierr, w
#endif
    ! The I/O rank waits for commands from start-up, so it must be told to
    ! quit even when no history was ever staged (hcap == 0); otherwise the
    ! run hangs at exit with wrt_file_his = .false.
    if (hcap > 0) call io_flush()
#ifdef USE_MPI
    if (have_io_server() .and. .not. is_io_rank() .and. .not. quit_sent) then
      hdr = 0
      hdr(1) = CMD_QUIT
      do w = 0, io_server_count() - 1
        call MPI_Send(hdr, NHDR, MPI_INTEGER, comm_writer_world(w), TAG_CMD, comm_world(), ierr)
      enddo
      quit_sent = .true.
    endif
#endif
    hcap = 0
  end subroutine io_finalize

  subroutine dispatch(idx, block)
    integer, intent(in) :: idx
    logical, intent(in) :: block
#ifdef USE_CUDA
    integer :: ierr
    if (block) then
      ierr = cudaEventSynchronize(ev(idx))
      if (ierr /= 0) error stop 'io: cudaEventSynchronize'
    endif
#endif
    if (have_io_server()) then
      call send_set(idx)
    else
      call write_set_local(idx)
    endif
    pend(idx)%sent = .true.
  end subroutine dispatch

  logical function set_free(idx)
    ! Set idx holds no record, or its record is handed over and every send
    ! of it has completed (tested without blocking).
    integer, intent(in) :: idx
#ifdef USE_MPI
    integer :: ierr
    logical :: flag
#endif
    set_free = .true.
    if (.not. pend(idx)%used) return
    set_free = .false.
    if (.not. pend(idx)%sent) return
#ifdef USE_MPI
    if (pend(idx)%nreq > 0) then
      call MPI_Testall(pend(idx)%nreq, pend(idx)%reqs, flag, MPI_STATUSES_IGNORE, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'io: MPI_Testall'
      if (.not. flag) return
      pend(idx)%nreq = 0
    endif
#endif
    set_free = .true.
  end function set_free

  subroutine wait_sends(idx)
    integer, intent(in) :: idx
#ifdef USE_MPI
    integer :: ierr
    if (pend(idx)%nreq > 0) then
      call MPI_Waitall(pend(idx)%nreq, pend(idx)%reqs, MPI_STATUSES_IGNORE, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'io: MPI_Waitall'
    endif
#endif
    pend(idx)%nreq = 0
  end subroutine wait_sends

  subroutine put_time_step(ncid, r)
    ! Restart records: time_step(1:6, rec) = (iic, rec, rec, 0, 0, 0), the CPU
    ! model's layout; a reader takes ntstart = time_step(1) + 1 and checks
    ! that the two records of an exact restart are one step apart.
    integer, intent(in) :: ncid
    type(rec_t), intent(in) :: r
    if (r%fkind /= IO_KIND_RST .or. r%tstep <= 0) return
    if (.not. nc_has_var(ncid, 'time_step')) return
    call nc_put_ints_rec(ncid, 'time_step', (/r%tstep, r%rec, r%rec, 0, 0, 0/), r%rec)
  end subroutine put_time_step

  subroutine put_avg_times(ncid, r)
    ! Averaging records (CDR output): avg_begin_time, avg_end_time; CDR
    ! gas-exchange records: the forcing time ddic_dco2_time, ddic_dalk_time.
    integer, intent(in) :: ncid
    type(rec_t), intent(in) :: r
    if (r%has_tavg) then
      if (nc_has_var(ncid, 'avg_begin_time')) call nc_put_1d_rec(ncid, 'avg_begin_time', r%tavg(1), r%rec)
      if (nc_has_var(ncid, 'avg_end_time')) call nc_put_1d_rec(ncid, 'avg_end_time', r%tavg(2), r%rec)
    endif
    if (r%has_tforce) then
      if (nc_has_var(ncid, 'ddic_dco2_time')) call nc_put_1d_rec(ncid, 'ddic_dco2_time', r%tforce, r%rec)
      if (nc_has_var(ncid, 'ddic_dalk_time')) call nc_put_1d_rec(ncid, 'ddic_dalk_time', r%tforce, r%rec)
    endif
  end subroutine put_avg_times

  subroutine pack_hdr(r, hdr)
    type(rec_t), intent(in) :: r
    integer, intent(out) :: hdr(NHDR)
    integer :: iv, b
    hdr = 0
    hdr(1) = CMD_PUT
    hdr(2) = r%fkind
    hdr(3) = r%rec
    hdr(4) = r%nvars
    hdr(5) = r%tstep
    hdr(6) = r%nfile
    hdr(7) = merge(1, 0, r%dbl)
    hdr(8) = merge(1, 0, r%has_tavg) + merge(2, 0, r%has_tforce)
    do iv = 1, r%nvars
      b = 8 + 6 * (iv - 1)
      hdr(b + 1) = r%n1(iv)
      hdr(b + 2) = r%n2(iv)
      hdr(b + 3) = r%n3(iv)
      hdr(b + 4) = r%nrank(iv)
      hdr(b + 5) = r%i0f(iv)
      hdr(b + 6) = r%j0f(iv)
    enddo
  end subroutine pack_hdr

  subroutine send_set(idx)
    integer, intent(in) :: idx
#ifdef USE_MPI
    integer :: ierr, iv, n, dest, comm, nw, w, nr, base
    comm = comm_world()
    nw = io_server_count()
    if (wgroup_size == 0) call writer_groups(nw, wgroup_size, wgroups)
    if (wgroup_size > MAXW) error stop 'io: too many writers per group (raise MAXW)'
    ! the writers of this file's group: writers base .. base + wgroup_size - 1
    base = file_group(pend(idx)%fkind, pend(idx)%nfile) * wgroup_size
    call pack_hdr(pend(idx), pend(idx)%hdr)
    pend(idx)%namebuf = ''
    do iv = 1, pend(idx)%nvars
      pend(idx)%namebuf((iv - 1) * NAMELEN + 1:iv * NAMELEN) = pend(idx)%names(iv)
    enddo
    ! data first (variable iv to the group's writer mod(iv-1, size)), all non-blocking
    do iv = 1, pend(idx)%nvars
      dest = comm_writer_world(base + mod(iv - 1, wgroup_size))
      n = pend(idx)%n1(iv) * pend(idx)%n2(iv) * pend(idx)%n3(iv)
      if (pend(idx)%dbl) then
        if (mod(idx, 2) == 1) then
          call MPI_Isend(hpin8_1(pend(idx)%off(iv)), n, MPI_REAL8, dest, TAG_DATA + iv, comm, &
                         pend(idx)%reqs(iv), ierr)
        else
          call MPI_Isend(hpin8_2(pend(idx)%off(iv)), n, MPI_REAL8, dest, TAG_DATA + iv, comm, &
                         pend(idx)%reqs(iv), ierr)
        endif
      elseif (mod(idx, 2) == 1) then
        call MPI_Isend(hpin1(pend(idx)%off(iv)), n, MPI_REAL4, dest, TAG_DATA + iv, comm, &
                       pend(idx)%reqs(iv), ierr)
      else
        call MPI_Isend(hpin2(pend(idx)%off(iv)), n, MPI_REAL4, dest, TAG_DATA + iv, comm, &
                       pend(idx)%reqs(iv), ierr)
      endif
    enddo
    nr = pend(idx)%nvars
    ! the open command and the times are filled once, before any send: every writer's
    ! Isend reads the same buffers, which must not change until the sends complete
    pend(idx)%ohdr = 0
    pend(idx)%ohdr(1) = CMD_OPEN
    pend(idx)%ohdr(2) = pend(idx)%fkind
    pend(idx)%ohdr(3) = pend(idx)%nfile
    pend(idx)%tmsg = (/real(pend(idx)%time, 8), real(pend(idx)%tavg(1), 8), real(pend(idx)%tavg(2), 8), &
                       real(pend(idx)%tforce, 8)/)
    ! every writer of the group gets the (optional) open command and the header
    do w = 0, wgroup_size - 1
      dest = comm_writer_world(base + w)
      if (len_trim(pend(idx)%open_path) > 0) then
        nr = nr + 1
        call MPI_Isend(pend(idx)%ohdr, NHDR, MPI_INTEGER, dest, TAG_CMD, comm, pend(idx)%reqs(nr), ierr)
        nr = nr + 1
        call MPI_Isend(pend(idx)%open_path, PATHLEN, MPI_CHARACTER, dest, TAG_STR, comm, pend(idx)%reqs(nr), ierr)
      endif
      nr = nr + 1
      call MPI_Isend(pend(idx)%hdr, NHDR, MPI_INTEGER, dest, TAG_CMD, comm, pend(idx)%reqs(nr), ierr)
      nr = nr + 1
      call MPI_Isend(pend(idx)%namebuf, NAMELEN * MAXV, MPI_CHARACTER, dest, TAG_STR, comm, pend(idx)%reqs(nr), ierr)
      nr = nr + 1
      call MPI_Isend(pend(idx)%tmsg, 4, MPI_DOUBLE_PRECISION, dest, TAG_TIME, comm, pend(idx)%reqs(nr), ierr)
    enddo
    pend(idx)%nreq = nr
#else
    error stop 'io: server requested without MPI'
#endif
  end subroutine send_set

  logical function mine(iv)
    ! Variable iv of a record belongs to the group's writer mod(iv-1, size).
    integer, intent(in) :: iv
    mine = (mod(iv - 1, wgroup_size) == me_g)
  end function mine

  ! ------------------------- CDF5 output: direct writes -------------------------

  subroutine direct_options()
    ! writer side: O_DIRECT, piece size, watchdog, record slots (environment)
    character(len=32) :: val
    integer :: stat, n
    io_direct = nc_io_direct()
    call get_environment_variable('ROMS_GPU_IO_PIECE_MB', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) n
      if (stat == 0 .and. n > 0) io_piece = int(n, 8) * 1048576_8
    endif
    n = 60
    call get_environment_variable('ROMS_GPU_IO_WATCHDOG', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) n
      if (stat /= 0) n = 60
    endif
    call cdf_watchdog_set(n)
    call get_environment_variable('ROMS_GPU_IO_NSLOT', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0) then
      read(val, *, iostat=stat) n
      if (stat == 0) nslot_use = max(1, min(NSLOT, n))
    endif
  end subroutine direct_options

  subroutine direct_close(k)
    ! Close the direct descriptor of kind k's file on every writer of the
    ! group (fsync, close), then writer 0 sets the length netCDF computes
    ! (the last record's page was written whole). Collective over the group.
    integer, intent(in) :: k
    integer(8) :: csz
    integer :: ierr
    csz = cdf_calcsize(dfile(k))
    if (.not. cdf_close_w(dfile(k), .true.)) &
         write(*,'(3a)') 'io_server: WARNING fsync or close of ', trim(opath(k)), ' failed'
    if (wgroup_size > 1) call MPI_Barrier(gcomm, ierr)
    if (me_g == 0) then
      if (.not. cdf_truncate(trim(opath(k)), csz)) &
           write(*,'(3a)') 'io_server: WARNING setting the length of ', trim(opath(k)), ' failed'
    endif
  end subroutine direct_close

  subroutine var_extent(isl, iv, gx, gy)
    ! global extents of variable iv of slot isl from the tiles' slabs
    integer, intent(in) :: isl, iv
    integer, intent(out) :: gx, gy
    integer :: t
    gx = 0; gy = 0
    do t = 0, size(tbuf, 1) - 1
      if (trim(tmeta(t, isl)%names(iv)) /= trim(tmeta(0, isl)%names(iv))) &
           error stop 'io_server: tiles disagree on the variable list'
      gx = max(gx, tmeta(t, isl)%i0f(iv) + tmeta(t, isl)%n1(iv) - 1)
      gy = max(gy, tmeta(t, isl)%j0f(iv) + tmeta(t, isl)%n2(iv) - 1)
    enddo
  end subroutine var_extent

  subroutine assemble_var(isl, iv, gx, gy, n3, dbl)
    ! variable iv of slot isl from the tiles' slabs into gbuf (gbuf8: dbl)
    integer, intent(in) :: isl, iv, gx, gy, n3
    logical, intent(in) :: dbl
    integer :: t, i, j, k, n1, n2, i0, j0, off
    integer(8) :: ng
    ng = int(gx, 8) * gy * n3
    if (dbl) then
      if (allocated(gbuf8)) then
        if (size(gbuf8, kind=8) < ng) deallocate(gbuf8)
      endif
      if (.not. allocated(gbuf8)) allocate(gbuf8(max(ng, 1_8)))
    else
      if (allocated(gbuf)) then
        if (size(gbuf, kind=8) < ng) deallocate(gbuf)
      endif
      if (.not. allocated(gbuf)) allocate(gbuf(max(ng, 1_8)))
    endif
    do t = 0, size(tbuf, 1) - 1
      n1 = tmeta(t, isl)%n1(iv); n2 = tmeta(t, isl)%n2(iv)
      i0 = tmeta(t, isl)%i0f(iv); j0 = tmeta(t, isl)%j0f(iv); off = tmeta(t, isl)%off(iv)
      if (dbl) then
        do k = 1, n3
          do j = 1, n2
            do i = 1, n1
              gbuf8((k - 1) * gx * gy + (j0 + j - 2) * gx + (i0 + i - 1)) = &
                   tbuf(t, isl)%b8(off + (k - 1) * n1 * n2 + (j - 1) * n1 + (i - 1))
            enddo
          enddo
        enddo
      else
        do k = 1, n3
          do j = 1, n2
            do i = 1, n1
              gbuf((k - 1) * gx * gy + (j0 + j - 2) * gx + (i0 + i - 1)) = &
                   tbuf(t, isl)%b(off + (k - 1) * n1 * n2 + (j - 1) * n1 + (i - 1))
            enddo
          enddo
        enddo
      endif
    enddo
  end subroutine assemble_var

  subroutine pack_head(isl, iv, gx, gy, dbl, xtype, ne, b)
    ! The first ne elements (file order) of variable iv of slot isl, taken
    ! from the tiles' slabs (the last tile that holds an element wins, as in
    ! assemble_var; none: 0), packed as the file's type into b at byte 0:
    ! the bytes of a shared page that the page's writer gets from this owner.
    integer, intent(in) :: isl, iv, gx, gy, xtype
    integer(8), intent(in) :: ne
    logical, intent(in) :: dbl
    type(cdf_buf_t), intent(inout) :: b
    real(4) :: h4(ne)
    real(8) :: h8(ne)
    integer(8) :: e, kk, rem
    integer :: t, ii, jj, n1, n2, i0, j0
    if (dbl) then
      h8 = 0.0d0
    else
      h4 = 0.0
    endif
    do e = 0, ne - 1
      kk = e / (int(gx, 8) * gy)
      rem = e - kk * gx * gy
      jj = int(rem / gx) + 1
      ii = int(rem - (jj - 1) * int(gx, 8)) + 1
      do t = size(tbuf, 1) - 1, 0, -1
        n1 = tmeta(t, isl)%n1(iv); n2 = tmeta(t, isl)%n2(iv)
        i0 = tmeta(t, isl)%i0f(iv); j0 = tmeta(t, isl)%j0f(iv)
        if (ii < i0 .or. ii > i0 + n1 - 1 .or. jj < j0 .or. jj > j0 + n2 - 1) cycle
        if (dbl) then
          h8(e + 1) = tbuf(t, isl)%b8(tmeta(t, isl)%off(iv) + kk * n1 * n2 + (jj - j0) * n1 + (ii - i0))
        else
          h4(e + 1) = tbuf(t, isl)%b(tmeta(t, isl)%off(iv) + kk * n1 * n2 + (jj - j0) * n1 + (ii - i0))
        endif
        exit
      enddo
    enddo
    if (dbl) then
      call cdf_pack_r8(h8, ne, xtype, b, 0_8)
    else
      call cdf_pack_r4(h4, ne, xtype, b, 0_8)
    endif
  end subroutine pack_head

  subroutine write_direct(k, isl)
    ! Write record slot isl into the open CDF5 file of kind k (this writer's
    ! part). Segments: the record's variables (owner: the writer that
    ! received them, mine()) and the small ones writer 0 fills (ocean_time,
    ! time_step, the averaging / forcing times), at their byte ranges for
    ! this record (time-less variables in the fixed-size section). Every
    ! page is written whole, once, by its writing segment's owner
    ! (cdf_plan); bytes of a later segment in that page arrive as a head
    ! (MPI on the group, all sent before any write); a page partly covered
    ! by this record (the previous record's tail, unwritten variables) is
    ! read first (O_DIRECT). Then the group's barrier and writer 0's record
    ! count. Timing: t_ext plan + heads, t_asm assembly, t_put pack + pwrite,
    ! t_sync barrier + record count.
    integer, intent(in) :: k, isl
    type(rec_t) :: r0
    integer :: ns, iv, vi, s, t, i, j, gx, gy, n3, nreq, ierr, w, xt
    integer :: sk(MAXSEG), sv(MAXSEG), own(MAXSEG), wseg(MAXSEG), ord(MAXSEG), rq(MAXSEG)
    integer :: sgx(MAXSEG), sgy(MAXSEG), sn3(MAXSEG)
    integer(8) :: soff(MAXSEG), snb(MAXSEG), wp1(MAXSEG), wp2(MAXSEG)
    integer(8) :: a, fp, base, nbuf, cov, hb, ne, xsz, tmpo, tmpn
    integer :: reqs(2 * MAXSEG)
    integer :: tsv(6)
    real(8) :: c0, c1, c2, vals(1)
    logical :: rmw1, lok, bdone
    integer :: breq
    t_ext = 0.0; t_asm = 0.0; t_put = 0.0; t_sync = 0.0; t_pack = 0.0; t_wr = 0.0; t_prog = 0.0
    d_bytes = 0; d_rmw = 0; d_heads = 0
    c0 = wall_time()
    r0 = tmeta(0, isl)
    a = dfile(k)%align
    ! ---- segments
    ns = 0
    call add_small('ocean_time', -1)
    if (r0%fkind == IO_KIND_RST .and. r0%tstep > 0) call add_small('time_step', -2)
    if (r0%has_tavg) then
      call add_small('avg_begin_time', -3)
      call add_small('avg_end_time', -4)
    endif
    if (r0%has_tforce) then
      call add_small('ddic_dco2_time', -5)
      call add_small('ddic_dalk_time', -5)
    endif
    do iv = 1, r0%nvars
      vi = cdf_var_index(dfile(k), trim(r0%names(iv)))
      if (vi == 0) then
        write(*,'(4a)') 'io_server: variable ', trim(r0%names(iv)), ' is not in ', trim(opath(k))
        error stop 'io_server: staged variable not in the CDF5 file'
      endif
      call var_extent(isl, iv, gx, gy)
      n3 = r0%n3(iv)
      if (int(gx, 8) * gy * n3 * dfile(k)%v(vi)%xsz /= dfile(k)%v(vi)%vraw) then
        write(*,'(3a,i0,a,i0,a,i0,a,i0)') 'io_server: variable ', trim(r0%names(iv)), ': staged ', gx, ' x ', gy, &
             ' x ', n3, ', file bytes ', dfile(k)%v(vi)%vraw
        error stop 'io_server: staged extent differs from the CDF5 variable'
      endif
      ns = ns + 1
      if (ns > MAXSEG) error stop 'io_server: too many segments (raise MAXSEG)'
      sk(ns) = iv; sv(ns) = vi; own(ns) = mod(iv - 1, wgroup_size)
      sgx(ns) = gx; sgy(ns) = gy; sn3(ns) = n3
      soff(ns) = dfile(k)%v(vi)%begin
      if (dfile(k)%v(vi)%isrec) soff(ns) = soff(ns) + int(r0%rec - 1, 8) * dfile(k)%recsize
      snb(ns) = dfile(k)%v(vi)%vraw
    enddo
    ! sorted by offset (insertion sort of the order, then the arrays in that order)
    do i = 1, ns
      ord(i) = i
    enddo
    do i = 2, ns
      j = i
      do while (j > 1)
        if (soff(ord(j - 1)) <= soff(ord(j))) exit
        t = ord(j); ord(j) = ord(j - 1); ord(j - 1) = t
        j = j - 1
      enddo
    enddo
    sk(1:ns) = sk(ord(1:ns)); sv(1:ns) = sv(ord(1:ns)); own(1:ns) = own(ord(1:ns))
    sgx(1:ns) = sgx(ord(1:ns)); sgy(1:ns) = sgy(ord(1:ns)); sn3(1:ns) = sn3(ord(1:ns))
    soff(1:ns) = soff(ord(1:ns)); snb(1:ns) = snb(ord(1:ns))
    if (.not. cdf_plan(ns, soff, snb, a, wseg, wp1, wp2)) then
      write(*,'(2a)') 'io_server: overlapping or empty variables in ', trim(opath(k))
      error stop 'io_server: cdf_plan'
    endif
    ! ---- heads: a segment that does not write its first page sends those bytes to that page's writer
    nreq = 0
    rq = 0
    do s = 1, ns
      if (wp1(s) == soff(s) / a) cycle
      w = own(wseg(s))
      hb = min(snb(s), (soff(s) / a + 1) * a - soff(s))
      if (own(s) == me_g) then
        if (w == me_g) then
          call head_bytes(s, hb, hrecv(s))          ! my own later segment in my page
        else
          call head_bytes(s, hb, hsend(s))
          nreq = nreq + 1
          call MPI_Isend(hsend(s)%u8(1), int(hb), MPI_BYTE, w, 100 + s, gcomm, reqs(nreq), ierr)
          d_heads = d_heads + 1
        endif
      elseif (w == me_g) then
        if (.not. cdf_buf_reserve(hrecv(s), 8192_8)) error stop 'io_server: head buffer'
        nreq = nreq + 1
        call MPI_Irecv(hrecv(s)%u8(1), int(hb), MPI_BYTE, own(s), 100 + s, gcomm, reqs(nreq), ierr)
        rq(s) = nreq
      endif
    enddo
    c1 = wall_time()
    t_ext = real(c1 - c0)
    ! ---- my pages
    do s = 1, ns
      if (own(s) /= me_g .or. wp1(s) > wp2(s)) cycle
      call io_progress()                         ! later records' parts, between variables
      fp = soff(s) / a
      base = fp * a
      nbuf = (wp2(s) - fp + 1) * a
      if (.not. cdf_buf_reserve(pbuf, nbuf)) error stop 'io_server: page buffer'
      rmw1 = (wp1(s) == fp .and. soff(s) > base)
      if (rmw1) then
        if (.not. cdf_read_page(dfile(k), pbuf, 0_8, base, a)) error stop 'io_server: page read'
        d_rmw = d_rmw + 1
      endif
      cov = min(soff(s) + snb(s), (wp2(s) + 1) * a) - max(soff(s), wp2(s) * a)
      do t = s + 1, ns
        if (soff(t) >= (wp2(s) + 1) * a) exit
        if (wseg(t) == s) cov = cov + min(snb(t), (soff(t) / a + 1) * a - soff(t))
      enddo
      if (cov < a .and. .not. (rmw1 .and. wp2(s) == fp)) then
        if (.not. cdf_read_page(dfile(k), pbuf, (wp2(s) - fp) * a, wp2(s) * a, a)) error stop 'io_server: page read'
        d_rmw = d_rmw + 1
      endif
      c1 = wall_time()
      xt = dfile(k)%v(sv(s))%xtype
      if (sk(s) > 0 .and. ((r0%dbl .and. xt == CDF_DOUBLE) .or. (.not. r0%dbl .and. xt == CDF_FLOAT))) then
        ! no conversion: each tile's rows go straight from its slab into the page buffer, byte-swapped
        ! (one pass instead of assembling into gbuf and packing that: half the memory traffic)
        call pack_rows(s, pbuf, soff(s) - base)
        c2 = wall_time()
        t_asm = t_asm + real(c2 - c1)
        c1 = c2
      elseif (sk(s) > 0) then
        call assemble_var(isl, sk(s), sgx(s), sgy(s), sn3(s), r0%dbl)
        c2 = wall_time()
        t_asm = t_asm + real(c2 - c1)
        c1 = c2
        call pack_seg(s, pbuf, soff(s) - base)
      else
        call pack_seg(s, pbuf, soff(s) - base)
      endif
      do t = s + 1, ns
        if (soff(t) >= (wp2(s) + 1) * a) exit
        if (wseg(t) /= s) cycle
        hb = min(snb(t), (soff(t) / a + 1) * a - soff(t))
        if (rq(t) > 0) call MPI_Wait(reqs(rq(t)), MPI_STATUS_IGNORE, ierr)
        call cdf_buf_copy(pbuf, soff(t) - base, hrecv(t), 0_8, hb)   ! memcpy: both were packed as words
      enddo
      c2 = wall_time()
      t_pack = t_pack + real(c2 - c1)
      ! in pieces of io_piece, taking in later records between them
      tmpo = 0
      tmpn = (wp2(s) - wp1(s) + 1) * a
      do while (tmpo < tmpn)
        if (tmpo > 0) call io_progress()
        if (.not. cdf_write(dfile(k), pbuf, (wp1(s) - fp) * a + tmpo, wp1(s) * a + tmpo, min(io_piece, tmpn - tmpo), &
                            io_piece)) then
          write(*,'(3a)') 'io_server: direct write to ', trim(opath(k)), ' failed'
          error stop 'io_server: direct write'
        endif
        tmpo = tmpo + min(io_piece, tmpn - tmpo)
      enddo
      c1 = wall_time()
      t_wr = t_wr + real(c1 - c2)
      d_bytes = d_bytes + (wp2(s) - wp1(s) + 1) * a
    enddo
    t_put = t_pack + t_wr
    ! ---- every head sent and received, the group done, then the record count
    c1 = wall_time()
    if (nreq > 0) call MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE, ierr)
    if (wgroup_size > 1) then
      ! the group's barrier, taking in later records while the others finish
      call MPI_Ibarrier(gcomm, breq, ierr)
      do
        call MPI_Test(breq, bdone, MPI_STATUS_IGNORE, ierr)
        if (bdone) exit
        call io_progress()
      enddo
    endif
    if (me_g == 0 .and. dfile(k)%begin_rec >= 0 .and. int(r0%rec, 8) > dfile(k)%numrecs) then
      if (.not. cdf_set_numrecs(dfile(k), int(r0%rec, 8))) error stop 'io_server: record count write'
    endif
    t_sync = real(wall_time() - c1)
  contains
    subroutine add_small(name, code)
      ! a small variable writer 0 fills, if the file has it
      character(len=*), intent(in) :: name
      integer, intent(in) :: code
      integer :: v
      v = cdf_var_index(dfile(k), name)
      if (v == 0) return
      ns = ns + 1
      sk(ns) = code; sv(ns) = v; own(ns) = 0
      sgx(ns) = 0; sgy(ns) = 0; sn3(ns) = 0
      soff(ns) = dfile(k)%v(v)%begin
      if (dfile(k)%v(v)%isrec) soff(ns) = soff(ns) + int(r0%rec - 1, 8) * dfile(k)%recsize
      snb(ns) = dfile(k)%v(v)%vraw
      if (name == 'ddic_dalk_time') sk(ns) = -6
    end subroutine add_small

    subroutine small_vals(code, n, v8, vi4)
      ! the values of a small segment: real(8) (v8) or, for time_step, integers (vi4)
      integer, intent(in) :: code
      integer(8), intent(in) :: n
      real(8), intent(out) :: v8(n)
      integer, intent(out) :: vi4(n)
      v8 = 0.0d0
      vi4 = 0
      select case (code)
      case (-1)
        v8 = real(r0%time, 8)
      case (-2)
        tsv = (/r0%tstep, r0%rec, r0%rec, 0, 0, 0/)
        vi4(1:min(n, 6_8)) = tsv(1:min(n, 6_8))
      case (-3)
        v8 = real(r0%tavg(1), 8)
      case (-4)
        v8 = real(r0%tavg(2), 8)
      case default
        v8 = real(r0%tforce, 8)
      end select
    end subroutine small_vals

    subroutine pack_seg(s, b, boff)
      ! the whole segment s into b at byte boff, as the file's type
      integer, intent(in) :: s
      type(cdf_buf_t), intent(inout) :: b
      integer(8), intent(in) :: boff
      integer(8) :: n
      real(8), allocatable :: v8(:)
      integer, allocatable :: vi4(:)
      xt = dfile(k)%v(sv(s))%xtype
      n = snb(s) / dfile(k)%v(sv(s))%xsz
      if (sk(s) > 0) then
        if (r0%dbl) then
          call cdf_pack_r8(gbuf8, n, xt, b, boff)
        else
          call cdf_pack_r4(gbuf, n, xt, b, boff)
        endif
        return
      endif
      allocate(v8(n), vi4(n))
      call small_vals(sk(s), n, v8, vi4)
      if (xt == CDF_INT) then
        call cdf_pack_i4(vi4, n, b, boff)
      else
        call cdf_pack_r8(v8, n, xt, b, boff)
      endif
    end subroutine pack_seg

    subroutine pack_rows(s, b, boff)
      ! Segment s (a staged variable stored in the file's own type) into b at
      ! byte boff, row by row from the tiles' slabs: the global element of
      ! tile row (j, k) starting at i0 is (k-1)*gx*gy + (j0+j-2)*gx + (i0-1).
      ! The same values and order as assemble_var + pack_seg (later tiles
      ! overwrite earlier ones where slabs overlap).
      integer, intent(in) :: s
      type(cdf_buf_t), intent(inout) :: b
      integer(8), intent(in) :: boff
      integer :: t, iv, j, kk, n1, n2, i0, j0, off, gx, gy, sz
      integer(8) :: e, src
      iv = sk(s)
      gx = sgx(s); gy = sgy(s)
      sz = dfile(k)%v(sv(s))%xsz
      do t = 0, size(tbuf, 1) - 1
        n1 = tmeta(t, isl)%n1(iv); n2 = tmeta(t, isl)%n2(iv)
        i0 = tmeta(t, isl)%i0f(iv); j0 = tmeta(t, isl)%j0f(iv); off = tmeta(t, isl)%off(iv)
        if (n1 <= 0) cycle
        do kk = 1, sn3(s)
          do j = 1, n2
            e = int(kk - 1, 8) * gx * gy + int(j0 + j - 2, 8) * gx + (i0 - 1)
            src = int(off, 8) + int(kk - 1, 8) * n1 * n2 + int(j - 1, 8) * n1
            if (r0%dbl) then
              call cdf_pack_r8(tbuf(t, isl)%b8(src), int(n1, 8), xt, b, boff + e * sz)
            else
              call cdf_pack_r4(tbuf(t, isl)%b(src), int(n1, 8), xt, b, boff + e * sz)
            endif
          enddo
        enddo
      enddo
    end subroutine pack_rows

    subroutine head_bytes(s, nb, b)
      ! the first nb bytes of segment s (file representation) into b
      integer, intent(in) :: s
      integer(8), intent(in) :: nb
      type(cdf_buf_t), intent(inout) :: b
      integer(8) :: n
      if (.not. cdf_buf_reserve(b, 8192_8)) error stop 'io_server: head buffer'
      xsz = dfile(k)%v(sv(s))%xsz
      n = (nb + xsz - 1) / xsz
      if (sk(s) > 0) then
        call pack_head(isl, sk(s), sgx(s), sgy(s), r0%dbl, dfile(k)%v(sv(s))%xtype, n, b)
      else
        call pack_seg(s, b, 0_8)                 ! a small variable: all of it (at most a page)
      endif
    end subroutine head_bytes
  end subroutine write_direct

  subroutine flush_slot_partial(isl)
    ! Write whatever tiles of slot isl have arrived, slab by slab, and free it.
    ! One writer only (several stop before): the direct descriptor is
    ! closed first (no direct write in flight), the slabs go through serial
    ! netCDF, and the next record of the file reopens it for direct writes.
    integer, intent(in) :: isl
    integer :: t, k, ncid
    k = slot_key(1, isl)
    if (dfile(k)%fd >= 0) call direct_close(k)
    call nc_open_write(trim(opath(k)), ncid)
    write(*,'(3a)') 'io_server: (CDF5: the partial record goes through serial netCDF; ', trim(opath(k)), &
         ' is closed for direct writes meanwhile)'
    do t = 0, size(tbuf, 1) - 1
      if (.not. tfill(t, isl)) cycle
      if (tmeta(t, isl)%dbl) then
        call write_record8(ncid, tmeta(t, isl), tbuf(t, isl)%b8)
      else
        call write_record(ncid, tmeta(t, isl), tbuf(t, isl)%b)
      endif
    enddo
    call nc_close(ncid)
    tfill(:, isl) = .false.; slot_n(isl) = 0; slot_key(:, isl) = -1
  end subroutine flush_slot_partial

  subroutine write_record8(ncid, r, buf)
    ! Write one staged double record slab by slab (no-server fallback).
    integer, intent(in) :: ncid
    type(rec_t), intent(in) :: r
    real(8), intent(in) :: buf(*)
    integer :: iv, off
    call nc_put_1d_rec(ncid, 'ocean_time', r%time, r%rec)
    call put_time_step(ncid, r)
    call put_avg_times(ncid, r)
    do iv = 1, r%nvars
      off = r%off(iv)
      if (r%nrank(iv) == 1) then
        call nc_put_vec_rec_d(ncid, trim(r%names(iv)), buf(off:off+r%n1(iv)-1), r%rec)
      else if (r%nrank(iv) == 2) then
        call nc_put_2d_rec_d(ncid, trim(r%names(iv)), buf(off), r%n1(iv), r%n2(iv), r%rec, r%i0f(iv), r%j0f(iv))
      else
        call nc_put_3d_rec_d(ncid, trim(r%names(iv)), buf(off), r%n1(iv), r%n2(iv), r%n3(iv), r%rec, r%i0f(iv), r%j0f(iv))
      endif
    enddo
  end subroutine write_record8

  ! ---------------- double-precision staging (restart records) ----------------
  subroutine io_init8(cap_record, cap_field)
    ! Called by every double writer (restart, CDR_LITE output) before the
    ! first record; the capacities are the largest asked for.
    integer, intent(in) :: cap_record, cap_field
    if (allocated(hpin8_1)) then
      if (cap_record > hcap8 .or. cap_field > dcap8) error stop 'io_init8: capacity raised after the first double record'
      return
    endif
    hcap8 = max(hcap8, cap_record, 1)
    dcap8 = max(dcap8, cap_field, 1)
    write(*,'(a,f6.2,a)') 'io: double-precision staging ', 2.0 * real(hcap8) * 8.0 / 2.0**30, &
         ' GB host (pageable, allocated at the first restart write)'
  end subroutine io_init8

  subroutine ensure_stage8()
    if (.not. allocated(hpin8_1)) allocate(hpin8_1(hcap8), hpin8_2(hcap8))
    if (.not. allocated(dstage8)) allocate(dstage8(dcap8))
  end subroutine ensure_stage8

  subroutine io_stage8_begin(fkind, rec, nfile, time, open_path, tstep, tavg, tforce)
    integer, intent(in) :: fkind, rec, nfile
    integer, intent(in), optional :: tstep
    real(rp), intent(in) :: time
    character(len=*), intent(in) :: open_path
    real(rp), intent(in), optional :: tavg(2), tforce
    if (hcap8 == 0) error stop 'io_stage8_begin: io_init8 not called'
    if (hcap == 0) error stop 'io_stage8_begin: io_init not called'
    call ensure_stage8()
    call io_stage_begin(fkind, rec, nfile, time, open_path, dbl=.true., tstep=tstep, tavg=tavg, tforce=tforce)
    pend(cur)%dbl = .true.
  end subroutine io_stage8_begin

  subroutine copy_to_host8(off, n)
    integer, intent(in) :: off, n
#ifdef USE_CUDA
    integer :: ierr
    if (mod(cur, 2) == 1) then
      ierr = cudaMemcpyAsync(hpin8_1(off:off+n-1), dstage8(1:n), n, cudaMemcpyDeviceToHost, work_stream)
    else
      ierr = cudaMemcpyAsync(hpin8_2(off:off+n-1), dstage8(1:n), n, cudaMemcpyDeviceToHost, work_stream)
    endif
    if (ierr /= 0) error stop 'io_stage8: cudaMemcpyAsync failed'
#else
    if (mod(cur, 2) == 1) then
      hpin8_1(off:off+n-1) = dstage8(1:n)
    else
      hpin8_2(off:off+n-1) = dstage8(1:n)
    endif
#endif
  end subroutine copy_to_host8

  subroutine add_var8(name, n1, n2, n3, nrank, off, di, dj)
    ! As add_var against the double staging capacity.
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3, nrank
    integer, intent(out) :: off
    integer, intent(in), optional :: di, dj
    call add_var(name, n1, n2, n3, nrank, off, di, dj)
  end subroutine add_var8

  subroutine io_stage8_vec(name, vals)
    ! A (n, time) double variable of a restart record (scalars, the same on every tile): every
    ! tile stages it at file position 1, the assembly overlays them (nrank 1).
    character(len=*), intent(in) :: name
    real(8), intent(in) :: vals(:)
    integer :: n, off, iv
    n = size(vals)
    call add_var8(name, n, 1, 1, 1, off)
    iv = pend(cur)%nvars
    pend(cur)%i0f(iv) = 1; pend(cur)%j0f(iv) = 1
    if (mod(cur, 2) == 1) then
      hpin8_1(off:off+n-1) = vals
    else
      hpin8_2(off:off+n-1) = vals
    endif
  end subroutine io_stage8_vec

  subroutine io_stage8_2d(name, a, i0, i1, j0, j1, di, dj)
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer, intent(in) :: i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    integer :: i, j, ni, nj, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    call add_var8(name, ni, nj, 1, 2, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        dstage8((j - j0) * ni + (i - i0) + 1) = real(a(i, j), 8)
      enddo
    enddo
    call copy_to_host8(off, ni * nj)
  end subroutine io_stage8_2d

  subroutine io_stage8_line(name, a, lb, ub, l0, l1, fpos)
    ! A line along one edge, a(l0:l1) of a(lb:ub), into a (n, time) file
    ! variable at file position fpos (staged as a (n, 1) block at (fpos, 1);
    ! nc_put_2d_rec_d writes it as a line). Every tile stages every
    ! variable: a tile without a part passes l1 < l0 (an empty block).
    character(len=*), intent(in) :: name
    integer, intent(in) :: lb, ub, l0, l1, fpos
    real(rp), WORK_ATTR intent(in) :: a(lb:ub)
    integer :: l, n, off, f0
    n = max(l1 - l0 + 1, 0)
    f0 = merge(fpos, 1, n > 0)
    call add_var8(name, n, 1, 1, 2, off, f0 - int(decomp%i0), 1 - int(decomp%j0))
    if (n == 0) return
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*,0,work_stream>>>
#endif
    do l = l0, l1
      dstage8(l - l0 + 1) = real(a(l), 8)
    enddo
    call copy_to_host8(off, n)
  end subroutine io_stage8_line

  subroutine io_stage8_3d(name, a, k0, k1, i0, i1, j0, j1, di, dj)
    character(len=*), intent(in) :: name
    integer, intent(in) :: k0, k1, i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var8(name, ni, nj, nk, 3, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage8((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(i, j, k), 8)
        enddo
      enddo
    enddo
    call copy_to_host8(off, ni * nj * nk)
  end subroutine io_stage8_3d

  subroutine io_stage8_3d_pt(name, a, ntr, itr, k0, k1, i0, i1, j0, j1, di, dj)
    character(len=*), intent(in) :: name
    integer, intent(in) :: ntr, itr, k0, k1, i0, i1, j0, j1
    integer, intent(in), optional :: di, dj
    ! Compressed passive layout (gpu_state tpcol): unstored points read column 0 (zero).
    real(pp), WORK_ATTR intent(in) :: a(ntr, 0:ntpcol, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var8(name, ni, nj, nk, 3, off, di, dj)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage8((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(itr, tpcol(i, j), k), 8)
        enddo
      enddo
    enddo
    call copy_to_host8(off, ni * nj * nk)
  end subroutine io_stage8_3d_pt

  subroutine io_stage8_end()
    call io_stage_end()
  end subroutine io_stage8_end

  subroutine write_record(ncid, r, buf)
    ! Write one staged record from a float32 buffer (compute fallback and
    ! server share this).
    integer, intent(in) :: ncid
    type(rec_t), intent(in) :: r
    real(4), intent(in) :: buf(*)
    integer :: iv, off
    call nc_put_1d_rec(ncid, 'ocean_time', r%time, r%rec)
    call put_time_step(ncid, r)
    call put_avg_times(ncid, r)
    do iv = 1, r%nvars
      off = r%off(iv)
      if (r%nrank(iv) == 2) then
        call nc_put_2d_rec_f(ncid, trim(r%names(iv)), buf(off), r%n1(iv), r%n2(iv), r%rec, r%i0f(iv), r%j0f(iv))
      else
        call nc_put_3d_rec_f(ncid, trim(r%names(iv)), buf(off), r%n1(iv), r%n2(iv), r%n3(iv), r%rec, r%i0f(iv), r%j0f(iv))
      endif
    enddo
  end subroutine write_record

  subroutine write_set_local(idx)
    integer, intent(in) :: idx
    integer :: ncid, k
    k = pend(idx)%fkind
    if (len_trim(pend(idx)%open_path) > 0) fname_sync(k) = pend(idx)%open_path
    if (len_trim(fname_sync(k)) == 0) error stop 'io: record without a file'
    call nc_open_write(trim(fname_sync(k)), ncid)
    if (pend(idx)%dbl) then
      if (mod(idx, 2) == 1) then
        call write_record8(ncid, pend(idx), hpin8_1)
      else
        call write_record8(ncid, pend(idx), hpin8_2)
      endif
    elseif (mod(idx, 2) == 1) then
      call write_record(ncid, pend(idx), hpin1)
    else
      call write_record(ncid, pend(idx), hpin2)
    endif
    call nc_close(ncid)
    call report_written(pend(idx))
  end subroutine write_set_local

  subroutine report_written(r)
    type(rec_t), intent(in) :: r
    if (r%fkind == IO_KIND_HIS) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote history t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_CDR) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote cdr_lite t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_CDRGAS) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote cdr_gas t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_CDROUT) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote cdr t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_AVG) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote averages t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_BGCAVG) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote BGC averages t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_BGC) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote BGC history t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_BGCDIA) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote BGC diagnostics t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_BGCDIAAVG) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote BGC diagnostic averages t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    elseif (r%fkind == IO_KIND_RND) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote random t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    else
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote restart t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    endif
    flush(6)
  end subroutine report_written

  subroutine io_server_run()
    ! I/O rank main loop. Returns after CMD_QUIT from every compute rank.
    ! Complete records wait in ready_q (completion order, which is the
    ! staging order on every writer of a group: each compute rank sends its
    ! records in that order) and are written one by one; while one is
    ! written, io_progress (between variables, between pwrite pieces, at the
    ! group's closing barrier) takes in the commands that have arrived, so
    ! the compute ranks' staging buffers free up during the write instead of
    ! after it. Completed records are only queued there, never written.
#ifdef USE_MPI
    integer :: hdr(NHDR), ierr, stat(MPI_STATUS_SIZE), k
    real(8) :: c_start, c_end
    character(len=8) :: val
    srv_comm = comm_world()
    nserv = io_server_count()
    me_w = io_server_index()
    srv_ncompute = comm_ncompute()   ! every compute rank sends its own slabs and its own quit (not the readers)
    call writer_groups(nserv, wgroup_size, wgroups)
    my_group = me_w / wgroup_size
    me_g = mod(me_w, wgroup_size)
    call MPI_Comm_split(comm_writers(), my_group, me_w, gcomm, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Comm_split groups'
    call direct_options()
    call get_environment_variable('ROMS_GPU_IO_PROGRESS', val, status=k)
    srv_progress = .not. (k == 0 .and. len_trim(val) > 0 .and. val(1:1) == '0')
    c_start = wall_time()
    trecv = 0.0
    srv_nquit = 0
    nready = 0
    n_early = 0
    write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0)') 'io_server: ready on world rank ', comm_world_rank(), '  writer ', me_w, ' of ', nserv, &
         '  group ', my_group, ' of ', wgroups
    if (me_w == 0) then
      write(*,'(a,a,a,i0,a,i0,a,i0,a)') 'io_server: CDF5 output, ', merge('O_DIRECT', 'buffered', io_direct), &
           ' writes in pieces of ', io_piece / 1048576_8, ' MiB, ', wgroups, ' group(s) of ', wgroup_size, &
           ' writers'
      if (.not. srv_progress) write(*,'(a)') 'io_server: records received between writes only (ROMS_GPU_IO_PROGRESS=0)'
      if (nslot_use < NSLOT) write(*,'(a,i0,a)') 'io_server: ', nslot_use, ' record slots (ROMS_GPU_IO_NSLOT, a test setting)'
    endif
    flush(6)
    do
      if (nready > 0) then
        call write_ready()
        cycle
      endif
      if (srv_nquit >= srv_ncompute) exit
      call MPI_Recv(hdr, NHDR, MPI_INTEGER, MPI_ANY_SOURCE, TAG_CMD, srv_comm, stat, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Recv cmd'
      if (hdr(1) == CMD_PUT) then
        if (find_slot(hdr) == 0) call evict()
      endif
      call serve_cmd(hdr, stat(MPI_SOURCE), .true.)
    enddo
    ! records still assembling: write their arrived tiles before closing
    do k = 1, nslot_use
      if (slot_n(k) > 0) then
        write(*,'(a,i0,a,i0,a)') 'io_server: WARNING record ', slot_key(3, k), ' incomplete at quit (', &
             slot_n(k), ' tiles); writing what arrived'
        if (wgroup_size > 1) then
          write(*,'(a)') 'io_server: (several writers: the incomplete record is dropped to keep the collective order)'
        else
          call open_for(slot_key(1, k), slot_key(2, k))
          call flush_slot_partial(k)
        endif
      endif
    enddo
    do k = 1, NKIND
      if (dfile(k)%fd >= 0) call direct_close(k)
      nfile_open(k) = -1
    enddo
    c_end = wall_time()
    write(*,'(a,i0,a,f10.2,a,i0,a)') 'io_server: writer ', me_w, ' done  t = ', &
         (c_end - c_start), ' s since the server started; ', n_early, ' record parts taken in while writing'
    flush(6)
#endif
  contains

    subroutine write_ready()
      ! Write the oldest complete record of the ready queue and free its slot.
      integer :: isl, k
      real(8) :: c1, c2, c3
      type(rec_t) :: r
      isl = ready_q(1)
      ready_q(1:nready - 1) = ready_q(2:nready)
      nready = nready - 1
      r = tmeta(0, isl)
      k = r%fkind
      call open_for(k, r%nfile)
      c1 = wall_time()
      call write_direct(k, isl)
      c2 = wall_time()
      c3 = c2 - t_sync
      write(*,'(a,i0,a,i0,a,i0,a,f7.2,a,f7.2,a,f7.2,a,f7.2,a)') 'io_server: writer ', me_w, ' kind ', k, &
           ' rec ', r%rec, ': extend ', t_ext, ' s  assemble ', t_asm, ' s  put ', t_put, &
           ' s  sync ', (c2 - c3), ' s'
      write(*,'(a,i0,a,f8.3,a,f7.2,a,f7.2,a,i0,a,i0)') 'io_server: writer ', me_w, ' direct ', &
           real(d_bytes, 8) / 1.0d9, ' GB  pack ', t_pack, ' s  pwrite ', t_wr, ' s  rmw pages ', d_rmw, &
           '  heads ', d_heads
      write(*,'(a,i0,a,f7.2,a)') 'io_server: writer ', me_w, ' took in later records for ', t_prog, ' s'
      call report_written(r)
      write(*,'(a,i0,a,f7.2,a,f7.2,a)') 'io_server:   parts ', srv_ncompute, '  recv ', trecv(isl), &
           ' s  write ', (c2 - c1), ' s'
      tfill(:, isl) = .false.; slot_n(isl) = 0; slot_key(:, isl) = -1; trecv(isl) = 0.0
    end subroutine write_ready

    subroutine evict()
      ! No free slot: write the oldest incomplete record's arrived tiles
      ! slab by slab (old path) and free its slot. Records arrive out of
      ! source order only when output is faster than the writer, so say so:
      ! a partially assembled record costs partial-chunk writes.
      integer :: isl, k
      isl = 1
      do k = 2, nslot_use
        if (slot_key(2, k) < slot_key(2, isl) .or. &
            (slot_key(2, k) == slot_key(2, isl) .and. slot_key(3, k) < slot_key(3, isl))) isl = k
      enddo
      write(*,'(a,i0,a,i0,a,i0,a)') 'io_server: WARNING all ', nslot_use, ' record slots busy; writing record ', &
           slot_key(3, isl), ' with ', slot_n(isl), ' of its tiles present (output faster than the writer)'
      if (wgroup_size > 1) error stop 'io_server: record slots overrun with several writers (output faster than the writers): raise NSLOT or lower the output frequency'
      call open_for(slot_key(1, isl), slot_key(2, isl))
      call flush_slot_partial(isl)
    end subroutine evict

    subroutine open_for(kind, nfile)
      ! Make file nfile of this kind the open one before a record of it is
      ! written: close the previous file and open the announced path.
      ! Every writer of the group opens it for direct writes on its own (no
      ! netCDF handle, no collective call); a file closed for a partial
      ! record (flush_slot_partial) is reopened here.
      integer, intent(in) :: kind, nfile
      integer :: a, hit
      character(len=256) :: msg
      if (nfile_open(kind) == nfile .and. dfile(kind)%fd >= 0) return
      if (nfile_open(kind) /= nfile) then
        hit = 0
        do a = 1, NANN
          if (ann_key(1, a) == kind .and. ann_key(2, a) == nfile) hit = a
        enddo
        if (hit == 0) then
          write(*,'(a,i0,a,i0,a,i0,a,i0)') 'io_server: writer ', me_w, ' record of kind ', kind, &
               ' for file ', nfile, ' before its open; file open: ', nfile_open(kind)
          error stop 'io_server: record before file open'
        endif
        if (dfile(kind)%fd >= 0) call direct_close(kind)
        opath(kind) = ann_path(hit)
        nfile_open(kind) = nfile
        ann_key(:, hit) = -1
        ann_path(hit) = ''
      endif
      if (.not. cdf_open_w(trim(opath(kind)), dfile(kind), io_direct, msg)) then
        if (io_direct) then
          write(*,'(5a)') 'io_server: ', trim(opath(kind)), ': no O_DIRECT open (', trim(msg), &
               '); buffered descriptors from now on (ROMS_GPU_IO_DIRECT=0)'
          io_direct = .false.
          if (.not. cdf_open_w(trim(opath(kind)), dfile(kind), io_direct, msg)) then
            write(*,'(4a)') 'io_server: ', trim(opath(kind)), ': ', trim(msg)
            error stop 'io_server: CDF5 open for writing'
          endif
        else
          write(*,'(4a)') 'io_server: ', trim(opath(kind)), ': ', trim(msg)
          error stop 'io_server: CDF5 open for writing'
        endif
      endif
      write(*,'(a,i0,a,a)') 'io_server: writer ', me_w, ' opened ', trim(opath(kind))
      flush(6)
    end subroutine open_for
  end subroutine io_server_run

  subroutine serve_cmd(hdr, src, main)
    ! One command (header hdr from world rank src; the rest of its messages
    ! follow from src). A record part needs a slot (find_slot): the main loop
    ! evicts first when there is none, io_progress calls this only while a
    ! slot is free. A record completed here is queued (ready_q), not written.
    ! main: called by the main loop (else by io_progress, while writing).
    integer, intent(in) :: hdr(NHDR), src
    logical, intent(in) :: main
#ifdef USE_MPI
    integer :: ierr, stat(MPI_STATUS_SIZE), sc, iv, off, n, b, k, ntot, isl
    real(8) :: c0, c1
    character(len=PATHLEN) :: path
    character(len=NAMELEN * MAXV) :: namebuf
    type(rec_t) :: r
    sc = comm_compute_index(src) ! its compute rank: the tile buffers' index
    select case (hdr(1))
    case (CMD_OPEN)
      call MPI_Recv(path, PATHLEN, MPI_CHARACTER, src, TAG_STR, srv_comm, stat, ierr)
      ! Only remember the path, under its (kind, file number): the open
      ! happens when the first record of that file is written, so that
      ! several writers reach it in the same order as the records (an eager
      ! open could pair with another writer's record write and deadlock),
      ! and a record still due in the previous file is written there (the
      ! OPEN may arrive from a faster rank before a slower rank's part of
      ! that record).
      call announce(hdr(2), hdr(3), path)
    case (CMD_PUT)
      if (.not. allocated(tbuf)) then
        allocate(tbuf(0:srv_ncompute - 1, NSLOT), tmeta(0:srv_ncompute - 1, NSLOT), tfill(0:srv_ncompute - 1, NSLOT))
        tfill = .false.
      endif
      ! slot for this (fkind, nfile, rec); a free slot if it is a new record (the main loop
      ! makes one free first when there is none: evict)
      isl = find_slot(hdr)
      if (isl == 0) error stop 'io_server: no free record slot (serve_cmd)'
      slot_key(:, isl) = (/hdr(2), hdr(6), hdr(3)/)
      r = rec_t()
      r%fkind = hdr(2)
      r%rec = hdr(3)
      r%nvars = hdr(4)
      r%tstep = hdr(5)
      r%nfile = hdr(6)
      call MPI_Recv(namebuf, NAMELEN * MAXV, MPI_CHARACTER, src, TAG_STR, srv_comm, stat, ierr)
      call MPI_Recv(r%tmsg, 4, MPI_DOUBLE_PRECISION, src, TAG_TIME, srv_comm, stat, ierr)
      r%time = r%tmsg(1)
      r%has_tavg = (iand(hdr(8), 1) /= 0)
      r%has_tforce = (iand(hdr(8), 2) /= 0)
      r%tavg = r%tmsg(2:3)
      r%tforce = r%tmsg(4)
      ntot = 0
      do iv = 1, r%nvars
        b = 8 + 6 * (iv - 1)
        r%n1(iv) = hdr(b + 1); r%n2(iv) = hdr(b + 2); r%n3(iv) = hdr(b + 3)
        r%nrank(iv) = hdr(b + 4)
        r%i0f(iv) = hdr(b + 5); r%j0f(iv) = hdr(b + 6)
        r%names(iv) = namebuf((iv - 1) * NAMELEN + 1:iv * NAMELEN)
        r%off(iv) = 0                       ! not this writer's variable
        if (mine(iv)) then
          r%off(iv) = ntot + 1
          ntot = ntot + r%n1(iv) * r%n2(iv) * r%n3(iv)
        endif
      enddo
      ntot = max(ntot, 1)
      r%dbl = (hdr(7) == 1)
      c0 = wall_time()
      if (r%dbl) then
        if (allocated(tbuf(sc, isl)%b8)) then
          if (size(tbuf(sc, isl)%b8) < ntot) deallocate(tbuf(sc, isl)%b8)
        endif
        if (.not. allocated(tbuf(sc, isl)%b8)) allocate(tbuf(sc, isl)%b8(ntot))
        do iv = 1, r%nvars
          if (.not. mine(iv)) cycle
          off = r%off(iv)
          n = r%n1(iv) * r%n2(iv) * r%n3(iv)
          call MPI_Recv(tbuf(sc, isl)%b8(off), n, MPI_REAL8, src, TAG_DATA + iv, srv_comm, stat, ierr)
          if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Recv data (double)'
        enddo
      else
        if (allocated(tbuf(sc, isl)%b)) then
          if (size(tbuf(sc, isl)%b) < ntot) deallocate(tbuf(sc, isl)%b)
        endif
        if (.not. allocated(tbuf(sc, isl)%b)) allocate(tbuf(sc, isl)%b(ntot))
        do iv = 1, r%nvars
          if (.not. mine(iv)) cycle
          off = r%off(iv)
          n = r%n1(iv) * r%n2(iv) * r%n3(iv)
          call MPI_Recv(tbuf(sc, isl)%b(off), n, MPI_REAL4, src, TAG_DATA + iv, srv_comm, stat, ierr)
          if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Recv data'
        enddo
      endif
      c1 = wall_time()
      tmeta(sc, isl) = r
      if (.not. tfill(sc, isl)) slot_n(isl) = slot_n(isl) + 1
      tfill(sc, isl) = .true.
      trecv(isl) = trecv(isl) + real(c1 - c0)
      if (.not. main) n_early = n_early + 1
      if (slot_n(isl) == srv_ncompute) then
        write(*,'(a,i0,a,i0,a,i0,a,i0)') 'io_server: writer ', me_w, ' record complete: kind ', r%fkind, ' rec ', &
             r%rec, ' file ', r%nfile
        if (.not. main) write(*,'(a,i0,a)') 'io_server: writer ', me_w, ' (record taken in while writing)'
        flush(6)
        if (nready >= NSLOT) error stop 'io_server: ready queue overflow'
        nready = nready + 1
        ready_q(nready) = isl
      endif
    case (CMD_QUIT)
      srv_nquit = srv_nquit + 1
      write(*,'(a,i0,a,i0,a,i0)') 'io_server: writer ', me_w, ' quit ', srv_nquit, ' of ', srv_ncompute
      flush(6)
    case default
      error stop 'io_server: unknown command'
    end select
#endif
  end subroutine serve_cmd


  integer function find_slot(hdr) result(isl)
    ! The slot of the record that header hdr (a CMD_PUT) belongs to, else a
    ! free one, else 0.
    integer, intent(in) :: hdr(NHDR)
    integer :: k
    isl = 0
    do k = 1, nslot_use
      if (slot_key(1, k) == hdr(2) .and. slot_key(2, k) == hdr(6) .and. slot_key(3, k) == hdr(3)) isl = k
    enddo
    if (isl > 0) return
    do k = 1, nslot_use
      if (isl == 0 .and. slot_n(k) == 0) isl = k
    enddo
  end function find_slot

  subroutine io_progress()
    ! While a record is written: take in every command that has arrived, as
    ! long as a record slot is free (a part of a new record needs one; with
    ! none free the commands wait for the main loop, which may evict).
    ! Completed records are queued (serve_cmd), never written from here.
#ifdef USE_MPI
    integer :: hdr(NHDR), ierr, stat(MPI_STATUS_SIZE), k
    logical :: flag, free
    real(8) :: c0
    if (in_progress .or. .not. srv_progress) return
    in_progress = .true.
    c0 = wall_time()
    do
      free = .false.
      do k = 1, nslot_use
        if (slot_n(k) == 0) free = .true.
      enddo
      if (.not. free) exit
      call MPI_Iprobe(MPI_ANY_SOURCE, TAG_CMD, srv_comm, flag, stat, ierr)
      if (.not. flag) exit
      call MPI_Recv(hdr, NHDR, MPI_INTEGER, stat(MPI_SOURCE), TAG_CMD, srv_comm, stat, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Recv cmd (progress)'
      call serve_cmd(hdr, stat(MPI_SOURCE), .false.)
    enddo
    t_prog = t_prog + real(wall_time() - c0)
    in_progress = .false.
#endif
  end subroutine io_progress

  subroutine announce(kind, nfile, fpath)
    ! Remember the path of file nfile of this kind (every compute rank
    ! announces it; keep one entry).
    integer, intent(in) :: kind, nfile
    character(len=*), intent(in) :: fpath
    integer :: a, free
    if (nfile_open(kind) == nfile) return
    free = 0
    do a = 1, NANN
      if (ann_key(1, a) == kind .and. ann_key(2, a) == nfile) return
      if (free == 0 .and. ann_key(1, a) < 0) free = a
    enddo
    if (free == 0) error stop 'io_server: too many files announced ahead of their records (raise NANN)'
    ann_key(:, free) = (/kind, nfile/)
    ann_path(free) = fpath
  end subroutine announce


end module io_server

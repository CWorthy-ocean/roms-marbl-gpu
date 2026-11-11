module fs_ipc
  ! Implicit free surface, fs_gpu_opt 256: the one-cell ring exchanges of the solve between ranks on
  ! the same node by CUDA IPC, device-initiated: a put kernel writes this rank's edge strips into the
  ! neighbours' receive buffers (peer memory) and raises their flags; a wait kernel spins on this
  ! rank's flags and unpacks the halos. No host synchronisation, no MPI per exchange; the same values
  ! land in the same halo cells as with the MPI ring (bit-identical). Receive slots alternate
  ! (sequence number q kept on the device, slot mod(q, 2)): a rank can be at most one exchange ahead
  ! of a neighbour, so the slot it writes is never the one being read. Used only when every rank's
  ! neighbours are MPI ranks on its own node or physical edges (no periodic self-wrap); otherwise the
  ! MPI ring is kept on all ranks. If opening a neighbour's IPC handle fails on any rank, every rank
  ! closes what it opened and keeps the MPI ring (gather) path: the decision is collective (a min
  ! over the compute ranks after the opens) and logged once. ROMS_GPU_FS_IPC_FAIL=<what>:<rank>
  ! (what = ring, gather or both; rank = a compute rank or all) makes the last open on that rank
  ! report a failure after it succeeded, to test the fallback.
  ! (compiled on its own like comm.F90: no main-command modules; the caller passes the tile
  ! geometry, the neighbours and the stream)
  use kinds
#ifdef USE_CUDA
  use cudafor
#endif
  use comm_mod, only: comm_ipc_exchange, comm_ipc_allgather, comm_allreduce_min, comm_allreduce_sum, compute_barrier, &
       comm_rank
  use iso_c_binding
  implicit none
  private
  public :: ipc_init, ipc_on, ipc_ring_fine, ipc_ring_plane, ipc_gather_init, ipcg_on, ipc_gather, ipc_gather_v
  public :: ipc_set_opts

  logical :: ipc_on = .false.
  ! the multigrid's vector gather (fs_mgg) by IPC, when all compute ranks (at most 4) share the node:
  ! every rank puts its block into every rank's gathered-vector buffer (2 slots) and raises its flag
  ! (one per source rank); a wait kernel spins on the flags and copies the slot to the caller's array.
  ! gld: the buffer's length per slot (gtot, or more when the coarse coefficients are gathered through
  ! the same buffers, fs_gpu_opt 2048: ipc_gather_v; the gathers share the sequence counter)
  logical :: ipcg_on = .false.
  integer :: gnp = 0, gme = 0, gtot = 0, go = 0, gnpeer = 0, gpr(3) = 0, gld = 0
  integer :: mv = 0                  ! values per direction and slot
  logical :: has(4) = .false.        ! W, E, S, N neighbour (an MPI rank on this node)
  integer :: g_nx = 0, g_ny = 0, g_is = 0, g_ie = 0, g_js = 0, g_je = 0, g_j0 = 0, g_j1 = 0, g_i0 = 0, g_i1 = 0
#ifdef USE_CUDA
  integer(kind=cuda_stream_kind) :: ist
#endif
  integer, parameter :: IPC_NT = 512
  ! fs_gpu_opt 16384 (o_fuse): a ring's put and wait in one kernel (kring_f, kring_p: every thread puts,
  ! thread 1 raises the flags and spins on this rank's own, then every thread unpacks); 32768 (o_mblk):
  ! the gathers' put by several blocks (kgput_m, kgput_vm; the last block to finish raises the flags)
  logical :: o_fuse = .false., o_mblk = .false.
  integer, parameter :: GP_NT = 256, GP_MAXB = 64
#ifdef USE_CUDA
  ! (mv, 2 slots, 4 arrival sides W E S N) and the flags per arrival side: own cudaMalloc blocks (an
  ! IPC handle maps a whole allocation); qc: the sequence counter
  real(8), device, pointer :: rbuf(:,:,:) => null()
  integer, device, pointer :: rflag(:) => null()
  integer, device, allocatable :: qc(:)
  type(c_devptr) :: rbuf_p, rflag_p
  real(8), device, pointer :: gbuf(:,:) => null(), gq1(:,:) => null(), gq2(:,:) => null(), gq3(:,:) => null()
  integer, device, pointer :: gfl(:) => null(), gf1(:) => null(), gf2(:) => null(), gf3(:) => null()
  integer, device, allocatable :: gqc(:)
  type(c_devptr) :: gbuf_p, gfl_p
  integer, device, allocatable :: gbc(:)      ! o_mblk: blocks finished (reset by the last block)
  real(8), device, pointer :: pw(:,:,:) => null(), pe(:,:,:) => null(), ps(:,:,:) => null(), pn(:,:,:) => null()
  integer, device, pointer :: fw(:) => null(), fe(:) => null(), fs(:) => null(), fn(:) => null()
  ! the CUDA runtime's IPC calls (not in the cudafor module); cudaIpcMemHandle_t is 64 bytes,
  ! passed by value to cudaIpcOpenMemHandle
  type, bind(c) :: ipch
    character(kind=c_char) :: r(64)
  end type ipch
  interface
    integer(c_int) function c_ipc_get(h, dptr) bind(c, name='cudaIpcGetMemHandle')
      import :: ipch, c_int, c_intptr_t
      type(ipch) :: h
      integer(c_intptr_t), value :: dptr
    end function c_ipc_get
    integer(c_int) function c_ipc_open(dptr, h, flags) bind(c, name='cudaIpcOpenMemHandle')
      import :: ipch, c_int, c_intptr_t
      integer(c_intptr_t) :: dptr
      type(ipch), value :: h
      integer(c_int), value :: flags
    end function c_ipc_open
    integer(c_int) function c_ipc_close(dptr) bind(c, name='cudaIpcCloseMemHandle')
      import :: c_int, c_intptr_t
      integer(c_intptr_t), value :: dptr
    end function c_ipc_close
  end interface
#endif

contains

  subroutine ipc_set_opts(fuse, mblk)
    ! the IPC kernel options (fs_gpu_opt 16384, 32768), before ipc_init
    logical, intent(in) :: fuse, mblk
    o_fuse = fuse; o_mblk = mblk
  end subroutine ipc_set_opts

  subroutine ipc_init(m, nbr, nx, ny, is, ie, js, je, edges, stream)
    ! collective over the compute ranks: decide, allocate, exchange and open the IPC handles.
    ! nbr: W, E, S, N neighbours (compute ranks, NBR_NONE, NBR_SELF); edges: physical W, E, S, N
    integer, intent(in) :: m, nbr(4), nx, ny, is, ie, js, je
    logical, intent(in) :: edges(4)
#ifdef USE_CUDA
    integer(kind=cuda_stream_kind), intent(in) :: stream
#else
    integer, intent(in) :: stream
#endif
#ifdef USE_CUDA
    integer :: nb(4), d, ierr, nop, ferr
    real(8) :: ok(1), gok(1)
    logical :: same(4)
    integer(c_intptr_t) :: opened(8)
    character(len=64) :: chb, chf, crhb(4), crhf(4)
    type(ipch) :: hb, hf, rhb(4), rhf(4)
    type(c_devptr) :: dp
    integer(c_intptr_t) :: addr
    nb = nbr
    ! one buffer layout on every rank: a neighbour writes into this rank's receive buffer with ITS
    ! leading dimension, so mv is the largest ring size over the ranks (tiles of unequal size give
    ! unequal m; with the local m the slots and sides were misplaced: exc basin 20/19 x 10/9 diverged)
    ok(1) = -real(m, 8)
    call comm_allreduce_min(1, ok, gok)
    mv = nint(-gok(1)); ist = stream
    g_nx = nx; g_ny = ny; g_is = is; g_ie = ie; g_js = js; g_je = je
    g_j0 = merge(0, 1, edges(3)); g_j1 = merge(ny + 1, ny, edges(4))
    g_i0 = merge(0, 1, edges(1)); g_i1 = merge(nx + 1, nx, edges(2))
    ierr = cudaMalloc(rbuf_p, int(mv, 8) * 2_8 * 4_8 * 8_8)
    if (ierr /= 0) error stop 'implicit_fs: ipc_init cudaMalloc'
    ierr = cudaMalloc(rflag_p, 4_8 * 4_8)
    if (ierr /= 0) error stop 'implicit_fs: ipc_init cudaMalloc'
    call c_f_pointer(rbuf_p, rbuf, [mv, 2, 4]); call c_f_pointer(rflag_p, rflag, [4])
    allocate(qc(1)); rbuf = 0; rflag = 0; qc = 0
    ok = 1
    if (any(nb == NBR_SELF)) ok = 0
    ierr = c_ipc_get(hb, transfer(rbuf_p, addr)); if (ierr /= 0) ok = 0
    ierr = c_ipc_get(hf, transfer(rflag_p, addr)); if (ierr /= 0) ok = 0
    chb = transfer(hb, chb); chf = transfer(hf, chf)
    call comm_ipc_exchange(nb, chb, chf, crhb, crhf, same)
    has = .false.
    do d = 1, 4
      if (nb(d) < 0) cycle
      if (.not. same(d)) ok = 0
      has(d) = .true.
      rhb(d) = transfer(crhb(d), rhb(d)); rhf(d) = transfer(crhf(d), rhf(d))
    enddo
    call comm_allreduce_min(1, ok, gok)
    if (gok(1) < 0.5d0) then
      ierr = cudaFree(rbuf_p); ierr = cudaFree(rflag_p); deallocate(qc)
      rbuf => null(); rflag => null()
      has = .false.
      return
    endif
    ! open the neighbours' handles; on the first failure stop opening, then decide collectively
    nop = 0; ferr = 0
    do d = 1, 4
      if (.not. has(d)) cycle
      ierr = c_ipc_open(addr, rhb(d), 1)
      if (ierr /= 0) then; ferr = ierr; exit; endif
      nop = nop + 1; opened(nop) = addr
      dp = transfer(addr, dp)
      select case (d)
      case (1); call c_f_pointer(dp, pw, [mv, 2, 4])
      case (2); call c_f_pointer(dp, pe, [mv, 2, 4])
      case (3); call c_f_pointer(dp, ps, [mv, 2, 4])
      case (4); call c_f_pointer(dp, pn, [mv, 2, 4])
      end select
      ierr = c_ipc_open(addr, rhf(d), 1)
      if (ierr /= 0) then; ferr = ierr; exit; endif
      nop = nop + 1; opened(nop) = addr
      dp = transfer(addr, dp)
      select case (d)
      case (1); call c_f_pointer(dp, fw, [4])
      case (2); call c_f_pointer(dp, fe, [4])
      case (3); call c_f_pointer(dp, fs, [4])
      case (4); call c_f_pointer(dp, fn, [4])
      end select
    enddo
    if (ferr == 0 .and. nop > 0 .and. sim_fail('ring')) ferr = -1
    if (.not. ipc_agree(ferr, 'rings')) then
      do d = 1, nop
        ierr = c_ipc_close(opened(d))
      enddo
      ierr = cudaFree(rbuf_p); ierr = cudaFree(rflag_p); deallocate(qc)
      rbuf => null(); rflag => null()
      pw => null(); pe => null(); ps => null(); pn => null()
      fw => null(); fe => null(); fs => null(); fn => null()
      has = .false.
      ierr = cudaDeviceSynchronize()
      call compute_barrier()
      return
    endif
    ! the missing directions point at this rank's own buffers (never written: has(d) false)
    if (.not. has(1)) then; pw => rbuf; fw => rflag; endif
    if (.not. has(2)) then; pe => rbuf; fe => rflag; endif
    if (.not. has(3)) then; ps => rbuf; fs => rflag; endif
    if (.not. has(4)) then; pn => rbuf; fn => rflag; endif
    ierr = cudaDeviceSynchronize()
    call compute_barrier()
    ipc_on = any(has)
#endif
  end subroutine ipc_init

  subroutine ipc_gather_init(np, me, tot, o, ld)
    ! collective: the gathered vector's buffers (tot values; this rank's block at offset o; ld >= tot
    ! values per slot, the same on every rank)
    integer, intent(in) :: np, me, tot, o, ld
#ifdef USE_CUDA
    integer :: ierr, k, n, nop, ferr
    logical :: same
    real(8) :: ok(1), gok(1)
    integer(c_intptr_t) :: opened(6)
    character(len=64) :: chb, chf
    character(len=64), allocatable :: crhb(:), crhf(:)
    type(ipch) :: hb, hf, h
    type(c_devptr) :: dp
    integer(c_intptr_t) :: addr
    ipcg_on = .false.
    if (.not. ipc_on .or. np < 2 .or. np > 4) return
    gnp = np; gme = me; gtot = tot; go = o; gld = max(ld, tot)
    ierr = cudaMalloc(gbuf_p, int(gld, 8) * 2_8 * 8_8)
    if (ierr /= 0) error stop 'implicit_fs: ipc_gather_init cudaMalloc'
    ierr = cudaMalloc(gfl_p, 4_8 * 4_8)
    if (ierr /= 0) error stop 'implicit_fs: ipc_gather_init cudaMalloc'
    call c_f_pointer(gbuf_p, gbuf, [gld, 2]); call c_f_pointer(gfl_p, gfl, [4])
    allocate(gqc(1)); gbuf = 0; gfl = 0; gqc = 0
    if (.not. allocated(gbc)) allocate(gbc(1))
    gbc = 0
    ok = 1
    ierr = c_ipc_get(hb, transfer(gbuf_p, addr)); if (ierr /= 0) ok = 0
    ierr = c_ipc_get(hf, transfer(gfl_p, addr)); if (ierr /= 0) ok = 0
    chb = transfer(hb, chb); chf = transfer(hf, chf)
    allocate(crhb(np), crhf(np))
    call comm_ipc_allgather(chb, chf, crhb, crhf, same)
    if (.not. same) ok = 0
    call comm_allreduce_min(1, ok, gok)
    if (gok(1) < 0.5d0) then
      ierr = cudaFree(gbuf_p); ierr = cudaFree(gfl_p); deallocate(gqc)
      gbuf => null(); gfl => null()
      return
    endif
    gnpeer = 0; nop = 0; ferr = 0
    do k = 0, np - 1
      if (k == me) cycle
      gnpeer = gnpeer + 1; gpr(gnpeer) = k
      h = transfer(crhb(k + 1), h)
      ierr = c_ipc_open(addr, h, 1)
      if (ierr /= 0) then; ferr = ierr; exit; endif
      nop = nop + 1; opened(nop) = addr
      dp = transfer(addr, dp)
      select case (gnpeer)
      case (1); call c_f_pointer(dp, gq1, [gld, 2])
      case (2); call c_f_pointer(dp, gq2, [gld, 2])
      case (3); call c_f_pointer(dp, gq3, [gld, 2])
      end select
      h = transfer(crhf(k + 1), h)
      ierr = c_ipc_open(addr, h, 1)
      if (ierr /= 0) then; ferr = ierr; exit; endif
      nop = nop + 1; opened(nop) = addr
      dp = transfer(addr, dp)
      select case (gnpeer)
      case (1); call c_f_pointer(dp, gf1, [4])
      case (2); call c_f_pointer(dp, gf2, [4])
      case (3); call c_f_pointer(dp, gf3, [4])
      end select
    enddo
    if (ferr == 0 .and. nop > 0 .and. sim_fail('gather')) ferr = -1
    if (.not. ipc_agree(ferr, 'multigrid gather')) then
      do k = 1, nop
        ierr = c_ipc_close(opened(k))
      enddo
      ierr = cudaFree(gbuf_p); ierr = cudaFree(gfl_p); deallocate(gqc)
      gbuf => null(); gfl => null()
      gq1 => null(); gq2 => null(); gq3 => null(); gf1 => null(); gf2 => null(); gf3 => null()
      gnpeer = 0
      ierr = cudaDeviceSynchronize()
      call compute_barrier()
      return
    endif
    if (gnpeer < 2) then; gq2 => gbuf; gf2 => gfl; endif
    if (gnpeer < 3) then; gq3 => gbuf; gf3 => gfl; endif
    ierr = cudaDeviceSynchronize()
    call compute_barrier()
    ipcg_on = .true.
#endif
  end subroutine ipc_gather_init

  logical function ipc_agree(ferr, what)
    ! collective: true if no compute rank failed to open an IPC handle (ferr /= 0); otherwise
    ! rank 0 logs the fallback once (number of failing ranks, the lowest failing rank and its error)
    integer, intent(in) :: ferr
    character(len=*), intent(in) :: what
    real(8) :: x(1), g(1), r(1), gr(1), e(1), ge(1)
    x(1) = merge(1.0d0, 0.0d0, ferr /= 0)
    call comm_allreduce_sum(1, x, g)
    ipc_agree = g(1) < 0.5d0
    if (ipc_agree) return
    r(1) = huge(1.0d0)
    if (ferr /= 0) r(1) = real(comm_rank(), 8)
    call comm_allreduce_min(1, r, gr)
    e(1) = huge(1.0d0)
    if (ferr /= 0 .and. real(comm_rank(), 8) == gr(1)) e(1) = real(ferr, 8)
    call comm_allreduce_min(1, e, ge)
    if (comm_rank() == 0) write(*,'(1x,A,A,A,I4,A,I6,A,I6,A)') 'IMPLICIT_FS: fs_gpu_opt 256: CUDA IPC ', what, &
         ': cudaIpcOpenMemHandle failed on', nint(g(1)), ' rank(s) (first: rank', nint(gr(1)), ', error', &
         nint(ge(1)), '; -1 = simulated by ROMS_GPU_FS_IPC_FAIL): all ranks use MPI'
  end function ipc_agree

  logical function sim_fail(what)
    ! ROMS_GPU_FS_IPC_FAIL=<ring|gather|both>:<compute rank|all>: simulate a failed open here
    character(len=*), intent(in) :: what
    character(len=64) :: v, w, r
    integer :: st, ln, c, rk, ios
    sim_fail = .false.
    call get_environment_variable('ROMS_GPU_FS_IPC_FAIL', v, ln, st)
    if (st /= 0 .or. ln == 0) return
    c = index(v, ':')
    if (c == 0) return
    w = v(1:c - 1); r = v(c + 1:)
    if (trim(w) /= what .and. trim(w) /= 'both') return
    if (trim(r) == 'all') then
      sim_fail = .true.
    else
      read(r, *, iostat=ios) rk
      sim_fail = (ios == 0 .and. rk == comm_rank())
    endif
  end function sim_fail

  subroutine ipc_gather(x, mgx, mgy, l, nxd, nyd, out)
    ! fs_mgg's vector gather: this rank's block x(1:nxd, 1:nyd, l) (i fastest) into every rank's
    ! gathered vector at offset go; out(1:gtot) = the gathered vector
    integer, intent(in) :: mgx, mgy, l, nxd, nyd
#ifdef USE_CUDA
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    real(8), device :: out(*)
    integer :: ierr, nb
    if (o_mblk) then
      nb = max(1, min(GP_MAXB, (nxd * nyd + GP_NT - 1) / GP_NT))
      call kgput_m<<<nb, GP_NT, 0, ist>>>(x, mgx, mgy, l, nxd, nyd, gld, go, gme, gnpeer, &
           gbuf, gq1, gq2, gq3, gf1, gf2, gf3, gqc, gbc)
    else
    call kgput<<<1, IPC_NT, 0, ist>>>(x, mgx, mgy, l, nxd, nyd, gld, go, gme, gnpeer, gpr(1), gpr(2), gpr(3), &
         gbuf, gq1, gq2, gq3, gf1, gf2, gf3, gqc)
    endif
    call kgwait<<<1, 32, 0, ist>>>(gnpeer, gpr(1), gpr(2), gpr(3), gfl, gqc)
    call kgcopy<<<(gtot + 255) / 256, 256, 0, ist>>>(gtot, gld, gbuf, gqc, out)
    ierr = cudaGetLastError()
    if (ierr /= 0) error stop 'implicit_fs: ipc_gather launch'
#else
    real(8) :: x(0:mgx + 1, 0:mgy + 1, *), out(*)
#endif
  end subroutine ipc_gather

  subroutine ipc_gather_v(xv, n, o, cnt, out)
    ! fs_gpu_opt 2048, fs_mgg's coarse-coefficient gather: this rank's packed values xv(1:n) into every
    ! rank's gathered buffer at offset o (the same buffers and sequence as ipc_gather; cnt <= gld
    ! values in all); out(1:cnt) = the gathered values
    integer, intent(in) :: n, o, cnt
#ifdef USE_CUDA
    real(8), device :: xv(*), out(*)
    integer :: ierr, nb
    if (cnt > gld) error stop 'implicit_fs: ipc_gather_v: buffer too short'
    if (o_mblk) then
      nb = max(1, min(GP_MAXB, (n + GP_NT - 1) / GP_NT))
      call kgput_vm<<<nb, GP_NT, 0, ist>>>(xv, n, gld, o, gme, gnpeer, gbuf, gq1, gq2, gq3, gf1, gf2, gf3, gqc, gbc)
    else
    call kgput_v<<<1, IPC_NT, 0, ist>>>(xv, n, gld, o, gme, gnpeer, gbuf, gq1, gq2, gq3, gf1, gf2, gf3, gqc)
    endif
    call kgwait<<<1, 32, 0, ist>>>(gnpeer, gpr(1), gpr(2), gpr(3), gfl, gqc)
    call kgcopy<<<(cnt + 255) / 256, 256, 0, ist>>>(cnt, gld, gbuf, gqc, out)
    ierr = cudaGetLastError()
    if (ierr /= 0) error stop 'implicit_fs: ipc_gather_v launch'
#else
    real(8) :: xv(*), out(*)
#endif
  end subroutine ipc_gather_v

  subroutine ipc_ring_fine(nf, a1, a2, a3, a4, a5, a6)
    ! fs_data's ring of nf fine-level fields (the same strips and halo cells)
    integer, intent(in) :: nf
#ifdef USE_CUDA
    real(8), device, dimension(g_is:g_ie, g_js:g_je) :: a1, a2, a3, a4, a5, a6
    integer :: nx, ny, j0, j1, i0, i1, is, ie, js, je, h(4), ierr
    nx = g_nx; ny = g_ny; is = g_is; ie = g_ie; js = g_js; je = g_je
    j0 = g_j0; j1 = g_j1; i0 = g_i0; i1 = g_i1
    h = merge(1, 0, has)
    if (o_fuse) then
      call kring_f<<<1, IPC_NT, 0, ist>>>(nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, a1, a2, a3, a4, a5, a6, &
           pw, pe, ps, pn, fw, fe, fs, fn, rbuf, rflag, qc, h(1), h(2), h(3), h(4))
    else
    call kput_f<<<1, IPC_NT, 0, ist>>>(nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, a1, a2, a3, a4, a5, a6, &
         pw, pe, ps, pn, fw, fe, fs, fn, qc, h(1), h(2), h(3), h(4))
    call kwait_f<<<1, IPC_NT, 0, ist>>>(nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, a1, a2, a3, a4, a5, a6, &
         rbuf, rflag, qc, h(1), h(2), h(3), h(4))
    endif
    ierr = cudaGetLastError()
    if (ierr /= 0) error stop 'implicit_fs: ipc_ring_fine launch'
#else
    real(8), dimension(g_is:g_ie, g_js:g_je) :: a1, a2, a3, a4, a5, a6
#endif
  end subroutine ipc_ring_fine

  subroutine ipc_ring_plane(x, mgx, mgy, l, nxl, nyl)
    ! fs_mg's ring_mg of level l >= 1 (zero halo toward a physical edge)
    integer, intent(in) :: mgx, mgy, l, nxl, nyl
#ifdef USE_CUDA
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    integer :: h(4), ierr
    h = merge(1, 0, has)
    if (o_fuse) then
      call kring_p<<<1, IPC_NT, 0, ist>>>(x, mgx, mgy, l, nxl, nyl, mv, pw, pe, ps, pn, fw, fe, fs, fn, rbuf, rflag, &
           qc, h(1), h(2), h(3), h(4))
    else
    call kput_p<<<1, IPC_NT, 0, ist>>>(x, mgx, mgy, l, nxl, nyl, mv, pw, pe, ps, pn, fw, fe, fs, fn, qc, &
         h(1), h(2), h(3), h(4))
    call kwait_p<<<1, IPC_NT, 0, ist>>>(x, mgx, mgy, l, nxl, nyl, mv, rbuf, rflag, qc, h(1), h(2), h(3), h(4))
    endif
    ierr = cudaGetLastError()
    if (ierr /= 0) error stop 'implicit_fs: ipc_ring_plane launch'
#else
    real(8) :: x(0:mgx + 1, 0:mgy + 1, *)
#endif
  end subroutine ipc_ring_plane

#ifdef USE_CUDA
  attributes(global) subroutine kgput(x, mgx, mgy, l, nxd, nyd, tot, o, me, npeer, p1, p2, p3, gb, q1, q2, q3, &
       f1, f2, f3, qc)
    integer, value :: mgx, mgy, l, nxd, nyd, tot, o, me, npeer, p1, p2, p3
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    real(8), device :: gb(tot, 2), q1(tot, 2), q2(tot, 2), q3(tot, 2)
    integer, device, volatile :: f1(4), f2(4), f3(4)
    integer, device :: qc(1)
    integer :: q, s, k, i, j
    real(8) :: v
    q = qc(1) + 1; s = mod(q, 2) + 1
    do k = threadIdx%x, nxd * nyd, blockDim%x
      j = (k - 1) / nxd + 1; i = k - (j - 1) * nxd
      v = x(i, j, l)
      gb(o + k, s) = v
      if (npeer >= 1) q1(o + k, s) = v
      if (npeer >= 2) q2(o + k, s) = v
      if (npeer >= 3) q3(o + k, s) = v
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      if (npeer >= 1) f1(me + 1) = q
      if (npeer >= 2) f2(me + 1) = q
      if (npeer >= 3) f3(me + 1) = q
    endif
  end subroutine kgput

  attributes(global) subroutine kgput_v(xv, n, ld, o, me, npeer, gb, q1, q2, q3, f1, f2, f3, qc)
    ! kgput for a packed vector xv(1:n)
    integer, value :: n, ld, o, me, npeer
    real(8), device :: xv(n)
    real(8), device :: gb(ld, 2), q1(ld, 2), q2(ld, 2), q3(ld, 2)
    integer, device, volatile :: f1(4), f2(4), f3(4)
    integer, device :: qc(1)
    integer :: q, s, k
    real(8) :: v
    q = qc(1) + 1; s = mod(q, 2) + 1
    do k = threadIdx%x, n, blockDim%x
      v = xv(k)
      gb(o + k, s) = v
      if (npeer >= 1) q1(o + k, s) = v
      if (npeer >= 2) q2(o + k, s) = v
      if (npeer >= 3) q3(o + k, s) = v
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      if (npeer >= 1) f1(me + 1) = q
      if (npeer >= 2) f2(me + 1) = q
      if (npeer >= 3) f3(me + 1) = q
    endif
  end subroutine kgput_v

  attributes(global) subroutine kgput_m(x, mgx, mgy, l, nxd, nyd, tot, o, me, npeer, gb, q1, q2, q3, &
       f1, f2, f3, qc, bc)
    ! kgput by several blocks: each block puts its share and fences; the last block to finish (counter
    ! bc, reset by it) raises the flags (fsr_k12's scheme)
    integer, value :: mgx, mgy, l, nxd, nyd, tot, o, me, npeer
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    real(8), device :: gb(tot, 2), q1(tot, 2), q2(tot, 2), q3(tot, 2)
    integer, device, volatile :: f1(4), f2(4), f3(4)
    integer, device :: qc(1), bc(1)
    integer, shared :: last
    integer :: q, s, k, i, j, old
    real(8) :: v
    q = qc(1) + 1; s = mod(q, 2) + 1
    do k = (blockIdx%x - 1) * blockDim%x + threadIdx%x, nxd * nyd, gridDim%x * blockDim%x
      j = (k - 1) / nxd + 1; i = k - (j - 1) * nxd
      v = x(i, j, l)
      gb(o + k, s) = v
      if (npeer >= 1) q1(o + k, s) = v
      if (npeer >= 2) q2(o + k, s) = v
      if (npeer >= 3) q3(o + k, s) = v
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      old = atomicadd(bc(1), 1)
      last = 0
      if (old == gridDim%x - 1) last = 1
    endif
    call syncthreads()
    if (last == 0) return
    if (threadIdx%x == 1) then
      call threadfence_system()
      bc(1) = 0
      if (npeer >= 1) f1(me + 1) = q
      if (npeer >= 2) f2(me + 1) = q
      if (npeer >= 3) f3(me + 1) = q
    endif
  end subroutine kgput_m

  attributes(global) subroutine kgput_vm(xv, n, ld, o, me, npeer, gb, q1, q2, q3, f1, f2, f3, qc, bc)
    ! kgput_v by several blocks (kgput_m's scheme)
    integer, value :: n, ld, o, me, npeer
    real(8), device :: xv(n)
    real(8), device :: gb(ld, 2), q1(ld, 2), q2(ld, 2), q3(ld, 2)
    integer, device, volatile :: f1(4), f2(4), f3(4)
    integer, device :: qc(1), bc(1)
    integer, shared :: last
    integer :: q, s, k, old
    real(8) :: v
    q = qc(1) + 1; s = mod(q, 2) + 1
    do k = (blockIdx%x - 1) * blockDim%x + threadIdx%x, n, gridDim%x * blockDim%x
      v = xv(k)
      gb(o + k, s) = v
      if (npeer >= 1) q1(o + k, s) = v
      if (npeer >= 2) q2(o + k, s) = v
      if (npeer >= 3) q3(o + k, s) = v
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      old = atomicadd(bc(1), 1)
      last = 0
      if (old == gridDim%x - 1) last = 1
    endif
    call syncthreads()
    if (last == 0) return
    if (threadIdx%x == 1) then
      call threadfence_system()
      bc(1) = 0
      if (npeer >= 1) f1(me + 1) = q
      if (npeer >= 2) f2(me + 1) = q
      if (npeer >= 3) f3(me + 1) = q
    endif
  end subroutine kgput_vm

  attributes(global) subroutine kring_f(nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, a1, a2, a3, a4, a5, a6, &
       pw, pe, ps, pn, fw, fe, fs, fn, rb, fl, qc, hw, he, hs, hn)
    ! kput_f and kwait_f in one kernel (one block): the same puts, flags, wait and unpacking in the
    ! same order (every rank puts before it waits)
    integer, value :: nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, hw, he, hs, hn
    real(8), device, dimension(is:ie, js:je) :: a1, a2, a3, a4, a5, a6
    real(8), device :: pw(mv, 2, 4), pe(mv, 2, 4), ps(mv, 2, 4), pn(mv, 2, 4)
    integer, device, volatile :: fw(4), fe(4), fs(4), fn(4)
    real(8), device, volatile :: rb(mv, 2, 4)
    integer, device, volatile :: fl(4)
    integer, device :: qc(1)
    integer :: q, s, k, f, m, n2, jj, ii
    real(8) :: vw, ve, vs, vn
    q = qc(1) + 1; s = mod(q, 2) + 1
    n2 = j1 - j0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; jj = j0 + m - 1
      select case (f)
      case (1); vw = a1(1, jj); ve = a1(nx, jj)
      case (2); vw = a2(1, jj); ve = a2(nx, jj)
      case (3); vw = a3(1, jj); ve = a3(nx, jj)
      case (4); vw = a4(1, jj); ve = a4(nx, jj)
      case (5); vw = a5(1, jj); ve = a5(nx, jj)
      case default; vw = a6(1, jj); ve = a6(nx, jj)
      end select
      if (hw /= 0) pw(k, s, 2) = vw
      if (he /= 0) pe(k, s, 1) = ve
    enddo
    n2 = i1 - i0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; ii = i0 + m - 1
      select case (f)
      case (1); vs = a1(ii, 1); vn = a1(ii, ny)
      case (2); vs = a2(ii, 1); vn = a2(ii, ny)
      case (3); vs = a3(ii, 1); vn = a3(ii, ny)
      case (4); vs = a4(ii, 1); vn = a4(ii, ny)
      case (5); vs = a5(ii, 1); vn = a5(ii, ny)
      case default; vs = a6(ii, 1); vn = a6(ii, ny)
      end select
      if (hs /= 0) ps(k, s, 4) = vs
      if (hn /= 0) pn(k, s, 3) = vn
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      if (hw /= 0) fw(2) = q
      if (he /= 0) fe(1) = q
      if (hs /= 0) fs(4) = q
      if (hn /= 0) fn(3) = q
      do while ((hw /= 0 .and. fl(1) < q) .or. (he /= 0 .and. fl(2) < q) .or. (hs /= 0 .and. fl(3) < q) &
           .or. (hn /= 0 .and. fl(4) < q))
      enddo
    endif
    call syncthreads()
    call threadfence()
    n2 = j1 - j0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; jj = j0 + m - 1
      if (hw /= 0) then
        vw = rb(k, s, 1)
        select case (f)
        case (1); a1(0, jj) = vw
        case (2); a2(0, jj) = vw
        case (3); a3(0, jj) = vw
        case (4); a4(0, jj) = vw
        case (5); a5(0, jj) = vw
        case default; a6(0, jj) = vw
        end select
      endif
      if (he /= 0) then
        ve = rb(k, s, 2)
        select case (f)
        case (1); a1(nx + 1, jj) = ve
        case (2); a2(nx + 1, jj) = ve
        case (3); a3(nx + 1, jj) = ve
        case (4); a4(nx + 1, jj) = ve
        case (5); a5(nx + 1, jj) = ve
        case default; a6(nx + 1, jj) = ve
        end select
      endif
    enddo
    n2 = i1 - i0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; ii = i0 + m - 1
      if (hs /= 0) then
        vs = rb(k, s, 3)
        select case (f)
        case (1); a1(ii, 0) = vs
        case (2); a2(ii, 0) = vs
        case (3); a3(ii, 0) = vs
        case (4); a4(ii, 0) = vs
        case (5); a5(ii, 0) = vs
        case default; a6(ii, 0) = vs
        end select
      endif
      if (hn /= 0) then
        vn = rb(k, s, 4)
        select case (f)
        case (1); a1(ii, ny + 1) = vn
        case (2); a2(ii, ny + 1) = vn
        case (3); a3(ii, ny + 1) = vn
        case (4); a4(ii, ny + 1) = vn
        case (5); a5(ii, ny + 1) = vn
        case default; a6(ii, ny + 1) = vn
        end select
      endif
    enddo
    call syncthreads()
    if (threadIdx%x == 1) qc(1) = q
  end subroutine kring_f

  attributes(global) subroutine kring_p(x, mgx, mgy, l, nxl, nyl, mv, pw, pe, ps, pn, fw, fe, fs, fn, rb, fl, qc, &
       hw, he, hs, hn)
    ! kput_p and kwait_p in one kernel (kring_f's scheme)
    integer, value :: mgx, mgy, l, nxl, nyl, mv, hw, he, hs, hn
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    real(8), device :: pw(mv, 2, 4), pe(mv, 2, 4), ps(mv, 2, 4), pn(mv, 2, 4)
    integer, device, volatile :: fw(4), fe(4), fs(4), fn(4)
    real(8), device, volatile :: rb(mv, 2, 4)
    integer, device, volatile :: fl(4)
    integer, device :: qc(1)
    integer :: q, s, m
    q = qc(1) + 1; s = mod(q, 2) + 1
    do m = threadIdx%x, max(nxl, nyl), blockDim%x
      if (m <= nyl) then
        if (hw /= 0) pw(m, s, 2) = x(1, m, l)
        if (he /= 0) pe(m, s, 1) = x(nxl, m, l)
      endif
      if (m <= nxl) then
        if (hs /= 0) ps(m, s, 4) = x(m, 1, l)
        if (hn /= 0) pn(m, s, 3) = x(m, nyl, l)
      endif
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      if (hw /= 0) fw(2) = q
      if (he /= 0) fe(1) = q
      if (hs /= 0) fs(4) = q
      if (hn /= 0) fn(3) = q
      do while ((hw /= 0 .and. fl(1) < q) .or. (he /= 0 .and. fl(2) < q) .or. (hs /= 0 .and. fl(3) < q) &
           .or. (hn /= 0 .and. fl(4) < q))
      enddo
    endif
    call syncthreads()
    call threadfence()
    do m = threadIdx%x, max(nxl, nyl), blockDim%x
      if (m <= nyl) then
        x(0, m, l) = 0.0d0; x(nxl + 1, m, l) = 0.0d0
        if (hw /= 0) x(0, m, l) = rb(m, s, 1)
        if (he /= 0) x(nxl + 1, m, l) = rb(m, s, 2)
      endif
      if (m <= nxl) then
        x(m, 0, l) = 0.0d0; x(m, nyl + 1, l) = 0.0d0
        if (hs /= 0) x(m, 0, l) = rb(m, s, 3)
        if (hn /= 0) x(m, nyl + 1, l) = rb(m, s, 4)
      endif
    enddo
    call syncthreads()
    if (threadIdx%x == 1) qc(1) = q
  end subroutine kring_p

  attributes(global) subroutine kgwait(npeer, p1, p2, p3, fl, qc)
    integer, value :: npeer, p1, p2, p3
    integer, device, volatile :: fl(4)
    integer, device :: qc(1)
    integer :: q
    q = qc(1) + 1
    if (threadIdx%x == 1) then
      do while ((npeer >= 1 .and. fl(p1 + 1) < q) .or. (npeer >= 2 .and. fl(p2 + 1) < q) .or. &
           (npeer >= 3 .and. fl(p3 + 1) < q))
      enddo
      call threadfence()
      qc(1) = q
    endif
  end subroutine kgwait

  attributes(global) subroutine kgcopy(tot, ld, gb, qc, out)
    integer, value :: tot, ld
    real(8), device, volatile :: gb(ld, 2)
    integer, device :: qc(1)
    real(8), device :: out(tot)
    integer :: k, s
    s = mod(qc(1), 2) + 1
    k = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (k <= tot) out(k) = gb(k, s)
  end subroutine kgcopy

  attributes(global) subroutine kput_f(nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, a1, a2, a3, a4, a5, a6, &
       pw, pe, ps, pn, fw, fe, fs, fn, qc, hw, he, hs, hn)
    integer, value :: nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, hw, he, hs, hn
    real(8), device, dimension(is:ie, js:je) :: a1, a2, a3, a4, a5, a6
    real(8), device :: pw(mv, 2, 4), pe(mv, 2, 4), ps(mv, 2, 4), pn(mv, 2, 4)
    integer, device, volatile :: fw(4), fe(4), fs(4), fn(4)
    integer, device :: qc(1)
    integer :: t, q, s, k, f, m, n2, jj, ii
    real(8) :: vw, ve, vs, vn
    q = qc(1) + 1; s = mod(q, 2) + 1
    n2 = j1 - j0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; jj = j0 + m - 1
      select case (f)
      case (1); vw = a1(1, jj); ve = a1(nx, jj)
      case (2); vw = a2(1, jj); ve = a2(nx, jj)
      case (3); vw = a3(1, jj); ve = a3(nx, jj)
      case (4); vw = a4(1, jj); ve = a4(nx, jj)
      case (5); vw = a5(1, jj); ve = a5(nx, jj)
      case default; vw = a6(1, jj); ve = a6(nx, jj)
      end select
      if (hw /= 0) pw(k, s, 2) = vw
      if (he /= 0) pe(k, s, 1) = ve
    enddo
    n2 = i1 - i0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; ii = i0 + m - 1
      select case (f)
      case (1); vs = a1(ii, 1); vn = a1(ii, ny)
      case (2); vs = a2(ii, 1); vn = a2(ii, ny)
      case (3); vs = a3(ii, 1); vn = a3(ii, ny)
      case (4); vs = a4(ii, 1); vn = a4(ii, ny)
      case (5); vs = a5(ii, 1); vn = a5(ii, ny)
      case default; vs = a6(ii, 1); vn = a6(ii, ny)
      end select
      if (hs /= 0) ps(k, s, 4) = vs
      if (hn /= 0) pn(k, s, 3) = vn
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      if (hw /= 0) fw(2) = q
      if (he /= 0) fe(1) = q
      if (hs /= 0) fs(4) = q
      if (hn /= 0) fn(3) = q
    endif
  end subroutine kput_f

  attributes(global) subroutine kwait_f(nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, a1, a2, a3, a4, a5, a6, &
       rb, fl, qc, hw, he, hs, hn)
    integer, value :: nf, nx, ny, is, ie, js, je, j0, j1, i0, i1, mv, hw, he, hs, hn
    real(8), device, dimension(is:ie, js:je) :: a1, a2, a3, a4, a5, a6
    real(8), device, volatile :: rb(mv, 2, 4)
    integer, device, volatile :: fl(4)
    integer, device :: qc(1)
    integer :: q, s, k, f, m, n2, jj, ii
    real(8) :: vw, ve, vs, vn
    q = qc(1) + 1; s = mod(q, 2) + 1
    if (threadIdx%x == 1) then
      do while ((hw /= 0 .and. fl(1) < q) .or. (he /= 0 .and. fl(2) < q) .or. (hs /= 0 .and. fl(3) < q) &
           .or. (hn /= 0 .and. fl(4) < q))
      enddo
    endif
    call syncthreads()
    call threadfence()
    n2 = j1 - j0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; jj = j0 + m - 1
      if (hw /= 0) then
        vw = rb(k, s, 1)
        select case (f)
        case (1); a1(0, jj) = vw
        case (2); a2(0, jj) = vw
        case (3); a3(0, jj) = vw
        case (4); a4(0, jj) = vw
        case (5); a5(0, jj) = vw
        case default; a6(0, jj) = vw
        end select
      endif
      if (he /= 0) then
        ve = rb(k, s, 2)
        select case (f)
        case (1); a1(nx + 1, jj) = ve
        case (2); a2(nx + 1, jj) = ve
        case (3); a3(nx + 1, jj) = ve
        case (4); a4(nx + 1, jj) = ve
        case (5); a5(nx + 1, jj) = ve
        case default; a6(nx + 1, jj) = ve
        end select
      endif
    enddo
    n2 = i1 - i0 + 1
    do k = threadIdx%x, nf * n2, blockDim%x
      f = (k - 1) / n2 + 1; m = k - (f - 1) * n2; ii = i0 + m - 1
      if (hs /= 0) then
        vs = rb(k, s, 3)
        select case (f)
        case (1); a1(ii, 0) = vs
        case (2); a2(ii, 0) = vs
        case (3); a3(ii, 0) = vs
        case (4); a4(ii, 0) = vs
        case (5); a5(ii, 0) = vs
        case default; a6(ii, 0) = vs
        end select
      endif
      if (hn /= 0) then
        vn = rb(k, s, 4)
        select case (f)
        case (1); a1(ii, ny + 1) = vn
        case (2); a2(ii, ny + 1) = vn
        case (3); a3(ii, ny + 1) = vn
        case (4); a4(ii, ny + 1) = vn
        case (5); a5(ii, ny + 1) = vn
        case default; a6(ii, ny + 1) = vn
        end select
      endif
    enddo
    call syncthreads()
    if (threadIdx%x == 1) qc(1) = q
  end subroutine kwait_f

  attributes(global) subroutine kput_p(x, mgx, mgy, l, nxl, nyl, mv, pw, pe, ps, pn, fw, fe, fs, fn, qc, hw, he, hs, hn)
    integer, value :: mgx, mgy, l, nxl, nyl, mv, hw, he, hs, hn
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    real(8), device :: pw(mv, 2, 4), pe(mv, 2, 4), ps(mv, 2, 4), pn(mv, 2, 4)
    integer, device, volatile :: fw(4), fe(4), fs(4), fn(4)
    integer, device :: qc(1)
    integer :: q, s, m
    q = qc(1) + 1; s = mod(q, 2) + 1
    do m = threadIdx%x, max(nxl, nyl), blockDim%x
      if (m <= nyl) then
        if (hw /= 0) pw(m, s, 2) = x(1, m, l)
        if (he /= 0) pe(m, s, 1) = x(nxl, m, l)
      endif
      if (m <= nxl) then
        if (hs /= 0) ps(m, s, 4) = x(m, 1, l)
        if (hn /= 0) pn(m, s, 3) = x(m, nyl, l)
      endif
    enddo
    call threadfence_system()
    call syncthreads()
    if (threadIdx%x == 1) then
      if (hw /= 0) fw(2) = q
      if (he /= 0) fe(1) = q
      if (hs /= 0) fs(4) = q
      if (hn /= 0) fn(3) = q
    endif
  end subroutine kput_p

  attributes(global) subroutine kwait_p(x, mgx, mgy, l, nxl, nyl, mv, rb, fl, qc, hw, he, hs, hn)
    integer, value :: mgx, mgy, l, nxl, nyl, mv, hw, he, hs, hn
    real(8), device :: x(0:mgx + 1, 0:mgy + 1, *)
    real(8), device, volatile :: rb(mv, 2, 4)
    integer, device, volatile :: fl(4)
    integer, device :: qc(1)
    integer :: q, s, m
    q = qc(1) + 1; s = mod(q, 2) + 1
    if (threadIdx%x == 1) then
      do while ((hw /= 0 .and. fl(1) < q) .or. (he /= 0 .and. fl(2) < q) .or. (hs /= 0 .and. fl(3) < q) &
           .or. (hn /= 0 .and. fl(4) < q))
      enddo
    endif
    call syncthreads()
    call threadfence()
    do m = threadIdx%x, max(nxl, nyl), blockDim%x
      if (m <= nyl) then
        x(0, m, l) = 0.0d0; x(nxl + 1, m, l) = 0.0d0
        if (hw /= 0) x(0, m, l) = rb(m, s, 1)
        if (he /= 0) x(nxl + 1, m, l) = rb(m, s, 2)
      endif
      if (m <= nxl) then
        x(m, 0, l) = 0.0d0; x(m, nyl + 1, l) = 0.0d0
        if (hs /= 0) x(m, 0, l) = rb(m, s, 3)
        if (hn /= 0) x(m, nyl + 1, l) = rb(m, s, 4)
      endif
    enddo
    call syncthreads()
    if (threadIdx%x == 1) qc(1) = q
  end subroutine kwait_p
#endif

end module fs_ipc

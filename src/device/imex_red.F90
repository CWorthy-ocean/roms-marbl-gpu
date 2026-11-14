#include "mem_attr.inc"
module imex_red
  ! Deterministic sums for the IMEX_IGW solver (imex_gpu_redu = 1): the
  ! kernels write one partial sum per water column (sequential in k) into
  ! red_part(c, s), c = i + (j-1) nx; red_sum adds them with one block of
  ! RED_NT threads in a fixed order (strided sequential sums, then a shared-
  ! memory tree), copies the results asynchronously into pinned host memory
  ! on work_stream and waits for that stream only. A CUF-kernel scalar
  ! reduction is neither bitwise reproducible from run to run (its
  ! cross-block combination order varies) nor asynchronous (the result
  ! comes back through a synchronous copy).
  use kinds
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: red_ensure, red_sum, red_set_split
  public :: it_set, itw, itw2, it_step, it_reb, it_print
  integer, parameter :: RED_NT = 512, RED_MAXS = 4
  ! imex_gpu_opt 4 (red_set_split): above RED_SPLIT_MIN columns the RED_NT strided sums run in RED_NT/RED_PW
  ! blocks of RED_PW threads (red_kpart, one partial per thread into red_tmp), the tree in a second
  ! one-block kernel (red_ktree): the same additions in the same order as red_kernel, on several
  ! multiprocessors instead of one
  integer, parameter :: RED_PW = 32, RED_SPLIT_MIN = 65536
  logical :: red_split = .false.
#ifdef USE_CUDA
  real(rp), device, allocatable, public :: red_part(:,:)
  real(rp), device, allocatable :: red_dres(:)
  real(rp), device, allocatable :: red_tmp(:,:)
  real(rp), pinned, allocatable :: red_hres(:)
#else
  real(rp), allocatable, public :: red_part(:,:)
#endif
  integer :: red_n = 0
  ! imex_gpu_timers (GPU only, imex_igw): wall-clock windows of the IMEX step and of the phases of
  ! the step that IMEX changes, with cudaDeviceSynchronize at entry and exit of each window (as
  ! fs_gpu_timers 2). 0 (default): every call returns at once, no synchronisation, no output;
  ! 1: the outer windows; 2: also the pieces inside the stage solve (more synchronisation points).
  ! No arithmetic is touched. Steps 1-10 are not counted; class 1 = steady step, 2 = b_ref rebuild
  ! step. At the last step one line per rank and window:
  !   IMEX_WIN <rank> <id> <name> <ms per steady step> <steady steps> <ms per rebuild step>
  !            <rebuild steps> <calls per steady step>
  integer, parameter :: IT_N = 30
  integer :: it_lev = 0, it_cls = 1, it_nstep = 0, it_ns(2) = 0
  real(8) :: it_acc(IT_N, 2) = 0.0d0, it_t0(IT_N) = 0.0d0
  integer :: it_cnt(IT_N, 2) = 0
  character(len=14), parameter :: it_name(IT_N) = [character(len=14) :: 'pg_store', 'stage_pred', 'stage_corr', &
       's.setup', 's.bc_exch', 's.rhs', 's.solve', 's.correct', 'p.start', 'p.zexch', 'p.ops', 'p.allreduce', &
       'p.up', 'p.finish', 'pred_transport', 'obc_faces', 'shift', 'corr_exch', 'visc3d', 'pre_step3d', &
       'step3d_uv2', 'fs_step', 'prsgrd', 'step', 'step3d_uv1', 'tracers_end', 'forces_pred', 'forces_corr', &
       'geom_pred', 'x30']

contains

  subroutine red_ensure(n)
    ! red_part(n, RED_MAXS): one partial per column of the tile
    integer(ip), intent(in) :: n
    if (red_n == n) return
    if (allocated(red_part)) deallocate(red_part)
    allocate(red_part(n, RED_MAXS)); red_part = 0.0_rp
#ifdef USE_CUDA
    if (.not. allocated(red_dres)) then
      allocate(red_dres(RED_MAXS), red_hres(RED_MAXS), red_tmp(RED_NT, RED_MAXS))
      red_dres = 0.0_rp; red_hres = 0.0_rp; red_tmp = 0.0_rp
    endif
#endif
    red_n = int(n)
  end subroutine red_ensure

#ifdef USE_CUDA
  attributes(global) subroutine red_kernel(n, m, part, res)
    ! One block: sum s of part(1:n, s) for s = 1..m, fixed order
    integer, value :: n, m
    real(rp), device, intent(in) :: part(n, RED_MAXS)
    real(rp), device :: res(RED_MAXS)
    real(rp), shared :: sh(RED_NT)
    integer :: t, s, c, w
    real(rp) :: a
    t = threadIdx%x
    do s = 1, m
      a = 0.0_rp
      do c = t, n, RED_NT
        a = a + part(c, s)
      enddo
      sh(t) = a
      call syncthreads()
      w = RED_NT / 2
      do while (w >= 1)
        if (t <= w) sh(t) = sh(t) + sh(t + w)
        call syncthreads()
        w = w / 2
      enddo
      if (t == 1) res(s) = sh(1)
      call syncthreads()
    enddo
  end subroutine red_kernel

  attributes(global) subroutine red_kpart(n, m, part, tmp)
    ! imex_gpu_opt 4: red_kernel's strided sums, thread t of the grid = thread t of red_kernel
    integer, value :: n, m
    real(rp), device, intent(in) :: part(n, RED_MAXS)
    real(rp), device :: tmp(RED_NT, RED_MAXS)
    integer :: t, s, c
    real(rp) :: a
    t = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    do s = 1, m
      a = 0.0_rp
      do c = t, n, RED_NT
        a = a + part(c, s)
      enddo
      tmp(t, s) = a
    enddo
  end subroutine red_kpart

  attributes(global) subroutine red_ktree(m, tmp, res)
    ! imex_gpu_opt 4: red_kernel's shared-memory tree over the RED_NT sums of red_kpart
    integer, value :: m
    real(rp), device, intent(in) :: tmp(RED_NT, RED_MAXS)
    real(rp), device :: res(RED_MAXS)
    real(rp), shared :: sh(RED_NT)
    integer :: t, s, w
    t = threadIdx%x
    do s = 1, m
      sh(t) = tmp(t, s)
      call syncthreads()
      w = RED_NT / 2
      do while (w >= 1)
        if (t <= w) sh(t) = sh(t) + sh(t + w)
        call syncthreads()
        w = w / 2
      enddo
      if (t == 1) res(s) = sh(1)
      call syncthreads()
    enddo
  end subroutine red_ktree
#endif

  subroutine red_set_split(on)
    ! imex_gpu_opt 4 (imex_igw check_settings)
    logical, intent(in) :: on
    red_split = on
  end subroutine red_set_split

  subroutine red_sum(m, out)
    ! out(1:m) = the sums of red_part(:, 1:m) (after the kernels that wrote
    ! them on work_stream or the legacy default stream)
    integer, intent(in) :: m
    real(rp), intent(out) :: out(m)
#ifdef USE_CUDA
    integer :: ierr
    if (red_split .and. red_n >= RED_SPLIT_MIN) then
      call red_kpart<<<RED_NT / RED_PW, RED_PW, 0, work_stream>>>(red_n, m, red_part, red_tmp)
      call red_ktree<<<1, RED_NT, 0, work_stream>>>(m, red_tmp, red_dres)
    else
      call red_kernel<<<1, RED_NT, 0, work_stream>>>(red_n, m, red_part, red_dres)
    endif
    ierr = cudaMemcpyAsync(red_hres, red_dres, m, cudaMemcpyDeviceToHost, work_stream)
    if (ierr /= 0) error stop 'red_sum: cudaMemcpyAsync'
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'red_sum: stream sync'
    out(1:m) = red_hres(1:m)
#else
    integer :: s
    do s = 1, m
      out(s) = sum(red_part(1:red_n, s))
    enddo
#endif
  end subroutine red_sum


  subroutine it_set(lev)
    ! imex_gpu_timers (imex_read_namelist)
    integer, intent(in) :: lev
    it_lev = lev
  end subroutine it_set

  subroutine it_step()
    ! start of a step: steps 1-10 are dropped
    if (it_lev <= 0) return
    it_nstep = it_nstep + 1
    if (it_nstep == 11) then
      it_acc = 0.0d0; it_cnt = 0; it_ns = 0
    endif
    it_cls = 1
  end subroutine it_step

  subroutine it_reb()
    if (it_lev <= 0) return
    it_cls = 2
  end subroutine it_reb

  subroutine itw(id, e)
    ! window id: e = 0 entry, 1 exit; the device is drained at both
    integer, intent(in) :: id, e
    integer(8) :: c, r
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (it_lev <= 0) return
#ifdef USE_CUDA
    ierr = cudaDeviceSynchronize()
#endif
    call system_clock(c, r)
    if (e == 0) then
      it_t0(id) = dble(c) / dble(r)
    else
      it_acc(id, it_cls) = it_acc(id, it_cls) + (dble(c) / dble(r) - it_t0(id))
      it_cnt(id, it_cls) = it_cnt(id, it_cls) + 1
      if (id == 24) it_ns(it_cls) = it_ns(it_cls) + 1
    endif
  end subroutine itw

  subroutine itw2(id, e)
    integer, intent(in) :: id, e
    if (it_lev < 2) return
    call itw(id, e)
  end subroutine itw2

  subroutine it_print(rank)
    integer, intent(in) :: rank
    integer :: i
    if (it_lev <= 0) return
    do i = 1, IT_N
      if (it_cnt(i, 1) + it_cnt(i, 2) == 0) cycle
      write(*,'(1x,a,i4,i4,1x,a14,f12.4,i6,f12.4,i6,f9.3)') 'IMEX_WIN', rank, i, it_name(i), &
           1.d3 * it_acc(i, 1) / max(it_ns(1), 1), it_ns(1), 1.d3 * it_acc(i, 2) / max(it_ns(2), 1), it_ns(2), &
           dble(it_cnt(i, 1)) / max(it_ns(1), 1)
    enddo
    flush(6)
  end subroutine it_print

end module imex_red

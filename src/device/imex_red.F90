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
  public :: red_ensure, red_sum
  integer, parameter :: RED_NT = 512, RED_MAXS = 4
#ifdef USE_CUDA
  real(rp), device, allocatable, public :: red_part(:,:)
  real(rp), device, allocatable :: red_dres(:)
  real(rp), pinned, allocatable :: red_hres(:)
#else
  real(rp), allocatable, public :: red_part(:,:)
#endif
  integer :: red_n = 0

contains

  subroutine red_ensure(n)
    ! red_part(n, RED_MAXS): one partial per column of the tile
    integer(ip), intent(in) :: n
    if (red_n == n) return
    if (allocated(red_part)) deallocate(red_part)
    allocate(red_part(n, RED_MAXS)); red_part = 0.0_rp
#ifdef USE_CUDA
    if (.not. allocated(red_dres)) then
      allocate(red_dres(RED_MAXS), red_hres(RED_MAXS))
      red_dres = 0.0_rp; red_hres = 0.0_rp
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
#endif

  subroutine red_sum(m, out)
    ! out(1:m) = the sums of red_part(:, 1:m) (after the kernels that wrote
    ! them on work_stream or the legacy default stream)
    integer, intent(in) :: m
    real(rp), intent(out) :: out(m)
#ifdef USE_CUDA
    integer :: ierr
    call red_kernel<<<1, RED_NT, 0, work_stream>>>(red_n, m, red_part, red_dres)
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

end module imex_red

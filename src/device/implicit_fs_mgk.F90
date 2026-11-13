#include "mem_attr.inc"
module fs_mgk
  ! Implicit free surface, fs_gpu_opt 32: the down and up legs of the global multigrid levels
  ! (fs_mgg's leg_down and leg_up: gzero, ghalo, gsmooth, gresid_restrict, gprolong and the
  ! coarsest vector's copies) each in one cooperative kernel whose steps are separated by grid
  ! synchronisations instead of kernel boundaries (one launch per leg instead of ~18). Every
  ! value gets the same expression with the same operands in the same order as in the separate
  ! kernels, so the result is bit-identical; the red-black passes visit only their colour's
  ! cells, and the planes of the down leg are zeroed in one step before the first sweep (no
  ! plane is read before its zeroing in the separate kernels either).
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: mgk_init, mgk_down, mgk_up

  integer, parameter :: MGK_NT = 256
#ifdef USE_CUDA
  integer, constant :: kgx(20), kgy(20)
#endif
  integer :: nblk = 0

contains

  subroutine mgk_init(glx, gly)
    integer, intent(in) :: glx(20), gly(20)
#ifdef USE_CUDA
    type(cudaDeviceProp) :: pr
    integer :: ierr, dev, o1, o2
    kgx = glx; kgy = gly
    ierr = cudaGetDevice(dev)
    ierr = cudaGetDeviceProperties(pr, dev)
    ! two blocks per SM, or fewer if both kernels cannot have that many resident (cooperative launch)
    o1 = 0; o2 = 0
    ierr = cudaOccupancyMaxActiveBlocksPerMultiprocessor(o1, kdn, MGK_NT, 0)
    ierr = cudaOccupancyMaxActiveBlocksPerMultiprocessor(o2, kup, MGK_NT, 0)
    nblk = max(1, min(2, o1, o2)) * pr%multiProcessorCount
#endif
  end subroutine mgk_init

  subroutine mgk_down(gx, gb, ge, gw, gs, gd, gt, dco, a, b, ng, nu, jac, px, py, om)
    integer, intent(in) :: a, b, ng, nu
    logical, intent(in) :: jac, px, py
    real(8), intent(in) :: om
#ifdef USE_CUDA
    real(8), device, dimension(0:a + 1, 0:b + 1, ng) :: gx, gb, ge, gw, gs, gd, gt
    real(8), device :: dco(*)
    integer :: ierr
    call kdn<<<nblk, MGK_NT, 0, work_stream>>>(gx, gb, ge, gw, gs, gd, gt, dco, a, b, ng, nu, &
         merge(1, 0, jac), merge(1, 0, px), merge(1, 0, py), om)
    ierr = cudaGetLastError()
    if (ierr /= 0) then
      write(*,*) 'implicit_fs: mgk_down launch: ', trim(cudaGetErrorString(ierr))
      error stop 'implicit_fs: mgk_down'
    endif
#else
    real(8), dimension(0:a + 1, 0:b + 1, ng) :: gx, gb, ge, gw, gs, gd, gt
    real(8) :: dco(*)
    error stop 'implicit_fs: fs_gpu_opt 32 needs the CUDA build'
#endif
  end subroutine mgk_down

  subroutine mgk_up(gx, gb, ge, gw, gs, gd, gt, dcx, mx, a, b, ng, nu, jac, px, py, om, i0, j0, nxd, nyd, mgx, mgy, l)
    integer, intent(in) :: a, b, ng, nu, i0, j0, nxd, nyd, mgx, mgy, l
    logical, intent(in) :: jac, px, py
    real(8), intent(in) :: om
#ifdef USE_CUDA
    real(8), device, dimension(0:a + 1, 0:b + 1, ng) :: gx, gb, ge, gw, gs, gd, gt
    real(8), device :: dcx(*)
    real(8), device :: mx(0:mgx + 1, 0:mgy + 1, *)
    integer :: ierr
    call kup<<<nblk, MGK_NT, 0, work_stream>>>(gx, gb, ge, gw, gs, gd, gt, dcx, mx, a, b, ng, nu, &
         merge(1, 0, jac), merge(1, 0, px), merge(1, 0, py), om, i0, j0, nxd, nyd, mgx, mgy, l)
    ierr = cudaGetLastError()
    if (ierr /= 0) then
      write(*,*) 'implicit_fs: mgk_up launch: ', trim(cudaGetErrorString(ierr))
      error stop 'implicit_fs: mgk_up'
    endif
#else
    real(8), dimension(0:a + 1, 0:b + 1, ng) :: gx, gb, ge, gw, gs, gd, gt
    real(8) :: dcx(*), mx(0:mgx + 1, 0:mgy + 1, *)
    error stop 'implicit_fs: fs_gpu_opt 32 needs the CUDA build'
#endif
  end subroutine mgk_up

#ifdef USE_CUDA
  attributes(grid_global) subroutine kdn(gx, gb, ge, gw, gs, gd, gt, dco, a, b, ng, nu, jac, px, py, om)
    use cooperative_groups
    integer, value :: a, b, ng, nu, jac, px, py
    real(8), value :: om
    real(8), device, dimension(0:a + 1, 0:b + 1, ng) :: gx, gb, ge, gw, gs, gd, gt
    real(8), device :: dco(*)
    type(grid_group) :: gg
    integer :: t0, nt, idx, i, j, gl, kk, c, col, p, hx, nx, ny, nxc, nyc, ic, jc, m, mm
    real(8) :: s, r, w
    gg = this_grid()
    t0 = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
    nt = gridDim%x * blockDim%x
    ! gzero of the planes 1..ng-1
    m = (a + 2) * (b + 2)
    do idx = t0, m * (ng - 1) - 1, nt
      gl = idx / m + 1; mm = idx - (gl - 1) * m
      i = mod(mm, a + 2); j = mm / (a + 2)
      gx(i, j, gl) = 0.0d0
    enddo
    call syncthreads(gg)
    do gl = 1, ng - 1
      nx = kgx(gl); ny = kgy(gl)
      do kk = 1, nu + 1
        ! ghalo (periodic wrap; zero halos stay zero): before each sweep and before the residual
        if (px /= 0 .or. py /= 0) then
          mm = max(nx, ny) + 2
          do idx = t0, mm, nt
            if (px /= 0 .and. idx >= 1 .and. idx <= ny) then
              gx(0, idx, gl) = gx(nx, idx, gl); gx(nx + 1, idx, gl) = gx(1, idx, gl)
            endif
            if (py /= 0 .and. idx >= 1 .and. idx <= nx) then
              gx(idx, 0, gl) = gx(idx, ny, gl); gx(idx, ny + 1, gl) = gx(idx, 1, gl)
            endif
          enddo
          call syncthreads(gg)
        endif
        if (kk > nu) exit
        ! gsmooth(gl, 1)
        if (jac /= 0) then
          do idx = t0, nx * ny - 1, nt
            i = mod(idx, nx) + 1; j = idx / nx + 1
            gt(i, j, gl) = gd(i, j, gl) * (gb(i, j, gl) + gw(i, j, gl) * gx(i - 1, j, gl) &
                 + gw(i + 1, j, gl) * gx(i + 1, j, gl) + gs(i, j, gl) * gx(i, j - 1, gl) &
                 + gs(i, j + 1, gl) * gx(i, j + 1, gl))
          enddo
          call syncthreads(gg)
          w = om
          do idx = t0, nx * ny - 1, nt
            i = mod(idx, nx) + 1; j = idx / nx + 1
            gx(i, j, gl) = (1.0d0 - w) * gx(i, j, gl) + w * gt(i, j, gl)
          enddo
          call syncthreads(gg)
        else
          hx = (nx + 1) / 2
          do c = 1, 2
            col = mod(c + 1, 2)
            p = mod(1 + col, 2)              ! the cells with mod(i + j + col, 2) == 1
            do idx = t0, hx * ny - 1, nt
              j = idx / hx + 1
              i = 1 + mod(1 + j + p, 2) + 2 * mod(idx, hx)
              if (i <= nx) gx(i, j, gl) = gd(i, j, gl) * (gb(i, j, gl) + gw(i, j, gl) * gx(i - 1, j, gl) &
                   + gw(i + 1, j, gl) * gx(i + 1, j, gl) + gs(i, j, gl) * gx(i, j - 1, gl) &
                   + gs(i, j + 1, gl) * gx(i, j + 1, gl))
            enddo
            call syncthreads(gg)
          enddo
        endif
      enddo
      ! gresid_restrict(gl)
      nxc = kgx(gl + 1); nyc = kgy(gl + 1)
      do idx = t0, nxc * nyc - 1, nt
        ic = mod(idx, nxc) + 1; jc = idx / nxc + 1
        s = 0.0d0
        do j = 2 * jc - 1, min(2 * jc, ny)
          do i = 2 * ic - 1, min(2 * ic, nx)
            r = 0.0d0
            if (ge(i, j, gl) > 0.0d0) r = gb(i, j, gl) - (ge(i, j, gl) * gx(i, j, gl) &
                 + gw(i, j, gl) * (gx(i, j, gl) - gx(i - 1, j, gl)) &
                 + gw(i + 1, j, gl) * (gx(i, j, gl) - gx(i + 1, j, gl)) &
                 + gs(i, j, gl) * (gx(i, j, gl) - gx(i, j - 1, gl)) &
                 + gs(i, j + 1, gl) * (gx(i, j, gl) - gx(i, j + 1, gl)))
            s = s + r
          enddo
        enddo
        gb(ic, jc, gl + 1) = s
      enddo
      call syncthreads(gg)
    enddo
    ! the coarsest level's rhs, packed i fastest
    nx = kgx(ng); ny = kgy(ng)
    do idx = t0, nx * ny - 1, nt
      i = mod(idx, nx) + 1; j = idx / nx + 1
      dco(i + (j - 1) * nx) = gb(i, j, ng)
    enddo
  end subroutine kdn

  attributes(grid_global) subroutine kup(gx, gb, ge, gw, gs, gd, gt, dcx, mx, a, b, ng, nu, jac, px, py, om, &
       i0, j0, nxd, nyd, mgx, mgy, l)
    use cooperative_groups
    integer, value :: a, b, ng, nu, jac, px, py, i0, j0, nxd, nyd, mgx, mgy, l
    real(8), value :: om
    real(8), device, dimension(0:a + 1, 0:b + 1, ng) :: gx, gb, ge, gw, gs, gd, gt
    real(8), device :: dcx(*)
    real(8), device :: mx(0:mgx + 1, 0:mgy + 1, *)
    type(grid_group) :: gg
    integer :: t0, nt, idx, i, j, gl, kk, c, col, p, hx, nx, ny, m
    real(8) :: v, w
    gg = this_grid()
    t0 = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
    nt = gridDim%x * blockDim%x
    ! gzero(ng) and the coarsest solution into its interior
    nx = kgx(ng); ny = kgy(ng)
    m = (a + 2) * (b + 2)
    do idx = t0, m - 1, nt
      i = mod(idx, a + 2); j = idx / (a + 2)
      v = 0.0d0
      if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) v = dcx(i + (j - 1) * nx)
      gx(i, j, ng) = v
    enddo
    call syncthreads(gg)
    do gl = ng - 1, 1, -1
      nx = kgx(gl); ny = kgy(gl)
      ! gprolong(gl)
      do idx = t0, nx * ny - 1, nt
        i = mod(idx, nx) + 1; j = idx / nx + 1
        if (ge(i, j, gl) > 0.0d0) gx(i, j, gl) = gx(i, j, gl) + gx((i + 1) / 2, (j + 1) / 2, gl + 1)
      enddo
      call syncthreads(gg)
      do kk = 1, nu
        if (px /= 0 .or. py /= 0) then
          m = max(nx, ny) + 2
          do idx = t0, m, nt
            if (px /= 0 .and. idx >= 1 .and. idx <= ny) then
              gx(0, idx, gl) = gx(nx, idx, gl); gx(nx + 1, idx, gl) = gx(1, idx, gl)
            endif
            if (py /= 0 .and. idx >= 1 .and. idx <= nx) then
              gx(idx, 0, gl) = gx(idx, ny, gl); gx(idx, ny + 1, gl) = gx(idx, 1, gl)
            endif
          enddo
          call syncthreads(gg)
        endif
        ! gsmooth(gl, 2)
        if (jac /= 0) then
          do idx = t0, nx * ny - 1, nt
            i = mod(idx, nx) + 1; j = idx / nx + 1
            gt(i, j, gl) = gd(i, j, gl) * (gb(i, j, gl) + gw(i, j, gl) * gx(i - 1, j, gl) &
                 + gw(i + 1, j, gl) * gx(i + 1, j, gl) + gs(i, j, gl) * gx(i, j - 1, gl) &
                 + gs(i, j + 1, gl) * gx(i, j + 1, gl))
          enddo
          call syncthreads(gg)
          w = om
          do idx = t0, nx * ny - 1, nt
            i = mod(idx, nx) + 1; j = idx / nx + 1
            gx(i, j, gl) = (1.0d0 - w) * gx(i, j, gl) + w * gt(i, j, gl)
          enddo
          call syncthreads(gg)
        else
          hx = (nx + 1) / 2
          do c = 1, 2
            col = mod(c + 2, 2)
            p = mod(1 + col, 2)
            do idx = t0, hx * ny - 1, nt
              j = idx / hx + 1
              i = 1 + mod(1 + j + p, 2) + 2 * mod(idx, hx)
              if (i <= nx) gx(i, j, gl) = gd(i, j, gl) * (gb(i, j, gl) + gw(i, j, gl) * gx(i - 1, j, gl) &
                   + gw(i + 1, j, gl) * gx(i + 1, j, gl) + gs(i, j, gl) * gx(i, j - 1, gl) &
                   + gs(i, j + 1, gl) * gx(i, j + 1, gl))
            enddo
            call syncthreads(gg)
          enddo
        endif
      enddo
    enddo
    ! this rank's block of global level 1 into the coarsest distributed level (zero halo)
    m = (mgx + 2) * (mgy + 2)
    do idx = t0, m - 1, nt
      i = mod(idx, mgx + 2); j = idx / (mgx + 2)
      v = 0.0d0
      if (i >= 1 .and. i <= nxd .and. j >= 1 .and. j <= nyd) v = gx(i0 + i, j0 + j, 1)
      mx(i, j, l) = v
    enddo
  end subroutine kup
#endif

end module fs_mgk

module fs_chol
  ! Implicit free surface, fs_gpu_opt 16: the coarsest multigrid level's banded Cholesky factor and
  ! solve (fs_host's factor and direct/fsh_direct_vec; the CPU's mg_factor and direct solve), in a
  ! file compiled on its own at -O2 -Mnovect (Makefile). The rest of fs_host stays at -O1 (nvfortran
  ! -O2 host code was miscompiled twice in this code base); here -O2 without vectorization keeps
  ! every operation and its order (vectorization reassociates the sums and changes the result; -O2
  ! -Mnovect was measured bit-identical to the -O1 build on the sargasso 24 x 24 level, factor and
  ! solve). The off-diagonal entries of a column and the forward substitution run in column order
  ! (each entry gets the same operations in the same order as in the row-ordered loops; independent
  ! updates interleaved). Measured on a Derecho CPU, 24 x 24 (bandwidth 24): factor 651 -> 162 us,
  ! solve 109 -> 24 us.
  implicit none
  private
  public :: chol_factor, chol_solve

contains

  subroutine chol_factor(nx, ny, e, cw, cs, perx, pery, n, b, ab, sv)
    integer, intent(in) :: nx, ny, n, b
    real(8), dimension(0:nx + 1, 0:ny + 1), intent(in) :: e, cw, cs
    logical, intent(in) :: perx, pery
    real(8), intent(inout) :: ab(0:b, n), sv(0:b + 1)
    integer :: i, j, k, kk, c, q, nr
    real(8) :: s, t
    ab = 0
    do j = 1, ny
      do i = 1, nx
        k = i + nx * (j - 1)
        if (e(i, j) > 0._8) then
          ab(0, k) = e(i, j) + cw(i, j) + cw(i + 1, j) + cs(i, j) + cs(i, j + 1)
        else
          ab(0, k) = 1._8
          cycle
        endif
        if (i > 1) call addl(k, k - 1, -cw(i, j))
        if (j > 1) call addl(k, k - nx, -cs(i, j))
        if (perx) then
          if (i == nx .and. nx > 2) call addl(k, k - nx + 1, -cw(1, j))
        endif
        if (pery) then
          if (j == ny .and. ny > 2) call addl(k, i, -cs(i, 1))
        endif
      enddo
    enddo
    do k = 1, n
      do kk = 1, min(b, n - k)
        i = mod(k + kk - 1, nx) + 1; j = (k + kk - 1) / nx + 1
        if (e(i, j) <= 0._8) ab(kk, k) = 0._8
      enddo
      i = mod(k - 1, nx) + 1; j = (k - 1) / nx + 1
      if (e(i, j) <= 0._8) ab(1:min(b, n - k), k) = 0._8
    enddo
    do c = 1, n
      s = ab(0, c)
      do kk = max(1, c - b), c - 1
        s = s - ab(c - kk, kk)**2
      enddo
      ab(0, c) = sqrt(s)
      nr = min(n, c + b) - c
      do q = 1, nr
        sv(q) = ab(q, c)
      enddo
      do kk = max(1, c + 1 - b), c - 1
        t = ab(c - kk, kk)
        do q = 1, min(nr, kk + b - c)
          sv(q) = sv(q) - ab(c + q - kk, kk) * t
        enddo
      enddo
      do q = 1, nr
        ab(q, c) = sv(q) / ab(0, c)
      enddo
    enddo
  contains
    subroutine addl(row, col, v)
      integer, intent(in) :: row, col
      real(8), intent(in) :: v
      integer :: rr, cc
      rr = max(row, col); cc = min(row, col)
      if (rr - cc <= b) ab(rr - cc, cc) = ab(rr - cc, cc) + v
    end subroutine addl
  end subroutine chol_factor

  subroutine chol_solve(n, b, ab, sol)
    ! sol: the rhs in, the solution out (forward substitution in column order, backward by rows)
    integer, intent(in) :: n, b
    real(8), intent(in) :: ab(0:b, n)
    real(8), intent(inout) :: sol(n)
    integer :: k, kk, q
    real(8) :: s, t
    do k = 1, n
      sol(k) = sol(k) / ab(0, k)
      t = sol(k)
      do q = 1, min(b, n - k)
        sol(k + q) = sol(k + q) - ab(q, k) * t
      enddo
    enddo
    do k = n, 1, -1
      s = sol(k)
      do kk = k + 1, min(n, k + b)
        s = s - ab(kk - k, k) * sol(kk)
      enddo
      sol(k) = s / ab(0, k)
    enddo
  end subroutine chol_solve

end module fs_chol

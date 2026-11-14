  ! Fused, W-scaled PCG of the stage solve (imex_pcg_fused = 1): one
  ! storage kind per inclusion (PFK = 4 or 8; imex_igw.F90 includes this
  ! file twice). The GPU port of the CPU's imex_pcgf8/imex_pcgf4,
  ! imex_pf_op8/4 and imex_pf_up8/4 (203a1a7a; FP32_SOLVER_SPEC.md).
  ! Storage of the coefficients rczk, czt, czb, cu, cv, hz, wr, the Jacobi
  ! reciprocal dinv and the vectors z, p, q is real(PFK); x, the residual
  ! rt, the start-up W M x0 (q0), the stored W M x and all arithmetic and
  ! reductions are FP64 (real(PFK) values converted on load, rounded once
  ! when stored, at the CPU's points). The CPU forms p in place row by row;
  ! here p has two buffers (pold, pnew) swapped every iteration.

  subroutine PF_SOLVE(r0, x, c2, tol, iters, resid, warm, stage, hv, reuse, &
                      rczk, czt, czb, cu, cv, hz, wr, ru, rv, dinv, z, pa, pb, q)
    ! CPU imex_pcgf<k>: start-up rt = W act b - W M x0 (W M x0 by an operator
    ! application on the FP64 x, or with reuse the stored W M x of the stage's
    ! previous solve), z = dt rt; per iteration the z exchange, sweep 1
    ! (deferred x += alpha p_old, p = z + beta p_old, q = W M p, <p,q>),
    ! sweep 2 (rt -= alpha q, z = dt rt, <rt,z>); stop on the recursive FP64
    ! residual; the pending x += alpha p; the stored W M x; x exchanged.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(rp), WORK_ATTR intent(inout) :: r0(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), intent(in) :: c2, tol
    integer, intent(out) :: iters
    real(rp), intent(out) :: resid
    logical, intent(in) :: warm, reuse
    integer, intent(in) :: stage
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: rczk(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: czt(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: czb(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: cu(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: cv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: hz(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: wr(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: ru(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rv(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(PFK), WORK_ATTR intent(in) :: dinv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 2)
    real(PFK), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: pa(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: pb(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: q(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz, s
    integer :: it, mode, cur
    real(rp) :: rho, rho0, pq, alpha, beta, tole, bnorm, s1, s2, al, c2l
    real(8) :: loc(2), glob(2)
    logical :: broke, pend, mqs
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    s = stage; c2l = c2
    broke = .false.; pend = .false.
    iters = 0; resid = 0.0_rp; tole = tol; bnorm = 0.0_rp; alpha = 0.0_rp; beta = 0.0_rp
    if (c2 <= 0.0_rp) return
    if (warm .and. .not. reuse) &                 ! W M x0 into pf_mq(:,:,:,3) (FP64 operand and result)
      call PF_OP(0, 0.0_rp, 0.0_rp, c2l, x, pf_mq, z, pa, pb, x, q, rczk, czt, czb, cu, cv, hz, wr, ru, rv, pq)
    if (.not. warm) call zero3(x)
    if (fs_on) then                                ! imex_gpu_stage: r0 formed in the start-up
      if (warm .and. reuse) then
        call PF_START_RHS(r0, hv, pf_mq, s, 1.0_rp, dinv, s, z, s1, s2)
      else if (warm) then
        call PF_START_RHS(r0, hv, pf_mq, 3_ip, 1.0_rp, dinv, s, z, s1, s2)
      else
        call PF_START_RHS(r0, hv, pf_mq, 3_ip, 0.0_rp, dinv, s, z, s1, s2)
      endif
    else if (warm .and. reuse) then
      call PF_START(r0, hv, pf_mq, s, 1.0_rp, dinv, s, z, s1, s2)      ! stored W M x of the stage
    else if (warm) then
      call PF_START(r0, hv, pf_mq, 3_ip, 1.0_rp, dinv, s, z, s1, s2)   ! W M x0 just applied
    else
      call PF_START(r0, hv, pf_mq, 3_ip, 0.0_rp, dinv, s, z, s1, s2)
    endif
    loc = [real(s1, 8), real(s2, 8)]
    call comm_allreduce_sum(2, loc, glob)
    rho = glob(1); rho0 = glob(1)
    if (warm) then
      bnorm = glob(2)
      tole = min(0.5_rp, tol * sqrt(bnorm / max(glob(1), 1.e-300_8)))
    endif
    if (.not. ieee_is_finite(rho0)) then           ! non-finite rhs or coefficients
      resid = 1.0_rp
      call breakdown('<r0,M^-1 r0>', rho0, 0)
      n_maxit_hits = n_maxit_hits + 1
      if (warm) call halo_fill_3d(x, STAGGER_RHO)
      ps_mver(s) = -1
      return
    endif
    if (rho0 <= 0.0_rp) then                       ! zero residual: x as it is
      if (warm) call halo_fill_3d(x, STAGGER_RHO)
      ps_mver(s) = -1
      return
    endif
    resid = 1.0_rp
    cur = 1                                        ! p of the last sweep 1: pa (1) or pb (2)
    do it = 1, imex_maxit
      call PF_ZEXCH(z)                             ! one cell, no corners: all the operator reads
      mode = 2
      if (it == 1) mode = 1                        ! p = z
      if (cur == 1) then                           ! p_old in pa, p_new into pb
        call PF_OP(mode, alpha, beta, c2l, x, pf_mq, z, pa, pb, x, q, rczk, czt, czb, cu, cv, hz, wr, ru, rv, pq)
      else
        call PF_OP(mode, alpha, beta, c2l, x, pf_mq, z, pb, pa, x, q, rczk, czt, czb, cu, cv, hz, wr, ru, rv, pq)
      endif
      cur = 3 - cur
      pend = .false.                               ! (x += alpha p_old done in the sweep)
      loc(1) = real(pq, 8)
      call comm_allreduce_sum(1, loc(1:1), glob(1:1))
      pq = glob(1)
      if (.not. ieee_is_finite(pq) .or. pq <= 0.0_rp) then
        call breakdown('<p,Ap>', pq, it)
        exit
      endif
      alpha = rho / pq
      call PF_UP(alpha, q, dinv, s, z, s1)
      pend = .true.                                ! x += alpha p pending
      loc(1) = real(s1, 8)
      call comm_allreduce_sum(1, loc(1:1), glob(1:1))
      iters = it
      if (.not. ieee_is_finite(glob(1)) .or. glob(1) < 0.0_8) then
        call breakdown('<r,M^-1 r>', glob(1), it)
        exit
      endif
      resid = sqrt(glob(1) / rho0)
      if (resid <= tole) exit
      beta = glob(1) / rho
      rho = glob(1)
    enddo
    if (fs_on) then                                ! imex_gpu_stage: the last x update, the stored
      al = alpha                                   ! W M x, one-cell x exchange
      mqs = (pcg_start == 3 .and. imex_warm == 1)
      if (cur == 1) then
        call PF_XMQ(pend, mqs, al, pa, x, r0, hv, s)
      else
        call PF_XMQ(pend, mqs, al, pb, x, r0, hv, s)
      endif
      if (broke .or. resid > tole) n_maxit_hits = n_maxit_hits + 1
      if (warm) resid = resid * sqrt(rho0 / max(bnorm, 1.e-300_8))
      if (mqs) then
        ps_mver(s) = coef_version; ps_mc2(s) = c2
      endif
      if (xring_on) then
        call pf_ring1_8(x)
      else
        call halo_fill_3d(x, STAGGER_RHO)
      endif
      return
    endif
    if (pend) then                                 ! the last x += alpha p
      al = alpha
      if (cur == 1) then
        call PF_XUPD(al, pa, x)
      else
        call PF_XUPD(al, pb, x)
      endif
    endif
    if (broke .or. resid > tole) n_maxit_hits = n_maxit_hits + 1
    if (warm) resid = resid * sqrt(rho0 / max(bnorm, 1.e-300_8))   ! relative to the cold solve's norm
    if (pcg_start == 3 .and. imex_warm == 1) then  ! W M x = W act b - rt for the next solve
      CUF_DO3
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            pf_mq(i, j, k, s) = act(i, j) * (darea(i, j) * hv(i, j, k) * (act(i, j) * r0(i, j, k) / ad_n2c(i, j, k))) &
                                - wk_r(i, j, k)
          enddo
        enddo
      enddo
      ps_mver(s) = coef_version; ps_mc2(s) = c2
    endif
    call halo_fill_3d(x, STAGGER_RHO)
  contains
    subroutine breakdown(what, val, iter)
      character(len=*), intent(in) :: what
      real(rp), intent(in) :: val
      integer, intent(in) :: iter
      broke = .true.
      if (talk) write(*,'(1x,a,a,a,es12.4,a,i4,a,es12.4)') 'IMEX_IGW WARNING: PCG breakdown, ', &
           what, ' =', val, ' at iteration', iter, '; stopped with relative residual', resid
    end subroutine breakdown
  end subroutine PF_SOLVE

  subroutine PF_START(r0, hv, q0, sq, wq, dinv, stage, z, rz, bb)
    ! rt = W act b - wq q0(:,:,:,sq) (W M x0), z = dt rt (rounded to real(PFK)),
    ! local <rt,z> and <W b, dt W b>; rt into wk_r
    real(rp), WORK_ATTR intent(in) :: r0(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: q0(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3)
    integer(ip), intent(in) :: sq
    real(rp), intent(in) :: wq
    real(PFK), WORK_ATTR intent(in) :: dinv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 2)
    integer, intent(in) :: stage
    real(PFK), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), intent(out) :: rz, bb
    integer(ip) :: i, j, k, nx, ny, nz, s, m
    real(rp) :: w, b1, bt, rt, a, b
    real(PFK) :: zs
    real(rp) :: ca, cb, rs(2)
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    s = stage; m = sq; w = wq; a = 0.0_rp; b = 0.0_rp
    if (redu_on) then                     ! column partials, deterministic sum (imex_red)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          ca = 0.0_rp; cb = 0.0_rp
          do k = 1, nz
            b1 = act(i, j) * r0(i, j, k) / ad_n2c(i, j, k)
            bt = darea(i, j) * hv(i, j, k) * b1
            rt = act(i, j) * bt - w * q0(i, j, k, m)
            zs = real(rt * real(dinv(i, j, k, s), 8), PFK)
            wk_r(i, j, k) = rt; z(i, j, k) = zs
            ca = ca + rt * real(zs, 8)
            cb = cb + bt * bt * real(dinv(i, j, k, s), 8)
          enddo
          red_part(i + (j - 1) * nx, 1) = ca; red_part(i + (j - 1) * nx, 2) = cb
        enddo
      enddo
      call red_sum(2, rs)
      rz = rs(1); bb = rs(2)
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          b1 = act(i, j) * r0(i, j, k) / ad_n2c(i, j, k)
          bt = darea(i, j) * hv(i, j, k) * b1
          rt = act(i, j) * bt - w * q0(i, j, k, m)
          zs = real(rt * real(dinv(i, j, k, s), 8), PFK)
          wk_r(i, j, k) = rt; z(i, j, k) = zs
          a = a + rt * real(zs, 8)
          b = b + bt * bt * real(dinv(i, j, k, s), 8)
        enddo
      enddo
    enddo
    rz = a; bb = b
  end subroutine PF_START

  subroutine PF_OP(mode, alpha, beta, c2, xo, qo, z, pold, pnew, x, q, rczk, czt, czb, cu, cv, hz, wr, &
                   ru, rv, pq)
    ! Sweep 1 (CPU imex_pf_op<k>; three kernels of the old apply_m plus the
    ! cell kernel):
    !   mode 0: operand xo (FP64), qo(:,:,:,3) = W M xo (FP64);
    !   mode 1: pnew = z (halos included), q = W M pnew;
    !   mode 2: x += alpha pold (interior), pnew = z + beta pold (halos
    !           included), q = W M pnew;
    ! W M p = wr p + c2 rczk S(p); pq the local <p, q> of the stored values.
    integer, intent(in) :: mode
    real(rp), intent(in) :: alpha, beta, c2
    real(rp), WORK_ATTR intent(in) :: xo(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: qo(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3)
    real(PFK), WORK_ATTR intent(in) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: pold(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: pnew(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: q(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: rczk(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: czt(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: czb(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: cu(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: cv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: hz(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: wr(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: ru(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rv(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(out) :: pq
    integer(ip) :: i, j, k, nx, ny, nz, md, inr
    real(rp) :: al, be, c2l, pr, sg, g, w, wm, dv, sv, qq, a, ca, rs(1)
    real(PFK) :: ps, qs
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    md = mode; al = alpha; be = beta; c2l = c2
    if (sweep_on) then
      call PF_OPS(mode, alpha, beta, c2, xo, qo, z, pold, pnew, x, q, rczk, czt, czb, cu, cv, hz, wr, ru, rv, pq)
      return
    endif
    ! (1) operand, zc = rczk p and dbar on the tile and one halo ring
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 0, nx + 1
        inr = 0
        if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) inr = 1
        do k = 1, nz
          if (md == 0) then
            pr = xo(i, j, k)
          else if (md == 1) then
            ps = z(i, j, k)
            pnew(i, j, k) = ps
            pr = real(ps, 8)
          else
            if (inr == 1) x(i, j, k) = x(i, j, k) + al * real(pold(i, j, k), 8)
            ps = real(real(z(i, j, k), 8) + be * real(pold(i, j, k), 8), PFK)
            pnew(i, j, k) = ps
            pr = real(ps, 8)
          endif
          op_zc(i, j, k) = real(rczk(i, j, k), 8) * pr
        enddo
        op_db(i, j, nz) = 0.0_rp
        sg = 0.0_rp
        do k = nz - 1, 1, -1
          sg = sg + real(czt(i, j, k), 8) * op_zc(i, j, k) - real(czb(i, j, k + 1), 8) * op_zc(i, j, k + 1)
          op_db(i, j, k) = -sg
        enddo
      enddo
    enddo
    ! (2) u-faces: g, its Hz_f-weighted column mean, F = Hz_f cxo (g - gbar)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        sg = 0.0_rp
        do k = 1, nz
          g = real(cu(i, j, k), 8) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
          op_fu(i, j, k) = g
          sg = sg + 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i - 1, j, k), 8)) * g
        enddo
        sg = sg * ru(i, j)
        do k = 1, nz
          op_fu(i, j, k) = 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i - 1, j, k), 8)) * cxo(i, j) &
                           * (op_fu(i, j, k) - sg)
        enddo
      enddo
    enddo
    ! (3) v-faces
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        sg = 0.0_rp
        do k = 1, nz
          g = real(cv(i, j, k), 8) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
          op_fv(i, j, k) = g
          sg = sg + 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i, j - 1, k), 8)) * g
        enddo
        sg = sg * rv(i, j)
        do k = 1, nz
          op_fv(i, j, k) = 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i, j - 1, k), 8)) * cyo(i, j) &
                           * (op_fv(i, j, k) - sg)
        enddo
      enddo
    enddo
    ! (4) cells: D with the vertical recursion of W; q = wr p + c2 rczk S
    a = 0.0_rp
    if (redu_on) then                     ! column partials, deterministic sum (imex_red)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          wm = 0.0_rp; ca = 0.0_rp
          do k = 1, nz
            dv = op_fu(i + 1, j, k) - op_fu(i, j, k) + op_fv(i, j + 1, k) - op_fv(i, j, k)
            w = 0.0_rp
            if (k < nz) w = wm - dv
            sv = real(cu(i + 1, j, k), 8) * op_fu(i + 1, j, k) - real(cu(i, j, k), 8) * op_fu(i, j, k) &
                 + real(cv(i, j + 1, k), 8) * op_fv(i, j + 1, k) - real(cv(i, j, k), 8) * op_fv(i, j, k) &
                 + real(czt(i, j, k), 8) * w - real(czb(i, j, k), 8) * wm
            wm = w
            if (md == 0) then
              pr = xo(i, j, k)
              qq = real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv
              qo(i, j, k, 3) = qq
            else
              pr = real(pnew(i, j, k), 8)
              qs = real(real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv, PFK)
              q(i, j, k) = qs
              qq = real(qs, 8)
            endif
            ca = ca + pr * qq
          enddo
          red_part(i + (j - 1) * nx, 1) = ca
        enddo
      enddo
      call red_sum(1, rs)
      pq = rs(1)
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        wm = 0.0_rp
        do k = 1, nz
          dv = op_fu(i + 1, j, k) - op_fu(i, j, k) + op_fv(i, j + 1, k) - op_fv(i, j, k)
          w = 0.0_rp
          if (k < nz) w = wm - dv
          sv = real(cu(i + 1, j, k), 8) * op_fu(i + 1, j, k) - real(cu(i, j, k), 8) * op_fu(i, j, k) &
               + real(cv(i, j + 1, k), 8) * op_fv(i, j + 1, k) - real(cv(i, j, k), 8) * op_fv(i, j, k) &
               + real(czt(i, j, k), 8) * w - real(czb(i, j, k), 8) * wm
          wm = w
          ! (one reduction update per iteration: a CUF reduction with two
          ! conditional updates of the same variable is not reduced)
          if (md == 0) then
            pr = xo(i, j, k)
            qq = real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv
            qo(i, j, k, 3) = qq
          else
            pr = real(pnew(i, j, k), 8)
            qs = real(real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv, PFK)
            q(i, j, k) = qs
            qq = real(qs, 8)
          endif
          a = a + pr * qq
        enddo
      enddo
    enddo
    pq = a
  end subroutine PF_OP

  subroutine PF_OPS(mode, alpha, beta, c2, xo, qo, z, pold, pnew, x, q, rczk, czt, czb, cu, cv, hz, wr, &
                    ru, rv, pq)
    ! imex_gpu_sweep = 1: PF_OP with the same arithmetic, restructured.
    ! (1) column quantities on the tile and its ring in one downward pass
    !     (x update, p, zc = rczk p, dbar), land columns (ad_act = 0: p, zc,
    !     dbar are 0 there) skipped;
    ! (2) u- and v-faces: g into op_fu, op_fv and its Hz_f-weighted column
    !     mean into op_sgu, op_sgv; closed faces (cxm, cym = 0) skipped;
    !     'modal': then g := (Y g)/sqrt(c2), mean 0, at the active open faces
    !     (obm_face_op; cxm, cym = dn_u, dm_v there, else cxo, cyo);
    ! (3) cells (land skipped: q = 0): F = Hz_f cxm (g - mean) formed here at
    !     the four faces (a closed face's F is 0 whatever its stored g), D
    !     with the vertical recursion of W, q, <p,q>.
    integer, intent(in) :: mode
    real(rp), intent(in) :: alpha, beta, c2
    real(rp), WORK_ATTR intent(in) :: xo(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: qo(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3)
    real(PFK), WORK_ATTR intent(in) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: pold(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: pnew(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(inout) :: q(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: rczk(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: czt(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: czb(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: cu(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: cv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: hz(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: wr(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: ru(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: rv(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(out) :: pq
    integer(ip) :: i, j, k, nx, ny, nz, md, inr
    real(rp) :: al, be, c2l, pr, sg, g, w, wm, dv, sv, qq, a, ca, rs(1), zcp, zck
    real(rp) :: fe, fw, fn, fs
    real(PFK) :: ps, qs
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    md = mode; al = alpha; be = beta; c2l = c2
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
    do j = 0, ny + 1
      do i = 0, nx + 1
        if (ad_act(i, j) > 0.0_rp) then
          inr = 0
          if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) inr = 1
          sg = 0.0_rp; zcp = 0.0_rp
          do k = nz, 1, -1
            if (md == 0) then
              pr = xo(i, j, k)
            else if (md == 1) then
              ps = z(i, j, k)
              pnew(i, j, k) = ps
              pr = real(ps, 8)
            else
              if (inr == 1) x(i, j, k) = x(i, j, k) + al * real(pold(i, j, k), 8)
              ps = real(real(z(i, j, k), 8) + be * real(pold(i, j, k), 8), PFK)
              pnew(i, j, k) = ps
              pr = real(ps, 8)
            endif
            zck = real(rczk(i, j, k), 8) * pr
            op_zc(i, j, k) = zck
            if (k == nz) then
              op_db(i, j, nz) = 0.0_rp
            else
              sg = sg + real(czt(i, j, k), 8) * zck - real(czb(i, j, k + 1), 8) * zcp
              op_db(i, j, k) = -sg
            endif
            zcp = zck
          enddo
        endif
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (j <= ny) then
          if (cxm(i, j) > 0.0_rp) then
            sg = 0.0_rp
            do k = 1, nz
              g = real(cu(i, j, k), 8) * (op_zc(i - 1, j, k) - op_zc(i, j, k)) + op_db(i - 1, j, k) - op_db(i, j, k)
              op_fu(i, j, k) = g
              sg = sg + 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i - 1, j, k), 8)) * g
            enddo
            op_sgu(i, j) = sg * ru(i, j)
          endif
        endif
        if (i <= nx) then
          if (cym(i, j) > 0.0_rp) then
            sg = 0.0_rp
            do k = 1, nz
              g = real(cv(i, j, k), 8) * (op_zc(i, j - 1, k) - op_zc(i, j, k)) + op_db(i, j - 1, k) - op_db(i, j, k)
              op_fv(i, j, k) = g
              sg = sg + 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i, j - 1, k), 8)) * g
            enddo
            op_sgv(i, j) = sg * rv(i, j)
          endif
        endif
      enddo
    enddo
    if (imex_obc_modal) call obm_face_op(c2l)   ! 'modal': the open faces' term (CPU obm_ufx, obm_vfx)
    a = 0.0_rp
    if (redu_on) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
      do j = 1, ny
        do i = 1, nx
          ca = 0.0_rp
          if (act(i, j) > 0.0_rp) then
            wm = 0.0_rp
            do k = 1, nz
              fe = 0.5_rp * (real(hz(i + 1, j, k), 8) + real(hz(i, j, k), 8)) * cxm(i + 1, j) &
                   * (op_fu(i + 1, j, k) - op_sgu(i + 1, j))
              fw = 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i - 1, j, k), 8)) * cxm(i, j) &
                   * (op_fu(i, j, k) - op_sgu(i, j))
              fn = 0.5_rp * (real(hz(i, j + 1, k), 8) + real(hz(i, j, k), 8)) * cym(i, j + 1) &
                   * (op_fv(i, j + 1, k) - op_sgv(i, j + 1))
              fs = 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i, j - 1, k), 8)) * cym(i, j) &
                   * (op_fv(i, j, k) - op_sgv(i, j))
              dv = fe - fw + fn - fs
              w = 0.0_rp
              if (k < nz) w = wm - dv
              sv = real(cu(i + 1, j, k), 8) * fe - real(cu(i, j, k), 8) * fw &
                   + real(cv(i, j + 1, k), 8) * fn - real(cv(i, j, k), 8) * fs &
                   + real(czt(i, j, k), 8) * w - real(czb(i, j, k), 8) * wm
              wm = w
              if (md == 0) then
                pr = xo(i, j, k)
                qq = real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv
                qo(i, j, k, 3) = qq
              else
                pr = real(pnew(i, j, k), 8)
                qs = real(real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv, PFK)
                q(i, j, k) = qs
                qq = real(qs, 8)
              endif
              ca = ca + pr * qq
            enddo
          endif
          red_part(i + (j - 1) * nx, 1) = ca
        enddo
      enddo
      call red_sum(1, rs)
      pq = rs(1)
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (act(i, j) > 0.0_rp) then
          wm = 0.0_rp
          do k = 1, nz
            fe = 0.5_rp * (real(hz(i + 1, j, k), 8) + real(hz(i, j, k), 8)) * cxm(i + 1, j) &
                 * (op_fu(i + 1, j, k) - op_sgu(i + 1, j))
            fw = 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i - 1, j, k), 8)) * cxm(i, j) &
                 * (op_fu(i, j, k) - op_sgu(i, j))
            fn = 0.5_rp * (real(hz(i, j + 1, k), 8) + real(hz(i, j, k), 8)) * cym(i, j + 1) &
                 * (op_fv(i, j + 1, k) - op_sgv(i, j + 1))
            fs = 0.5_rp * (real(hz(i, j, k), 8) + real(hz(i, j - 1, k), 8)) * cym(i, j) &
                 * (op_fv(i, j, k) - op_sgv(i, j))
            dv = fe - fw + fn - fs
            w = 0.0_rp
            if (k < nz) w = wm - dv
            sv = real(cu(i + 1, j, k), 8) * fe - real(cu(i, j, k), 8) * fw &
                 + real(cv(i, j + 1, k), 8) * fn - real(cv(i, j, k), 8) * fs &
                 + real(czt(i, j, k), 8) * w - real(czb(i, j, k), 8) * wm
            wm = w
            if (md == 0) then
              pr = xo(i, j, k)
              qq = real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv
              qo(i, j, k, 3) = qq
            else
              pr = real(pnew(i, j, k), 8)
              qs = real(real(wr(i, j, k), 8) * pr + c2l * real(rczk(i, j, k), 8) * sv, PFK)
              q(i, j, k) = qs
              qq = real(qs, 8)
            endif
            a = a + pr * qq
          enddo
        endif
      enddo
    enddo
    pq = a
  end subroutine PF_OPS

  subroutine PF_UP(alpha, q, dinv, stage, z, rz)
    ! Sweep 2 (CPU imex_pf_up<k>): rt -= alpha q (FP64, wk_r), z = dt rt
    ! (rounded to real(PFK)), local <rt,z>
    real(rp), intent(in) :: alpha
    real(PFK), WORK_ATTR intent(in) :: q(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(PFK), WORK_ATTR intent(in) :: dinv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 2)
    integer, intent(in) :: stage
    real(PFK), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), intent(out) :: rz
    integer(ip) :: i, j, k, nx, ny, nz, s
    real(rp) :: al, rr, a, ca, rs(1)
    real(PFK) :: zs
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    al = alpha; s = stage; a = 0.0_rp
    if (sweep_on) then                    ! imex_gpu_sweep: land columns skipped (rt, q, z, dinv are 0 there)
      if (redu_on) then
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
        do j = 1, ny
          do i = 1, nx
            ca = 0.0_rp
            if (act(i, j) > 0.0_rp) then
              do k = 1, nz
                rr = wk_r(i, j, k) - al * real(q(i, j, k), 8)
                wk_r(i, j, k) = rr
                zs = real(rr * real(dinv(i, j, k, s), 8), PFK)
                z(i, j, k) = zs
                ca = ca + rr * real(zs, 8)
              enddo
            endif
            red_part(i + (j - 1) * nx, 1) = ca
          enddo
        enddo
        call red_sum(1, rs)
        rz = rs(1)
        return
      endif
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then
            do k = 1, nz
              rr = wk_r(i, j, k) - al * real(q(i, j, k), 8)
              wk_r(i, j, k) = rr
              zs = real(rr * real(dinv(i, j, k, s), 8), PFK)
              z(i, j, k) = zs
              a = a + rr * real(zs, 8)
            enddo
          endif
        enddo
      enddo
      rz = a
      return
    endif
    if (redu_on) then                     ! column partials, deterministic sum (imex_red)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          ca = 0.0_rp
          do k = 1, nz
            rr = wk_r(i, j, k) - al * real(q(i, j, k), 8)
            wk_r(i, j, k) = rr
            zs = real(rr * real(dinv(i, j, k, s), 8), PFK)
            z(i, j, k) = zs
            ca = ca + rr * real(zs, 8)
          enddo
          red_part(i + (j - 1) * nx, 1) = ca
        enddo
      enddo
      call red_sum(1, rs)
      rz = rs(1)
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          rr = wk_r(i, j, k) - al * real(q(i, j, k), 8)
          wk_r(i, j, k) = rr
          zs = real(rr * real(dinv(i, j, k, s), 8), PFK)
          z(i, j, k) = zs
          a = a + rr * real(zs, 8)
        enddo
      enddo
    enddo
    rz = a
  end subroutine PF_UP

  subroutine PF_START_RHS(r0, hv, q0, sq, wq, dinv, stage, z, rz, bb)
    ! imex_gpu_stage = 1: PF_START with the stage rhs formed in the same
    ! column pass: r0 = 0 + act dts n2c D(F) (stage_increment; F = wk_fu,
    ! wk_fv of rhs_transport) - cfbak (hs1 + hs2) (predictor; the original's
    ! dpw = dps + dp_corr) or - hs1 (corrector; dps), stored in r0; then
    ! rt, z and the sums as PF_START. The same expressions in the same
    ! order as stage_increment, imex_solve_stage and PF_START.
    real(rp), WORK_ATTR intent(inout) :: r0(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: q0(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3)
    integer(ip), intent(in) :: sq
    real(rp), intent(in) :: wq
    real(PFK), WORK_ATTR intent(in) :: dinv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 2)
    integer, intent(in) :: stage
    real(PFK), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), intent(out) :: rz, bb
    integer(ip) :: i, j, k, nx, ny, nz, s, m, pr
    real(rp) :: w, b1, bt, rt, a, b, ca, cb, rs(2), dtl, cf, zr, wv, wm, d, rr
    real(PFK) :: zs
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    s = stage; m = sq; w = wq; a = 0.0_rp; b = 0.0_rp
    dtl = fs_dtl; cf = fs_cb; zr = fs_zero
    pr = merge(1_ip, 0_ip, stage == imex_stage_pred)
    if (pr == 1 .and. fs_dpw) pr = 2               ! dpw rescaled at this step's b_ref rebuild
    if (redu_on) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
      do j = 1, ny
        do i = 1, nx
          ca = 0.0_rp; cb = 0.0_rp; wm = 0.0_rp
          if (act(i, j) > 0.0_rp) then     ! (act = 0: r0, rt, z and the sums are 0)
          do k = 1, nz
            wv = 0.0_rp
            if (k < nz) wv = wm - (wk_fu(i + 1, j, k) - wk_fu(i, j, k) + wk_fv(i, j + 1, k) - wk_fv(i, j, k))
            d = -ad_en2(i, j, k) * (ad_cu(i + 1, j, k) * wk_fu(i + 1, j, k) - ad_cu(i, j, k) * wk_fu(i, j, k) &
                + ad_cv(i, j + 1, k) * wk_fv(i, j + 1, k) - ad_cv(i, j, k) * wk_fv(i, j, k) &
                + ad_czt(i, j, k) * wv - ad_czb(i, j, k) * wm)
            wm = wv
            rr = zr + act(i, j) * dtl * ad_n2c(i, j, k) * d
            if (pr == 1) then
              rr = rr - cf * (hs1(i, j, k) + hs2(i, j, k))
            else if (pr == 2) then
              rr = rr - cf * dpw(i, j, k)
            else
              rr = rr - hs1(i, j, k)
            endif
            r0(i, j, k) = rr
            b1 = act(i, j) * rr / ad_n2c(i, j, k)
            bt = darea(i, j) * hv(i, j, k) * b1
            rt = act(i, j) * bt - w * q0(i, j, k, m)
            zs = real(rt * real(dinv(i, j, k, s), 8), PFK)
            wk_r(i, j, k) = rt; z(i, j, k) = zs
            ca = ca + rt * real(zs, 8)
            cb = cb + bt * bt * real(dinv(i, j, k, s), 8)
          enddo
          endif
          red_part(i + (j - 1) * nx, 1) = ca; red_part(i + (j - 1) * nx, 2) = cb
        enddo
      enddo
      call red_sum(2, rs)
      rz = rs(1); bb = rs(2)
      return
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        wm = 0.0_rp
        if (act(i, j) > 0.0_rp) then
        do k = 1, nz
          wv = 0.0_rp
          if (k < nz) wv = wm - (wk_fu(i + 1, j, k) - wk_fu(i, j, k) + wk_fv(i, j + 1, k) - wk_fv(i, j, k))
          d = -ad_en2(i, j, k) * (ad_cu(i + 1, j, k) * wk_fu(i + 1, j, k) - ad_cu(i, j, k) * wk_fu(i, j, k) &
              + ad_cv(i, j + 1, k) * wk_fv(i, j + 1, k) - ad_cv(i, j, k) * wk_fv(i, j, k) &
              + ad_czt(i, j, k) * wv - ad_czb(i, j, k) * wm)
          wm = wv
          rr = zr + act(i, j) * dtl * ad_n2c(i, j, k) * d
          if (pr == 1) then
            rr = rr - cf * (hs1(i, j, k) + hs2(i, j, k))
          else if (pr == 2) then
            rr = rr - cf * dpw(i, j, k)
          else
            rr = rr - hs1(i, j, k)
          endif
          r0(i, j, k) = rr
          b1 = act(i, j) * rr / ad_n2c(i, j, k)
          bt = darea(i, j) * hv(i, j, k) * b1
          rt = act(i, j) * bt - w * q0(i, j, k, m)
          zs = real(rt * real(dinv(i, j, k, s), 8), PFK)
          wk_r(i, j, k) = rt; z(i, j, k) = zs
          a = a + rt * real(zs, 8)
          b = b + bt * bt * real(dinv(i, j, k, s), 8)
        enddo
        endif
      enddo
    enddo
    rz = a; bb = b
  end subroutine PF_START_RHS

  subroutine PF_XMQ(pend, mqs, alpha, p, x, r0, hv, stage)
    ! imex_gpu_stage = 1: PF_XUPD (pend) and the stored W M x = W act b - rt
    ! of the stage (mqs) in one pass (interior), the same expressions
    logical, intent(in) :: pend, mqs
    real(rp), intent(in) :: alpha
    real(PFK), WORK_ATTR intent(in) :: p(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: r0(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(in) :: hv(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer, intent(in) :: stage
    integer(ip) :: i, j, k, nx, ny, nz, s, ipd, imq
    real(rp) :: al
    if (.not. (pend .or. mqs)) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    al = alpha; s = stage
    ipd = merge(1_ip, 0_ip, pend); imq = merge(1_ip, 0_ip, mqs)
    CUF_DO3
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (act(i, j) > 0.0_rp) then     ! (act = 0: p = 0 and W M x = 0 there)
            if (ipd == 1) x(i, j, k) = x(i, j, k) + al * real(p(i, j, k), 8)
            if (imq == 1) pf_mq(i, j, k, s) = act(i, j) * (darea(i, j) * hv(i, j, k) * (act(i, j) * r0(i, j, k) &
                                              / ad_n2c(i, j, k))) - wk_r(i, j, k)
          endif
        enddo
      enddo
    enddo
  end subroutine PF_XMQ

  subroutine PF_XUPD(alpha, p, x)
    ! x += alpha p (interior)
    real(rp), intent(in) :: alpha
    real(PFK), WORK_ATTR intent(in) :: p(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    real(rp), WORK_ATTR intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: al
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    al = alpha
    CUF_DO3
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          x(i, j, k) = x(i, j, k) + al * real(p(i, j, k), 8)
        enddo
      enddo
    enddo
  end subroutine PF_XUPD

  subroutine PF_ZEXCH(z)
    ! z's halo, one cell wide (all the operator reads; corners are not
    ! read): real(8) the 2-wide exchange of z itself; real(4) the tile's
    ! edge ring through the FP64 wk_z (exact), the 2-wide exchange, the halo
    ! ring back (CPU pf_zexch4)
    real(PFK), WORK_ATTR intent(inout) :: z(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
#if PFK == 4
    integer(ip) :: i, j, k, nx, ny, nz
#endif
#if PFK == 4
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    CUF_DO2
    do k = 1, nz
      do j = 1, ny
        wk_z(1, j, k) = real(z(1, j, k), 8); wk_z(nx, j, k) = real(z(nx, j, k), 8)
      enddo
    enddo
    CUF_DO2
    do k = 1, nz
      do i = 1, nx
        wk_z(i, 1, k) = real(z(i, 1, k), 8); wk_z(i, ny, k) = real(z(i, ny, k), 8)
      enddo
    enddo
    call halo_fill_3d(wk_z, STAGGER_RHO)
    CUF_DO2
    do k = 1, nz
      do j = 1, ny
        z(0, j, k) = real(wk_z(0, j, k), 4); z(nx + 1, j, k) = real(wk_z(nx + 1, j, k), 4)
      enddo
    enddo
    CUF_DO2
    do k = 1, nz
      do i = 1, nx
        z(i, 0, k) = real(wk_z(i, 0, k), 4); z(i, ny + 1, k) = real(wk_z(i, ny + 1, k), 4)
      enddo
    enddo
#else
    call halo_fill_3d(z, STAGGER_RHO)
#endif
  end subroutine PF_ZEXCH

#if PFK == 8
  subroutine PF_RING1(a)
    ! imex_gpu_stage's x exchange (xring_on): one-cell halo without corners of a real(8)
    ! rho-point field in one round: the four edge rows of the tile (columns
    ! 1, nx for rows 1..ny; rows 1, ny for columns 1..nx) packed by one
    ! kernel, exchanged with the four neighbours at once (GPU-aware MPI,
    ! comm_ring_dev<k>; periodic self-wrap read directly from the send
    ! buffers), unpacked by one kernel. Physical-edge ghosts and corners are
    ! left as they are (the operator multiplies the former by rczk = 0 and
    ! never reads the latter). The values are those of the exchanges above.
    real(PFK), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip) :: k, m, nx, ny, nz, mx, nb(4), w, e, s, n
    integer :: ierr
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nb = [decomp%nbr_w, decomp%nbr_e, decomp%nbr_s, decomp%nbr_n]
    if (all(nb == NBR_NONE)) return
    if (.not. allocated(RG_SW)) then
      allocate(RG_SW(ny * nz), RG_SE(ny * nz), RG_RW(ny * nz), RG_RE(ny * nz))
      allocate(RG_SS(nx * nz), RG_SN(nx * nz), RG_RS(nx * nz), RG_RN(nx * nz))
      RG_SW = 0; RG_SE = 0; RG_RW = 0; RG_RE = 0; RG_SS = 0; RG_SN = 0; RG_RS = 0; RG_RN = 0
    endif
    mx = max(nx, ny)
    CUF_DO2
    do k = 1, nz
      do m = 1, mx
        if (m <= ny) then
          RG_SW(m + (k - 1) * ny) = a(1, m, k); RG_SE(m + (k - 1) * ny) = a(nx, m, k)
        endif
        if (m <= nx) then
          RG_SS(m + (k - 1) * nx) = a(m, 1, k); RG_SN(m + (k - 1) * nx) = a(m, ny, k)
        endif
      enddo
    enddo
    if (any(nb >= 0)) then
#ifdef USE_CUDA
      ierr = cudaStreamSynchronize(work_stream)
      if (ierr /= 0) error stop 'pf_ring1: stream sync'
#endif
      call PF_COMMRING(nb, int(ny * nz), int(nx * nz), RG_SW, RG_SE, RG_SS, RG_SN, RG_RW, RG_RE, RG_RS, RG_RN)
    endif
    w = nb(1); e = nb(2); s = nb(3); n = nb(4)
    CUF_DO2
    do k = 1, nz
      do m = 1, mx
        if (m <= ny) then
          if (w == NBR_SELF) then
            a(0, m, k) = RG_SE(m + (k - 1) * ny)
          else if (w /= NBR_NONE) then
            a(0, m, k) = RG_RW(m + (k - 1) * ny)
          endif
          if (e == NBR_SELF) then
            a(nx + 1, m, k) = RG_SW(m + (k - 1) * ny)
          else if (e /= NBR_NONE) then
            a(nx + 1, m, k) = RG_RE(m + (k - 1) * ny)
          endif
        endif
        if (m <= nx) then
          if (s == NBR_SELF) then
            a(m, 0, k) = RG_SN(m + (k - 1) * nx)
          else if (s /= NBR_NONE) then
            a(m, 0, k) = RG_RS(m + (k - 1) * nx)
          endif
          if (n == NBR_SELF) then
            a(m, ny + 1, k) = RG_SS(m + (k - 1) * nx)
          else if (n /= NBR_NONE) then
            a(m, ny + 1, k) = RG_RN(m + (k - 1) * nx)
          endif
        endif
      enddo
    enddo
  end subroutine PF_RING1
#endif

#include "mem_attr.inc"
#include "cuf_launch.inc"
#ifdef USE_CUDA
#define FSD device,
#else
#define FSD
#endif
module implicit_fs_gpu
  ! Implicit free surface (namelist group &implicit_fs, fs_mode =
  ! 'implicit'): the GPU port of ucla-roms implicit_fs_mod.F90 (branch
  ! implicit-fs-defaults; the discrete algorithm, index ranges and time
  ! levels of the CPU module, docs/model_settings/implicit_free_surface.md
  ! there).
  ! One generalized-alpha step of the depth-integrated momentum and
  ! continuity equations per baroclinic step, reduced to a symmetric
  ! positive-definite 5-point Helmholtz problem for delta = zeta^n+1 -
  ! zeta^n, solved by PCG (Jacobi or a geometric multigrid V-cycle),
  ! replaces step2d's split-explicit fast loop. It fills what the fast loop
  ! filled: zeta(knew), Zt_avg1, ubar/vbar(knew), DU_avg1 = U^n+1, DU_avg2 =
  ! the flux that advanced zeta, DU_avg_bak, and calls set_depth.
  ! fs_mode = 'split' (the default, also when the group is absent) leaves
  ! the split path untouched.
  !
  ! Layout: one tile per rank, interior 1:nx, 1:ny (the CPU's istr:iend,
  ! jstr:jend), arrays decomp%is:ie, js:je as gpu_state; the CPU's
  ! WESTERN_EDGE ... are decomp%edge_w ..., OBC_WEST ... the open flags
  ! obc_west ... (Flather M2 + Chapman zeta, the only open-boundary type of
  ! roms_gpu). All per-step work is device kernels on work_stream; the host
  ! sees only the reduction results (deterministic fixed-order sums) and the coarse multigrid levels, which are held on
  ! the host as on the CPU (fs_host, implicit_fs_host.F90).
  !
  ! Multigrid (FS_SPEC 5): level 0 the fine arrays; 1..mg_ld distributed
  ! coarse levels (2x2 aggregation in the tile) on the device in planes of
  ! (0:mx+1, 0:my+1, l); the coarsest distributed level is gathered on the
  ! host into one global grid (fs_host). The level rule is the CPU's
  ! (fs_mg_levels); the GPU-only key fs_gpu_gather > 0 gathers instead at
  ! the first level whose global cell count is <= fs_gpu_gather.
  ! Smoothers: 'rbgs' (tile-local red-black colours, one exchange before
  ! each sweep: the CPU's block-hybrid order, reproduced exactly for the
  ! same tiling) and 'jacobi' (damped, omega 0.8).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state, only: zeta, ubar, vbar, h, pm, pn, dn_u, dm_v, f, rmask, umask, vmask, swflx, r_D, dndx, dmde, &
                       riv_umask, riv_vmask, &
                       rufrc, rvfrc, DU_avg1, DU_avg2, DU_avg_bak, DV_avg1, DV_avg2, DV_avg_bak, &
                       zt_avg1, zeta_west, zeta_east, zeta_south, zeta_north, ubar_west, ubar_east, &
                       vbar_south, vbar_north, vbar_west, vbar_east, ubar_south, ubar_north, &
                       obc_west, obc_east, obc_south, obc_north, ubind, rhoS, rhoA
  use time_state, only: dt, iic, ntstart, ntimes, knew, kstp, fast
  use formulas, only: g
  use halo_mod, only: halo_fill_zuv_n
  use geom, only: set_depth
  use comm_mod, only: comm_allreduce_sum, comm_allreduce_max, comm_world_rank, comm_abort
  use rivers_mod, only: river_on, rf_flx, rivers_fs_map
  use cdr_src_mod, only: cdr_vol_on, cdr_fs_src
  use run_timers, only: wall_time
  use fs_data
  use fs_mg, only: mg_build_layout, mg_setup, mg_vcycle
  use fs_ipc, only: ipc_init, ipc_on
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private

  public :: fs_read_namelist, fs_step, fs_on, fs_rst_prepare, fs_rst_loaded
  public :: fs_src_diag, fs_srcdiag, fs_rst_scal_vals, fs_rst_scal_in, fs_mg_loaded
  public :: fs_global_atts

contains

  !======================================================================
  ! Namelist
  !======================================================================
  subroutine fs_read_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios, i
    real(rp) :: alf                        ! alpha_f
    character(len=512) :: msg
    talk = (comm_world_rank() == 0)
    rewind(unit)
    read(unit, nml=implicit_fs, iostat=ios, iomsg=msg)
    if (ios > 0) then
      do i = 1, len(msg)                     ! nvfortran pads iomsg with NULs (as nc_io's iomsg_text)
        if (iachar(msg(i:i)) < 32) msg(i:i) = ' '
      enddo
      write(*,*) 'IMPLICIT_FS: could not read the IMPLICIT_FS section of the namelist file ('// &
           trim(adjustl(msg))//')'
      error stop 'read_namelist: unreadable namelist group (see above)'
    endif
    if (ios < 0) then
      fs_mode = 'split'
      fs_on = .false.
      return
    endif
    fs_on = (trim(fs_mode) == 'implicit')
    fs_srcdiag = (fs_verbose >= 2)
    if (.not. fs_on .and. trim(fs_mode) /= 'split') error stop "IMPLICIT_FS: fs_mode must be 'split' or 'implicit'"
    if (.not. fs_on) return
    if (fs_theta < 0.5_rp .or. fs_theta > 1.0_rp .or. fs_theta_div < 0.5_rp .or. fs_theta_div > 1.0_rp) &
         error stop 'IMPLICIT_FS: fs_theta and fs_theta_div must lie in [0.5, 1]'
    if (fs_rho_inf < 0.0_rp .or. fs_rho_inf >= 1.0_rp) error stop 'IMPLICIT_FS: fs_rho_inf must lie in [0, 1):'// &
         " 1 (Crank-Nicolson) is resonant with the 3-D step's 2 dt mode"
    if (fs_nsub < 1 .or. fs_nsub > 4) error stop 'IMPLICIT_FS: fs_nsub must lie in [1, 4]'
    call fs_scheme_coef()
    alf = 1.0_rp / (1.0_rp + fs_rho_inf)
    ! forcing time of the step: alpha_f; with sub-steps (rufrc held over the step) the mean of the
    ! sub-steps' forcing times n + (k-1+alpha_f)/fs_nsub (CPU b6325f06)
    sft = alf
    if (fs_nsub > 1) sft = (alf + 0.5_rp * real(fs_nsub - 1, rp)) / real(fs_nsub, rp)
    fs_bry_phase = 1.0_rp
    if (fs_nsub > 1) fs_bry_phase = 0.5_rp * real(fs_nsub + 1, rp) / real(fs_nsub, rp)
    select case (trim(fs_pot_time))        ! the tidal potential at n+sft (tide_io: fs_pot_phase)
    case ('af')
      fs_pot_phase = 1.0_rp / (1.0_rp + fs_rho_inf)
      if (fs_nsub > 1) fs_pot_phase = sft
    case ('mid'); fs_pot_phase = 0.5_rp
    case default; error stop "IMPLICIT_FS: fs_pot_time must be 'af' or 'mid'"
    end select
    if (trim(fs_mg_faces) /= 'width' .and. trim(fs_mg_faces) /= 'half') &
         error stop "IMPLICIT_FS: fs_mg_faces must be 'width' or 'half'"
    if (fs_tol <= 0.0_rp .or. fs_maxit < 1) error stop 'IMPLICIT_FS: fs_tol > 0 and fs_maxit >= 1 required'
    if (trim(fs_precond) /= 'jacobi' .and. trim(fs_precond) /= 'mg') &
         error stop "IMPLICIT_FS: fs_precond must be 'jacobi' or 'mg'"
    mg_on = (trim(fs_precond) == 'mg')
    if (fs_mg_nu < 1 .or. fs_mg_lag < 1 .or. (fs_mg_smoother(1:4) /= 'rbgs' .and. fs_mg_smoother(1:6) /= 'jacobi')) &
         error stop "IMPLICIT_FS: fs_mg_nu >= 1, fs_mg_lag >= 1, fs_mg_smoother 'rbgs' or 'jacobi' required"
    jac_sm = (fs_mg_smoother(1:1) == 'j')
    if (fs_gpu_gather < 0) error stop 'IMPLICIT_FS: fs_gpu_gather >= 0 required'
    if (fs_tang_nudge < 0.0_rp) error stop 'IMPLICIT_FS: fs_tang_nudge must be >= 0'
    if (trim(fs_gpu_coarse) /= 'device' .and. trim(fs_gpu_coarse) /= 'host') &
         error stop "IMPLICIT_FS: fs_gpu_coarse must be 'device' or 'host'"
    dev_glob = (trim(fs_gpu_coarse) == 'device')
    if (fs_gpu_opt < 0 .or. fs_gpu_opt > 511) error stop 'IMPLICIT_FS: fs_gpu_opt must lie in [0, 511]'
    if (fs_gpu_timers < 0 .or. fs_gpu_timers > 2) error stop 'IMPLICIT_FS: fs_gpu_timers must be 0, 1 or 2'
    call fs_opt_flags()
    if (talk) then
      write(*,'(/1x,A)') 'IMPLICIT_FS: fs_mode = implicit (the step2d fast loop is replaced)'
      write(*,'(1x,A,F6.3,A,F7.4,A,F7.4,A,F7.4,A,F7.4,A,F7.4)') 'IMPLICIT_FS: fs_rho_inf =', fs_rho_inf, &
           ' -> beta1 =', beta1, ' th_p =', thp, ' th_d =', thd, ' gamma =', gam_a, ' (fs_theta, fs_theta_div ', &
           fs_theta
      write(*,'(1x,A,F7.4,A,ES9.2,A,I6,A,A,A,L1,A,I2,A,I2)') 'IMPLICIT_FS: fs_theta_div =', fs_theta_div, &
           ' fs_tol =', fs_tol, ' fs_maxit =', fs_maxit, ' fs_precond = ', trim(fs_precond), &
           ' fs_drag_implicit = ', fs_drag_implicit, ' fs_warm =', fs_warm, ' fs_verbose =', fs_verbose
      write(*,'(1x,A,I3,A,I3,A,A,A,I3,A,I8,A,I2)') 'IMPLICIT_FS: fs_mg_levels =', fs_mg_levels, &
           ' fs_mg_nu =', fs_mg_nu, ' fs_mg_smoother = ', trim(fs_mg_smoother), ' fs_mg_lag =', fs_mg_lag, &
           ' fs_gpu_gather =', fs_gpu_gather, ' fs_gpu_timers =', fs_gpu_timers
      write(*,'(1x,A,F8.3,A,L1)') 'IMPLICIT_FS: fs_tang_nudge =', fs_tang_nudge, ' fs_cor_metric = ', fs_cor_metric
      write(*,'(1x,A,I2,A,F7.4,A,F7.4,A)') 'IMPLICIT_FS: fs_nsub =', fs_nsub, ' (implicit sub-steps of dt/fs_nsub;'// &
           ' explicit forcing at time +', sft, ' dt, boundary tidal harmonics at time +', fs_bry_phase, ' dt)'
      write(*,'(1x,A,L1)') 'IMPLICIT_FS: fs_pg_rho = ', fs_pg_rho
      write(*,'(1x,A,L1)') 'IMPLICIT_FS: fs_flather_zghost = ', fs_flather_zghost
      write(*,'(1x,A,A,A,F7.4,A)') 'IMPLICIT_FS: fs_pot_time = ', trim(fs_pot_time), ' (tidal potential at time +', &
           fs_pot_phase, ' dt)'
      write(*,'(1x,A,A,A,I6,A,A,A,L1,A)') 'IMPLICIT_FS: fs_mg_faces = ', trim(fs_mg_faces), &
           ' (CPU layout keys read, not used: fs_mg_gather =', fs_mg_gather, ' fs_mg_global_on = ', &
           trim(fs_mg_global_on), ' fs_exch_thin = ', fs_exch_thin, ')'
      write(*,'(1x,A,A,A,I2,A,I3)') 'IMPLICIT_FS: fs_gpu_coarse = ', trim(fs_gpu_coarse), ' fs_gpu_graph =', &
           fs_gpu_graph, ' fs_gpu_opt =', fs_gpu_opt
      if (fs_selftest > 0) write(*,'(1x,A)') 'IMPLICIT_FS: note: fs_selftest is not implemented on the GPU (ignored)'
    endif
  end subroutine fs_read_namelist

  subroutine fs_global_atts(ncid)
    ! Global attributes of an output file (the CPU's put_global_atts, ucla-roms e34f73bd): fs_mode
    ! always ('split' also when the group is absent); with 'implicit' also fs_rho_inf (double) and fs_nsub
    ! (int). Called by the output modules where they write the file's title (compute rank 0, which has
    ! read the namelist).
    use netcdf, only: nf90_put_att, nf90_global, nf90_noerr, nf90_strerror
    integer, intent(in) :: ncid
    integer :: ierr
    ierr = nf90_put_att(ncid, nf90_global, 'fs_mode', trim(fs_mode))
    if (ierr == nf90_noerr .and. fs_on) then
      ierr = nf90_put_att(ncid, nf90_global, 'fs_rho_inf', real(fs_rho_inf, 8))
      if (ierr == nf90_noerr) ierr = nf90_put_att(ncid, nf90_global, 'fs_nsub', int(fs_nsub, 4))
    endif
    if (ierr /= nf90_noerr) then
      write(*,'(1x,A,A)') 'fs_global_atts: ', trim(nf90_strerror(ierr))
      error stop 'fs_global_atts: netCDF error'
    endif
  end subroutine fs_global_atts

  subroutine fs_scheme_coef()
    ! generalized-alpha written as a theta-step with memory (FS_SPEC 3.1)
    real(rp) :: am, af
    am = 0.5_rp * (3.0_rp - fs_rho_inf) / (1.0_rp + fs_rho_inf)
    af = 1.0_rp / (1.0_rp + fs_rho_inf)
    gam_a = 0.5_rp + am - af
    beta1 = gam_a / am
    ths = gam_a * af / am
    thp = ths + fs_theta - 0.5_rp
    thd = ths + fs_theta_div - 0.5_rp
  end subroutine fs_scheme_coef

  !======================================================================
  ! Set-up
  !======================================================================
  subroutine fs_init()
    integer(ip) :: is, ie, js, je, nx, ny
    integer :: m
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nx = decomp%nx; ny = decomp%ny
    allocate(unu(is:ie, js:je), unv(is:ie, js:je), usu(is:ie, js:je), usv(is:ie, js:je), &
             au(is:ie, js:je), av(is:ie, js:je), cu(is:ie, js:je), cv(is:ie, js:je), &
             ftu(is:ie, js:je), ftv(is:ie, js:je), fbu(is:ie, js:je), fbv(is:ie, js:je), &
             u1u(is:ie, js:je), u1v(is:ie, js:je), cbu(is:ie, js:je), cbv(is:ie, js:je), &
             cxu(is:ie, js:je), w1u(is:ie, js:je), w0u(is:ie, js:je), cxv(is:ie, js:je), w1v(is:ie, js:je), &
             w0v(is:ie, js:je), &
             wd(is:ie, js:je), ar(is:ie, js:je), rb(is:ie, js:je), dinv(is:ie, js:je), &
             qs(is:ie, js:je), act(is:ie, js:je), bb(is:ie, js:je), &
             rr(is:ie, js:je), zz(is:ie, js:je), pd(is:ie, js:je), qq(is:ie, js:je), wk2(is:ie, js:je), &
             du1o(is:ie, js:je), dv1o(is:ie, js:je), fdum(is:ie, js:je), &
             xru(is:ie, js:je), xrv(is:ie, js:je), cou(is:ie, js:je), cov(is:ie, js:je), fcr(is:ie, js:je))
    xru = 0; xrv = 0; cou = 0; cov = 0; fcr = 0
    cxu = 0; w1u = 0; w0u = 0; cxv = 0; w1v = 0; w0v = 0
    call fs_alloc_rates()      ! kept if a restart file supplied them
    if (fs_nsub > 1) then
      allocate(fsu(is:ie, js:je), fsv(is:ie, js:je), asu(is:ie, js:je), asv(is:ie, js:je))
      fsu = 0; fsv = 0; asu = 0; asv = 0
    endif
    unu = 0; unv = 0; usu = 0; usv = 0; au = 0; av = 0; cu = 0; cv = 0; ftu = 0; ftv = 0
    fbu = 0; fbv = 0; u1u = 0; u1v = 0; cbu = 0; cbv = 0; wd = 0; ar = 0; rb = 0; dinv = 0
    qs = 0; act = 0; bb = 0; rr = 0; zz = 0; pd = 0; qq = 0; wk2 = 0
    du1o = 0; dv1o = 0; fdum = 0
    allocate(rivu(is:ie, js:je), rivv(is:ie, js:je))
    rivu = 0; rivv = 0
    if (river_on) call rivers_fs_map(rivu, rivv)
    ncell = int(nx * ny)
    allocate(fs_part(ncell, FSR_MAXS), fs_bpart(FSR_NB, FSR_MAXS))
    fs_part = 0; fs_bpart = 0
    m = int(RG_NF * (max(nx, ny) + 2))
    call fs_alloc_bufs(m)
    if (o_ipc) then
      call ipc_init(int(m), [int(decomp%nbr_w), int(decomp%nbr_e), int(decomp%nbr_s), int(decomp%nbr_n)], &
           int(nx), int(ny), int(is), int(ie), int(js), int(je), &
           [decomp%edge_w, decomp%edge_e, decomp%edge_s, decomp%edge_n], work_stream)
      if (talk) write(*,'(1x,A,L1)') 'IMPLICIT_FS: fs_gpu_opt 256: rings by CUDA IPC: ', ipc_on
    endif
    if (mg_on) call mg_build_layout()
    ready = .true.
  end subroutine fs_init

  !======================================================================
  ! The step (FS_SPEC 3)
  !======================================================================
  subroutine fs_step()
    ! The implicit barotropic step; the caller has rotated kstp, knew.
    integer(ip) :: i, j, nx, ny, ks, kn, istrU, istrR, iendR, jstrV, jstrR, jendR, nmax, m
    integer(ip) :: k, l, nl, lsc(2), k0
    real(rp) :: b1, th, td, dtl, gl, ga, cdu, d, df, ainv, gam, mm, hf, cb, rf, cx, cff, dft, dfx, ce, ub, dfn
    real(rp) :: zc, xd, c, rn
    logical :: ew, ee, es, en, ow, oe, os, on, wf, ef, sf, nf, drag, first, rv, cmet, sub, k1, last1, pgr, zg
    real(rp) :: h1, h2, z1, z2, s1, s2, a1, a2, fac, w1, w0
    integer :: its
    real(rp) :: resid
    logical :: conv

    if (.not. ready) call fs_init()
    nx = decomp%nx; ny = decomp%ny
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    wf = ew .and. ow; ef = ee .and. oe; sf = es .and. os; nf = en .and. on   ! Flather faces
    istrU = merge(2, 1, ew); istrR = merge(0, 1, ew); iendR = merge(nx + 1, nx, ee)
    jstrV = merge(2, 1, es); jstrR = merge(0, 1, es); jendR = merge(ny + 1, ny, en)
    ! (sub-)step length dts = dt/fs_nsub (CPU b6325f06)
    dts = dt
    if (fs_nsub > 1) dts = dt / real(fs_nsub, rp)
    th = thp; td = thd; dtl = dts; gl = g; ga = gam_a
    drag = fs_drag_implicit
    rv = river_on
    cmet = fs_cor_metric
    k0 = kstp
    last1 = (fs_nsub == 1)
    pgr = fs_pg_rho
    zg = fs_flather_zghost
    call tw(0)
    call tq0()
    ! Flather weights of the open normal faces (the CPU's fs_flather_wts): cx = dts pm sqrt(g h) at the first
    ! interior rho point (the radiation condition of the boundary sea level), and with fs_flather_zghost
    ! w1 = (1 + 2 cx)/(2 (1 + cx)), w0 = 1/(2 (1 + cx)); u faces 1 (west) and nx+1 (east), v faces 1 (south)
    ! and ny+1 (north). Once per step (dts may change between runs), read by the setup, the Robin diagonal,
    ! the boundary flux, the boundary sea level and the normal boundary velocity
    nmax = max(nx, ny)
    CUF_DO1
    do m = 1, nmax
      if (m <= ny) then
        if (wf) then
          cx = dtl * pm(1, m) * sqrt(gl * h(1, m)); cxu(1, m) = cx
          w1u(1, m) = 0.5_rp * (1.0_rp + 2.0_rp * cx) / (1.0_rp + cx); w0u(1, m) = 0.5_rp / (1.0_rp + cx)
        endif
        if (ef) then
          cx = dtl * pm(nx, m) * sqrt(gl * h(nx, m)); cxu(nx + 1, m) = cx
          w1u(nx + 1, m) = 0.5_rp * (1.0_rp + 2.0_rp * cx) / (1.0_rp + cx); w0u(nx + 1, m) = 0.5_rp / (1.0_rp + cx)
        endif
      endif
      if (m <= nx) then
        if (sf) then
          cx = dtl * pn(m, 1) * sqrt(gl * h(m, 1)); cxv(m, 1) = cx
          w1v(m, 1) = 0.5_rp * (1.0_rp + 2.0_rp * cx) / (1.0_rp + cx); w0v(m, 1) = 0.5_rp / (1.0_rp + cx)
        endif
        if (nf) then
          cx = dtl * pn(m, ny) * sqrt(gl * h(m, ny)); cxv(m, ny + 1) = cx
          w1v(m, ny + 1) = 0.5_rp * (1.0_rp + 2.0_rp * cx) / (1.0_rp + cx); w0v(m, ny + 1) = 0.5_rp / (1.0_rp + cx)
        endif
      endif
    enddo

    ! ---- (A) DU_avg_bak bookkeeping (EXTRAP_BAR_FLUXES): U^n for the end of the step
    CUF_DO2
    do j = jstrR, jendR
      do i = istrR, iendR
        du1o(i, j) = DU_avg1(i, j); dv1o(i, j) = DV_avg1(i, j)
      enddo
    enddo

    ! ---- sub-steps k = 1..fs_nsub of length dts (CPU b6325f06), each from the source level ks to the
    ! level kn: ks = kstp (zeta^n) for k = 1, else the level of sub-step k-1; kn = knew for the last
    ! sub-step, else one of the two time levels other than kstp and knew (unused during the barotropic
    ! step; the levels alternate), so that no sub-step reads a level it writes. rufrc (with the 3-D
    ! pressure gradient of zeta^n) is held over the step; sub-steps k > 1 add the pressure gradient of
    ! zeta(ks) - zeta^n. fs_nsub = 1 is the single step.
    nl = 0
    do l = 1, 4
      if (l /= kstp .and. l /= knew .and. nl < 2) then
        nl = nl + 1; lsc(nl) = l
      endif
    enddo
    do k = 1, fs_nsub
    ks = kstp
    if (k > 1) ks = lsc(mod(k - 2, 2) + 1)
    kn = knew
    if (k < fs_nsub) kn = lsc(mod(k - 1, 2) + 1)
    sub = (k > 1); k1 = (k == 1)
    b1 = beta1
    if (ga_first) b1 = 1.0_rp                ! no rates yet: the first sub-step of the run without memory
    first = ga_first

    ! D^n of the (sub-)step (level ks)
    CUF_DO2
    do j = 0, ny + 1
      do i = 0, nx + 1
        wd(i, j) = h(i, j) + zeta(i, j, ks)
        ar(i, j) = rmask(i, j) / (pm(i, j) * pn(i, j))
      enddo
    enddo

    ! Coriolis estimate of the momentum memory: depth-integrated velocities [m2/s] of U^n, V^n incl. one
    ! halo row/column
    CUF_DO2
    do j = 0, ny + 1
      do i = 0, nx + 1
        if (i >= 1) cou(i, j) = 0.5_rp * (wd(i - 1, j) + wd(i, j)) * ubar(i, j, ks)
        if (j >= 1) cov(i, j) = 0.5_rp * (wd(i, j - 1) + wd(i, j)) * vbar(i, j, ks)
      enddo
    enddo
    ! the Coriolis factor per cell (cells istrU-1..nx x jstrV-1..ny), with the curvilinear
    ! metric terms of the stock rhs (fs_cor_metric; CPU 22c05bc1 fcr: fomn + 0.5 ((v(j) + v(j+1))
    ! dndx - (u(i) + u(i+1)) dmde), ubar, vbar(kstp); dndx = dmde = 0 on uniform grids)
    CUF_DO2
    do j = jstrV - 1, ny
      do i = istrU - 1, nx
        if (cmet) then
          fcr(i, j) = f(i, j) / (pm(i, j) * pn(i, j)) + 0.5_rp * ((vbar(i, j, ks) + vbar(i, j + 1, ks)) * dndx(i, j) &
               - (ubar(i, j, ks) + ubar(i + 1, j, ks)) * dmde(i, j))
        else
          fcr(i, j) = f(i, j) / (pm(i, j) * pn(i, j))
        endif
      enddo
    enddo

    ! ---- (B) per-face quantities; every face of the array written (0 where
    ! the CPU's whole-array zeroing leaves 0)
    CUF_DO2
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        au(i, j) = 0; cu(i, j) = 0; ftu(i, j) = 0; usu(i, j) = 0; cbu(i, j) = 0
        av(i, j) = 0; cv(i, j) = 0; ftv(i, j) = 0; usv(i, j) = 0; cbv(i, j) = 0
        fbu(i, j) = 0; fbv(i, j) = 0; u1u(i, j) = 0; u1v(i, j) = 0; xru(i, j) = 0; xrv(i, j) = 0
        ! U^n, V^n
        if (j >= jstrR .and. j <= jendR .and. i >= 1 .and. i <= nx + 1) &
             unu(i, j) = 0.5_rp * (wd(i - 1, j) + wd(i, j)) * dn_u(i, j) * ubar(i, j, ks)
        if (j >= 1 .and. j <= ny + 1 .and. i >= istrR .and. i <= iendR) &
             unv(i, j) = 0.5_rp * (wd(i, j - 1) + wd(i, j)) * dm_v(i, j) * vbar(i, j, ks)
        ! owned u faces
        if (j >= 1 .and. j <= ny .and. i >= istrU .and. i <= nx) then
          df = 0.5_rp * (wd(i - 1, j) + wd(i, j))
          ainv = 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j))
          gam = 0.0_rp
          if (drag) gam = dtl * 0.5_rp * (r_D(i - 1, j) + r_D(i, j)) / df
          mm = umask(i, j)
          au(i, j) = th * dtl * gl * dn_u(i, j) * dn_u(i, j) * df * ainv * mm / (1.0_rp + gam)
          if (pgr) then
            ! fs_pg_rho: the density factor of the face coefficient (CPU fs_rho_fac). The depth integral of
            ! the 3-D pressure gradient responds to the sea level as step2d's rubar, not as -g D grad zeta;
            ! its derivative with respect to the difference of zeta across the face, divided by g D_face:
            !   1 + [ (h1+h2) (s1+s2)/2 + s1 z1 + s2 z2 + (h1-h2) ((s1-a1 - s2+a2)/2 + (a1-a2)/3) ] / (h1+h2+z1+z2)
            ! (cells 1, 2 on either side, s = rhoS, a = rhoA); the part multiplying the mean of the
            ! increment across the face stays explicit
            h1 = h(i - 1, j); h2 = h(i, j); z1 = zeta(i - 1, j, ks); z2 = zeta(i, j, ks)
            s1 = rhoS(i - 1, j); s2 = rhoS(i, j); a1 = rhoA(i - 1, j); a2 = rhoA(i, j)
            fac = 1.0_rp + (0.5_rp * (h1 + h2) * (s1 + s2) + s1 * z1 + s2 * z2 &
                 + (h1 - h2) * (0.5_rp * ((s1 - a1) - (s2 - a2)) + (a1 - a2) / 3.0_rp)) / (h1 + h2 + z1 + z2)
            au(i, j) = au(i, j) * fac
          endif
          xd = dtl * dn_u(i, j) * ainv * rufrc(i, j)
          hf = dtl * dn_u(i, j) * ainv * 0.25_rp * (fcr(i, j) * (cov(i, j) + cov(i, j + 1)) &
               + fcr(i - 1, j) * (cov(i - 1, j) + cov(i - 1, j + 1)))
          usu(i, j) = mm * (unu(i, j) + (hf + b1 * (xd - hf) + (1.0_rp - b1) * dtl * udu(i, j)) / (1.0_rp + gam))
          xru(i, j) = mm * hf / (1.0_rp + gam)
          ftu(i, j) = mm * (b1 * unu(i, j) + (1.0_rp - b1) * gru(i, j) + td * (usu(i, j) - unu(i, j)))
          cu(i, j) = td * dtl * au(i, j)
          if (sub) then
            ! sub-steps after the first: rufrc holds the pressure gradient of zeta^n, so the explicit
            ! pressure gradient of zeta(ks) - zeta^n is added in the discrete form of the implicit
            ! increment's (au/th per unit difference) with the weight b1 of the explicit rhs
            c = b1 * (au(i, j) / th) * ((zeta(i - 1, j, ks) - zeta(i - 1, j, k0)) - (zeta(i, j, ks) - zeta(i, j, k0)))
            usu(i, j) = usu(i, j) + c
            ftu(i, j) = ftu(i, j) + td * c
          endif
        endif
        ! owned v faces
        if (j >= jstrV .and. j <= ny .and. i >= 1 .and. i <= nx) then
          df = 0.5_rp * (wd(i, j - 1) + wd(i, j))
          ainv = 0.25_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1))
          gam = 0.0_rp
          if (drag) gam = dtl * 0.5_rp * (r_D(i, j - 1) + r_D(i, j)) / df
          mm = vmask(i, j)
          av(i, j) = th * dtl * gl * dm_v(i, j) * dm_v(i, j) * df * ainv * mm / (1.0_rp + gam)
          if (pgr) then
            h1 = h(i, j - 1); h2 = h(i, j); z1 = zeta(i, j - 1, ks); z2 = zeta(i, j, ks)
            s1 = rhoS(i, j - 1); s2 = rhoS(i, j); a1 = rhoA(i, j - 1); a2 = rhoA(i, j)
            fac = 1.0_rp + (0.5_rp * (h1 + h2) * (s1 + s2) + s1 * z1 + s2 * z2 &
                 + (h1 - h2) * (0.5_rp * ((s1 - a1) - (s2 - a2)) + (a1 - a2) / 3.0_rp)) / (h1 + h2 + z1 + z2)
            av(i, j) = av(i, j) * fac
          endif
          xd = dtl * dm_v(i, j) * ainv * rvfrc(i, j)
          hf = -dtl * dm_v(i, j) * ainv * 0.25_rp * (fcr(i, j) * (cou(i, j) + cou(i + 1, j)) &
               + fcr(i, j - 1) * (cou(i, j - 1) + cou(i + 1, j - 1)))
          usv(i, j) = mm * (unv(i, j) + (hf + b1 * (xd - hf) + (1.0_rp - b1) * dtl * udv(i, j)) / (1.0_rp + gam))
          xrv(i, j) = mm * hf / (1.0_rp + gam)
          ftv(i, j) = mm * (b1 * unv(i, j) + (1.0_rp - b1) * grv(i, j) + td * (usv(i, j) - unv(i, j)))
          cv(i, j) = td * dtl * av(i, j)
          if (sub) then
            c = b1 * (av(i, j) / th) * ((zeta(i, j - 1, ks) - zeta(i, j - 1, k0)) - (zeta(i, j, ks) - zeta(i, j, k0)))
            usv(i, j) = usv(i, j) + c
            ftv(i, j) = ftv(i, j) + td * c
          endif
        endif
        ! open normal faces: Flather on the th_d-level flux
        if (wf .and. i == 1 .and. j >= 1 .and. j <= ny) then
          mm = umask(1, j)
          df = 0.5_rp * (wd(0, j) + wd(1, j))
          hf = 0.5_rp * (h(0, j) + h(1, j))
          cb = mm * dn_u(1, j) * df * sqrt(gl / hf)
          ftu(1, j) = mm * (b1 * unu(1, j) + (1.0_rp - b1) * gru(1, j) + td * (df * dn_u(1, j) * ubar_west(j) &
               - cb * (zeta(1, j, ks) - zeta_west(j)) - unu(1, j)))
          cbu(1, j) = cb
          if (zg) then
            ! fs_flather_zghost: the Flather sea level (zeta_1^n+1 + zeta_b^n+1)/2 with zeta_b^n+1 from the
            ! radiation condition = w1 zeta_1^n+1 + w0 zeta_b^n: Robin coefficient cb w1, w0 zeta_b explicit
            w1 = w1u(1, j); w0 = w0u(1, j)
            ftu(1, j) = mm * (b1 * unu(1, j) + (1.0_rp - b1) * gru(1, j) + td * (df * dn_u(1, j) * ubar_west(j) &
                 - cb * (w1 * zeta(1, j, ks) + w0 * zeta(0, j, ks) - zeta_west(j)) - unu(1, j)))
          endif
        endif
        if (ef .and. i == nx + 1 .and. j >= 1 .and. j <= ny) then
          mm = umask(nx + 1, j)
          df = 0.5_rp * (wd(nx, j) + wd(nx + 1, j))
          hf = 0.5_rp * (h(nx, j) + h(nx + 1, j))
          cb = mm * dn_u(nx + 1, j) * df * sqrt(gl / hf)
          ftu(nx + 1, j) = mm * (b1 * unu(nx + 1, j) + (1.0_rp - b1) * gru(nx + 1, j) + td * (df * dn_u(nx + 1, j) &
               * ubar_east(j) + cb * (zeta(nx, j, ks) - zeta_east(j)) - unu(nx + 1, j)))
          cbu(nx + 1, j) = cb
          if (zg) then
            w1 = w1u(nx + 1, j); w0 = w0u(nx + 1, j)
            ftu(nx + 1, j) = mm * (b1 * unu(nx + 1, j) + (1.0_rp - b1) * gru(nx + 1, j) + td * (df * dn_u(nx + 1, j) &
                 * ubar_east(j) + cb * (w1 * zeta(nx, j, ks) + w0 * zeta(nx + 1, j, ks) - zeta_east(j)) - unu(nx + 1, j)))
          endif
        endif
        if (sf .and. j == 1 .and. i >= 1 .and. i <= nx) then
          mm = vmask(i, 1)
          df = 0.5_rp * (wd(i, 0) + wd(i, 1))
          hf = 0.5_rp * (h(i, 0) + h(i, 1))
          cb = mm * dm_v(i, 1) * df * sqrt(gl / hf)
          ftv(i, 1) = mm * (b1 * unv(i, 1) + (1.0_rp - b1) * grv(i, 1) + td * (df * dm_v(i, 1) * vbar_south(i) &
               - cb * (zeta(i, 1, ks) - zeta_south(i)) - unv(i, 1)))
          cbv(i, 1) = cb
          if (zg) then
            w1 = w1v(i, 1); w0 = w0v(i, 1)
            ftv(i, 1) = mm * (b1 * unv(i, 1) + (1.0_rp - b1) * grv(i, 1) + td * (df * dm_v(i, 1) * vbar_south(i) &
                 - cb * (w1 * zeta(i, 1, ks) + w0 * zeta(i, 0, ks) - zeta_south(i)) - unv(i, 1)))
          endif
        endif
        if (nf .and. j == ny + 1 .and. i >= 1 .and. i <= nx) then
          mm = vmask(i, ny + 1)
          df = 0.5_rp * (wd(i, ny) + wd(i, ny + 1))
          hf = 0.5_rp * (h(i, ny) + h(i, ny + 1))
          cb = mm * dm_v(i, ny + 1) * df * sqrt(gl / hf)
          ftv(i, ny + 1) = mm * (b1 * unv(i, ny + 1) + (1.0_rp - b1) * grv(i, ny + 1) + td * (df * dm_v(i, ny + 1) &
               * vbar_north(i) + cb * (zeta(i, ny, ks) - zeta_north(i)) - unv(i, ny + 1)))
          cbv(i, ny + 1) = cb
          if (zg) then
            w1 = w1v(i, ny + 1); w0 = w0v(i, ny + 1)
            ftv(i, ny + 1) = mm * (b1 * unv(i, ny + 1) + (1.0_rp - b1) * grv(i, ny + 1) + td * (df * dm_v(i, ny + 1) &
                 * vbar_north(i) + cb * (w1 * zeta(i, ny, ks) + w0 * zeta(i, ny + 1, ks) - zeta_north(i)) - unv(i, ny + 1)))
          endif
        endif
      enddo
    enddo

    ! river faces: Dirichlet transport (owned faces, as step2d)
    if (rv) then
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          if (i >= istrU .and. rivu(i, j) > 0) then
            rf = rf_flx(rivu(i, j))
            au(i, j) = 0; cu(i, j) = 0; xru(i, j) = 0
            usu(i, j) = rf
            ftu(i, j) = rf                 ! the step's discharge (= DU_avg2, as the split delivers it)
          endif
          if (j >= jstrV .and. rivv(i, j) > 0) then
            rf = rf_flx(rivv(i, j))
            av(i, j) = 0; cv(i, j) = 0; xrv(i, j) = 0
            usv(i, j) = rf
            ftv(i, j) = rf
          endif
        enddo
      enddo
    endif
    ! ---- volume sources and the Robin diagonal (cells)
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        rb(i, j) = 0.0_rp
        ! (fs_flather_zghost: + dt th_d cb (w1 - 1) after each edge's term, the CPU's order)
        if (wf .and. i == 1) then
          rb(i, j) = rb(i, j) + dtl * td * cbu(1, j)
          if (zg) then
            w1 = w1u(1, j)
            rb(i, j) = rb(i, j) + dtl * td * cbu(1, j) * (w1 - 1.0_rp)
          endif
        endif
        if (ef .and. i == nx) then
          rb(i, j) = rb(i, j) + dtl * td * cbu(nx + 1, j)
          if (zg) then
            w1 = w1u(nx + 1, j)
            rb(i, j) = rb(i, j) + dtl * td * cbu(nx + 1, j) * (w1 - 1.0_rp)
          endif
        endif
        if (sf .and. j == 1) then
          rb(i, j) = rb(i, j) + dtl * td * cbv(i, 1)
          if (zg) then
            w1 = w1v(i, 1)
            rb(i, j) = rb(i, j) + dtl * td * cbv(i, 1) * (w1 - 1.0_rp)
          endif
        endif
        if (nf .and. j == ny) then
          rb(i, j) = rb(i, j) + dtl * td * cbv(i, ny + 1)
          if (zg) then
            w1 = w1v(i, ny + 1)
            rb(i, j) = rb(i, j) + dtl * td * cbv(i, ny + 1) * (w1 - 1.0_rp)
          endif
        endif
        qs(i, j) = swflx(i, j)
      enddo
    enddo
    if (cdr_vol_on) call cdr_fs_src(qs)     ! Qbar = Q (the volume the 3-D step adds)
    call tq(T_SETUP)

    ! ---- (C) E1: the face coefficients and explicit fluxes
    call ring(4, cu, cv, ftu, ftv, fdum, fdum, .true.)
    call tq(T_EXCO)

    ! ---- (D) operator diagonal, rhs
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        d = ar(i, j) + cu(i, j) + cu(i + 1, j) + cv(i, j) + cv(i, j + 1) + rb(i, j)
        act(i, j) = 0.0_rp
        if (rmask(i, j) > 0.0_rp .and. d > 0.0_rp) act(i, j) = 1.0_rp
        dinv(i, j) = 0.0_rp
        if (act(i, j) > 0.0_rp) dinv(i, j) = 1.0_rp / d
        bb(i, j) = act(i, j) * (-dtl * (ftu(i + 1, j) - ftu(i, j) + ftv(i, j + 1) - ftv(i, j)) &
             + dtl * ar(i, j) * qs(i, j))
      enddo
    enddo
    if (fs_warm == 0) call zero2(xx)
    call tq(T_SETUP)
    if (mg_on) then
      if (mg_rpend) call mg_replay()       ! restart within a fs_mg_lag period (FS_SPEC 9b-C)
      if (mod(n_solve + mg_ph0, fs_mg_lag) == 0) then
        call mg_setup()
        mg_built = mg_built + 1
        if (fs_mg_lag > 1) call mg_keep_fine()
      endif
    endif

    ! ---- (E) solve A delta = b
    call fs_pcg(its, resid, conv)
    n_solve = n_solve + 1; n_its = n_its + its; its_last = its; res_last = resid
    if (.not. conv) n_unconv = n_unconv + 1
    if (.not. conv) call fs_unconv_stop(k, its, resid)
    if (fs_nsub > 1) then                   ! the step's FS_SOLVE statistics
      if (k1) then
        its_sum = 0; its_max = 0; res_max = 0.0_rp; dz_max = 0.0_rp
      endif
      its_sum = its_sum + its; its_max = max(its_max, its); res_max = max(res_max, resid)
      if (fs_verbose >= 1) dz_max = max(dz_max, xx_absmax())
    endif

    ! ---- (F) E2, momentum and fluxes, generalized-alpha rates
    call ring(1, xx, fdum, fdum, fdum, fdum, fdum, .true.)
    call tq(T_EXFIN)
    CUF_DO2
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (i >= istrU .and. i <= nx .and. j <= ny) then
          d = xx(i - 1, j) - xx(i, j)
          u1u(i, j) = usu(i, j) + au(i, j) * d
          fbu(i, j) = ftu(i, j) + td * au(i, j) * d
        endif
        if (i <= nx .and. j >= jstrV .and. j <= ny) then
          d = xx(i, j - 1) - xx(i, j)
          u1v(i, j) = usv(i, j) + av(i, j) * d
          fbv(i, j) = ftv(i, j) + td * av(i, j) * d
        endif
        if (ew .and. i == 1 .and. j <= ny) then
          fbu(1, j) = ftu(1, j)
          if (ow) then
            if (zg) then                     ! fs_flather_zghost: the Robin coefficient cb w1
              w1 = w1u(1, j)
              fbu(1, j) = fbu(1, j) - td * (cbu(1, j) * w1) * xx(1, j)
            else
              fbu(1, j) = fbu(1, j) - td * cbu(1, j) * xx(1, j)
            endif
          endif
        endif
        if (ee .and. i == nx + 1 .and. j <= ny) then
          fbu(nx + 1, j) = ftu(nx + 1, j)
          if (oe) then
            if (zg) then
              w1 = w1u(nx + 1, j)
              fbu(nx + 1, j) = fbu(nx + 1, j) + td * (cbu(nx + 1, j) * w1) * xx(nx, j)
            else
              fbu(nx + 1, j) = fbu(nx + 1, j) + td * cbu(nx + 1, j) * xx(nx, j)
            endif
          endif
        endif
        if (es .and. j == 1 .and. i <= nx) then
          fbv(i, 1) = ftv(i, 1)
          if (os) then
            if (zg) then
              w1 = w1v(i, 1)
              fbv(i, 1) = fbv(i, 1) - td * (cbv(i, 1) * w1) * xx(i, 1)
            else
              fbv(i, 1) = fbv(i, 1) - td * cbv(i, 1) * xx(i, 1)
            endif
          endif
        endif
        if (en .and. j == ny + 1 .and. i <= nx) then
          fbv(i, ny + 1) = ftv(i, ny + 1)
          if (on) then
            if (zg) then
              w1 = w1v(i, ny + 1)
              fbv(i, ny + 1) = fbv(i, ny + 1) + td * (cbv(i, ny + 1) * w1) * xx(i, ny)
            else
              fbv(i, ny + 1) = fbv(i, ny + 1) + td * cbv(i, ny + 1) * xx(i, ny)
            endif
          endif
        endif
        ! rates (u faces 1..nx+1 x 1..ny, v faces 1..nx x 1..ny+1, cells)
        if (first) then
          if (j <= ny) then
            udu(i, j) = (u1u(i, j) - unu(i, j) - xru(i, j)) / dtl; gru(i, j) = fbu(i, j)
          endif
          if (i <= nx) then
            udv(i, j) = (u1v(i, j) - unv(i, j) - xrv(i, j)) / dtl; grv(i, j) = fbv(i, j)
          endif
          if (i <= nx .and. j <= ny) qd(i, j) = qs(i, j)
        else
          if (j <= ny) then
            udu(i, j) = ((u1u(i, j) - unu(i, j) - xru(i, j)) / dtl - (1.0_rp - ga) * udu(i, j)) / ga
            gru(i, j) = (fbu(i, j) - (1.0_rp - ga) * gru(i, j)) / ga
          endif
          if (i <= nx) then
            udv(i, j) = ((u1v(i, j) - unv(i, j) - xrv(i, j)) / dtl - (1.0_rp - ga) * udv(i, j)) / ga
            grv(i, j) = (fbv(i, j) - (1.0_rp - ga) * grv(i, j)) / ga
          endif
          if (i <= nx .and. j <= ny) qd(i, j) = (qs(i, j) - (1.0_rp - ga) * qd(i, j)) / ga
        endif
      enddo
    enddo
    ga_first = .false.
    call tq(T_FIN)
    call ring(2, fbu, fbv, fdum, fdum, fdum, fdum, .true.)         ! E3
    call tq(T_EXFIN)

    ! ---- (G) zeta^n+1 from the exact divergence; boundary rho points
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        zeta(i, j, kn) = rmask(i, j) * (zeta(i, j, ks) - dtl * pm(i, j) * pn(i, j) * (fbu(i + 1, j) - fbu(i, j) &
             + fbv(i, j + 1) - fbv(i, j)) + dtl * qs(i, j))
      enddo
    enddo
    nmax = max(nx, ny)
    CUF_DO1
    do m = 1, nmax
      if (m <= ny) then
        if (ew) then
          if (ow) then
            cx = cxu(1, m)
            zeta(0, m, kn) = rmask(0, m) * (zeta(0, m, ks) + cx * zeta(1, m, kn)) / (1.0_rp + cx)
          else
            zeta(0, m, kn) = rmask(0, m) * zeta(1, m, kn)
          endif
        endif
        if (ee) then
          if (oe) then
            cx = cxu(nx + 1, m)
            zeta(nx + 1, m, kn) = rmask(nx + 1, m) * (zeta(nx + 1, m, ks) + cx * zeta(nx, m, kn)) / (1.0_rp + cx)
          else
            zeta(nx + 1, m, kn) = rmask(nx + 1, m) * zeta(nx, m, kn)
          endif
        endif
      endif
      if (m <= nx) then
        if (es) then
          if (os) then
            cx = cxv(m, 1)
            zeta(m, 0, kn) = rmask(m, 0) * (zeta(m, 0, ks) + cx * zeta(m, 1, kn)) / (1.0_rp + cx)
          else
            zeta(m, 0, kn) = rmask(m, 0) * zeta(m, 1, kn)
          endif
        endif
        if (en) then
          if (on) then
            cx = cxv(m, ny + 1)
            zeta(m, ny + 1, kn) = rmask(m, ny + 1) * (zeta(m, ny + 1, ks) + cx * zeta(m, ny, kn)) / (1.0_rp + cx)
          else
            zeta(m, ny + 1, kn) = rmask(m, ny + 1) * zeta(m, ny, kn)
          endif
        endif
      endif
    enddo
    if ((ew .or. ee) .and. (es .or. en)) then
      CUF_SERIAL
      do m = 1, 1
        if (ew .and. es) zeta(0, 0, kn) = 0.5_rp * (zeta(0, 1, kn) + zeta(1, 0, kn))
        if (ee .and. es) zeta(nx + 1, 0, kn) = 0.5_rp * (zeta(nx + 1, 1, kn) + zeta(nx, 0, kn))
        if (ew .and. en) zeta(0, ny + 1, kn) = 0.5_rp * (zeta(0, ny, kn) + zeta(1, ny + 1, kn))
        if (ee .and. en) zeta(nx + 1, ny + 1, kn) = 0.5_rp * (zeta(nx + 1, ny, kn) + zeta(nx, ny + 1, kn))
      enddo
    endif
    call tq(T_FIN)
    call ring_slot(zeta, kn)                                       ! E4a (one cell)
    call tq(T_EXFIN)

    ! ---- (H) ubar, vbar(knew): owned faces, normal and tangential boundary faces
    CUF_DO2
    do j = 0, ny + 1
      do i = 0, nx + 1
        wd(i, j) = h(i, j) + zeta(i, j, kn)
      enddo
    enddo
    CUF_DO2
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (i >= istrU .and. i <= nx .and. j <= ny) &
             ubar(i, j, kn) = 2.0_rp * u1u(i, j) / ((wd(i, j) + wd(i - 1, j)) * dn_u(i, j))
        if (i <= nx .and. j >= jstrV .and. j <= ny) &
             vbar(i, j, kn) = 2.0_rp * u1v(i, j) / ((wd(i, j) + wd(i, j - 1)) * dm_v(i, j))
        ! closed physical edges (no OBC, not periodic): no normal flow, every step (CPU 43949d66
        ! fs_normal_bc, as u2dbc/v2dbc's closed branch: west/east j = 1..ny, south/north i = 1..nx)
        if (ew .and. .not. ow .and. i == 1 .and. j <= ny) ubar(1, j, kn) = 0.0_rp
        if (ee .and. .not. oe .and. i == nx + 1 .and. j <= ny) ubar(nx + 1, j, kn) = 0.0_rp
        if (es .and. .not. os .and. j == 1 .and. i <= nx) vbar(i, 1, kn) = 0.0_rp
        if (en .and. .not. on .and. j == ny + 1 .and. i <= nx) vbar(i, ny + 1, kn) = 0.0_rp
        if (wf .and. i == 1 .and. j <= ny) then
          mm = umask(1, j)
          dfn = 0.5_rp * (h(0, j) + zeta(0, j, ks) + h(1, j) + zeta(1, j, ks))
          if (zg) then                       ! fs_flather_zghost: zeta_1^n+1 -> w1 zeta_1^n+1 + w0 zeta_b(ks)
            w1 = w1u(1, j); w0 = w0u(1, j)
            ub = mm * (dfn * dn_u(1, j) * ubar_west(j) - (cbu(1, j) * w1) * (zeta(1, j, kn) &
                 + (w0 * zeta(0, j, ks) - zeta_west(j)) / w1))
          else
          ub = mm * (dfn * dn_u(1, j) * ubar_west(j) - cbu(1, j) * (zeta(1, j, kn) - zeta_west(j)))
          endif
          ubar(1, j, kn) = 2.0_rp * ub / ((wd(0, j) + wd(1, j)) * dn_u(1, j))
        endif
        if (ef .and. i == nx + 1 .and. j <= ny) then
          mm = umask(nx + 1, j)
          dfn = 0.5_rp * (h(nx, j) + zeta(nx, j, ks) + h(nx + 1, j) + zeta(nx + 1, j, ks))
          if (zg) then
            w1 = w1u(nx + 1, j); w0 = w0u(nx + 1, j)
            ub = mm * (dfn * dn_u(nx + 1, j) * ubar_east(j) + (cbu(nx + 1, j) * w1) * (zeta(nx, j, kn) &
                 + (w0 * zeta(nx + 1, j, ks) - zeta_east(j)) / w1))
          else
          ub = mm * (dfn * dn_u(nx + 1, j) * ubar_east(j) + cbu(nx + 1, j) * (zeta(nx, j, kn) - zeta_east(j)))
          endif
          ubar(nx + 1, j, kn) = 2.0_rp * ub / ((wd(nx, j) + wd(nx + 1, j)) * dn_u(nx + 1, j))
        endif
        if (sf .and. j == 1 .and. i <= nx) then
          mm = vmask(i, 1)
          dfn = 0.5_rp * (h(i, 0) + zeta(i, 0, ks) + h(i, 1) + zeta(i, 1, ks))
          if (zg) then
            w1 = w1v(i, 1); w0 = w0v(i, 1)
            ub = mm * (dfn * dm_v(i, 1) * vbar_south(i) - (cbv(i, 1) * w1) * (zeta(i, 1, kn) &
                 + (w0 * zeta(i, 0, ks) - zeta_south(i)) / w1))
          else
          ub = mm * (dfn * dm_v(i, 1) * vbar_south(i) - cbv(i, 1) * (zeta(i, 1, kn) - zeta_south(i)))
          endif
          vbar(i, 1, kn) = 2.0_rp * ub / ((wd(i, 0) + wd(i, 1)) * dm_v(i, 1))
        endif
        if (nf .and. j == ny + 1 .and. i <= nx) then
          mm = vmask(i, ny + 1)
          dfn = 0.5_rp * (h(i, ny) + zeta(i, ny, ks) + h(i, ny + 1) + zeta(i, ny + 1, ks))
          if (zg) then
            w1 = w1v(i, ny + 1); w0 = w0v(i, ny + 1)
            ub = mm * (dfn * dm_v(i, ny + 1) * vbar_north(i) + (cbv(i, ny + 1) * w1) * (zeta(i, ny, kn) &
                 + (w0 * zeta(i, ny + 1, ks) - zeta_north(i)) / w1))
          else
          ub = mm * (dfn * dm_v(i, ny + 1) * vbar_north(i) + cbv(i, ny + 1) * (zeta(i, ny, kn) - zeta_north(i)))
          endif
          vbar(i, ny + 1, kn) = 2.0_rp * ub / ((wd(i, ny) + wd(i, ny + 1)) * dm_v(i, ny + 1))
        endif
      enddo
    enddo
    call fs_tang_bc(ks, kn, dtl)
    call tq(T_BCUV)

    ! ---- (I) hand-over: DU_avg1 = U^n+1, DU_avg2 = Fbar, Zt_avg1, DU_avg_bak
    CUF_DO2
    do j = 0, ny + 1
      do i = 0, nx + 1
        ! u faces
        if (i >= istrU .and. i <= nx .and. j >= 1 .and. j <= ny) then
          DU_avg1(i, j) = u1u(i, j); DU_avg2(i, j) = fbu(i, j)
        endif
        if (ew .and. i == 1 .and. j >= jstrR .and. j <= jendR) then
          DU_avg1(1, j) = 0.5_rp * (wd(1, j) + wd(0, j)) * ubar(1, j, kn) * dn_u(1, j)
          if (j >= 1 .and. j <= ny) then
            DU_avg2(1, j) = fbu(1, j)
          else
            DU_avg2(1, j) = td * DU_avg1(1, j) + (1.0_rp - td) * unu(1, j)
          endif
        endif
        if (ee .and. i == nx + 1 .and. j >= jstrR .and. j <= jendR) then
          DU_avg1(nx + 1, j) = 0.5_rp * (wd(nx + 1, j) + wd(nx, j)) * ubar(nx + 1, j, kn) * dn_u(nx + 1, j)
          if (j >= 1 .and. j <= ny) then
            DU_avg2(nx + 1, j) = fbu(nx + 1, j)
          else
            DU_avg2(nx + 1, j) = td * DU_avg1(nx + 1, j) + (1.0_rp - td) * unu(nx + 1, j)
          endif
        endif
        if (es .and. j == 0 .and. i >= istrU .and. i <= nx) then
          DU_avg1(i, 0) = 0.5_rp * (wd(i, 0) + wd(i - 1, 0)) * ubar(i, 0, kn) * dn_u(i, 0)
          DU_avg2(i, 0) = td * DU_avg1(i, 0) + (1.0_rp - td) * unu(i, 0)
        endif
        if (en .and. j == ny + 1 .and. i >= istrU .and. i <= nx) then
          DU_avg1(i, ny + 1) = 0.5_rp * (wd(i, ny + 1) + wd(i - 1, ny + 1)) * ubar(i, ny + 1, kn) * dn_u(i, ny + 1)
          DU_avg2(i, ny + 1) = td * DU_avg1(i, ny + 1) + (1.0_rp - td) * unu(i, ny + 1)
        endif
        ! v faces
        if (i >= 1 .and. i <= nx .and. j >= jstrV .and. j <= ny) then
          DV_avg1(i, j) = u1v(i, j); DV_avg2(i, j) = fbv(i, j)
        endif
        if (ew .and. i == 0 .and. j >= jstrV .and. j <= ny) then
          DV_avg1(0, j) = 0.5_rp * (wd(0, j) + wd(0, j - 1)) * vbar(0, j, kn) * dm_v(0, j)
          DV_avg2(0, j) = td * DV_avg1(0, j) + (1.0_rp - td) * unv(0, j)
        endif
        if (ee .and. i == nx + 1 .and. j >= jstrV .and. j <= ny) then
          DV_avg1(nx + 1, j) = 0.5_rp * (wd(nx + 1, j) + wd(nx + 1, j - 1)) * vbar(nx + 1, j, kn) * dm_v(nx + 1, j)
          DV_avg2(nx + 1, j) = td * DV_avg1(nx + 1, j) + (1.0_rp - td) * unv(nx + 1, j)
        endif
        if (es .and. j == 1 .and. i >= istrR .and. i <= iendR) then
          DV_avg1(i, 1) = 0.5_rp * (wd(i, 1) + wd(i, 0)) * vbar(i, 1, kn) * dm_v(i, 1)
          if (i >= 1 .and. i <= nx) then
            DV_avg2(i, 1) = fbv(i, 1)
          else
            DV_avg2(i, 1) = td * DV_avg1(i, 1) + (1.0_rp - td) * unv(i, 1)
          endif
        endif
        if (en .and. j == ny + 1 .and. i >= istrR .and. i <= iendR) then
          DV_avg1(i, ny + 1) = 0.5_rp * (wd(i, ny + 1) + wd(i, ny)) * vbar(i, ny + 1, kn) * dm_v(i, ny + 1)
          if (i >= 1 .and. i <= nx) then
            DV_avg2(i, ny + 1) = fbv(i, ny + 1)
          else
            DV_avg2(i, ny + 1) = td * DV_avg1(i, ny + 1) + (1.0_rp - td) * unv(i, ny + 1)
          endif
        endif
        ! rho points and the predictor's extrapolation (fs_nsub > 1: after the mean of the sub-steps)
        if (last1 .and. i >= istrR .and. i <= iendR .and. j >= jstrR .and. j <= jendR) then
          zt_avg1(i, j) = zeta(i, j, kn)
          DU_avg_bak(i, j) = ((1.5_rp - 3.63_rp) * DU_avg1(i, j) - 0.5_rp * du1o(i, j) + 4.47_rp * DU_avg2(i, j)) &
               / 2.05_rp
          DV_avg_bak(i, j) = ((1.5_rp - 3.63_rp) * DV_avg1(i, j) - 0.5_rp * dv1o(i, j) + 4.47_rp * DV_avg2(i, j)) &
               / 2.05_rp
        endif
      enddo
    enddo
    if (rv) then
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          if (i >= istrU .and. rivu(i, j) > 0) &
               ubar(i, j, kn) = 2.0_rp * u1u(i, j) / (dn_u(i, j) * (wd(i - 1, j) + wd(i, j)))
          if (j >= jstrV .and. rivv(i, j) > 0) &
               vbar(i, j, kn) = 2.0_rp * u1v(i, j) / (dm_v(i, j) * (wd(i, j - 1) + wd(i, j)))
        enddo
      enddo
    endif
    call tq(T_FIN)
    if (fs_nsub > 1) then
      ! sums over the sub-steps of Fbar (fsu, fsv) and of DU_avg2 (asu, asv; every face it is set on)
      CUF_DO2
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          if (k1) then
            fsu(i, j) = fbu(i, j); fsv(i, j) = fbv(i, j)
          else
            fsu(i, j) = fsu(i, j) + fbu(i, j); fsv(i, j) = fsv(i, j) + fbv(i, j)
          endif
          if (i >= istrR .and. i <= iendR .and. j >= jstrR .and. j <= jendR) then
            if (k1) then
              asu(i, j) = DU_avg2(i, j); asv(i, j) = DV_avg2(i, j)
            else
              asu(i, j) = asu(i, j) + DU_avg2(i, j); asv(i, j) = asv(i, j) + DV_avg2(i, j)
            endif
          endif
        enddo
      enddo
      call tq(T_FIN)
    endif
    if (k < fs_nsub) then
      ! E4b of the sub-step: zeta, ubar, vbar(kn) with their two-cell halos for the next sub-step
      fast%kstp = ks; fast%knew = kn
      call halo_fill_zuv_n(zeta, ubar, vbar)
      n_ring = n_ring + 2
      call tq(T_EXFIN)
    endif
    enddo                                   ! sub-steps

    if (fs_nsub > 1) then
      ! DU_avg2 = the mean flux of the sub-steps: zeta^n+1 = zeta^n - dt pm pn Div(DU_avg2) + dt Q (to
      ! rounding); fbu, fbv likewise for the step's volume check; then Zt_avg1 and DU_avg_bak
      rn = 1.0_rp / real(fs_nsub, rp)
      CUF_DO2
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          fbu(i, j) = rn * fsu(i, j); fbv(i, j) = rn * fsv(i, j)
          if (i >= istrR .and. i <= iendR .and. j >= jstrR .and. j <= jendR) then
            DU_avg2(i, j) = rn * asu(i, j); DV_avg2(i, j) = rn * asv(i, j)
            zt_avg1(i, j) = zeta(i, j, kn)
            DU_avg_bak(i, j) = ((1.5_rp - 3.63_rp) * DU_avg1(i, j) - 0.5_rp * du1o(i, j) + 4.47_rp * DU_avg2(i, j)) &
                 / 2.05_rp
            DV_avg_bak(i, j) = ((1.5_rp - 3.63_rp) * DV_avg1(i, j) - 0.5_rp * dv1o(i, j) + 4.47_rp * DV_avg2(i, j)) &
                 / 2.05_rp
          endif
        enddo
      enddo
      call tq(T_FIN)
    endif
    if (fs_verbose >= 2) call fs_volume_check()                     ! the whole step
    n_step = n_step + 1
    ! the seam faces of the hand-over (nx+1, ny+1 toward a neighbour; the
    ! split loop forms them redundantly, read by set_HUV1 before its own
    ! exchanges)
    call ring(6, DU_avg1, DU_avg2, DU_avg_bak, DV_avg1, DV_avg2, DV_avg_bak, .true.)
    ! E4b: zeta, ubar, vbar(knew) with their two-cell halos (as the last fast step)
    fast%kstp = kstp; fast%knew = knew
    call halo_fill_zuv_n(zeta, ubar, vbar)
    n_ring = n_ring + 2
    call tq(T_EXFIN)
    call set_depth()
    call tq(T_SETD)
    call tw(1)

    if (fs_verbose >= 1) call fs_solve_line()
    if (iic == ntstart + ntimes - 1) call fs_cost_print()
  end subroutine fs_step

  subroutine fs_unconv_stop(k, its, resid)
    ! The solve of sub-step k reached fs_maxit without meeting fs_tol: stop the run (the CPU's
    ! fs_unconv_stop, same message). conv comes from global reductions, so every compute rank comes
    ! here: compute rank 0 writes the message, every compute rank meets the others once more (so the
    ! message is out before any rank aborts) and stops all ranks, writers included, by comm_abort
    ! (MPI_Abort), as the MARBL solver failure does.
    integer(ip), intent(in) :: k
    integer, intent(in) :: its
    real(rp), intent(in) :: resid
    character(len=320) :: msg
    real(rp) :: x(1), y(1)
    write(msg, '(A,I0,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,I0,A)') 'IMPLICIT_FS: the free-surface solve did not'// &
         ' converge (step ', iic, ', sub-step ', k, ' of ', fs_nsub, ': ', its, ' iterations, relative residual', &
         resid, ', fs_tol =', fs_tol, ', fs_maxit = ', fs_maxit, '): the run is unstable, or fs_tol is below what'// &
         ' rounding allows'
    if (talk) then
      write(*,'(1x,A)') trim(msg)
      flush(6)
    endif
    x = 0.0_rp
    call comm_allreduce_max(1, x, y)
    call comm_abort(1)
  end subroutine fs_unconv_stop

  subroutine fs_tang_bc(ks, kn, dts_)
    ! Tangential boundary velocities (FS_SPEC 3.7.5, the CPU's fs_tang_bc):
    ! open edges the backward-Euler form of stock u2dbc/v2dbc (upwind advection by the
    ! normal flow, inflow nudging at fs_tang_nudge |u_n|/dx; the former fs_tang = 'adv');
    ! closed edges gamma2 (= 1, roms_gpu supports free slip only) v_int;
    ! then the corners between two open edges.
    ! ks, kn, dts_: the (sub-)step's source and new levels and its length
    integer(ip), intent(in) :: ks, kn
    real(rp), intent(in) :: dts_
    integer(ip) :: m, nx, ny, nmax, istrU, iendR, jstrV, jendR
    real(rp) :: dtl, ca, cy, tnu, cc, rr_, gmm, gpp
    logical :: ew, ee, es, en, ow, oe, os, on
    real(rp), parameter :: gamma2 = 1.0_rp
    nx = decomp%nx; ny = decomp%ny
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    ow = obc_west; oe = obc_east; os = obc_south; on = obc_north
    if (.not. (ew .or. ee .or. es .or. en)) return
    istrU = merge(2, 1, ew); iendR = merge(nx + 1, nx, ee)
    jstrV = merge(2, 1, es); jendR = merge(ny + 1, ny, en)
    dtl = dts_; tnu = fs_tang_nudge
    nmax = max(nx, ny) + 1
    CUF_DO1
    do m = 1, nmax
      if (ew) then
        if (ow) then
          if (m >= jstrV .and. m <= ny) then
            ca = -0.125_rp * dtl * (ubar(1, m, ks) + ubar(1, m - 1, ks)) &
                 * (pm(0, m) + pm(0, m - 1) + pm(1, m) + pm(1, m - 1))
            cy = 0.125_rp * dtl * (vbar(0, m, ks) + vbar(1, m, ks)) &
                 * (pn(0, m) + pn(0, m - 1) + pn(1, m) + pn(1, m - 1))
            ! the CPU's tnew, inline (no host scalars passed by reference)
            gmm = vbar(0, m, ks) - vbar(0, m - 1, ks); gpp = vbar(0, m + 1, ks) - vbar(0, m, ks)
            cc = min(1.0_rp, max(-1.0_rp, cy))
            rr_ = vbar(0, m, ks) - max(cc, 0.0_rp) * gmm - min(cc, 0.0_rp) * gpp
            if (ca > 0.0_rp) then
              rr_ = (rr_ + ca * vbar(1, m, kn)) / (1.0_rp + ca)
            else
              cc = -ca * tnu
              rr_ = (rr_ + cc * vbar_west(m)) / (1.0_rp + cc)
            endif
            vbar(0, m, kn) = rr_
            vbar(0, m, kn) = vbar(0, m, kn) * vmask(0, m)
          endif
        else
          if (m >= 1 .and. m <= jendR) then
            vbar(0, m, kn) = gamma2 * vbar(1, m, kn)
            vbar(0, m, kn) = vbar(0, m, kn) * vmask(0, m)
          endif
        endif
      endif
      if (ee) then
        if (oe) then
          if (m >= jstrV .and. m <= ny) then
            ca = 0.125_rp * dtl * (ubar(nx + 1, m, ks) + ubar(nx + 1, m - 1, ks)) &
                 * (pm(nx, m) + pm(nx, m - 1) + pm(nx + 1, m) + pm(nx + 1, m - 1))
            cy = 0.125_rp * dtl * (vbar(nx, m, ks) + vbar(nx + 1, m, ks)) &
                 * (pn(nx, m) + pn(nx, m - 1) + pn(nx + 1, m) + pn(nx + 1, m - 1))
            ! the CPU's tnew, inline (no host scalars passed by reference)
            gmm = vbar(nx + 1, m, ks) - vbar(nx + 1, m - 1, ks); gpp = vbar(nx + 1, m + 1, ks) - vbar(nx + 1, m, ks)
            cc = min(1.0_rp, max(-1.0_rp, cy))
            rr_ = vbar(nx + 1, m, ks) - max(cc, 0.0_rp) * gmm - min(cc, 0.0_rp) * gpp
            if (ca > 0.0_rp) then
              rr_ = (rr_ + ca * vbar(nx, m, kn)) / (1.0_rp + ca)
            else
              cc = -ca * tnu
              rr_ = (rr_ + cc * vbar_east(m)) / (1.0_rp + cc)
            endif
            vbar(nx + 1, m, kn) = rr_
            vbar(nx + 1, m, kn) = vbar(nx + 1, m, kn) * vmask(nx + 1, m)
          endif
        else
          if (m >= 1 .and. m <= jendR) then
            vbar(nx + 1, m, kn) = gamma2 * vbar(nx, m, kn)
            vbar(nx + 1, m, kn) = vbar(nx + 1, m, kn) * vmask(nx + 1, m)
          endif
        endif
      endif
      if (es) then
        if (os) then
          if (m >= istrU .and. m <= nx) then
            ca = -0.125_rp * dtl * (vbar(m, 1, ks) + vbar(m - 1, 1, ks)) &
                 * (pn(m, 0) + pn(m - 1, 0) + pn(m, 1) + pn(m - 1, 1))
            cy = 0.125_rp * dtl * (ubar(m, 0, ks) + ubar(m, 1, ks)) &
                 * (pm(m, 0) + pm(m - 1, 0) + pm(m, 1) + pm(m - 1, 1))
            ! the CPU's tnew, inline (no host scalars passed by reference)
            gmm = ubar(m, 0, ks) - ubar(m - 1, 0, ks); gpp = ubar(m + 1, 0, ks) - ubar(m, 0, ks)
            cc = min(1.0_rp, max(-1.0_rp, cy))
            rr_ = ubar(m, 0, ks) - max(cc, 0.0_rp) * gmm - min(cc, 0.0_rp) * gpp
            if (ca > 0.0_rp) then
              rr_ = (rr_ + ca * ubar(m, 1, kn)) / (1.0_rp + ca)
            else
              cc = -ca * tnu
              rr_ = (rr_ + cc * ubar_south(m)) / (1.0_rp + cc)
            endif
            ubar(m, 0, kn) = rr_
            ubar(m, 0, kn) = ubar(m, 0, kn) * umask(m, 0)
          endif
        else
          if (m >= 1 .and. m <= iendR) then
            ubar(m, 0, kn) = gamma2 * ubar(m, 1, kn)
            ubar(m, 0, kn) = ubar(m, 0, kn) * umask(m, 0)
          endif
        endif
      endif
      if (en) then
        if (on) then
          if (m >= istrU .and. m <= nx) then
            ca = 0.125_rp * dtl * (vbar(m, ny + 1, ks) + vbar(m - 1, ny + 1, ks)) &
                 * (pn(m, ny) + pn(m - 1, ny) + pn(m, ny + 1) + pn(m - 1, ny + 1))
            cy = 0.125_rp * dtl * (ubar(m, ny, ks) + ubar(m, ny + 1, ks)) &
                 * (pm(m, ny) + pm(m - 1, ny) + pm(m, ny + 1) + pm(m - 1, ny + 1))
            ! the CPU's tnew, inline (no host scalars passed by reference)
            gmm = ubar(m, ny + 1, ks) - ubar(m - 1, ny + 1, ks); gpp = ubar(m + 1, ny + 1, ks) - ubar(m, ny + 1, ks)
            cc = min(1.0_rp, max(-1.0_rp, cy))
            rr_ = ubar(m, ny + 1, ks) - max(cc, 0.0_rp) * gmm - min(cc, 0.0_rp) * gpp
            if (ca > 0.0_rp) then
              rr_ = (rr_ + ca * ubar(m, ny, kn)) / (1.0_rp + ca)
            else
              cc = -ca * tnu
              rr_ = (rr_ + cc * ubar_north(m)) / (1.0_rp + cc)
            endif
            ubar(m, ny + 1, kn) = rr_
            ubar(m, ny + 1, kn) = ubar(m, ny + 1, kn) * umask(m, ny + 1)
          endif
        else
          if (m >= 1 .and. m <= iendR) then
            ubar(m, ny + 1, kn) = gamma2 * ubar(m, ny, kn)
            ubar(m, ny + 1, kn) = ubar(m, ny + 1, kn) * umask(m, ny + 1)
          endif
        endif
      endif
    enddo
    if ((ew .and. ow .or. ee .and. oe) .and. (es .and. os .or. en .and. on)) then
      CUF_SERIAL
      do m = 1, 1
        if (ew .and. es .and. ow .and. os) then
          ubar(1, 0, kn) = 0.5_rp * (ubar(2, 0, kn) + ubar(1, 1, kn))
          vbar(0, 1, kn) = 0.5_rp * (vbar(0, 2, kn) + vbar(1, 1, kn))
        endif
        if (ee .and. es .and. oe .and. os) then
          ubar(nx + 1, 0, kn) = 0.5_rp * (ubar(nx, 0, kn) + ubar(nx + 1, 1, kn))
          vbar(nx + 1, 1, kn) = 0.5_rp * (vbar(nx + 1, 2, kn) + vbar(nx, 1, kn))
        endif
        if (ew .and. en .and. ow .and. on) then
          ubar(1, ny + 1, kn) = 0.5_rp * (ubar(2, ny + 1, kn) + ubar(1, ny, kn))
          vbar(0, ny + 1, kn) = 0.5_rp * (vbar(0, ny, kn) + vbar(1, ny + 1, kn))
        endif
        if (ee .and. en .and. oe .and. on) then
          ubar(nx + 1, ny + 1, kn) = 0.5_rp * (ubar(nx, ny + 1, kn) + ubar(nx + 1, ny, kn))
          vbar(nx + 1, ny + 1, kn) = 0.5_rp * (vbar(nx + 1, ny, kn) + vbar(nx, ny + 1, kn))
        endif
      enddo
    endif
  end subroutine fs_tang_bc

  !======================================================================
  ! PCG (FS_SPEC 4)
  !======================================================================
  subroutine fs_pcg(its, resid, conv)
    integer, intent(out) :: its
    real(rp), intent(out) :: resid
    logical, intent(out) :: conv
    integer(ip) :: i, j, nx, ny
    integer :: it
    real(rp) :: loc(4), glob(4), rz, rznew, alpha, beta, pq, tol2, bn2
    nx = decomp%nx; ny = decomp%ny
    conv = .true.; its = 0; resid = 0.0_rp
    call ring(1, xx, fdum, fdum, fdum, fdum, fdum, .true.)
    call tq(T_PEX)
    call fs_apply(xx, qq, .false.)
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        xx(i, j) = act(i, j) * xx(i, j)
        rr(i, j) = bb(i, j) - qq(i, j)
        fs_part(i + (j - 1) * nx, 4) = act(i, j) * bb(i, j) * bb(i, j) * pm(i, j) * pn(i, j)
      enddo
    enddo
    call tq(T_START)
    call precond()
    CUF_DO2
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        pd(i, j) = zz(i, j)
      enddo
    enddo
    call tq(T_START)
    call fs_sum(4, loc)
    call comm_allreduce_sum(4, loc, glob)
    call tq(T_RED)
    rz = glob(1); bn2 = glob(4)
    if (bn2 <= 0.0_rp) then
      call zero2(xx)
      return
    endif
    tol2 = fs_tol * fs_tol * bn2
    resid = sqrt(glob(2) / bn2)
    if (glob(2) <= tol2) return
    conv = .false.
    do it = 1, fs_maxit
      its = it
      call ring(1, pd, fdum, fdum, fdum, fdum, fdum, .true.)
      call tq(T_PEX)
      call fs_apply(pd, qq, .true.)
      call tq(T_OP)
      call fs_sum(1, loc)
      call comm_allreduce_sum(1, loc(1:1), glob(1:1))
      call tq(T_RED)
      pq = glob(1)
      alpha = rz / pq
      if (mg_on) then
        CUF_DO2
        do j = 1, ny
          do i = 1, nx
            xx(i, j) = xx(i, j) + alpha * pd(i, j)
            rr(i, j) = rr(i, j) - alpha * qq(i, j)
          enddo
        enddo
        call tq(T_VEC)
        call precond()
      else
        CUF_DO2
        do j = 1, ny
          do i = 1, nx
            xx(i, j) = xx(i, j) + alpha * pd(i, j)
            rr(i, j) = rr(i, j) - alpha * qq(i, j)
            zz(i, j) = dinv(i, j) * rr(i, j)
            fs_part(i + (j - 1) * nx, 1) = rr(i, j) * zz(i, j)
            fs_part(i + (j - 1) * nx, 2) = act(i, j) * rr(i, j) * rr(i, j) * pm(i, j) * pn(i, j)
          enddo
        enddo
        call tq(T_PC)
      endif
      call fs_sum(merge(3, 2, mg_on), loc)
      call comm_allreduce_sum(merge(3, 2, mg_on), loc(1:3), glob(1:3))
      call tq(T_RED)
      rznew = glob(1)
      resid = sqrt(glob(2) / bn2)
      if (glob(2) <= tol2) then
        conv = .true.; exit
      endif
      if (mg_on) then
        beta = -alpha * glob(3) / rz
      else
        beta = rznew / rz
      endif
      rz = rznew
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          pd(i, j) = zz(i, j) + beta * pd(i, j)
        enddo
      enddo
      call tq(T_VEC)
    enddo
    its = min(its, fs_maxit)
  end subroutine fs_pcg

  subroutine fs_apply(x, y, dot)
    ! y = A x on the interior (FS_SPEC 3.6); dot: partial sums x y in slot 1
    real(rp), FSD intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), FSD intent(inout) :: y(decomp%is:decomp%ie, decomp%js:decomp%je)
    logical, intent(in) :: dot
    integer(ip) :: i, j, nx, ny
    logical :: dt_
    nx = decomp%nx; ny = decomp%ny; dt_ = dot
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        y(i, j) = act(i, j) * ((ar(i, j) + rb(i, j)) * x(i, j) &
             + cu(i + 1, j) * (x(i, j) - x(i + 1, j)) + cu(i, j) * (x(i, j) - x(i - 1, j)) &
             + cv(i, j + 1) * (x(i, j) - x(i, j + 1)) + cv(i, j) * (x(i, j) - x(i, j - 1)))
        if (dt_) fs_part(i + (j - 1) * nx, 1) = x(i, j) * y(i, j)
      enddo
    enddo
  end subroutine fs_apply

  subroutine precond()
    ! z = M^-1 r; partial sums <r,z> (slot 1), sum act r^2 pm pn (2), <z,q> (3, MG)
    integer(ip) :: i, j, nx, ny
    nx = decomp%nx; ny = decomp%ny
    if (mg_on) then
      call mg_vcycle()
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          fs_part(i + (j - 1) * nx, 1) = rr(i, j) * zz(i, j)
          fs_part(i + (j - 1) * nx, 2) = act(i, j) * rr(i, j) * rr(i, j) * pm(i, j) * pn(i, j)
          fs_part(i + (j - 1) * nx, 3) = zz(i, j) * qq(i, j)
        enddo
      enddo
      call tq(T_PC)
      return
    endif
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        zz(i, j) = dinv(i, j) * rr(i, j)
        fs_part(i + (j - 1) * nx, 1) = rr(i, j) * zz(i, j)
        fs_part(i + (j - 1) * nx, 2) = act(i, j) * rr(i, j) * rr(i, j) * pm(i, j) * pn(i, j)
      enddo
    enddo
    call tq(T_PC)
  end subroutine precond

  !======================================================================
  ! Multigrid preconditioner (FS_SPEC 5)
  !======================================================================








  !======================================================================
  ! Exchanges
  !======================================================================




  !======================================================================
  ! Deterministic sums
  !======================================================================


  !======================================================================
  ! Helpers, diagnostics, timing
  !======================================================================

  subroutine fs_volume_check()
    ! fs_verbose >= 2: the step's global volume budget (FS_VOL line)
    integer(ip) :: i, j, nx, ny, ks, kn
    real(rp) :: loc(5), glob(5), dz, dtl, s3
    logical :: ew, ee, es, en
    nx = decomp%nx; ny = decomp%ny; ks = kstp; kn = knew; dtl = dt
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        fs_part(i + (j - 1) * nx, 1) = 0.0_rp; fs_part(i + (j - 1) * nx, 2) = 0.0_rp
        fs_part(i + (j - 1) * nx, 3) = 0.0_rp; fs_part(i + (j - 1) * nx, 4) = 0.0_rp
        fs_part(i + (j - 1) * nx, 5) = 0.0_rp
        if (ar(i, j) > 0.0_rp) then
          dz = zeta(i, j, kn) - zeta(i, j, ks)
          fs_part(i + (j - 1) * nx, 1) = ar(i, j) * dz
          fs_part(i + (j - 1) * nx, 4) = abs(ar(i, j) * dz)
          fs_part(i + (j - 1) * nx, 5) = ar(i, j) * zeta(i, j, kn)
          fs_part(i + (j - 1) * nx, 2) = dtl * ar(i, j) * qs(i, j)
          s3 = 0.0_rp
          if (i == 1 .and. ew) then
            s3 = s3 + dtl * fbu(i, j)
          else if (ar(i - 1, j) <= 0.0_rp) then
            s3 = s3 + dtl * fbu(i, j)
          endif
          if (i == nx .and. ee) then
            s3 = s3 - dtl * fbu(i + 1, j)
          else if (ar(i + 1, j) <= 0.0_rp) then
            s3 = s3 - dtl * fbu(i + 1, j)
          endif
          if (j == 1 .and. es) then
            s3 = s3 + dtl * fbv(i, j)
          else if (ar(i, j - 1) <= 0.0_rp) then
            s3 = s3 + dtl * fbv(i, j)
          endif
          if (j == ny .and. en) then
            s3 = s3 - dtl * fbv(i, j + 1)
          else if (ar(i, j + 1) <= 0.0_rp) then
            s3 = s3 - dtl * fbv(i, j + 1)
          endif
          fs_part(i + (j - 1) * nx, 3) = s3
        endif
      enddo
    enddo
    call fs_sum(5, loc)
    call comm_allreduce_sum(5, loc, glob)
    if (talk) write(*,'(1x,A,I8,A,ES14.6,A,ES14.6,A,ES10.2,A,ES20.12)') 'FS_VOL', iic, &
         ' dV', glob(1), ' dt(Q+inflow)', glob(2) + glob(3), &
         ' rel_err', (glob(1) - glob(2) - glob(3)) / max(glob(4), 1.e-300_rp), ' V', glob(5)
  end subroutine fs_volume_check

  real(rp) function xx_absmax()
    ! max |delta| over the tile (the sub-step's FS_SOLVE statistic)
    integer(ip) :: i, j, nx, ny
    real(rp) :: amax
    nx = decomp%nx; ny = decomp%ny
    amax = 0.0_rp
    !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(max:amax)
    do j = 1, ny
      do i = 1, nx
        amax = max(amax, abs(xx(i, j)))
      enddo
    enddo
    xx_absmax = amax
  end function xx_absmax

  subroutine fs_solve_line()
    integer(ip) :: i, j, nx, ny
    real(rp) :: loc(1), glob(1), amax
    nx = decomp%nx; ny = decomp%ny
    if (fs_nsub > 1) then
      ! sub-steps: iterations summed over the step (and the largest of one sub-step), the worst
      ! residual, the largest increment of a sub-step (CPU b6325f06)
      loc(1) = dz_max
      call comm_allreduce_max(1, loc, glob)
      if (talk) write(*,'(1x,A,I8,A,I6,A,ES10.3,A,ES11.4,A,I2,A,I6)') 'FS_SOLVE', iic, ' its', its_sum, &
           ' rel_res', res_max, ' max|dzeta|', glob(1), ' sub-steps', fs_nsub, ' max its', its_max
      return
    endif
    amax = 0.0_rp
    !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(max:amax)
    do j = 1, ny
      do i = 1, nx
        amax = max(amax, abs(xx(i, j)))
      enddo
    enddo
    loc(1) = amax
    call comm_allreduce_max(1, loc, glob)
    if (talk) write(*,'(1x,A,I8,A,I6,A,ES10.3,A,ES11.4)') 'FS_SOLVE', iic, ' its', its_last, &
         ' rel_res', res_last, ' max|dzeta|', glob(1)
  end subroutine fs_solve_line

  subroutine fs_cost_print()
    ! end of the run: FS_STAT (as the CPU, always); with fs_verbose >= 1 also the GPU's communication
    ! counts per solve (FS_STAT per solve); with fs_gpu_timers 1 the FS_COST pieces, with 2 FS_WIN
    ! (their reductions only then)
    real(8) :: gm(ntm), lm(ntm), x(2), y(2)
    integer :: i
    if (talk) then
      write(*,'(1x,A,I7,A,F9.2,A,I6)') 'FS_STAT solves', n_solve, ' mean its', real(n_its, 8) / max(n_solve, 1), &
           ' unconverged', n_unconv
      if (fs_nsub > 1) write(*,'(1x,A,I7,A,I2,A)') 'FS_STAT steps', n_step, ' (', fs_nsub, ' sub-steps each)'
      if (fs_verbose >= 1) write(*,'(1x,A,4(A,F9.2))') 'FS_STAT per solve:', ' ring rounds', &
           real(n_ring, 8) / max(n_solve, 1), ' reductions', real(n_red, 8) / max(n_solve, 1), &
           ' V-cycles', real(n_vc, 8) / max(n_solve, 1), ' host gathers', real(n_gath, 8) / max(n_solve, 1)
    endif
    if (fs_gpu_timers < 1) return
    lm = tm
    call comm_allreduce_max(ntm, lm, gm)
    x(1) = tm_mgsetup; x(2) = tm_win
    call comm_allreduce_max(2, x, y)
    if (talk) then
      if (fs_gpu_timers == 2) write(*,'(1x,A,F9.3,A,I6,A)') 'FS_WIN', 1.e3_8 * y(2) / max(n_win - 10, 1), &
           ' ms per step (FS window, device synchronised at entry and exit; max rank, mean over', n_win - 10, &
           ' steps after the first 10)'
      if (fs_gpu_timers == 1) then
        write(*,'(1x,A,I6,A)') 'FS_COST [ms per step, max rank over', n_step, ' steps; synchronised pieces]:'
        do i = 1, ntm
          write(*,'(1x,A,1x,A12,F9.3)') 'FS_COST', tm_name(i), 1.e3_8 * gm(i) / max(n_step, 1)
        enddo
        write(*,'(1x,A,1x,A12,F9.3)') 'FS_COST', 'mg_setup', 1.e3_8 * y(1) / max(n_step, 1)
        write(*,'(1x,A,1x,A12,F9.3)') 'FS_COST', 'SUM', 1.e3_8 * (sum(gm) + y(1)) / max(n_step, 1)
      endif
    endif
  end subroutine fs_cost_print




  subroutine fs_rst_prepare()
    ! before a restart record is written (his_io): the rates and delta exist; before the
    ! first implicit step there are none: zeros with qd = 1e30 ('no rates', as the CPU's
    ! fs_rst_wrt); the device is drained so the host can read the managed arrays
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (.not. allocated(udu)) then
      call fs_alloc_rates()
      qd = 1.e30_rp
    endif
    if (mg_on .and. fs_mg_lag > 1) call mg_alloc_keep()   ! (no build yet: zeros, flag 5 = 0)
#ifdef USE_CUDA
    ierr = cudaDeviceSynchronize()
#endif
  end subroutine fs_rst_prepare

  subroutine mg_alloc_keep()
    integer(ip) :: is, ie, js, je
    if (allocated(mg_se0)) return
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    allocate(mg_se0(is:ie, js:je), mg_scu(is:ie, js:je), mg_scv(is:ie, js:je))
    mg_se0 = 0; mg_scu = 0; mg_scv = 0
  end subroutine mg_alloc_keep

  subroutine mg_keep_fine()
    ! fs_mg_lag > 1: the fine operator the coarse operators were just built from (restart
    ! file; mg_setup reads act (ar + rb) on the tile, cu on faces 1..nx+1, cv on 1..ny+1)
    integer(ip) :: i, j, nx, ny, is, ie, js, je
    call mg_alloc_keep()
    nx = decomp%nx; ny = decomp%ny
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) mg_se0(i, j) = act(i, j) * (ar(i, j) + rb(i, j))
        mg_scu(i, j) = cu(i, j); mg_scv(i, j) = cv(i, j)
      enddo
    enddo
  end subroutine mg_keep_fine

  subroutine mg_replay()
    ! the first solve after a restart within a fs_mg_lag period: the coarse operators of the
    ! last build from its fine operator by mg_setup with act = 1, ar = mg_se0, rb = 0, cu, cv =
    ! mg_scu, mg_scv (act (ar + rb) = mg_se0 exactly); the step's arrays restored (CPU mg_replay)
    integer(ip) :: i, j, is, ie, js, je
    real(rp), FSD allocatable, dimension(:,:) :: s_act, s_ar, s_rb, s_cu, s_cv
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call ring(3, mg_se0, mg_scu, mg_scv, fdum, fdum, fdum, .true.)   ! the halo faces read from the file
    allocate(s_act(is:ie, js:je), s_ar(is:ie, js:je), s_rb(is:ie, js:je), s_cu(is:ie, js:je), s_cv(is:ie, js:je))
    CUF_DO2
    do j = js, je
      do i = is, ie
        s_act(i, j) = act(i, j); s_ar(i, j) = ar(i, j); s_rb(i, j) = rb(i, j); s_cu(i, j) = cu(i, j)
        s_cv(i, j) = cv(i, j)
        act(i, j) = 1.0_rp; ar(i, j) = mg_se0(i, j); rb(i, j) = 0.0_rp; cu(i, j) = mg_scu(i, j)
        cv(i, j) = mg_scv(i, j)
      enddo
    enddo
    call mg_setup()
    mg_built = mg_built + 1
    CUF_DO2
    do j = js, je
      do i = is, ie
        act(i, j) = s_act(i, j); ar(i, j) = s_ar(i, j); rb(i, j) = s_rb(i, j); cu(i, j) = s_cu(i, j)
        cv(i, j) = s_cv(i, j)
      enddo
    enddo
    deallocate(s_act, s_ar, s_rb, s_cu, s_cv)
    mg_rpend = .false.
  end subroutine mg_replay

  subroutine fs_rst_scal_vals(sc)
    ! fs_rst_scal of the record (the CPU's layout): 1 the scalars are set (1), 2 fs_mg_lag, 3 phase of the
    ! coarse-operator rebuild counter for the next step, 4 format version of the stored state (rst_format),
    ! 5 the fine operator of the last build is in fs_mge0, fs_mgcu, fs_mgcv (1/0), 6 fs_nsub of the run
    real(8), intent(out) :: sc(fs_nsc)
    sc = 0.0d0
    if (.not. fs_on) return
    sc(1) = 1.0d0; sc(2) = dble(fs_mg_lag); sc(3) = dble(mod(n_solve + mg_ph0, fs_mg_lag))
    sc(4) = dble(rst_format)
    if (mg_built > 0 .and. allocated(mg_se0)) sc(5) = 1.0d0
    sc(6) = dble(fs_nsub)
  end subroutine fs_rst_scal_vals

  subroutine fs_rst_scal_in(have, sc, take_rates, take_mg)
    ! case_io, before the rates are read (the CPU's fs_rst_get): a record whose stored state has another
    ! format version than rst_format gives a cold start of the memory (rates not read); a file without
    ! the scalars: the rates are used, the coarse operators are built at the first step; with the same
    ! fs_mg_lag > 1 and a phase /= 0 the stored fine operator is read (take_mg) and the coarse
    ! operators of the last build are rebuilt from it before the first solve
    logical, intent(in) :: have
    real(8), intent(in) :: sc(fs_nsc)
    logical, intent(out) :: take_rates, take_mg
    take_rates = .true.; take_mg = .false.
    if (.not. fs_on) return
    if (have .and. nint(sc(1)) == 1 .and. nint(sc(4)) /= rst_format) then
      if (talk) write(*,'(1x,A,I0,A,I0,A)') 'IMPLICIT_FS: the restart record''s implicit free-surface'// &
           ' state has format version ', nint(sc(4)), ', this model reads version ', rst_format, ': cold start'// &
           ' of the generalized-alpha memory (not an exact restart)'
      take_rates = .false.
      return
    endif
    if (have .and. nint(sc(6)) > 0 .and. nint(sc(6)) /= fs_nsub .and. talk) write(*,'(1x,A,I2,A,I2,A)') &
         'IMPLICIT_FS: the restart record was written with fs_nsub =', nint(sc(6)), ', this run uses', fs_nsub, &
         ' (the rates are per second and are used as they are; not an exact restart)'
    mg_ph0 = 0; mg_rpend = .false.
    if (.not. have) return
    if (mg_on .and. fs_mg_lag > 1 .and. nint(sc(1)) == 1 .and. nint(sc(2)) == fs_mg_lag .and. nint(sc(3)) /= 0) then
      if (nint(sc(5)) == 1) then           ! within a lag period: the coarse operators of the last build
        call mg_alloc_keep()
        mg_ph0 = nint(sc(3)); take_mg = .true.
      else if (talk) then
        write(*,'(1x,A)') 'IMPLICIT_FS: fs_mg_lag > 1 restart within a lag period without the stored'// &
             ' operator: coarse operators rebuilt at the first step (not exact)'
      endif
    else if (mg_on .and. fs_mg_lag > 1 .and. nint(sc(3)) /= 0 .and. talk) then
      write(*,'(1x,A)') 'IMPLICIT_FS: fs_mg_lag differs from the run that wrote the restart: coarse'// &
           ' operators rebuilt at the first step (not exact)'
    endif
  end subroutine fs_rst_scal_in

  subroutine fs_mg_loaded()
    ! after case_io has read fs_mge0, fs_mgcu, fs_mgcv: their halo faces (the E1 ring of cu,
    ! cv), the replay before the first solve
    mg_rpend = .true.
  end subroutine fs_mg_loaded

  subroutine fs_src_diag()
    ! fs_verbose >= 2, both fs_mode values (step_driver, after the barotropic step): FS_SRC
    ! line, global sums [m3/s] of the prescribed volume sources of the step and of what the
    ! barotropic step delivered (the CPU's fs_src_diag): riv_map = sum |Q_r| over the owned
    ! river faces, riv_del = sum sign(Q_r) DU_avg2 there, src_pc = pipes + CDR volume (no pipes
    ! on the GPU; the CDR volume is not summed here: 0), src_sw = sum A swflx, qbar = sum A Qbar
    ! (implicit only). CUF-kernel sums (diagnostic; not bitwise reproducible).
    integer(ip) :: i, j, nx, ny, istrU, jstrV, is, ie, js, je
    real(rp) :: s1, s2, s4, s5, rf, loc(5), glob(5)
    logical :: rv, fo
    nx = decomp%nx; ny = decomp%ny
    istrU = merge(2, 1, decomp%edge_w); jstrV = merge(2, 1, decomp%edge_s)
    rv = river_on; fo = fs_on
    if (rv .and. .not. allocated(rivu)) then        ! fs_mode = 'split': the face map only
      is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
      allocate(rivu(is:ie, js:je), rivv(is:ie, js:je))
      rivu = 0; rivv = 0
      call rivers_fs_map(rivu, rivv)
    endif
    s1 = 0.0_rp; s2 = 0.0_rp; s4 = 0.0_rp; s5 = 0.0_rp
    if (rv) then
      !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(+:s1, s2)
      do j = 1, ny
        do i = 1, nx
          if (i >= istrU .and. rivu(i, j) > 0) then
            rf = rf_flx(rivu(i, j))
            s1 = s1 + abs(rf); s2 = s2 + sign(1.0_rp, rf) * DU_avg2(i, j)
          endif
          if (j >= jstrV .and. rivv(i, j) > 0) then
            rf = rf_flx(rivv(i, j))
            s1 = s1 + abs(rf); s2 = s2 + sign(1.0_rp, rf) * DV_avg2(i, j)
          endif
        enddo
      enddo
    endif
    !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(+:s4)
    do j = 1, ny
      do i = 1, nx
        s4 = s4 + rmask(i, j) * swflx(i, j) / (pm(i, j) * pn(i, j))
      enddo
    enddo
    if (fo) then
      !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(+:s5)
      do j = 1, ny
        do i = 1, nx
          s5 = s5 + ar(i, j) * qs(i, j)
        enddo
      enddo
    endif
    loc = [s1, s2, 0.0_rp, s4, s5]
    call comm_allreduce_sum(5, loc, glob)
    if (talk) write(*,'(1x,A,I8,A,ES20.12,A,ES20.12,A,ES20.12,A,ES20.12,A,ES20.12)') 'FS_SRC', iic, &
         ' riv_map', glob(1), ' riv_del', glob(2), ' src_pc', glob(3), ' src_sw', glob(4), ' qbar', glob(5)
    call fs_wall_diag()
  end subroutine fs_src_diag

  subroutine fs_wall_diag()
    ! FS_WALL line (CPU 1d174880, FS_SPEC 9b-J item 3; fs_verbose >= 2, both fs_mode values): the
    ! normal velocity the barotropic step left on closed faces, global maxima of |ubar(knew)|,
    ! |DU_avg1|, |DU_avg2| (1-3) over the normal faces of the closed physical edges (not open, not
    ! periodic) and (4-6) over the masked faces (umask = 0, not a river face) of the owned faces.
    ! One unconditional max per variable and kernel (masks as factors: CUF reductions with
    ! conditional updates were wrong, session 5).
    integer(ip) :: i, j, nx, ny, istrU, jstrV, kn, ii, jj
    real(rp) :: a1, a2, a3, a4, a5, a6, w
    real(rp) :: loc(6), glob(6)
    nx = decomp%nx; ny = decomp%ny; kn = knew
    istrU = merge(2, 1, decomp%edge_w); jstrV = merge(2, 1, decomp%edge_s)
    loc = 0.0_rp
    a1 = 0.0_rp; a2 = 0.0_rp; a3 = 0.0_rp
    if (.not. decomp%periodic_x) then
      if (decomp%edge_w .and. .not. obc_west) then
        ii = 1
        !$cuf kernel do(1) <<<*,*,0,work_stream>>> reduce(max:a1, a2, a3)
        do j = 1, ny
          a1 = max(a1, abs(ubar(ii, j, kn))); a2 = max(a2, abs(DU_avg1(ii, j))); a3 = max(a3, abs(DU_avg2(ii, j)))
        enddo
        loc(1) = max(loc(1), a1); loc(2) = max(loc(2), a2); loc(3) = max(loc(3), a3)
      endif
      a1 = 0.0_rp; a2 = 0.0_rp; a3 = 0.0_rp
      if (decomp%edge_e .and. .not. obc_east) then
        ii = nx + 1
        !$cuf kernel do(1) <<<*,*,0,work_stream>>> reduce(max:a1, a2, a3)
        do j = 1, ny
          a1 = max(a1, abs(ubar(ii, j, kn))); a2 = max(a2, abs(DU_avg1(ii, j))); a3 = max(a3, abs(DU_avg2(ii, j)))
        enddo
        loc(1) = max(loc(1), a1); loc(2) = max(loc(2), a2); loc(3) = max(loc(3), a3)
      endif
    endif
    a1 = 0.0_rp; a2 = 0.0_rp; a3 = 0.0_rp
    if (.not. decomp%periodic_y) then
      if (decomp%edge_s .and. .not. obc_south) then
        jj = 1
        !$cuf kernel do(1) <<<*,*,0,work_stream>>> reduce(max:a1, a2, a3)
        do i = 1, nx
          a1 = max(a1, abs(vbar(i, jj, kn))); a2 = max(a2, abs(DV_avg1(i, jj))); a3 = max(a3, abs(DV_avg2(i, jj)))
        enddo
        loc(1) = max(loc(1), a1); loc(2) = max(loc(2), a2); loc(3) = max(loc(3), a3)
      endif
      a1 = 0.0_rp; a2 = 0.0_rp; a3 = 0.0_rp
      if (decomp%edge_n .and. .not. obc_north) then
        jj = ny + 1
        !$cuf kernel do(1) <<<*,*,0,work_stream>>> reduce(max:a1, a2, a3)
        do i = 1, nx
          a1 = max(a1, abs(vbar(i, jj, kn))); a2 = max(a2, abs(DV_avg1(i, jj))); a3 = max(a3, abs(DV_avg2(i, jj)))
        enddo
        loc(1) = max(loc(1), a1); loc(2) = max(loc(2), a2); loc(3) = max(loc(3), a3)
      endif
    endif
    a4 = 0.0_rp; a5 = 0.0_rp; a6 = 0.0_rp
    !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(max:a4, a5, a6)
    do j = 1, ny
      do i = istrU, nx
        w = merge(1.0_rp, 0.0_rp, umask(i, j) == 0.0_rp .and. riv_umask(i, j) == 0.0_rp)
        a4 = max(a4, w * abs(ubar(i, j, kn))); a5 = max(a5, w * abs(DU_avg1(i, j))); a6 = max(a6, w * abs(DU_avg2(i, j)))
      enddo
    enddo
    loc(4) = a4; loc(5) = a5; loc(6) = a6
    a4 = 0.0_rp; a5 = 0.0_rp; a6 = 0.0_rp
    !$cuf kernel do(2) <<<*,*,0,work_stream>>> reduce(max:a4, a5, a6)
    do j = jstrV, ny
      do i = 1, nx
        w = merge(1.0_rp, 0.0_rp, vmask(i, j) == 0.0_rp .and. riv_vmask(i, j) == 0.0_rp)
        a4 = max(a4, w * abs(vbar(i, j, kn))); a5 = max(a5, w * abs(DV_avg1(i, j))); a6 = max(a6, w * abs(DV_avg2(i, j)))
      enddo
    enddo
    loc(4) = max(loc(4), a4); loc(5) = max(loc(5), a5); loc(6) = max(loc(6), a6)
    call comm_allreduce_max(6, loc, glob)
    if (talk) write(*,'(1x,A,I8,A,3ES10.2,A,3ES10.2)') 'FS_WALL', iic, ' closed_edges ubar,DU_avg1,DU_avg2', &
         glob(1:3), ' masked', glob(4:6)
  end subroutine fs_wall_diag

  subroutine fs_rst_loaded()
    ! after case_io has read fs_udu ... fs_delta from the initial (restart) file into the
    ! managed arrays (the CPU's fs_rst_get): the 'no rates' marker gives a cold start
    integer(ip) :: i, j
    real(rp) :: qm, x(1), y(1)
    qm = 0.0_rp
    !$cuf kernel do(2) <<<*,*>>> reduce(max:qm)
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        qm = max(qm, abs(qd(i, j)))
      enddo
    enddo
    x(1) = qm
    call comm_allreduce_max(1, x, y)
    if (y(1) >= 1.e29_rp) then
      udu = 0; udv = 0; gru = 0; grv = 0; qd = 0; xx = 0
      if (talk) write(*,'(1x,A)') 'IMPLICIT_FS: the restart record has no rates (written before the first'// &
           ' implicit step): cold start of the generalized-alpha memory'
      return
    endif
    ga_first = .false.
    if (talk) write(*,'(1x,A)') 'IMPLICIT_FS: generalized-alpha rates and delta read from the initial file'// &
         ' (exact restart)'
  end subroutine fs_rst_loaded

end module implicit_fs_gpu

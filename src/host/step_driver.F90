module step_driver
  ! Host time loop. Named steps match roms_bugfix/src/main.F90.
  ! nfast stays on the host. Each fast step is a CUDA graph of
  ! {step2d interiors, halo_fill} with host OBC between the sections.
  ! Physics kernels that are not written yet are explicit stubs.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use geom
  use halo_mod
  use step2d_mod, only: step2d_with_halo
  use prsgrd_mod, only: prsgrd, dump_pgf_col
  use pflx_mod, only: calc_pflx, calc_pressure_flux
  use advect, only: tracer_advect_fluxes
  use rho_eos_mod, only: rho_eos
  use pre_step3d_mod, only: pre_step3d, ncol_probe, icol_probe, jcol_probe
  use step3d_uv_mod, only: step3d_uv1, step3d_uv2
  use marbl_gpu, only: marbl_gpu_step
  use visc3d_mod, only: visc3d
  use step3d_t_mod, only: step3d_t
  use t3dmix_mod, only: t3dmix
  use passive_mod, only: ntp, passive_t3dmix, passive_exchange_mid
  use lmd_mod, only: lmd_vmix
  use frc_io, only: set_forces, frc_prefetch
  use fs_data, only: fs_bry_phase, fs_pot_phase
  use river_io, only: set_rivers
  use cdr_io, only: set_cdr, set_cdr_pred
  use cdr_lite_io, only: set_cdr_lite
  use bry_io, only: set_bry_all, dump_bry_north
  use tide_io, only: set_tides
  use run_timers, only: rt_dev_collect, wall_time
  use arch_mod, only: sync_arch
  use upscale_out, only: wrt_upscale
  use imex_igw, only: imex_on, imex_begin_step, imex_solve_stage, imex_stage_corr, &
                      imex_shift_levels, imex_tsmix_active
  use implicit_fs_gpu, only: fs_on, fs_step, fs_srcdiag, fs_src_diag
  implicit none
  private

  public :: run_geom_only
  public :: run_baroclinic_step
  public :: run_barotropic_only
  public :: run_prsgrd_only
  public :: run_advect_only
  public :: run_lmd_only
  public :: not_implemented

  logical, save :: clock_on = .false.
  real(8), save :: clock_start = 0.0d0
  real(8), save :: clock_last = 0.0d0
contains

  subroutine run_geom_only()
    real(rp) :: umis, vmis, vol

    iic = 0
    knew = 1
    nrhs = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call set_HUV()
    call omega()
    call check_mode_coupling(umis, vmis, vol)
    write(*,'(a,es14.6)') 'geom_only: volume m^3 = ', vol
    write(*,'(a,es14.6)') 'geom_only: max |Hz|   = ', maxval(abs(Hz))
    write(*,'(a,es14.6)') 'geom_only: max |We|   = ', maxval(abs(We))
  end subroutine run_geom_only

  subroutine run_baroclinic_step()
    integer(ip) :: n
    real(rp) :: umis, vmis, vol, umax, akvmax, hblmax
    logical :: info_step

    if (iic < ntstart) iic = ntstart
    call dry_skip_update(iic - ntstart)
    call hg_audit_step()
    call update_clock()
    call set_probe_windows()
    probe_on = (iic - ntstart < probe_steps)
    fast_probe_on = (iic - ntstart >= fast_probe_from .and. &
         iic - ntstart < fast_probe_to)
    call begin_predictor_indices()
    call step_timer_init()
    if (imex_on) call imex_begin_step()   ! IMEX_IGW: this step's weights; n-1 stores on the first

    ! --- predictor (nrhs = n, nnew = 3) ---
    frc_slot = 0
    call set_forces()
    call set_rivers(0)                    ! river data at time n (predictor)
    call set_cdr_pred()                   ! CDR volume releases at time n (predictor, first omegas)
    call dump_swflx_b('after pred set_forces')
    call step_mark('set_forces')
    frc_slot = 1
    call set_bry_all()
    call step_mark('set_bry')
    call set_tides(pot_phase=fs_pot_phase)
    call dump_bry_north('after tide', frc_modtime())
    call step_mark('set_tides')
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call step_mark('halo zeta')
    ! u, v and t in slot nrhs (= nstp) are the previous step's nnew, whose
    ! halos step3d_uv2's group and rho_eos(nnew)'s group exchanged after
    ! the last writes; nothing touches them in between, so no exchange here.
    call dump_vic('after halo v')
    call set_depth()
    call step_mark('set_depth')
    call dump_pgf_col('after set_depth')
    call dump_rim('after set_depth')
    call rho_eos(nrhs, 1)
    call step_mark('rho_eos')
    call dump_pgf_col('after rho_eos')
    call set_HUV()
    call step_mark('set_HUV')
    call dump_huv_probe('after set_HUV')
    call dump_p1_fun('set_HUV')
    call dump_rim('set_HUV')
    call omega()
    call dump_we_b('after pred omega')
    call step_mark('omega')
    call lmd_vmix(nstp)
    call step_mark('lmd_vmix')
    if (ntp > 0) call passive_exchange_mid()   ! previous step's nnew: east-west done, post north-south
    call prsgrd()
    call step_mark('prsgrd')
    call dump_col_probe('start', nrhs)
    call pre_step3d()
    call step_mark('pre_step3d')
    call dump_col_probe('after pre_step3d', nnew)
    call dump_uvt('after pre_step3d', nnew)
    call dump_vic('after pre_step3d')

    call set_HUV1()
    call step_mark('set_HUV1')
    call dump_huv_probe('after set_HUV1')
    call dump_p1_fun('set_HUV1')
    call dump_rim('set_HUV1')
    call dump_uvt('after set_HUV1', nnew)
    call dump_vic('after set_HUV1')
    call begin_corrector_indices()
    ! CPU: corr omega uses the still-current (time-n) swflx, then
    ! set_forces interpolates n+1/2, then bry/tides, then prsgrd/uv1.
    call dump_swflx_b('before corr omega')
    call omega()
    call dump_we_b('after corr omega')
    call dump_swflx_b('after corr omega')
    call rho_eos(nrhs, 2)
    frc_slot = 1
    call set_forces()
    call set_rivers(1)                    ! river data at n+1/2 (fast loop, corrector)
    call set_cdr()                        ! CDR release fluxes at n+1/2 (corrector)
    call set_cdr_lite()                   ! CDR_LITE surface fluxes at n+1/2 (corrector)
    call dump_swflx_b('after corr set_forces')
    call lmd_vmix(nrhs)
    ! Boundary data for n+1 (CPU main.F90 since 69a23382; before it read
    ! n+3/2). The boundary tidal harmonics are added to the 2D boundary
    ! arrays, which the split fast loop reads (uv2dbc). The fast loop
    ! advances the barotropic mode from n to n+1 with these values held
    ! fixed, so they are evaluated at the centre of that interval, n+1/2;
    ! at n+1 the boundary-forced tide leads by about omega*dt/2 (CPU
    ! ab490df8). The potential is at n+1/2 as well, where prsgrd uses it
    ! (the implicit free surface moves it: fs_pot_phase). With the
    ! implicit free surface (fs_on) the same arrays are read by the
    ! implicit step, whose boundary face takes data at n+alpha_f, so fs_on
    ! keeps the harmonics at n+1 (CPU 6a2d56da; fs_bry_phase = 1); with
    ! fs_nsub sub-steps, which hold these values, the mean of the
    ! sub-steps' end times, (fs_nsub+1)/(2 fs_nsub).
    frc_slot = 2
    call set_bry_all()
    call set_tides(merge(fs_bry_phase, 0.5_rp, fs_on), fs_pot_phase)
    call dump_bry_north('after tide', frc_modtime())
    call step_mark('corrector forces')
    call dump_vic('before uv1')
    call dump_p1_fun('before_uv1')
    call dump_rim('before_uv1')
    call prsgrd()
    call step_mark('corrector rhs')
    if (calc_pflx) then
      ! the CPU's calc_pressure_flux (prsgrd, corrector stage, DIAGNOSTICS)
      call calc_pressure_flux()
      call step_mark('calc_pflx')
    endif
    call step3d_uv1()
    if (imex_on) then
      ! IMEX_IGW: elliptic correction of u, v(nnew) (Hz-weighted); rufrc
      ! keeps the explicit rhs; then the halos of the corrected faces
      call imex_solve_stage(imex_stage_corr)
      call hg_begin()
      call hg_add(F_U, nnew)
      call hg_add(F_V, nnew)
      call hg_end()
    endif
    call visc3d()
    call step_mark('step3d_uv1+visc')
    if (ntp > 0) call passive_exchange_mid()   ! slot 3 of the passives: east-west done, post north-south
    call dump_uvt('after step3d_uv1', nnew)
    call dump_bt2d_probe('after step3d_uv1+visc')
    call dump_rvS('after step3d_uv1+visc')
    call dump_rufrc_build('after visc')
    call dump_ztd_pt('after step3d_uv1+visc')
    call dump_rim('after step3d_uv1+visc')

    if (fs_on) then
      ! implicit free surface (implicit_fs.F90, FS_SPEC.md): one step
      ! replaces the fast loop; kstp then points at the new level
      call cycle_fast_index()
      call fs_step()
      kstp = knew
    else
    do n = 1, nfast
      iif = n
      call cycle_fast_index()
      call step2d_with_halo(iif)
      if (n == 1) call dump_bt2d_probe('after iif=1')
      if (n == nfast) then
        call dump_bt2d_probe('after last fast')
        call dump_ztd_pt('after last fast')
        call dump_rim('after last fast')
      endif
      call dump_fast_probe()
    enddo
    endif
    if (fs_srcdiag) call fs_src_diag()   ! IMPLICIT_FS fs_verbose >= 2: FS_SRC line (both fs_mode values)
    call step_mark('step2d')
    call dump_ic_jet_2d('after step2d')

    call step3d_uv2()
    call step_mark('step3d_uv2')
    call dump_huv_probe('after step3d_uv2')
    call dump_p1_fun('after step3d_uv2')
    call dump_uvt('after step3d_uv2', nnew)
    call dump_rim('after step3d_uv2')
    call omega()
    call dump_we_b('after uv2 omega')
    call dump_swflx_b('after uv2 omega')
    call step3d_t()
    ! Read the next forcing records ahead while the passive corrector
    ! kernels queued by step3d_t run on the device (before marbl_gpu_step,
    ! which drains the device on some calls for its solver report).
    call frc_prefetch()
    ! upscaling: this step's open-boundary fluxes (slot nrhs, unchanged by
    ! MARBL), as the CPU's wrt_upscale at the end of its tracer loop
    call wrt_upscale()
    call marbl_gpu_step()
    call dump_max_t('after step3d_t')
    call dump_col_probe('after step3d_t', nnew)
    call dump_rim('after step3d_t')
    if (imex_tsmix_active()) then
      ! IMEX_IGW: lateral tracer mixing from t^n (nstp), not
      ! the predictor state (nrhs = 3)
      n = nrhs
      nrhs = nstp
      call t3dmix()
      if (ntp > 0) call passive_t3dmix()
      nrhs = n
    else
    call t3dmix()
    if (ntp > 0) call passive_t3dmix()
    endif
    call dump_col_probe('after t3dmix', nnew)
    call rho_eos(nnew, 3)
    call step_mark('step3d_t')
    ! Step diagnostics are device reductions, every ninfo steps.
    info_step = (mod(iic - ntstart, ninfo_steps) == 0)
    if (info_step) then
      call check_mode_coupling(umis, vmis, vol)
      write(*,'(a,3es14.6)') 'coupling: u_mis, v_mis, vol = ', umis, vmis, vol
      call step_maxima(umax, akvmax, hblmax)
      write(*,'(a,es14.6)') 'step: max |u| = ', umax
    endif
    call dump_ztd_pt('end of step')
    call dump_rim('end of step')
    call dump_ztd_stats()
    call dump_max_t('end of step')
    if (info_step) then
      write(*,'(a,es14.6)') 'step: max Akv = ', akvmax
      write(*,'(a,es14.6)') 'step: max hbls = ', hblmax
    endif
    if (imex_on) call imex_shift_levels()   ! IMEX_IGW: level n becomes n-1
    call step_mark('baroclinic step')
    call rt_dev_collect()        ! the device is drained here
  end subroutine run_baroclinic_step

  subroutine run_barotropic_only()
    integer(ip) :: n
    real(rp) :: vol0, vol1

    iic = 1
    knew = 1
    kstp = 1
    call hg_begin()
    call hg_add(F_ZETA, 1_ip)
    call hg_add(F_UBAR, 1_ip)
    call hg_add(F_VBAR, 1_ip)
    call hg_end()
    call set_depth()

    vol0 = sum((h(1:decomp%nx, 1:decomp%ny) + zeta(1:decomp%nx, 1:decomp%ny, 1)) / &
               (pm(1:decomp%nx, 1:decomp%ny) * pn(1:decomp%nx, 1:decomp%ny)))

    do n = 1, nfast
      iif = n
      call cycle_fast_index()
      call step2d_with_halo(iif)
    enddo

    vol1 = sum((h(1:decomp%nx, 1:decomp%ny) + zeta(1:decomp%nx, 1:decomp%ny, knew)) / &
               (pm(1:decomp%nx, 1:decomp%ny) * pn(1:decomp%nx, 1:decomp%ny)))
    write(*,'(a,i4)') 'barotropic: nfast = ', nfast
    write(*,'(a,2es14.6)') 'barotropic: vol0, vol1 = ', vol0, vol1
    write(*,'(a,es14.6)') 'barotropic: max |zeta| = ', maxval(abs(zeta(1:decomp%nx,1:decomp%ny,knew)))
    write(*,'(a,es14.6)') 'barotropic: max |ubar| = ', maxval(abs(ubar(1:decomp%nx+1,1:decomp%ny,knew)))
  end subroutine run_barotropic_only

  subroutine run_advect_only()
    integer(ip) :: tidx, itrc

    tidx = 1
    itrc = 1
    iic = 0
    knew = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call set_HUV()
    call omega()
    ! Ideal box is at rest, so volume fluxes are 0. Impose a uniform
    ! transport so the kernel has something to interpolate.
    FlxU = 1.0_rp
    FlxV = 1.0_rp
    We = 1.0_rp
    call tracer_advect_fluxes(tidx, itrc, .true.)
    write(*,'(a,es14.6)') 'advect C4: max |FX| = ', maxval(abs(wrk3a(1:decomp%nx+1,1:decomp%ny,:)))
    write(*,'(a,es14.6)') 'advect C4: max |FE| = ', maxval(abs(wrk3b(1:decomp%nx,1:decomp%ny+1,:)))
    write(*,'(a,es14.6)') 'advect C4: max |Fz| = ', maxval(abs(col_wc(1:decomp%nx,1:decomp%ny,1:decomp%nz-1)))
    call tracer_advect_fluxes(tidx, itrc, .false.)
    write(*,'(a,es14.6)') 'advect U3: max |Fz| = ', maxval(abs(col_wc(1:decomp%nx,1:decomp%ny,1:decomp%nz-1)))
  end subroutine run_advect_only

  subroutine run_lmd_only()
    iic = 1
    knew = 1
    nstp = 1
    nrhs = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call rho_eos(nstp)
    call lmd_vmix(nstp)
    write(*,'(a,es14.6)') 'lmd: max Akv  = ', maxval(Akv(1:decomp%nx,1:decomp%ny,:))
    write(*,'(a,es14.6)') 'lmd: max AktT = ', maxval(Akt(1:decomp%nx,1:decomp%ny,:,itemp))
    write(*,'(a,es14.6)') 'lmd: max hbls = ', maxval(hbls(1:decomp%nx,1:decomp%ny))
    write(*,'(a,es14.6)') 'lmd: max hbbl = ', maxval(hbbl(1:decomp%nx,1:decomp%ny))
    write(*,'(a,es14.6)') 'lmd: min swr  = ', minval(swr_frac(1:decomp%nx,1:decomp%ny,0))
  end subroutine run_lmd_only

  subroutine run_prsgrd_only()
    iic = 0
    knew = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call prsgrd()
    write(*,'(a,es14.6)') 'prsgrd: max |ru| = ', maxval(abs(ru(1:decomp%nx+1,1:decomp%ny,:)))
    write(*,'(a,es14.6)') 'prsgrd: max |rv| = ', maxval(abs(rv(1:decomp%nx,1:decomp%ny+1,:)))
    write(*,'(a,es14.6)') 'prsgrd: max |P|  = ', maxval(abs(P(1:decomp%nx,1:decomp%ny,:)))
  end subroutine run_prsgrd_only

  subroutine dump_max_t(tag)
    ! After Thomas, t(:,:,:,nnew) is temperature. Print the hot cell
    ! together with the 1/Hz flux pieces that can heat a 3 cm box.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, im, jm, km, ih, jh, kh
    real(rp) :: tmax, hmin

    if (.not. probe_on) return
    call sync_arch()
    tmax = 0.0_rp
    hmin = 1.0e30_rp
    im = 0; jm = 0; km = 0
    ih = 0; jh = 0; kh = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (Hz(i, j, k) < hmin) then
            hmin = Hz(i, j, k)
            ih = i; jh = j; kh = k
          endif
          if (abs(t(i, j, k, nnew, itemp)) > tmax) then
            tmax = abs(t(i, j, k, nnew, itemp))
            im = i; jm = j; km = k
          endif
        enddo
      enddo
    enddo
    write(*,'(a,a,a,es14.6,a,3i6)') 'diag: ', trim(tag), '  max |t| = ', tmax, &
         ' at i,j,k =', im, jm, km
    if (im > 0) call dump_t_cell(tag, 'hot', im, jm, km)
    if (ih > 0 .and. (ih /= im .or. jh /= jm .or. kh /= km)) then
      write(*,'(a,a,a,es14.6,a,3i6)') 'diag: ', trim(tag), '  min Hz = ', hmin, &
           ' at i,j,k =', ih, jh, kh
      call dump_t_cell(tag, 'thin', ih, jh, kh)
    endif
    flush(6)
  end subroutine dump_max_t

  subroutine dump_t_cell(tag, role, i, j, k)
    character(len=*), intent(in) :: tag, role
    integer(ip), intent(in) :: i, j, k
    integer(ip) :: kg
    real(rp) :: Tval, hzv, Dcol, dTst, dTrain, dTswr

    if (.not. probe_on) return
    Tval = t(i, j, k, nnew, itemp)
    hzv = max(Hz(i, j, k), 1.0e-30_rp)
    Dcol = h(i, j) + zeta(i, j, knew)
    dTst = dt * stflx(i, j, itemp) / hzv
    if (have_bulk) then
      dTrain = dt * swflx(i, j) * tair(i, j) / hzv
    else
      dTrain = dt * swflx(i, j) * Tval / hzv
    endif
    dTswr = dt * srflx(i, j) * swr_frac(i, j, 0) / hzv
    kg = max(k - 1, 0)
    write(*,'(a,a,a,a,a,3i6)') 'diag: ', trim(tag), '  ', trim(role), ' i,j,k=', i, j, k
    write(*,'(a,a,a,a,a,5es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' T,Hz,h,zeta,D=', Tval, Hz(i, j, k), h(i, j), zeta(i, j, knew), Dcol
    write(*,'(a,a,a,a,a,5es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' stflx,srflx,swflx,swr0,tair=', stflx(i, j, itemp), srflx(i, j), &
         swflx(i, j), swr_frac(i, j, 0), tair(i, j)
    write(*,'(a,a,a,a,a,4es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' dT(stflx,rain,swr0)/step,ghat=', dTst, dTrain, dTswr, ghat(i, j, kg)
    write(*,'(a,a,a,a,a,4es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' Tbot,Tsrf,hbls,Akt=', t(i, j, 1, nnew, itemp), &
         t(i, j, decomp%nz, nnew, itemp), hbls(i, j), Akt(i, j, max(k - 1, 1), itemp)
  end subroutine dump_t_cell

  subroutine dump_uvt(tag, slot)
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: slot
    integer(ip) :: n, i, j, k, im, jm, km, iu, ju, ku
    real(rp) :: tmax, umax, hzu
    if (.not. probe_on) return
    n = slot
    call sync_arch()
    write(*,'(a,a,a,i0,a,es14.6,a,es14.6)') 'diag: ', trim(tag), '  slot=', n, &
         '  max |u| = ', maxval(abs(u(1:decomp%nx+1,1:decomp%ny,:,n)), &
         mask=spread(umask(1:decomp%nx+1,1:decomp%ny) > 0.5_rp, 3, decomp%nz)), &
         '  max |t| = ', maxval(abs(t(1:decomp%nx,1:decomp%ny,:,n,1)), &
         mask=spread(rmask(1:decomp%nx,1:decomp%ny) > 0.5_rp, 3, decomp%nz))
    umax = 0.0_rp
    iu = 0; ju = 0; ku = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx + 1
        if (umask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (abs(u(i, j, k, n)) > umax) then
            umax = abs(u(i, j, k, n))
            iu = i; ju = j; ku = k
          endif
        enddo
      enddo
    enddo
    if (iu > 0) then
      hzu = 0.5_rp * (Hz(iu, ju, ku) + Hz(iu - 1, ju, ku))
      write(*,'(a,a,a,3i6,a,3es14.6)') 'diag: ', trim(tag), '  u at i,j,k=', iu, ju, ku, &
           '  Hz_u,u,u/Hz_u=', hzu, u(iu, ju, ku, n), &
           u(iu, ju, ku, n) / max(hzu, 1.0e-30_rp)
    endif
    if (ic_u_i > 0) then
      hzu = 0.5_rp * (Hz(ic_u_i, ic_u_j, ic_u_k) + Hz(ic_u_i - 1, ic_u_j, ic_u_k))
      write(*,'(a,a,a,3i6,a,3es14.6)') 'diag: ', trim(tag), '  IC-jet u at i,j,k=', &
           ic_u_i, ic_u_j, ic_u_k, '  Hz_u,u,u/Hz_u=', hzu, &
           u(ic_u_i, ic_u_j, ic_u_k, n), &
           u(ic_u_i, ic_u_j, ic_u_k, n) / max(hzu, 1.0e-30_rp)
    endif
    tmax = 0.0_rp
    im = 0; jm = 0; km = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (abs(t(i, j, k, n, 1)) > tmax) then
            tmax = abs(t(i, j, k, n, 1))
            im = i; jm = j; km = k
          endif
        enddo
      enddo
    enddo
    if (im > 0) then
      write(*,'(a,a,a,3i6,a,3es14.6)') 'diag: ', trim(tag), '  t at i,j,k=', im, jm, km, &
           '  Hz,Hz_fwd,t/Hz=', Hz(im, jm, km), Hz_fwd(im, jm, km), &
           t(im, jm, km, n, 1) / max(Hz(im, jm, km), 1.0e-30_rp)
    endif
    flush(6)
    call step_timer_lap()
  end subroutine dump_uvt

  subroutine dump_col_probe(tag, slot)
    ! Same "diag: col" lines as roms_bugfix smoke_dump_col.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: slot
    integer(ip) :: n, i, j, k
    if (.not. probe_on) return
    call sync_arch()
    do n = 1, ncol_probe
      i = icol_probe(n)
      j = jcol_probe(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(*,'(a,a,a,i1,a,i6,8es22.14)') 'diag: colx ', trim(tag), ' P', n, &
           ' iic,hbls,hbbl,r_D,u1W,u1E,v1S,v1N(slot),u1W(nstp)=', iic, &
           hbls(i, j), hbbl(i, j), r_D(i, j), u(i, j, 1, slot), &
           u(i + 1, j, 1, slot), v(i, j, 1, slot), v(i, j + 1, 1, slot), &
           u(i, j, 1, nstp)
      do k = 1, decomp%nz
        write(*,'(a,a,a,i1,a,i6,i4,4es22.14)') 'diag: col ', trim(tag), ' P', n, &
             ' iic,k,T,S,AktT,Akv=', iic, k, t(i, j, k, slot, itemp), &
             t(i, j, k, slot, isalt), Akt(i, j, k, itemp), Akv(i, j, k)
      enddo
      do k = 1, decomp%nz
        if (k /= 1 .and. k /= 2 .and. k /= 3 .and. k /= 5 .and. k /= 10) cycle
        write(*,'(a,a,a,i1,a,i6,i4,4es22.14)') 'diag: cold ', trim(tag), ' P', n, &
             ' iic,k,Hz,z_r,We,Wi=', iic, k, Hz(i, j, k), z_r(i, j, k), &
             We(i, j, k), Wi(i, j, k)
      enddo
    enddo
    flush(6)
  end subroutine dump_col_probe

  subroutine dump_ic_jet_2d(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k
    real(rp) :: Dcol, dn, du_bar
    if (.not. probe_on) return
    if (ic_u_i <= 0) return
    call sync_arch()
    i = ic_u_i
    j = ic_u_j
    Dcol = 0.0_rp
    do k = 1, decomp%nz
      Dcol = Dcol + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
    enddo
    dn = dn_u(i, j)
    du_bar = DU_avg1(i, j) / max(Dcol * dn, 1.0e-30_rp)
    write(*,'(a,a,a,2i6,a,4es14.6)') 'diag: ', trim(tag), '  IC-jet 2D i,j=', i, j, &
         '  Dcol,ubar,DU/D,DU_avg1=', Dcol, ubar(i, j, knew), du_bar, DU_avg1(i, j)
    flush(6)
    call step_timer_lap()
  end subroutine dump_ic_jet_2d

  subroutine dump_ztd_pt(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: n, i, j
    if (.not. probe_on) return
    call sync_arch()
    do n = 1, nprb_pt
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(*,'(a,i1,a,a,a,i8,a,6es22.14)') 'diag: ztdP', n, '  ', trim(tag), &
           '  iic=', iic, '  zeta,Zt_avg1,Hz1,zr1,T1,We1=', &
           zeta(i, j, knew), zt_avg1(i, j), Hz(i, j, 1), z_r(i, j, 1), &
           t(i, j, 1, nnew, itemp), We(i, j, 1)
    enddo
    flush(6)
  end subroutine dump_ztd_pt

  subroutine dump_ztd_stats()
    integer(ip) :: i, j, iz, jz, iT, jT, ir, jr
    real(rp) :: zabs, tabs, rabs, zmax, Tmax, rmax, z2, nwet
    if (.not. probe_on) return
    call sync_arch()
    zmax = 0.0_rp; Tmax = 0.0_rp; rmax = 0.0_rp; z2 = 0.0_rp; nwet = 0.0_rp
    iz = 0; jz = 0; iT = 0; jT = 0; ir = 0; jr = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        zabs = abs(zeta(i, j, knew))
        if (zabs > zmax) then
          zmax = zabs
          iz = i; jz = j
        endif
        z2 = z2 + zeta(i, j, knew)**2
        nwet = nwet + 1.0_rp
        if (h(i, j) > 2000.0_rp) then
          tabs = abs(t(i, j, 1, nnew, itemp))
          if (tabs > Tmax) then
            Tmax = tabs
            iT = i; jT = j
          endif
        endif
        rabs = abs(rufrc(i, j))
        if (rabs > rmax) then
          rmax = rabs
          ir = i; jr = j
        endif
      enddo
    enddo
    write(*,'(a,i8,a,es26.17,a,2i6,a,es26.17,a,es26.17,a,2i6,a,es26.17,a,2i6)') &
         'diag: ztd  iic=', iic, '  max|zeta|=', zmax, '  at i,j=', iz, jz, &
         '  rms=', sqrt(z2 / max(nwet, 1.0_rp)), &
         '  max|Tdeep|=', Tmax, '  at i,j=', iT, jT, &
         '  max|rufrc|=', rmax, '  at i,j=', ir, jr
    flush(6)
  end subroutine dump_ztd_stats

  subroutine set_probe_windows()
    ! Diagnostic probes are off unless ROMS_GPU_PROBES is set to a non-zero
    ! value (unset, empty or 0: off). The off state is the time_state
    ! default, so it holds from the start of the run.
    logical, save :: checked = .false.
    character(len=16) :: val
    integer :: stat, ios, ival
    if (checked) return
    checked = .true.
    call get_environment_variable('ROMS_GPU_PROBES', val, status=stat)
    if (stat > 0 .or. len_trim(val) == 0) return
    read(val, *, iostat=ios) ival
    if (ios == 0 .and. ival == 0) return
    probe_steps = 24
    fast_probe_from = 4
    fast_probe_to = 8
    kpp_probe_from = 0
    kpp_probe_to = 23
    if (decomp%rank == 0) write(*,'(3a)') 'step_driver: diagnostic probes on (ROMS_GPU_PROBES=', &
         trim(val), ')'
    call set_probe_points('rho', 0, 0, iprb_pt_g, jprb_pt_g, iprb_pt, jprb_pt)
    call set_probe_points('u', 1, 0, iprb_u_g, jprb_u_g, iprb_u, jprb_u)
    call set_probe_points('v', 0, 1, iprb_v_g, jprb_v_g, iprb_v, jprb_v)
  end subroutine set_probe_windows

  subroutine set_probe_points(stag, su, sv, ig, jg, il, jl)
    ! Rank-local probe points from the global ones: the local index where
    ! the point is inside the grid and owned by this rank, else prb_off.
    ! su = 1 for U-faces (global i = 1..gnx+1, the east boundary face owned
    ! by the east-edge tile), sv = 1 for V-faces (j = 1..gny+1). Rank 0
    ! names each point outside the grid once.
    character(len=*), intent(in) :: stag
    integer(ip), intent(in) :: su, sv, ig(:), jg(:)
    integer(ip), intent(out) :: il(:), jl(:)
    integer(ip) :: n, gimax, gjmax, iimax, jjmax, li, lj
    gimax = decomp%gnx
    gjmax = decomp%gny
    iimax = decomp%nx
    jjmax = decomp%ny
    if (su == 1 .and. .not. decomp%periodic_x) gimax = gimax + 1
    if (sv == 1 .and. .not. decomp%periodic_y) gjmax = gjmax + 1
    if (su == 1 .and. decomp%edge_e) iimax = iimax + 1
    if (sv == 1 .and. decomp%edge_n) jjmax = jjmax + 1
    do n = 1, size(ig)
      il(n) = prb_off
      jl(n) = prb_off
      if (ig(n) < 1 .or. ig(n) > gimax .or. jg(n) < 1 .or. jg(n) > gjmax) then
        if (decomp%rank == 0) write(*,'(3a,i2,a,2i6,a,i6,a,i6,a)') &
             'step_driver: probe point ', stag, ' P', n, ' (i,j =', ig(n), jg(n), &
             ') is outside the', gimax, ' x', gjmax, ' grid: skipped'
        cycle
      endif
      li = ig(n) - decomp%i0 + 1
      lj = jg(n) - decomp%j0 + 1
      if (li < 1 .or. li > iimax .or. lj < 1 .or. lj > jjmax) cycle
      il(n) = li
      jl(n) = lj
    enddo
  end subroutine set_probe_points

  subroutine step_timer_init()
    clock_start = wall_time()
    clock_last = clock_start
    clock_on = .true.
  end subroutine step_timer_init

  subroutine step_timer_lap()
    if (.not. clock_on) return
    clock_last = wall_time()
  end subroutine step_timer_lap

  subroutine step_mark(msg)
    ! Phase stamps: each one drains the device (cudaDeviceSynchronize) and
    ! prints; by default only the end-of-step stamp is kept (its cumul is
    ! the step time). ROMS_GPU_STEP_MARKS=1 restores every phase stamp.
    character(len=*), intent(in) :: msg
    real(8) :: now
    real(rp) :: dt_sec, tot_sec
    logical, save :: checked = .false., all_marks = .false.
    character(len=16) :: val
    integer :: stat

    if (.not. checked) then
      checked = .true.
      call get_environment_variable('ROMS_GPU_STEP_MARKS', val, status=stat)
      all_marks = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) /= '0')
      if (.not. all_marks) write(*,'(a)') 'step_driver: phase stamps off (ROMS_GPU_STEP_MARKS=1 to print them)'
    endif
    if (.not. all_marks .and. msg /= 'baroclinic step') return
    call sync_arch()
    if (.not. clock_on) then
      write(*,'(a,a)') 'step: ', trim(msg)
    else
      now = wall_time()
      dt_sec = real(now - clock_last, rp)
      tot_sec = real(now - clock_start, rp)
      write(*,'(a,a,a,f10.4,a,f10.4,a)') 'step: ', trim(msg), &
           '  dt =', dt_sec, ' s  cumul =', tot_sec, ' s'
      clock_last = now
    endif
    flush(6)
  end subroutine step_mark

  subroutine not_implemented(name)
    character(len=*), intent(in) :: name
    write(*,'(a,a)') 'stub (not yet implemented): ', trim(name)
    error stop 'step_driver: refused to run an unimplemented kernel'
  end subroutine not_implemented

end module step_driver

#include "mem_attr.inc"
module step3d_uv_mod
  ! Corrector UV: apply dt*ru/rv with implicit Akv (uv1), then couple
  ! the barotropic increment and rebuild FlxU/FlxV (uv2).
  !
  ! Matches CPU step3d_uv2_tile: first couple interior (istrU/jstrV),
  ! apply OBC, then couple the extended range and rebuild fluxes.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use halo_mod
  use uv_rhs
  use obc, only: u3dbc, v3dbc
  use rivers_mod, only: river_on, rivers_uv, RIV_COR
  use arch_mod, only: sync_arch
  use geom, only: dump_rufrc_build, dump_rv3d
  use imex_igw, only: imex_on, wflx_corr, u_nm1, v_nm1, imex_obc_faces, imex_stage_corr
  implicit none
  private
  public :: step3d_uv1
  public :: step3d_uv2

  ! P4 / IC-jet probe values captured inside the uv2 coupling kernels and
  ! printed by the host afterwards (see step3d_uv2). Layout:
  !  1-3 couple1 u uint,Dcol,mis   4-7 u(1,50,99,nz) before the shift
  !  8-10 couple1 v uint,Dcol,mis 11-14 v(1,50,99,nz) before shift/zero
  ! 15-19 IC-jet Hz_u,u,uint/D,DU/D,mis
  ! 20-24 couple2 u Dcol,uint,flx_int,mis_u,mis_f   25-29 the same for v
  ! 30-34 flags (couple1 u, couple1 v, IC-jet, couple2 u, couple2 v)
  integer, parameter :: NCPL_PRB = 40
  real(rp), WORK_ATTR allocatable :: cpl_prb(:)

contains

  subroutine step3d_uv1()
    integer(ip) :: i, j, k, nx, ny, nz, n_new, i_u0, i_u1, j_v0, lim
    real(rp) :: cff, dt_loc
    real(rp) :: fck, wck, cfk, fcm, wcm, cfm, dc0, dck, rf
    logical :: keep

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_new = nnew
    dt_loc = dt
    i_u0 = 1
    if (decomp%edge_w) i_u0 = 2
    i_u1 = nx
    j_v0 = 1
    if (decomp%edge_s) j_v0 = 2
    ! u, v(nrhs), Hz, Wi, We are current (pre_step3d, set_depth, omega);
    ! only u, v(nnew) need their halo here (2026-10-08 audit).
    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_end()

    call dump_ru_probe('after prsgrd', .false.)
    call dump_rv3d('after prsgrd', 0.0_rp)
    call dump_rhs_b('after prsgrd')
    call add_horiz_uv_rhs_all(.true.)
    call dump_ru_probe('after horiz adv+cor', .false.)
    call dump_rv3d('after horiz adv+cor', 0.0_rp)
    call dump_vn3('after corr horiz')
    call dump_rhs_b('after horiz adv+cor')
    call add_vert_uv_rhs(1, ny)
    call dump_ru_probe('after vert adv', .false.)
    call dump_rv3d('after vert adv', 0.0_rp)
    call dump_vn3('after corr vert')
    call dump_rhs_b('after vert adv')
    call dump_uv1_b('before impl')
    keep = probe_on
    lim = dry_lim(G_MOM)

    ! Column Thomas has a k-recurrence: one thread per column, k serial.
    ! col_dc(0), the back-substituted col_dc and rufrc/rvfrc stay in
    ! registers (stored on probe steps, keep, for dump_thom_b); the forward
    ! col_dc/col_cf are stored for the back-substitution. Same operations.
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dt_loc * 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j))
        if (keep) col_dc(i, j, 0) = dc0
        fck = 2.0_rp * dt_loc * (Akv(i, j, nz - 1) + Akv(i - 1, j, nz - 1)) / &
             (Hz(i, j, nz) + Hz(i - 1, j, nz) + Hz(i, j, nz - 1) + Hz(i - 1, j, nz - 1))
        wck = dc0 * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i - 1, j, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz(i, j, nz) + Hz(i - 1, j, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        dck = cff * (u(i, j, nz, n_new) + dc0 * ru(i, j, nz) + dt_loc * sustr(i, j))
        col_dc(i, j, nz) = dck
        if (keep) then
          col_fc(i, j, nz - 1) = fck
          col_wc(i, j, nz - 1) = wck
        endif
        do k = nz - 1, 2, -1
          fcm = 2.0_rp * dt_loc * (Akv(i, j, k - 1) + Akv(i - 1, j, k - 1)) / &
               (Hz(i, j, k) + Hz(i - 1, j, k) + Hz(i, j, k - 1) + Hz(i - 1, j, k - 1))
          wcm = dc0 * 0.5_rp * (Wi(i, j, k - 1) + Wi(i - 1, j, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          dck = cff * (u(i, j, k, n_new) + dc0 * ru(i, j, k) + &
                            dck * (fck - min(wck, 0.0_rp)))
          col_dc(i, j, k) = dck
          if (keep) then
            col_fc(i, j, k - 1) = fcm
            col_wc(i, j, k - 1) = wcm
          endif
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        dck = (u(i, j, 1, n_new) + dc0 * ru(i, j, 1) + &
                    dck * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz(i, j, 1) + Hz(i - 1, j, 1)) + 0.5_rp * dt_loc * (r_D(i, j) + r_D(i - 1, j)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        if (keep) col_dc(i, j, 1) = dck
        ! CPU step3d_uv1 writes u_new unmasked; umask is applied in set_HUV1.
        u(i, j, 1, n_new) = dck * 0.5_rp * (Hz(i, j, 1) + Hz(i - 1, j, 1))
        rf = apply_mask(ru(i, j, 1) + dm_u(i, j) * dn_u(i, j) * (sustr(i, j) - &
                      0.5_rp * (r_D(i - 1, j) + r_D(i, j)) * dck), umask(i, j))
        do k = 2, nz
          dck = col_dc(i, j, k) + col_cf(i, j, k - 1) * dck
          if (keep) col_dc(i, j, k) = dck
          u(i, j, k, n_new) = dck * 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
          rf = apply_mask(rf + ru(i, j, k), umask(i, j))
        enddo
        rufrc(i, j) = rf
      enddo
    enddo
    call dump_thom_b('u')

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        col_dc(i, j, 0) = dt_loc * 0.25_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1))
        fck = 2.0_rp * dt_loc * (Akv(i, j, nz - 1) + Akv(i, j - 1, nz - 1)) / &
             (Hz(i, j, nz) + Hz(i, j - 1, nz) + Hz(i, j, nz - 1) + Hz(i, j - 1, nz - 1))
        wck = col_dc(i, j, 0) * 0.5_rp * (Wi(i, j, nz - 1) + Wi(i, j - 1, nz - 1))
        cff = 1.0_rp / (0.5_rp * (Hz(i, j, nz) + Hz(i, j - 1, nz)) + &
             fck - min(wck, 0.0_rp))
        cfk = cff * (fck + max(wck, 0.0_rp))
        col_cf(i, j, nz - 1) = cfk
        col_dc(i, j, nz) = cff * (v(i, j, nz, n_new) + col_dc(i, j, 0) * rv(i, j, nz) + dt_loc * svstr(i, j))
        if (keep) then
          col_fc(i, j, nz - 1) = fck
          col_wc(i, j, nz - 1) = wck
        endif
        do k = nz - 1, 2, -1
          fcm = 2.0_rp * dt_loc * (Akv(i, j, k - 1) + Akv(i, j - 1, k - 1)) / &
               (Hz(i, j, k) + Hz(i, j - 1, k) + Hz(i, j, k - 1) + Hz(i, j - 1, k - 1))
          wcm = col_dc(i, j, 0) * 0.5_rp * (Wi(i, j, k - 1) + Wi(i, j - 1, k - 1))
          cff = 1.0_rp / (0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) + &
                          fcm - min(wcm, 0.0_rp) + &
                          fck + max(wck, 0.0_rp) - &
                          cfk * (fck - min(wck, 0.0_rp)))
          cfm = cff * (fcm + max(wcm, 0.0_rp))
          col_cf(i, j, k - 1) = cfm
          col_dc(i, j, k) = cff * (v(i, j, k, n_new) + col_dc(i, j, 0) * rv(i, j, k) + &
                            col_dc(i, j, k + 1) * (fck - min(wck, 0.0_rp)))
          if (keep) then
            col_fc(i, j, k - 1) = fcm
            col_wc(i, j, k - 1) = wcm
          endif
          fck = fcm; wck = wcm; cfk = cfm
        enddo
        col_dc(i, j, 1) = (v(i, j, 1, n_new) + col_dc(i, j, 0) * rv(i, j, 1) + &
                    col_dc(i, j, 2) * (fck - min(wck, 0.0_rp))) / &
             (0.5_rp * (Hz(i, j, 1) + Hz(i, j - 1, 1)) + 0.5_rp * dt_loc * (r_D(i, j) + r_D(i, j - 1)) + &
              fck + max(wck, 0.0_rp) - &
              cfk * (fck - min(wck, 0.0_rp)))
        v(i, j, 1, n_new) = col_dc(i, j, 1) * 0.5_rp * (Hz(i, j, 1) + Hz(i, j - 1, 1))
        rvfrc(i, j) = apply_mask(rv(i, j, 1) + dm_v(i, j) * dn_v(i, j) * (svstr(i, j) - &
                      0.5_rp * (r_D(i, j - 1) + r_D(i, j)) * col_dc(i, j, 1)), vmask(i, j))
        do k = 2, nz
          col_dc(i, j, k) = col_dc(i, j, k) + col_cf(i, j, k - 1) * col_dc(i, j, k - 1)
          v(i, j, k, n_new) = col_dc(i, j, k) * 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
          rvfrc(i, j) = apply_mask(rvfrc(i, j) + rv(i, j, k), vmask(i, j))
        enddo
      enddo
    enddo
    call dump_thom_b('v')

    call dump_ru_probe('after rufrc', .true.)
    call dump_rv3d_after_rufrc()
    call dump_rufrc_build('after rufrc')
    call dump_vn3('after corr impl')
    call dump_uv1_b('after impl')

    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_add(F_RUFRC)
    call hg_add(F_RVFRC)
    call hg_end()
    call dump_uv2_b('after uv1')
  end subroutine step3d_uv1

  subroutine dump_ru_probe(tag, with_frc)
    ! Depth sums and top/bottom levels of ru (west U-face) and rv (south
    ! V-face) at each probe point, one line per stage of the 3-D RHS.
    character(len=*), intent(in) :: tag
    logical, intent(in) :: with_frc
    integer(ip) :: n, i, j, nz

    if (.not. probe_on) return
    call sync_arch()
    nz = decomp%nz
    do n = 1, nprb_pt
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(*,'(a,i1,a,a,a,i8,a,8es26.17)') 'diag: ruP', n, '  ', trim(tag), '  iic=', iic, &
           '  sum ruW, ruW k=1,nz-1,nz, sum rvS, rvS k=1,nz-1,nz=', &
           sum(ru(i, j, 1:nz)), ru(i, j, 1), ru(i, j, nz - 1), ru(i, j, nz), &
           sum(rv(i, j, 1:nz)), rv(i, j, 1), rv(i, j, nz - 1), rv(i, j, nz)
      if (with_frc) then
        write(*,'(a,i1,a,a,a,2es26.17)') 'diag: ruP', n, '  ', trim(tag), &
             '  rufrcW, rvfrcS=', rufrc(i, j), rvfrc(i, j)
      endif
    enddo
    call flush(6)
  end subroutine dump_ru_probe

  subroutine dump_rv3d_after_rufrc()
    integer(ip) :: i, j, nz
    real(rp) :: extra
    if (.not. probe_on) return
    i = iprb_pt(1)
    j = jprb_pt(1)
    extra = 0.0_rp
    if (i >= 1 .and. i <= decomp%nx .and. j >= 1 .and. j <= decomp%ny) then
      call sync_arch()
      nz = decomp%nz
      extra = rvfrc(i, j) - sum(rv(i, j, 1:nz))
    endif
    call dump_rv3d('after rufrc', extra)
  end subroutine dump_rv3d_after_rufrc

  subroutine dump_cpl1(i, j, uint, Dcol, mis, vel)
    ! Couple1 V-face ingredients at P1 neighbors. uint/Dcol are the
    ! column integral of Hz*v and the V-face thickness sum; v(:,:,:,nnew)
    ! is already divided by Hz_v but not yet shifted by mis. vel, if
    ! given, holds v(1,50,99,nz) captured in the kernel before the shift.
    integer(ip), intent(in) :: i, j
    real(rp), intent(in) :: uint, Dcol, mis
    real(rp), intent(in), optional :: vel(4)
    real(rp) :: vd(4)
    real(rp) :: dmv, DVavg1, vmsk, den, termU, termD
    real(rp) :: Hz1, Hz50, Hz99, Hz100
    integer(ip) :: nz
    ! Levels 50 and 99 print 0 when they are above nz.

    if (.not. probe_on) return
    if (i /= iprb_pt(4) .or. j /= jprb_pt(4)) return
    call sync_arch()
    nz = decomp%nz
    dmv = dm_v(i, j)
    DVavg1 = DV_avg1(i, j)
    vmsk = vmask(i, j)
    den = Dcol * dmv
    termU = 0.0_rp
    termD = 0.0_rp
    if (abs(Dcol) > 1.0e-30_rp) termU = uint / Dcol
    if (abs(den) > 1.0e-30_rp) termD = DVavg1 / den
    Hz1 = 0.5_rp * (Hz(i, j, 1) + Hz(i, j - 1, 1))
    Hz50 = 0.0_rp; Hz99 = 0.0_rp
    if (nz >= 50) Hz50 = 0.5_rp * (Hz(i, j, 50) + Hz(i, j - 1, 50))
    if (nz >= 99) Hz99 = 0.5_rp * (Hz(i, j, 99) + Hz(i, j - 1, 99))
    Hz100 = 0.5_rp * (Hz(i, j, nz) + Hz(i, j - 1, nz))
    if (present(vel)) then
      vd = vel
    else
      vd = 0.0_rp
      vd(1) = v(i, j, 1, nnew); vd(4) = v(i, j, nz, nnew)
      if (nz >= 50) vd(2) = v(i, j, 50, nnew)
      if (nz >= 99) vd(3) = v(i, j, 99, nnew)
    endif
    write(*,'(a,i6,a,i6,a,i8,a,8es26.17)') 'diag: cpl1v B  i=', i, &
         ' j=', j, '  couple1  iic=', iic, &
         '  uint,Dcol,dmv,DVavg1,mis,vmask,termU,termD=', &
         uint, Dcol, dmv, DVavg1, mis, vmsk, termU, termD
    write(*,'(a,i6,a,i6,a,i8,a,8es26.17)') 'diag: cpl1v B  i=', i, &
         ' j=', j, '  couple1 Hz  iic=', iic, &
         '  Hz1,Hz50,Hz99,Hz100,vd1,vd50,vd99,vd100=', &
         Hz1, Hz50, Hz99, Hz100, vd
    flush(6)
  end subroutine dump_cpl1

  subroutine dump_cpl1_u(i, j, uint, Dcol, mis, vel)
    ! vel, if given, holds u(1,50,99,nz) captured in the kernel before
    ! the mis shift (see dump_cpl1).
    integer(ip), intent(in) :: i, j
    real(rp), intent(in) :: uint, Dcol, mis
    real(rp), intent(in), optional :: vel(4)
    real(rp) :: dnu, DUavg1, umsk, den, termU, termD, ud(4), hzl(4)
    integer(ip) :: nz
    ! Levels 50 and 99 print 0 when they are above nz.

    if (.not. probe_on) return
    if (i /= iprb_pt(4) .or. j /= jprb_pt(4)) return
    call sync_arch()
    nz = decomp%nz
    dnu = dn_u(i, j)
    DUavg1 = DU_avg1(i, j)
    umsk = umask(i, j)
    den = Dcol * dnu
    termU = 0.0_rp
    termD = 0.0_rp
    if (abs(Dcol) > 1.0e-30_rp) termU = uint / Dcol
    if (abs(den) > 1.0e-30_rp) termD = DUavg1 / den
    if (present(vel)) then
      ud = vel
    else
      ud = 0.0_rp
      ud(1) = u(i, j, 1, nnew); ud(4) = u(i, j, nz, nnew)
      if (nz >= 50) ud(2) = u(i, j, 50, nnew)
      if (nz >= 99) ud(3) = u(i, j, 99, nnew)
    endif
    hzl = 0.0_rp
    hzl(1) = 0.5_rp * (Hz(i, j, 1) + Hz(i - 1, j, 1))
    if (nz >= 50) hzl(2) = 0.5_rp * (Hz(i, j, 50) + Hz(i - 1, j, 50))
    if (nz >= 99) hzl(3) = 0.5_rp * (Hz(i, j, 99) + Hz(i - 1, j, 99))
    hzl(4) = 0.5_rp * (Hz(i, j, nz) + Hz(i - 1, j, nz))
    write(*,'(a,i6,a,i6,a,i8,a,8es26.17)') 'diag: cpl1u B  i=', i, &
         ' j=', j, '  couple1  iic=', iic, &
         '  uint,Dcol,dnu,DUavg1,mis,umask,termU,termD=', &
         uint, Dcol, dnu, DUavg1, mis, umsk, termU, termD
    write(*,'(a,i6,a,i6,a,i8,a,8es26.17)') 'diag: cpl1u B  i=', i, &
         ' j=', j, '  couple1 Hz  iic=', iic, &
         '  Hz1,Hz50,Hz99,Hz100,ud1,ud50,ud99,ud100=', hzl, ud
    flush(6)
  end subroutine dump_cpl1_u

  subroutine dump_uv1_b(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, kw, nz
    real(rp) :: hzu, hzv

    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    nz = decomp%nz
    do k = 1, nz
      kw = max(k - 1, 0)
      hzu = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
      hzv = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
      write(*,'(a,a,a,i8,a,i4,a,12es26.17)') 'diag: uv1 B u  ', trim(tag), &
           '  iic=', iic, '  k=', k, &
           '  u_stp,u_rhs,u_new,HzU,ru,AkvW,AkvE,WiW,WiE,sustr,rDW,rDE=', &
           u(i, j, k, nstp), u(i, j, k, nrhs), u(i, j, k, nnew), hzu, ru(i, j, k), &
           Akv(i - 1, j, kw), Akv(i, j, kw), Wi(i - 1, j, kw), Wi(i, j, kw), &
           sustr(i, j), r_D(i - 1, j), r_D(i, j)
      write(*,'(a,a,a,i8,a,i4,a,12es26.17)') 'diag: uv1 B v  ', trim(tag), &
           '  iic=', iic, '  k=', k, &
           '  v_stp,v_rhs,v_new,HzV,rv,AkvS,AkvN,WiS,WiN,svstr,rDS,rDN=', &
           v(i, j, k, nstp), v(i, j, k, nrhs), v(i, j, k, nnew), hzv, rv(i, j, k), &
           Akv(i, j - 1, kw), Akv(i, j, kw), Wi(i, j - 1, kw), Wi(i, j, kw), &
           svstr(i, j), r_D(i, j - 1), r_D(i, j)
    enddo
    write(*,'(a,a,a,i8,a,6es26.17)') 'diag: uv1 B 2d  ', trim(tag), &
         '  iic=', iic, '  DC0u,DC0v,pmW,pmE,pnS,pnN=', &
         dt * 0.25_rp * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j)), &
         dt * 0.25_rp * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1)), &
         pm(i - 1, j), pm(i, j), pn(i, j - 1), pn(i, j)
    flush(6)
  end subroutine dump_uv1_b

  subroutine dump_rhs_b(tag)
    ! ru/rv at family B after each 3-D RHS stage (PGF, horiz, vert).
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, nz
    real(rp) :: hzu, hzv

    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    nz = decomp%nz
    do k = 1, nz
      hzu = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
      hzv = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
      write(*,'(a,a,a,i8,a,i4,a,5es26.17)') 'diag: rhs B u  ', trim(tag), &
           '  iic=', iic, '  k=', k, '  ru,u_rhs,u_new,HzU,umask=', &
           ru(i, j, k), u(i, j, k, nrhs), u(i, j, k, nnew), hzu, umask(i, j)
      write(*,'(a,a,a,i8,a,i4,a,5es26.17)') 'diag: rhs B v  ', trim(tag), &
           '  iic=', iic, '  k=', k, '  rv,v_rhs,v_new,HzV,vmask=', &
           rv(i, j, k), v(i, j, k, nrhs), v(i, j, k, nnew), hzv, vmask(i, j)
    enddo
    flush(6)
  end subroutine dump_rhs_b

  subroutine dump_thom_b(face)
    character(len=*), intent(in) :: face
    integer(ip) :: i, j, k, kw, nz
    real(rp) :: hzu, ruv, vel, akw, ake, sstr, rdf

    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    nz = decomp%nz
    do k = 1, nz
      kw = min(max(k - 1, 1), nz - 1)
      if (trim(face) == 'u') then
        hzu = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
        ruv = ru(i, j, k)
        vel = u(i, j, k, nnew)
        akw = Akv(i - 1, j, kw)
        ake = Akv(i, j, kw)
        sstr = sustr(i, j)
        rdf = 0.5_rp * (r_D(i, j) + r_D(i - 1, j))
      else
        hzu = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
        ruv = rv(i, j, k)
        vel = v(i, j, k, nnew)
        akw = Akv(i, j - 1, kw)
        ake = Akv(i, j, kw)
        sstr = svstr(i, j)
        rdf = 0.5_rp * (r_D(i, j) + r_D(i, j - 1))
      endif
      write(*,'(a,a,a,i8,a,i4,a,12es26.17)') 'diag: thom B ', trim(face), &
           '  iic=', iic, '  k=', k, &
           '  DC0,FC,WC,CF,DC,ru,u_new,HzU,AkvW,AkvE,sstr,rD=', &
           col_dc(i, j, 0), col_fc(i, j, kw), col_wc(i, j, kw), col_cf(i, j, kw), &
           col_dc(i, j, k), ruv, vel, hzu, akw, ake, sstr, rdf
    enddo
    flush(6)
  end subroutine dump_thom_b

  subroutine dump_uv2_b(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, nz
    real(rp) :: hzu, hzv

    if (.not. probe_on) return
    call sync_arch()
    i = iprb_pt(4)
    j = jprb_pt(4)
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    nz = decomp%nz
    do k = 1, nz
      hzu = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
      hzv = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
      write(*,'(a,a,a,i8,a,i4,a,10es26.17)') 'diag: uv2 B u  ', trim(tag), &
           '  iic=', iic, '  k=', k, &
           '  u_stp,u_rhs,u_new,HzU,dn_u,umask,DU1,DU2,FlxU,ru=', &
           u(i, j, k, nstp), u(i, j, k, nrhs), u(i, j, k, nnew), &
           hzu, dn_u(i, j), umask(i, j), DU_avg1(i, j), DU_avg2(i, j), &
           FlxU(i, j, k), ru(i, j, k)
      write(*,'(a,a,a,i8,a,i4,a,10es26.17)') 'diag: uv2 B v  ', trim(tag), &
           '  iic=', iic, '  k=', k, &
           '  v_stp,v_rhs,v_new,HzV,dm_v,vmask,DV1,DV2,FlxV,rv=', &
           v(i, j, k, nstp), v(i, j, k, nrhs), v(i, j, k, nnew), &
           hzv, dm_v(i, j), vmask(i, j), DV_avg1(i, j), DV_avg2(i, j), &
           FlxV(i, j, k), rv(i, j, k)
    enddo
    flush(6)
  end subroutine dump_uv2_b

  subroutine dump_cpl2_b(face, i, j, Dcol, uint, flx_int, mis_u, mis_f, avg1, avg2)
    character(len=*), intent(in) :: face
    integer(ip), intent(in) :: i, j
    real(rp), intent(in) :: Dcol, uint, flx_int, mis_u, mis_f, avg1, avg2

    if (.not. probe_on) return
    if (i /= iprb_pt(4) .or. j /= jprb_pt(4)) return
    call sync_arch()
    write(*,'(a,a,a,i8,a,8es26.17)') 'diag: cpl2 B ', trim(face), &
         '  iic=', iic, &
         '  Dcol,uint,flx_int,mis_u,mis_f,avg1,avg2,vel100=', &
         Dcol, uint, flx_int, mis_u, mis_f, avg1, avg2, 0.0_rp
    if (trim(face) == 'u') then
      write(*,'(a,a,a,i8,a,4es26.17)') 'diag: cpl2 B ', trim(face), &
           '  iic=', iic, '  u100,FlxU100,HzU100,dn_u=', &
           u(i, j, decomp%nz, nnew), FlxU(i, j, decomp%nz), &
           0.5_rp * (Hz(i, j, decomp%nz) + Hz(i - 1, j, decomp%nz)), dn_u(i, j)
    else
      write(*,'(a,a,a,i8,a,4es26.17)') 'diag: cpl2 B ', trim(face), &
           '  iic=', iic, '  v100,FlxV100,HzV100,dm_v=', &
           v(i, j, decomp%nz, nnew), FlxV(i, j, decomp%nz), &
           0.5_rp * (Hz(i, j, decomp%nz) + Hz(i, j - 1, decomp%nz)), dm_v(i, j)
    endif
    flush(6)
  end subroutine dump_cpl2_b

  subroutine step3d_uv2_couple2_imex(j_ur0, j_ur1, i_ur1, j_vr1, i_vr0, i_vr1, lim)
    ! step3d_uv2's second coupling with the IMEX_IGW corrector transport
    ! (CPU step3d_uv2 under IMEX_IGW): Flx_corr = mu Flx* + Hz_u dn_u
    ! (theta_c u^n+1 + nu u^n + rho u^n-1), weights wflx_corr on (u^n+1,
    ! Flx*, u^n, u^n-1); otherwise the stock kernels.
    integer(ip), intent(in) :: j_ur0, j_ur1, i_ur1, j_vr1, i_vr0, i_vr1, lim
    integer(ip) :: i, j, k, nz, n_new, n_stp, kn, lm
    real(rp) :: Dcol, uint, mis_u, mis_f, flx_int, flx_k, hz_dn, u_k, w1, w2, w3, w4
    nz = decomp%nz
    n_new = nnew
    n_stp = nstp
    kn = knew
    lm = lim
    w1 = wflx_corr(1); w2 = wflx_corr(2); w3 = wflx_corr(3); w4 = wflx_corr(4)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_ur0, j_ur1
      do i = 1, i_ur1
        if (lm > 0) then; if (colact(i, j) == 0) cycle; end if
        if (umask(i, j) + riv_umask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)
            Dcol = Dcol + hz_dn
            uint = uint + hz_dn * u(i, j, k, n_new)
          enddo
          ubar(i, j, kn) = DU_avg1(i, j) / Dcol
          mis_u = (uint - DU_avg1(i, j)) / Dcol
          ! shift and flux in one pass (hz_dn recomputed, same expression)
          flx_int = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)
            u_k = apply_mask((u(i, j, k, n_new) - mis_u), (umask(i, j)))
            u(i, j, k, n_new) = u_k
            flx_k = w2 * FlxU(i, j, k) + hz_dn * &
                    (w1 * u_k + w3 * u(i, j, k, n_stp) + w4 * u_nm1(i, j, k))
            col_dc(i, j, k) = flx_k
            flx_int = flx_int + flx_k
          enddo
          mis_f = (flx_int - DU_avg2(i, j)) / Dcol
          do k = 1, nz
            FlxU(i, j, k) = col_dc(i, j, k) - (0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)) * mis_f
          enddo
        else
          ubar(i, j, kn) = 0.0_rp
          do k = 1, nz
            u(i, j, k, n_new) = 0.0_rp
            FlxU(i, j, k) = 0.0_rp
          enddo
        endif
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, j_vr1
      do i = i_vr0, i_vr1
        if (lm > 0) then; if (colact(i, j) == 0) cycle; end if
        if (vmask(i, j) + riv_vmask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)
            Dcol = Dcol + hz_dn
            uint = uint + hz_dn * v(i, j, k, n_new)
          enddo
          vbar(i, j, kn) = DV_avg1(i, j) / Dcol
          mis_u = (uint - DV_avg1(i, j)) / Dcol
          ! shift and flux in one pass (hz_dn recomputed, same expression)
          flx_int = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)
            u_k = apply_mask((v(i, j, k, n_new) - mis_u), (vmask(i, j)))
            v(i, j, k, n_new) = u_k
            flx_k = w2 * FlxV(i, j, k) + hz_dn * &
                    (w1 * u_k + w3 * v(i, j, k, n_stp) + w4 * v_nm1(i, j, k))
            col_dc(i, j, k) = flx_k
            flx_int = flx_int + flx_k
          enddo
          mis_f = (flx_int - DV_avg2(i, j)) / Dcol
          do k = 1, nz
            FlxV(i, j, k) = col_dc(i, j, k) - (0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)) * mis_f
          enddo
        else
          vbar(i, j, kn) = 0.0_rp
          do k = 1, nz
            v(i, j, k, n_new) = 0.0_rp
            FlxV(i, j, k) = 0.0_rp
          enddo
        endif
      enddo
    enddo
  end subroutine step3d_uv2_couple2_imex

  subroutine step3d_uv2()
    integer(ip) :: i, j, k, nx, ny, nz, n_new, n_stp, kn
    integer(ip) :: i_u0, i_u1, j_v0, i_ur1, j_vr1, j_ur0, j_ur1, i_vr0, i_vr1
    real(rp) :: cff, Dcol, uint, mis, mis_u, mis_f, flx_int, flx_k, hz_dn, u_k
    integer(ip) :: ip4, jp4, icu_i, icu_j, icu_k, lim
    logical :: probe_c, ic_on

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_new = nnew
    n_stp = nstp
    kn = knew
    i_u0 = 1
    if (decomp%edge_w) i_u0 = 2
    i_u1 = nx
    j_v0 = 1
    if (decomp%edge_s) j_v0 = 2
    i_ur1 = nx
    if (decomp%edge_e) i_ur1 = nx + 1
    j_vr1 = ny
    if (decomp%edge_n) j_vr1 = ny + 1
    j_ur0 = 1; j_ur1 = ny; i_vr0 = 1; i_vr1 = nx
    if (decomp%edge_s) j_ur0 = 0
    if (decomp%edge_n) j_ur1 = ny + 1
    if (decomp%edge_w) i_vr0 = 0
    if (decomp%edge_e) i_vr1 = nx + 1

    ! First coupling: interior only (CPU istrU:iend / jstrV:jend).
    ! On entry u,v(:,:,:,nnew) are Hz*velocity from step3d_uv1.
    ! Each column (vertical sum, divide by Hz, remove the barotropic
    ! mismatch) is one device thread; the k-loops stay serial in-thread.
    if (.not. allocated(cpl_prb)) then
      allocate(cpl_prb(NCPL_PRB))
      cpl_prb = 0.0_rp
    endif
    probe_c = probe_on
    ! The IC-jet line prints on probe steps only: each print synchronises
    ! the device and reads the managed cpl_prb on the host.
    ic_on = (ic_u_k >= 1) .and. probe_on
    icu_i = ic_u_i; icu_j = ic_u_j; icu_k = ic_u_k
    ip4 = iprb_pt(4); jp4 = jprb_pt(4)
    if (probe_c .or. ic_on) then
      call sync_arch()
      cpl_prb = 0.0_rp
    endif
    call dump_uv2_b('before couple1')
    lim = dry_lim(G_MOM)
    if (.not. probe_c) then
    ! One pass fewer: the vertical sums first, then u/cff - mis per level
    ! (the same two roundings as storing u/cff and subtracting mis). The
    ! IC-jet value u/cff at icu_k is the same expression. Probe steps keep
    ! the stored-quotient form below (the couple1 probe prints it).
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (umask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            cff = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
            Dcol = Dcol + cff
            uint = uint + u(i, j, k, n_new)
          enddo
          mis = (uint * dn_u(i, j) - DU_avg1(i, j)) / (Dcol * dn_u(i, j))
          if (ic_on .and. i == icu_i .and. j == icu_j) then
            cpl_prb(15) = 0.5_rp * (Hz(i, j, icu_k) + Hz(i - 1, j, icu_k))
            cpl_prb(16) = u(i, j, icu_k, n_new) / (0.5_rp * (Hz(i, j, icu_k) + Hz(i - 1, j, icu_k)))
            cpl_prb(17) = uint / max(Dcol, 1.0e-30_rp)
            cpl_prb(18) = DU_avg1(i, j) / max(Dcol * dn_u(i, j), 1.0e-30_rp)
            cpl_prb(19) = mis
            cpl_prb(32) = 1.0_rp
          endif
          do k = 1, nz
            cff = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
            u(i, j, k, n_new) = u(i, j, k, n_new) / cff - mis
          enddo
        else
          do k = 1, nz
            u(i, j, k, n_new) = 0.0_rp
          enddo
        endif
      enddo
    enddo
    else
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = i_u0, i_u1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (umask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            cff = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
            Dcol = Dcol + cff
            uint = uint + u(i, j, k, n_new)
            u(i, j, k, n_new) = u(i, j, k, n_new) / cff
          enddo
          mis = (uint * dn_u(i, j) - DU_avg1(i, j)) / (Dcol * dn_u(i, j))
          if (probe_c .and. i == ip4 .and. j == jp4) then
            cpl_prb(1) = uint; cpl_prb(2) = Dcol; cpl_prb(3) = mis
            cpl_prb(4) = u(i, j, 1, n_new); cpl_prb(7) = u(i, j, nz, n_new)
            if (nz >= 50) cpl_prb(5) = u(i, j, 50, n_new)   ! else 0 (zeroed on the host)
            if (nz >= 99) cpl_prb(6) = u(i, j, 99, n_new)
            cpl_prb(30) = 1.0_rp
          endif
          if (ic_on .and. i == icu_i .and. j == icu_j) then
            cpl_prb(15) = 0.5_rp * (Hz(i, j, icu_k) + Hz(i - 1, j, icu_k))
            cpl_prb(16) = u(i, j, icu_k, n_new)
            cpl_prb(17) = uint / max(Dcol, 1.0e-30_rp)
            cpl_prb(18) = DU_avg1(i, j) / max(Dcol * dn_u(i, j), 1.0e-30_rp)
            cpl_prb(19) = mis
            cpl_prb(32) = 1.0_rp
          endif
          do k = 1, nz
            u(i, j, k, n_new) = u(i, j, k, n_new) - mis
          enddo
        else
          do k = 1, nz
            u(i, j, k, n_new) = 0.0_rp
          enddo
        endif
      enddo
    enddo
    endif
    if (probe_c .or. ic_on) then
      call sync_arch()
      if (cpl_prb(30) == 1.0_rp) call dump_cpl1_u(ip4, jp4, cpl_prb(1), cpl_prb(2), &
           cpl_prb(3), cpl_prb(4:7))
      if (cpl_prb(32) == 1.0_rp) then
        write(*,'(a,3i6,a,6es14.6)') 'diag: uv2 couple1 IC-jet i,j,k=', &
             icu_i, icu_j, icu_k, '  Hz_u,u,u-mis,uint/D,DU/D,mis=', &
             cpl_prb(15), cpl_prb(16), cpl_prb(16) - cpl_prb(19), &
             cpl_prb(17), cpl_prb(18), cpl_prb(19)
      endif
    endif
    if (.not. probe_c) then
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (vmask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            cff = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
            Dcol = Dcol + cff
            uint = uint + v(i, j, k, n_new)
          enddo
          mis = (uint * dm_v(i, j) - DV_avg1(i, j)) / (Dcol * dm_v(i, j))
          do k = 1, nz
            cff = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
            v(i, j, k, n_new) = v(i, j, k, n_new) / cff - mis
          enddo
        else
          do k = 1, nz
            v(i, j, k, n_new) = 0.0_rp
          enddo
        endif
      enddo
    enddo
    else
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_v0, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (vmask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            cff = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k))
            Dcol = Dcol + cff
            uint = uint + v(i, j, k, n_new)
            v(i, j, k, n_new) = v(i, j, k, n_new) / cff
          enddo
          mis = (uint * dm_v(i, j) - DV_avg1(i, j)) / (Dcol * dm_v(i, j))
          if (probe_c .and. i == ip4 .and. j == jp4) then
            cpl_prb(8) = uint; cpl_prb(9) = Dcol; cpl_prb(10) = mis
            cpl_prb(11) = v(i, j, 1, n_new); cpl_prb(14) = v(i, j, nz, n_new)
            if (nz >= 50) cpl_prb(12) = v(i, j, 50, n_new)
            if (nz >= 99) cpl_prb(13) = v(i, j, 99, n_new)
            cpl_prb(31) = 1.0_rp
          endif
          do k = 1, nz
            v(i, j, k, n_new) = v(i, j, k, n_new) - mis
          enddo
        else
          if (probe_c .and. i == ip4 .and. j == jp4) then
            cpl_prb(8) = 0.0_rp; cpl_prb(9) = 0.0_rp; cpl_prb(10) = 0.0_rp
            cpl_prb(11) = v(i, j, 1, n_new); cpl_prb(14) = v(i, j, nz, n_new)
            if (nz >= 50) cpl_prb(12) = v(i, j, 50, n_new)
            if (nz >= 99) cpl_prb(13) = v(i, j, 99, n_new)
            cpl_prb(31) = 1.0_rp
          endif
          do k = 1, nz
            v(i, j, k, n_new) = 0.0_rp
          enddo
        endif
      enddo
    enddo
    endif
    if (probe_c) then
      call sync_arch()
      if (cpl_prb(31) == 1.0_rp) call dump_cpl1(ip4, jp4, cpl_prb(8), cpl_prb(9), &
           cpl_prb(10), cpl_prb(11:14))
    endif
    call dump_vn3('after corr couple1')
    call dump_uv2_b('after couple1')

    call u3dbc()
    call v3dbc()
    if (imex_on) call imex_obc_faces(imex_stage_corr)    ! 'modal': the normal open faces (CPU imex_obc_faces)
    call dump_vn3('after corr v3dbc')
    call dump_uv2_b('after v3dbc')

    ! Second coupling over the extended physical range, then mass fluxes.
    ! CPU builds Flx from u,v AFTER subtracting the barotropic mismatch.
    ! One device thread per column; col_dc holds the column's Hz*dn, then
    ! its fluxes. River faces (umask = 0, riv_umask = 1) are coupled as
    ! the CPU does on every face: ubar = DU_avg1/D and the flux correction,
    ! with u(nnew) masked to 0 (rivers_uv sets it afterwards).
    if (imex_on) then
      ! IMEX_IGW: the tracer transport with the IMEX weights (wflx_corr)
      call step3d_uv2_couple2_imex(j_ur0, j_ur1, i_ur1, j_vr1, i_vr0, i_vr1, lim)
    else
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j_ur0, j_ur1
      do i = 1, i_ur1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (umask(i, j) + riv_umask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)
            Dcol = Dcol + hz_dn
            uint = uint + hz_dn * u(i, j, k, n_new)
          enddo
          ubar(i, j, kn) = DU_avg1(i, j) / Dcol
          mis_u = (uint - DU_avg1(i, j)) / Dcol
          ! shift and flux in one pass (hz_dn recomputed, same expression)
          flx_int = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)
            u_k = apply_mask((u(i, j, k, n_new) - mis_u), (umask(i, j)))
            u(i, j, k, n_new) = u_k
            flx_k = FLUX_DELTA * FlxU(i, j, k) + FLUX_EPSIL * hz_dn * &
                    (u(i, j, k, n_stp) + u_k)
            col_dc(i, j, k) = flx_k
            flx_int = flx_int + flx_k
          enddo
          mis_f = (flx_int - DU_avg2(i, j)) / Dcol
          do k = 1, nz
            FlxU(i, j, k) = col_dc(i, j, k) - (0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j)) * mis_f
          enddo
          if (probe_c .and. i == ip4 .and. j == jp4) then
            cpl_prb(20) = Dcol; cpl_prb(21) = uint; cpl_prb(22) = flx_int
            cpl_prb(23) = mis_u; cpl_prb(24) = mis_f
            cpl_prb(33) = 1.0_rp
          endif
        else
          ubar(i, j, kn) = 0.0_rp
          do k = 1, nz
            u(i, j, k, n_new) = 0.0_rp
            FlxU(i, j, k) = 0.0_rp
          enddo
        endif
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, j_vr1
      do i = i_vr0, i_vr1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        if (vmask(i, j) + riv_vmask(i, j) > 0.5_rp) then
          Dcol = 0.0_rp
          uint = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)
            Dcol = Dcol + hz_dn
            uint = uint + hz_dn * v(i, j, k, n_new)
          enddo
          vbar(i, j, kn) = DV_avg1(i, j) / Dcol
          mis_u = (uint - DV_avg1(i, j)) / Dcol
          ! shift and flux in one pass (hz_dn recomputed, same expression)
          flx_int = 0.0_rp
          do k = 1, nz
            hz_dn = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)
            u_k = apply_mask((v(i, j, k, n_new) - mis_u), (vmask(i, j)))
            v(i, j, k, n_new) = u_k
            flx_k = FLUX_DELTA * FlxV(i, j, k) + FLUX_EPSIL * hz_dn * &
                    (v(i, j, k, n_stp) + u_k)
            col_dc(i, j, k) = flx_k
            flx_int = flx_int + flx_k
          enddo
          mis_f = (flx_int - DV_avg2(i, j)) / Dcol
          do k = 1, nz
            FlxV(i, j, k) = col_dc(i, j, k) - (0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j)) * mis_f
          enddo
          if (probe_c .and. i == ip4 .and. j == jp4) then
            cpl_prb(25) = Dcol; cpl_prb(26) = uint; cpl_prb(27) = flx_int
            cpl_prb(28) = mis_u; cpl_prb(29) = mis_f
            cpl_prb(34) = 1.0_rp
          endif
        else
          vbar(i, j, kn) = 0.0_rp
          do k = 1, nz
            v(i, j, k, n_new) = 0.0_rp
            FlxV(i, j, k) = 0.0_rp
          enddo
        endif
      enddo
    enddo
    endif
    if (probe_c) then
      call sync_arch()
      if (cpl_prb(33) == 1.0_rp) call dump_cpl2_b('u', ip4, jp4, cpl_prb(20), cpl_prb(21), &
           cpl_prb(22), cpl_prb(23), cpl_prb(24), DU_avg1(ip4, jp4), DU_avg2(ip4, jp4))
      if (cpl_prb(34) == 1.0_rp) call dump_cpl2_b('v', ip4, jp4, cpl_prb(25), cpl_prb(26), &
           cpl_prb(27), cpl_prb(28), cpl_prb(29), DV_avg1(ip4, jp4), DV_avg2(ip4, jp4))
    endif

    call dump_uv2_b('after couple2')
    if (river_on) call rivers_uv(RIV_COR)    ! CPU step3d_uv2: river faces before the exchange
    call hg_begin()
    call hg_add(F_U, nnew)
    call hg_add(F_V, nnew)
    call hg_add(F_FLXU)
    call hg_add(F_FLXV)
    call hg_add(F_UBAR, knew)
    call hg_add(F_VBAR, knew)
    call hg_end()
  end subroutine step3d_uv2

end module step3d_uv_mod

#include "cuf_launch.inc"
#ifdef USE_CUDA
! Rounded multiply: keeps a*b from contracting into an FMA on the device.
#define BAK_MUL(a, b) __dmul_rn(a, b)
#else
#define BAK_MUL(a, b) ((a) * (b))
#endif
module step2d_mod
  ! One fused FB AB3-AM4 fast step. GPU: pre/tail graphs; mid stays
  ! live so FIRST_2D_STEP does not replay. Host resets DU_avg* on
  ! iif=1, then kernels always accumulate. nfast stays on the host.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use geom, only: set_depth, apply_volume_sources, dump_rufrc_build, &
       dump_zeta_obc, dump_vbar_obc, dump_fast_zeta_inc, dump_rvS
  use halo_mod
  use formulas
  use obc, only: zetabc, uv2dbc, uv2dbc_corners
  use rivers_mod, only: river_on, rivers_ubar, rivers_duon, rf_flx, riv_ufc, riv_vfc, nrf_s2df
  use cdr_src_mod, only: cdr_vol_on, cdr_zsrc
  use comm_mod, only: comm_size
  use arch_mod, only: sync_arch, on_gpu, graph_is_ready, graph_begin, &
                      graph_end, graph_launch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
  use cudadevice, only: __dmul_rn
#endif
  implicit none
  private
  public :: step2d_fused
  public :: step2d_with_halo

  ! Host copies of fast%is_last / fast%half_w1 (fast lives on the GPU).
  logical :: fast_is_last_h = .false.
  real(rp) :: fast_half_w1_h = 0.0_rp

contains

  subroutine fill_fast_pack(iif_in)
    ! CUF kernels take `fast` by value (a snapshot of the host copy at
    ! each launch), so the host may rewrite it as soon as the previous
    ! fast step's kernels have been launched: no device sync is needed.
    ! (A device-side write would only change a kernel's own argument copy.)
    integer(ip), intent(in) :: iif_in
    integer(ip) :: kbak, kold
    integer(ip) :: l_iif, l_kstp, l_knew, l_first, l_last, l_nx, l_ny
    integer(ip) :: l_is, l_ie, l_js, l_je, l_iu0, l_iu1, l_jv0, l_jv1
    real(rp) :: fwd, fwd1, fwd2, bkw_new, bkw, bkw1, bkw2
    real(rp) :: l_w1, l_w2, l_dtf, l_hdt, l_hw1

    iif = iif_in
    call fb_ab3_am4_weights(iif_in, fwd, fwd1, fwd2, bkw_new, bkw, bkw1, bkw2)
    call fb_time_indices(iif_in, kstp, kbak, kold)

    l_iif = iif_in
    l_kstp = kstp
    l_knew = knew
    l_first = merge(1_ip, 0_ip, iif_in == 1)
    l_last = merge(1_ip, 0_ip, iif_in == nfast)
    l_nx = decomp%nx
    l_ny = decomp%ny
    l_is = decomp%is
    l_ie = decomp%ie
    l_js = decomp%js
    l_je = decomp%je
    l_iu0 = merge(2, 1, decomp%edge_w)
    l_iu1 = merge(decomp%nx, decomp%nx + 1, decomp%edge_e)
    l_jv0 = merge(2, 1, decomp%edge_s)
    l_jv1 = merge(decomp%ny, decomp%ny + 1, decomp%edge_n)
    l_w1 = weight(1, iif_in)
    l_w2 = weight(2, iif_in)
    l_dtf = dtfast
    l_hdt = 0.5_rp * dtfast
    l_hw1 = 0.5_rp * weight(1, iif_in)
    fast_is_last_h = (l_last == 1)
    fast_half_w1_h = l_hw1

    fast%iif = l_iif
    fast%kstp = l_kstp
    fast%knew = l_knew
    fast%kbak = kbak
    fast%kold = kold
    fast%is_first = l_first
    fast%is_last = l_last
    fast%nx = l_nx
    fast%ny = l_ny
    fast%is = l_is
    fast%ie = l_ie
    fast%js = l_js
    fast%je = l_je
    fast%i_u_lo = l_iu0
    fast%i_u_hi = l_iu1
    fast%j_v_lo = l_jv0
    fast%j_v_hi = l_jv1
    fast%fwd = fwd
    fast%fwd1 = fwd1
    fast%fwd2 = fwd2
    fast%bkw_new = bkw_new
    fast%bkw = bkw
    fast%bkw1 = bkw1
    fast%bkw2 = bkw2
    fast%w1 = l_w1
    fast%w2 = l_w2
    fast%dtfast = l_dtf
    fast%half_dt = l_hdt
    fast%half_w1 = l_hw1
  end subroutine fill_fast_pack

  subroutine run_device_section(id)
    integer, intent(in) :: id
#ifdef USE_CUDA
    ! Section 2 (mid) has FIRST_2D_STEP stores. A graph captured on
    ! iif=1 freezes is_first=1, so every fast step re-zeros DU_avg1.
    ! Pre/tail graphs are safe: they do not reset the averages.
    if (on_gpu() .and. comm_size() == 1 .and. id /= 2 .and. graphs_enabled()) then
      if (graph_is_ready(id)) then
        call graph_launch(id)
        return
      endif
      ! Capture records kernels but does not run them. Launch immediately
      ! so the first fast step actually computes zeta_new / tail.
      call graph_begin()
      call device_section(id)
      call graph_end(id)
      call graph_launch(id)
      return
    endif
#endif
    call device_section(id)
  end subroutine run_device_section

  logical function graphs_enabled()
    ! Off by default: replayed pre/tail graphs keep iif=1 fast-step
    ! values and damp zeta ~3x vs CPU. ROMS_GPU_GRAPH=1 re-enables.
    logical, save :: checked = .false., enabled = .false.
    character(len=16) :: val
    integer :: stat
    if (.not. checked) then
      call get_environment_variable('ROMS_GPU_GRAPH', val, status=stat)
      enabled = stat == 0 .and. len_trim(val) > 0 .and. val(1:1) /= '0'
      if (enabled) then
        write(*,'(a)') 'step2d: CUDA graphs enabled (ROMS_GPU_GRAPH)'
      else
        write(*,'(a)') 'step2d: CUDA graphs disabled (set ROMS_GPU_GRAPH=1 to enable)'
      endif
      checked = .true.
    endif
    graphs_enabled = enabled
  end function graphs_enabled

  subroutine device_section(id)
    integer, intent(in) :: id
    select case (id)
    case (1)
      call step2d_pre()
    case (2)
      call step2d_mid()
    case (3)
      call step2d_tail()
    end select
  end subroutine device_section

  subroutine step2d_with_halo(iif_in)
    integer(ip), intent(in) :: iif_in
    call fill_fast_pack(iif_in)
    call halo_ensure()
    call run_device_section(1)
    call host_zeta_obc()
    call dump_fast_zeta_inc()
    call run_device_section(2)
    call dump_zeta_obc()
    call host_uv_obc(.false.)
    call dump_vbar_obc()
    ! (apply_volume_sources is empty; if it ever writes ubar/vbar it must
    ! stay before accumulate_boundary_avg1, which step2d_tail now queues
    ! behind the exchange pack.)
    call apply_volume_sources()
    if (river_on) call rivers_ubar(knew)     ! CPU step2d: after u2dbc/v2dbc (no river face is a boundary face)
    call run_device_section(3)
    if (fast_is_last_h) call set_depth()
  end subroutine step2d_with_halo

  subroutine step2d_fused(iif_in)
    integer(ip), intent(in) :: iif_in
    call fill_fast_pack(iif_in)
    call halo_ensure()
    call step2d_pre()
    call host_zeta_obc()
    call step2d_mid()                        ! exchanges zeta(knew) before its halo PGF terms
    call host_uv_obc(.true.)
    if (river_on) call rivers_ubar(knew)
    call halo_fill_uv_n(ubar, vbar)          ! every fast step, as the CPU step2d exchange
    if (fast_is_last_h) then
      call step2d_last_zeta()
      call halo_fill_2d_n(zeta, STAGGER_RHO)
      call set_depth()
    endif
  end subroutine step2d_fused

  subroutine step2d_pre()
    ! Three launches: Drhs; DUon and DVom together (both need Drhs of a
    ! neighbour); zeta_new (needs DUon/DVom of neighbours).
    integer(ip) :: i, j, ks, kb, ko, nx, ny, is, ie, js, je, ju0, ju1, iv0, iv1, lim
    integer(ip) :: iu_lo, iu_hi, jv_lo, jv_hi, iz_lo, iz_hi, jz_lo, jz_hi, cvo
    real(rp) :: fwd, fwd1, fwd2, dt_fast, urhs, vrhs

    nx = decomp%nx
    ny = decomp%ny
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    lim = dry_lim(G_2D)

    CUF_DO2
    do j = js, je
      do i = is, ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        ks = fast%kstp
        kb = fast%kbak
        ko = fast%kold
        fwd = fast%fwd
        fwd1 = fast%fwd1
        fwd2 = fast%fwd2
        Drhs(i, j) = h(i, j) + fwd * zeta(i, j, ks) &
                     + fwd1 * zeta(i, j, kb) + fwd2 * zeta(i, j, ko)
      enddo
    enddo

    ! Physical edges extend to the boundary rows/columns (CPU jstrR:jendR,
    ! istrR:iendR) so DU_avg2/DV_avg2 exist there for step3d_uv2.
    ju0 = 1; ju1 = ny; iv0 = 1; iv1 = nx
    if (decomp%edge_s) ju0 = 0
    if (decomp%edge_n) ju1 = ny + 1
    if (decomp%edge_w) iv0 = 0
    if (decomp%edge_e) iv1 = nx + 1
    ! zeta_new is also formed on the first halo ring of MPI seams (one
    ! exchange per fast step, see step2d_tail): DUon/DVom are needed one
    ! face further there. Physical edges keep the former ranges.
    call seam_ring(iz_lo, iz_hi, jz_lo, jz_hi)
    iu_lo = merge(1_ip, 0_ip, decomp%edge_w); iu_hi = merge(nx + 1, nx + 2, decomp%edge_e)
    jv_lo = merge(1_ip, 0_ip, decomp%edge_s); jv_hi = merge(ny + 1, ny + 2, decomp%edge_n)
    ju0 = min(ju0, jz_lo); ju1 = max(ju1, jz_hi)
    iv0 = min(iv0, iz_lo); iv1 = max(iv1, iz_hi)

    CUF_DO2
    do j = 0, ny + 2
      do i = 0, nx + 2
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        ks = fast%kstp
        kb = fast%kbak
        ko = fast%kold
        fwd = fast%fwd
        fwd1 = fast%fwd1
        fwd2 = fast%fwd2
        if (j >= ju0 .and. j <= ju1 .and. i >= iu_lo .and. i <= iu_hi) then
          urhs = fwd * ubar(i, j, ks) + fwd1 * ubar(i, j, kb) + fwd2 * ubar(i, j, ko)
          DUon(i, j) = 0.5_rp * (Drhs(i, j) + Drhs(i - 1, j)) * dn_u(i, j) * urhs
        endif
        if (i >= iv0 .and. i <= iv1 .and. j >= jv_lo .and. j <= jv_hi) then
          vrhs = fwd * vbar(i, j, ks) + fwd1 * vbar(i, j, kb) + fwd2 * vbar(i, j, ko)
          DVom(i, j) = 0.5_rp * (Drhs(i, j) + Drhs(i, j - 1)) * dm_v(i, j) * vrhs
        endif
      enddo
    enddo
    ! River faces carry the river transport (CPU ad6107f7), in the same
    ! ranges, before zeta_new and DU_avg2/DV_avg2 use DUon/DVom
    if (river_on) call rivers_duon(iu_lo, iu_hi, ju0, ju1, iv0, iv1, jv_lo, jv_hi)

    ! CDR volume releases (CPU c859943a): dtfast*cdr_zsrc into zeta_new
    ! after swflx, as the pipe source; cdr_zsrc is exchanged, so the seam
    ! ring gets the neighbour's releases. The water enters with zero
    ! momentum (it is not in zeta(kstp), hence not in Drhs/DUon).
    cvo = merge(1_ip, 0_ip, cdr_vol_on)

    CUF_DO2
    do j = jz_lo, jz_hi
      do i = iz_lo, iz_hi
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        ks = fast%kstp
        dt_fast = fast%dtfast
        zeta_new(i, j) = zeta(i, j, ks) + dt_fast * pm(i, j) * pn(i, j) * &
                         (DUon(i, j) - DUon(i + 1, j) + DVom(i, j) - DVom(i, j + 1)) &
                         + dt_fast * swflx(i, j)
        if (cvo == 1) zeta_new(i, j) = zeta_new(i, j) + dt_fast * cdr_zsrc(i, j)
        zeta_new(i, j) = apply_mask(zeta_new(i, j), rmask(i, j))
      enddo
    enddo
  end subroutine step2d_pre

  subroutine host_zeta_obc()
    ! zetabc writes the four zeta_new edges in one launch; this kernel
    ! copies the non-corner edge cells into zeta(knew) and then one thread
    ! sets the four corners in both arrays: the average of the two
    ! adjacent edge values where two physical edges meet (as before), or
    ! the single edge's value otherwise. Corners are written by that one
    ! thread only, so no other thread's copy can race it.
    integer(ip) :: m, nx, ny, kn, mmax
    logical :: ew, ee, es, en
    real(rp) :: zc
    nx = decomp%nx
    ny = decomp%ny
    kn = knew
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    if (.not. (ew .or. ee .or. es .or. en)) return
    call zetabc()
    mmax = max(nx, ny)
    CUF_DO1
    do m = 0, mmax
      if (m >= 1 .and. m <= ny) then
        if (ew) zeta(0, m, kn) = zeta_new(0, m)
        if (ee) zeta(nx + 1, m, kn) = zeta_new(nx + 1, m)
      endif
      if (m >= 1 .and. m <= nx) then
        if (es) zeta(m, 0, kn) = zeta_new(m, 0)
        if (en) zeta(m, ny + 1, kn) = zeta_new(m, ny + 1)
      endif
      if (m == 0) then
        if (es .and. ew) then
          zc = 0.5_rp * (zeta_new(1, 0) + zeta_new(0, 1))
          zeta_new(0, 0) = zc; zeta(0, 0, kn) = zc
        elseif (es .or. ew) then
          zeta(0, 0, kn) = zeta_new(0, 0)
        endif
        if (es .and. ee) then
          zc = 0.5_rp * (zeta_new(nx, 0) + zeta_new(nx + 1, 1))
          zeta_new(nx + 1, 0) = zc; zeta(nx + 1, 0, kn) = zc
        elseif (es .or. ee) then
          zeta(nx + 1, 0, kn) = zeta_new(nx + 1, 0)
        endif
        if (en .and. ew) then
          zc = 0.5_rp * (zeta_new(1, ny + 1) + zeta_new(0, ny))
          zeta_new(0, ny + 1) = zc; zeta(0, ny + 1, kn) = zc
        elseif (en .or. ew) then
          zeta(0, ny + 1, kn) = zeta_new(0, ny + 1)
        endif
        if (en .and. ee) then
          zc = 0.5_rp * (zeta_new(nx, ny + 1) + zeta_new(nx + 1, ny))
          zeta_new(nx + 1, ny + 1) = zc; zeta(nx + 1, ny + 1, kn) = zc
        elseif (en .or. ee) then
          zeta(nx + 1, ny + 1, kn) = zeta_new(nx + 1, ny + 1)
        endif
      endif
    enddo
  end subroutine host_zeta_obc

  subroutine step2d_mid()
    ! Three launches in the steady state: (1) per point: zeta(knew) copy,
    ! Dnew, the FIRST_2D_STEP bak/zero, the fast averages and the PGF
    ! state; (2) the barotropic PGF (needs neighbours of 1) plus
    ! DUon = h + zeta(kstp); (3) the ubar/vbar update. Per-point
    ! arithmetic is unchanged; iif = 1 adds the rufrc corrector kernels.
    integer(ip) :: i, j, ks, kn, kb, ko, nx, ny, is, ie, js, je, ju0, ju1, iv0, iv1
    integer(ip) :: i_u_lo, i_u_hi, j_v_lo, j_v_hi, ir0, ir1, jr0, jr1, lim
    integer(ip) :: iz_lo, iz_hi, jz_lo, jz_hi
    real(rp) :: cff, cff1, DUnew, DVnew, zk, z_old, term, flux
    integer(ip) :: fr
    logical :: first, rv_on

    nx = decomp%nx
    ny = decomp%ny
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    i_u_lo = merge(2, 1, decomp%edge_w)
    i_u_hi = merge(nx, nx + 1, decomp%edge_e)
    j_v_lo = merge(2, 1, decomp%edge_s)
    j_v_hi = merge(ny, ny + 1, decomp%edge_n)
    call rho_bounds_r(ir0, ir1, jr0, jr1)
    ju0 = 1; ju1 = ny; iv0 = 1; iv1 = nx
    if (decomp%edge_s) ju0 = 0
    if (decomp%edge_n) ju1 = ny + 1
    if (decomp%edge_w) iv0 = 0
    if (decomp%edge_e) iv1 = nx + 1
    first = (iif == 1)
    lim = dry_lim(G_2D)
    call seam_ring(iz_lo, iz_hi, jz_lo, jz_hi)

    CUF_DO2
    do j = js, je
      do i = is, ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        kn = fast%knew
        ks = fast%kstp
        kb = fast%kbak
        ko = fast%kold
        if (i >= iz_lo .and. i <= iz_hi .and. j >= jz_lo .and. j <= jz_hi) zeta(i, j, kn) = zeta_new(i, j)
        zk = zeta(i, j, kn)
        if (first) then
          ! FIRST_2D_STEP: bak over istrR:iendR / jstrR:jendR (the north
          ! U-row j=ny+1 that set_HUV1 reads included), then zero the
          ! fast averages before this substep's contribution.
          if (i >= ir0 .and. i <= ir1 .and. j >= jr0 .and. j <= jr1) then
            DU_avg_bak(i, j) = DU_avg1(i, j) - BAK_MUL(EXTRAP_AVG_BAK_FAC, DU_avg2(i, j))
            DV_avg_bak(i, j) = DV_avg1(i, j) - BAK_MUL(EXTRAP_AVG_BAK_FAC, DV_avg2(i, j))
          endif
          DU_avg1(i, j) = 0.0_rp
          DV_avg1(i, j) = 0.0_rp
          DU_avg2(i, j) = 0.0_rp
          DV_avg2(i, j) = 0.0_rp
          zt_avg1(i, j) = 0.0_rp
        endif
        ! Physical-edge ghost cells are averaged too: zeta(knew)=zt_avg1
        ! there sets Hz at the boundary, as in the CPU (istrR:iendR).
        if (i >= ir0 .and. i <= ir1 .and. j >= jr0 .and. j <= jr1) &
             zt_avg1(i, j) = zt_avg1(i, j) + fast%w1 * zk
        if (i >= 1 .and. i <= nx + 1 .and. j >= ju0 .and. j <= ju1) &
             DU_avg2(i, j) = DU_avg2(i, j) + fast%w2 * DUon(i, j)
        if (i >= iv0 .and. i <= iv1 .and. j >= 1 .and. j <= ny + 1) &
             DV_avg2(i, j) = DV_avg2(i, j) + fast%w2 * DVom(i, j)
      enddo
    enddo
    ! Dnew and the PGF state at the first seam halo ring (the east/north
    ! tile's u(1)/v(1) update uses Dnew(0) and rzeta(0)) come from the
    ! zeta(knew) formed there above with the neighbour's arithmetic, so no
    ! exchange is needed here; step2d_tail exchanges zeta with ubar/vbar.
    CUF_DO2
    do j = js, je
      do i = is, ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        kn = fast%knew
        ks = fast%kstp
        kb = fast%kbak
        ko = fast%kold
        zk = zeta(i, j, kn)
        Dnew(i, j) = h(i, j) + zk
        ! fill_pgf_state(.false.)
        z_old = zeta(i, j, ks)
        zwrk(i, j) = fast%bkw_new * zk + fast%bkw * z_old &
                     + fast%bkw1 * zeta(i, j, kb) + fast%bkw2 * zeta(i, j, ko)
        call density_weighted_zeta(zwrk(i, j), rhoS(i, j), rhoA(i, j), &
                                   rzeta(i, j), rzetaSA(i, j))
        rzeta2(i, j) = rzeta(i, j) * zwrk(i, j)
      enddo
    enddo

    ! apply_pgf(.false.) for u and v, plus DUon = h + zeta(kstp).
    CUF_DO2
    do j = js, je
      do i = is, ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        ks = fast%kstp
        if (i >= 1 .and. i <= nx + 1 .and. j >= 1 .and. j <= ny) then
          term = barotropic_pgf(h(i - 1, j), h(i, j), &
                                rzeta(i - 1, j), rzeta(i, j), &
                                rzeta2(i - 1, j), rzeta2(i, j), &
                                rzetaSA(i - 1, j), rzetaSA(i, j), &
                                rhoA(i - 1, j), rhoA(i, j), &
                                zwrk(i - 1, j), zwrk(i, j), dn_u(i, j))
          rubar(i, j) = term
        endif
        if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny + 1) then
          term = barotropic_pgf(h(i, j - 1), h(i, j), &
                                rzeta(i, j - 1), rzeta(i, j), &
                                rzeta2(i, j - 1), rzeta2(i, j), &
                                rzetaSA(i, j - 1), rzetaSA(i, j), &
                                rhoA(i, j - 1), rhoA(i, j), &
                                zwrk(i, j - 1), zwrk(i, j), dm_v(i, j))
          rvbar(i, j) = term
        endif
        DUon(i, j) = h(i, j) + zeta(i, j, ks)
      enddo
    enddo

    if (first) then
      call dump_rvS('before subtract')
      call host_subtract_barotropic_from_rufrc()
      call dump_rvS('after subtract')
      call dump_rufrc_build('after subtract rubar')
    endif
    ! rufrc/rvfrc change only on the first fast step (the subtraction
    ! above; visc3d's additions are exchanged here too); later fast steps
    ! would copy the same halo values again.
    if (first) then
      call hg_begin()
      call hg_add(F_RUFRC)
      call hg_add(F_RVFRC)
      call hg_end()
      call fill_pgf_state(.true.)
      call apply_pgf(.true.)
    endif

    ! River faces away from the physical edges (rivers_mod RIV_S2DF): the
    ! CPU's override (ubar = river_flux*2/(dn_u*(Dnew(i-1)+Dnew(i))),
    ! DU_avg1 = river_flux) right after the update, in the same thread;
    ! rivers_ubar does the faces next to an edge after the open-boundary code.
    ! Two copies of the kernel: the river version uses ~30 more registers,
    ! so tiles without such faces (and runs without rivers) keep the plain one.
    rv_on = river_on
    if (rv_on) rv_on = (nrf_s2df > 0)
    if (rv_on) then
    CUF_DO2
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        ks = fast%kstp
        kn = fast%knew
        cff = fast%half_dt
        cff1 = fast%half_w1
        if (j <= ny .and. i >= i_u_lo .and. i <= i_u_hi) then
          DUnew = ((DUon(i, j) + DUon(i - 1, j)) * ubar(i, j, ks) + &
                   cff * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j)) * &
                   (rubar(i, j) + rufrc(i, j)))
          DUnew = apply_mask(DUnew, umask(i, j))
          ubar(i, j, kn) = DUnew / max(Dnew(i, j) + Dnew(i - 1, j), 1.0e-30_rp)
          DU_avg1(i, j) = DU_avg1(i, j) + cff1 * dn_u(i, j) * DUnew
          fr = riv_ufc(i, j)
          if (fr > 0) then
            flux = rf_flx(fr)                 ! river_flux of the face, this step
            ubar(i, j, kn) = flux * 2.0_rp / (dn_u(i, j) * (Dnew(i - 1, j) + Dnew(i, j)))
            DU_avg1(i, j) = flux
          endif
        endif
        if (i <= nx .and. j >= j_v_lo .and. j <= j_v_hi) then
          DVnew = ((DUon(i, j) + DUon(i, j - 1)) * vbar(i, j, ks) + &
                   cff * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1)) * &
                   (rvbar(i, j) + rvfrc(i, j)))
          DVnew = apply_mask(DVnew, vmask(i, j))
          vbar(i, j, kn) = DVnew / max(Dnew(i, j) + Dnew(i, j - 1), 1.0e-30_rp)
          DV_avg1(i, j) = DV_avg1(i, j) + cff1 * dm_v(i, j) * DVnew
          fr = riv_vfc(i, j)
          if (fr > 0) then
            flux = rf_flx(fr)
            vbar(i, j, kn) = flux * 2.0_rp / (dm_v(i, j) * (Dnew(i, j - 1) + Dnew(i, j)))
            DV_avg1(i, j) = flux
          endif
        endif
      enddo
    enddo
    else
    CUF_DO2
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        ks = fast%kstp
        kn = fast%knew
        cff = fast%half_dt
        cff1 = fast%half_w1
        if (j <= ny .and. i >= i_u_lo .and. i <= i_u_hi) then
          DUnew = ((DUon(i, j) + DUon(i - 1, j)) * ubar(i, j, ks) + &
                   cff * (pm(i, j) + pm(i - 1, j)) * (pn(i, j) + pn(i - 1, j)) * &
                   (rubar(i, j) + rufrc(i, j)))
          DUnew = apply_mask(DUnew, umask(i, j))
          ubar(i, j, kn) = DUnew / max(Dnew(i, j) + Dnew(i - 1, j), 1.0e-30_rp)
          DU_avg1(i, j) = DU_avg1(i, j) + cff1 * dn_u(i, j) * DUnew
        endif
        if (i <= nx .and. j >= j_v_lo .and. j <= j_v_hi) then
          DVnew = ((DUon(i, j) + DUon(i, j - 1)) * vbar(i, j, ks) + &
                   cff * (pm(i, j) + pm(i, j - 1)) * (pn(i, j) + pn(i, j - 1)) * &
                   (rvbar(i, j) + rvfrc(i, j)))
          DVnew = apply_mask(DVnew, vmask(i, j))
          vbar(i, j, kn) = DVnew / max(Dnew(i, j) + Dnew(i, j - 1), 1.0e-30_rp)
          DV_avg1(i, j) = DV_avg1(i, j) + cff1 * dm_v(i, j) * DVnew
        endif
      enddo
    enddo
    endif
  end subroutine step2d_mid

  subroutine host_begin_fast_averages()
    ! FIRST_2D_STEP: save the extrapolated bak averages and zero the fast
    ! averages. Device kernels (stream-ordered, no sync); the product is
    ! rounded before the subtract (BAK_MUL) to match the host arithmetic.
    integer(ip) :: i, j, is, ie, js, je
    integer(ip) :: ir0, ir1, jr0, jr1
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    ! CPU FIRST_2D_STEP writes bak over istrR:iendR / jstrR:jendR, including
    ! the north U-row j=ny+1 that set_HUV1 later reads.
    call rho_bounds_r(ir0, ir1, jr0, jr1)
    CUF_DO2
    do j = jr0, jr1
      do i = ir0, ir1
        DU_avg_bak(i, j) = DU_avg1(i, j) - BAK_MUL(EXTRAP_AVG_BAK_FAC, DU_avg2(i, j))
        DV_avg_bak(i, j) = DV_avg1(i, j) - BAK_MUL(EXTRAP_AVG_BAK_FAC, DV_avg2(i, j))
      enddo
    enddo
    CUF_DO2
    do j = js, je
      do i = is, ie
        DU_avg1(i, j) = 0.0_rp
        DV_avg1(i, j) = 0.0_rp
        DU_avg2(i, j) = 0.0_rp
        DV_avg2(i, j) = 0.0_rp
        zt_avg1(i, j) = 0.0_rp
      enddo
    enddo
  end subroutine host_begin_fast_averages

  subroutine host_subtract_barotropic_from_rufrc()
    integer(ip) :: i, j, nx, ny, lim
    nx = decomp%nx
    ny = decomp%ny
    lim = dry_lim(G_2D)
    CUF_DO2
    do j = 1, ny
      do i = 1, nx + 1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        rufrc(i, j) = rufrc(i, j) - rubar(i, j)
      enddo
    enddo
    CUF_DO2
    do j = 1, ny + 1
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        rvfrc(i, j) = rvfrc(i, j) - rvbar(i, j)
      enddo
    enddo
  end subroutine host_subtract_barotropic_from_rufrc

  subroutine host_uv_obc(with_avg)
    ! with_avg = .false.: step2d_tail runs accumulate_boundary_avg1 behind
    ! the exchange pack (it reads only physical-edge cells, which no
    ! exchange writes), so it overlaps the MPI transfer.
    logical, intent(in) :: with_avg
    call uv2dbc()
    call uv2dbc_corners()
    if (with_avg) call accumulate_boundary_avg1(fast_half_w1_h)
  end subroutine host_uv_obc

  subroutine boundary_avg1_now()
    call accumulate_boundary_avg1(fast_half_w1_h)
  end subroutine boundary_avg1_now

  subroutine rho_bounds_r(ir0, ir1, jr0, jr1)
    integer(ip), intent(out) :: ir0, ir1, jr0, jr1
    ir0 = merge(0, 1, decomp%edge_w)
    ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s)
    jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
  end subroutine rho_bounds_r

  subroutine step2d_last_zeta()
    integer(ip) :: i, j, kn, ir0, ir1, jr0, jr1
    call rho_bounds_r(ir0, ir1, jr0, jr1)
    CUF_DO2
    do j = jr0, jr1
      do i = ir0, ir1
        if (fast%is_last == 1) then
          kn = fast%knew
          zeta(i, j, kn) = zt_avg1(i, j)
        endif
      enddo
    enddo
  end subroutine step2d_last_zeta

  subroutine step2d_tail()
    ! zeta(knew) was exchanged inside step2d_mid before its halo PGF terms;
    ! only the last fast step rewrites it (zt_avg1). ubar and vbar go in one
    ! exchange per direction.
    if (fast_is_last_h) call step2d_last_zeta()
    call halo_fill_zuv_n(zeta, ubar, vbar, boundary_avg1_now)
  end subroutine step2d_tail

  subroutine seam_ring(iz_lo, iz_hi, jz_lo, jz_hi)
    ! Range of zeta_new: the interior plus the first halo ring on MPI seams
    ! (physical edges stay with the boundary conditions).
    integer(ip), intent(out) :: iz_lo, iz_hi, jz_lo, jz_hi
    iz_lo = merge(1_ip, 0_ip, decomp%edge_w); iz_hi = merge(decomp%nx, decomp%nx + 1, decomp%edge_e)
    jz_lo = merge(1_ip, 0_ip, decomp%edge_s); jz_hi = merge(decomp%ny, decomp%ny + 1, decomp%edge_n)
  end subroutine seam_ring

  subroutine fill_pgf_state(fb_correct)
    logical, intent(in) :: fb_correct
    integer(ip) :: i, j, is, ie, js, je, kn, ks, kb, ko, lim
    real(rp) :: z_new, z_old
    logical :: correct

    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    correct = fb_correct
    lim = dry_lim(G_2D)

    CUF_DO2
    do j = js, je
      do i = is, ie
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        kn = fast%knew
        ks = fast%kstp
        kb = fast%kbak
        ko = fast%kold
        z_new = zeta(i, j, kn)
        z_old = zeta(i, j, ks)
        if (correct) then
          zwrk(i, j) = z_new - z_old
        else
          zwrk(i, j) = fast%bkw_new * z_new + fast%bkw * z_old &
                       + fast%bkw1 * zeta(i, j, kb) + fast%bkw2 * zeta(i, j, ko)
        endif
        call density_weighted_zeta(zwrk(i, j), rhoS(i, j), rhoA(i, j), &
                                   rzeta(i, j), rzetaSA(i, j))
        if (correct) then
          rzeta2(i, j) = rzeta(i, j) * (z_new + z_old)
        else
          rzeta2(i, j) = rzeta(i, j) * zwrk(i, j)
        endif
      enddo
    enddo
  end subroutine fill_pgf_state

  subroutine apply_pgf(add)
    logical, intent(in) :: add
    integer(ip) :: i, j, nx, ny, lim
    real(rp) :: term
    logical :: do_add

    nx = decomp%nx
    ny = decomp%ny
    do_add = add
    lim = dry_lim(G_2D)

    CUF_DO2
    do j = 1, ny
      do i = 1, nx + 1
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        term = barotropic_pgf(h(i - 1, j), h(i, j), &
                              rzeta(i - 1, j), rzeta(i, j), &
                              rzeta2(i - 1, j), rzeta2(i, j), &
                              rzetaSA(i - 1, j), rzetaSA(i, j), &
                              rhoA(i - 1, j), rhoA(i, j), &
                              zwrk(i - 1, j), zwrk(i, j), dn_u(i, j))
        if (do_add) then
          rubar(i, j) = rubar(i, j) + term
        else
          rubar(i, j) = term
        endif
      enddo
    enddo
    CUF_DO2
    do j = 1, ny + 1
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        term = barotropic_pgf(h(i, j - 1), h(i, j), &
                              rzeta(i, j - 1), rzeta(i, j), &
                              rzeta2(i, j - 1), rzeta2(i, j), &
                              rzetaSA(i, j - 1), rzetaSA(i, j), &
                              rhoA(i, j - 1), rhoA(i, j), &
                              zwrk(i, j - 1), zwrk(i, j), dm_v(i, j))
        if (do_add) then
          rvbar(i, j) = rvbar(i, j) + term
        else
          rvbar(i, j) = term
        endif
      enddo
    enddo
  end subroutine apply_pgf

  subroutine accumulate_boundary_avg1(cff1)
    ! One launch over the longest edge; the four edges write disjoint
    ! cells, so the pieces of the former per-edge kernels run together.
    real(rp), intent(in) :: cff1
    integer(ip) :: m, nx, ny, kn, mmax, jr0, jr1, ir0, ir1, jv_lo, iu_lo
    real(rp) :: w
    logical :: ew, ee, es, en
    nx = decomp%nx
    ny = decomp%ny
    kn = knew
    w = cff1
    ew = decomp%edge_w; ee = decomp%edge_e; es = decomp%edge_s; en = decomp%edge_n
    if (.not. (ew .or. ee .or. es .or. en)) return
    mmax = max(nx, ny) + 1
    ! Match CPU step2d: WEST/EAST DU_avg1 over jstrR:jendR, DV_avg1 over
    ! jstrV:jend; NORTH/SOUTH DU_avg1 over istrU:iend, DV_avg1 over
    ! istrR:iendR. The CPU tile ranges: jstrR = 0 / jstrV = 2 only at a
    ! physical south edge (else 1: the seam row is this tile's interior),
    ! jendR = ny+1 only at a physical north edge (else ny), and likewise in i.
    call rho_bounds_r(ir0, ir1, jr0, jr1)
    jv_lo = merge(2, 1, es)
    iu_lo = merge(2, 1, ew)
    CUF_DO1
    do m = 0, mmax
      if (m >= jr0 .and. m <= jr1) then
        if (ew) then
          DU_avg1(1, m) = DU_avg1(1, m) + w * (Dnew(1, m) + Dnew(0, m)) * &
                          ubar(1, m, kn) * dn_u(1, m)
        endif
        if (ee) then
          DU_avg1(nx + 1, m) = DU_avg1(nx + 1, m) + w * (Dnew(nx + 1, m) + Dnew(nx, m)) * &
                               ubar(nx + 1, m, kn) * dn_u(nx + 1, m)
        endif
      endif
      if (m >= jv_lo .and. m <= ny) then
        if (ew) then
          DV_avg1(0, m) = DV_avg1(0, m) + w * (Dnew(0, m) + Dnew(0, m - 1)) * &
                          vbar(0, m, kn) * dm_v(0, m)
        endif
        if (ee) then
          DV_avg1(nx + 1, m) = DV_avg1(nx + 1, m) + w * (Dnew(nx + 1, m) + Dnew(nx + 1, m - 1)) * &
                               vbar(nx + 1, m, kn) * dm_v(nx + 1, m)
        endif
      endif
      if (m >= iu_lo .and. m <= nx) then
        if (es) then
          DU_avg1(m, 0) = DU_avg1(m, 0) + w * (Dnew(m, 0) + Dnew(m - 1, 0)) * &
                          ubar(m, 0, kn) * dn_u(m, 0)
        endif
        if (en) then
          DU_avg1(m, ny + 1) = DU_avg1(m, ny + 1) + w * (Dnew(m, ny + 1) + Dnew(m - 1, ny + 1)) * &
                               ubar(m, ny + 1, kn) * dn_u(m, ny + 1)
        endif
      endif
      if (m >= ir0 .and. m <= ir1) then
        if (es) then
          DV_avg1(m, 1) = DV_avg1(m, 1) + w * (Dnew(m, 1) + Dnew(m, 0)) * &
                          vbar(m, 1, kn) * dm_v(m, 1)
        endif
        if (en) then
          DV_avg1(m, ny + 1) = DV_avg1(m, ny + 1) + w * (Dnew(m, ny + 1) + Dnew(m, ny)) * &
                               vbar(m, ny + 1, kn) * dm_v(m, ny + 1)
        endif
      endif
    enddo
  end subroutine accumulate_boundary_avg1

end module step2d_mod

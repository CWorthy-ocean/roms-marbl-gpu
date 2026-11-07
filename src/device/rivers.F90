#include "mem_attr.inc"
#include "cuf_launch.inc"
module rivers_mod
  ! River sources (UCLA river_frc), on the device. A river cell is a land
  ! cell with river_index > 0; each of its faces to an ocean cell carries
  ! the fraction river_fraction / (number of such faces) of that river's
  ! volume flux. The CPU model handles these faces in six places, and the
  ! kernels below repeat its expressions there:
  !   set_HUV     the predictor transport FlxU/FlxV on the faces, from the
  !               river volume at time n (rivers_huv; CPU 2805cbef);
  !   step2d      ubar/vbar(knew) and DU/DV_avg1 on the faces, every fast
  !               step (rivers_ubar);
  !   pre_step3d  u/v(nnew) on the faces, before u3dbc (rivers_uv, RIV_PRE);
  !   step3d_uv2  u/v(nnew) on the faces, before the exchange (rivers_uv,
  !               RIV_COR);
  !   tracers     the horizontal tracer flux through the faces, predictor
  !               and corrector, for T/S (rivers_ts) and the passive
  !               tracers (rivers_tp).
  ! The time-stepping kernels mask every face flux with umask/vmask, which
  ! is 0 on a river face, so rivers_ts/_tp add the river flux divergence to
  ! the ocean cell next to each face after the horizontal pass (the two
  ! differ from the CPU's single expression by rounding only). They run
  ! per ocean cell, summing its river faces in a fixed order: no atomics,
  ! the same result on every run and every tiling.
  !
  ! River data: two records of river_volume and river_tracer for the
  ! rivers this tile touches, slot 1 = the earlier record. The kernels
  ! interpolate in time with the weights of the predictor (time n) or the
  ! fast loop and corrector (n+1/2), set by river_io from the CPU's
  ! set_frc_data formula. Records change once per record interval (daily),
  ! so the per-step work is kernels only.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: river_on, rivers_set_lists, rivers_set_record, rivers_copy_upper, rivers_set_weights
  public :: rivers_set_map
  public :: rivers_ubar, rivers_uv, rivers_ts, rivers_tp, rivers_huv
  public :: RIV_S2D, RIV_PRE, RIV_COR, RIV_S2DF, RIV_RNG
  ! for the fast-step override fused into step2d_mid's ubar/vbar kernel
  public :: rf_flx, riv_ufc, riv_vfc

  ! Stage bits of a face: the CPU's loop ranges differ by stage. A face in
  ! the fast-loop range gets RIV_S2DF when step2d_mid's ubar/vbar kernel
  ! applies its override (riv_ufc/riv_vfc point to it), RIV_S2D when
  ! rivers_ubar does: faces within two cells of a physical edge, so that
  ! the open-boundary code still sees them before the override, as the CPU.
  ! RIV_RNG: every face of the river scan (cells 0:nx+1 x 0:ny+1, as the
  ! CPU's calc_river_flux); rivers_huv tests the ranges of the loop it
  ! follows (set_HUV's) itself.
  integer, parameter :: RIV_S2D = 1, RIV_PRE = 2, RIV_COR = 4, RIV_S2DF = 8, RIV_RNG = 16

  logical :: river_on = .false.
  integer(ip) :: nrf = 0, nrc = 0, nrl = 0, nrt = 0   ! faces, cells, rivers, tracers of the river data
  integer(ip), public :: nrf_s2d = 0, nrf_s2df = 0      ! fast-loop faces done by rivers_ubar / fused
  real(rp) :: w_pre(2) = (/1.0_rp, 0.0_rp/), w_cor(2) = (/1.0_rp, 0.0_rp/)
  integer :: geo_stage = 0          ! stage whose face geometry rg_h/rg_v hold (rivers_ts)
  integer, parameter :: RIV_NKB = 4  ! level chunks per column in rivers_tp

  ! Faces: location (rf_i, rf_j), type (1 = u, 2 = v), river (local index),
  ! stage bits, and the signed fraction (riv_uflx - 10*iriver of the CPU).
  ! Cells: location, passive column (0: none), number of river faces and
  ! their face indices and signs (+1: west/south face, -1: east/north).
  ! rf_flx(f): river_flux = riv_vol*frac of face f at n+1/2, formed once per
  ! step (rivers_set_weights) for the fast loop and the corrector, which all
  ! use the same value. rc_r: the river of each cell face slot. rg_h/rg_v:
  ! the stage's face geometry (Hz(a)+Hz(b) per level, riv_vol*frac/riv_depth)
  ! from rivers_ts, reused by rivers_tp.
#ifdef USE_CUDA
  integer(ip), device, allocatable :: riv_ufc(:,:), riv_vfc(:,:)   ! face index on fused faces, else 0
  real(rp),    device, allocatable :: rf_flx(:), rg_h(:,:,:), rg_v(:,:)
  integer(ip), device, allocatable :: rc_r(:,:)
  integer(ip), device, allocatable :: rf_i(:), rf_j(:), rf_t(:), rf_r(:), rf_s(:)
  real(rp),    device, allocatable :: rf_frac(:)
  integer(ip), device, allocatable :: rc_i(:), rc_j(:), rc_col(:), rc_n(:), rc_f(:,:)
  real(rp),    device, allocatable :: rc_sg(:,:)
  real(rp),    device, allocatable :: rv_vol(:,:), rv_trc(:,:,:)
  ! row of rv_trc for passive tracer itp: its index in the river file's
  ! tracer axis (the CPU's tracer order), or the extra zero row for a
  ! CDR_LITE tracer the file does not carry (see river_io)
  integer(ip), device, allocatable :: rv_map(:)
#else
  integer(ip), allocatable :: riv_ufc(:,:), riv_vfc(:,:)
  real(rp),    allocatable :: rf_flx(:), rg_h(:,:,:), rg_v(:,:)
  integer(ip), allocatable :: rc_r(:,:)
  integer(ip), allocatable :: rf_i(:), rf_j(:), rf_t(:), rf_r(:), rf_s(:)
  real(rp),    allocatable :: rf_frac(:)
  integer(ip), allocatable :: rc_i(:), rc_j(:), rc_col(:), rc_n(:), rc_f(:,:)
  real(rp),    allocatable :: rc_sg(:,:)
  real(rp),    allocatable :: rv_vol(:,:), rv_trc(:,:,:)
  integer(ip), allocatable :: rv_map(:)
#endif

contains

  subroutine rivers_set_lists(nf, fi, fj, ft, fr, fs, ffrac, nc, ci, cj, ccol, cn, cf, csg, nloc, ntrr)
    ! Copies the host-built lists to the device and allocates the river
    ! data slots (filled by rivers_set_record).
    integer(ip), intent(in) :: nf, nc, nloc, ntrr
    integer(ip), intent(in) :: fi(:), fj(:), ft(:), fr(:), fs(:)
    real(rp),    intent(in) :: ffrac(:)
    integer(ip), intent(in) :: ci(:), cj(:), ccol(:), cn(:), cf(:,:)
    real(rp),    intent(in) :: csg(:,:)
    integer(ip), allocatable :: mu(:,:), mv(:,:), cr(:,:)
    integer(ip) :: m, c
    nrf = nf; nrc = nc; nrl = nloc; nrt = ntrr
    allocate(mu(decomp%is:decomp%ie, decomp%js:decomp%je), mv(decomp%is:decomp%ie, decomp%js:decomp%je))
    mu = 0; mv = 0
    nrf_s2d = count(iand(fs(1:nf), int(RIV_S2D, ip)) /= 0)
    nrf_s2df = count(iand(fs(1:nf), int(RIV_S2DF, ip)) /= 0)
    do m = 1, nf
      if (iand(fs(m), int(RIV_S2DF, ip)) == 0) cycle
      if (ft(m) == 1) then
        mu(fi(m), fj(m)) = m
      else
        mv(fi(m), fj(m)) = m
      endif
    enddo
    allocate(riv_ufc(decomp%is:decomp%ie, decomp%js:decomp%je), riv_vfc(decomp%is:decomp%ie, decomp%js:decomp%je))
    riv_ufc = mu; riv_vfc = mv
    deallocate(mu, mv)
    allocate(rf_i(max(nf, 1)), rf_j(max(nf, 1)), rf_t(max(nf, 1)), rf_r(max(nf, 1)), &
             rf_s(max(nf, 1)), rf_frac(max(nf, 1)))
    allocate(rc_i(max(nc, 1)), rc_j(max(nc, 1)), rc_col(max(nc, 1)), rc_n(max(nc, 1)), &
             rc_f(4, max(nc, 1)), rc_sg(4, max(nc, 1)))
    allocate(rv_vol(max(nloc, 1), 2), rv_trc(max(ntrr, 1), max(nloc, 1), 2))
    allocate(rf_flx(max(nf, 1)), rc_r(4, max(nc, 1)), rg_h(4, max(nc, 1), decomp%nz), rg_v(4, max(nc, 1)))
    allocate(cr(4, max(nc, 1)))
    cr = 1
    do c = 1, nc
      do m = 1, cn(c)
        cr(m, c) = fr(cf(m, c))
      enddo
    enddo
    rc_r = cr
    deallocate(cr)
    rf_flx = 0.0_rp; rg_h = 0.0_rp; rg_v = 0.0_rp
    if (nf > 0) then
      rf_i = fi(1:nf); rf_j = fj(1:nf); rf_t = ft(1:nf); rf_r = fr(1:nf)
      rf_s = fs(1:nf); rf_frac = ffrac(1:nf)
    endif
    if (nc > 0) then
      rc_i = ci(1:nc); rc_j = cj(1:nc); rc_col = ccol(1:nc); rc_n = cn(1:nc)
      rc_f = cf(:, 1:nc); rc_sg = csg(:, 1:nc)
    endif
    rv_vol = 0.0_rp
    rv_trc = 0.0_rp
    river_on = .true.
  end subroutine rivers_set_lists

  subroutine rivers_set_map(map)
    ! rv_trc row of each passive tracer (river_io).
    integer(ip), intent(in) :: map(:)
    if (allocated(rv_map)) deallocate(rv_map)
    allocate(rv_map(max(size(map), 1)))
    if (size(map) > 0) rv_map(1:size(map)) = map
  end subroutine rivers_set_map

  subroutine rivers_set_record(slot, vol, trc)
    ! One record of this tile's rivers into slot 1 or 2. The device is
    ! drained first: kernels already queued may still read the slot.
    integer,  intent(in) :: slot
    real(rp), intent(in) :: vol(:), trc(:,:)
    if (nrl == 0) return
    call sync_arch()
    rv_vol(1:nrl, slot) = vol(1:nrl)
    rv_trc(1:nrt, 1:nrl, slot) = trc(1:nrt, 1:nrl)
    call sync_arch()
  end subroutine rivers_set_record

  subroutine rivers_copy_upper()
    ! The bracket moved on by one record: the upper slot becomes the lower.
    if (nrl == 0) return
    call sync_arch()
    rv_vol(:, 1) = rv_vol(:, 2)
    rv_trc(:, :, 1) = rv_trc(:, :, 2)
    call sync_arch()
  end subroutine rivers_copy_upper

  subroutine rivers_set_weights(sbit, c1, c2)
    ! Time weights of slots 1 and 2: RIV_PRE for the predictor (time n),
    ! RIV_COR for the fast loop and the corrector (n+1/2).
    integer,  intent(in) :: sbit
    real(rp), intent(in) :: c1, c2
    integer(ip) :: f, n, r
    real(rp) :: d1, d2
    if (sbit == RIV_PRE) then
      w_pre(1) = c1; w_pre(2) = c2
    else
      w_cor(1) = c1; w_cor(2) = c2
      ! the fast loop's and the corrector's river_flux, once per step
      if (nrf == 0) return
      n = nrf; d1 = c1; d2 = c2
      CUF_DO1
      do f = 1, n
        r = rf_r(f)
        rf_flx(f) = (d1 * rv_vol(r, 1) + d2 * rv_vol(r, 2)) * rf_frac(f)
      enddo
    endif
  end subroutine rivers_set_weights

  subroutine rivers_ubar(kn)
    ! CPU step2d, after u2dbc/v2dbc: on every river face
    !   ubar(knew) = river_flux*2/(dn_u*(Dnew(i-1)+Dnew(i))), DU_avg1 = river_flux.
    ! Faces in the fast-loop update range (the seam face included, as the
    ! fast step forms it redundantly there).
    integer(ip), intent(in) :: kn
    integer(ip) :: f, i, j, n, k_n
    real(rp) :: flux
    if (nrf_s2d == 0) return
    n = nrf; k_n = kn
    CUF_DO1
    do f = 1, n
      if (iand(rf_s(f), RIV_S2D) == 0) cycle
      i = rf_i(f); j = rf_j(f)
      flux = rf_flx(f)
      if (rf_t(f) == 1) then
        ubar(i, j, k_n) = flux * 2.0_rp / (dn_u(i, j) * (Dnew(i - 1, j) + Dnew(i, j)))
        DU_avg1(i, j) = flux
      else
        vbar(i, j, k_n) = flux * 2.0_rp / (dm_v(i, j) * (Dnew(i, j - 1) + Dnew(i, j)))
        DV_avg1(i, j) = flux
      endif
    enddo
  end subroutine rivers_ubar

  subroutine rivers_huv(iu0, iu1, ju0, ju1, iv0, iv1, jv0, jv1)
    ! CPU set_HUV (2805cbef): on every river face in set_HUV's ranges (u
    ! faces iu0:iu1 x ju0:ju1, v faces iv0:iv1 x jv0:jv1) the predictor
    ! transport is the river transport at the predictor's forcing time
    ! (time n), distributed over the column as the river tracer flux is:
    !   FlxU = Hz_u * (riv_vol*frac / riv_depth),
    ! riv_depth the mean column depth of the two cells. u(nrhs) on a river
    ! face holds the previous step's river velocity (or 0 from an initial
    ! file without it), so a varying river would otherwise enter the
    ! predictor's volume (omega, the Hz predictor) with a different volume
    ! than its tracers. Called before set_HUV's exchange.
    integer(ip), intent(in) :: iu0, iu1, ju0, ju1, iv0, iv1, jv0, jv1
    integer(ip) :: f, k, i, j, r, n, nz, a0, a1, b0, b1, c0, c1_, d0, d1
    real(rp) :: c1, c2, q, dep, rflx
    if (nrf == 0) return
    n = nrf; nz = decomp%nz
    a0 = iu0; a1 = iu1; b0 = ju0; b1 = ju1; c0 = iv0; c1_ = iv1; d0 = jv0; d1 = jv1
    c1 = w_pre(1); c2 = w_pre(2)
    CUF_DO2
    do k = 1, nz
      do f = 1, n
        if (iand(rf_s(f), RIV_RNG) == 0) cycle
        i = rf_i(f); j = rf_j(f); r = rf_r(f)
        if (rf_t(f) == 1) then
          if (i < a0 .or. i > a1 .or. j < b0 .or. j > b1) cycle
          q = (c1 * rv_vol(r, 1) + c2 * rv_vol(r, 2)) * rf_frac(f)
          dep = 0.5_rp * (z_w(i - 1, j, nz) - z_w(i - 1, j, 0) + z_w(i, j, nz) - z_w(i, j, 0))
          rflx = q / dep
          FlxU(i, j, k) = Hz_u(i, j, k) * rflx
        else
          if (i < c0 .or. i > c1_ .or. j < d0 .or. j > d1) cycle
          q = (c1 * rv_vol(r, 1) + c2 * rv_vol(r, 2)) * rf_frac(f)
          dep = 0.5_rp * (z_w(i, j - 1, nz) - z_w(i, j - 1, 0) + z_w(i, j, nz) - z_w(i, j, 0))
          rflx = q / dep
          FlxV(i, j, k) = Hz_v(i, j, k) * rflx
        endif
      enddo
    enddo
  end subroutine rivers_huv

  subroutine rivers_uv(sbit)
    ! CPU pre_step3d (RIV_PRE) and step3d_uv2 (RIV_COR): on every river
    ! face u(nnew) = riv_vol*frac/(dn_u*riv_depth) at all levels, with
    ! riv_depth the mean column depth of the two cells.
    integer, intent(in) :: sbit
    integer(ip) :: f, k, i, j, r, n, nz, n_new, sb
    real(rp) :: c1, c2, q, dep
    if (nrf == 0) return
    n = nrf; nz = decomp%nz; n_new = nnew; sb = sbit
    if (sbit == RIV_PRE) then
      c1 = w_pre(1); c2 = w_pre(2)
    else
      c1 = w_cor(1); c2 = w_cor(2)
    endif
    CUF_DO2
    do k = 1, nz
      do f = 1, n
        if (iand(rf_s(f), sb) == 0) cycle
        i = rf_i(f); j = rf_j(f); r = rf_r(f)
        q = (c1 * rv_vol(r, 1) + c2 * rv_vol(r, 2)) * rf_frac(f)
        if (rf_t(f) == 1) then
          dep = 0.5_rp * (z_w(i - 1, j, nz) - z_w(i - 1, j, 0) + z_w(i, j, nz) - z_w(i, j, 0))
          u(i, j, k, n_new) = q / (dn_u(i, j) * dep)
        else
          dep = 0.5_rp * (z_w(i, j - 1, nz) - z_w(i, j - 1, 0) + z_w(i, j, nz) - z_w(i, j, 0))
          v(i, j, k, n_new) = q / (dm_v(i, j) * dep)
        endif
      enddo
    enddo
  end subroutine rivers_uv

  ! The cell kernels: rivers_ts runs one thread per (level, ocean cell) and
  ! forms the geometry of the cell's river faces (up to 4: river, sign,
  ! Hz(a)+Hz(b), riv_vol*frac/riv_depth) once for all its tracers, and keeps
  ! it for rivers_tp in the same stage (Hz and z_w do not change in
  ! between). FX = riv_trc*0.5*(Hz(a)+Hz(b))*vel and the sum over the faces
  ! keep the CPU's expression and order. (The first version, one thread per
  ! tracer forming the geometry itself, took 227 us per call for 32 passives
  ! on an Iceland0 tile.) With T and S only, a thread per level is fastest:
  ! level chunks as in rivers_tp took 105 us per call on MiniPac rank 2
  ! against 64 us. rivers_tp, with 32 tracers per cell, splits a column into
  ! RIV_NKB chunks: 309 us per call with a thread per level, 144 us with
  ! chunks of 10 levels, 137 us with 25, 158 us with 100 (nz = 100, 2181
  ! cells).

  subroutine rivers_ts(sbit, n_new, dtp, itr0, itr1)
    ! River flux divergence into T/S (tracers itr0..itr1) of slot n_new,
    ! after the horizontal advection: t += dtp*pm*pn*sum(sign*FX).
    integer,     intent(in) :: sbit
    integer(ip), intent(in) :: n_new, itr0, itr1
    real(rp),    intent(in) :: dtp
    integer(ip) :: c, k, f, i, j, ic, jc, itr, n, nz, nn, i0, i1, nm, r1, r2, r3, r4
    real(rp) :: c1, c2, dtl, dtpp, s, q, dep, tr
    real(rp) :: s1, s2, s3, s4, h1, h2, h3, h4, v1, v2, v3, v4
    if (nrc == 0) return
    n = nrc; nz = decomp%nz; nn = n_new; dtl = dtp; i0 = itr0; i1 = itr1
    if (sbit == RIV_PRE) then
      c1 = w_pre(1); c2 = w_pre(2)
    else
      c1 = w_cor(1); c2 = w_cor(2)
    endif
    CUF_DO2
    do k = 1, nz
      do c = 1, n
        ic = rc_i(c); jc = rc_j(c)
        dtpp = dtl * pm(ic, jc) * pn(ic, jc)
        nm = rc_n(c)
        r2 = 1; r3 = 1; r4 = 1
        s2 = 0.0_rp; s3 = 0.0_rp; s4 = 0.0_rp; h2 = 0.0_rp; h3 = 0.0_rp; h4 = 0.0_rp
        v2 = 0.0_rp; v3 = 0.0_rp; v4 = 0.0_rp
        f = rc_f(1, c)
        i = rf_i(f); j = rf_j(f); r1 = rf_r(f); s1 = rc_sg(1, c)
        q = (c1 * rv_vol(r1, 1) + c2 * rv_vol(r1, 2)) * rf_frac(f)
        if (rf_t(f) == 1) then
          dep = 0.5_rp * (z_w(i - 1, j, nz) - z_w(i - 1, j, 0) + z_w(i, j, nz) - z_w(i, j, 0))
          h1 = Hz(i - 1, j, k) + Hz(i, j, k)
        else
          dep = 0.5_rp * (z_w(i, j - 1, nz) - z_w(i, j - 1, 0) + z_w(i, j, nz) - z_w(i, j, 0))
          h1 = Hz(i, j - 1, k) + Hz(i, j, k)
        endif
        v1 = q / dep
        if (nm >= 2) then
          f = rc_f(2, c)
          i = rf_i(f); j = rf_j(f); r2 = rf_r(f); s2 = rc_sg(2, c)
          q = (c1 * rv_vol(r2, 1) + c2 * rv_vol(r2, 2)) * rf_frac(f)
          if (rf_t(f) == 1) then
            dep = 0.5_rp * (z_w(i - 1, j, nz) - z_w(i - 1, j, 0) + z_w(i, j, nz) - z_w(i, j, 0))
            h2 = Hz(i - 1, j, k) + Hz(i, j, k)
          else
            dep = 0.5_rp * (z_w(i, j - 1, nz) - z_w(i, j - 1, 0) + z_w(i, j, nz) - z_w(i, j, 0))
            h2 = Hz(i, j - 1, k) + Hz(i, j, k)
          endif
          v2 = q / dep
        endif
        if (nm >= 3) then
          f = rc_f(3, c)
          i = rf_i(f); j = rf_j(f); r3 = rf_r(f); s3 = rc_sg(3, c)
          q = (c1 * rv_vol(r3, 1) + c2 * rv_vol(r3, 2)) * rf_frac(f)
          if (rf_t(f) == 1) then
            dep = 0.5_rp * (z_w(i - 1, j, nz) - z_w(i - 1, j, 0) + z_w(i, j, nz) - z_w(i, j, 0))
            h3 = Hz(i - 1, j, k) + Hz(i, j, k)
          else
            dep = 0.5_rp * (z_w(i, j - 1, nz) - z_w(i, j - 1, 0) + z_w(i, j, nz) - z_w(i, j, 0))
            h3 = Hz(i, j - 1, k) + Hz(i, j, k)
          endif
          v3 = q / dep
        endif
        if (nm >= 4) then
          f = rc_f(4, c)
          i = rf_i(f); j = rf_j(f); r4 = rf_r(f); s4 = rc_sg(4, c)
          q = (c1 * rv_vol(r4, 1) + c2 * rv_vol(r4, 2)) * rf_frac(f)
          if (rf_t(f) == 1) then
            dep = 0.5_rp * (z_w(i - 1, j, nz) - z_w(i - 1, j, 0) + z_w(i, j, nz) - z_w(i, j, 0))
            h4 = Hz(i - 1, j, k) + Hz(i, j, k)
          else
            dep = 0.5_rp * (z_w(i, j - 1, nz) - z_w(i, j - 1, 0) + z_w(i, j, nz) - z_w(i, j, 0))
            h4 = Hz(i, j - 1, k) + Hz(i, j, k)
          endif
          v4 = q / dep
        endif
        rg_h(1, c, k) = h1; rg_h(2, c, k) = h2; rg_h(3, c, k) = h3; rg_h(4, c, k) = h4
        if (k == 1) then
          rg_v(1, c) = v1; rg_v(2, c) = v2; rg_v(3, c) = v3; rg_v(4, c) = v4
        endif
        do itr = i0, i1
          s = 0.0_rp
          tr = c1 * rv_trc(itr, r1, 1) + c2 * rv_trc(itr, r1, 2)
          s = s + s1 * (tr * 0.5_rp * h1 * v1)
          if (nm >= 2) then
            tr = c1 * rv_trc(itr, r2, 1) + c2 * rv_trc(itr, r2, 2)
            s = s + s2 * (tr * 0.5_rp * h2 * v2)
          endif
          if (nm >= 3) then
            tr = c1 * rv_trc(itr, r3, 1) + c2 * rv_trc(itr, r3, 2)
            s = s + s3 * (tr * 0.5_rp * h3 * v3)
          endif
          if (nm >= 4) then
            tr = c1 * rv_trc(itr, r4, 1) + c2 * rv_trc(itr, r4, 2)
            s = s + s4 * (tr * 0.5_rp * h4 * v4)
          endif
          t(ic, jc, k, nn, itr) = t(ic, jc, k, nn, itr) + dtpp * s
        enddo
      enddo
    enddo
    geo_stage = sbit
  end subroutine rivers_ts

  subroutine rivers_tp(tp, ntr, sbit, n_new, dtp)
    ! As rivers_ts for the passive tracers in their column storage; passive
    ! tracer itr takes row rv_map(itr) of the river data (the CPU's order).
    ! One thread per (cell, tracer, chunk of levels), the tracers
    ! across the warp, so tp and rv_trc are read and written contiguously.
    ! Everything but the layer thickness is the same at every level: the
    ! river tracer at this stage's time (times 0.5, which is exact), the
    ! sign and q/dep of each face are loaded once per chunk; the loop over
    ! the chunk reads only rg_h (stored by this stage's rivers_ts) and tp.
    ! The product keeps the order of the per-level form, ((tr*0.5)*h)*v.
    integer(ip), intent(in) :: ntr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer,     intent(in) :: sbit
    integer(ip), intent(in) :: n_new
    real(rp),    intent(in) :: dtp
    integer(ip) :: c, k, kb, ic, jc, itr, n, nz, nn, col, nta, nm, r, kch, nkb, k0, k1, rw
    real(rp) :: c1, c2, dtl, dtpp, s
    real(rp) :: t1, t2, t3, t4, g1, g2, g3, g4, v1, v2, v3, v4
    if (nrc == 0 .or. ntr <= 0) return
    if (geo_stage /= sbit) error stop 'rivers_tp: no face geometry for this stage (rivers_ts runs first)'
    n = nrc; nz = decomp%nz; nn = n_new; dtl = dtp; nta = ntr
    kch = (nz + RIV_NKB - 1) / RIV_NKB; nkb = (nz + kch - 1) / kch
    if (sbit == RIV_PRE) then
      c1 = w_pre(1); c2 = w_pre(2)
    else
      c1 = w_cor(1); c2 = w_cor(2)
    endif
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, (32, 4, 1), 0, work_stream>>>
#endif
    do kb = 1, nkb
      do c = 1, n
        do itr = 1, nta
          col = rc_col(c)
          if (col == 0) cycle
          ic = rc_i(c); jc = rc_j(c)
          dtpp = dtl * pm(ic, jc) * pn(ic, jc)
          nm = rc_n(c)
          rw = rv_map(itr)
          r = rc_r(1, c)
          t1 = (c1 * rv_trc(rw, r, 1) + c2 * rv_trc(rw, r, 2)) * 0.5_rp
          g1 = rc_sg(1, c); v1 = rg_v(1, c)
          t2 = 0.0_rp; g2 = 0.0_rp; v2 = 0.0_rp
          t3 = 0.0_rp; g3 = 0.0_rp; v3 = 0.0_rp
          t4 = 0.0_rp; g4 = 0.0_rp; v4 = 0.0_rp
          if (nm >= 2) then
            r = rc_r(2, c)
            t2 = (c1 * rv_trc(rw, r, 1) + c2 * rv_trc(rw, r, 2)) * 0.5_rp
            g2 = rc_sg(2, c); v2 = rg_v(2, c)
          endif
          if (nm >= 3) then
            r = rc_r(3, c)
            t3 = (c1 * rv_trc(rw, r, 1) + c2 * rv_trc(rw, r, 2)) * 0.5_rp
            g3 = rc_sg(3, c); v3 = rg_v(3, c)
          endif
          if (nm >= 4) then
            r = rc_r(4, c)
            t4 = (c1 * rv_trc(rw, r, 1) + c2 * rv_trc(rw, r, 2)) * 0.5_rp
            g4 = rc_sg(4, c); v4 = rg_v(4, c)
          endif
          k0 = (kb - 1) * kch + 1; k1 = min(kb * kch, nz)
          do k = k0, k1
            s = 0.0_rp
            s = s + g1 * (t1 * rg_h(1, c, k) * v1)
            if (nm >= 2) s = s + g2 * (t2 * rg_h(2, c, k) * v2)
            if (nm >= 3) s = s + g3 * (t3 * rg_h(3, c, k) * v3)
            if (nm >= 4) s = s + g4 * (t4 * rg_h(4, c, k) * v4)
            tp(itr, col, k, nn) = real(real(tp(itr, col, k, nn), rp) + dtpp * s, pp)
          enddo
        enddo
      enddo
    enddo
  end subroutine rivers_tp

end module rivers_mod

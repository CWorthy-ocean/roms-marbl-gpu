#include "mem_attr.inc"
#include "cuf_launch.inc"
module cdr_src_mod
  ! CDR releases (UCLA cdr_frc, parameterized releases with cdr_trcflx) on
  ! the device. A release spreads its tracer flux [mmol/s] over cells and
  ! levels with weights w(k) (a Gaussian in the horizontal and the vertical,
  ! normalized so the weights of one release sum to 1 over the domain);
  ! cdr_io builds them on the host. As the CPU (step3d_t, per tracer after
  ! the horizontal advection), the corrector adds to the thickness-weighted
  ! tracer of slot nnew
  !   t += dt*pm*pn*w(k)*flux(release, tracer)
  ! for every release of the cell, in release order, with the flux at the
  ! corrector's forcing time (n+1/2): c1*f1 + c2*f2 of two records, or the
  ! earlier record without time interpolation (c1 = 1, c2 = 0).
  ! The CPU adds to every tracer at every level. Where the flux of a tracer
  ! is 0 in both records, or the weight of a level is 0 in every profile of
  ! the cell, its addition is an exact zero, so those are skipped: the
  ! kernel visits only the active tracers (nonzero flux in either record)
  ! and each cell's levels from its first to its last nonzero weight.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use arch_mod, only: sync_arch
  use halo_mod, only: halo_fill_2d
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: cdr_on, cdr_set_lists, cdr_set_record, cdr_copy_upper, cdr_set_weights, cdr_apply_tp
  public :: cdr_src_out
  public :: cdr_mode, CDR_MODE_FLUX, CDR_MODE_VOL, CDR_MODE_DP, CDR_MODE_3D
  public :: cdr_set_vol, cdr_set_record_vol, cdr_apply_ts, cdr_zsrc, cdr_vol_on
  public :: cdr_fs_src
  public :: cdr_cmap, cd_vol, cq_w, cc_p0, cq_rel
  public :: cdr_set_3d, cdr_3d_buf, cdr_src_out3d
  public :: cdr_set_dp, cdr_dp_gather_hz, cdr_set_dp_weights, cdr_dp_hzc

  integer, parameter :: CDR_NKB = 4                    ! chunks of levels per (cell, tracer)
  logical :: cdr_on = .false.
  ! Release mode: parameterized tracer fluxes (step 3), or depth profiles
  ! (cdr_forcing_depth_profiles: one cell per release, weights per tracer
  ! from the host remap every corrector, flux 1 as the CPU's cdr_flx).
  ! 3D forcing (cdr_forcing_3d): gridded ALK and DIC fluxes [mmol/s per
  ! cell], two records of this tile's interior in cdr_3d_buf(i, j, k, m, slot)
  ! (m = 1 ALK, 2 DIC), interpolated in the apply kernel.
  ! Volume releases (cdr_volume): water cd_v(r, slot) [m3/s] carrying
  ! tracers at concentrations cd_flx (passive) and cd_ts (T, S); the flux is
  ! vol*conc with both interpolated in time (CPU set_cdr_frc), and the
  ! water also enters zeta (step2d, every fast step, through cdr_zsrc) and
  ! Wi (omega) with the release's weights; cd_vol(r) is the volume at the
  ! latest set_cdr.
  integer, parameter :: CDR_MODE_FLUX = 1, CDR_MODE_VOL = 2, CDR_MODE_DP = 3, CDR_MODE_3D = 4
  integer :: cdr_mode = CDR_MODE_FLUX
  ! depth profiles: mapped passive tracers cdp_itr(1:ndp), weights
  ! cq_wt(k, a, q) of mapped tracer a in profile q, and the release
  ! columns' layer thicknesses gathered for the host (cdr_dp_hzc(k, c))
  integer(ip) :: ndp = 0
  integer(ip), allocatable :: cdp_itr_h(:)
  integer(ip) :: ncc = 0, ncq = 0, nrel = 0, ntrf = 0   ! cells, profiles, releases, passive tracers
  integer(ip) :: nact = 0                              ! active tracers
  real(rp) :: w_cor(2) = (/1.0_rp, 0.0_rp/)
  real(rp), allocatable :: h_flx(:,:,:)                ! host copy of cd_flx

  ! Cells of this tile with a release: local (i, j), passive column, the
  ! levels cc_k0 .. cc_k1 with a nonzero weight, and their profiles
  ! cq_p0(c) .. cq_p0(c+1)-1 (release order): release cq_rel and weights
  ! cq_w(k, q). cd_flx(itp, release, slot): two flux records of each
  ! passive tracer, slot 1 the earlier. ca_itr(1:nact): the active tracers.
  real(rp), WORK_ATTR allocatable :: cdr_dp_hzc(:,:)
  real(rp), WORK_ATTR allocatable :: cdr_3d_buf(:,:,:,:,:)
  logical :: cdr_vol_on = .false.
  real(rp), allocatable :: h_ts(:,:,:)
  ! Volume releases: the free-surface source [m/s], sum over the cell's
  ! profiles of pm*pn*vol*sum_k(w), at the latest set_cdr, on the owned
  ! cells and exchanged (CPU cdr_frc.F90 cdr_zsrc, c859943a); step2d adds
  ! dtfast*cdr_zsrc to zeta_new, the halo ring included.
  real(rp), WORK_ATTR allocatable :: cdr_zsrc(:,:)
#ifdef USE_CUDA
  real(rp),    device, allocatable :: cd_v(:,:), cd_vol(:), cd_ts(:,:,:), cq_qsum(:)
  integer(ip), device, allocatable :: cdr_cmap(:,:)
#else
  real(rp),    allocatable :: cd_v(:,:), cd_vol(:), cd_ts(:,:,:), cq_qsum(:)
  integer(ip), allocatable :: cdr_cmap(:,:)
#endif
  integer(ip) :: c3_alk = 0, c3_dic = 0
#ifdef USE_CUDA
  integer(ip), device, allocatable :: cdp_itr(:)
  real(rp),    device, allocatable :: cq_wt(:,:,:)
#else
  integer(ip), allocatable :: cdp_itr(:)
  real(rp),    allocatable :: cq_wt(:,:,:)
#endif
#ifdef USE_CUDA
  integer(ip), device, allocatable :: cc_i(:), cc_j(:), cc_col(:), cc_k0(:), cc_k1(:), cc_p0(:), cq_rel(:)
  integer(ip), device, allocatable :: ca_itr(:)
  real(rp),    device, allocatable :: cq_w(:,:), cd_flx(:,:,:)
#else
  integer(ip), allocatable :: cc_i(:), cc_j(:), cc_col(:), cc_k0(:), cc_k1(:), cc_p0(:), cq_rel(:)
  integer(ip), allocatable :: ca_itr(:)
  real(rp),    allocatable :: cq_w(:,:), cd_flx(:,:,:)
#endif

contains

  subroutine cdr_set_lists(nc, ci, cj, ccol, cp0, nq, qrel, qw, nr, nt_p)
    ! Copies the host-built cells and profiles (cdr_io) to the device and
    ! allocates the flux records (filled by cdr_set_record).
    integer(ip), intent(in) :: nc, nq, nr, nt_p
    integer(ip), intent(in) :: ci(:), cj(:), ccol(:), cp0(:), qrel(:)
    real(rp),    intent(in) :: qw(:,:)
    integer(ip), allocatable :: k0(:), k1(:)
    integer(ip) :: c, q, k, nz
    ncc = nc; ncq = nq; nrel = nr; ntrf = nt_p; nz = decomp%nz
    allocate(cc_i(max(nc, 1)), cc_j(max(nc, 1)), cc_col(max(nc, 1)), cc_p0(nc + 1))
    allocate(cc_k0(max(nc, 1)), cc_k1(max(nc, 1)))
    allocate(cq_rel(max(nq, 1)), cq_w(nz, max(nq, 1)))
    allocate(cd_flx(max(nt_p, 1), max(nr, 1), 2), h_flx(max(nt_p, 1), max(nr, 1), 2))
    allocate(ca_itr(max(nt_p, 1)))
    cc_i = 0; cc_j = 0; cc_col = 0; cc_k0 = 1; cc_k1 = 0; cq_rel = 1; cq_w = 0.0_rp
    cd_flx = 0.0_rp; h_flx = 0.0_rp; ca_itr = 1; nact = 0
    cc_p0 = cp0(1:nc + 1)
    if (nc > 0) then
      cc_i = ci(1:nc); cc_j = cj(1:nc); cc_col = ccol(1:nc)
      allocate(k0(nc), k1(nc))
      k0 = nz + 1; k1 = 0
      do c = 1, nc
        do q = cp0(c), cp0(c + 1) - 1
          do k = 1, nz
            if (qw(k, q) /= 0.0_rp) then
              k0(c) = min(k0(c), k); k1(c) = max(k1(c), k)
            endif
          enddo
        enddo
      enddo
      cc_k0 = k0; cc_k1 = k1
      deallocate(k0, k1)
    endif
    if (nq > 0) then
      cq_rel = qrel(1:nq); cq_w = qw(:, 1:nq)
    endif
    cdr_on = .true.
  end subroutine cdr_set_lists

  subroutine set_active()
    ! The tracers with a nonzero flux in either record, from the host copy.
    integer(ip) :: itr, act(max(ntrf, 1))
    nact = 0
    do itr = 1, ntrf
      if (any(h_flx(itr, 1:nrel, 1:2) /= 0.0_rp)) then
        nact = nact + 1
        act(nact) = itr
      endif
    enddo
    if (nact > 0) ca_itr(1:nact) = act(1:nact)
  end subroutine set_active

  subroutine cdr_set_record(slot, flx)
    ! One flux record (passive tracer, release) into slot 1 or 2. The device
    ! is drained first: kernels already queued may still read the slot.
    integer,  intent(in) :: slot
    real(rp), intent(in) :: flx(:,:)
    call sync_arch()
    h_flx(1:ntrf, 1:nrel, slot) = flx(1:ntrf, 1:nrel)
    cd_flx(1:ntrf, 1:nrel, slot) = flx(1:ntrf, 1:nrel)
    call set_active()
    call sync_arch()
  end subroutine cdr_set_record

  subroutine cdr_copy_upper()
    ! The bracket moved on by one record: the upper slot becomes the lower.
    call sync_arch()
    h_flx(:, :, 1) = h_flx(:, :, 2)
    cd_flx(:, :, 1) = cd_flx(:, :, 2)
    if (cdr_vol_on) then
      h_ts(:, :, 1) = h_ts(:, :, 2)
      cd_ts(:, :, 1) = cd_ts(:, :, 2)
      cd_v(:, 1) = cd_v(:, 2)
    endif
    call set_active()
    call sync_arch()
  end subroutine cdr_copy_upper

  subroutine cdr_set_weights(c1, c2)
    ! Time weights of the two records at the corrector's forcing time.
    real(rp), intent(in) :: c1, c2
    w_cor(1) = c1; w_cor(2) = c2
    if (cdr_vol_on) call cdr_vol_interp()
  end subroutine cdr_set_weights

  subroutine cdr_set_dp(nd, itrs)
    ! Depth profiles: the mapped passive tracers (after cdr_set_lists).
    integer(ip), intent(in) :: nd, itrs(:)
    cdr_mode = CDR_MODE_DP
    ndp = nd
    allocate(cdp_itr(max(nd, 1)), cq_wt(decomp%nz, max(nd, 1), max(ncq, 1)))
    allocate(cdr_dp_hzc(decomp%nz, max(ncc, 1)))
    allocate(cdp_itr_h(max(nd, 1)))
    cdp_itr_h = 1
    if (nd > 0) cdp_itr_h(1:nd) = itrs(1:nd)
    cdp_itr = cdp_itr_h
    cq_wt = 0.0_rp
    cdr_dp_hzc = 0.0_rp
  end subroutine cdr_set_dp

  subroutine cdr_dp_gather_hz()
    ! The layer thicknesses of the release columns into cdr_dp_hzc, then
    ! the device is drained so the host can read them.
    integer(ip) :: c, k, n, nz
    if (ncc == 0) return
    n = ncc; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do c = 1, n
      do k = 1, nz
        cdr_dp_hzc(k, c) = Hz(cc_i(c), cc_j(c), k)
      enddo
    enddo
    call sync_arch()
  end subroutine cdr_dp_gather_hz

  subroutine cdr_set_dp_weights(wt)
    ! This corrector's depth-profile weights wt(k, a, q) (the device is
    ! drained before the copy: kernels queued earlier may read them).
    real(rp), intent(in) :: wt(:,:,:)
    if (ncq == 0 .or. ndp == 0) return
    call sync_arch()
    cq_wt(:, 1:ndp, 1:ncq) = wt(:, 1:ndp, 1:ncq)
  end subroutine cdr_set_dp_weights

  subroutine cdr_set_vol(qsum, cmap)
    ! Volume releases (after cdr_set_lists): the weight sum of every profile
    ! (sum over k, for zeta) and the cell of every point (0: none; omega).
    real(rp), intent(in) :: qsum(:)
    integer(ip), intent(in) :: cmap(decomp%is:decomp%ie, decomp%js:decomp%je)
    cdr_mode = CDR_MODE_VOL
    cdr_vol_on = .true.
    allocate(cd_v(max(nrel, 1), 2), cd_vol(max(nrel, 1)), cd_ts(2, max(nrel, 1), 2), cq_qsum(max(ncq, 1)))
    allocate(cdr_cmap(decomp%is:decomp%ie, decomp%js:decomp%je), h_ts(2, max(nrel, 1), 2))
    allocate(cdr_zsrc(decomp%is:decomp%ie, decomp%js:decomp%je))
    cd_v = 0.0_rp; cd_vol = 0.0_rp; cd_ts = 0.0_rp; cq_qsum = 0.0_rp; h_ts = 0.0_rp
    cdr_zsrc = 0.0_rp
    if (ncq > 0) cq_qsum = qsum(1:ncq)
    cdr_cmap = cmap
  end subroutine cdr_set_vol

  subroutine cdr_set_record_vol(slot, conc, ts, vol)
    ! One volume record: passive concentrations conc(itp, r), T/S ts(2, r),
    ! volume vol(r), into slot 1 or 2 (device drained before the copy).
    integer,  intent(in) :: slot
    real(rp), intent(in) :: conc(:,:), ts(:,:), vol(:)
    call sync_arch()
    h_flx(1:ntrf, 1:nrel, slot) = conc(1:ntrf, 1:nrel)
    cd_flx(1:ntrf, 1:nrel, slot) = conc(1:ntrf, 1:nrel)
    h_ts(:, 1:nrel, slot) = ts(:, 1:nrel)
    cd_ts(:, 1:nrel, slot) = ts(:, 1:nrel)
    cd_v(1:nrel, slot) = vol(1:nrel)
    call set_active()
    call sync_arch()
  end subroutine cdr_set_record_vol

  subroutine cdr_vol_interp()
    ! cd_vol = c1*v1 + c2*v2 at the latest weights (CPU set_frc_data),
    ! stream-ordered before the kernels that read it; then cdr_zsrc from it
    ! (CPU set_cdr_zsrc, after every update of cdr_vol).
    integer(ip) :: r, n
    real(rp) :: c1, c2
    n = nrel; c1 = w_cor(1); c2 = w_cor(2)
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, *, 0, work_stream>>>
#endif
    do r = 1, n
      cd_vol(r) = c1 * cd_v(r, 1) + c2 * cd_v(r, 2)
    enddo
    call cdr_zsrc_set()
  end subroutine cdr_vol_interp

  subroutine cdr_zsrc_set()
    ! cdr_zsrc(i, j) = sum of ((pm*pn)*vol)*sum_k(w) over the profiles of
    ! the cell, in release order (CPU set_cdr_zsrc), on the owned release
    ! cells, then exchanged: the list holds owned cells only, and step2d
    ! forms zeta_new on the first halo ring of MPI seams too, where the
    ! neighbour's release must enter as well. Every rank exchanges, with or
    ! without release cells. Points that never hold a release stay 0.
    integer(ip) :: c, i, j, q, n, r
    real(rp) :: z
    n = ncc
    if (n > 0) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*, *, 0, work_stream>>>
#endif
      do c = 1, n
        i = cc_i(c); j = cc_j(c)
        z = 0.0_rp
        do q = cc_p0(c), cc_p0(c + 1) - 1
          r = cq_rel(q)
          z = z + pm(i, j) * pn(i, j) * cd_vol(r) * cq_qsum(q)
        enddo
        cdr_zsrc(i, j) = z
      enddo
    endif
    call halo_fill_2d(cdr_zsrc, STAGGER_RHO)
  end subroutine cdr_zsrc_set

  subroutine cdr_apply_ts(n_new, dtp, s0, s1)
    ! Volume releases: temperature and salinity (t(:,:,:,n_new,s0:s1)) take
    ! t += ((dt*pm*pn)*w)*(vol*conc) for every profile of the cell, in
    ! release order (the CPU's step3d_t for T and S).
    integer(ip), intent(in) :: n_new, s0, s1
    real(rp), intent(in) :: dtp
    integer(ip) :: c, k, i, j, q, n, nz, nn, r, sa, sb, s
    real(rp) :: dtl, cff, val, f, c1, c2
    if (.not. cdr_vol_on .or. ncc == 0) return
    n = ncc; nz = decomp%nz; nn = n_new; dtl = dtp; sa = s0; sb = s1
    c1 = w_cor(1); c2 = w_cor(2)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do c = 1, n
        i = cc_i(c); j = cc_j(c)
        cff = dtl * pm(i, j) * pn(i, j)
        do s = sa, sb
          val = t(i, j, k, nn, s)
          do q = cc_p0(c), cc_p0(c + 1) - 1
            r = cq_rel(q)
            f = cd_vol(r) * (c1 * cd_ts(s, r, 1) + c2 * cd_ts(s, r, 2))
            val = val + cff * cq_w(k, q) * f
          enddo
          t(i, j, k, nn, s) = val
        enddo
      enddo
    enddo
  end subroutine cdr_apply_ts

  subroutine cdr_set_3d(ialk, idic)
    ! 3D forcing: the passive indices of MARBL's ALK and DIC.
    integer(ip), intent(in) :: ialk, idic
    cdr_mode = CDR_MODE_3D
    c3_alk = ialk; c3_dic = idic
    allocate(cdr_3d_buf(decomp%nx, decomp%ny, decomp%nz, 2, 2))
    cdr_3d_buf = 0.0_rp
    cdr_on = .true.
  end subroutine cdr_set_3d

  subroutine cdr_apply_3d_tp(tp, ntr, n_new, dtp)
    ! 3D forcing (CPU step3d_t): t += ((dt*pm)*pn)*flx for ALK and DIC, flx
    ! = c1*f1 + c2*f2 of the two records (set_frc_data), on the wet columns
    ! of the interior.
    integer(ip), intent(in) :: ntr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: n_new
    real(rp),    intent(in) :: dtp
    integer(ip) :: c, k, i, j, m, nz, nn, itr, ia, id
    real(rp) :: dtl, c1, c2, f
    nz = decomp%nz; nn = n_new; dtl = dtp; c1 = w_cor(1); c2 = w_cor(2); ia = c3_alk; id = c3_dic
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do c = 1, nwet
        i = wet_i(c); j = wet_j(c)
        do m = 1, 2
          itr = merge(ia, id, m == 1)
          f = c1 * cdr_3d_buf(i, j, k, m, 1) + c2 * cdr_3d_buf(i, j, k, m, 2)
          tp(itr, c, k, nn) = real(real(tp(itr, c, k, nn), rp) + dtl * pm(i, j) * pn(i, j) * f, pp)
        enddo
      enddo
    enddo
  end subroutine cdr_apply_3d_tp

  subroutine cdr_src_out3d(a, m, om, cf, inst)
    ! CDR output, 3D forcing (CPU calc_cdr_source / add_cdr_source_sums): the
    ! ALK (m = 1) or DIC (m = 2) source c1*f1 + c2*f2 on the interior;
    ! inst: a = source, otherwise a = a*om + source*cf.
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    integer(ip), intent(in) :: m
    real(rp), intent(in) :: om, cf
    logical, intent(in) :: inst
    integer(ip) :: i, j, k, nx, ny, nz, mm, md
    real(rp) :: c1, c2, src
    if (cdr_mode /= CDR_MODE_3D) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; mm = m; c1 = w_cor(1); c2 = w_cor(2)
    md = merge(1_ip, 0_ip, inst)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          src = c1 * cdr_3d_buf(i, j, k, mm, 1) + c2 * cdr_3d_buf(i, j, k, mm, 2)
          if (md == 1) then
            a(i, j, k) = src
          else
            a(i, j, k) = a(i, j, k) * om + src * cf
          endif
        enddo
      enddo
    enddo
  end subroutine cdr_src_out3d

  subroutine cdr_apply_dp_tp(tp, ntr, n_new, dtp)
    ! Depth profiles: t += ((dt*pm*pn)*prf(k))*1 for each mapped tracer and
    ! each profile of the cell, in release order (the CPU's cidx loop with
    ! cdr_flx = 1); one thread per (mapped tracer, cell) over the levels.
    integer(ip), intent(in) :: ntr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: n_new
    real(rp),    intent(in) :: dtp
    integer(ip) :: c, k, i, j, a, m, nm, na, nz, nn, col, q, itr
    real(rp) :: dtl, cff, val
    if (ncc == 0 .or. ndp == 0) return
    na = ndp; nm = ndp * ncc; nz = decomp%nz; nn = n_new; dtl = dtp
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, 128, 0, work_stream>>>
#endif
    do m = 1, nm
      a = mod(m - 1, na) + 1
      c = (m - 1) / na + 1
      col = cc_col(c)
      if (col == 0) cycle
      itr = cdp_itr(a)
      i = cc_i(c); j = cc_j(c)
      cff = dtl * pm(i, j) * pn(i, j)
      do k = 1, nz
        val = real(tp(itr, col, k, nn), rp)
        do q = cc_p0(c), cc_p0(c + 1) - 1
          val = val + cff * cq_wt(k, a, q) * 1.0_rp
        enddo
        tp(itr, col, k, nn) = real(val, pp)
      enddo
    enddo
  end subroutine cdr_apply_dp_tp

  subroutine cdr_apply_tp(tp, ntr, n_new, dtp)
    ! The releases into the passive tracers of slot n_new (thickness-weighted
    ! after the corrector's horizontal pass). One thread per (active tracer
    ! and cell, chunk of levels), the (tracer, cell) pairs across the warp.
    ! At each level the cell's releases are added in release order, as the
    ! CPU's cidx loop does, each as t + ((dt*pm*pn)*w)*flux; the fluxes of
    ! the first two releases are formed once per thread.
    integer(ip), intent(in) :: ntr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: n_new
    real(rp),    intent(in) :: dtp
    integer(ip) :: c, k, i, j, itr, q, a, m, nm, na, nz, nn, col, kb, kch, nkb, k0, k1, p0, p1, r, vo
    real(rp) :: c1, c2, dtl, cff, val, f, f1, f2
    if (cdr_mode == CDR_MODE_DP) then
      if (cdr_on) call cdr_apply_dp_tp(tp, ntr, n_new, dtp)
      return
    endif
    if (cdr_mode == CDR_MODE_3D) then
      if (cdr_on) call cdr_apply_3d_tp(tp, ntr, n_new, dtp)
      return
    endif
    if (.not. cdr_on .or. ncc == 0 .or. nact == 0 .or. ntr <= 0) return
    na = nact; nm = nact * ncc; nz = decomp%nz; nn = n_new; dtl = dtp
    vo = merge(1_ip, 0_ip, cdr_vol_on)          ! volume releases: flux = vol*conc
    kch = (nz + CDR_NKB - 1) / CDR_NKB; nkb = (nz + kch - 1) / kch
    c1 = w_cor(1); c2 = w_cor(2)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, (32, 4, 1), 0, work_stream>>>
#endif
    do kb = 1, nkb
      do m = 1, nm
        a = mod(m - 1, na) + 1
        c = (m - 1) / na + 1
        col = cc_col(c)
        if (col == 0) cycle
        k0 = max((kb - 1) * kch + 1, cc_k0(c))
        k1 = min(kb * kch, nz, cc_k1(c))
        if (k0 > k1) cycle
        itr = ca_itr(a)
        i = cc_i(c); j = cc_j(c)
        cff = dtl * pm(i, j) * pn(i, j)
        p0 = cc_p0(c); p1 = cc_p0(c + 1) - 1
        r = cq_rel(p0)
        f1 = c1 * cd_flx(itr, r, 1) + c2 * cd_flx(itr, r, 2)
        if (vo == 1) f1 = cd_vol(r) * f1
        f2 = 0.0_rp
        if (p1 > p0) then
          r = cq_rel(p0 + 1)
          f2 = c1 * cd_flx(itr, r, 1) + c2 * cd_flx(itr, r, 2)
          if (vo == 1) f2 = cd_vol(r) * f2
        endif
        do k = k0, k1
          val = real(tp(itr, col, k, nn), rp)
          val = val + cff * cq_w(k, p0) * f1
          if (p1 > p0) val = val + cff * cq_w(k, p0 + 1) * f2
          do q = p0 + 2, p1
            r = cq_rel(q)
            f = c1 * cd_flx(itr, r, 1) + c2 * cd_flx(itr, r, 2)
            if (vo == 1) f = cd_vol(r) * f
            val = val + cff * cq_w(k, q) * f
          enddo
          tp(itr, col, k, nn) = real(val, pp)
        enddo
      enddo
    enddo
  end subroutine cdr_apply_tp

  subroutine cdr_src_out(a, itr, om, cf, inst)
    ! CDR_LITE output (the CPU's calc_cdr_lite_source): the release source
    ! of passive tracer itr [per cell and level, the flux unit, mmol/s],
    ! sum over the cell's releases in release order of w(k)*flux, with the
    ! corrector's flux. inst: a = source on the release cells (the caller
    ! zeroes a first); otherwise the average a = a*om + source*cf (a stays
    ! 0 elsewhere, as the CPU's 0*(1-coef) + 0*coef).
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    integer(ip), intent(in) :: itr
    real(rp), intent(in) :: om, cf
    logical, intent(in) :: inst
    integer(ip) :: c, k, i, j, q, n, nz, it, r, md, ma, ia, dp, vo
    real(rp) :: c1, c2, src, f
    if (.not. cdr_on .or. ncc == 0 .or. cdr_mode == CDR_MODE_3D) return
    n = ncc; nz = decomp%nz; it = itr; c1 = w_cor(1); c2 = w_cor(2)
    md = merge(1_ip, 0_ip, inst)
    vo = merge(1_ip, 0_ip, cdr_vol_on)
    ! depth profiles: the source of a mapped tracer is sum of prf*1 (0 for
    ! the others), as the CPU's cdr_prf*cdr_flx
    dp = 0; ia = 0
    if (cdr_mode == CDR_MODE_DP) then
      dp = 1
      do ma = 1, ndp
        if (cdp_itr_h(ma) == itr) ia = ma
      enddo
    endif
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do c = 1, n
        i = cc_i(c); j = cc_j(c)
        src = 0.0_rp
        if (dp == 1) then
          if (ia > 0) then
            do q = cc_p0(c), cc_p0(c + 1) - 1
              src = src + cq_wt(k, ia, q) * 1.0_rp
            enddo
          endif
        else
          do q = cc_p0(c), cc_p0(c + 1) - 1
            r = cq_rel(q)
            f = c1 * cd_flx(it, r, 1) + c2 * cd_flx(it, r, 2)
            if (vo == 1) f = cd_vol(r) * f
            src = src + cq_w(k, q) * f
          enddo
        endif
        if (md == 1) then
          a(i, j, k) = src
        else
          a(i, j, k) = a(i, j, k) * om + src * cf
        endif
      enddo
    enddo
  end subroutine cdr_src_out

  subroutine cdr_fs_src(qs)
    ! Implicit free surface (implicit_fs.F90; CPU implicit_fs_mod, CDR_FORCING
    ! cdr_volume): the volume releases as a source Q [m/s] of the step,
    ! qs += pm pn vol sum_k(w) for every profile of the cell, in release order.
#ifdef USE_CUDA
    real(rp), device, intent(inout) :: qs(decomp%is:decomp%ie, decomp%js:decomp%je)
#else
    real(rp), intent(inout) :: qs(decomp%is:decomp%ie, decomp%js:decomp%je)
#endif
    integer(ip) :: c, i, j, q, n, r
    if (.not. cdr_vol_on .or. ncc == 0) return
    n = ncc
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, *, 0, work_stream>>>
#endif
    do c = 1, n
      i = cc_i(c); j = cc_j(c)
      do q = cc_p0(c), cc_p0(c + 1) - 1
        r = cq_rel(q)
        qs(i, j) = qs(i, j) + pm(i, j) * pn(i, j) * cd_vol(r) * cq_qsum(q)
      enddo
    enddo
  end subroutine cdr_fs_src

end module cdr_src_mod

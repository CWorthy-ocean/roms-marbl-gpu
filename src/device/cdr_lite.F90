#include "mem_attr.inc"
#include "cuf_launch.inc"
module cdr_lite_mod
  ! CDR_LITE surface forcing on the device (CPU tracers.F90: CDR_LITE in
  ! set_surf_tracer_flx, set_gas_transfer_velocity,
  ! subtract_gas_exchange_from_tracer_flx). Each CDR_LITE DIC tracer
  ! (CDR_OAE_DIC<n>, CDR_DOR_DIC<n>) takes the surface flux
  !   stflx = <name>_flx + FG,   FG = -(k_gas/beta)*(DIC' - eta*ALK')*rmask
  ! with beta = ddic_dco2 and eta = ddic_dalk (no FG where beta is not
  ! > 0, NaN included) and ALK' the paired CDR_OAE_ALK<n> (0 for the DOR
  ! tracers); k_gas = xkw*(uwnd^2+vwnd^2)*sqrt(660/Sc) with the Schmidt
  ! number Sc a polynomial in SST, and k_gas = 0 where SST <= -1.8 C (the
  ! CPU's SEA_ICE_NOFLUX, always defined in set_global_definitions.h, as
  ! the bulk heat flux on the GPU). The ALK tracers have no surface flux.
  ! As the CPU, Sc, beta, eta and ALK' are single precision (default real
  ! in the CPU build) and the rest double. The CPU sets the flux in
  ! set_forces at n+1/2 from the predicted state (slot nrhs = 3), and only
  ! the corrector applies it (dt*stflx on the top layer after the vertical
  ! advection, before the implicit mixing; passive_column_run).
  ! cdrl_psf(itp, c): the surface flux of passive tracer itp at interior
  ! wet column c (0 for every tracer but the DIC tracers);
  ! cdrl_gas(i, j, m): FG of DIC tracer m (output).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: cdrl_on, cdrl_nd, cdrl_setup, cdrl_ensure, cdrl_sflx, cdrl_psf_full, psf_cflx
  public :: cdrl_psf, cdrl_ddco2, cdrl_ddalk, cdrl_flx, cdrl_gas, cdrl_dic_h, cdrl_alk_h
  public :: cdrl_out_alloc, cdrl_out_accum, cdrl_out_inst_h, cdrl_out_inst_iz, cdrl_out_zero
  public :: cdrl_out_gather3, cdrl_out_gather2
  public :: co_trc, co_h, co_iz, co_fg, co_src, co_tmp3, co_tmp2

  integer, parameter :: sp = kind(1.0)      ! the CPU build's default real
  logical :: cdrl_on = .false.
  integer(ip) :: cdrl_nd = 0                ! DIC tracers
  integer(ip), allocatable :: cdrl_dic_h(:), cdrl_alk_h(:)   ! passive index of DIC tracer m, its ALK (0: none)
#ifdef USE_CUDA
  integer(ip), device, allocatable :: cl_dic(:), cl_alk(:)
#else
  integer(ip), allocatable :: cl_dic(:), cl_alk(:)
#endif
  ! cdrl_psf is the passive corrector's one surface-flux array: CDR_LITE
  ! writes its DIC tracers' fluxes, the CFLX_CORR restoring (frc_io,
  ! psf_cflx) those of DIC, ALK and their _ALT_CO2 twins; 0 elsewhere.
  real(rp), WORK_ATTR allocatable :: cdrl_psf(:,:)          ! (ntp, nwet)
  logical :: psf_cflx = .false.
  real(rp), WORK_ATTR allocatable :: cdrl_ddco2(:,:), cdrl_ddalk(:,:)   ! (is:ie, js:je)
  real(rp), WORK_ATTR allocatable :: cdrl_flx(:,:,:), cdrl_gas(:,:,:)   ! (is:ie, js:je, nd)

  ! CDR_LITE output (cdr_lite_out): output tracers o = 1..no (passive
  ! co_itr(o)); averages of the tracers, the thickness-weighted tracers,
  ! their vertical integrals, the release sources (ns) and the DIC tracers'
  ! gas flux (nd); scratch fields for the instantaneous ones. The tracer,
  ! thickness-weighted and integral averages are kept over the wet columns
  ! the tile writes, tracer innermost as tp: co_trc(o, w, k), co_iz(o, w),
  ! with column w = 1..co_nw at tp column co_wcol(w), point (co_wi, co_wj);
  ! co_wmap(i, j) = w (0: land, whose entry w = 0 stays 0). Written
  ! records gather them back to (i, j, k).
  integer(ip) :: co_n = 0, co_nw = 0
#ifdef USE_CUDA
  integer(ip), device, allocatable :: co_itr(:), co_wcol(:), co_wi(:), co_wj(:), co_wmap(:,:)
#else
  integer(ip), allocatable :: co_itr(:), co_wcol(:), co_wi(:), co_wj(:), co_wmap(:,:)
#endif
  real(rp), WORK_ATTR allocatable :: co_trc(:,:,:), co_h(:,:,:), co_iz(:,:)
  real(rp), WORK_ATTR allocatable :: co_fg(:,:,:), co_src(:,:,:,:)
  real(rp), WORK_ATTR allocatable :: co_tmp3(:,:,:), co_tmp2(:,:)

contains

  subroutine cdrl_setup(nd, dic, alk, ntr)
    ! DIC tracer m is passive tracer dic(m), paired with ALK tracer alk(m)
    ! (0: none).
    integer(ip), intent(in) :: nd, ntr
    integer(ip), intent(in) :: dic(:), alk(:)
    integer :: is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    cdrl_nd = nd
    allocate(cdrl_dic_h(max(nd, 1)), cdrl_alk_h(max(nd, 1)), cl_dic(max(nd, 1)), cl_alk(max(nd, 1)))
    cdrl_dic_h = 1; cdrl_alk_h = 0
    if (nd > 0) then
      cdrl_dic_h(1:nd) = dic(1:nd); cdrl_alk_h(1:nd) = alk(1:nd)
    endif
    cl_dic = cdrl_dic_h; cl_alk = cdrl_alk_h
    if (allocated(cdrl_psf)) deallocate(cdrl_psf)
    allocate(cdrl_psf(max(ntr, 1), max(nwet, 1)))
    allocate(cdrl_ddco2(is:ie, js:je), cdrl_ddalk(is:ie, js:je))
    allocate(cdrl_flx(is:ie, js:je, max(nd, 1)), cdrl_gas(is:ie, js:je, max(nd, 1)))
    cdrl_psf = 0.0_rp; cdrl_ddco2 = 0.0_rp; cdrl_ddalk = 0.0_rp
    cdrl_flx = 0.0_rp; cdrl_gas = 0.0_rp
    cdrl_on = nd > 0
  end subroutine cdrl_setup

  subroutine cdrl_ensure()
    ! passive_column_run takes cdrl_psf in every run: a placeholder when
    ! there are no CDR_LITE DIC tracers.
    if (.not. allocated(cdrl_psf)) then
      allocate(cdrl_psf(1, 1))
      cdrl_psf = 0.0_rp
    endif
  end subroutine cdrl_ensure

  subroutine cdrl_psf_full(ntr)
    ! cdrl_psf at its full size (ntr, nwet), zero, unless it already is
    ! (CDR_LITE's cdrl_setup); cdrl_ensure leaves a (1, 1) placeholder.
    integer(ip), intent(in) :: ntr
    if (allocated(cdrl_psf)) then
      if (size(cdrl_psf, 1) == max(ntr, 1) .and. size(cdrl_psf, 2) == max(nwet, 1)) return
      deallocate(cdrl_psf)
    endif
    allocate(cdrl_psf(max(ntr, 1), max(nwet, 1)))
    cdrl_psf = 0.0_rp
  end subroutine cdrl_psf_full

  subroutine cdrl_sflx(tp, ntr, nrs, uw, vw, have_wind)
    ! Surface fluxes of the DIC tracers from slot nrs of T and the passive
    ! tracers, one thread per interior wet column. Without bulk winds
    ! (have_wind false) k_gas is 0.
    integer(ip), intent(in) :: ntr, nrs
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), WORK_ATTR intent(in) :: uw(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: vw(decomp%is:decomp%ie, decomp%js:decomp%je)
    logical, intent(in) :: have_wind
    real(sp), parameter :: a = 2116.8, b = -136.25, c3 = 4.7353, d4 = -0.092307, e5 = 0.0007555
    real(sp), parameter :: xkw_coef = 6.97e-07
    integer(ip) :: c, i, j, m, nz, nd, nr, itr, ia, hw
    real(sp) :: sst, sc, beta, eta, calk
    real(rp) :: w2, kg, g, rm
    if (.not. cdrl_on .or. nwet <= 0) return
    nz = decomp%nz; nd = cdrl_nd; nr = nrs
    hw = merge(1_ip, 0_ip, have_wind)
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, 128, 0, work_stream>>>
#endif
    do c = 1, nwet
      i = wet_i(c); j = wet_j(c)
      sst = real(t(i, j, nz, nr, itemp), sp)
      sc = a + sst * (b + sst * (c3 + sst * (d4 + e5 * sst)))
      w2 = 0.0_rp
      if (hw == 1) w2 = uw(i, j)**2 + vw(i, j)**2
      kg = real(xkw_coef, rp) * w2 * real(sqrt(660.0_sp / sc), rp)
      if (sst <= -1.8_sp) kg = 0.0_rp          ! SEA_ICE_NOFLUX (CPU set_global_definitions.h)
      beta = real(cdrl_ddco2(i, j), sp)
      eta = real(cdrl_ddalk(i, j), sp)
      rm = rmask(i, j)
      do m = 1, nd
        itr = cl_dic(m); ia = cl_alk(m)
        g = 0.0_rp
        if (beta > 0.0_sp) then
          calk = 0.0_sp
          if (ia > 0) calk = real(tp(ia, c, nz, nr), sp)
          g = -(kg / real(beta, rp) * (real(tp(itr, c, nz, nr), rp) - real(eta * calk, rp)) * rm)
        endif
        cdrl_gas(i, j, m) = g
        cdrl_psf(itr, c) = cdrl_flx(i, j, m) + g
      enddo
    enddo
    ! The CPU's flux loop also covers i = iend+1, which on a tile at the
    ! east edge is the boundary column: its gas flux reaches the output only.
    if (decomp%edge_e) call cdrl_gas_east(tp, ntr, nr, uw, vw, hw)
  end subroutine cdrl_sflx

  subroutine cdrl_gas_east(tp, ntr, nr, uw, vw, hw)
    integer(ip), intent(in) :: ntr, nr, hw
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), WORK_ATTR intent(in) :: uw(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: vw(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(sp), parameter :: a = 2116.8, b = -136.25, c3 = 4.7353, d4 = -0.092307, e5 = 0.0007555
    real(sp), parameter :: xkw_coef = 6.97e-07
    integer(ip) :: i, j, m, nz, nd, ny, col, itr, ia
    real(sp) :: sst, sc, beta, eta, calk
    real(rp) :: w2, kg, g, rm
    nz = decomp%nz; nd = cdrl_nd; ny = decomp%ny; i = decomp%nx + 1
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, 128, 0, work_stream>>>
#endif
    do j = 1, ny
      col = tpcol(i, j)
      sst = real(t(i, j, nz, nr, itemp), sp)
      sc = a + sst * (b + sst * (c3 + sst * (d4 + e5 * sst)))
      w2 = 0.0_rp
      if (hw == 1) w2 = uw(i, j)**2 + vw(i, j)**2
      kg = real(xkw_coef, rp) * w2 * real(sqrt(660.0_sp / sc), rp)
      if (sst <= -1.8_sp) kg = 0.0_rp          ! SEA_ICE_NOFLUX (CPU set_global_definitions.h)
      beta = real(cdrl_ddco2(i, j), sp)
      eta = real(cdrl_ddalk(i, j), sp)
      rm = rmask(i, j)
      do m = 1, nd
        itr = cl_dic(m); ia = cl_alk(m)
        g = 0.0_rp
        if (beta > 0.0_sp) then
          calk = 0.0_sp
          if (ia > 0) calk = real(tp(ia, col, nz, nr), sp)
          g = -(kg / real(beta, rp) * (real(tp(itr, col, nz, nr), rp) - real(eta * calk, rp)) * rm)
        endif
        cdrl_gas(i, j, m) = g
      enddo
    enddo
  end subroutine cdrl_gas_east

  ! ------------------------------------------------------------ output
  subroutine cdrl_out_alloc(no, oitr, avg, wtrc, wh, wiz, ns, wfg, i0, i1, j0, j1)
    ! Output tracers oitr(1:no); the averages that are written (avg: the
    ! file holds time averages), ns release sources, wfg: gas fluxes; the
    ! points the tile writes, i0:i1 x j0:j1.
    integer(ip), intent(in) :: no, ns, oitr(:), i0, i1, j0, j1
    logical, intent(in) :: avg, wtrc, wh, wiz, wfg
    integer :: is, ie, js, je, nz, i, j
    integer(ip) :: nw
    integer(ip), allocatable :: wc(:), wi(:), wj(:), wm(:,:)
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    co_n = no
    allocate(co_itr(max(no, 1)))
    co_itr = 1
    if (no > 0) co_itr(1:no) = oitr(1:no)
    ! the wet columns this tile writes (stored tp columns)
    allocate(wc((i1 - i0 + 1) * (j1 - j0 + 1)), wi(size(wc)), wj(size(wc)), wm(is:ie, js:je))
    wm = 0; nw = 0
    do j = j0, j1
      do i = i0, i1
        if (tpcol_h(i, j) > 0) then
          nw = nw + 1
          wc(nw) = tpcol_h(i, j); wi(nw) = i; wj(nw) = j; wm(i, j) = nw
        endif
      enddo
    enddo
    co_nw = nw
    allocate(co_wcol(max(nw, 1)), co_wi(max(nw, 1)), co_wj(max(nw, 1)), co_wmap(is:ie, js:je))
    co_wcol = 0; co_wi = is; co_wj = js
    if (nw > 0) then
      co_wcol(1:nw) = wc(1:nw); co_wi(1:nw) = wi(1:nw); co_wj(1:nw) = wj(1:nw)
    endif
    co_wmap = wm
    deallocate(wc, wi, wj, wm)
    allocate(co_tmp3(is:ie, js:je, nz), co_tmp2(is:ie, js:je))
    allocate(co_trc(max(no, 1), 0:merge(nw, 0_ip, avg .and. wtrc), nz))
    allocate(co_h(max(no, 1), 0:merge(nw, 0_ip, avg .and. wh), nz))
    allocate(co_iz(max(no, 1), 0:merge(nw, 0_ip, avg .and. wiz)))
    allocate(co_src(is:ie, js:je, nz, max(ns, 1)))
    allocate(co_fg(is:ie, js:je, merge(max(cdrl_nd, 1), 1, avg .and. wfg)))
    ! zeroed on the device: managed pages first touched there stay there
    ! (zeroing ~0.5 GB on the host cost seconds of start-up and migrations)
    call zero_flat(co_tmp3, size(co_tmp3, kind=8)); call zero_flat(co_tmp2, size(co_tmp2, kind=8))
    call zero_flat(co_trc, size(co_trc, kind=8)); call zero_flat(co_h, size(co_h, kind=8))
    call zero_flat(co_iz, size(co_iz, kind=8)); call zero_flat(co_src, size(co_src, kind=8))
    call zero_flat(co_fg, size(co_fg, kind=8))
  end subroutine cdrl_out_alloc

  subroutine zero_flat(a, n)
    ! a(1:n) = 0 on the device (any managed array, by sequence association).
    integer(8), intent(in) :: n
    real(rp), WORK_ATTR intent(inout) :: a(n)
    integer(8) :: i
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*, 256, 0, work_stream>>>
#endif
    do i = 1, n
      a(i) = 0.0_rp
    enddo
  end subroutine zero_flat

  subroutine cdrl_out_accum(tp, ntr, nn, om, cf, wtrc, wh, wiz, wfg, i0, i1, j0, j1)
    ! One step of the CPU's calc_average over the written points:
    ! avg = avg*(1-coef) + x*coef with the single-precision coef = 1./navg
    ! (om = 1-coef, cf = coef, both widened); x the tracer of slot nn, its
    ! thickness-weighted value (t*Hz)*coef, its vertical integral (sum
    ! over k = 1..nz of t*Hz), and the DIC tracers' gas flux. One thread per
    ! (output tracer, written wet column) over the levels; land stays 0, as
    ! the CPU's 0*(1-coef) + 0*coef.
    integer(ip), intent(in) :: ntr, nn, i0, i1, j0, j1
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), intent(in) :: om, cf
    logical, intent(in) :: wtrc, wh, wiz, wfg
    integer(ip) :: i, j, k, o, m, w, no, nm, nz, nd, col, n, itr, ft, fh, fi, ff
    real(rp) :: x, h, sm
    no = co_n; nz = decomp%nz; nd = cdrl_nd; n = nn; nm = co_n * co_nw
    ft = merge(1_ip, 0_ip, wtrc); fh = merge(1_ip, 0_ip, wh)
    fi = merge(1_ip, 0_ip, wiz); ff = merge(1_ip, 0_ip, wfg)
    if ((wtrc .or. wh .or. wiz) .and. nm > 0) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*, 128, 0, work_stream>>>
#endif
      do m = 1, nm
        o = mod(m - 1, no) + 1
        w = (m - 1) / no + 1
        col = co_wcol(w); i = co_wi(w); j = co_wj(w); itr = co_itr(o)
        sm = 0.0_rp
        do k = 1, nz
          x = real(tp(itr, col, k, n), rp)
          h = Hz(i, j, k)
          if (ft == 1) co_trc(o, w, k) = co_trc(o, w, k) * om + x * cf
          if (fh == 1) co_h(o, w, k) = co_h(o, w, k) * om + x * h * cf
          sm = sm + x * h
        enddo
        if (fi == 1) co_iz(o, w) = co_iz(o, w) * om + sm * cf
      enddo
    endif
    if (wfg) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do j = j0, j1
        do i = i0, i1
          do m = 1, nd
            co_fg(i, j, m) = co_fg(i, j, m) * om + cdrl_gas(i, j, m) * cf
          enddo
        enddo
      enddo
    endif
  end subroutine cdrl_out_accum

  subroutine cdrl_out_gather3(which, o, i0, i1, j0, j1)
    ! co_tmp3 = the average of output tracer o (which 1) or its
    ! thickness-weighted average (2) at the written points (land 0).
    integer, intent(in) :: which
    integer(ip), intent(in) :: o, i0, i1, j0, j1
    integer(ip) :: i, j, k, nz, oo, w
    nz = decomp%nz; oo = o
    if (which == 1) then
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = j0, j1
          do i = i0, i1
            co_tmp3(i, j, k) = co_trc(oo, co_wmap(i, j), k)
          enddo
        enddo
      enddo
    else
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = j0, j1
          do i = i0, i1
            co_tmp3(i, j, k) = co_h(oo, co_wmap(i, j), k)
          enddo
        enddo
      enddo
    endif
  end subroutine cdrl_out_gather3

  subroutine cdrl_out_gather2(o, i0, i1, j0, j1)
    ! co_tmp2 = the vertical-integral average of output tracer o.
    integer(ip), intent(in) :: o, i0, i1, j0, j1
    integer(ip) :: i, j, oo
    oo = o
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        co_tmp2(i, j) = co_iz(oo, co_wmap(i, j))
      enddo
    enddo
  end subroutine cdrl_out_gather2

  subroutine cdrl_out_inst_h(tp, ntr, nn, o, i0, i1, j0, j1)
    ! co_tmp3 = t*Hz of output tracer o (the CPU's multiply_by_thickness).
    integer(ip), intent(in) :: ntr, nn, o, i0, i1, j0, j1
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip) :: i, j, k, nz, n, oo
    nz = decomp%nz; n = nn; oo = o
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          co_tmp3(i, j, k) = real(tp(co_itr(oo), tpcol(i, j), k, n), rp) * Hz(i, j, k)
        enddo
      enddo
    enddo
  end subroutine cdrl_out_inst_h

  subroutine cdrl_out_inst_iz(tp, ntr, nn, o, i0, i1, j0, j1)
    ! co_tmp2 = sum over k of t*Hz of output tracer o.
    integer(ip), intent(in) :: ntr, nn, o, i0, i1, j0, j1
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip) :: i, j, k, nz, n, oo, col
    real(rp) :: sm
    nz = decomp%nz; n = nn; oo = o
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        col = tpcol(i, j)
        sm = 0.0_rp
        do k = 1, nz
          sm = sm + real(tp(co_itr(oo), col, k, n), rp) * Hz(i, j, k)
        enddo
        co_tmp2(i, j) = sm
      enddo
    enddo
  end subroutine cdrl_out_inst_iz

  subroutine cdrl_out_zero(which, idx)
    ! After a record is staged (stream-ordered after its gathers): zero one
    ! average. which: 1 tracer, 2 thickness-weighted, 3 integral, 4 source,
    ! 5 gas flux, 6 co_tmp3.
    integer, intent(in) :: which
    integer(ip), intent(in) :: idx
    integer(ip) :: i, j, k, w, is, ie, js, je, nz, ix, wh, nw
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; ix = idx
    wh = which; nw = co_nw
    select case (which)
    case (1, 2)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do w = 1, nw
          if (wh == 1) co_trc(ix, w, k) = 0.0_rp
          if (wh == 2) co_h(ix, w, k) = 0.0_rp
        enddo
      enddo
    case (3)
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*, *, 0, work_stream>>>
#endif
      do w = 1, nw
        co_iz(ix, w) = 0.0_rp
      enddo
    case (4, 6)
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = js, je
          do i = is, ie
            if (wh == 4) co_src(i, j, k, ix) = 0.0_rp
            if (wh == 6) co_tmp3(i, j, k) = 0.0_rp
          enddo
        enddo
      enddo
    case (5)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
      do j = js, je
        do i = is, ie
          co_fg(i, j, ix) = 0.0_rp
        enddo
      enddo
    end select
  end subroutine cdrl_out_zero

end module cdr_lite_mod

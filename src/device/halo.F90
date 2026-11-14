#include "mem_attr.inc"
#include "cuf_launch.inc"
module halo_mod
  ! Pack/unpack launch on the GPU work stream. NBR_SELF deliver is a
  ! device copy so {step2d, halo_fill} can live in a CUDA graph.
  ! EW then NS so corners ride along (ROMS do_corners).
  use kinds
  use decomp_mod, only: decomp
  use time_state, only: fast, nt, iic
  use comm_mod, only: comm_exchange2, comm_exchange3, comm_sendrecv_raw, comm_sendrecv_dev, comm_gpu_direct, &
                      comm_post_dev, comm_wait_dev, comm_rank
  use gpu_state, only: Hz, z_r, z_w, FlxU, FlxV, We, Wi, u, v, t, rho1, qp1, rho, bvf, rhoA, rhoS, ru, rv, &
                       umask, vmask, rmask, sustr, svstr, stflx, srflx, f, hbls, hbbl, Akv, Akt, ghat, &
                       rufrc, rvfrc, wrk3a, wrk3b, ubar, vbar, zeta, &
                       tpcol, ntpcol
#ifdef USE_CUDA
  use cudafor
  use iso_c_binding, only: c_f_pointer
#endif
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private

  ! Exchange buffers are plain device arrays (not managed): the pack and
  ! unpack kernels fill them, and the MPI path moves them with cudaMemcpy
  ! to and from pinned host memory, so no page migrates between the host
  ! and the device on an exchange.
#ifdef USE_CUDA
  real(rp), device, allocatable :: ew_slo(:,:,:), ew_shi(:,:,:)
  real(rp), device, allocatable :: ew_rlo(:,:,:), ew_rhi(:,:,:)
  real(rp), device, allocatable :: ns_slo(:,:,:), ns_shi(:,:,:)
  real(rp), device, allocatable :: ns_rlo(:,:,:), ns_rhi(:,:,:)
#else
  real(rp), allocatable :: ew_slo(:,:,:), ew_shi(:,:,:)
  real(rp), allocatable :: ew_rlo(:,:,:), ew_rhi(:,:,:)
  real(rp), allocatable :: ns_slo(:,:,:), ns_shi(:,:,:)
  real(rp), allocatable :: ns_rlo(:,:,:), ns_rhi(:,:,:)
#endif
  real(rp), WORK_ATTR allocatable :: work2d(:,:)
  real(rp), WORK_ATTR allocatable :: work3d(:,:,:)
  real(rp), WORK_ATTR allocatable :: work3dw(:,:,:)
  ! Group exchange state and buffers (device; grown on demand).
  integer, parameter :: HG_MAX = 40
  integer :: hg_n = 0, hg_nlev_tot = 0
  logical :: hg_batch = .true., hg_batch_checked = .false.    ! ROMS_GPU_HG_BATCH=n fields per launch (0: off)
  integer :: hg_stat, hg_per_launch = 4
  character(len=4) :: hg_val
  integer :: hg_id(HG_MAX), hg_slot(HG_MAX), hg_itr(HG_MAX), hg_lev0(HG_MAX), hg_nlev(HG_MAX)
#ifdef USE_CUDA
  ! device address and level range of each registered field, so one kernel
  ! can pack/unpack up to HG_PER_LAUNCH fields with the same level range
  integer, parameter :: HG_PER_LAUNCH = 4
  type(c_devptr) :: hg_ptr(HG_MAX)
  ! ROMS_GPU_HG_AUDIT=1 (debug): every group exchange snapshots its fields
  ! and counts the halo values the exchange changed; per position of the
  ! exchange within the step (aud_seq) and field, the number of calls and
  ! of calls that changed something is printed at the end of the run.
  integer, parameter :: AUD_MAX = 200
  logical :: hg_audit_on = .false., hg_audit_checked = .false.
  integer :: aud_seq = 0, aud_nseq = 0
  integer :: aud_nf(AUD_MAX) = 0, aud_ids(HG_MAX, AUD_MAX) = 0, aud_slots(HG_MAX, AUD_MAX) = 0
  integer :: aud_calls(AUD_MAX) = 0, aud_chg(HG_MAX, AUD_MAX) = 0
  integer(kind=8) :: aud_nval(HG_MAX, AUD_MAX) = 0
#ifdef USE_CUDA
  real(rp), device, allocatable :: aud_buf(:)
#else
  real(rp), allocatable :: aud_buf(:)
#endif
  integer(ip) :: hg_kb(HG_MAX), hg_ke(HG_MAX)
#endif
#ifdef USE_CUDA
  real(rp), device, allocatable :: gew_slo(:,:,:), gew_shi(:,:,:), gew_rlo(:,:,:), gew_rhi(:,:,:)
  real(rp), device, allocatable :: gns_slo(:,:,:), gns_shi(:,:,:), gns_rlo(:,:,:), gns_rhi(:,:,:)
#else
  real(rp), allocatable :: gew_slo(:,:,:), gew_shi(:,:,:), gew_rlo(:,:,:), gew_rhi(:,:,:)
  real(rp), allocatable :: gns_slo(:,:,:), gns_shi(:,:,:), gns_rlo(:,:,:), gns_rhi(:,:,:)
#endif
  integer :: hg_cap = 0
#ifdef USE_CUDA
  ! Pinned host staging for the MPI path: packed device buffers are copied
  ! here asynchronously on work_stream, exchanged, and copied back. No
  ! managed-memory page migration and no per-call allocation.
  real(rp), pinned, allocatable :: hs_lo(:), hs_hi(:), hr_lo(:), hr_hi(:)
  integer :: hbuf_n = 0
#endif
  ! All-tracer passive exchange buffers: (h*ntr, js:je, nz) and (h*ntr, is:ie, nz), tracer innermost
#ifdef USE_CUDA
  real(rp), device, allocatable :: tew_slo(:,:,:), tew_shi(:,:,:), tew_rlo(:,:,:), tew_rhi(:,:,:)
  real(rp), device, allocatable :: tns_slo(:,:,:), tns_shi(:,:,:), tns_rlo(:,:,:), tns_rhi(:,:,:)
#else
  real(rp), allocatable :: tew_slo(:,:,:), tew_shi(:,:,:), tew_rlo(:,:,:), tew_rhi(:,:,:)
  real(rp), allocatable :: tns_slo(:,:,:), tns_shi(:,:,:), tns_rlo(:,:,:), tns_rhi(:,:,:)
#endif
#ifdef USE_CUDA
  ! Event after the fast-loop exchange pack (halo_fill_zuv_n / zuv_mid).
  type(cudaEvent) :: zuv_ev
  logical :: zuv_ev_made = .false.
#endif
  ! Non-blocking all-tracer exchange in flight (halo_tp_start/mid/finish):
  ! 0 none, 1 east-west posted, 2 north-south posted.
  integer :: tp_pending = 0, tp_slot_p = 0, tp_ntr_p = 0
  logical :: tp_ew_posted = .false., tp_ns_posted = .false.
  integer :: tp_req_ew(4) = 0, tp_req_ns(4) = 0
#ifdef USE_CUDA
  real(pp), device, allocatable :: tp_chk(:,:,:)     ! ROMS_GPU_TP_CHECK scratch copy of one slot
#else
  real(pp), allocatable :: tp_chk(:,:,:)
#endif
  logical :: tp_devsync = .false.   ! ROMS_GPU_TP_DEVSYNC=1: cudaDeviceSynchronize after each wait (test)

  public :: halo_fill_2d
  public :: halo_fill_2d_n
  public :: halo_fill_uv_n
  public :: halo_fill_zuv_n
  public :: hg_begin, hg_add, hg_end
  public :: hg_audit_step, hg_audit_report
  ! Field ids for the group exchange (hg_*): several fields in one
  ! east-west and one north-south round. All fields use the rho-cell
  ! exchange bounds (see interior_bounds), so staggers may be mixed.
  integer, parameter, public :: F_HZ = 1, F_ZR = 2, F_ZW = 3, F_FLXU = 4, F_FLXV = 5, F_WE = 6, F_WI = 7, &
       F_U = 8, F_V = 9, F_T = 10, F_RHO1 = 11, F_QP1 = 12, F_RHO = 13, F_BVF = 14, F_RHOA = 15, F_RHOS = 16, &
       F_RU = 17, F_RV = 18, F_UMASK = 19, F_VMASK = 20, F_RMASK = 21, F_SUSTR = 22, F_SVSTR = 23, &
       F_STFLX = 24, F_SRFLX = 25, F_F = 26, F_HBLS = 27, F_HBBL = 28, F_AKV = 29, F_AKT = 30, F_GHAT = 31, &
       F_RUFRC = 32, F_RVFRC = 33, F_WRK3A = 34, F_WRK3B = 35, F_UBAR = 36, F_VBAR = 37, F_ZETA = 38
  public :: halo_fill_2d_s
  public :: halo_fill_2d_st
  public :: halo_fill_2d_sw
  public :: halo_fill_3d
  public :: halo_fill_3d_s
  public :: halo_fill_3d_t
  public :: halo_fill_3d_tp, halo_fill_3d_tp_k, halo_fill_3d_k
  public :: halo_tp_start, halo_tp_mid, halo_tp_finish, tp_overlap_on
  public :: halo_fill_3d_w
  public :: halo_fill_3d_w_s
  public :: halo_ensure
  public :: halo_free

contains

  subroutine ensure_bufs()
    integer(ip) :: h, is, ie, js, je, nz
    if (allocated(ew_slo)) then
      if (lbound(ew_slo, 2) == decomp%js .and. ubound(ew_slo, 2) == decomp%je .and. &
          ubound(ew_slo, 3) == decomp%nz) return
      call halo_free()
    endif
    h = decomp%halo
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nz = decomp%nz
    allocate(ew_slo(h, js:je, 0:nz), ew_shi(h, js:je, 0:nz))
    allocate(ew_rlo(h, js:je, 0:nz), ew_rhi(h, js:je, 0:nz))
    allocate(ns_slo(is:ie, h, 0:nz), ns_shi(is:ie, h, 0:nz))
    allocate(ns_rlo(is:ie, h, 0:nz), ns_rhi(is:ie, h, 0:nz))
    allocate(work2d(is:ie, js:je))
    allocate(work3d(is:ie, js:je, 1:nz))
    allocate(work3dw(is:ie, js:je, 0:nz))
    ew_slo = 0.0_rp; ew_shi = 0.0_rp; ew_rlo = 0.0_rp; ew_rhi = 0.0_rp
    ns_slo = 0.0_rp; ns_shi = 0.0_rp; ns_rlo = 0.0_rp; ns_rhi = 0.0_rp
    work2d = 0.0_rp; work3d = 0.0_rp; work3dw = 0.0_rp
  end subroutine ensure_bufs

  subroutine halo_ensure()
    ! Nothing to exchange on a tile without neighbours: no buffers.
    if (no_neighbors()) return
    call ensure_bufs()
  end subroutine halo_ensure

  subroutine halo_free()
    if (allocated(ew_slo)) deallocate(ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (allocated(ns_slo)) deallocate(ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (allocated(work2d)) deallocate(work2d)
    if (allocated(work3d)) deallocate(work3d)
    if (allocated(work3dw)) deallocate(work3dw)
    if (allocated(tew_slo)) deallocate(tew_slo, tew_shi, tew_rlo, tew_rhi)
    if (allocated(tns_slo)) deallocate(tns_slo, tns_shi, tns_rlo, tns_rhi)
  end subroutine halo_free

  subroutine halo_fill_2d(a, stagger)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_2d(a, i1, i2, j1, j2)
    call exchange_ns_2d(a, i1, i2, j1, j2)
  end subroutine halo_fill_2d

  subroutine halo_fill_2d_n(a, stagger)
    ! Halo a(:,:,fast%knew). The slot is read from managed memory so a
    ! captured graph can replay after the host advances knew.
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_2d_n(a, i1, i2, j1, j2)
    call exchange_ns_2d_n(a, i1, i2, j1, j2)
  end subroutine halo_fill_2d_n

  subroutine halo_fill_uv_n(u2, v2)
    ! ubar and vbar at slot fast%knew in ONE exchange per direction: both are
    ! packed as levels 0 and 1 of the 3-D buffers (the rho-cell exchange
    ! bounds are the same for every stagger). Pure copies.
    real(rp), WORK_ATTR intent(inout) :: u2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    real(rp), WORK_ATTR intent(inout) :: v2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip) :: i1, i2, j1, j2, h, i, j, m, n, ilo, ihi
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    h = decomp%halo
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          ew_slo(m, j, 0) = u2(i1 + m - 1, j, n)
          ew_shi(m, j, 0) = u2(i2 - h + m, j, n)
          ew_slo(m, j, 1) = v2(i1 + m - 1, j, n)
          ew_shi(m, j, 1) = v2(i2 - h + m, j, n)
        enddo
      enddo
      call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo(:, :, 0:1), ew_shi(:, :, 0:1), &
                    ew_rlo(:, :, 0:1), ew_rhi(:, :, 0:1))
      if (decomp%nbr_w /= NBR_NONE) then
        CUF_DO2
        do j = j1, j2
          do m = 1, h
            n = fast%knew
            u2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
            v2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 1)
          enddo
        enddo
      endif
      if (decomp%nbr_e /= NBR_NONE) then
        CUF_DO2
        do j = j1, j2
          do m = 1, h
            n = fast%knew
            u2(i2 + m, j, n) = ew_rhi(m, j, 0)
            v2(i2 + m, j, n) = ew_rhi(m, j, 1)
          enddo
        enddo
      endif
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          ns_slo(i, m, 0) = u2(i, j1 + m - 1, n)
          ns_shi(i, m, 0) = u2(i, j2 - h + m, n)
          ns_slo(i, m, 1) = v2(i, j1 + m - 1, n)
          ns_shi(i, m, 1) = v2(i, j2 - h + m, n)
        enddo
      enddo
      call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo(:, :, 0:1), ns_shi(:, :, 0:1), &
                    ns_rlo(:, :, 0:1), ns_rhi(:, :, 0:1))
      if (decomp%nbr_s /= NBR_NONE) then
        CUF_DO2
        do m = 1, h
          do i = ilo, ihi
            n = fast%knew
            u2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
            v2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 1)
          enddo
        enddo
      endif
      if (decomp%nbr_n /= NBR_NONE) then
        CUF_DO2
        do m = 1, h
          do i = ilo, ihi
            n = fast%knew
            u2(i, j2 + m, n) = ns_rhi(i, m, 0)
            v2(i, j2 + m, n) = ns_rhi(i, m, 1)
          enddo
        enddo
      endif
    endif
  end subroutine halo_fill_uv_n

  subroutine halo_fill_zuv_n(z2, u2, v2, mid)
    ! zeta, ubar and vbar at slot fast%knew in ONE exchange per direction
    ! (levels 2, 0 and 1 of the 3-D buffers; the rho-cell exchange bounds
    ! are the same for every stagger, as halo_fill_2d_n uses for zeta).
    ! mid (optional): device work queued right after the first pack; the
    ! MPI transfer waits for the pack only (event), so mid runs during it.
    ! It must not touch the exchanged halo cells or the buffers.
    real(rp), WORK_ATTR intent(inout) :: z2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    real(rp), WORK_ATTR intent(inout) :: u2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    real(rp), WORK_ATTR intent(inout) :: v2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    interface
      subroutine mid_i()
      end subroutine mid_i
    end interface
    procedure(mid_i), optional :: mid
    integer(ip) :: i1, i2, j1, j2, h, i, j, m, n, ilo, ihi
    logical :: lo, hi, mid_done
    if (no_neighbors()) then
      if (present(mid)) call mid()
      return
    endif
    call ensure_bufs()
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    h = decomp%halo
    mid_done = .not. present(mid)
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          ew_slo(m, j, 0) = u2(i1 + m - 1, j, n)
          ew_shi(m, j, 0) = u2(i2 - h + m, j, n)
          ew_slo(m, j, 1) = v2(i1 + m - 1, j, n)
          ew_shi(m, j, 1) = v2(i2 - h + m, j, n)
          ew_slo(m, j, 2) = z2(i1 + m - 1, j, n)
          ew_shi(m, j, 2) = z2(i2 - h + m, j, n)
        enddo
      enddo
      call zuv_mid(mid, mid_done)
      call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo(:, :, 0:2), ew_shi(:, :, 0:2), &
                    ew_rlo(:, :, 0:2), ew_rhi(:, :, 0:2), ev_wait=present(mid))
      lo = (decomp%nbr_w /= NBR_NONE); hi = (decomp%nbr_e /= NBR_NONE)
      ! west and east halo columns in one launch (disjoint cells)
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          if (lo) then
            u2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
            v2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 1)
            z2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 2)
          endif
          if (hi) then
            u2(i2 + m, j, n) = ew_rhi(m, j, 0)
            v2(i2 + m, j, n) = ew_rhi(m, j, 1)
            z2(i2 + m, j, n) = ew_rhi(m, j, 2)
          endif
        enddo
      enddo
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          ns_slo(i, m, 0) = u2(i, j1 + m - 1, n)
          ns_shi(i, m, 0) = u2(i, j2 - h + m, n)
          ns_slo(i, m, 1) = v2(i, j1 + m - 1, n)
          ns_shi(i, m, 1) = v2(i, j2 - h + m, n)
          ns_slo(i, m, 2) = z2(i, j1 + m - 1, n)
          ns_shi(i, m, 2) = z2(i, j2 - h + m, n)
        enddo
      enddo
      call zuv_mid(mid, mid_done)
      call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo(:, :, 0:2), ns_shi(:, :, 0:2), &
                    ns_rlo(:, :, 0:2), ns_rhi(:, :, 0:2), ev_wait=present(mid))
      lo = (decomp%nbr_s /= NBR_NONE); hi = (decomp%nbr_n /= NBR_NONE)
      ! south and north halo rows in one launch (disjoint cells)
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          if (lo) then
            u2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
            v2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 1)
            z2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 2)
          endif
          if (hi) then
            u2(i, j2 + m, n) = ns_rhi(i, m, 0)
            v2(i, j2 + m, n) = ns_rhi(i, m, 1)
            z2(i, j2 + m, n) = ns_rhi(i, m, 2)
          endif
        enddo
      enddo
    endif
    if (.not. mid_done) call mid()
  end subroutine halo_fill_zuv_n

  subroutine zuv_mid(mid, mid_done)
    ! After a pack: record the pack event (the transfer waits for it), then
    ! queue mid once.
    interface
      subroutine mid_i()
      end subroutine mid_i
    end interface
    procedure(mid_i), optional :: mid
    logical, intent(inout) :: mid_done
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (.not. present(mid)) return
#ifdef USE_CUDA
    if (.not. zuv_ev_made) then
      ierr = cudaEventCreateWithFlags(zuv_ev, cudaEventDisableTiming)
      if (ierr /= 0) error stop 'halo: zuv event create'
      zuv_ev_made = .true.
    endif
    ierr = cudaEventRecord(zuv_ev, work_stream)
    if (ierr /= 0) error stop 'halo: zuv event record'
#endif
    if (.not. mid_done) then
      call mid()
      mid_done = .true.
    endif
  end subroutine zuv_mid

  ! ------------------------------------------------------------ group exchange
  subroutine hg_begin()
    hg_n = 0; hg_nlev_tot = 0
  end subroutine hg_begin

  subroutine hg_add(id, slot, itr)
    ! Register a field (optionally a time slot and a tracer/level index).
    integer, intent(in) :: id
    integer(ip), intent(in), optional :: slot, itr
    integer :: nl
    if (no_neighbors()) return
    hg_n = hg_n + 1
    if (hg_n > HG_MAX) error stop 'hg_add: too many fields in one group'
    hg_id(hg_n) = id
    hg_slot(hg_n) = 1; hg_itr(hg_n) = 1
    if (present(slot)) hg_slot(hg_n) = int(slot)
    if (present(itr)) hg_itr(hg_n) = int(itr)
    select case (id)
    case (F_ZW, F_WE, F_WI, F_BVF, F_AKV, F_AKT, F_GHAT)
      nl = int(decomp%nz) + 1
    case (F_HZ, F_ZR, F_FLXU, F_FLXV, F_U, F_V, F_T, F_RHO1, F_QP1, F_RHO, F_RU, F_RV, F_WRK3A, F_WRK3B)
      nl = int(decomp%nz)
    case default
      nl = 1
    end select
    hg_lev0(hg_n) = hg_nlev_tot
    hg_nlev(hg_n) = nl
    hg_nlev_tot = hg_nlev_tot + nl
  end subroutine hg_add

  subroutine hg_end()
    ! One east-west round for every registered field, then one north-south
    ! round (including the east-west halo columns), as the single fills do.
    integer(ip) :: i1, i2, j1, j2, h, is, ie, js, je, k
    integer :: L
    if (no_neighbors() .or. hg_n == 0) then
      hg_n = 0; hg_nlev_tot = 0
      return
    endif
    call ensure_bufs()
    h = decomp%halo; is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    L = hg_nlev_tot
    if (hg_cap < L) then
      if (allocated(gew_slo)) deallocate(gew_slo, gew_shi, gew_rlo, gew_rhi, gns_slo, gns_shi, gns_rlo, gns_rhi)
      allocate(gew_slo(h, js:je, L), gew_shi(h, js:je, L), gew_rlo(h, js:je, L), gew_rhi(h, js:je, L))
      allocate(gns_slo(is:ie, h, L), gns_shi(is:ie, h, L), gns_rlo(is:ie, h, L), gns_rhi(is:ie, h, L))
      gew_slo = 0.0_rp; gew_shi = 0.0_rp; gew_rlo = 0.0_rp; gew_rhi = 0.0_rp
      gns_slo = 0.0_rp; gns_shi = 0.0_rp; gns_rlo = 0.0_rp; gns_rhi = 0.0_rp
      hg_cap = L
    endif
#ifdef USE_CUDA
    if (.not. hg_batch_checked) then
      call get_environment_variable('ROMS_GPU_HG_BATCH', hg_val, status=hg_stat)
      ! default: 4 fields per launch; ROMS_GPU_HG_BATCH=0 restores one launch per field
      hg_batch = .true.
      if (hg_stat == 0 .and. len_trim(hg_val) > 0) then
        read(hg_val, *, iostat=hg_stat) hg_per_launch
        if (hg_stat /= 0) hg_per_launch = HG_PER_LAUNCH
        hg_batch = (hg_per_launch >= 1)
      endif
      hg_per_launch = max(1, min(hg_per_launch, HG_PER_LAUNCH))
      hg_batch_checked = .true.
    endif
    if (.not. hg_batch) then
      call hg_end_single(i1, i2, j1, j2, L)
      hg_n = 0; hg_nlev_tot = 0
      return
    endif
    do k = 1, hg_n
      call hg_resolve(int(k))
    enddo
    if (.not. hg_audit_checked) then
      hg_audit_checked = .true.
      call get_environment_variable('ROMS_GPU_HG_AUDIT', hg_val, status=hg_stat)
      hg_audit_on = (hg_stat == 0 .and. len_trim(hg_val) > 0 .and. hg_val(1:1) == '1')
      if (hg_audit_on) write(*,'(a)') 'halo: group-exchange audit on (ROMS_GPU_HG_AUDIT)'
    endif
    if (hg_audit_on) call hg_audit_before()
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      call hg_run_phase(1, i1, i2, j1, j2)
      call deliver3(decomp%nbr_w, decomp%nbr_e, gew_slo(:, :, 1:L), gew_shi(:, :, 1:L), &
                    gew_rlo(:, :, 1:L), gew_rhi(:, :, 1:L))
      call hg_run_phase(2, i1, i2, j1, j2)
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      call hg_run_phase(3, i1, i2, j1, j2)
      call deliver3(decomp%nbr_s, decomp%nbr_n, gns_slo(:, :, 1:L), gns_shi(:, :, 1:L), &
                    gns_rlo(:, :, 1:L), gns_rhi(:, :, 1:L))
      call hg_run_phase(4, i1, i2, j1, j2)
    endif
    if (hg_audit_on) call hg_audit_after(i1, i2, j1, j2)
#else
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      do k = 1, hg_n
        call hg_dispatch(k, 1, i1, i2, j1, j2)
      enddo
      call deliver3(decomp%nbr_w, decomp%nbr_e, gew_slo(:, :, 1:L), gew_shi(:, :, 1:L), &
                    gew_rlo(:, :, 1:L), gew_rhi(:, :, 1:L))
      do k = 1, hg_n
        call hg_dispatch(k, 2, i1, i2, j1, j2)
      enddo
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      do k = 1, hg_n
        call hg_dispatch(k, 3, i1, i2, j1, j2)
      enddo
      call deliver3(decomp%nbr_s, decomp%nbr_n, gns_slo(:, :, 1:L), gns_shi(:, :, 1:L), &
                    gns_rlo(:, :, 1:L), gns_rhi(:, :, 1:L))
      do k = 1, hg_n
        call hg_dispatch(k, 4, i1, i2, j1, j2)
      enddo
    endif
#endif
    hg_n = 0; hg_nlev_tot = 0
  end subroutine hg_end

#ifdef USE_CUDA
  subroutine hg_audit_step()
    ! Start of a baroclinic step: exchange positions count from 1 again.
    aud_seq = 0
  end subroutine hg_audit_step

  subroutine hg_audit_before()
#ifdef USE_CUDA
    integer :: k
    integer(ip) :: ni, nj, nel, off, tot, n
    real(rp), device, pointer, contiguous :: p(:)
    ni = decomp%ie - decomp%is + 1; nj = decomp%je - decomp%js + 1
    tot = 0
    do k = 1, hg_n
      tot = tot + ni * nj * (hg_ke(k) - hg_kb(k) + 1)
    enddo
    if (allocated(aud_buf)) then
      if (size(aud_buf) < tot) deallocate(aud_buf)
    endif
    if (.not. allocated(aud_buf)) allocate(aud_buf(tot))
    off = 0
    do k = 1, hg_n
      nel = ni * nj * (hg_ke(k) - hg_kb(k) + 1)
      call c_f_pointer(hg_ptr(k), p, (/nel/))
      !$cuf kernel do(1) <<<*,*,0,work_stream>>>
      do n = 1, nel
        aud_buf(off + n) = p(n)
      enddo
      off = off + nel
    enddo
#endif
  end subroutine hg_audit_before

  subroutine hg_audit_after(i1, i2, j1, j2)
#ifdef USE_CUDA
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer :: k, q
    integer(ip) :: ni, nj, nel, off, n, ii, jj, nchg, nhal, is, js
    real(rp), device, pointer, contiguous :: p(:)
    ni = decomp%ie - decomp%is + 1; nj = decomp%je - decomp%js + 1
    is = decomp%is; js = decomp%js
    aud_seq = aud_seq + 1
    q = min(aud_seq, AUD_MAX)
    aud_nseq = max(aud_nseq, q)
    if (aud_calls(q) == 0) then
      aud_nf(q) = hg_n
      aud_ids(1:hg_n, q) = hg_id(1:hg_n)
      aud_slots(1:hg_n, q) = hg_slot(1:hg_n)
    endif
    aud_calls(q) = aud_calls(q) + 1
    off = 0
    do k = 1, hg_n
      nel = ni * nj * (hg_ke(k) - hg_kb(k) + 1)
      call c_f_pointer(hg_ptr(k), p, (/nel/))
      nchg = 0; nhal = 0
      !$cuf kernel do(1) <<<*,*,0,work_stream>>>
      do n = 1, nel
        ii = mod(n - 1, ni) + is
        jj = mod((n - 1) / ni, nj) + js
        if (ii < i1 .or. ii > i2 .or. jj < j1 .or. jj > j2) then
          nhal = nhal + 1
          if (p(n) /= aud_buf(off + n)) nchg = nchg + 1
        endif
      enddo
      if (nchg > 0) aud_chg(k, q) = aud_chg(k, q) + 1
      aud_nval(k, q) = aud_nval(k, q) + nchg
      off = off + nel
    enddo
#endif
  end subroutine hg_audit_after

  subroutine hg_audit_report()
    integer :: q, k
    character(len=600) :: line
    if (.not. hg_audit_on) return
    do q = 1, aud_nseq
      line = ''
      do k = 1, aud_nf(q)
        write(line(len_trim(line) + 2:), '(i0,a,i0,a,i0,a)') aud_ids(k, q), '/', aud_slots(k, q), ':', aud_chg(k, q), ''
      enddo
      write(*,'(a,i0,a,i3,a,i5,a,a)') 'hg audit rank ', comm_rank(), ' seq ', q, ' calls ', aud_calls(q), &
           '  id/slot:changed', trim(line)
    enddo
    flush(6)
  end subroutine hg_audit_report

  subroutine hg_resolve(k)
    ! Device address + level range of field k (same table as hg_dispatch).
    integer, intent(in) :: k
    integer(ip) :: is, js, nz, sl, it
    is = decomp%is; js = decomp%js; nz = decomp%nz
    sl = hg_slot(k); it = hg_itr(k)
    select case (hg_id(k))
    case (F_HZ); hg_ptr(k) = c_devloc(Hz(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_ZR); hg_ptr(k) = c_devloc(z_r(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_ZW); hg_ptr(k) = c_devloc(z_w(is, js, 0)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_FLXU); hg_ptr(k) = c_devloc(FlxU(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_FLXV); hg_ptr(k) = c_devloc(FlxV(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_WE); hg_ptr(k) = c_devloc(We(is, js, 0)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_WI); hg_ptr(k) = c_devloc(Wi(is, js, 0)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_U); hg_ptr(k) = c_devloc(u(is, js, 1, sl)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_V); hg_ptr(k) = c_devloc(v(is, js, 1, sl)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_T); hg_ptr(k) = c_devloc(t(is, js, 1, sl, it)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_RHO1); hg_ptr(k) = c_devloc(rho1(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_QP1); hg_ptr(k) = c_devloc(qp1(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_RHO); hg_ptr(k) = c_devloc(rho(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_BVF); hg_ptr(k) = c_devloc(bvf(is, js, 0)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_RHOA); hg_ptr(k) = c_devloc(rhoA(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_RHOS); hg_ptr(k) = c_devloc(rhoS(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_RU); hg_ptr(k) = c_devloc(ru(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_RV); hg_ptr(k) = c_devloc(rv(is, js, 1)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_UMASK); hg_ptr(k) = c_devloc(umask(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_VMASK); hg_ptr(k) = c_devloc(vmask(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_RMASK); hg_ptr(k) = c_devloc(rmask(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_SUSTR); hg_ptr(k) = c_devloc(sustr(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_SVSTR); hg_ptr(k) = c_devloc(svstr(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_STFLX); hg_ptr(k) = c_devloc(stflx(is, js, it)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_SRFLX); hg_ptr(k) = c_devloc(srflx(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_F); hg_ptr(k) = c_devloc(f(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_HBLS); hg_ptr(k) = c_devloc(hbls(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_HBBL); hg_ptr(k) = c_devloc(hbbl(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_AKV); hg_ptr(k) = c_devloc(Akv(is, js, 0)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_AKT); hg_ptr(k) = c_devloc(Akt(is, js, 0, it)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_GHAT); hg_ptr(k) = c_devloc(ghat(is, js, 0)); hg_kb(k) = 0_ip; hg_ke(k) = nz
    case (F_RUFRC); hg_ptr(k) = c_devloc(rufrc(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_RVFRC); hg_ptr(k) = c_devloc(rvfrc(is, js)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_WRK3A); hg_ptr(k) = c_devloc(wrk3a(is, js, 0)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_WRK3B); hg_ptr(k) = c_devloc(wrk3b(is, js, 0)); hg_kb(k) = 1_ip; hg_ke(k) = nz
    case (F_UBAR); hg_ptr(k) = c_devloc(ubar(is, js, sl)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_VBAR); hg_ptr(k) = c_devloc(vbar(is, js, sl)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case (F_ZETA); hg_ptr(k) = c_devloc(zeta(is, js, sl)); hg_kb(k) = 1_ip; hg_ke(k) = 1_ip
    case default
      error stop 'hg_resolve: unknown field id'
    end select
  end subroutine hg_resolve

  subroutine hg_run_phase(phase, i1, i2, j1, j2)
    ! Fields with the same level range, up to HG_PER_LAUNCH at a time, in
    ! one kernel launch each.
    integer, intent(in) :: phase
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer :: k, n, g
    integer(ip) :: ni, nj, nel
    ! contiguous: pass the base address, never a device temporary (an
    ! intent(inout) temporary would lose the unpacked halos)
    real(rp), device, pointer, contiguous :: p1(:), p2(:), p3(:), p4(:)
    ni = decomp%ie - decomp%is + 1; nj = decomp%je - decomp%js + 1
    k = 1
    do while (k <= hg_n)
      n = 1
      do while (k + n <= hg_n .and. n < hg_per_launch)
        if (hg_kb(k + n) /= hg_kb(k) .or. hg_ke(k + n) /= hg_ke(k)) exit
        n = n + 1
      enddo
      nel = ni * nj * (hg_ke(k) - hg_kb(k) + 1)
      call c_f_pointer(hg_ptr(k), p1, (/nel/))
      p2 => p1; p3 => p1; p4 => p1
      if (n >= 2) call c_f_pointer(hg_ptr(k + 1), p2, (/nel/))
      if (n >= 3) call c_f_pointer(hg_ptr(k + 2), p3, (/nel/))
      if (n >= 4) call c_f_pointer(hg_ptr(k + 3), p4, (/nel/))
      g = min(n, HG_PER_LAUNCH)
      call hg_slab4(g, p1, p2, p3, p4, hg_kb(k), hg_ke(k), hg_lev0(k), hg_lev0(min(k + 1, hg_n)), &
                    hg_lev0(min(k + 2, hg_n)), hg_lev0(min(k + 3, hg_n)), phase, i1, i2, j1, j2)
      k = k + n
    enddo
  end subroutine hg_run_phase

  subroutine hg_slab4(n, a1, a2, a3, a4, kb, ke, l1, l2, l3, l4, phase, i1, i2, j1, j2)
    ! Pack / unpack up to four slabs with the same level range kb:ke at the
    ! group level offsets l1..l4 (pure copies, as hg_slab).
    integer, intent(in) :: n, phase, l1, l2, l3, l4
    integer(ip), intent(in) :: kb, ke, i1, i2, j1, j2
    real(rp), device, intent(inout) :: a1(decomp%is:decomp%ie, decomp%js:decomp%je, kb:ke)
    real(rp), device, intent(inout) :: a2(decomp%is:decomp%ie, decomp%js:decomp%je, kb:ke)
    real(rp), device, intent(inout) :: a3(decomp%is:decomp%ie, decomp%js:decomp%je, kb:ke)
    real(rp), device, intent(inout) :: a4(decomp%is:decomp%ie, decomp%js:decomp%je, kb:ke)
    integer(ip) :: i, j, m, kk, h, ilo, ihi, c
    integer :: nn
    h = decomp%halo
    ilo = i1 - h; ihi = i2 + h
    nn = n
    select case (phase)
    case (1)
      CUF_DO3
      do kk = kb, ke
        do j = j1, j2
          do m = 1, h
            c = kk - kb + 1
            gew_slo(m, j, l1 + c) = a1(i1 + m - 1, j, kk); gew_shi(m, j, l1 + c) = a1(i2 - h + m, j, kk)
            if (nn >= 2) then
              gew_slo(m, j, l2 + c) = a2(i1 + m - 1, j, kk); gew_shi(m, j, l2 + c) = a2(i2 - h + m, j, kk)
            endif
            if (nn >= 3) then
              gew_slo(m, j, l3 + c) = a3(i1 + m - 1, j, kk); gew_shi(m, j, l3 + c) = a3(i2 - h + m, j, kk)
            endif
            if (nn >= 4) then
              gew_slo(m, j, l4 + c) = a4(i1 + m - 1, j, kk); gew_shi(m, j, l4 + c) = a4(i2 - h + m, j, kk)
            endif
          enddo
        enddo
      enddo
    case (2)
      if (decomp%nbr_w /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do j = j1, j2
            do m = 1, h
              c = kk - kb + 1
              a1(i1 - h + m - 1, j, kk) = gew_rlo(m, j, l1 + c)
              if (nn >= 2) a2(i1 - h + m - 1, j, kk) = gew_rlo(m, j, l2 + c)
              if (nn >= 3) a3(i1 - h + m - 1, j, kk) = gew_rlo(m, j, l3 + c)
              if (nn >= 4) a4(i1 - h + m - 1, j, kk) = gew_rlo(m, j, l4 + c)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_e /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do j = j1, j2
            do m = 1, h
              c = kk - kb + 1
              a1(i2 + m, j, kk) = gew_rhi(m, j, l1 + c)
              if (nn >= 2) a2(i2 + m, j, kk) = gew_rhi(m, j, l2 + c)
              if (nn >= 3) a3(i2 + m, j, kk) = gew_rhi(m, j, l3 + c)
              if (nn >= 4) a4(i2 + m, j, kk) = gew_rhi(m, j, l4 + c)
            enddo
          enddo
        enddo
      endif
    case (3)
      CUF_DO3
      do kk = kb, ke
        do m = 1, h
          do i = ilo, ihi
            c = kk - kb + 1
            gns_slo(i, m, l1 + c) = a1(i, j1 + m - 1, kk); gns_shi(i, m, l1 + c) = a1(i, j2 - h + m, kk)
            if (nn >= 2) then
              gns_slo(i, m, l2 + c) = a2(i, j1 + m - 1, kk); gns_shi(i, m, l2 + c) = a2(i, j2 - h + m, kk)
            endif
            if (nn >= 3) then
              gns_slo(i, m, l3 + c) = a3(i, j1 + m - 1, kk); gns_shi(i, m, l3 + c) = a3(i, j2 - h + m, kk)
            endif
            if (nn >= 4) then
              gns_slo(i, m, l4 + c) = a4(i, j1 + m - 1, kk); gns_shi(i, m, l4 + c) = a4(i, j2 - h + m, kk)
            endif
          enddo
        enddo
      enddo
    case (4)
      if (decomp%nbr_s /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do m = 1, h
            do i = ilo, ihi
              c = kk - kb + 1
              a1(i, j1 - h + m - 1, kk) = gns_rlo(i, m, l1 + c)
              if (nn >= 2) a2(i, j1 - h + m - 1, kk) = gns_rlo(i, m, l2 + c)
              if (nn >= 3) a3(i, j1 - h + m - 1, kk) = gns_rlo(i, m, l3 + c)
              if (nn >= 4) a4(i, j1 - h + m - 1, kk) = gns_rlo(i, m, l4 + c)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_n /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do m = 1, h
            do i = ilo, ihi
              c = kk - kb + 1
              a1(i, j2 + m, kk) = gns_rhi(i, m, l1 + c)
              if (nn >= 2) a2(i, j2 + m, kk) = gns_rhi(i, m, l2 + c)
              if (nn >= 3) a3(i, j2 + m, kk) = gns_rhi(i, m, l3 + c)
              if (nn >= 4) a4(i, j2 + m, kk) = gns_rhi(i, m, l4 + c)
            enddo
          enddo
        enddo
      endif
    end select
  end subroutine hg_slab4
#endif

  subroutine hg_end_single(i1, i2, j1, j2, L)
    ! One launch per field and phase (the validated path).
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer, intent(in) :: L
    integer(ip) :: k
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      do k = 1, hg_n
        call hg_dispatch(k, 1, i1, i2, j1, j2)
      enddo
      call deliver3(decomp%nbr_w, decomp%nbr_e, gew_slo(:, :, 1:L), gew_shi(:, :, 1:L), &
                    gew_rlo(:, :, 1:L), gew_rhi(:, :, 1:L))
      do k = 1, hg_n
        call hg_dispatch(k, 2, i1, i2, j1, j2)
      enddo
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      do k = 1, hg_n
        call hg_dispatch(k, 3, i1, i2, j1, j2)
      enddo
      call deliver3(decomp%nbr_s, decomp%nbr_n, gns_slo(:, :, 1:L), gns_shi(:, :, 1:L), &
                    gns_rlo(:, :, 1:L), gns_rhi(:, :, 1:L))
      do k = 1, hg_n
        call hg_dispatch(k, 4, i1, i2, j1, j2)
      enddo
    endif
  end subroutine hg_end_single

  subroutine hg_dispatch(k, phase, i1, i2, j1, j2)
    ! Resolve the field id to its slab (first element + level range) and run
    ! phase 1: pack EW, 2: unpack EW, 3: pack NS, 4: unpack NS.
    integer, intent(in) :: k, phase
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: is, js, nz, sl, it
    is = decomp%is; js = decomp%js; nz = decomp%nz
    sl = hg_slot(k); it = hg_itr(k)
    select case (hg_id(k))
    case (F_HZ);    call hg_slab(Hz(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_ZR);    call hg_slab(z_r(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_ZW);    call hg_slab(z_w(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_FLXU);  call hg_slab(FlxU(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_FLXV);  call hg_slab(FlxV(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_WE);    call hg_slab(We(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_WI);    call hg_slab(Wi(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_U);     call hg_slab(u(is, js, 1, sl), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_V);     call hg_slab(v(is, js, 1, sl), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_T);     call hg_slab(t(is, js, 1, sl, it), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RHO1);  call hg_slab(rho1(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_QP1);   call hg_slab(qp1(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RHO);   call hg_slab(rho(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_BVF);   call hg_slab(bvf(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RHOA);  call hg_slab(rhoA(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RHOS);  call hg_slab(rhoS(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RU);    call hg_slab(ru(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RV);    call hg_slab(rv(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_UMASK); call hg_slab(umask(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_VMASK); call hg_slab(vmask(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RMASK); call hg_slab(rmask(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_SUSTR); call hg_slab(sustr(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_SVSTR); call hg_slab(svstr(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_STFLX); call hg_slab(stflx(is, js, it), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_SRFLX); call hg_slab(srflx(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_F);     call hg_slab(f(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_HBLS);  call hg_slab(hbls(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_HBBL);  call hg_slab(hbbl(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_AKV);   call hg_slab(Akv(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_AKT);   call hg_slab(Akt(is, js, 0, it), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_GHAT);  call hg_slab(ghat(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RUFRC); call hg_slab(rufrc(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RVFRC); call hg_slab(rvfrc(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    ! wrk3a/wrk3b are (0:nz) arrays that the single fill exchanged as nz
    ! levels from level 0 (sequence association); keep exactly that slab.
    case (F_WRK3A); call hg_slab(wrk3a(is, js, 0), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_WRK3B); call hg_slab(wrk3b(is, js, 0), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_UBAR);  call hg_slab(ubar(is, js, sl), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_VBAR);  call hg_slab(vbar(is, js, sl), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_ZETA);  call hg_slab(zeta(is, js, sl), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case default
      error stop 'hg_dispatch: unknown field id'
    end select
  end subroutine hg_dispatch

  subroutine hg_slab(a, kb, ke, k, phase, i1, i2, j1, j2)
    ! Pack / unpack one slab a(is:ie, js:je, kb:ke) at group level offset
    ! hg_lev0(k) (pure copies).
    integer(ip), intent(in) :: kb, ke, i1, i2, j1, j2
    integer, intent(in) :: k, phase
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, kb:ke)
    integer(ip) :: i, j, m, kk, h, l0, ilo, ihi
    h = decomp%halo
    l0 = hg_lev0(k)
    ilo = i1 - h; ihi = i2 + h
    select case (phase)
    case (1)
      CUF_DO3
      do kk = kb, ke
        do j = j1, j2
          do m = 1, h
            gew_slo(m, j, l0 + kk - kb + 1) = a(i1 + m - 1, j, kk)
            gew_shi(m, j, l0 + kk - kb + 1) = a(i2 - h + m, j, kk)
          enddo
        enddo
      enddo
    case (2)
      if (decomp%nbr_w /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do j = j1, j2
            do m = 1, h
              a(i1 - h + m - 1, j, kk) = gew_rlo(m, j, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_e /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do j = j1, j2
            do m = 1, h
              a(i2 + m, j, kk) = gew_rhi(m, j, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
    case (3)
      CUF_DO3
      do kk = kb, ke
        do m = 1, h
          do i = ilo, ihi
            gns_slo(i, m, l0 + kk - kb + 1) = a(i, j1 + m - 1, kk)
            gns_shi(i, m, l0 + kk - kb + 1) = a(i, j2 - h + m, kk)
          enddo
        enddo
      enddo
    case (4)
      if (decomp%nbr_s /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do m = 1, h
            do i = ilo, ihi
              a(i, j1 - h + m - 1, kk) = gns_rlo(i, m, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_n /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do m = 1, h
            do i = ilo, ihi
              a(i, j2 + m, kk) = gns_rhi(i, m, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
    end select
  end subroutine hg_slab

  subroutine halo_fill_3d(a, stagger)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_3d(a, i1, i2, j1, j2, 1, decomp%nz)
    call exchange_ns_3d(a, i1, i2, j1, j2, 1, decomp%nz)
  end subroutine halo_fill_3d

  subroutine halo_fill_3d_k(a, stagger, k0, k1)
    ! halo_fill_3d for the levels k0..k1 only
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip), intent(in) :: stagger, k0, k1
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    if (k1 < k0) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_3d(a, i1, i2, j1, j2, k0, k1)
    call exchange_ns_3d(a, i1, i2, j1, j2, k0, k1)
  end subroutine halo_fill_3d_k

  subroutine halo_fill_3d_w(a, stagger)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_3dw(a, i1, i2, j1, j2, 0, decomp%nz)
    call exchange_ns_3dw(a, i1, i2, j1, j2, 0, decomp%nz)
  end subroutine halo_fill_3d_w

  ! Slot fills copy one time/tracer level onto an explicit-shape work
  ! array, then reuse halo_fill_2d/3d. Assumed-shape managed dummies
  ! become host descriptors in nvfortran and are illegal in CUF.
  ! The copies are kernels on work_stream, ordered after the caller's
  ! kernels; host loops here would race them on managed memory.
  subroutine halo_fill_2d_s(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, n, is, ie, js, je
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        work2d(i, j) = a(i, j, n)
      enddo
    enddo
    call halo_fill_2d(work2d, stagger)
    CUF_DO2
    do j = js, je
      do i = is, ie
        a(i, j, n) = work2d(i, j)
      enddo
    enddo
  end subroutine halo_fill_2d_s

  subroutine halo_fill_2d_st(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, nt)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, n, is, ie, js, je
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        work2d(i, j) = a(i, j, n)
      enddo
    enddo
    call halo_fill_2d(work2d, stagger)
    CUF_DO2
    do j = js, je
      do i = is, ie
        a(i, j, n) = work2d(i, j)
      enddo
    enddo
  end subroutine halo_fill_2d_st

  subroutine halo_fill_2d_sw(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, n, is, ie, js, je
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        work2d(i, j) = a(i, j, n)
      enddo
    enddo
    call halo_fill_2d(work2d, stagger)
    CUF_DO2
    do j = js, je
      do i = is, ie
        a(i, j, n) = work2d(i, j)
      enddo
    enddo
  end subroutine halo_fill_2d_sw

  subroutine halo_fill_3d_s(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, k, n, is, ie, js, je, k0, k1
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    k0 = 1; k1 = decomp%nz
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          work3d(i, j, k) = a(i, j, k, n)
        enddo
      enddo
    enddo
    call halo_fill_3d(work3d, stagger)
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          a(i, j, k, n) = work3d(i, j, k)
        enddo
      enddo
    enddo
  end subroutine halo_fill_3d_s

  subroutine halo_fill_3d_t(a, stagger, slot, itrc)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3, nt)
    integer(ip), intent(in) :: stagger, slot, itrc
    integer(ip) :: i, j, k, n, itr, is, ie, js, je, k0, k1
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    itr = itrc
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    k0 = 1; k1 = decomp%nz
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          work3d(i, j, k) = a(i, j, k, n, itr)
        enddo
      enddo
    enddo
    call halo_fill_3d(work3d, stagger)
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          a(i, j, k, n, itr) = work3d(i, j, k)
        enddo
      enddo
    enddo
  end subroutine halo_fill_3d_t

  subroutine tp_ensure_bufs(nk, ntr)
    ! Buffer layout tracer-innermost, (q, j, k) east-west and (q, i, k)
    ! north-south with q = (m-1)*ntr + itr, so a warp packs/unpacks
    ! consecutive tracers of one cell (one coalesced line of tp) and writes
    ! consecutive buffer elements. Same element count as (h, j, nz*ntr).
    integer(ip), intent(in) :: nk, ntr
    integer(ip) :: h, is, ie, js, je, nq, nz
    h = decomp%halo; is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nq = h * ntr; nz = nk / ntr
    if (allocated(tew_slo)) then
      if (ubound(tew_slo, 1) /= nq .or. ubound(tew_slo, 3) /= nz .or. lbound(tew_slo, 2) /= js .or. &
          ubound(tew_slo, 2) /= je .or. lbound(tns_slo, 2) /= is .or. ubound(tns_slo, 2) /= ie) then
        deallocate(tew_slo, tew_shi, tew_rlo, tew_rhi, tns_slo, tns_shi, tns_rlo, tns_rhi)
      endif
    endif
    if (.not. allocated(tew_slo)) then
      allocate(tew_slo(nq, js:je, nz), tew_shi(nq, js:je, nz), tew_rlo(nq, js:je, nz), tew_rhi(nq, js:je, nz))
      allocate(tns_slo(nq, is:ie, nz), tns_shi(nq, is:ie, nz), tns_rlo(nq, is:ie, nz), tns_rhi(nq, is:ie, nz))
      tew_slo = 0.0_rp; tew_shi = 0.0_rp; tew_rlo = 0.0_rp; tew_rhi = 0.0_rp
      tns_slo = 0.0_rp; tns_shi = 0.0_rp; tns_rlo = 0.0_rp; tns_rhi = 0.0_rp
    endif
  end subroutine tp_ensure_bufs

  subroutine tp_pack_ew(a, ntr, nsl, n, kb, nz, h, i1, i2, j1, j2)
    ! slot n of a(ntr, 0:ntpcol, nz, nsl), levels kb..kb+nz-1 <-> buffer planes 1..nz
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, nsl)
    integer(ip), intent(in) :: ntr, nsl, n, kb, nz, h, i1, i2, j1, j2
    integer(ip) :: q, j, m, k, itr
    CUF_DO3
    do k = 1, nz
      do j = j1, j2
        do q = 1, h * ntr
          m = (q - 1) / ntr + 1
          itr = q - (m - 1) * ntr
          tew_slo(q, j, k) = real(a(itr, tpcol(i1 + m - 1, j), kb + k - 1, n), rp)
          tew_shi(q, j, k) = real(a(itr, tpcol(i2 - h + m, j), kb + k - 1, n), rp)
        enddo
      enddo
    enddo
  end subroutine tp_pack_ew

  subroutine tp_unpack_ew(a, ntr, nsl, n, kb, nz, h, i1, i2, j1, j2)
    ! slot n of a(ntr, 0:ntpcol, nz, nsl), levels kb..kb+nz-1 <-> buffer planes 1..nz
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, nsl)
    integer(ip), intent(in) :: ntr, nsl, n, kb, nz, h, i1, i2, j1, j2
    integer(ip) :: q, j, m, k, itr
    logical :: w, e
    w = (decomp%nbr_w /= NBR_NONE); e = (decomp%nbr_e /= NBR_NONE)
    if (.not. (w .or. e)) return
    ! west and east halo columns in one launch (disjoint cells)
    CUF_DO3
    do k = 1, nz
      do j = j1, j2
        do q = 1, h * ntr
          m = (q - 1) / ntr + 1
          itr = q - (m - 1) * ntr
          if (w) then
            if (tpcol(i1 - h + m - 1, j) > 0) a(itr, tpcol(i1 - h + m - 1, j), kb + k - 1, n) = real(tew_rlo(q, j, k), pp)
          endif
          if (e) then
            if (tpcol(i2 + m, j) > 0) a(itr, tpcol(i2 + m, j), kb + k - 1, n) = real(tew_rhi(q, j, k), pp)
          endif
        enddo
      enddo
    enddo
  end subroutine tp_unpack_ew

  subroutine tp_pack_ns(a, ntr, nsl, n, kb, nz, h, ilo, ihi, j1, j2)
    ! slot n of a(ntr, 0:ntpcol, nz, nsl), levels kb..kb+nz-1 <-> buffer planes 1..nz
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, nsl)
    integer(ip), intent(in) :: ntr, nsl, n, kb, nz, h, ilo, ihi, j1, j2
    integer(ip) :: q, i, m, k, itr
    CUF_DO3
    do k = 1, nz
      do i = ilo, ihi
        do q = 1, h * ntr
          m = (q - 1) / ntr + 1
          itr = q - (m - 1) * ntr
          tns_slo(q, i, k) = real(a(itr, tpcol(i, j1 + m - 1), kb + k - 1, n), rp)
          tns_shi(q, i, k) = real(a(itr, tpcol(i, j2 - h + m), kb + k - 1, n), rp)
        enddo
      enddo
    enddo
  end subroutine tp_pack_ns

  subroutine tp_unpack_ns(a, ntr, nsl, n, kb, nz, h, ilo, ihi, j1, j2)
    ! slot n of a(ntr, 0:ntpcol, nz, nsl), levels kb..kb+nz-1 <-> buffer planes 1..nz
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, nsl)
    integer(ip), intent(in) :: ntr, nsl, n, kb, nz, h, ilo, ihi, j1, j2
    integer(ip) :: q, i, m, k, itr
    logical :: so, no
    so = (decomp%nbr_s /= NBR_NONE); no = (decomp%nbr_n /= NBR_NONE)
    if (.not. (so .or. no)) return
    ! south and north halo rows in one launch (disjoint cells)
    CUF_DO3
    do k = 1, nz
      do i = ilo, ihi
        do q = 1, h * ntr
          m = (q - 1) / ntr + 1
          itr = q - (m - 1) * ntr
          if (so) then
            if (tpcol(i, j1 - h + m - 1) > 0) a(itr, tpcol(i, j1 - h + m - 1), kb + k - 1, n) = real(tns_rlo(q, i, k), pp)
          endif
          if (no) then
            if (tpcol(i, j2 + m) > 0) a(itr, tpcol(i, j2 + m), kb + k - 1, n) = real(tns_rhi(q, i, k), pp)
          endif
        enddo
      enddo
    enddo
  end subroutine tp_unpack_ns

  subroutine sync_work_stream()
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'halo: work stream sync'
#endif
  end subroutine sync_work_stream

  subroutine halo_fill_3d_tp(a, ntr, slot)
    ! Passive tracers, tracer-innermost layout a(ntr, is:ie, js:je, nz, 3):
    ! one exchange of all tracers in slot `slot` at rho points. Pure copies,
    ! so a tiled run reproduces the single-tile interior bit for bit. The
    ! combined buffer index is kk = (k-1)*ntr + itr; east-west first, then
    ! north-south including the east-west halo columns (as halo_fill_3d).
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: ntr, slot
    integer(ip) :: n, h, nk, i1, i2, j1, j2, ilo, ihi
    if (no_neighbors()) return
    if (ntr <= 0) return
    if (tp_pending /= 0) error stop 'halo_fill_3d_tp: a non-blocking all-tracer exchange is pending'
    h = decomp%halo; nk = decomp%nz * ntr; n = slot
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    call tp_ensure_bufs(nk, ntr)
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      call tp_pack_ew(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, i1, i2, j1, j2)
      call deliver3(decomp%nbr_w, decomp%nbr_e, tew_slo, tew_shi, tew_rlo, tew_rhi)
      call tp_unpack_ew(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, i1, i2, j1, j2)
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      call tp_pack_ns(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, ilo, ihi, j1, j2)
      call deliver3(decomp%nbr_s, decomp%nbr_n, tns_slo, tns_shi, tns_rlo, tns_rhi)
      call tp_unpack_ns(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, ilo, ihi, j1, j2)
    endif
  end subroutine halo_fill_3d_tp

  subroutine halo_fill_3d_tp_k(a, ntr, nsl, slot, k0, k1)
    ! halo_fill_3d_tp for the levels k0..k1 only of slot `slot` of
    ! a(ntr, 0:ntpcol, nz, nsl): only those planes are packed and sent (the
    ! BGC subcycling window's sub-passes; nsl = 1 for its shadow tracer)
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, nsl)
    integer(ip), intent(in) :: ntr, nsl, slot, k0, k1
    integer(ip) :: h, nk, i1, i2, j1, j2, ilo, ihi
    if (no_neighbors()) return
    if (ntr <= 0 .or. k1 < k0) return
    if (tp_pending /= 0) error stop 'halo_fill_3d_tp_k: a non-blocking all-tracer exchange is pending'
    h = decomp%halo; nk = k1 - k0 + 1
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    call tp_ensure_bufs(decomp%nz * ntr, ntr)
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      call tp_pack_ew(a, ntr, nsl, slot, k0, nk, h, i1, i2, j1, j2)
      call deliver3(decomp%nbr_w, decomp%nbr_e, tew_slo(:, :, 1:nk), tew_shi(:, :, 1:nk), &
                    tew_rlo(:, :, 1:nk), tew_rhi(:, :, 1:nk))
      call tp_unpack_ew(a, ntr, nsl, slot, k0, nk, h, i1, i2, j1, j2)
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      call tp_pack_ns(a, ntr, nsl, slot, k0, nk, h, ilo, ihi, j1, j2)
      call deliver3(decomp%nbr_s, decomp%nbr_n, tns_slo(:, :, 1:nk), tns_shi(:, :, 1:nk), &
                    tns_rlo(:, :, 1:nk), tns_rhi(:, :, 1:nk))
      call tp_unpack_ns(a, ntr, nsl, slot, k0, nk, h, ilo, ihi, j1, j2)
    endif
  end subroutine halo_fill_3d_tp_k

  ! ---- the same exchange in three non-blocking pieces, so that the MPI
  ! transfers run under unrelated kernels: start packs and posts the
  ! east-west round; mid waits for it, unpacks, packs and posts the
  ! north-south round (which carries the east-west halo columns); finish
  ! waits and unpacks. Between start and finish the slot's interior must
  ! not change and its halo must not be read. The same call sequence runs
  ! on every rank. Without GPU-direct MPI, start does the blocking exchange.
  subroutine halo_tp_start(a, ntr, slot)
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: ntr, slot
    integer(ip) :: n, h, nk, i1, i2, j1, j2
    if (no_neighbors()) return
    if (ntr <= 0) return
    if (tp_pending /= 0) error stop 'halo_tp_start: the previous all-tracer exchange was not finished'
    if (.not. comm_gpu_direct) then
      call halo_fill_3d_tp(a, ntr, slot)
      return
    endif
    if (tp_check_wanted()) continue
    h = decomp%halo; nk = decomp%nz * ntr; n = slot
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    call tp_ensure_bufs(nk, ntr)
    tp_slot_p = int(slot); tp_ntr_p = int(ntr)
    tp_ew_posted = .false.; tp_ns_posted = .false.
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      call tp_pack_ew(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, i1, i2, j1, j2)
      call sync_work_stream()
      call comm_post_dev(decomp%nbr_w, decomp%nbr_e, int(size(tew_slo)), tew_slo, tew_shi, tew_rlo, tew_rhi, &
                         321, 322, tp_req_ew)
      tp_ew_posted = .true.
    endif
    tp_pending = 1
  end subroutine halo_tp_start

  subroutine halo_tp_mid(a, ntr)
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: ntr
    integer(ip) :: n, h, nk, i1, i2, j1, j2, ilo, ihi
    if (tp_pending /= 1) return
    if (int(ntr) /= tp_ntr_p) error stop 'halo_tp_mid: tracer count changed'
    h = decomp%halo; nk = decomp%nz * ntr; n = int(tp_slot_p, ip)
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    if (tp_ew_posted) then
      call comm_wait_dev(tp_req_ew)
      if (tp_devsync) call sync_arch()
      call tp_unpack_ew(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, i1, i2, j1, j2)
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      call tp_pack_ns(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, ilo, ihi, j1, j2)
      call sync_work_stream()
      call comm_post_dev(decomp%nbr_s, decomp%nbr_n, int(size(tns_slo)), tns_slo, tns_shi, tns_rlo, tns_rhi, &
                         323, 324, tp_req_ns)
      tp_ns_posted = .true.
      tp_pending = 2
    else
      tp_pending = 0
    endif
  end subroutine halo_tp_mid

  subroutine halo_tp_finish(a, ntr)
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: ntr
    integer(ip) :: n, h, nk, i1, i2, j1, j2, ilo, ihi
    if (tp_pending == 1) call halo_tp_mid(a, ntr)
    if (tp_pending /= 2) then
      tp_pending = 0
      return
    endif
    h = decomp%halo; nk = decomp%nz * ntr; n = int(tp_slot_p, ip)
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    ilo = i1 - h; ihi = i2 + h
    call comm_wait_dev(tp_req_ns)
    if (tp_devsync) call sync_arch()
    call tp_unpack_ns(a, ntr, 3_ip, n, 1_ip, decomp%nz, h, ilo, ihi, j1, j2)
    tp_pending = 0
    if (tp_check_wanted()) call tp_check(a, ntr, n)
  end subroutine halo_tp_finish

  logical function tp_overlap_on(window)
    ! ROMS_GPU_TP_OVERLAP bitmask (default 3): bit 1 = the slot-3 exchange
    ! posted by pre_step3d, bit 2 = the end-of-step exchange posted by
    ! passive_t3dmix. A window that is off runs the blocking exchange.
    integer, intent(in) :: window
    logical, save :: checked = .false.
    integer, save :: mask = 3
    character(len=16) :: val
    integer :: stat
    if (.not. checked) then
      checked = .true.
      call get_environment_variable('ROMS_GPU_TP_OVERLAP', val, status=stat)
      if (stat == 0 .and. len_trim(val) > 0) then
        read(val, *, iostat=stat) mask
        if (stat /= 0) mask = 3
      endif
      write(*,'(a,i0)') 'halo: all-tracer exchange overlap mask = ', mask
    endif
    tp_overlap_on = (iand(mask, window) /= 0)
  end function tp_overlap_on

  logical function tp_check_wanted()
    ! ROMS_GPU_TP_CHECK=1: after every overlapped all-tracer exchange, redo
    ! it blockingly and report halo cells that differ (debug aid).
    logical, save :: checked = .false., wanted = .false.
    character(len=16) :: val
    integer :: stat
    if (.not. checked) then
      checked = .true.
      call get_environment_variable('ROMS_GPU_TP_CHECK', val, status=stat)
      wanted = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '1')
      if (wanted) write(*,'(a)') 'halo: overlapped all-tracer exchanges are re-checked (ROMS_GPU_TP_CHECK)'
      call get_environment_variable('ROMS_GPU_TP_DEVSYNC', val, status=stat)
      tp_devsync = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '1')
      if (tp_devsync) write(*,'(a)') 'halo: device sync after each overlapped exchange wait (ROMS_GPU_TP_DEVSYNC)'
    endif
    tp_check_wanted = wanted
  end function tp_check_wanted

  subroutine tp_check(a, ntr, n)
    ! Exact check: keep a device copy of slot n, redo the exchange
    ! blockingly, count halo values that differ (integer reduction).
    real(pp), WORK_ATTR intent(inout) :: a(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: ntr, n
    integer(ip) :: i1, i2, j1, j2, i, j, k, itr, nz, is, ie, js, je, nbad, c
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    nz = decomp%nz; is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    if (.not. allocated(tp_chk)) allocate(tp_chk(ntr, 0:ntpcol, nz))
    CUF_DO3
    do k = 1, nz
      do c = 0, ntpcol
        do itr = 1, ntr
          tp_chk(itr, c, k) = a(itr, c, k, n)
        enddo
      enddo
    enddo
    call sync_work_stream()
    call halo_fill_3d_tp(a, ntr, n)
    nbad = 0
    CUF_DO3
    do k = 1, nz
      do j = js, je
        do i = is, ie
          if (i < i1 .or. i > i2 .or. j < j1 .or. j > j2) then
            c = tpcol(i, j)
            do itr = 1, ntr
              if (tp_chk(itr, c, k) /= a(itr, c, k, n)) nbad = nbad + 1
            enddo
          endif
        enddo
      enddo
    enddo
    call sync_work_stream()
    if (nbad /= 0) then
      write(*,'(a,i0,a,i0,a,i0,a,i0)') 'TP EXCHANGE MISMATCH rank ', comm_rank(), ' iic ', iic, &
           ' slot ', n, ' differing halo values ', nbad
      flush(6)
    endif
  end subroutine tp_check

  subroutine halo_fill_3d_w_s(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz, 2)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, k, n, is, ie, js, je, k0, k1
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    k0 = 0; k1 = decomp%nz
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          work3dw(i, j, k) = a(i, j, k, n)
        enddo
      enddo
    enddo
    call halo_fill_3d_w(work3dw, stagger)
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          a(i, j, k, n) = work3dw(i, j, k)
        enddo
      enddo
    enddo
  end subroutine halo_fill_3d_w_s

  logical function no_neighbors()
    ! Single non-periodic tile: every halo point is a physical-edge ghost
    ! set by the boundary code, so a fill has nothing to deliver.
    no_neighbors = decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE .and. &
                   decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE
  end function no_neighbors

  subroutine interior_bounds(stagger, i1, i2, j1, j2)
    integer(ip), intent(in)  :: stagger
    integer(ip), intent(out) :: i1, i2, j1, j2
    ! Exchange bounds are the rho-cell range for every stagger. A u (v)
    ! face on a seam is the interior point 1 of the east (north) tile and
    ! the point nx+1 (ny+1) of the west (south) tile: the kernels compute it
    ! on the east/north side (uv_rhs stops at nx toward a neighbor), so the
    ! exchange carries it, with one more point beyond, to the west/south
    ! tile as its halo a(nx+1), a(nx+2). Periodic wrap gives a(nx+1) = a(1).
    ! The physical boundary rows (j = 0 at a south edge, ny+1 at a north
    ! edge) are computed by the boundary routines with tangential neighbours
    ! (u2dbc/v2dbc/u3dbc/v3dbc/t3dbc radiation), so an east-west seam on an
    ! open edge needs them exchanged too. The north-south round already
    ! covers the columns 0 and nx+1 (it packs i1-h .. i2+h).
    i1 = 1
    j1 = 1
    i2 = decomp%nx
    j2 = decomp%ny
    if (decomp%edge_s) j1 = 0
    if (decomp%edge_n) j2 = decomp%ny + 1
  end subroutine interior_bounds

  subroutine exchange_ew_2d(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, j, m
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO2
    do j = j1, j2
      do m = 1, h
        ew_slo(m, j, 0) = a(i1 + m - 1, j)
        ew_shi(m, j, 0) = a(i2 - h + m, j)
      enddo
    enddo
    call deliver2_ew(decomp%nbr_w, decomp%nbr_e)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i1 - h + m - 1, j) = ew_rlo(m, j, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i2 + m, j) = ew_rhi(m, j, 0)
        enddo
      enddo
    endif
  end subroutine exchange_ew_2d

  subroutine exchange_ns_2d(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, i, m, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO2
    do m = 1, h
      do i = ilo, ihi
        ns_slo(i, m, 0) = a(i, j1 + m - 1)
        ns_shi(i, m, 0) = a(i, j2 - h + m)
      enddo
    enddo
    call deliver2_ns(decomp%nbr_s, decomp%nbr_n)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j1 - h + m - 1) = ns_rlo(i, m, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j2 + m) = ns_rhi(i, m, 0)
        enddo
      enddo
    endif
  end subroutine exchange_ns_2d

  subroutine exchange_ew_2d_n(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, j, m, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO2
    do j = j1, j2
      do m = 1, h
        n = fast%knew
        ew_slo(m, j, 0) = a(i1 + m - 1, j, n)
        ew_shi(m, j, 0) = a(i2 - h + m, j, n)
      enddo
    enddo
    call deliver2_ew(decomp%nbr_w, decomp%nbr_e)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          a(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          a(i2 + m, j, n) = ew_rhi(m, j, 0)
        enddo
      enddo
    endif
  end subroutine exchange_ew_2d_n

  subroutine exchange_ns_2d_n(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, i, m, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO2
    do m = 1, h
      do i = ilo, ihi
        n = fast%knew
        ns_slo(i, m, 0) = a(i, j1 + m - 1, n)
        ns_shi(i, m, 0) = a(i, j2 - h + m, n)
      enddo
    enddo
    call deliver2_ns(decomp%nbr_s, decomp%nbr_n)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          a(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          a(i, j2 + m, n) = ns_rhi(i, m, 0)
        enddo
      enddo
    endif
  end subroutine exchange_ns_2d_n

  subroutine exchange_ew_3d(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, j, m, k
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k)
          ew_shi(m, j, k) = a(i2 - h + m, j, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ew_3d

  subroutine exchange_ns_3d(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, i, m, k, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k)
          ns_shi(i, m, k) = a(i, j2 - h + m, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ns_3d

  subroutine exchange_ew_3dw(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, j, m, k
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k)
          ew_shi(m, j, k) = a(i2 - h + m, j, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ew_3dw

  subroutine exchange_ns_3dw(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, i, m, k, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k)
          ns_shi(i, m, k) = a(i, j2 - h + m, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ns_3dw

  subroutine exchange_ew_2d_s(a, i1, i2, j1, j2, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, slot
    integer(ip) :: h, j, m, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    CUF_DO2
    do j = j1, j2
      do m = 1, h
        ew_slo(m, j, 0) = a(i1 + m - 1, j, n)
        ew_shi(m, j, 0) = a(i2 - h + m, j, n)
      enddo
    enddo
    call deliver2_ew(decomp%nbr_w, decomp%nbr_e)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i2 + m, j, n) = ew_rhi(m, j, 0)
        enddo
      enddo
    endif
  end subroutine exchange_ew_2d_s

  subroutine exchange_ns_2d_s(a, i1, i2, j1, j2, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, slot
    integer(ip) :: h, i, m, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO2
    do m = 1, h
      do i = ilo, ihi
        ns_slo(i, m, 0) = a(i, j1 + m - 1, n)
        ns_shi(i, m, 0) = a(i, j2 - h + m, n)
      enddo
    enddo
    call deliver2_ns(decomp%nbr_s, decomp%nbr_n)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j2 + m, n) = ns_rhi(i, m, 0)
        enddo
      enddo
    endif
  end subroutine exchange_ns_2d_s

  subroutine exchange_ew_3d_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, j, m, k, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k, n)
          ew_shi(m, j, k) = a(i2 - h + m, j, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k, n) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k, n) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ew_3d_s

  subroutine exchange_ns_3d_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, i, m, k, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k, n)
          ns_shi(i, m, k) = a(i, j2 - h + m, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k, n) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k, n) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ns_3d_s

  subroutine exchange_ew_3d_t(a, i1, i2, j1, j2, k0, k1, slot, itrc)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot, itrc
    integer(ip) :: h, j, m, k, n, itr
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    itr = itrc
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k, n, itr)
          ew_shi(m, j, k) = a(i2 - h + m, j, k, n, itr)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k, n, itr) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k, n, itr) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ew_3d_t

  subroutine exchange_ns_3d_t(a, i1, i2, j1, j2, k0, k1, slot, itrc)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot, itrc
    integer(ip) :: h, i, m, k, n, itr, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    itr = itrc
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k, n, itr)
          ns_shi(i, m, k) = a(i, j2 - h + m, k, n, itr)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k, n, itr) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k, n, itr) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ns_3d_t

  subroutine exchange_ew_3dw_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, j, m, k, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k, n)
          ew_shi(m, j, k) = a(i2 - h + m, j, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k, n) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k, n) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ew_3dw_s

  subroutine exchange_ns_3dw_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, i, m, k, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k, n)
          ns_shi(i, m, k) = a(i, j2 - h + m, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k, n) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k, n) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
  end subroutine exchange_ns_3dw_s

  ! Host-side sections of a managed array are legal. A CUF kernel on a
  ! section is not: nvfortran materializes a host temp. Index the parent
  ! pack buffers (k = 0 for 2-D) instead of passing ew_slo(:,:,0).
  subroutine deliver2_ew(nbr_lo, nbr_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer(ip) :: i, j, lo, hi, h, js, je
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
#ifdef USE_CUDA
      call deliver_dev(nbr_lo, nbr_hi, ew_slo(:,:,0:0), ew_shi(:,:,0:0), ew_rlo(:,:,0:0), ew_rhi(:,:,0:0))
#else
      call sync_arch()
      call comm_exchange2(nbr_lo, nbr_hi, ew_slo(:,:,0), ew_shi(:,:,0), &
                          ew_rlo(:,:,0), ew_rhi(:,:,0))
#endif
      return
    endif
    h = decomp%halo
    js = decomp%js
    je = decomp%je
    lo = nbr_lo
    hi = nbr_hi
    CUF_DO2
    do j = js, je
      do i = 1, h
        ew_rlo(i, j, 0) = 0.0_rp
        ew_rhi(i, j, 0) = 0.0_rp
        if (lo == NBR_SELF .and. hi == NBR_SELF) then
          ew_rhi(i, j, 0) = ew_slo(i, j, 0)
          ew_rlo(i, j, 0) = ew_shi(i, j, 0)
        elseif (lo == NBR_SELF) then
          ew_rlo(i, j, 0) = ew_shi(i, j, 0)
        elseif (hi == NBR_SELF) then
          ew_rhi(i, j, 0) = ew_slo(i, j, 0)
        endif
      enddo
    enddo
  end subroutine deliver2_ew

  subroutine deliver2_ns(nbr_lo, nbr_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer(ip) :: i, m, lo, hi, h, is, ie
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
#ifdef USE_CUDA
      call deliver_dev(nbr_lo, nbr_hi, ns_slo(:,:,0:0), ns_shi(:,:,0:0), ns_rlo(:,:,0:0), ns_rhi(:,:,0:0))
#else
      call sync_arch()
      call comm_exchange2(nbr_lo, nbr_hi, ns_slo(:,:,0), ns_shi(:,:,0), &
                          ns_rlo(:,:,0), ns_rhi(:,:,0))
#endif
      return
    endif
    h = decomp%halo
    is = decomp%is
    ie = decomp%ie
    lo = nbr_lo
    hi = nbr_hi
    CUF_DO2
    do m = 1, h
      do i = is, ie
        ns_rlo(i, m, 0) = 0.0_rp
        ns_rhi(i, m, 0) = 0.0_rp
        if (lo == NBR_SELF .and. hi == NBR_SELF) then
          ns_rhi(i, m, 0) = ns_slo(i, m, 0)
          ns_rlo(i, m, 0) = ns_shi(i, m, 0)
        elseif (lo == NBR_SELF) then
          ns_rlo(i, m, 0) = ns_shi(i, m, 0)
        elseif (hi == NBR_SELF) then
          ns_rhi(i, m, 0) = ns_slo(i, m, 0)
        endif
      enddo
    enddo
  end subroutine deliver2_ns

  subroutine deliver3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi, ev_wait)
    ! ev_wait: the packs were followed by zuv_ev; wait for that event, not
    ! for everything queued on work_stream (GPU-direct path).
    logical, intent(in), optional :: ev_wait
    integer(ip), intent(in) :: nbr_lo, nbr_hi
#ifdef USE_CUDA
    real(rp), device, intent(inout), contiguous :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), device, intent(inout), contiguous :: recv_lo(:,:,:), recv_hi(:,:,:)
#else
    real(rp), intent(inout), contiguous :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), intent(inout), contiguous :: recv_lo(:,:,:), recv_hi(:,:,:)
#endif
    integer(ip) :: i, j, k, n1, n2, n3, lo, hi
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
#ifdef USE_CUDA
      call deliver_dev(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi, ev_wait)
#else
      call sync_arch()
      call comm_exchange3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
#endif
      return
    endif
    n1 = size(send_lo, 1)
    n2 = size(send_lo, 2)
    n3 = size(send_lo, 3)
    lo = nbr_lo
    hi = nbr_hi
    CUF_DO3
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          recv_lo(i, j, k) = 0.0_rp
          recv_hi(i, j, k) = 0.0_rp
          if (lo == NBR_SELF .and. hi == NBR_SELF) then
            recv_hi(i, j, k) = send_lo(i, j, k)
            recv_lo(i, j, k) = send_hi(i, j, k)
          elseif (lo == NBR_SELF) then
            recv_lo(i, j, k) = send_hi(i, j, k)
          elseif (hi == NBR_SELF) then
            recv_hi(i, j, k) = send_lo(i, j, k)
          endif
        enddo
      enddo
    enddo
  end subroutine deliver3

#ifdef USE_CUDA
  subroutine deliver_dev(nbr_lo, nbr_hi, s_lo, s_hi, r_lo, r_hi, ev_wait)
    ! MPI delivery of packed device buffers through pinned host memory,
    ! stream-ordered after the pack kernels and before the unpack kernels.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), device, intent(inout), contiguous :: s_lo(:,:,:), s_hi(:,:,:)
    real(rp), device, intent(inout), contiguous :: r_lo(:,:,:), r_hi(:,:,:)
    logical, intent(in), optional :: ev_wait
    integer :: n, ierr
    logical :: use_ev
    use_ev = .false.
    if (present(ev_wait)) use_ev = ev_wait
    n = size(s_lo)
    if (size(s_hi) /= n .or. size(r_lo) /= n .or. size(r_hi) /= n) &
         error stop 'deliver_dev: buffer size mismatch'
    if (comm_gpu_direct) then
      ! GPU-aware MPI: the pack kernels (work_stream) must have completed;
      ! the blocking Sendrecv returns with r_lo/r_hi filled on the device.
      if (use_ev) then
        ierr = cudaEventSynchronize(zuv_ev)
      else
        ierr = cudaStreamSynchronize(work_stream)
      endif
      if (ierr /= 0) error stop 'deliver_dev: stream sync'
      call comm_sendrecv_dev(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi)
      return
    endif
    if (hbuf_n < n) then
      if (allocated(hs_lo)) deallocate(hs_lo, hs_hi, hr_lo, hr_hi)
      allocate(hs_lo(n), hs_hi(n), hr_lo(n), hr_hi(n))
      hbuf_n = n
    endif
    ! Synchronous copies: the legacy default stream orders them after the
    ! pack kernels and before the unpack kernels on work_stream (a blocking
    ! stream). cudaMemcpyAsync does not resolve for these managed dummies.
    ierr = cudaMemcpy(hs_lo, s_lo, n)
    if (ierr /= 0) error stop 'deliver_dev: D2H lo'
    ierr = cudaMemcpy(hs_hi, s_hi, n)
    if (ierr /= 0) error stop 'deliver_dev: D2H hi'
    call comm_sendrecv_raw(nbr_lo, nbr_hi, n, hs_lo, hs_hi, hr_lo, hr_hi)
    ierr = cudaMemcpy(r_lo, hr_lo, n)
    if (ierr /= 0) error stop 'deliver_dev: H2D lo'
    ierr = cudaMemcpy(r_hi, hr_hi, n)
    if (ierr /= 0) error stop 'deliver_dev: H2D hi'
  end subroutine deliver_dev
#endif

end module halo_mod

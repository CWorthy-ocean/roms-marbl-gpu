#include "mem_attr.inc"
module time_state
  ! Same index dance as UCLA ROMS. Do not simplify on the GPU.
  use kinds
  implicit none
  private

  integer(ip), public :: nstp = 1
  integer(ip), public :: nnew = 2
  integer(ip), public :: nrhs = 1
  integer(ip), public :: knew = 1
  integer(ip), public :: kstp = 1
  integer(ip), public :: iic  = 0
  integer(ip), public :: iif  = 0
  integer(ip), public :: nfast = 0
  integer(ip), public :: ndtfast = 60
  integer(ip), public :: ntimes = 0
  ! Step diagnostics (coupling, max |u|, Akv, hbls) every ninfo_steps steps.
  integer(ip), public :: ninfo_steps = 1
  integer(ip), public :: nt = 2          ! temp, salt in v1; grow later

  real(rp), public :: dt = 100.0_rp
  real(rp), public :: dtfast = 0.0_rp
  real(rp), public :: time = 0.0_rp
  real(rp), public :: start_time = 0.0_rp
  real(rp), public :: tdays = 0.0_rp
  real(rp), parameter, public :: sec2day = 1.0_rp / 86400.0_rp
  integer(ip), public :: frc_slot = 0   ! 0=n, 1=n+1/2, 2=n+1, 3=n+3/2

  real(rp), WORK_ATTR public, allocatable :: weight(:,:)   ! (2, 2*ndtfast)
  real(rp), public :: theta_s = 6.0_rp
  real(rp), public :: theta_b = 6.0_rp
  real(rp), public :: hc = 250.0_rp
  real(rp), WORK_ATTR public, allocatable :: Cs_w(:)       ! 0:nz
  real(rp), WORK_ATTR public, allocatable :: Cs_r(:)       ! 1:nz

  integer(ip), public :: ntstart = 1
  ! Diagnostic probes are off unless ROMS_GPU_PROBES is set to a non-zero
  ! value; set_probe_windows (step_driver) then opens the windows below:
  ! the first 24 steps, the P1 fast-loop window at steps 5-8
  ! (iic-ntstart = 4..7) and the KPP surface-layer window at steps 1-24
  ! after the restart (iic-ntstart = 0..23).
  integer(ip), public :: probe_steps = 0
  logical, public :: probe_on = .false.
  integer(ip), public :: fast_probe_from = 0, fast_probe_to = 0
  logical, public :: fast_probe_on = .false.
  integer(ip), public :: kpp_probe_from = 0, kpp_probe_to = -1
  ! Probe points, global model indices: the local indices of a 1 x 1 run
  ! (and the 0-based indices of a history field that includes the
  ! boundary points), chosen on the development grid from the day-run
  ! CPU-GPU residual:
  !   P1 (433,432) NtT (closed)
  !   P2 (372,2)   south-rim
  !   P3 (512,266) east-edge
  !   P4 (456,56)  west U-face of the SW Ri neighbor
  ! P1 NtT, P2 wet AkvW rho (455,56), P3 east, P4 U-face (456,56).
  ! The probes read the rank-local copies iprb_*/jprb_*, which
  ! set_probe_windows (step_driver) fills when probes are on: the local
  ! index if the point is inside the grid and owned by this rank,
  ! otherwise prb_off, which every probe's range check rejects.
  integer(ip), parameter, public :: prb_off = -100000
  ! Rho points; a probe at rho cell (i,j) also reads its west U-face and
  ! south V-face.
  integer(ip), parameter, public :: nprb_pt = 4
  integer(ip), parameter, public :: iprb_pt_g(nprb_pt) = (/ 433, 455, 512, 456 /)
  integer(ip), parameter, public :: jprb_pt_g(nprb_pt) = (/ 432, 56, 266, 56 /)
  ! U-faces at the retargeted rho cells.
  integer(ip), parameter, public :: nprb_u = 4
  integer(ip), parameter, public :: iprb_u_g(nprb_u) = (/ 433, 372, 512, 456 /)
  integer(ip), parameter, public :: jprb_u_g(nprb_u) = (/ 432, 2, 266, 56 /)
  ! V-faces: NtT last interior / NtT north OBC / south OBC / south first interior.
  integer(ip), parameter, public :: nprb_v = 4
  integer(ip), parameter, public :: iprb_v_g(nprb_v) = (/ 433, 433, 372, 372 /)
  integer(ip), parameter, public :: jprb_v_g(nprb_v) = (/ 432, 433, 1, 2 /)
  integer(ip), public :: iprb_pt(nprb_pt) = prb_off, jprb_pt(nprb_pt) = prb_off
  integer(ip), public :: iprb_u(nprb_u) = prb_off, jprb_u(nprb_u) = prb_off
  integer(ip), public :: iprb_v(nprb_v) = prb_off, jprb_v(nprb_v) = prb_off
  ! FIRST_TIME_STEP is iic==forw_start. Exact restart keeps this at 1 so
  ! the first step after a two-record load uses both AB/AM time levels.
  ! Approximate / single-record start sets it to ntstart (forward step).
  integer(ip), public :: forw_start = 1
  ! Wet-face location of max |u| in the loaded IC (slot 1).
  integer(ip), public :: ic_u_i = 0, ic_u_j = 0, ic_u_k = 0
  real(rp), public :: ic_u_val = 0.0_rp

  ! Host writes this each iif; step2d/halo kernels read it so a CUDA
  ! graph can replay with new time indices and FB weights.
  type, public :: fast_pack_t
    integer(ip) :: iif = 0, kstp = 1, knew = 1, kbak = 1, kold = 1
    integer(ip) :: is_first = 0, is_last = 0
    integer(ip) :: i_u_lo = 1, i_u_hi = 1, j_v_lo = 1, j_v_hi = 1
    integer(ip) :: nx = 0, ny = 0, is = 0, ie = 0, js = 0, je = 0
    real(rp) :: fwd = 1.0_rp, fwd1 = 0.0_rp, fwd2 = 0.0_rp
    real(rp) :: bkw_new = 0.0_rp, bkw = 1.0_rp, bkw1 = 0.0_rp, bkw2 = 0.0_rp
    real(rp) :: w1 = 0.0_rp, w2 = 0.0_rp, half_dt = 0.0_rp, half_w1 = 0.0_rp
    real(rp) :: dtfast = 0.0_rp
  end type fast_pack_t
  type(fast_pack_t), WORK_ATTR public, save :: fast

  public :: init_time_state
  public :: set_scoord
  public :: set_fast_weights
  public :: advance_slow_indices
  public :: cycle_fast_index
  public :: begin_predictor_indices
  public :: begin_corrector_indices
  public :: is_first_time_step
  public :: is_first_run_step
  public :: update_clock
  public :: frc_modtime

contains

  subroutine init_time_state(nz_in, ndtfast_in, dt_in, ntimes_in, nt_in)
    use formulas, only: AM3_crv
    integer(ip), intent(in) :: nz_in, ndtfast_in, ntimes_in, nt_in
    real(rp),    intent(in) :: dt_in

    ndtfast = ndtfast_in
    dt = dt_in
    dtfast = dt / real(ndtfast, rp)
    ntimes = ntimes_in
    nt = nt_in
    nstp = 1
    nnew = 2
    nrhs = 1
    knew = 1
    kstp = 1
    iic = 0
    iif = 0
    ntstart = 1
    forw_start = 1
    time = 0.0_rp
    tdays = 0.0_rp
    frc_slot = 0
    call set_fast_weights()
    call set_scoord(nz_in)
    if (AM3_crv <= 0.0_rp) error stop 'AM3_crv must be 1/6'
  end subroutine init_time_state

  subroutine set_scoord(nz)
    use formulas, only: csf
    integer(ip), intent(in) :: nz
    integer(ip) :: k
    real(rp) :: ds, sc

    if (allocated(Cs_w)) deallocate(Cs_w)
    if (allocated(Cs_r)) deallocate(Cs_r)
    allocate(Cs_w(0:nz), Cs_r(1:nz))
    ds = 1.0_rp / real(nz, rp)
    Cs_w(nz) = 0.0_rp
    do k = nz - 1, 1, -1
      sc = ds * real(k - nz, rp)
      Cs_w(k) = csf(sc, theta_s, theta_b)
    enddo
    Cs_w(0) = -1.0_rp
    do k = 1, nz
      sc = ds * (real(k - nz, rp) - 0.5_rp)
      Cs_r(k) = csf(sc, theta_s, theta_b)
    enddo
  end subroutine set_scoord

  subroutine set_fast_weights()
    ! Primary + secondary weights from set_weights_mod.F90 (p=2, q=4).
    integer(ip) :: i, j, iter
    real(rp) :: p, q, r, scale, cff, sumw, shft

    if (allocated(weight)) deallocate(weight)
    allocate(weight(2, 2 * ndtfast))
    weight = 0.0_rp
    p = 2.0_rp
    q = 4.0_rp
    r = 0.25_rp
    scale = (p + 1.0_rp) * (p + q + 1.0_rp) / &
            ((p + 2.0_rp) * (p + q + 2.0_rp) * real(ndtfast, rp))

    do iter = 1, 16
      nfast = 0
      do i = 1, 2 * ndtfast
        cff = scale * real(i, rp)
        weight(1, i) = cff**p - cff**(p + q) - r * cff
        if (weight(1, i) > 0.0_rp) nfast = i
        if (nfast > 0 .and. weight(1, i) < 0.0_rp) weight(1, i) = 0.0_rp
      enddo
      sumw = 0.0_rp
      shft = 0.0_rp
      do i = 1, nfast
        sumw = sumw + weight(1, i)
        shft = shft + weight(1, i) * real(i, rp)
      enddo
      scale = scale * shft / (sumw * real(ndtfast, rp))
    enddo

    do iter = 1, ndtfast
      sumw = 0.0_rp
      shft = 0.0_rp
      do i = 1, nfast
        sumw = sumw + weight(1, i)
        shft = shft + real(i, rp) * weight(1, i)
      enddo
      shft = shft / sumw
      cff = real(ndtfast, rp) - shft
      if (cff > 1.0_rp) then
        nfast = nfast + 1
        do i = nfast, 2, -1
          weight(1, i) = weight(1, i - 1)
        enddo
        weight(1, 1) = 0.0_rp
      elseif (cff > 0.0_rp) then
        sumw = 1.0_rp - cff
        do i = nfast, 2, -1
          weight(1, i) = sumw * weight(1, i) + cff * weight(1, i - 1)
        enddo
        weight(1, 1) = sumw * weight(1, 1)
      elseif (cff < -1.0_rp) then
        nfast = nfast - 1
        do i = 1, nfast
          weight(1, i) = weight(1, i + 1)
        enddo
        weight(1, nfast + 1) = 0.0_rp
      elseif (cff < 0.0_rp) then
        sumw = 1.0_rp + cff
        do i = 1, nfast - 1
          weight(1, i) = sumw * weight(1, i) - cff * weight(1, i + 1)
        enddo
        weight(1, nfast) = sumw * weight(1, nfast)
      endif
    enddo

    weight(2, :) = 0.0_rp
    do j = 1, nfast
      cff = weight(1, j)
      do i = 1, j
        weight(2, i) = weight(2, i) + cff
      enddo
    enddo
    sumw = sum(weight(1, 1:nfast))
    cff  = sum(weight(2, 1:nfast))
    weight(1, 1:nfast) = weight(1, 1:nfast) / sumw
    weight(2, 1:nfast) = weight(2, 1:nfast) / cff
  end subroutine set_fast_weights

  subroutine advance_slow_indices()
    ! After a baroclinic step: nstp <-> nnew, nrhs = 3 during corrector.
    integer(ip) :: tmp
    tmp = nstp
    nstp = nnew
    nnew = tmp
    nrhs = nstp
  end subroutine advance_slow_indices

  subroutine cycle_fast_index()
    kstp = knew
    knew = kstp + 1
    if (knew > 4) knew = 1
  end subroutine cycle_fast_index

  subroutine begin_predictor_indices()
    ! main.F90: nstp=1+mod(iic-ntstart,2); nrhs=nstp; nnew=3
    nstp = 1 + mod(iic - ntstart, 2)
    nrhs = nstp
    nnew = 3
  end subroutine begin_predictor_indices

  subroutine begin_corrector_indices()
    ! main.F90: nrhs=3; nnew=3-nstp
    nrhs = 3
    nnew = 3 - nstp
  end subroutine begin_corrector_indices

  logical function is_first_time_step()
    is_first_time_step = (iic == forw_start)
  end function is_first_time_step

  logical function is_first_run_step()
    ! The first baroclinic step of this run (fresh start or restart). Some
    ! halo exchanges are only needed there: afterwards the halos they fill
    ! are already current (2026-10-08 exchange audit).
    is_first_run_step = (iic <= ntstart)
  end function is_first_run_step

  subroutine update_clock()
    time = start_time + dt * real(iic - ntstart, rp)
    tdays = time * sec2day
  end subroutine update_clock

  real(rp) function frc_modtime()
    ! days. slot 0 = n, 1 = n+1/2, 2 = n+1.
    ! slot 3 = n+3/2: CPU corrector does tdays=(time+0.5*dt)*sec2day
    ! then frc_time='forward' (adds another dt) before set_bry_all.
    if (frc_slot <= 0) then
      frc_modtime = tdays
    elseif (frc_slot == 1) then
      frc_modtime = tdays + 0.5_rp * dt * sec2day
    elseif (frc_slot == 2) then
      ! The CPU's expression for the corrector's boundary time (main.F90:603, then
      ! roms_read_write.F90:606): tdays shifted half a step, plus half a step. It differs
      ! from tdays + dt*sec2day by 1 ulp on about 2 steps in 3.
      frc_modtime = (time + 0.5_rp * dt) * sec2day + 0.5_rp * dt * sec2day
    else
      frc_modtime = tdays + 1.5_rp * dt * sec2day
    endif
  end function frc_modtime

end module time_state

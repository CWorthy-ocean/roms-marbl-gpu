#include "mem_attr.inc"
#include "cuf_launch.inc"
module bgc_subcycle_mod
  ! BGC tracer subcycling (UCLA ROMS bgc_subcycle_mod, after 0.14.0, #428
  ! and #430): keys bgc_physics_subcycle, bgc_physics_timestep of the
  ! namelist group BGC_SETTINGS (read in avg_out), and the testing keys
  ! bgc_subcycle_pos and bgc_subcycle_diag.
  !
  ! The tracers beyond T and S (tp: MARBL, generic passives, CDR_LITE; the
  ! CPU's itrc > iTandS) are to be transported, mixed and forced once per
  ! window of k = bgc_physics_timestep/dt steps, from the transports summed
  ! every step; MARBL runs at the window ends that are MARBL steps. T, S and
  ! the dynamics are unchanged.
  !
  ! *** STAGE-1 STUB: the window transport is not written yet. ***
  ! What exists (GPU_BGC_SUBCYCLE_SCOPE.md, stage 1b): the CPU's start-up
  ! checks and messages, the state copied to all time slots at start, the
  ! stock passive tracer step switched off (passive_mod tp_subcycled), the
  ! every-step sums of the corrector transports and diffusivity, and a
  ! window end that does everything but the transport: the window's
  ! continuity residual (bgc_subcycle_diag > 0), MARBL at MARBL steps, the
  ! constant-Fe hold, the exchange, the copy to the other time slots and
  ! the reset of the sums. So with the stub the BGC tracers are NOT
  ! transported, mixed or forced: they change only through MARBL. The stub
  ! runs only with the environment variable ROMS_GPU_BGCSUB_STUB=1 (tests
  ! and timing of the port); otherwise bgc_physics_subcycle = .true. stops
  ! the run at start-up, after the CPU's checks.
  !
  ! Between window ends the BGC state is held in time slots 1 and 2 of tp
  ! and at the surface of slot 3 (the only part of slot 3 read between
  ! windows: CFLX_CORR restoring and the CDR_LITE gas flux read slot nrhs at
  ! the surface); with bgc_subcycle_diag > 0 the rest of slot 3 is filled
  ! with a signalling NaN, as on the CPU.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state, only: Hz, FlxU, FlxV, We, Wi, Akt, pm, pn, rmask
  use time_state, only: iic, nnew, nstp, dt, ntstart, ntimes
  use passive_mod, only: ntp, tp, passive_hold, passive_init, tp_subcycled
  use halo_mod, only: halo_fill_3d_tp
  use marbl_gpu, only: marbl_on, marbl_timestep_ratio, marbl_gpu_step
  use his_io, only: wrt_file_rst, output_period_rst
  use upscale_out, only: do_upscale
  use pflx_mod, only: ub_tune
  use comm_mod, only: comm_rank, comm_allreduce_max
  use run_timers, only: rt_ev_begin, rt_ev_end, RT_BGCACC, RT_BGCWIN
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_signaling_nan
  implicit none
  private

  ! Namelist keys (group BGC_SETTINGS, read in avg_out)
  logical, public :: bgc_physics_subcycle = .false.
  real(rp), public :: bgc_physics_timestep = 0.0_rp   ! [s]; 0: marbl_timestep
  logical, public :: bgc_subcycle_pos = .false.       ! testing/expert key
  integer, public :: bgc_subcycle_diag = 0            ! testing/expert key
  integer(ip), public :: bgcsub_k = 1
  public :: bgcsub_init, bgcsub_accumulate, bgcsub_step

  character(len=*), parameter :: module_name = 'bgc_subcycle'
  integer(ip) :: nwin = 0           ! steps accumulated in the current window
  integer(ip) :: nwindows = 0
  integer :: nerr = 0               ! start-up errors raised (bgcsub_init)
  ! accumulated transports [m^3], vertical transport and diffusivity summed
  ! over the window; window-start thickness [m]
  real(rp), WORK_ATTR allocatable :: VX(:,:,:), VY(:,:,:), WA(:,:,:), KA(:,:,:), HZS(:,:,:)

contains

  ! ----------------------------------------------------------------------
  subroutine bgcsub_init()
    ! After the initial state and set_depth (load_case): the CPU's checks,
    ! allocation, window-start thickness, BGC state copied to all slots.
    character(len=512) :: info, val
    integer(ip) :: nrst, is, ie, js, je, nz, l, c, k, m, n, ns, ncol
    integer :: st
    logical :: stub_ok

    if (.not. bgc_physics_subcycle) return
    nerr = 0
    if (ntp < 1) call raise('bgcsub_init', &
         'bgc_physics_subcycle: there are no tracers beyond temperature/salinity')
    ! k = bgc_physics_timestep/dt (0: marbl_timestep as MARBL rounds it, i.e.
    ! marbl_timestep_ratio). MARBL keeps its own interval and runs at the end
    ! of a window, so every MARBL step must end one: k must divide
    ! marbl_timestep/dt.
    if (marbl_on) then
      if (bgc_physics_timestep == 0.0_rp) then
        bgcsub_k = marbl_timestep_ratio
      else
        call bgcsub_window_steps(int(marbl_timestep_ratio, ip))
      endif
    else
      if (bgc_physics_timestep == 0.0_rp) then
        call raise('bgcsub_init', 'bgc_physics_subcycle without MARBL needs bgc_physics_timestep > 0'// &
             ' (the window length in s, a whole multiple of dt = '//secs(dt)//' s)')
      else
        call bgcsub_window_steps(0_ip)
      endif
    endif
    ! Restarts only on window boundaries (the window sums are not in the
    ! file). Windows end on steps iic with modulo(iic, k) == 0 (every MARBL
    ! step among them); restart records are written every output_period_rst
    ! from the start of the run, so the run must also start on a window
    ! boundary. (monthly_restarts is not on the GPU: feature_guard stops it.)
    if (wrt_file_rst .and. bgcsub_k > 1) then
      nrst = nint(output_period_rst / dt, ip)
      if (abs(nrst * dt - output_period_rst) > 1.0e-6_rp * dt .or. &
          modulo(nrst, bgcsub_k) /= 0 .or. modulo(ntstart - 1, bgcsub_k) /= 0) then
        write(info,'(A,I0,A,I0,A,I0,A)') 'bgc_physics_subcycle: restarts must fall on window'// &
             ' boundaries: output_period_rst must be a multiple of bgc_physics_timestep = '// &
             secs(bgcsub_k * dt)//' s (', bgcsub_k, ' x dt), monthly_restarts off, and the run'// &
             ' must start on a window boundary (time step ', ntstart - 1, ' of the initial file'// &
             ' a multiple of ', bgcsub_k, ')'
        call raise('bgcsub_init', info)
      endif
    endif
    ! Not with subcycling (UCLA ROMS: UPSCALING output, SPONGE_TUNE ub_tune;
    ! GPU only: the salinity-mirror validation tracer, and what the GPU does
    ! not have yet)
    if (do_upscale) call raise('bgcsub_init', 'bgc_physics_subcycle does not support UPSCALING output')
    if (ub_tune) call raise('bgcsub_init', 'bgc_physics_subcycle does not support SPONGE_TUNE ub_tune')
    if (trim(passive_init) == 'mirror_salt') call raise('bgcsub_init', &
         "bgc_physics_subcycle does not support &passive_tracers init_mode = 'mirror_salt'")
    if (bgc_subcycle_pos) call raise('bgcsub_init', &
         'bgc_subcycle_pos (the positivity limiter) is not on the GPU yet')
    if (bgc_subcycle_diag >= 2) call raise('bgcsub_init', &
         'bgc_subcycle_diag >= 2 (the positivity test) is not on the GPU yet')
    if (pp /= rp) call raise('bgcsub_init', &
         'bgc_physics_subcycle needs double-precision passive tracers (a build with -DPASSIVE_FP64)')
    if (nerr > 0) error stop 'bgc_subcycle: bgc_physics_subcycle settings rejected (see above)'

    ! The window transport is not written yet: run the stub only on request.
    call get_environment_variable('ROMS_GPU_BGCSUB_STUB', val, status=st)
    stub_ok = (st == 0 .and. trim(val) == '1')
    if (.not. stub_ok) error stop 'BGC_SETTINGS: bgc_physics_subcycle = .true.: BGC tracer '// &
         'subcycling is not on the GPU yet; set bgc_physics_subcycle = .false. to run on the GPU'

    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    allocate(VX(is:ie, js:je, nz), VY(is:ie, js:je, nz), HZS(is:ie, js:je, nz))
    allocate(WA(is:ie, js:je, 0:nz), KA(is:ie, js:je, 0:nz))
    call reset_sums()
    nwin = 0

    ! the BGC state into every slot (all stored columns, halos included)
    ns = nstp
    n = ntp
    ncol = ntpcol_n()
    do l = 1, 3
      if (l == ns) cycle
      CUF_DO3
      do k = 1, nz
        do c = 0, ncol
          do m = 1, n
            tp(m, c, k, l) = tp(m, c, k, ns)
          enddo
        enddo
      enddo
    enddo
    tp_subcycled = .true.

    if (comm_rank() == 0) then
      write(*,'(/1x,A,I0,A,I0,A)') 'BGC subcycling: tracers ', 3, '-', 2 + ntp, &
           ', bgc_physics_timestep = '//secs(bgcsub_k * dt)//' s'
      write(*,'(1x,A,I0,A,L1,A,I0)') '  k = ', bgcsub_k, ', positivity limiter ', &
           bgc_subcycle_pos, ', diagnostics ', bgc_subcycle_diag
      if (marbl_on) write(*,'(1x,A,I0,A)') '  MARBL at the end of every ', &
           marbl_timestep_ratio / bgcsub_k, ' window(s)'
      if (modulo(ntstart - 1, bgcsub_k) /= 0) write(*,'(1x,A)') &
           '  the run starts inside a window: the first window is shorter'
      if (modulo(ntstart + ntimes - 1, bgcsub_k) /= 0) write(*,'(1x,A,I0,A)') &
           '  the run ends inside a window: the BGC transport of its last ', &
           modulo(ntstart + ntimes - 1, bgcsub_k), ' steps is not applied'
      write(*,'(1x,A)') '  *** GPU STAGE-1 STUB (ROMS_GPU_BGCSUB_STUB=1): NO BGC transport, mixing or'// &
           ' sources; the BGC tracers are held frozen and change only through MARBL. Testing only. ***'
    endif
  end subroutine bgcsub_init

  integer(ip) function ntpcol_n()
    ! last stored column of tp (tp(:, 0:ntpcol, :, :))
    ntpcol_n = int(ubound(tp, 2), ip)
  end function ntpcol_n

  ! ----------------------------------------------------------------------
  subroutine raise(where, info)
    ! the CPU's error_log%raise_global, printed at once by compute rank 0
    character(len=*), intent(in) :: where, info
    nerr = nerr + 1
    if (comm_rank() == 0) then
      write(*,*) 'ERROR [global] [', module_name//'/'//where, ']:'
      write(*,*) trim(info)
      flush(6)
    endif
  end subroutine raise

  ! ----------------------------------------------------------------------
  subroutine bgcsub_window_steps(nm)
    ! bgcsub_k = bgc_physics_timestep/dt. It must be a whole multiple of dt
    ! (relative tolerance as for output_period_rst), at least dt, and with
    ! MARBL (nm = marbl_timestep/dt > 0) divide marbl_timestep. Otherwise
    ! stop, naming the nearest valid values: never round silently.
    integer(ip), intent(in) :: nm
    character(len=512) :: info
    real(rp) :: x
    integer(ip) :: kl, kh

    x = min(bgc_physics_timestep / dt, 1.0e8_rp)
    bgcsub_k = max(1_ip, nint(x, ip))
    if (x < 1.0_rp - 1.0e-6_rp) then
      info = 'bgc_physics_timestep = '//secs(bgc_physics_timestep)// &
           ' s is shorter than dt = '//secs(dt)//' s'
    else if (abs(bgcsub_k * dt - bgc_physics_timestep) > 1.0e-6_rp * dt) then
      info = 'bgc_physics_timestep = '//secs(bgc_physics_timestep)// &
           ' s is not a whole multiple of dt = '//secs(dt)//' s'
    else if (nm > 0 .and. modulo(nm, bgcsub_k) /= 0) then
      info = 'marbl_timestep = '//secs(nm * dt)//' s (as MARBL rounds it) is not'// &
           ' a whole multiple of bgc_physics_timestep = '//secs(bgc_physics_timestep)//' s'
    else
      return
    endif
    ! nearest valid window lengths below and above (with MARBL: divisors of nm)
    kl = floor(x + 1.0e-6_rp, ip)
    if (nm > 0) kl = min(kl, nm)
    do while (kl >= 1)
      if (nm == 0) exit
      if (modulo(nm, kl) == 0) exit
      kl = kl - 1
    enddo
    kh = max(1_ip, ceiling(x - 1.0e-6_rp, ip))
    do while (nm > 0 .and. kh <= nm)
      if (modulo(nm, kh) == 0) exit
      kh = kh + 1
    enddo
    info = trim(info)//'; nearest valid bgc_physics_timestep:'
    if (kl >= 1) info = trim(info)//' '//secs(kl * dt)//' s'
    if (kl >= 1 .and. (nm == 0 .or. kh <= nm)) info = trim(info)//' or'
    if (nm == 0 .or. kh <= nm) info = trim(info)//' '//secs(kh * dt)//' s'
    if (nm > 0) info = trim(info)//' (0 = marbl_timestep)'
    call raise('bgcsub_window_steps', info)
  end subroutine bgcsub_window_steps

  ! ----------------------------------------------------------------------
  function secs(v) result(str)
    ! A time in s for messages: an integer when it is one.
    real(rp), intent(in) :: v
    character(len=:), allocatable :: str
    character(len=32) :: buf
    if (abs(v) < 1.0e9_rp .and. abs(v - anint(v)) <= 1.0e-9_rp * max(1.0_rp, abs(v))) then
      write(buf,'(I0)') nint(v, kind=8)
    else
      write(buf,'(G0.6)') v
    endif
    str = trim(adjustl(buf))
  end function secs

  ! ----------------------------------------------------------------------
  subroutine bgcsub_accumulate()
    ! Every step, after step3d_uv2 + omega (the corrector transports that
    ! step3d_t uses): sum dt*FlxU, dt*FlxV, dt*(We+Wi) and dt*Akt(salt) over
    ! the tile with its halo (FlxU/FlxV exchanged by step3d_uv2, We/Wi set
    ! by omega there). One fused kernel. The source sums (surface fluxes,
    ! rivers, CDR) come with the window transport (stage 4).
    integer(ip) :: i, j, k, is, ie, js, je, nz
    real(rp) :: dtl
    if (.not. bgc_physics_subcycle) return
    call rt_ev_begin(RT_BGCACC)
    nwin = nwin + 1
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    dtl = dt
    CUF_DO3
    do k = 0, nz
      do j = js, je
        do i = is, ie
          if (k >= 1) then
            VX(i, j, k) = VX(i, j, k) + dtl * FlxU(i, j, k)
            VY(i, j, k) = VY(i, j, k) + dtl * FlxV(i, j, k)
          endif
          WA(i, j, k) = WA(i, j, k) + dtl * (We(i, j, k) + Wi(i, j, k))
          KA(i, j, k) = KA(i, j, k) + dtl * Akt(i, j, k, 2)
        enddo
      enddo
    enddo
    call rt_ev_end(RT_BGCACC)
  end subroutine bgcsub_accumulate

  ! ----------------------------------------------------------------------
  subroutine bgcsub_step()
    ! After step3d_t, in place of marbl_gpu_step. At a window end (STUB):
    ! continuity residual (diagnostics), MARBL at MARBL steps, the Fe hold,
    ! the exchange of slot nnew, the copy to the other slots, the reset.
    integer(ip) :: nn, l, c, k, m, n, nz, ncol, nwin_w
    real(rp) :: cres
    real(pp) :: snan
    real(8) :: lb(1), gb(1)
    if (modulo(iic, bgcsub_k) /= 0) return
    call rt_ev_begin(RT_BGCWIN)
    nwindows = nwindows + 1
    nwin_w = nwin
    nn = nnew; nz = decomp%nz; n = ntp; ncol = ntpcol_n()
    ! window continuity (the CPU's cont): Hz_end = v1 - d_z W, k < nz
    if (bgc_subcycle_diag > 0) then
      cres = window_cont()
      lb(1) = real(cres, 8)
      call comm_allreduce_max(1, lb, gb)
      cres = real(gb(1), rp)
    endif
    ! STUB: the window transport, mixing, sources and boundary conditions
    ! go here (stages 2-4).
    call marbl_gpu_step()               ! returns unless iic is a MARBL step
    call passive_hold(nn)               ! use_constant_Fe (no-op otherwise)
    call halo_fill_3d_tp(tp, ntp, nn)
    ! Copy to the other slots: nnew is 1 or 2 here, and until the next
    ! window end only the other of the two is read in full; slot 3 is read
    ! only at the surface (see the module header).
    l = 3 - nn
    CUF_DO3
    do k = 1, nz
      do c = 0, ncol
        do m = 1, n
          tp(m, c, k, l) = tp(m, c, k, nn)
          if (k == nz) tp(m, c, k, 3) = tp(m, c, k, nn)
        enddo
      enddo
    enddo
    ! Test runs (bgc_subcycle_diag > 0): the rest of slot 3 is a signalling
    ! NaN, so that a reader of it between windows fails loudly.
    if (bgc_subcycle_diag > 0) then
      snan = ieee_value(1.0_pp, ieee_signaling_nan)
      CUF_DO3
      do k = 1, nz - 1
        do c = 0, ncol
          do m = 1, n
            tp(m, c, k, 3) = snan
          enddo
        enddo
      enddo
    endif
    call reset_sums()
    nwin = 0
    call rt_ev_end(RT_BGCWIN)
    if (bgc_subcycle_diag > 0 .and. comm_rank() == 0) then
      write(*,'(1x,A,I7,I3,A,ES9.2,A)') 'BGCSUB-STUB', iic, nwin_w, ' cont ', cres, &
           '  (stage-1 stub: no BGC transport, mixing or sources)'
    endif
  end subroutine bgcsub_step

  ! ----------------------------------------------------------------------
  real(rp) function window_cont() result(cres)
    ! max over interior wet cells, k = 1..nz-1, of |Hz - (v1 - d_z W)|/Hz,
    ! v1 = HZS - pm pn (dVX/dx + dVY/dy), W = WA pm pn with W = 0 at the
    ! bottom (the CPU's cres, same operand order). A device reduction.
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: cm, v1, wk, wkm
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    cm = 0.0_rp
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz - 1
      do j = 1, ny
        do i = 1, nx
          v1 = HZS(i, j, k) - pm(i, j) * pn(i, j) * (VX(i + 1, j, k) - VX(i, j, k) + VY(i, j + 1, k) - VY(i, j, k))
          wk = WA(i, j, k) * pm(i, j) * pn(i, j)
          wkm = 0.0_rp
          if (k > 1) wkm = WA(i, j, k - 1) * pm(i, j) * pn(i, j)
          cm = max(cm, abs(Hz(i, j, k) - (v1 - (wk - wkm))) / Hz(i, j, k) * rmask(i, j))
        enddo
      enddo
    enddo
    cres = cm
  end function window_cont

  ! ----------------------------------------------------------------------
  subroutine reset_sums()
    ! sums to 0, window-start thickness = Hz
    integer(ip) :: i, j, k, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    CUF_DO3
    do k = 0, nz
      do j = js, je
        do i = is, ie
          if (k >= 1) then
            VX(i, j, k) = 0.0_rp; VY(i, j, k) = 0.0_rp
            HZS(i, j, k) = Hz(i, j, k)
          endif
          WA(i, j, k) = 0.0_rp; KA(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine reset_sums

end module bgc_subcycle_mod

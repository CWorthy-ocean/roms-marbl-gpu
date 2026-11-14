module run_timers
  ! Run-level wall-clock regions for the CPU/GPU comparison, summarized at
  ! the end as max and mean over the compute ranks (the layout of the CPU
  ! model's print_region_timers). Host regions use the wall clock; MARBL is
  ! timed on the device with two events on work_stream (no added
  ! synchronization): the interval runs from the end of the work queued
  ! before it to the end of its last kernel, and is read back after the
  ! end-of-step device sync.
  ! Wall time comes from wall_time() (clock_gettime, CLOCK_MONOTONIC), not
  ! system_clock: nvfortran's default-integer system_clock wraps every
  ! 2**31 microseconds (2147 s) and, when it does, re-bases the integer(8)
  ! clock as well, so integer(8) intervals spanning a wrap came out short
  ! in a run that also made default-integer calls (a ~5950 s stepping loop
  ! reported 1660 s, one step -2147 s; a test with only integer(8) calls
  ! measured 2200 s correctly, and with default-integer calls every 20 s
  ! the same interval read 20 s).
  use iso_c_binding, only: c_int, c_long
  use kinds
  use comm_mod, only: comm_rank, comm_size, comm_reduce_maxsum
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  integer, parameter, public :: RT_LOOP = 1, RT_OUTPUT = 2, RT_HIS = 3, RT_RST = 4, &
                                RT_MARBL = 5, RT_FINAL = 6, RT_INIT = 7, RT_BGCACC = 8, RT_BGCWIN = 9, &
                                RT_BGCH0 = 10, RT_BGCH = 11, RT_BGCV = 12, RT_BGCBC = 13, RT_BGCHS = 14
  integer, parameter :: NRT = 14
  character(len=34), parameter :: rt_name(NRT) = [character(len=34) :: &
       'stepping loop', 'output (maybe_wrt_output)', '  history staging (nested)', &
       '  restart staging (nested)', 'MARBL (device events)', 'end of run (io_finalize)', &
       'init (start-up to first step)', 'BGC subcycle sums (device events)', &
       'BGC window, incl. MARBL (dev. ev.)', '  window Courant numbers (nested)', &
       '  window horizontal pass (nested)', '  window vertical pass (nested)', &
       '  window bc+MARBL+exchange (nested)', '    window sub-passes 2+ (nested)']
  real(8) :: rt_t(NRT) = 0.0d0
  real(8) :: rt_c0(NRT) = 0.0d0
  integer :: rt_n(NRT) = 0
  integer, public :: rt_nsteps = 0
  ! Wall time of every step (baroclinic step + output) on this rank and
  ! the steps that staged a restart, for the TIMING lines (the same lines
  ! as the CPU model's timers module).
  real(8), allocatable :: step_t(:)
  real(8) :: step_c0 = 0.0d0
  integer, parameter :: MAX_RST_STEPS = 16
  integer :: rst_steps(MAX_RST_STEPS) = 0, n_rst_steps = 0
#ifdef USE_CUDA
  type(cudaEvent) :: ev_a, ev_b
  logical :: ev_made = .false., ev_pending = .false.
  ! BGC subcycling (bgc_subcycle_mod): the window sums and the window end,
  ! timed like MARBL with their own event pairs (rt_ev_begin/rt_ev_end)
  type(cudaEvent) :: evb_a(RT_BGCACC:RT_BGCHS), evb_b(RT_BGCACC:RT_BGCHS)
  logical :: evb_made = .false., evb_pending(RT_BGCACC:RT_BGCHS) = .false.
#endif
  public :: rt_tic, rt_toc, rt_dev_begin, rt_dev_end, rt_dev_collect, rt_report
  public :: rt_ev_begin, rt_ev_end
  public :: rt_steps_init, rt_step_begin, rt_step_end, rt_mark_rst
  public :: wall_time

  type, bind(C) :: c_timespec
    integer(c_long) :: tv_sec, tv_nsec
  end type c_timespec
  integer(c_int), parameter :: CLOCK_MONOTONIC = 1     ! Linux <time.h>
  interface
    integer(c_int) function c_clock_gettime(clk, ts) bind(C, name='clock_gettime')
      import :: c_int, c_timespec
      integer(c_int), value :: clk
      type(c_timespec), intent(out) :: ts
    end function c_clock_gettime
  end interface
contains
  real(8) function wall_time()
    ! Seconds on the monotonic clock (arbitrary origin; differences only).
    type(c_timespec) :: ts
    integer(c_int) :: ierr
    ierr = c_clock_gettime(CLOCK_MONOTONIC, ts)
    wall_time = real(ts%tv_sec, 8) + 1.0d-9 * real(ts%tv_nsec, 8)
  end function wall_time

  subroutine rt_tic(ir)
    integer, intent(in) :: ir
    rt_c0(ir) = wall_time()
  end subroutine rt_tic

  subroutine rt_toc(ir)
    integer, intent(in) :: ir
    rt_t(ir) = rt_t(ir) + (wall_time() - rt_c0(ir))
    rt_n(ir) = rt_n(ir) + 1
  end subroutine rt_toc

  subroutine rt_steps_init(n)
    integer, intent(in) :: n
    if (allocated(step_t)) deallocate(step_t)
    allocate(step_t(max(n, 1)))
    step_t = 0.0d0
  end subroutine rt_steps_init

  subroutine rt_step_begin()
    step_c0 = wall_time()
  end subroutine rt_step_begin

  subroutine rt_step_end()
    ! after rt_nsteps has been advanced for this step
    real(8) :: now
    now = wall_time()
    if (allocated(step_t)) then
      if (rt_nsteps >= 1 .and. rt_nsteps <= size(step_t)) step_t(rt_nsteps) = now - step_c0
    endif
  end subroutine rt_step_end

  subroutine rt_mark_rst()
    ! called while step rt_nsteps + 1 is being taken
    if (n_rst_steps < MAX_RST_STEPS) then
      n_rst_steps = n_rst_steps + 1
      rst_steps(n_rst_steps) = rt_nsteps + 1
    endif
  end subroutine rt_mark_rst

  subroutine rt_dev_begin()
#ifdef USE_CUDA
    integer :: ierr
    if (.not. ev_made) then
      ierr = cudaEventCreate(ev_a)
      ierr = cudaEventCreate(ev_b)
      ev_made = .true.
    endif
    ierr = cudaEventRecord(ev_a, work_stream)
#endif
  end subroutine rt_dev_begin

  subroutine rt_dev_end()
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaEventRecord(ev_b, work_stream)
    ev_pending = .true.
#endif
  end subroutine rt_dev_end

  subroutine rt_ev_begin(ir)
    ! Device interval of region ir (RT_BGCACC or RT_BGCWIN), as rt_dev_begin
    integer, intent(in) :: ir
#ifdef USE_CUDA
    integer :: ierr, k
    if (.not. evb_made) then
      do k = RT_BGCACC, RT_BGCHS
        ierr = cudaEventCreate(evb_a(k))
        ierr = cudaEventCreate(evb_b(k))
      enddo
      evb_made = .true.
    endif
    ierr = cudaEventRecord(evb_a(ir), work_stream)
#else
    call rt_tic(ir)
#endif
  end subroutine rt_ev_begin

  subroutine rt_ev_end(ir)
    integer, intent(in) :: ir
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaEventRecord(evb_b(ir), work_stream)
    evb_pending(ir) = .true.
#else
    call rt_toc(ir)
#endif
  end subroutine rt_ev_end

  subroutine rt_dev_collect()
    ! After a device sync: add the pending MARBL interval (and the BGC
    ! subcycling intervals, waiting for their end events).
#ifdef USE_CUDA
    integer :: ierr, k
    real :: ms
    do k = RT_BGCACC, RT_BGCHS
      if (.not. evb_pending(k)) cycle
      ierr = cudaEventSynchronize(evb_b(k))
      ierr = cudaEventElapsedTime(ms, evb_a(k), evb_b(k))
      if (ierr == 0) then
        rt_t(k) = rt_t(k) + real(ms, 8) * 1.0d-3
        rt_n(k) = rt_n(k) + 1
      endif
      evb_pending(k) = .false.
    enddo
    if (.not. ev_pending) return
    ierr = cudaEventElapsedTime(ms, ev_a, ev_b)
    if (ierr == 0) then
      rt_t(RT_MARBL) = rt_t(RT_MARBL) + real(ms, 8) * 1.0d-3
      rt_n(RT_MARBL) = rt_n(RT_MARBL) + 1
    endif
    ev_pending = .false.
#endif
  end subroutine rt_dev_collect

  subroutine rt_report()
    real(8) :: tmax(NRT), tsum(NRT), cnt(NRT), cmax(NRT), csum(NRT), np
    integer :: ir, ns
    call rt_dev_collect()
    cnt = real(rt_n, 8)
    call comm_reduce_maxsum(NRT, rt_t, tmax, tsum)
    call comm_reduce_maxsum(NRT, cnt, cmax, csum)
    if (comm_rank() /= 0) return
    np = real(max(comm_size(), 1), 8)
    ns = max(rt_nsteps, 1)
    write(*,'(/1x,a)') 'Run timers (wall seconds over the run, compute ranks):'
    write(*,'(1x,a34,2x,a10,2x,a10,2x,a7,2x,a12)') 'region', 'max rank', 'mean rank', 'calls', 'max per call'
    do ir = 1, NRT
      if (ir >= RT_BGCACC .and. cmax(ir) <= 0.0d0) cycle   ! BGC subcycling rows only when used
      write(*,'(1x,a34,2x,f10.3,2x,f10.3,2x,i7,2x,f12.5)') rt_name(ir), tmax(ir), tsum(ir) / np, &
           nint(cmax(ir)), tmax(ir) / max(cmax(ir), 1.0d0)
    enddo
    write(*,'(1x,a,f12.3,a,i8,a,f10.5,a)') 'stepping loop (max rank):', tmax(RT_LOOP), ' s over', &
         rt_nsteps, ' steps =', tmax(RT_LOOP) / real(ns, 8), ' s/step'
    write(*,'(1x,a,f10.5,a,f10.5,a/)') 'MARBL per step (max rank):', tmax(RT_MARBL) / real(ns, 8), &
         ' s   restart staging per write (max rank):', tmax(RT_RST) / max(cmax(RT_RST), 1.0d0), ' s'
    write(*,'(1x,a,f12.3)') 'TIMING init_s', tmax(RT_INIT)
    write(*,'(1x,a,i8,a,f12.6,a,f12.6)') 'TIMING marbl_calls', nint(cmax(RT_MARBL)), &
         ' marbl_per_call_s', tmax(RT_MARBL) / max(cmax(RT_MARBL), 1.0d0), &
         ' marbl_per_step_s', tmax(RT_MARBL) / real(ns, 8)
    write(*,'(1x,a,i8,a,f12.6)') 'TIMING restart_writes', nint(cmax(RT_RST)), &
         ' restart_write_s', tmax(RT_RST) / max(cmax(RT_RST), 1.0d0)
    if (cmax(RT_BGCACC) > 0.0d0) then
      write(*,'(1x,a,i8,a,f12.6,a,i8,a,f12.6,a,f12.6)') 'TIMING bgcsub_sum_calls', nint(cmax(RT_BGCACC)), &
           ' sum_per_call_s', tmax(RT_BGCACC) / max(cmax(RT_BGCACC), 1.0d0), &
           ' window_calls', nint(cmax(RT_BGCWIN)), &
           ' window_per_call_s', tmax(RT_BGCWIN) / max(cmax(RT_BGCWIN), 1.0d0), &
           ' window_per_step_s', tmax(RT_BGCWIN) / real(ns, 8)
      write(*,'(1x,a,4(a,f12.6))') 'TIMING bgcsub_window_parts_per_call_s', &
           ' courant', tmax(RT_BGCH0) / max(cmax(RT_BGCH0), 1.0d0), &
           ' horizontal', tmax(RT_BGCH) / max(cmax(RT_BGCH), 1.0d0), &
           ' vertical', tmax(RT_BGCV) / max(cmax(RT_BGCV), 1.0d0), &
           ' bc_marbl_exchange', tmax(RT_BGCBC) / max(cmax(RT_BGCBC), 1.0d0)
      if (cmax(RT_BGCHS) > 0.0d0) write(*,'(1x,a,i8,a,f12.6)') 'TIMING bgcsub_subpass_windows', &
           nint(cmax(RT_BGCHS)), ' subpasses_2plus_per_window_s', tmax(RT_BGCHS) / max(cmax(RT_BGCHS), 1.0d0)
    endif
    if (allocated(step_t)) call print_step_stats(step_t, rt_nsteps, rst_steps, n_rst_steps)
    flush(6)
  end subroutine rt_report

  subroutine print_step_stats(st, n, rsteps, nrs)
    ! Steps 1-10 are the warm-up; the steady average is over the remaining
    ! steps that staged no restart; each restart step is reported with its
    ! excess over the steady median (as the CPU model's print_step_stats).
    real(8), intent(in) :: st(:)
    integer, intent(in) :: n, rsteps(:), nrs
    integer, parameter :: NWARM = 10
    real(8), allocatable :: v(:)
    real(8) :: x, med
    integer :: i, j, m, nw
    logical :: is_rst
    nw = min(NWARM, n)
    allocate(v(max(n, 1)))
    v = 0.0d0
    m = 0
    do i = nw + 1, n
      is_rst = .false.
      do j = 1, nrs
        if (rsteps(j) == i) is_rst = .true.
      enddo
      if (.not. is_rst) then
        m = m + 1; v(m) = st(i)
      endif
    enddo
    do i = 2, m
      x = v(i); j = i - 1
      do while (j >= 1)
        if (v(j) <= x) exit
        v(j + 1) = v(j); j = j - 1
      enddo
      v(j + 1) = x
    enddo
    med = 0.0d0
    if (m > 0) med = 0.5d0 * (v((m + 1) / 2) + v(m / 2 + 1))
    write(*,'(1x,a,i8,a,f12.6,a,i8,a,f12.6,a,f12.6,a,f12.6,a,f12.6)') 'TIMING steps', n, &
         ' warm_mean_s', sum(st(1:nw)) / real(max(nw, 1), 8), ' nsteady', m, &
         ' steady_mean_s', sum(v(1:m)) / real(max(m, 1), 8), ' steady_median_s', med, &
         ' steady_min_s', v(1), ' steady_max_s', v(max(m, 1))
    do j = 1, nrs
      if (rsteps(j) >= 1 .and. rsteps(j) <= n) then
        write(*,'(1x,a,i8,a,f12.6,a,f12.6)') 'TIMING restart_step', rsteps(j), &
             ' step_s', st(rsteps(j)), ' excess_s', st(rsteps(j)) - med
      endif
    enddo
    do i = 1, n, 10
      write(*,'(1x,a,i6,10f10.5)') 'TIMING step_times', i, st(i:min(i + 9, n))
    enddo
    deallocate(v)
  end subroutine print_step_stats
end module run_timers

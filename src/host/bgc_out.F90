#include "mem_attr.inc"
module bgc_out
  ! The BGC output files of the CPU's bgc_io (&bgc_settings), in double:
  !  - <root>_bgc: the MARBL tracers (wrt_bgc_his), from the start-up state;
  !  - <root>_bgc_avg: their averages (wrt_bgc_avg);
  !  - <root>_bgc_dia: MARBL diagnostics (wrt_bgc_dia_his), from the start-up
  !    state's (marbl_gpu_init_diagnostics);
  !  - <root>_bgc_dia_avg: their averages (wrt_bgc_dia_avg).
  ! Each has a record every output_period_bgc_* and nrpf_bgc_* records per
  ! file; a file is named after the time of its first record. A record is
  ! stamped with the time at the write; the averages also carry the bounds
  ! of their window, avg_begin_time and avg_end_time. The start-up state is
  ! in no average.
  !
  ! Tracers: those marbl_tracers_to_write selects (avg_out select_tracers);
  ! at slot nnew (time n+1), averaged with a running mean.
  !
  ! Diagnostics: those marbl_diagnostics_to_write names, or all of them if
  ! it names none that MARBL knows, among the ones the GPU computes
  ! (marbl_gpu dia_name). A name that MARBL has and the GPU does not stops
  ! the run. They change only on MARBL steps, so they are added to the
  ! averaging sums right after each, weighted by the steps since the
  ! previous addition (the CPU's accumulate_dia_bgc);
  ! output_period_bgc_avg_dia must be a multiple of the MARBL timestep.
  use kinds
  use his_io, only: output_fname, refdatestr, def_avg_times
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use implicit_fs_gpu, only: fs_global_atts
  use sponge_io, only: sponge_global_atts
  use io_server, only: IO_KIND_BGC, IO_KIND_BGCAVG, IO_KIND_BGCDIA, IO_KIND_BGCDIAAVG, io_init, io_init8, &
                       io_stage8_begin, io_stage8_2d, io_stage8_3d, io_stage8_3d_pt, io_stage8_end
  use comm_mod, only: comm_rank, compute_barrier
  use passive_mod, only: ntp, n_marbl, tp, passive_name
  use marbl_gpu, only: marbl_on, marbl_timestep_ratio, marbl_need_zsat, xco2_default, marbl_dia_list, &
                       NDIA2, NDIA3, dia2_name, dia3_name, dia2_slot, DIA2_ZSATC, DIA2_ZSATA
  use marbl_host, only: marbl_tracer_units, marbl_tracer_lnames, marbl_diag_info
  use marbl_sflux_mod, only: sf_diag
  use marbl_interior_mod, only: it_co3, it_co3a, it_co3sat, it_co3sata, it_zsatc, it_zsata
  use frc_io, only: use_xco2air_forcing, use_nox_forcing, use_nhy_forcing
  use avg_out, only: select_tracers
  ! BGC tracer subcycling keys of &bgc_settings (UCLA ROMS after 0.14.0, #428),
  ! owned by bgc_subcycle_mod and read here with the rest of the group
  use bgc_subcycle_mod, only: bgc_physics_subcycle, bgc_physics_timestep, bgc_subcycle_pos, bgc_subcycle_diag
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: read_bgc_namelist, set_bgc_root, init_bgc_output, wrt_bgc

  ! &bgc_settings: the CPU's keys and defaults
  logical  :: wrt_bgc_his = .false., wrt_bgc_avg = .false., wrt_bgc_dia_his = .false., wrt_bgc_dia_avg = .false.
  real(rp) :: output_period_bgc_his = 0.0_rp, output_period_bgc_avg = 0.0_rp
  real(rp) :: output_period_bgc_his_dia = 0.0_rp, output_period_bgc_avg_dia = 0.0_rp
  integer  :: nrpf_bgc_his = 0, nrpf_bgc_avg = 0, nrpf_bgc_his_dia = 0, nrpf_bgc_avg_dia = 0

  type stream_t
    logical :: on = .false., avg = .false., dia = .false.
    integer :: kind = 0, nrpf = 0, rec = 0, nfile = 0
    integer :: nsum = 0                       ! steps in the current average
    real(rp) :: period = 0.0_rp, otime = 0.0_rp, begin_time = 0.0_rp
    character(len=16) :: name = ''
    character(len=512) :: path = ''
  end type stream_t
  type(stream_t) :: his, avg, dhis, davg

  character(len=512) :: root = './output'
  integer :: ir0, ir1, jr0, jr1, gxi, geta, gxi_u, geta_v
  ! tracers written: passive indices bsrc(1:nba); averages ba(m, column, k)
  integer :: nba = 0
  integer(ip), allocatable :: bsrc(:)
  integer(ip), WORK_ATTR allocatable :: bsrc_d(:)
  real(pp), WORK_ATTR allocatable :: ba(:,:,:)
  ! diagnostics written: d2(1:nd2) of dia2_name, d3(1:nd3) of dia3_name, with
  ! MARBL's long names and units; sums a2, a3
  integer :: nd2 = 0, nd3 = 0, d2(NDIA2) = 0, d3(NDIA3) = 0
  character(len=256) :: d2_lname(NDIA2) = '', d3_lname(NDIA3) = ''
  character(len=64)  :: d2_units(NDIA2) = '', d3_units(NDIA3) = ''
  integer :: nheld = 0
  logical :: dia_first = .true.
  real(rp), WORK_ATTR allocatable :: a2(:,:,:), a3(:,:,:,:)

contains

  subroutine read_bgc_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    logical :: interp_bgc_frc
    real(rp) :: xco2air_default
    character(len=512) :: msg
    namelist /bgc_settings/ wrt_bgc_his, wrt_bgc_avg, wrt_bgc_dia_his, wrt_bgc_dia_avg, interp_bgc_frc, &
         output_period_bgc_his, nrpf_bgc_his, output_period_bgc_avg, nrpf_bgc_avg, &
         output_period_bgc_his_dia, nrpf_bgc_his_dia, output_period_bgc_avg_dia, nrpf_bgc_avg_dia, &
         use_xco2air_forcing, use_nox_forcing, use_nhy_forcing, xco2air_default, &
         bgc_physics_subcycle, bgc_physics_timestep, bgc_subcycle_pos, bgc_subcycle_diag
    xco2air_default = 284.7_rp     ! ppm, the CPU's default
    rewind(unit)
    read(unit, nml=bgc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'BGC_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_bgc_namelist'
    endif
    ! MARBL's atmospheric xCO2, for both carbonate systems, with
    ! use_xco2air_forcing = .false. (marbl_gpu)
    xco2_default = xco2air_default
  end subroutine read_bgc_namelist

  subroutine set_bgc_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_bgc_root

  subroutine check_stream(on, switch, period, period_name, nrpf, nrpf_name)
    ! The CPU's bgc_precheck, and its checks at each stream's write.
    logical, intent(in) :: on
    character(len=*), intent(in) :: switch, period_name, nrpf_name
    real(rp), intent(in) :: period
    integer, intent(in) :: nrpf
    if (.not. on) return
    if (.not. (period > 0.0_rp .and. nrpf > 0)) then
      write(*,'(7a)') 'BGC_SETTINGS: ', switch, ' = .true. needs ', period_name, ' and ', nrpf_name, ' above 0'
      error stop 'init_bgc_output: output stream without a period or records per file'
    endif
    if (abs(period / dt - anint(period / dt)) > 1.0e-9_rp) then
      write(*,'(2a)') 'BGC_SETTINGS: dt is not a factor of ', period_name
      error stop 'init_bgc_output: output period'
    endif
  end subroutine check_stream

  subroutine init_bgc_output()
    ! After marbl_gpu_init_diagnostics and init_output; writes the start-up
    ! records.
    integer :: nxy, nz, is, ie, js, je, m, cap
    real(rp) :: r
    logical :: sel_ts(2), sel_cdr(max(int(ntp - n_marbl), 1)), sel_marbl(max(int(n_marbl), 1))
    if (.not. marbl_on) return      ! as the CPU, which then does not use the group
    call check_stream(wrt_bgc_his, 'wrt_bgc_his', output_period_bgc_his, 'output_period_bgc_his', &
                      nrpf_bgc_his, 'nrpf_bgc_his')
    call check_stream(wrt_bgc_avg, 'wrt_bgc_avg', output_period_bgc_avg, 'output_period_bgc_avg', &
                      nrpf_bgc_avg, 'nrpf_bgc_avg')
    call check_stream(wrt_bgc_dia_his, 'wrt_bgc_dia_his', output_period_bgc_his_dia, 'output_period_bgc_his_dia', &
                      nrpf_bgc_his_dia, 'nrpf_bgc_his_dia')
    call check_stream(wrt_bgc_dia_avg, 'wrt_bgc_dia_avg', output_period_bgc_avg_dia, 'output_period_bgc_avg_dia', &
                      nrpf_bgc_avg_dia, 'nrpf_bgc_avg_dia')
    if (wrt_bgc_dia_avg) then
      ! the CPU's check_avg_ends_on_marbl_step
      r = output_period_bgc_avg_dia / (dt * real(marbl_timestep_ratio, rp))
      if (abs(r - anint(r)) > 1.0e-9_rp) then
        write(*,'(a,es14.6,a,es14.6)') 'BGC_SETTINGS: output_period_bgc_avg_dia = ', output_period_bgc_avg_dia, &
             ' is not a whole multiple of the MARBL timestep used, ', dt * real(marbl_timestep_ratio, rp)
        error stop 'init_bgc_output: output_period_bgc_avg_dia'
      endif
    endif
    call set_stream(his, wrt_bgc_his, 'bgc', IO_KIND_BGC, .false., .false., output_period_bgc_his, nrpf_bgc_his)
    call set_stream(avg, wrt_bgc_avg, 'bgc_avg', IO_KIND_BGCAVG, .true., .false., output_period_bgc_avg, nrpf_bgc_avg)
    call set_stream(dhis, wrt_bgc_dia_his, 'bgc_dia', IO_KIND_BGCDIA, .false., .true., &
                    output_period_bgc_his_dia, nrpf_bgc_his_dia)
    call set_stream(davg, wrt_bgc_dia_avg, 'bgc_dia_avg', IO_KIND_BGCDIAAVG, .true., .true., &
                    output_period_bgc_avg_dia, nrpf_bgc_avg_dia)
    if (.not. (his%on .or. avg%on .or. dhis%on .or. davg%on)) return

    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    gxi_u = merge(int(decomp%gnx), int(decomp%gnx) + 1, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    geta_v = merge(int(decomp%gny), int(decomp%gny) + 1, decomp%periodic_y)
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    nxy = (ir1 - ir0 + 1) * (jr1 - jr0 + 1)
    cap = 1
    if (his%on .or. avg%on) then
      call select_tracers(sel_ts, sel_cdr, sel_marbl)
      nba = count(sel_marbl(1:int(n_marbl)))
      allocate(bsrc(max(nba, 1)), bsrc_d(max(nba, 1)))
      nba = 0; bsrc = 0
      do m = 1, int(n_marbl)
        if (sel_marbl(m)) then
          nba = nba + 1; bsrc(nba) = m
        endif
      enddo
      bsrc_d = bsrc
      if (avg%on) allocate(ba(max(nba, 1), 0:ntpcol, nz))
      cap = max(cap, nba * nz * nxy)
    endif
    if (dhis%on .or. davg%on) then
      call select_diagnostics()
      if (any(d2(1:nd2) == DIA2_ZSATC) .or. any(d2(1:nd2) == DIA2_ZSATA)) marbl_need_zsat = .true.
      if (davg%on) allocate(a2(is:ie, js:je, max(nd2, 1)), a3(is:ie, js:je, nz, max(nd3, 1)))
      cap = max(cap, (nd2 + nd3 * nz) * nxy)
    endif
    call zero_all()
    call io_init(1, 1)
    call io_init8(cap, nxy * nz)
    if (his%on) call stage_tracers(his, start_time, 1_ip)
    if (dhis%on) call stage_dia(dhis, start_time)
  end subroutine init_bgc_output

  subroutine set_stream(s, on, name, kind, avg_stream, dia_stream, period, nrpf)
    type(stream_t), intent(out) :: s
    logical, intent(in) :: on, avg_stream, dia_stream
    character(len=*), intent(in) :: name
    integer, intent(in) :: kind, nrpf
    real(rp), intent(in) :: period
    s%on = on; s%name = name; s%kind = kind; s%avg = avg_stream; s%dia = dia_stream
    s%period = period; s%nrpf = nrpf
    if (on .and. comm_rank() == 0) write(*,'(3a,es12.4,a,i0,a)') 'BGC output _', trim(name), ': period ', period, &
         ' s, ', nrpf, ' records per file'
  end subroutine set_stream

  subroutine select_diagnostics()
    ! The CPU's marbldrv_parse_diagnostic_output_list, over the diagnostics
    ! the GPU computes.
    integer :: l, n
    logical :: s2(NDIA2), s3(NDIA3), known
    character(len=256) :: ln
    character(len=64) :: un
    s2 = .false.; s3 = .false.
    do l = 1, size(marbl_dia_list)
      if (len_trim(marbl_dia_list(l)) == 0) exit
      known = .false.
      do n = 1, NDIA2
        if (dia2_name(n) == marbl_dia_list(l)) then
          s2(n) = .true.; known = .true.
        endif
      enddo
      do n = 1, NDIA3
        if (dia3_name(n) == marbl_dia_list(l)) then
          s3(n) = .true.; known = .true.
        endif
      enddo
      if (known) cycle
      call marbl_diag_info(marbl_dia_list(l), known, ln, un)
      if (known) then
        write(*,'(3a)') 'MARBL_BIOGEOCHEMISTRY_SETTINGS: marbl_diagnostics_to_write names ', &
             trim(marbl_dia_list(l)), ', which the GPU model does not compute. It computes'
        write(*,'(a,*(1x,a))') ' ', (trim(dia2_name(n)), n = 1, NDIA2), (trim(dia3_name(n)), n = 1, NDIA3)
        error stop 'init_bgc_output: marbl_diagnostics_to_write not available'
      elseif (comm_rank() == 0) then
        write(*,'(3a)') 'REQUESTED DIAGNOSTIC ', trim(marbl_dia_list(l)), ' NOT RECOGNIZED'
      endif
    enddo
    if (.not. (any(s2) .or. any(s3))) then
      s2 = .true.; s3 = .true.
      if (comm_rank() == 0) write(*,'(a,i0,a)') 'marbl_diagnostics_to_write names no diagnostic: writing the ', &
           NDIA2 + NDIA3, ' the GPU model computes (UCLA ROMS writes all of MARBL''s)'
    endif
    nd2 = 0; nd3 = 0
    do n = 1, NDIA2
      if (.not. s2(n)) cycle
      nd2 = nd2 + 1; d2(nd2) = n
      call marbl_diag_info(dia2_name(n), known, d2_lname(nd2), d2_units(nd2))
    enddo
    do n = 1, NDIA3
      if (.not. s3(n)) cycle
      nd3 = nd3 + 1; d3(nd3) = n
      call marbl_diag_info(dia3_name(n), known, d3_lname(nd3), d3_units(nd3))
    enddo
  end subroutine select_diagnostics

  subroutine wrt_bgc()
    ! After every step (the CPU's wrt_bgc).
    real(rp) :: tnow
    if (.not. (his%on .or. avg%on .or. dhis%on .or. davg%on)) return
    tnow = start_time + dt * real(iic - ntstart + 1, rp)
    if (his%on) then
      his%otime = his%otime + dt
      if (his%otime >= his%period) call stage_tracers(his, tnow, nnew)
    endif
    if (avg%on) then
      call average_tracers(tnow)
      avg%otime = avg%otime + dt
      if (avg%otime >= avg%period) call stage_tracers(avg, tnow, nnew)
    endif
    if (davg%on) then
      if (davg%nsum == 0) then
        davg%begin_time = tnow - dt
        nheld = 0
        dia_first = .true.
      endif
      davg%nsum = davg%nsum + 1
      nheld = nheld + 1
      if (modulo(iic, marbl_timestep_ratio) == 0) call accumulate_dia()
      davg%otime = davg%otime + dt
      if (davg%otime >= davg%period) then
        call finalize_dia()
        call stage_dia(davg, tnow)
      endif
    endif
    if (dhis%on) then
      dhis%otime = dhis%otime + dt
      if (dhis%otime >= dhis%period) call stage_dia(dhis, tnow)
    endif
  end subroutine wrt_bgc

  subroutine average_tracers(tnow)
    ! The CPU's calc_avg_bgc: the running mean of the tracers at n+1.
    real(rp), intent(in) :: tnow
    real(rp) :: coef, omc
    integer :: m, c, k, nz, nb, nn
    if (avg%nsum == 0) avg%begin_time = tnow - dt
    avg%nsum = avg%nsum + 1
    coef = 1.0_rp / real(avg%nsum, rp)
    omc = 1.0_rp - coef
    nz = decomp%nz; nb = nba; nn = nnew
    if (nb < 1) return
    ! one thread per (tracer, column, level), tracers fastest: the tp and
    ! ba accesses of a warp are contiguous
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do c = 0, ntpcol
        do m = 1, nb
          ba(m, c, k) = real(real(ba(m, c, k), rp) * omc + real(tp(bsrc_d(m), c, k, nn), rp) * coef, pp)
        enddo
      enddo
    enddo
  end subroutine average_tracers

  subroutine accumulate_dia()
    ! The CPU's accumulate_dia_bgc: the diagnostics of the MARBL step just
    ! made, weighted by the steps since the last addition.
    real(rp) :: c0, w
    integer :: n, is, js, nz
    if (nheld == 0) return
    w = real(nheld, rp)
    c0 = merge(0.0_rp, 1.0_rp, dia_first)
    is = decomp%is; js = decomp%js; nz = decomp%nz
    do n = 1, nd2
      select case (d2(n))
      case (DIA2_ZSATC); call add2(a2(is, js, n), it_zsatc(is, js), c0, w)
      case (DIA2_ZSATA); call add2(a2(is, js, n), it_zsata(is, js), c0, w)
      case default;      call add2(a2(is, js, n), sf_diag(is, js, dia2_slot(d2(n))), c0, w)
      end select
    enddo
    do n = 1, nd3
      select case (d3(n))
      case (1); call add3(a3(is, js, 1, n), it_co3(is, js, 1), c0, w)
      case (2); call add3(a3(is, js, 1, n), it_co3a(is, js, 1), c0, w)
      case (3); call add3(a3(is, js, 1, n), it_co3sat(is, js, 1), c0, w)
      case (4); call add3(a3(is, js, 1, n), it_co3sata(is, js, 1), c0, w)
      end select
    enddo
    nheld = 0
    dia_first = .false.
  end subroutine accumulate_dia

  subroutine finalize_dia()
    ! The CPU's finalize_dia_bgc: sums to averages.
    real(rp) :: r
    integer :: i, j, k, n, i0, i1, j0, j1, nz, n2, n3
    call accumulate_dia()
    r = 1.0_rp / real(davg%nsum, rp)
    i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1; nz = decomp%nz; n2 = nd2; n3 = nd3
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          do n = 1, n3
            a3(i, j, k, n) = r * a3(i, j, k, n)
          enddo
          if (k == 1) then
            do n = 1, n2
              a2(i, j, n) = r * a2(i, j, n)
            enddo
          endif
        enddo
      enddo
    enddo
  end subroutine finalize_dia

  subroutine add2(a, x, c0, w)
    ! a = c0*a + w*x
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: c0, w
    integer :: i, j, i0, i1, j0, j1
    real(rp) :: c, ww
    c = c0; ww = w; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        a(i, j) = c * a(i, j) + ww * x(i, j)
      enddo
    enddo
  end subroutine add2

  subroutine add3(a, x, c0, w)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), WORK_ATTR intent(in) :: x(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
    real(rp), intent(in) :: c0, w
    integer :: i, j, k, i0, i1, j0, j1, nz
    real(rp) :: c, ww
    c = c0; ww = w; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          a(i, j, k) = c * a(i, j, k) + ww * x(i, j, k)
        enddo
      enddo
    enddo
  end subroutine add3

  subroutine zero_all()
    ! the sums to 0 on the device (the first update multiplies the old
    ! value by 0: no NaN may be there)
    integer :: i, j, k, m, c, n, is, ie, js, je, nz, nb, n2, n3
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    if (avg%on) then
      nb = size(ba, 1)
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do c = 0, ntpcol
          do m = 1, nb
            ba(m, c, k) = 0.0_pp
          enddo
        enddo
      enddo
    endif
    if (davg%on) then
      n2 = size(a2, 3); n3 = size(a3, 4)
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
      do k = 1, nz
        do j = js, je
          do i = is, ie
            do n = 1, n3
              a3(i, j, k, n) = 0.0_rp
            enddo
            if (k == 1) then
              do n = 1, n2
                a2(i, j, n) = 0.0_rp
              enddo
            endif
          enddo
        enddo
      enddo
    endif
  end subroutine zero_all

  subroutine begin_record(s, tnow)
    ! A new file when the last one is full, then the record's header.
    type(stream_t), intent(inout) :: s
    real(rp), intent(in) :: tnow
    character(len=512) :: open_path
    open_path = ''
    if (mod(s%rec, s%nrpf) == 0) then
      s%nfile = s%nfile + 1
      s%rec = 0
      call output_fname(root, trim(s%name), tnow, s%path)
      if (comm_rank() == 0) call create_file(s)
      call compute_barrier()       ! the file exists before any tile announces it
      open_path = s%path
    endif
    s%rec = s%rec + 1
    if (s%avg) then
      call io_stage8_begin(s%kind, s%rec, s%nfile, tnow, open_path, tavg=(/s%begin_time, tnow/))
    else
      call io_stage8_begin(s%kind, s%rec, s%nfile, tnow, open_path)
    endif
  end subroutine begin_record

  subroutine end_record(s, tnow)
    type(stream_t), intent(inout) :: s
    real(rp), intent(in) :: tnow
    call io_stage8_end()
    s%otime = 0.0_rp
    s%nsum = 0
    if (comm_rank() == 0) write(*,'(3a,es14.6,a,i0,a,i0)') 'staged _', trim(s%name), ' t = ', tnow, '  rec = ', &
         s%rec, '  file ', s%nfile
  end subroutine end_record

  subroutine stage_tracers(s, tnow, nn)
    type(stream_t), intent(inout) :: s
    real(rp), intent(in) :: tnow
    integer(ip), intent(in) :: nn           ! slot of the history's tracers
    integer :: m, nz
    nz = decomp%nz
    call begin_record(s, tnow)
    do m = 1, nba
      if (s%avg) then
        call io_stage8_3d_pt(trim(passive_name(bsrc(m))), ba(1, 0, 1), max(nba, 1), m, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      else
        call io_stage8_3d_pt(trim(passive_name(bsrc(m))), tp(1, 0, 1, nn), int(ntp), int(bsrc(m)), 1, nz, &
                             ir0, ir1, jr0, jr1, ir0, jr0)
      endif
    enddo
    call end_record(s, tnow)
  end subroutine stage_tracers

  subroutine stage_dia(s, tnow)
    type(stream_t), intent(inout) :: s
    real(rp), intent(in) :: tnow
    integer :: n, is, js, nz
    character(len=32) :: nm
    is = decomp%is; js = decomp%js; nz = decomp%nz
    call begin_record(s, tnow)
    do n = 1, nd2
      nm = dia2_name(d2(n))
      if (s%avg) then
        call io_stage8_2d(trim(nm), a2(is, js, n), ir0, ir1, jr0, jr1, ir0, jr0)
      else
        select case (d2(n))
        case (DIA2_ZSATC); call io_stage8_2d(trim(nm), it_zsatc(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
        case (DIA2_ZSATA); call io_stage8_2d(trim(nm), it_zsata(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
        case default
          call io_stage8_2d(trim(nm), sf_diag(is, js, dia2_slot(d2(n))), ir0, ir1, jr0, jr1, ir0, jr0)
        end select
      endif
    enddo
    do n = 1, nd3
      nm = dia3_name(d3(n))
      if (s%avg) then
        call io_stage8_3d(trim(nm), a3(is, js, 1, n), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      else
        select case (d3(n))
        case (1); call io_stage8_3d(trim(nm), it_co3(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        case (2); call io_stage8_3d(trim(nm), it_co3a(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        case (3); call io_stage8_3d(trim(nm), it_co3sat(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        case (4); call io_stage8_3d(trim(nm), it_co3sata(is, js, 1), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        end select
      endif
    enddo
    call end_record(s, tnow)
  end subroutine stage_dia

  subroutine create_file(s)
    ! The CPU's variables (double), long names and units.
    type(stream_t), intent(in) :: s
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, n, id
    character(len=4) :: pre
    n = index(s%path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//s%path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(s%path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    id = nc_def_dim(ncid, 's_w', decomp%nz + 1)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    if (s%avg) call def_avg_times(ncid, d_t)
    if (s%dia) then
      do n = 1, nd2
        call def(dia2_name(d2(n)), (/d_xi, d_eta, d_t/), d2_lname(n), d2_units(n))
      enddo
      do n = 1, nd3
        call def(dia3_name(d3(n)), (/d_xi, d_eta, d_s, d_t/), d3_lname(n), d3_units(n))
      enddo
    else
      pre = ''
      if (s%avg) pre = 'avg_'
      do n = 1, nba
        call def(passive_name(bsrc(n)), (/d_xi, d_eta, d_s, d_t/), trim(pre)//marbl_tracer_lnames(bsrc(n)), &
                 marbl_tracer_units(bsrc(n)))
      enddo
    endif
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu '//trim(s%name))
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call sponge_global_atts(ncid)   ! sponge layer (&v_sponge_settings)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(4a)') 'created _', trim(s%name), ' file: ', trim(s%path)
  contains
    subroutine def(name, dims, ln, un)
      character(len=*), intent(in) :: name, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      vid = nc_def_var(ncid, trim(name), dims)
      call nc_put_att_text(ncid, trim(name), 'long_name', trim(ln))
      call nc_put_att_text(ncid, trim(name), 'units', trim(un))
    end subroutine def
  end subroutine create_file

end module bgc_out

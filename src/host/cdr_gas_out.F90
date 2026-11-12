#include "mem_attr.inc"
module cdr_gas_out
  ! CDR gas-exchange output (CPU cdr_gas_exch_output.F90): the surface
  ! carbonate sensitivities ddic_dco2 (beta) and ddic_dalk (eta) of the
  ! ALT_CO2 carbon system, written in the form the CDR_LITE forcing reads
  ! (variables and their ddic_*_time axes), so the files can force a later
  ! run. &cdr_gas_exch_output_settings; files
  ! <root>_cdrgas.<date>.nc (the CPU naming, his_io output_fname), double.
  !  - an instantaneous "bracket" record at the start of the run and,
  !    unless the last record already sits there, at its end; bracket
  !    records do not count toward nrpf_cdr_gas;
  !  - every output_period_cdr_gas a record: instantaneous, or with
  !    wrt_cdr_gas_avg the window means that reproduce the mean gas
  !    exchange coefficients, beta = n/sum(1/beta), eta = sum(eta/beta)/
  !    sum(1/beta), over the steps where the solve succeeded (0 where none
  !    did), stamped at the middle of the window;
  !  - ocean_time, avg_begin_time/avg_end_time (with wrt_cdr_gas_avg) and
  !    ddic_dco2_time = ddic_dalk_time (the record's forcing time, days).
  ! Not on the GPU yet: cdr_gas_monthly_averages (stops at start-up).
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use implicit_fs_gpu, only: fs_global_atts
  use io_server, only: IO_KIND_CDRGAS, io_init, io_init8, io_stage8_begin, io_stage8_2d, io_stage8_end
  use comm_mod, only: comm_rank, compute_barrier
  use passive_mod, only: ntp, tp
  use carb_sens_mod, only: cs_surface
  use cdr_lite_io, only: alt_co2_tracers
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: read_cdr_gas_output_namelist, set_cdr_gas_output_root, init_cdr_gas_output
  public :: wrt_cdr_gas, wrt_cdr_gas_final

  real(rp) :: output_period_cdr_gas = 3600.0_rp
  integer  :: nrpf_cdr_gas = 4
  logical  :: wrt_cdr_gas_avg = .false., cdr_gas_monthly_averages = .false., do_cdr_gas_exch_output = .false.
  namelist /cdr_gas_exch_output_settings/ output_period_cdr_gas, nrpf_cdr_gas, &
       wrt_cdr_gas_avg, cdr_gas_monthly_averages, do_cdr_gas_exch_output

  logical :: on = .false.
  character(len=512) :: root = './output', fname = ''
  integer :: record = 0, periodic_in_file = 0, nfile = 0, navg = 0
  real(rp) :: output_time = 0.0_rp, avg_begin_time = 0.0_rp
  real(rp) :: last_record_days = -huge(1.0_rp)
  integer(ip) :: ialk = 0, idic = 0, ipo4 = 0, isio3 = 0
  integer :: ir0, ir1, jr0, jr1, gxi, geta, gxi_u, geta_v
  real(rp), WORK_ATTR allocatable :: gb(:,:), ge(:,:), sib(:,:), seib(:,:)
  integer, WORK_ATTR allocatable :: nok(:,:)

contains

  subroutine read_cdr_gas_output_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    rewind(unit)
    read(unit, nml=cdr_gas_exch_output_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'CDR_GAS_EXCH_OUTPUT_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_cdr_gas_output_namelist'
    endif
    if (ios < 0) do_cdr_gas_exch_output = .false.
  end subroutine read_cdr_gas_output_namelist

  subroutine set_cdr_gas_output_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_cdr_gas_output_root

  subroutine init_cdr_gas_output()
    ! After the other output (staging sized); writes the bracket record at
    ! the start time from the initial state (slot 1).
    integer :: nxy, is, ie, js, je
    on = .false.
    if (.not. do_cdr_gas_exch_output) return
    if (cdr_gas_monthly_averages) error stop 'init_cdr_gas_output: cdr_gas_monthly_averages is not on the GPU model yet'
    if (nrpf_cdr_gas < 1) error stop 'init_cdr_gas_output: nrpf_cdr_gas must be >= 1'
    call alt_co2_tracers(ialk, idic, ipo4, isio3)
    if (ialk == 0 .or. idic == 0 .or. ipo4 == 0 .or. isio3 == 0) &
      error stop 'init_cdr_gas_output: cdr_gas_exch_output requires tracers PO4, SiO3, ALK_ALT_CO2 and DIC_ALT_CO2'
    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    gxi_u = merge(int(decomp%gnx), int(decomp%gnx) + 1, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    geta_v = merge(int(decomp%gny), int(decomp%gny) + 1, decomp%periodic_y)
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    allocate(gb(is:ie, js:je), ge(is:ie, js:je), sib(is:ie, js:je), seib(is:ie, js:je), nok(is:ie, js:je))
    call zero_state(.true.)
    nxy = (ir1 - ir0 + 1) * (jr1 - jr0 + 1)
    call io_init(1, 1)
    call io_init8(2 * nxy, nxy)
    record = 0; periodic_in_file = 0; nfile = 0; navg = 0; output_time = 0.0_rp
    last_record_days = -huge(1.0_rp)
    on = .true.
    if (comm_rank() == 0) write(*,'(a,es12.4,a,i0,a,l2)') 'cdr_gas output: period ', output_period_cdr_gas, &
         ' s, ', nrpf_cdr_gas, ' records per file, averages', wrt_cdr_gas_avg
    call write_record(.true., start_time, 1_ip)
  end subroutine init_cdr_gas_output

  subroutine zero_state(all)
    ! Averaging sums to 0 (and beta/eta when all), on the device.
    logical, intent(in) :: all
    integer :: i, j, is, ie, js, je, fa
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    fa = merge(1, 0, all)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = js, je
      do i = is, ie
        sib(i, j) = 0.0_rp; seib(i, j) = 0.0_rp; nok(i, j) = 0
        if (fa == 1) then
          gb(i, j) = 0.0_rp; ge(i, j) = 0.0_rp
        endif
      enddo
    enddo
  end subroutine zero_state

  subroutine accumulate()
    ! The CPU's calc_average: where beta > 0, add 1/beta and eta/beta.
    integer :: i, j
    integer(ip) :: i0, i1, j0, j1
    i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        if (gb(i, j) > 0.0_rp) then
          sib(i, j) = sib(i, j) + 1.0_rp / gb(i, j)
          seib(i, j) = seib(i, j) + ge(i, j) / gb(i, j)
          nok(i, j) = nok(i, j) + 1
        endif
      enddo
    enddo
  end subroutine accumulate

  subroutine finish_average()
    ! The CPU's finish_average: window means into gb/ge (0 where no step of
    ! the window had a valid solve), then the sums are reset.
    integer :: i, j
    integer(ip) :: i0, i1, j0, j1
    i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        if (nok(i, j) > 0) then
          gb(i, j) = nok(i, j) / sib(i, j)
          ge(i, j) = seib(i, j) / sib(i, j)
        else
          gb(i, j) = 0.0_rp
          ge(i, j) = 0.0_rp
        endif
      enddo
    enddo
  end subroutine finish_average

  subroutine wrt_cdr_gas()
    ! Every step, after the step (the CPU's wrt_cdr_gas).
    real(rp) :: tnow
    if (.not. on) return
    tnow = start_time + dt * real(iic - ntstart + 1, rp)
    if (wrt_cdr_gas_avg) then
      if (navg == 0) avg_begin_time = tnow - dt
      navg = navg + 1
      call cs_surface(tp, ntp, nnew, ialk, idic, ipo4, isio3, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip), gb, ge)
      call accumulate()
    endif
    output_time = output_time + dt
    if (output_time >= output_period_cdr_gas) then
      call write_record(.false., tnow, nnew)
      output_time = 0.0_rp
    endif
  end subroutine wrt_cdr_gas

  subroutine wrt_cdr_gas_final()
    ! After the last step: a bracket record at the end time, unless the last
    ! record already sits there (within a quarter step).
    real(rp) :: tnow
    if (.not. on) return
    tnow = start_time + dt * real(iic - ntstart + 1, rp)
    if (abs(tnow * sec2day - last_record_days) < 0.25_rp * dt * sec2day) return
    call write_record(.true., tnow, nnew)
  end subroutine wrt_cdr_gas_final

  subroutine write_record(bracket, tnow, slot)
    logical, intent(in) :: bracket
    real(rp), intent(in) :: tnow
    integer(ip), intent(in) :: slot
    character(len=512) :: open_path
    logical :: use_avg, new_file
    real(rp) :: tforce, tbeg
    integer :: is, js
    is = decomp%is; js = decomp%js
    use_avg = wrt_cdr_gas_avg .and. .not. bracket
    if (use_avg) then
      tforce = 0.5_rp * (avg_begin_time + tnow) * sec2day
      tbeg = avg_begin_time
    else
      tforce = tnow * sec2day
      tbeg = tnow
    endif
    new_file = (record == 0)
    if (.not. bracket) new_file = new_file .or. periodic_in_file >= nrpf_cdr_gas
    open_path = ''
    if (new_file) then
      nfile = nfile + 1
      call output_fname(root, 'cdrgas', tnow, fname)
      if (comm_rank() == 0) call create_file(fname)
      call compute_barrier()
      open_path = fname
      record = 0
      periodic_in_file = 0
    endif
    record = record + 1
    if (.not. bracket) periodic_in_file = periodic_in_file + 1
    if (use_avg) then
      call finish_average()
    else
      call cs_surface(tp, ntp, slot, ialk, idic, ipo4, isio3, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip), gb, ge)
    endif
    if (wrt_cdr_gas_avg) then
      call io_stage8_begin(IO_KIND_CDRGAS, record, nfile, tnow, open_path, tavg=(/tbeg, tnow/), tforce=tforce)
    else
      call io_stage8_begin(IO_KIND_CDRGAS, record, nfile, tnow, open_path, tforce=tforce)
    endif
    call io_stage8_2d('ddic_dco2', gb(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
    call io_stage8_2d('ddic_dalk', ge(is, js), ir0, ir1, jr0, jr1, ir0, jr0)
    call io_stage8_end()
    if (use_avg) then
      call zero_state(.false.)
      navg = 0
    endif
    last_record_days = tforce
    if (comm_rank() == 0) write(*,'(a,a,f11.4,a,i0,a,i0)') 'staged cdr_gas ', merge('bracket ', 'record  ', bracket), &
         tnow * sec2day, ' days  rec = ', record, '  file ', nfile
  end subroutine write_record

  subroutine create_file(path)
    ! The CPU's dimensions, variables (double) and attributes.
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, id, n
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    call def1('avg_begin_time', (/d_t/), 'Time at beginning of averaging period', 'seconds')
    call def1('avg_end_time', (/d_t/), 'Time at end of averaging period', 'seconds')
    if (wrt_cdr_gas_avg) then
      call def1('ddic_dco2', (/d_xi, d_eta, d_t/), 'surface carbonate sensitivity beta = dDIC/dCO2 (ALT_CO2); '// &
                'averaged records: 1/<1/beta> over the window', 'nondimensional')
      call def1('ddic_dalk', (/d_xi, d_eta, d_t/), 'surface carbonate sensitivity eta = dDIC/dALK (ALT_CO2); '// &
                'averaged records: <eta/beta>/<1/beta> over the window', 'nondimensional')
    else
      call def1('ddic_dco2', (/d_xi, d_eta, d_t/), 'surface carbonate sensitivity beta = dDIC/dCO2 (ALT_CO2)', &
                'nondimensional')
      call def1('ddic_dalk', (/d_xi, d_eta, d_t/), 'surface carbonate sensitivity eta = dDIC/dALK (ALT_CO2)', &
                'nondimensional')
    endif
    call def1('ddic_dco2_time', (/d_t/), 'time of ddic_dco2 record, days since reference date', 'days')
    call def1('ddic_dalk_time', (/d_t/), 'time of ddic_dalk record, days since reference date', 'days')
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu CDR gas-exchange output')
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created cdr_gas file: ', trim(path)
  contains
    subroutine def1(name, dims, ln, un)
      character(len=*), intent(in) :: name, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      vid = nc_def_var(ncid, name, dims)
      call nc_put_att_text(ncid, name, 'long_name', ln)
      call nc_put_att_text(ncid, name, 'units', un)
    end subroutine def1
  end subroutine create_file

end module cdr_gas_out

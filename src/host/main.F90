program roms_gpu
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use step_driver
  use case_io, only: load_case, report_loaded_case
  use his_io, only: maybe_wrt_output
  use cdr_lite_out, only: wrt_cdr_lite
  use random_out, only: wrt_random
  use cdr_gas_out, only: wrt_cdr_gas, wrt_cdr_gas_final
  use cdr_out, only: wrt_cdr
  use avg_out, only: wrt_avg, step_bgc_avg
  use extract_out, only: wrt_extract
  use sponge_io, only: adjust_orlanski
  use sponge_out, only: wrt_spn
  use upscale_out, only: wrt_upscale_final
  use arch_mod, only: init_arch, arch_name, select_device
  use comm_mod, only: init_comm, finalize_comm, comm_rank, comm_size, is_io_rank, comm_node_rank, comm_world_rank, &
                      is_reader_rank
  use frc_reader, only: frc_reader_run, frc_reader_stop
  use io_server, only: io_server_run, io_finalize
  use passive_mod, only: passive_exchange_finish
  use halo_mod, only: hg_audit_report
  use frc_io, only: frc_prefetch_drain
  use feature_guard, only: check_namelist_features
  use run_timers, only: rt_tic, rt_toc, rt_report, rt_nsteps, RT_LOOP, RT_OUTPUT, RT_FINAL, RT_INIT, &
                        rt_steps_init, rt_step_begin, rt_step_end
  implicit none
#ifndef ROMS_GPU_VERSION
#define ROMS_GPU_VERSION "unknown"
#endif

  integer(ip) :: gnx, gny, nz, halo, npx, npy, rank
  integer(ip) :: ndtfast_in, ntimes_in, nt_in, istep
  real(rp) :: dt_in, H0, dx, f0
  character(len=32) :: mode
  character(len=512) :: nml_path

  mode = 'geom'
  if (command_argument_count() >= 1) call get_command_argument(1, mode)
  call init_comm()
  if (comm_world_rank() == 0) write(*,'(2a)') 'roms_gpu version ', ROMS_GPU_VERSION
  if (trim(mode) == 'load' .or. trim(mode) == 'run') then
    ! Every rank, writers included, so that all stop together.
    nml_path = 'namelist.nml'
    if (command_argument_count() >= 2) call get_command_argument(2, nml_path)
    call check_namelist_features(nml_path)
  endif
  if (is_io_rank()) then
    ! Output server: no GPU context, no model state. Serves until QUIT.
    call io_server_run()
    call finalize_comm()
    stop
  endif
  if (is_reader_rank()) then
    ! Forcing reader (frc_reader): no GPU context, no model state. Reads the
    ! records compute rank 0 posts until it says quit.
    call frc_reader_run()
    call finalize_comm()
    stop
  endif
  call wall_stamp('start')
  call rt_tic(RT_INIT)
  call select_device(comm_node_rank())
  call init_arch()

  ! Ideal-box defaults stay one tile. A loaded case uses namelist
  ! np_xi/np_eta when comm_size matches, otherwise 1 x 1.
  gnx = 32; gny = 32; nz = 16; halo = 2
  npx = 1; npy = 1; rank = comm_rank()
  ndtfast_in = 60; ntimes_in = 3; nt_in = 2
  dt_in = 100.0_rp
  H0 = 4000.0_rp; dx = 12000.0_rp; f0 = 1.0e-4_rp

  if (trim(mode) == 'load' .or. trim(mode) == 'run') then
    nml_path = 'namelist.nml'
    if (command_argument_count() >= 2) call get_command_argument(2, nml_path)
    call load_case(nml_path)
    call wall_stamp('init done')
    write(*,'(a,a)') 'roms_gpu  architecture = ', trim(arch_name())
    write(*,'(a,2i6)') '  MPI rank, size = ', comm_rank(), comm_size()
    write(*,'(a)') 'roms_gpu  (loaded case)'
    write(*,'(a,3i6)') '  local nx,ny,nz = ', decomp%nx, decomp%ny, decomp%nz
    write(*,'(a,2i6)') '  nfast, ndtfast = ', nfast, ndtfast
    write(*,'(a,4l2)') '  edges W E S N  = ', decomp%edge_w, decomp%edge_e, &
                        decomp%edge_s, decomp%edge_n
    write(*,'(a,3l2,a,4l2)') '  flux,bulk,bry = ', have_flux, have_bulk, have_bry, &
                        '  OBC WENS =', obc_west, obc_east, obc_south, obc_north
    write(*,'(a,2l2)') '  pot,bry tides = ', pot_tides, bry_tides
    call report_loaded_case()
    if (trim(mode) == 'run') then
      call rt_toc(RT_INIT)
      call rt_steps_init(max(ntimes, 1))
      call rt_tic(RT_LOOP)
      do istep = 1, max(ntimes, 1)
        iic = ntstart + istep - 1
        call rt_step_begin()
        call run_baroclinic_step()
        ! ub_tune: the CPU's adjust_orlanski, before the outputs here, so that a
        ! restart holds the coefficients the next step uses (exact restart)
        call adjust_orlanski()
        call wrt_spn()                  ! ub_tune + wrt_sponge: the _spn file
        call rt_tic(RT_OUTPUT)
        call maybe_wrt_output()
        call wrt_avg()
        call wrt_extract()
        call wrt_cdr_lite()
        call wrt_random()
        call wrt_cdr_gas()
        call wrt_cdr()
        call step_bgc_avg()
        call rt_toc(RT_OUTPUT)
        rt_nsteps = rt_nsteps + 1
        call rt_step_end()
      enddo
      call rt_toc(RT_LOOP)
      call wrt_upscale_final()        ! the CPU's final wrt_upscale
      call wrt_cdr_gas_final()        ! end-time bracket record of the CDR gas-exchange output
      call wall_stamp('loop done')
      call rt_tic(RT_FINAL)
      call frc_reader_stop()
      call io_finalize()
      call rt_toc(RT_FINAL)
      call rt_report()
    endif
  else
  call init_decomp(gnx, gny, nz, halo, npx, npy, rank, .true., .true.)
  call init_time_state(nz, ndtfast_in, dt_in, ntimes_in, nt_in)
  call alloc_state(nt)
  call init_ideal_box(H0, dx, f0)

  write(*,'(a,a)') 'roms_gpu  architecture = ', trim(arch_name())
  write(*,'(a,2i6)') '  MPI rank, size = ', comm_rank(), comm_size()
  write(*,'(a,3i6)') '  local nx,ny,nz = ', decomp%nx, decomp%ny, decomp%nz
  write(*,'(a,2i6)') '  nfast, ndtfast = ', nfast, ndtfast
  write(*,'(a,4l2)') '  edges W E S N  = ', decomp%edge_w, decomp%edge_e, &
                      decomp%edge_s, decomp%edge_n
  write(*,'(a,4i4)') '  nbrs  W E S N  = ', decomp%nbr_w, decomp%nbr_e, &
                      decomp%nbr_s, decomp%nbr_n

  if (trim(mode) == 'geom') then
    call run_geom_only()
  elseif (trim(mode) == 'baro') then
    call run_barotropic_only()
  elseif (trim(mode) == 'prsgrd') then
    call run_prsgrd_only()
  elseif (trim(mode) == 'advect') then
    call run_advect_only()
  elseif (trim(mode) == 'step') then
    call run_baroclinic_step()
  elseif (trim(mode) == 'lmd') then
    call run_lmd_only()
  else
    write(*,*) 'usage: roms_gpu [geom|baro|prsgrd|advect|step|lmd|load|run [namelist]]'
    error stop
  endif
  endif

  call passive_exchange_finish()
  call frc_prefetch_drain()
  call hg_audit_report()
  call frc_reader_stop()
  call io_finalize()
  call free_state()
  call finalize_comm()
end program roms_gpu

subroutine wall_stamp(tag)
  ! Wall clock since the first stamp, printed by every compute rank.
  use comm_mod, only: comm_rank, is_io_rank, is_reader_rank
  use run_timers, only: wall_time
  implicit none
  character(len=*), intent(in) :: tag
  real(8), save :: t0 = -1.0d0
  real(8) :: now
  if (is_io_rank() .or. is_reader_rank()) return
  now = wall_time()
  if (t0 < 0.0d0) t0 = now
  write(*,'(a,i0,a,a,a,f10.2,a)') 'wall: rank ', comm_rank(), '  ', tag, '  t = ', now - t0, ' s'
  if (tag == 'init done' .or. tag == 'restart staged' .or. tag == 'loop done') call gpu_mem_report(tag)
  flush(6)
end subroutine wall_stamp

subroutine gpu_mem_report(tag)
  ! Device memory in use on this rank's GPU (resident managed pages included).
  use comm_mod, only: comm_rank
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  character(len=*), intent(in) :: tag
#ifdef USE_CUDA
  integer(kind=cuda_count_kind) :: freeb, totalb
  integer :: ierr
  ierr = cudaMemGetInfo(freeb, totalb)
  if (ierr == 0) write(*,'(a,i0,a,a,a,f7.2,a,f7.2,a)') 'gpumem: rank ', comm_rank(), '  ', tag, &
       '  used ', real(totalb - freeb, 8) / 2.0d0**30, ' GB of ', real(totalb, 8) / 2.0d0**30, ' GB'
#endif
end subroutine gpu_mem_report

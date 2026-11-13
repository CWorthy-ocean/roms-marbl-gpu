#include "mem_attr.inc"
module random_out
  ! The CPU's random_output module (&random_output_settings:
  ! output_period_random, nrpf_random, do_random), modified to write the
  ! surface (top-level) field of every generic passive tracer
  ! (passive_tracer<n>, param_settings nt_passive). Every
  ! output_period_random seconds after the step (the CPU's wrt_random) one
  ! record is written to <root>_rnd.<date>.nc (his_io output_fname)
  ! (nrpf_random records per file). The fields are written in the passive
  ! storage precision (double with -DPASSIVE_FP64, float otherwise); the
  ! time stays double.
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use nc_io
  use implicit_fs_gpu, only: fs_global_atts
  use io_server, only: IO_KIND_RND, io_init, io_init8, io_stage8_begin, io_stage8_2d, io_stage8_end
  use comm_mod, only: comm_rank, compute_barrier
  use passive_mod, only: ntp, tp, n_marbl, n_generic, passive_name
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: read_random_namelist, set_random_output_root, init_random_output, wrt_random

  real(rp) :: output_period_random = 3600.0_rp
  integer  :: nrpf_random = 10
  logical  :: do_random = .false.
  namelist /random_output_settings/ output_period_random, nrpf_random, do_random
  ! Output fields in float when the passive tracers are stored in float32.
  logical, parameter :: out_float = (precision(1.0_pp) < precision(1.0_rp))

  logical :: on = .false.
  character(len=512) :: root = './output', fname = ''
  integer :: record = 0, nfile = 0
  real(rp) :: output_time = 0.0_rp
  real(rp), WORK_ATTR allocatable :: rnd_sfc(:,:)       ! (is:ie, js:je) scratch
  integer :: ir0, ir1, jr0, jr1, nxi, neta, gxi, geta

contains

  subroutine read_random_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    rewind(unit)
    read(unit, nml=random_output_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'RANDOM_OUTPUT_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_random_namelist'
    endif
    if (ios < 0) do_random = .false.
  end subroutine read_random_namelist

  subroutine set_random_output_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_random_output_root

  subroutine init_random_output()
    ! After init_output (history and restart staging are sized).
    integer :: nxy
    on = .false.
    if (.not. do_random) return
    if (n_generic <= 0) error stop 'init_random_output: do_random is .true. but there are no passive tracers '// &
         '(nt_passive is 0 in PARAM_SETTINGS)'
    if (nrpf_random < 1) error stop 'init_random_output: nrpf_random must be >= 1'
    if (output_period_random <= 0.0_rp) error stop 'init_random_output: output_period_random must be > 0'
    ! the points this tile writes (as his_io): interior plus a physical
    ! boundary row/column
    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    nxi = ir1 - ir0 + 1; neta = jr1 - jr0 + 1
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    allocate(rnd_sfc(decomp%is:decomp%ie, decomp%js:decomp%je))
    rnd_sfc = 0.0_rp
    nxy = nxi * neta
    call io_init(1, 1)               ! no-op when history or restarts are on
    call io_init8(int(n_generic) * nxy, nxy * decomp%nz)
    record = nrpf_random; nfile = 0; output_time = 0.0_rp
    on = .true.
    if (comm_rank() == 0) write(*,'(a,es12.4,a,i0,a,i0,a,a)') 'random output: surface passive tracers every ', &
         output_period_random, ' s, ', nrpf_random, ' records per file, ', n_generic, ' tracers, fields ', &
         merge('float ', 'double', out_float)
  end subroutine init_random_output

  subroutine wrt_random()
    ! Every step, after the step (the CPU's wrt_random).
    real(rp) :: tnow
    if (.not. on) return
    output_time = output_time + dt
    if (output_time >= output_period_random) then
      tnow = start_time + dt * real(iic - ntstart + 1, rp)
      call write_record(tnow)
      output_time = 0.0_rp
    endif
  end subroutine wrt_random

  subroutine write_record(tnow)
    real(rp), intent(in) :: tnow
    character(len=512) :: open_path
    integer(ip) :: m
    open_path = ''
    if (record >= nrpf_random) then
      nfile = nfile + 1
      record = 0
      call output_fname(root, 'rnd', tnow, fname)
      if (comm_rank() == 0) call create_file(fname)
      call compute_barrier()
      open_path = fname
    endif
    record = record + 1
    call io_stage8_begin(IO_KIND_RND, record, nfile, tnow, open_path)
    ! one field at a time through rnd_sfc (stream order: each gather follows
    ! the staging of the field before it)
    do m = 1, n_generic
      call gather_surface(n_marbl + m)
      call io_stage8_2d(trim(passive_name(n_marbl + m)), rnd_sfc, ir0, ir1, jr0, jr1, ir0, jr0)
    enddo
    call io_stage8_end()
    if (comm_rank() == 0) write(*,'(a,es14.6,a,i0,a,i0)') 'staged random t = ', tnow, '  rec = ', record, &
         '  file ', nfile
  end subroutine write_record

  subroutine gather_surface(itp)
    ! rnd_sfc = top level of tp tracer itp, slot nnew (0 on land: column 0).
    integer(ip), intent(in) :: itp
    integer(ip) :: i, j, nz, n, it, col, i0, i1, j0, j1
    nz = decomp%nz; n = nnew; it = itp
    i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        col = tpcol(i, j)
        rnd_sfc(i, j) = real(tp(it, col, nz, n), rp)
      enddo
    enddo
  end subroutine gather_surface

  subroutine create_file(path)
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_eta, id, n
    integer(ip) :: m
    character(len=32) :: name
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    do m = 1, n_generic
      name = passive_name(n_marbl + m)
      if (out_float) then
        id = nc_def_var_f(ncid, trim(name), (/d_xi, d_eta, d_t/))
      else
        id = nc_def_var(ncid, trim(name), (/d_xi, d_eta, d_t/))
      endif
      call nc_put_att_text(ncid, trim(name), 'long_name', 'surface '//trim(name))
      call nc_put_att_text(ncid, trim(name), 'units', 'tracer units')
    enddo
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu random output: surface passive tracers')
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created random file: ', trim(path)
  end subroutine create_file

end module random_out

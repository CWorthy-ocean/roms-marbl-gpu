#include "mem_attr.inc"
module cdr_lite_out
  ! CDR_LITE output (CPU cdr_lite_output.F90): &cdr_lite_output_settings,
  ! one file series <root>_cdrtrc.<date>.nc (the CPU naming, his_io
  ! output_fname) of nrpf_cdr_lite records every output_period_cdr_lite,
  ! with the CPU's dimensions, variables and attributes. The fields are
  ! written in the passive storage precision: double with -DPASSIVE_FP64
  ! (as the CPU), float in the float32 build (computed and averaged in
  ! double, rounded once by netCDF at the write); the times stay double:
  !   CDR_<..><n>          tracers (averages with wrt_cdr_lite_avg)
  !   hCDR_<..><n>         thickness-weighted, instantaneous at the write
  !   hCDR_<..><n>_avg     thickness-weighted average (wrt_cdr_lite_avg)
  !   int_z_CDR_<..><n>    vertical integral (average or instantaneous)
  !   CDR_OAE_ALK<n>_source, CDR_DOR_DIC<n>_source   release sources
  !   FG_CDR_<..>_DIC<n>   air-sea CO2 flux of the DIC tracers
  ! wrt_alk selects the OAE ALK tracers, wrt_dic the OAE and DOR DIC
  ! tracers. Averages follow the CPU's calc_average: every step after the
  ! step, avg = avg*(1-coef) + x*coef with coef = 1./navg in single
  ! precision, avg_begin_time = time - dt at the first step of a period,
  ! avg_end_time = ocean_time = time (n+1). Not on the GPU yet:
  ! cdr_lite_monthly_averages (stops at start-up).
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use implicit_fs_gpu, only: fs_global_atts
  use io_server, only: IO_KIND_CDR, io_init, io_init8, io_stage8_begin, io_stage8_2d, io_stage8_3d, &
                       io_stage8_3d_pt, io_stage8_end
  use comm_mod, only: comm_rank, compute_barrier
  use passive_mod, only: ntp, tp, n_oae, n_dor, ncdr, cdr0, passive_name
  use cdr_lite_mod
  use cdr_src_mod, only: cdr_on, cdr_src_out
  implicit none
  private
  ! Output fields in float when the passive tracers are stored in float32.
  logical, parameter :: out_float = (precision(1.0_pp) < precision(1.0_rp))
  public :: read_cdr_lite_output_namelist, set_cdr_lite_output_root, init_cdr_lite_output, wrt_cdr_lite

  real(rp) :: output_period_cdr_lite = 3600.0_rp
  integer  :: nrpf_cdr_lite = 4
  logical  :: wrt_cdr_lite_avg = .false., cdr_lite_monthly_averages = .false., do_cdr_lite_output = .false.
  logical  :: wrt_tracers = .false., wrt_vertical_integrals = .false., wrt_thickness_weighted = .false.
  logical  :: wrt_sources = .false.
  logical  :: wrt_alk = .true., wrt_dic = .true.
  logical  :: wrt_gas_exchange = .false.
  namelist /cdr_lite_output_settings/ output_period_cdr_lite, nrpf_cdr_lite, &
       wrt_cdr_lite_avg, cdr_lite_monthly_averages, do_cdr_lite_output, &
       wrt_tracers, wrt_vertical_integrals, wrt_thickness_weighted, &
       wrt_sources, wrt_alk, wrt_dic, wrt_gas_exchange

  logical :: on = .false., wsrc = .false., wfg = .false.
  character(len=512) :: root = './output', fname = ''
  integer :: record = 0, nfile = 0, navg = 0
  real(rp) :: output_time = 0.0_rp, avg_begin_time = 0.0_rp
  integer(ip) :: no = 0, ns = 0
  integer(ip), allocatable :: oitr(:), sitr(:)          ! output tracers, source tracers (passive indices)
  character(len=32), allocatable :: oname(:), sunit(:)
  character(len=48), allocatable :: sname(:)
  character(len=64), allocatable :: olname(:), slname(:)
  integer :: ir0, ir1, jr0, jr1, nxi, neta, gxi, geta, gxi_u, geta_v

contains

  subroutine read_cdr_lite_output_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    rewind(unit)
    read(unit, nml=cdr_lite_output_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'CDR_LITE_OUTPUT_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_cdr_lite_output_namelist'
    endif
    if (ios < 0) do_cdr_lite_output = .false.
  end subroutine read_cdr_lite_output_namelist

  subroutine set_cdr_lite_output_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_cdr_lite_output_root

  subroutine init_cdr_lite_output()
    ! After init_output (history and restart staging) and init_cdr_lite.
    integer(ip) :: m, k
    integer :: nxy, n3, n2
    character(len=8) :: num
    on = .false.
    if (.not. do_cdr_lite_output) return
    if (ncdr == 0) error stop 'init_cdr_lite_output: do_cdr_lite_output is .true. but no CDR_LITE tracers '// &
         'are configured (nt_cdr_oae and nt_cdr_dor are both 0 in PARAM_SETTINGS)'
    if (cdr_lite_monthly_averages) error stop 'init_cdr_lite_output: cdr_lite_monthly_averages is not on the GPU model yet'
    if (nrpf_cdr_lite < 1) error stop 'init_cdr_lite_output: nrpf_cdr_lite must be >= 1'
    ! output tracers in the CPU's order: OAE ALK, OAE DIC, DOR DIC
    allocate(oitr(ncdr), oname(ncdr), olname(ncdr))
    no = 0
    if (wrt_alk) then
      do m = 1, n_oae
        no = no + 1; oitr(no) = cdr0 + 2 * m - 1
        write(num, '(i0)') m; olname(no) = 'CDR OAE ALK tracer'//trim(num)
      enddo
    endif
    if (wrt_dic) then
      do m = 1, n_oae
        no = no + 1; oitr(no) = cdr0 + 2 * m
        write(num, '(i0)') m; olname(no) = 'CDR OAE DIC tracer'//trim(num)
      enddo
      do m = 1, n_dor
        no = no + 1; oitr(no) = cdr0 + 2 * n_oae + m
        write(num, '(i0)') m; olname(no) = 'CDR DOR DIC tracer'//trim(num)
      enddo
    endif
    do k = 1, no
      oname(k) = passive_name(oitr(k))
    enddo
    ! release sources (with the releases on): OAE ALK, DOR DIC
    wsrc = cdr_on .and. wrt_sources
    allocate(sitr(ncdr), sname(ncdr), slname(ncdr), sunit(ncdr))
    ns = 0
    if (wsrc) then
      if (wrt_alk) then
        do m = 1, n_oae
          ns = ns + 1; sitr(ns) = cdr0 + 2 * m - 1
          slname(ns) = 'CDR OAE ALK source from CDR module'; sunit(ns) = 'meq/s'
        enddo
      endif
      if (wrt_dic) then
        do m = 1, n_dor
          ns = ns + 1; sitr(ns) = cdr0 + 2 * n_oae + m
          slname(ns) = 'CDR DOR DIC source from CDR module'; sunit(ns) = 'mmol/s'
        enddo
      endif
      do k = 1, ns
        sname(k) = trim(passive_name(sitr(k)))//'_source'
      enddo
    endif
    wfg = wrt_gas_exchange .and. wrt_dic
    call set_bounds()
    call cdrl_out_alloc(no, oitr, wrt_cdr_lite_avg, wrt_tracers, wrt_thickness_weighted, &
                        wrt_vertical_integrals, ns, wfg, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
    ! double staging: the largest record of this tile
    nxy = nxi * neta
    n3 = 0; n2 = 0
    if (wrt_tracers) n3 = n3 + int(no)
    if (wrt_thickness_weighted) n3 = n3 + int(no) * merge(2, 1, wrt_cdr_lite_avg)
    if (wrt_vertical_integrals) n2 = n2 + int(no)
    n3 = n3 + int(ns)
    if (wfg) n2 = n2 + int(cdrl_nd)
    call io_init(1, 1)               ! no-op when history or restarts are on
    call io_init8(n3 * nxy * decomp%nz + n2 * nxy, nxy * decomp%nz)
    record = nrpf_cdr_lite; nfile = 0; navg = 0; output_time = 0.0_rp
    on = .true.
    if (comm_rank() == 0) then
      write(*,'(a,es12.4,a,i0,a,l2,a,i0,a,i0,a,l2,a,a)') 'cdr_lite output: period ', output_period_cdr_lite, &
           ' s, ', nrpf_cdr_lite, ' records per file, averages', wrt_cdr_lite_avg, ', tracers ', no, &
           ', sources ', ns, ', gas exchange', wfg, ', fields ', merge('float ', 'double', out_float)
    endif
  end subroutine init_cdr_lite_output

  subroutine set_bounds()
    ! The points this tile writes (as his_io): interior plus a physical
    ! boundary row/column.
    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    nxi = ir1 - ir0 + 1
    neta = jr1 - jr0 + 1
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    gxi_u = merge(int(decomp%gnx), int(decomp%gnx) + 1, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    geta_v = merge(int(decomp%gny), int(decomp%gny) + 1, decomp%periodic_y)
  end subroutine set_bounds

  subroutine wrt_cdr_lite()
    ! Every step, after the step (the CPU's wrt_cdr_lite).
    real(rp) :: tnow, om, cf
    real :: coef
    integer(ip) :: k
    if (.not. on) return
    tnow = start_time + dt * real(iic - ntstart + 1, rp)
    if (wrt_cdr_lite_avg) then
      if (navg == 0) avg_begin_time = tnow - dt
      navg = navg + 1
      coef = 1. / navg
      om = real(1 - coef, rp); cf = real(coef, rp)
      call cdrl_out_accum(tp, ntp, nnew, om, cf, wrt_tracers, wrt_thickness_weighted, &
                          wrt_vertical_integrals, wfg, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
      do k = 1, ns
        call cdr_src_out(co_src(decomp%is, decomp%js, 1, k), sitr(k), om, cf, .false.)
      enddo
    endif
    output_time = output_time + dt
    if (output_time >= output_period_cdr_lite) then
      call write_record(tnow)
      output_time = 0.0_rp
    endif
  end subroutine wrt_cdr_lite

  subroutine write_record(tnow)
    real(rp), intent(in) :: tnow
    character(len=512) :: open_path
    integer(ip) :: o, k, m, is, js, nz
    is = decomp%is; js = decomp%js; nz = decomp%nz
    open_path = ''
    if (record >= nrpf_cdr_lite) then
      nfile = nfile + 1
      record = 0
      call output_fname(root, 'cdrtrc', tnow, fname)
      if (comm_rank() == 0) call create_file(fname)
      call compute_barrier()
      open_path = fname
    endif
    record = record + 1
    if (wrt_cdr_lite_avg) then
      call io_stage8_begin(IO_KIND_CDR, record, nfile, tnow, open_path, tavg=(/avg_begin_time, tnow/))
    else
      call io_stage8_begin(IO_KIND_CDR, record, nfile, tnow, open_path)
    endif
    if (wrt_cdr_lite_avg) then
      ! the averages are gathered into co_tmp3/co_tmp2 field by field
      ! (stream order: each gather follows the staging of the field before)
      if (wrt_tracers) then
        do o = 1, no
          call cdrl_out_gather3(1, o, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
          call io_stage8_3d(trim(oname(o)), co_tmp3, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
          call cdrl_out_zero(1, o)
        enddo
      endif
      if (wrt_thickness_weighted) then
        do o = 1, no
          call cdrl_out_inst_h(tp, ntp, nnew, o, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
          call io_stage8_3d('h'//trim(oname(o)), co_tmp3, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
          call cdrl_out_gather3(2, o, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
          call io_stage8_3d('h'//trim(oname(o))//'_avg', co_tmp3, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
          call cdrl_out_zero(2, o)
        enddo
      endif
      if (wrt_vertical_integrals) then
        do o = 1, no
          call cdrl_out_gather2(o, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
          call io_stage8_2d('int_z_'//trim(oname(o)), co_tmp2, ir0, ir1, jr0, jr1, ir0, jr0)
          call cdrl_out_zero(3, o)
        enddo
      endif
      do k = 1, ns
        call io_stage8_3d(trim(sname(k)), co_src(is, js, 1, k), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        call cdrl_out_zero(4, k)
      enddo
      if (wfg) then
        do m = 1, cdrl_nd
          call io_stage8_2d('FG_'//trim(passive_name(cdrl_dic_h(m))), co_fg(is, js, m), ir0, ir1, jr0, jr1, ir0, jr0)
          call cdrl_out_zero(5, m)
        enddo
      endif
      navg = 0
    else
      if (wrt_tracers) then
        do o = 1, no
          call io_stage8_3d_pt(trim(oname(o)), tp(1, 0, 1, nnew), ntp, oitr(o), 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        enddo
      endif
      if (wrt_thickness_weighted) then
        do o = 1, no
          call cdrl_out_inst_h(tp, ntp, nnew, o, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
          call io_stage8_3d('h'//trim(oname(o)), co_tmp3, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
        enddo
      endif
      if (wrt_vertical_integrals) then
        do o = 1, no
          call cdrl_out_inst_iz(tp, ntp, nnew, o, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip))
          call io_stage8_2d('int_z_'//trim(oname(o)), co_tmp2, ir0, ir1, jr0, jr1, ir0, jr0)
        enddo
      endif
      do k = 1, ns
        call cdrl_out_zero(6, 1_ip)
        call cdr_src_out(co_tmp3, sitr(k), 0.0_rp, 0.0_rp, .true.)
        call io_stage8_3d(trim(sname(k)), co_tmp3, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
      enddo
      if (wfg) then
        do m = 1, cdrl_nd
          call io_stage8_2d('FG_'//trim(passive_name(cdrl_dic_h(m))), cdrl_gas(is, js, m), ir0, ir1, jr0, jr1, ir0, jr0)
        enddo
      endif
    endif
    call io_stage8_end()
    write(*,'(a,es14.6,a,i0,a,i0)') 'staged cdr_lite t = ', tnow, '  rec = ', record, '  file ', nfile
  end subroutine write_record

  subroutine create_file(path)
    ! The CPU's dimensions, variables and attributes; fields in the passive
    ! storage precision (deff), the time variables in double (def1).
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, id, o, k, m, n
    integer :: d3(4), d2(3)
    character(len=32) :: dname
    character(len=128) :: lname
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    d3 = (/d_xi, d_eta, d_s, d_t/); d2 = (/d_xi, d_eta, d_t/)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    call def1('avg_begin_time', (/d_t/), 'Time at beginning of averaging period', 'seconds')
    call def1('avg_end_time', (/d_t/), 'Time at end of averaging period', 'seconds')
    if (wrt_tracers) then
      do o = 1, no
        call deff(trim(oname(o)), d3, trim(olname(o)), 'mMol m-3')
      enddo
    endif
    if (wrt_vertical_integrals) then
      do o = 1, no
        ! the CPU names the DIC integrals 'instantaneous' even when averaged
        if (index(oname(o), '_ALK') > 0) then
          lname = 'depth-integrated '//trim(olname(o))
        else
          lname = 'instantaneous depth-integrated '//trim(olname(o))
        endif
        call deff('int_z_'//trim(oname(o)), d2, trim(lname), 'meters * mMol m-3')
      enddo
    endif
    if (wrt_thickness_weighted) then
      do o = 1, no
        call deff('h'//trim(oname(o)), d3, 'instantaneous thickness-weighted '//trim(olname(o)), 'meters mMol m-3')
      enddo
      if (wrt_cdr_lite_avg) then
        do o = 1, no
          call deff('h'//trim(oname(o))//'_avg', d3, 'time-averaged thickness-weighted '//trim(olname(o)), &
                    'meters mMol m-3')
        enddo
      endif
    endif
    do k = 1, ns
      call deff(trim(sname(k)), d3, trim(slname(k)), trim(sunit(k)))
    enddo
    if (wfg) then
      do m = 1, cdrl_nd
        dname = passive_name(cdrl_dic_h(m))
        lname = 'air-sea CO2 flux into '//trim(olname_of(cdrl_dic_h(m)))//' (positive into ocean)'
        call deff('FG_'//trim(dname), d2, trim(lname), 'mmol/m^2/s')
      enddo
    endif
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu CDR_LITE output')
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created cdr_lite file: ', trim(path)
  contains
    subroutine def1(name, dims, ln, un)
      character(len=*), intent(in) :: name, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      vid = nc_def_var(ncid, name, dims)
      call nc_put_att_text(ncid, name, 'long_name', ln)
      call nc_put_att_text(ncid, name, 'units', un)
    end subroutine def1
    subroutine deff(name, dims, ln, un)
      character(len=*), intent(in) :: name, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      if (out_float) then
        vid = nc_def_var_f(ncid, name, dims)
      else
        vid = nc_def_var(ncid, name, dims)
      endif
      call nc_put_att_text(ncid, name, 'long_name', ln)
      call nc_put_att_text(ncid, name, 'units', un)
    end subroutine deff
  end subroutine create_file

  function olname_of(itp) result(ln)
    integer(ip), intent(in) :: itp
    character(len=64) :: ln
    integer(ip) :: m
    character(len=8) :: num
    if (itp <= cdr0 + 2 * n_oae) then
      m = (itp - cdr0 + 1) / 2
      write(num, '(i0)') m
      if (mod(itp - cdr0, 2_ip) == 1) then
        ln = 'CDR OAE ALK tracer'//trim(num)
      else
        ln = 'CDR OAE DIC tracer'//trim(num)
      endif
    else
      write(num, '(i0)') itp - cdr0 - 2 * n_oae
      ln = 'CDR DOR DIC tracer'//trim(num)
    endif
  end function olname_of

end module cdr_lite_out

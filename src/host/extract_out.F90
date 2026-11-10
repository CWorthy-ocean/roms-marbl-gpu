#include "mem_attr.inc"
module extract_out
  ! Output for nested child grids (CPU extract_data, its PARALLEL_IO path):
  ! the parent's fields interpolated to the points of a child grid's
  ! boundaries and written as the child's boundary file.
  !
  ! &extract_data_settings as the CPU. The objects file (extract_file) has
  ! one variable per object, (n2, n1): n1 points of a child boundary in the
  ! parent's absolute index space ([0, gnx] x [0, gny]; n2 = 2: i, j for
  ! rho points; n2 = 3: i, j and the child's grid angle for u/v points),
  ! named <set>_<side>_<stagger> (side north/south/east/west), with the
  ! fields in its output_vars attribute (matched as substrings, as the CPU's
  ! findstr: zeta, ubar, vbar, u, v, temp, salt, passive, cdr_oae, cdr_dor,
  ! bgc, up, vp; w is not supported). The child's sizes come from the
  ! dimensions child_xi_rho and child_eta_rho; N_chd, theta_s_chd,
  ! theta_b_chd and hc_chd give the child's vertical grid (as the CPU, the
  ! dimensions <prefix>_xi_rho / <prefix>_eta_rho are taken first, <prefix>
  ! the set name of the first object).
  !
  ! As the CPU: each tile takes the points inside it (the first contiguous
  ! run); the interpolation is bilinear with the mask in the weights,
  ! normalized (rho points rmask, u/v points umask/vmask); u/v are rotated
  ! by the parent angle minus the child angle; when the child's vertical
  ! grid differs, every column is remapped (the CPU's flux-conserving PPM)
  ! from the parent layers at the point (Hz interpolated; at u/v points
  ! the mean of the two rho columns) to the child layers of the same total
  ! depth. Fields: zeta, ubar, vbar at knew (time n+1); u, v and the tracers
  ! at slot nstp (time n); the pressure fluxes up and vp of this step
  ! (calc_pflx, pflx_mod) as the CPU after its fix 4b07d30e: the interior
  ! values (1:nx, 1:ny) of each tile, exchanged (zero beyond a physical
  ! edge), interpolated at the u points (up) and the v points (vp) of a
  ! u- or v-object and rotated as ubar/vbar. A record every
  ! output_period_extract (after the step), nrpf_extract per file,
  ! bry_time in days (time n+1).
  !
  ! Files <extract_root_name>_bry.<date>.nc (the CPU naming, his_io
  ! output_fname), in double, the CPU's dimensions and variables: for each child
  ! side present, zeta, ubar, vbar, every tracer, u and v, and up/vp where an
  ! object of the side asks for them; only the fields an object asks for
  ! are written. The interpolation runs on the device;
  ! rotation and remap on the host of the tile; compute rank 0 writes the
  ! sum over the ranks of the boundary lines.
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod
  use gpu_state
  use time_state
  use run_timers, only: wall_time
  use netcdf
  use nc_io, only: nc_create, nc_def_dim, nc_def_var, nc_put_att_text, nc_enddef, nc_close, nc_open_write, &
                   nc_put_1d_rec
  use comm_mod, only: comm_rank, comm_reduce_sum
  use passive_mod, only: ntp, n_marbl, n_generic, tp, passive_name, passive_marbl, ncdr, cdr0, n_oae, n_dor
  use marbl_host, only: marbl_tracer_units, marbl_tracer_lnames
  use cdr_io, only: remap_src_to_grid
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  use pflx_mod, only: calc_pflx, up, vp
  use halo_mod, only: halo_fill_2d
  implicit none
  private
  public :: read_extract_namelist, init_extract, wrt_extract

  ! &extract_data_settings
  real(rp) :: output_period_extract = 0.0_rp
  character(len=256) :: extract_file = 'sample_edata.nc', extract_root_name = ''
  integer :: nrpf_extract = 0, N_chd = 0
  real(rp) :: theta_s_chd = 5.0_rp, theta_b_chd = 2.0_rp, hc_chd = 250.0_rp
  logical :: do_extract = .false.

  type obj_t
    character(len=60) :: name = ''
    character(len=20) :: set = '', bnd = ''      ! bnd with its leading '_'
    integer :: dsize = 0, np = 0, start = 1, side = 0, p0 = 0
    logical :: scalar = .true.
    logical :: zeta = .false., ubar = .false., vbar = .false., u = .false., v = .false., w = .false., &
               temp = .false., salt = .false., passive = .false., cdr_oae = .false., cdr_dor = .false., &
               up = .false., vp = .false., bgc = .false.
    real(rp), allocatable :: ipos(:), jpos(:), ang(:), cosa(:), sina(:)
  end type obj_t
  type(obj_t), allocatable :: obj(:)
  integer :: nobj = 0, LLm_chd = 0, MMm_chd = 0, nlev = 0
  logical :: on = .false., mismatch = .true.
  logical :: side_on(4) = .false.               ! 1 north, 2 south, 3 east, 4 west (the CPU's order)
  logical :: side_up(4) = .false., side_vp(4) = .false.   ! the CPU's child_up / child_vp
  ! the CPU's upe/vpe: up/vp on the tile interior, exchanged
  real(rp), WORK_ATTR allocatable :: upe(:,:), vpe(:,:)
  logical :: want_pflx = .false.
  character(len=5), parameter :: side_name(4) = (/'north', 'south', 'east ', 'west '/)
  integer :: dim_t(4), dim_u(4), dim_v(4)       ! child boundary lengths (rho, u, v points)
  real(rp), allocatable :: Cs_w_chd(:)
  ! interpolation sets for all local points p (1:npt): set 1 rho, 2 rho at
  ! i-1 (u-point thickness), 3 rho at j-1 (v-point thickness), 4 u, 5 v
  integer :: npt = 0
  integer(ip), WORK_ATTR allocatable :: cxi(:,:), cxj(:,:)
  real(rp), WORK_ATTR allocatable :: cxc(:,:,:)
  ! one record's interpolations, all objects and fields, queued on the
  ! device into dbuf (field f of object o at fld_off(o, f): (np, nk)
  ! column-major) and copied to the host once (hb)
  integer, parameter :: F_HZ1 = 1, F_ZETA = 2, F_UBAR = 3, F_VBAR = 4, F_U = 5, F_V = 6, F_HZ2 = 7, F_HZ3 = 8, &
                        F_UP = 9, F_VP = 10, F_TRC = 10
  integer, allocatable :: fld_off(:,:)
  integer :: nbtot = 0
#ifdef USE_CUDA
  real(rp), device, allocatable :: dbuf(:)
#else
  real(rp), allocatable :: dbuf(:)
#endif
  real(rp), allocatable :: hb(:)
  ! tracers in the CPU's order: kind 1 temp, 2 salt, 3 generic passive, 4 CDR_LITE, 5 MARBL
  integer :: ntrc = 0
  integer, allocatable :: trc_kind(:), trc_src(:)
  character(len=32), allocatable :: trc_name(:)
  character(len=128), allocatable :: trc_lname(:), trc_units(:)
  ! output variables: name, side, levels, length, offset in the line buffer
  integer :: nvar = 0, nbuf = 0
  character(len=64), allocatable :: var_name(:)
  integer, allocatable :: var_len(:), var_lev(:), var_off(:)
  logical, allocatable :: var_wr(:)
  real(8), allocatable :: hbuf(:), gbuf(:)
  integer :: record = 0, nfile = 0
  real(rp) :: otime = 0.0_rp
  character(len=512) :: fname = ''

contains

  subroutine read_extract_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    namelist /extract_data_settings/ output_period_extract, extract_file, nrpf_extract, N_chd, theta_s_chd, &
         theta_b_chd, hc_chd, do_extract, extract_root_name
    rewind(unit)
    read(unit, nml=extract_data_settings, iostat=ios)
    if (ios > 0) error stop 'read_extract_namelist: extract_data_settings'
    if (ios < 0) do_extract = .false.
  end subroutine read_extract_namelist

  logical function findstr(string, pattern)
    ! the CPU's findstr: pattern (untrimmed) is a substring of trim(string)
    character(len=*), intent(in) :: string, pattern
    integer :: i, nl, nlv
    nl = len_trim(string); nlv = len(pattern)
    findstr = .false.
    do i = 1, nl - nlv + 1
      if (string(i:i + nlv - 1) == pattern) then
        findstr = .true.
        return
      endif
    enddo
  end function findstr

  subroutine ex_check(ierr, what)
    integer, intent(in) :: ierr
    character(len=*), intent(in) :: what
    if (ierr /= nf90_noerr) then
      write(*,'(4a)') 'extract_out: ', trim(what), ': ', trim(nf90_strerror(ierr))
      error stop 'extract_out: netCDF error'
    endif
  end subroutine ex_check

  subroutine init_extract()
    ! The CPU's init_extract_data and read_extraction_objects.
    integer :: ncid, io, dimids(2), n1, n2, cxr, cer, s, l, i, np, p, k
    character(len=60) :: oname, pfx
    character(len=150) :: vars
    real(rp), allocatable :: ob(:,:)
    on = do_extract
    if (.not. on) return
    if (output_period_extract <= 0.0_rp) error stop 'init_extract: output_period_extract must be > 0'
    if (nrpf_extract < 1) error stop 'init_extract: nrpf_extract must be >= 1'
    if (N_chd < 1) error stop 'init_extract: N_chd must be >= 1'
    call ex_check(nf90_open(trim(extract_file), nf90_nowrite, ncid), 'open '//trim(extract_file))
    call ex_check(nf90_inquire(ncid, nVariables=nobj), 'inquire')
    ! The CPU's child size: <prefix>_xi_rho / <prefix>_eta_rho, <prefix> the
    ! set name of the first object (roms-tools make_nesting_info prefix),
    ! else child_xi_rho / child_eta_rho.
    call ex_check(nf90_inquire_variable(ncid, 1, name=oname), 'inquire variable')
    pfx = oname
    if (index(oname, '_') > 0) pfx = oname(1:index(oname, '_') - 1)
    if (nf90_inq_dimid(ncid, trim(pfx)//'_xi_rho', cxr) /= nf90_noerr) &
      call ex_check(nf90_inq_dimid(ncid, 'child_xi_rho', cxr), trim(pfx)//'_xi_rho (or child_xi_rho)')
    if (nf90_inq_dimid(ncid, trim(pfx)//'_eta_rho', cer) /= nf90_noerr) &
      call ex_check(nf90_inq_dimid(ncid, 'child_eta_rho', cer), trim(pfx)//'_eta_rho (or child_eta_rho)')
    call ex_check(nf90_inquire_dimension(ncid, cxr, len=LLm_chd), 'child_xi_rho')
    call ex_check(nf90_inquire_dimension(ncid, cer, len=MMm_chd), 'child_eta_rho')
    allocate(obj(nobj))
    do io = 1, nobj
      call ex_check(nf90_inquire_variable(ncid, io, name=oname, dimids=dimids), 'inquire variable')
      call ex_check(nf90_inquire_dimension(ncid, dimids(1), len=n1), 'object dimension')
      call ex_check(nf90_inquire_dimension(ncid, dimids(2), len=n2), 'object dimension')
      obj(io)%name = oname
      obj(io)%dsize = n1
      obj(io)%scalar = (n2 == 2)
      allocate(ob(n1, n2))
      call ex_check(nf90_get_var(ncid, io, ob), 'read '//trim(oname))
      call local_points(obj(io), ob(:, 1), ob(:, 2))
      if (.not. obj(io)%scalar .and. obj(io)%np > 0) &
        obj(io)%ang = ob(obj(io)%start:obj(io)%start + obj(io)%np - 1, 3)
      deallocate(ob)
      ! set = name up to the first '_', bnd = '_<side>' up to the second
      l = len_trim(oname)
      s = index(oname, '_')
      if (s > 0) then
        obj(io)%set = oname(1:s - 1)
        i = index(oname(s + 1:l), '_')
        if (i > 0) then
          obj(io)%bnd = oname(s:s + i - 1)
        else
          obj(io)%bnd = ' '
        endif
      else
        obj(io)%set = oname
      endif
      select case (trim(obj(io)%bnd))
      case ('_north'); obj(io)%side = 1
      case ('_south'); obj(io)%side = 2
      case ('_east');  obj(io)%side = 3
      case ('_west');  obj(io)%side = 4
      end select
      if (obj(io)%side > 0) side_on(obj(io)%side) = .true.
      vars = ''
      call ex_check(nf90_get_att(ncid, io, 'output_vars', vars), 'output_vars of '//trim(oname))
      obj(io)%zeta = findstr(vars, 'zeta'); obj(io)%temp = findstr(vars, 'temp')
      obj(io)%salt = findstr(vars, 'salt'); obj(io)%passive = findstr(vars, 'passive')
      obj(io)%cdr_oae = findstr(vars, 'cdr_oae'); obj(io)%cdr_dor = findstr(vars, 'cdr_dor')
      obj(io)%ubar = findstr(vars, 'ubar'); obj(io)%vbar = findstr(vars, 'vbar')
      obj(io)%u = findstr(vars, 'u'); obj(io)%v = findstr(vars, 'v'); obj(io)%w = findstr(vars, 'w')
      obj(io)%up = findstr(vars, 'up'); obj(io)%vp = findstr(vars, 'vp'); obj(io)%bgc = findstr(vars, 'bgc')
      if ((obj(io)%up .or. obj(io)%vp) .and. .not. calc_pflx) &
        error stop 'init_extract: up/vp extraction needs &calc_pflx_settings calc_pflx = .true.'
      if (obj(io)%w) error stop 'init_extract: w extraction is not supported (the CPU''s parallel-I/O path does not write it)'
      if (obj(io)%bgc .and. .not. passive_marbl) error stop 'init_extract: bgc extraction needs MARBL'
      if ((obj(io)%u .or. obj(io)%v .or. obj(io)%ubar .or. obj(io)%vbar .or. obj(io)%up .or. obj(io)%vp) &
          .and. obj(io)%scalar) &
        error stop 'init_extract: velocities need an object with angles (i, j, angle)'
      if (obj(io)%side == 0) error stop 'init_extract: object names must be <set>_<north|south|east|west>_...'
      if (obj(io)%up) side_up(obj(io)%side) = .true.
      if (obj(io)%vp) side_vp(obj(io)%side) = .true.
      if (obj(io)%up .or. obj(io)%vp) want_pflx = .true.
    enddo
    call ex_check(nf90_close(ncid), 'close')
    dim_t = (/LLm_chd, LLm_chd, MMm_chd, MMm_chd/)
    dim_u = (/LLm_chd - 1, LLm_chd - 1, MMm_chd, MMm_chd/)
    dim_v = (/LLm_chd, LLm_chd, MMm_chd - 1, MMm_chd - 1/)
    mismatch = .not. (decomp%nz == N_chd .and. theta_s == theta_s_chd .and. theta_b == theta_b_chd .and. &
                      hc == hc_chd)
    nlev = merge(N_chd, int(decomp%nz), mismatch)
    if (mismatch) then
      allocate(Cs_w_chd(0:N_chd))
      call set_chd_scoord()
    endif
    if (comm_rank() == 0) write(*,'(a,i0,a,i0,a,i0,a,l2)') 'extract: ', nobj, ' objects, child ', LLm_chd, ' x ', &
         MMm_chd, ', vertical remap', mismatch
    if (want_pflx) then
      allocate(upe(decomp%is:decomp%ie, decomp%js:decomp%je), vpe(decomp%is:decomp%ie, decomp%js:decomp%je))
      upe = 0.0_rp; vpe = 0.0_rp
    endif
    call set_coefs()
    call set_tracers()
    call set_vars()
    call set_fields()
    record = nrpf_extract; nfile = 0; otime = 0.0_rp
  end subroutine init_extract

  subroutine local_points(o, gi, gj)
    ! The CPU's find_local_points: the first contiguous run of points inside
    ! this tile ([0, nx) x [0, ny) after the tile offset), in tile-absolute
    ! coordinates.
    type(obj_t), intent(inout) :: o
    real(rp), intent(in) :: gi(:), gj(:)
    real(rp), allocatable :: li(:), lj(:)
    integer :: i, n, s, e
    n = size(gi)
    allocate(li(n), lj(n))
    li = gi - real(decomp%i0 - 1, rp)
    lj = gj - real(decomp%j0 - 1, rp)
    s = 0
    do i = 1, n
      if (li(i) >= 0.0_rp .and. li(i) < real(decomp%nx, rp) .and. lj(i) >= 0.0_rp .and. lj(i) < real(decomp%ny, rp)) then
        s = i
        exit
      endif
    enddo
    e = n
    if (s > 0) then
      do i = s, n
        if (li(i) < 0.0_rp .or. li(i) >= real(decomp%nx, rp) .or. lj(i) < 0.0_rp .or. lj(i) >= real(decomp%ny, rp)) then
          e = i - 1
          exit
        endif
      enddo
      o%np = e - s + 1
      o%start = s
      o%ipos = li(s:e)
      o%jpos = lj(s:e)
    else
      o%np = 0
      o%start = 1
    endif
  end subroutine local_points

  subroutine coef_set(set, p0, ipos, jpos, mask)
    ! The CPU's compute_coef: bilinear weights times the mask, normalized.
    integer, intent(in) :: set, p0
    real(rp), intent(in) :: ipos(:), jpos(:)
    real(rp), WORK_ATTR intent(in) :: mask(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: p, ip_, jp_
    real(rp) :: cfx, cfy, c(4)
    do p = 1, size(ipos)
      ip_ = floor(ipos(p)); cfx = ipos(p) - real(ip_, rp)
      jp_ = floor(jpos(p)); cfy = jpos(p) - real(jp_, rp)
      c(1) = (1 - cfx) * (1 - cfy) * mask(ip_, jp_)
      c(2) = cfx * (1 - cfy) * mask(ip_ + 1, jp_)
      c(3) = (1 - cfx) * cfy * mask(ip_, jp_ + 1)
      c(4) = cfx * cfy * mask(ip_ + 1, jp_ + 1)
      c = c / sum(c)
      cxi(p0 + p, set) = ip_; cxj(p0 + p, set) = jp_
      cxc(p0 + p, 1:4, set) = c
    enddo
  end subroutine coef_set

  subroutine set_coefs()
    ! the CPU's init_extract_data: rho-index = absolute + 0.5; u-index =
    ! rho-index + 0.5; v-index = rho-index (j + 0.5); the rho weights at
    ! i-1 and j-1 for the u/v layer thickness; the rotation from the
    ! interpolated parent angle
    integer :: io, np, p0
    real(rp), allocatable :: ri(:), rj(:), angp(:)
    call sync_arch()
    npt = 0
    do io = 1, nobj
      obj(io)%p0 = npt
      npt = npt + obj(io)%np
    enddo
    allocate(cxi(max(npt, 1), 5), cxj(max(npt, 1), 5), cxc(max(npt, 1), 4, 5))
    cxi = 0; cxj = 0; cxc = 0.0_rp
    do io = 1, nobj
      np = obj(io)%np
      if (np == 0) cycle
      p0 = obj(io)%p0
      allocate(ri(np), rj(np))
      ri = obj(io)%ipos + 0.5_rp
      rj = obj(io)%jpos + 0.5_rp
      call coef_set(1, p0, ri, rj, rmask)
      if (.not. obj(io)%scalar) then
        call coef_set(2, p0, ri - 1.0_rp, rj, rmask)
        call coef_set(3, p0, ri, rj - 1.0_rp, rmask)
        allocate(angp(np))
        call interp2_host(angler, p0, np, angp)
        obj(io)%cosa = cos(angp - obj(io)%ang)
        obj(io)%sina = sin(angp - obj(io)%ang)
        deallocate(angp)
        call coef_set(4, p0, ri + 0.5_rp, rj, umask)
        call coef_set(5, p0, ri, rj + 0.5_rp, vmask)
      endif
      deallocate(ri, rj)
    enddo
  end subroutine set_coefs

  subroutine interp2_host(f, p0, np, out)
    real(rp), WORK_ATTR intent(in) :: f(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer, intent(in) :: p0, np
    real(rp), intent(out) :: out(np)
    integer :: p, i, j
    do p = 1, np
      i = cxi(p0 + p, 1); j = cxj(p0 + p, 1)
      out(p) = f(i, j) * cxc(p0 + p, 1, 1) + f(i + 1, j) * cxc(p0 + p, 2, 1) + &
               f(i, j + 1) * cxc(p0 + p, 3, 1) + f(i + 1, j + 1) * cxc(p0 + p, 4, 1)
    enddo
  end subroutine interp2_host

  subroutine set_tracers()
    ! every tracer, the CPU's order and metadata (T, S, generic passive,
    ! CDR_LITE, MARBL)
    integer :: m, n, ngen
    character(len=8) :: num
    ngen = int(n_generic)
    ntrc = 2 + ngen + int(ncdr) + int(n_marbl)
    allocate(trc_kind(ntrc), trc_src(ntrc), trc_name(ntrc), trc_lname(ntrc), trc_units(ntrc))
    n = 1; trc_kind(n) = 1; trc_src(n) = itemp; trc_name(n) = 'temp'
    trc_lname(n) = 'potential temperature'; trc_units(n) = 'Celsius'
    n = 2; trc_kind(n) = 2; trc_src(n) = isalt; trc_name(n) = 'salt'
    trc_lname(n) = 'salinity'; trc_units(n) = 'PSU'
    do m = 1, ngen
      n = n + 1; trc_kind(n) = 3; trc_src(n) = int(n_marbl) + m      ! tp index of generic tracer m
      write(num, '(i0)') m
      trc_name(n) = 'passive_tracer'//trim(num); trc_lname(n) = 'passive tracer'//trim(num)
      trc_units(n) = 'mMol m-3'
    enddo
    do m = 1, int(ncdr)
      n = n + 1; trc_kind(n) = 4; trc_src(n) = int(cdr0) + m
      trc_name(n) = passive_name(cdr0 + m); trc_units(n) = 'mMol m-3'
      if (m <= 2 * int(n_oae)) then
        write(num, '(i0)') (m + 1) / 2
        if (mod(m, 2) == 1) then
          trc_lname(n) = 'CDR OAE ALK tracer'//trim(num)
        else
          trc_lname(n) = 'CDR OAE DIC tracer'//trim(num)
        endif
      else
        write(num, '(i0)') m - 2 * int(n_oae)
        trc_lname(n) = 'CDR DOR DIC tracer'//trim(num)
      endif
    enddo
    if (passive_marbl) then
      do m = 1, int(n_marbl)
        n = n + 1; trc_kind(n) = 5; trc_src(n) = m
        trc_name(n) = passive_name(m); trc_lname(n) = marbl_tracer_lnames(m); trc_units(n) = marbl_tracer_units(m)
      enddo
    endif
  end subroutine set_tracers

  logical function trc_wanted(o, n)
    ! the object's tracer flags (temp, salt, passive, cdr_oae, cdr_dor, bgc)
    type(obj_t), intent(in) :: o
    integer, intent(in) :: n
    select case (trc_kind(n))
    case (1); trc_wanted = o%temp
    case (2); trc_wanted = o%salt
    case (3); trc_wanted = o%passive
    case (4)
      if (trc_src(n) - int(cdr0) <= 2 * int(n_oae)) then
        trc_wanted = o%cdr_oae
      else
        trc_wanted = o%cdr_dor
      endif
    case default; trc_wanted = o%bgc
    end select
  end function trc_wanted

  subroutine set_vars()
    ! The file's variables per side (the CPU's create_edata_file order:
    ! zeta, ubar, vbar, every tracer, u, v) and which of them are written.
    integer :: s, n, io
    nvar = 0
    allocate(var_name(4 * (ntrc + 7)), var_len(4 * (ntrc + 7)), var_lev(4 * (ntrc + 7)), &
             var_off(4 * (ntrc + 7)), var_wr(4 * (ntrc + 7)))
    nbuf = 0
    do s = 1, 4
      if (.not. side_on(s)) cycle
      call add('zeta_'//trim(side_name(s)), dim_t(s), 1, any(obj(:)%zeta .and. obj(:)%side == s))
      call add('ubar_'//trim(side_name(s)), dim_u(s), 1, any(obj(:)%ubar .and. obj(:)%side == s))
      call add('vbar_'//trim(side_name(s)), dim_v(s), 1, any(obj(:)%vbar .and. obj(:)%side == s))
      do n = 1, ntrc
        call add(trim(trc_name(n))//'_'//trim(side_name(s)), dim_t(s), nlev, wanted_any(n, s))
      enddo
      call add('u_'//trim(side_name(s)), dim_u(s), nlev, any(obj(:)%u .and. obj(:)%side == s))
      call add('v_'//trim(side_name(s)), dim_v(s), nlev, any(obj(:)%v .and. obj(:)%side == s))
      if (side_up(s)) call add('up_'//trim(side_name(s)), dim_u(s), 1, .true.)
      if (side_vp(s)) call add('vp_'//trim(side_name(s)), dim_v(s), 1, .true.)
    enddo
    allocate(hbuf(max(nbuf, 1)), gbuf(max(nbuf, 1)))
  contains
    subroutine add(nm, nl, nk, wr)
      character(len=*), intent(in) :: nm
      integer, intent(in) :: nl, nk
      logical, intent(in) :: wr
      nvar = nvar + 1
      var_name(nvar) = nm; var_len(nvar) = nl; var_lev(nvar) = nk; var_wr(nvar) = wr
      var_off(nvar) = nbuf
      if (wr) nbuf = nbuf + nl * nk
    end subroutine add
    logical function wanted_any(nn, ss)
      integer, intent(in) :: nn, ss
      wanted_any = .false.
      do io = 1, nobj
        if (obj(io)%side == ss .and. trc_wanted(obj(io), nn)) wanted_any = .true.
      enddo
    end function wanted_any
  end subroutine set_vars

  integer function var_index(nm) result(iv)
    character(len=*), intent(in) :: nm
    do iv = 1, nvar
      if (trim(var_name(iv)) == trim(nm)) return
    enddo
    iv = 0
  end function var_index

  ! ---------------------------------------------------------------- device interpolation
  subroutine set_fields()
    ! the offsets of every object's fields in the record buffer
    integer :: io, np, nz, n, off
    nz = decomp%nz
    allocate(fld_off(nobj, F_TRC + ntrc))
    fld_off = -1
    off = 0
    do io = 1, nobj
      np = obj(io)%np
      if (np == 0) cycle
      if (mismatch) call take(F_HZ1, nz)
      if (obj(io)%zeta) call take(F_ZETA, 1)
      do n = 1, ntrc
        if (trc_wanted(obj(io), n)) call take(F_TRC + n, nz)
      enddo
      if (.not. obj(io)%scalar) then
        if (obj(io)%ubar .or. obj(io)%vbar) then
          call take(F_UBAR, 1); call take(F_VBAR, 1)
        endif
        if (obj(io)%u .or. obj(io)%v) then
          call take(F_U, nz); call take(F_V, nz)
        endif
        if (obj(io)%up .or. obj(io)%vp) then
          call take(F_UP, 1); call take(F_VP, 1)
        endif
        if (obj(io)%u .and. mismatch) call take(F_HZ2, nz)
        if (obj(io)%v .and. mismatch) call take(F_HZ3, nz)
      endif
    enddo
    nbtot = off
    allocate(dbuf(max(nbtot, 1)), hb(max(nbtot, 1)))
  contains
    subroutine take(f, nk)
      integer, intent(in) :: f, nk
      fld_off(io, f) = off
      off = off + np * nk
    end subroutine take
  end subroutine set_fields

  subroutine interp3(f, k0, k1, set, p0, np, off)
    ! dbuf(off + (k - k0)*np + p): f at the four corners of set 'set'
    ! with its weights, summed in the CPU's order
    integer, intent(in) :: k0, k1, set, p0, np, off
    real(rp), WORK_ATTR intent(in) :: f(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: p, k, i, j, st, a, o, n
    st = set; a = p0; o = off; n = np
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = k0, k1
      do p = 1, n
        i = cxi(a + p, st); j = cxj(a + p, st)
        dbuf(o + (k - k0) * n + p) = f(i, j, k) * cxc(a + p, 1, st) + f(i + 1, j, k) * cxc(a + p, 2, st) + &
             f(i, j + 1, k) * cxc(a + p, 3, st) + f(i + 1, j + 1, k) * cxc(a + p, 4, st)
      enddo
    enddo
  end subroutine interp3

  subroutine interp_tp(itr, p0, np, off)
    ! a passive tracer (compressed layout, slot nstp) at rho points
    integer, intent(in) :: itr, p0, np, off
    integer :: p, k, i, j, a, nz, ns, it, o, n
    a = p0; nz = decomp%nz; ns = nstp; it = itr; o = off; n = np
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do p = 1, n
        i = cxi(a + p, 1); j = cxj(a + p, 1)
        dbuf(o + (k - 1) * n + p) = real(tp(it, tpcol(i, j), k, ns), rp) * cxc(a + p, 1, 1) + &
             real(tp(it, tpcol(i + 1, j), k, ns), rp) * cxc(a + p, 2, 1) + &
             real(tp(it, tpcol(i, j + 1), k, ns), rp) * cxc(a + p, 3, 1) + &
             real(tp(it, tpcol(i + 1, j + 1), k, ns), rp) * cxc(a + p, 4, 1)
      enddo
    enddo
  end subroutine interp_tp

  subroutine queue_record()
    ! every interpolation of the record, on the device
    integer :: io, np, p0, nz, n, kn, ns, is, js
    nz = decomp%nz; kn = knew; ns = nstp; is = decomp%is; js = decomp%js
    do io = 1, nobj
      np = obj(io)%np
      if (np == 0) cycle
      p0 = obj(io)%p0
      if (fld_off(io, F_HZ1) >= 0) call interp3(Hz, 1, nz, 1, p0, np, fld_off(io, F_HZ1))
      if (fld_off(io, F_ZETA) >= 0) call interp3(zeta(is, js, kn), 1, 1, 1, p0, np, fld_off(io, F_ZETA))
      do n = 1, ntrc
        if (fld_off(io, F_TRC + n) < 0) cycle
        if (trc_kind(n) <= 2) then
          call interp3(t(is, js, 1, ns, trc_src(n)), 1, nz, 1, p0, np, fld_off(io, F_TRC + n))
        else
          call interp_tp(trc_src(n), p0, np, fld_off(io, F_TRC + n))
        endif
      enddo
      if (fld_off(io, F_UBAR) >= 0) call interp3(ubar(is, js, kn), 1, 1, 4, p0, np, fld_off(io, F_UBAR))
      if (fld_off(io, F_VBAR) >= 0) call interp3(vbar(is, js, kn), 1, 1, 5, p0, np, fld_off(io, F_VBAR))
      if (fld_off(io, F_U) >= 0) call interp3(u(is, js, 1, ns), 1, nz, 4, p0, np, fld_off(io, F_U))
      if (fld_off(io, F_V) >= 0) call interp3(v(is, js, 1, ns), 1, nz, 5, p0, np, fld_off(io, F_V))
      if (fld_off(io, F_UP) >= 0) call interp3(upe, 1, 1, 4, p0, np, fld_off(io, F_UP))
      if (fld_off(io, F_VP) >= 0) call interp3(vpe, 1, 1, 5, p0, np, fld_off(io, F_VP))
      if (fld_off(io, F_HZ2) >= 0) call interp3(Hz, 1, nz, 2, p0, np, fld_off(io, F_HZ2))
      if (fld_off(io, F_HZ3) >= 0) call interp3(Hz, 1, nz, 3, p0, np, fld_off(io, F_HZ3))
    enddo
  end subroutine queue_record

  subroutine set_upe()
    ! The CPU's upe(1:nx,1:ny) = up(1:nx,1:ny) (vpe alike), then the
    ! exchange; the rest stays 0 (as allocated).
    integer :: i, j, nx, ny
    nx = decomp%nx; ny = decomp%ny
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = 1, ny
      do i = 1, nx
        upe(i, j) = up(i, j)
        vpe(i, j) = vp(i, j)
      enddo
    enddo
    call halo_fill_2d(upe, STAGGER_U)
    call halo_fill_2d(vpe, STAGGER_V)
  end subroutine set_upe

  subroutine fetch(io, f, nk, out)
    ! field f of object io from the host copy of the record
    integer, intent(in) :: io, f, nk
    real(rp), intent(out) :: out(obj(io)%np, nk)
    out = reshape(hb(fld_off(io, f) + 1:fld_off(io, f) + obj(io)%np * nk), (/obj(io)%np, nk/))
  end subroutine fetch

  ! ---------------------------------------------------------------- the record
  subroutine wrt_extract()
    ! After every step, after the other outputs (the CPU's do_extract_data).
    if (.not. on) return
    otime = otime + dt
    if (otime < output_period_extract) return
    if (record == nrpf_extract) then
      nfile = nfile + 1
      record = 0
      call output_fname(extract_root_name, 'bry', start_time + dt * real(iic - ntstart + 1, rp), fname)
      if (comm_rank() == 0) call create_file(fname)
    endif
    record = record + 1
    call extract_record()
    otime = 0.0_rp
  end subroutine wrt_extract

  subroutine extract_record()
    integer :: io, np, p0, nz, k, j, n, iv, kn, ns, ncid
    real(8) :: c0, c1, c2, c3, c4
    real(rp), allocatable :: hzp(:,:), hzc(:,:), hzu(:,:), hzuc(:,:), hzv(:,:), hzvc(:,:), val(:,:), vc(:,:), &
                             ui(:,:), vi(:,:), wk(:,:), h(:)
    nz = decomp%nz; kn = knew; ns = nstp
    c0 = wall_time()
    if (want_pflx) call set_upe()
    call queue_record()
    if (nbtot > 0) hb(1:nbtot) = dbuf(1:nbtot)        ! one copy (after the queued kernels)
    c1 = wall_time()
    hbuf(1:max(nbuf, 1)) = 0.0d0
    do io = 1, nobj
      np = obj(io)%np
      if (np == 0) cycle
      p0 = obj(io)%p0
      allocate(hzp(np, nz), hzc(np, nlev), val(np, nz), vc(np, nlev), h(np))
      if (mismatch) then
        ! parent layers at the point and the child's layers of the same depth
        call fetch(io, F_HZ1, nz, hzp)
        do j = 1, np
          h(j) = 0.0_rp
          do k = 1, nz
            h(j) = h(j) + hzp(j, k)
          enddo
          call child_thickness(h(j), hzc(j, :))
        enddo
      endif
      if (obj(io)%zeta) then
        call fetch(io, F_ZETA, 1, val(:, 1:1))
        call put('zeta', obj(io), val(:, 1:1), 1)
      endif
      do n = 1, ntrc
        if (.not. trc_wanted(obj(io), n)) cycle
        call fetch(io, F_TRC + n, nz, val)
        call put3(trim(trc_name(n)), obj(io), val, hzp, hzc)
      enddo
      if (.not. obj(io)%scalar) then
        allocate(ui(np, nz), vi(np, nz), wk(np, nz))
        if (obj(io)%ubar .or. obj(io)%vbar) then
          call fetch(io, F_UBAR, 1, ui(:, 1:1))
          call fetch(io, F_VBAR, 1, vi(:, 1:1))
          if (obj(io)%ubar) then
            wk(:, 1) = obj(io)%cosa * ui(:, 1) - obj(io)%sina * vi(:, 1)
            call put('ubar', obj(io), wk(:, 1:1), 1)
          endif
          if (obj(io)%vbar) then
            wk(:, 1) = obj(io)%sina * ui(:, 1) + obj(io)%cosa * vi(:, 1)
            call put('vbar', obj(io), wk(:, 1:1), 1)
          endif
        endif
        if (obj(io)%up .or. obj(io)%vp) then
          call fetch(io, F_UP, 1, ui(:, 1:1))
          call fetch(io, F_VP, 1, vi(:, 1:1))
          if (obj(io)%up) then
            wk(:, 1) = obj(io)%cosa * ui(:, 1) - obj(io)%sina * vi(:, 1)
            call put('up', obj(io), wk(:, 1:1), 1)
          endif
          if (obj(io)%vp) then
            wk(:, 1) = obj(io)%sina * ui(:, 1) + obj(io)%cosa * vi(:, 1)
            call put('vp', obj(io), wk(:, 1:1), 1)
          endif
        endif
        if (obj(io)%u .or. obj(io)%v) then
          call fetch(io, F_U, nz, ui)
          call fetch(io, F_V, nz, vi)
        endif
        if (obj(io)%u) then
          allocate(hzu(np, nz), hzuc(np, nlev))
          if (mismatch) then
            ! u-point layers: the mean of the rho columns at the point and at i-1
            call fetch(io, F_HZ2, nz, hzu)
            hzu = 0.5_rp * (hzp + hzu)
            do j = 1, np
              h(j) = 0.0_rp
              do k = 1, nz
                h(j) = h(j) + hzu(j, k)
              enddo
              call child_thickness(h(j), hzuc(j, :))
            enddo
          endif
          do k = 1, nz
            wk(:, k) = obj(io)%cosa * ui(:, k) - obj(io)%sina * vi(:, k)
          enddo
          call put3('u', obj(io), wk, hzu, hzuc)
          deallocate(hzu, hzuc)
        endif
        if (obj(io)%v) then
          allocate(hzv(np, nz), hzvc(np, nlev))
          if (mismatch) then
            call fetch(io, F_HZ3, nz, hzv)
            hzv = 0.5_rp * (hzp + hzv)
            do j = 1, np
              h(j) = 0.0_rp
              do k = 1, nz
                h(j) = h(j) + hzv(j, k)
              enddo
              call child_thickness(h(j), hzvc(j, :))
            enddo
          endif
          do k = 1, nz
            wk(:, k) = obj(io)%sina * ui(:, k) + obj(io)%cosa * vi(:, k)
          enddo
          call put3('v', obj(io), wk, hzv, hzvc)
          deallocate(hzv, hzvc)
        endif
        deallocate(ui, vi, wk)
      endif
      deallocate(hzp, hzc, val, vc, h)
    enddo
    c2 = wall_time()
    call comm_reduce_sum(max(nbuf, 1), hbuf, gbuf)
    c3 = wall_time()
    if (comm_rank() == 0) then
      call nc_open_write(trim(fname), ncid)
      ! tdays of the CPU at the write: time n+1 in days
      call nc_put_1d_rec(ncid, 'bry_time', (start_time + dt * real(iic - ntstart + 1, rp)) * sec2day, record)
      do iv = 1, nvar
        if (.not. var_wr(iv)) cycle
        call put_var(ncid, iv)
      enddo
      call nc_close(ncid)
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote extract t = ', (start_time + dt * real(iic - ntstart + 1, rp)) * sec2day, &
           ' days  rec = ', record, &
           '  file ', nfile
      c4 = wall_time()
      write(*,'(a,4f9.4,a)') 'extract timing (device+copy, host, reduce, write): ', (c1 - c0), &
           (c2 - c1), (c3 - c2), (c4 - c3), ' s'
    endif
  contains
    subroutine put3(nm, o, x, hp, hc)
      ! remap to the child layers if needed, then into the line buffer
      character(len=*), intent(in) :: nm
      type(obj_t), intent(in) :: o
      real(rp), intent(in) :: x(:,:), hp(:,:), hc(:,:)
      real(rp), allocatable :: y(:,:)
      integer :: jj
      if (mismatch) then
        allocate(y(size(x, 1), N_chd))
        do jj = 1, size(x, 1)
          call remap_src_to_grid(int(decomp%nz), hp(jj, :), x(jj, :), N_chd, hc(jj, :), y(jj, :))
        enddo
        call put(nm, o, y, N_chd)
        deallocate(y)
      else
        call put(nm, o, x, int(decomp%nz))
      endif
    end subroutine put3
  end subroutine extract_record

  subroutine put(nm, o, x, nk)
    ! object points start..start+np-1 of variable nm_<side>
    character(len=*), intent(in) :: nm
    type(obj_t), intent(in) :: o
    integer, intent(in) :: nk
    real(rp), intent(in) :: x(:,:)
    integer :: iv, p, k, off
    iv = var_index(trim(nm)//'_'//trim(side_name(o%side)))
    if (iv == 0) error stop 'extract_out: no such variable'
    if (o%start + o%np - 1 > var_len(iv)) error stop 'extract_out: object longer than the child boundary'
    off = var_off(iv)
    do k = 1, nk
      do p = 1, o%np
        hbuf(off + (k - 1) * var_len(iv) + o%start + p - 1) = x(p, k)
      enddo
    enddo
  end subroutine put

  subroutine put_var(ncid, iv)
    integer, intent(in) :: ncid, iv
    integer :: vid
    call ex_check(nf90_inq_varid(ncid, trim(var_name(iv)), vid), 'inq '//trim(var_name(iv)))
    if (var_lev(iv) == 1) then
      call ex_check(nf90_put_var(ncid, vid, gbuf(var_off(iv) + 1:var_off(iv) + var_len(iv)), &
                    start=(/1, record/), count=(/var_len(iv), 1/)), 'put '//trim(var_name(iv)))
    else
      call ex_check(nf90_put_var(ncid, vid, &
                    reshape(gbuf(var_off(iv) + 1:var_off(iv) + var_len(iv) * var_lev(iv)), (/var_len(iv), var_lev(iv)/)), &
                    start=(/1, 1, record/), count=(/var_len(iv), var_lev(iv), 1/)), 'put '//trim(var_name(iv)))
    endif
  end subroutine put_var

  ! ---------------------------------------------------------------- child vertical grid
  subroutine set_chd_scoord()
    integer :: k
    real(rp) :: ds, sc
    ds = 1.0_rp / real(N_chd, rp)
    Cs_w_chd(N_chd) = 0.0_rp
    do k = N_chd - 1, 1, -1
      sc = ds * real(k - N_chd, rp)
      call calc_cs(sc, Cs_w_chd(k))
    enddo
    Cs_w_chd(0) = -1.0_rp
  end subroutine set_chd_scoord

  subroutine calc_cs(sc, cs)
    real(rp), intent(in) :: sc
    real(rp), intent(out) :: cs
    real(rp) :: csrf
    if (theta_s_chd > 0.0_rp) then
      csrf = (1.0_rp - cosh(theta_s_chd * sc)) / (cosh(theta_s_chd) - 1.0_rp)
    else
      csrf = -sc**2
    endif
    if (theta_b_chd > 0.0_rp) then
      cs = (exp(theta_b_chd * csrf) - 1.0_rp) / (1.0_rp - exp(-theta_b_chd))
    else
      cs = csrf
    endif
  end subroutine calc_cs

  subroutine child_thickness(h_par, hz_chd)
    ! the CPU's get_child_thickness
    real(rp), intent(in) :: h_par
    real(rp), intent(out) :: hz_chd(:)
    integer :: k
    real(rp) :: ds, hinv, zw(0:N_chd), cff_w, cff1_w
    ds = 1.0_rp / real(N_chd, rp)
    hinv = 1.0_rp / (h_par + hc_chd)
    zw(0) = -h_par
    do k = 1, N_chd
      cff_w = hc_chd * ds * real(k - N_chd, rp)
      cff1_w = Cs_w_chd(k)
      zw(k) = h_par * (cff_w + cff1_w * h_par) * hinv
      hz_chd(k) = zw(k) - zw(k - 1)
    enddo
  end subroutine child_thickness

  ! ---------------------------------------------------------------- the file

  subroutine create_file(path)
    ! The CPU's create_edata_file (PARALLEL_IO): dimensions time, xi_rho,
    ! xi_u, eta_rho, eta_v, s_rho; bry_time; per side zeta, ubar, vbar,
    ! every tracer, u, v. netCDF-4, one chunk per record (the CPU writes
    ! CDF5; a CDF5 record of ~170 variables took 150 ms here, this 60 ms).
    character(len=*), intent(in) :: path
    integer :: ncid, d0, d1, d2, d3, d4, d5, s, n, id, dt_(4), du_(4), dv_(4)
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d0 = nc_def_dim(ncid, 'time', 0)
    d1 = nc_def_dim(ncid, 'xi_rho', LLm_chd)
    d2 = nc_def_dim(ncid, 'xi_u', LLm_chd - 1)
    d3 = nc_def_dim(ncid, 'eta_rho', MMm_chd)
    d4 = nc_def_dim(ncid, 'eta_v', MMm_chd - 1)
    d5 = nc_def_dim(ncid, 's_rho', N_chd)
    id = nc_def_var(ncid, 'bry_time', (/d0/))
    call nc_put_att_text(ncid, 'bry_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'bry_time', 'units', 'days')
    dt_ = (/d1, d1, d3, d3/); du_ = (/d2, d2, d3, d3/); dv_ = (/d1, d1, d4, d4/)
    do s = 1, 4
      if (.not. side_on(s)) cycle
      call def('zeta_'//trim(side_name(s)), (/dt_(s), d0/), 'free-surface elevation', 'meter')
      call def('ubar_'//trim(side_name(s)), (/du_(s), d0/), 'vertically averaged u-momentum component', 'meter second-1')
      call def('vbar_'//trim(side_name(s)), (/dv_(s), d0/), 'vertically averaged v-momentum component', 'meter second-1')
      do n = 1, ntrc
        call def(trim(trc_name(n))//'_'//trim(side_name(s)), (/dt_(s), d5, d0/), trim(trc_lname(n)), trim(trc_units(n)))
      enddo
      call def('u_'//trim(side_name(s)), (/du_(s), d5, d0/), 'u-momentum component', 'meter second-1')
      call def('v_'//trim(side_name(s)), (/dv_(s), d5, d0/), 'v-momentum component', 'meter second-1')
      ! the CPU's names and attributes (the values are m^4 s^-3: W/m / rho0)
      if (side_up(s)) call def('up_'//trim(side_name(s)), (/du_(s), d0/), 'u-momentum pressure flux', &
                               'meter3 second-2')
      if (side_vp(s)) call def('vp_'//trim(side_name(s)), (/dv_(s), d0/), 'v-momentum pressure flux', &
                               'meter3 second-2')
    enddo
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu child boundary extraction')
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created extract file: ', trim(path)
  contains
    subroutine def(nm, dims, ln, un)
      character(len=*), intent(in) :: nm, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      vid = nc_def_var(ncid, nm, dims)
      call nc_put_att_text(ncid, nm, 'long_name', ln)
      call nc_put_att_text(ncid, nm, 'units', un)
    end subroutine def
  end subroutine create_file

end module extract_out

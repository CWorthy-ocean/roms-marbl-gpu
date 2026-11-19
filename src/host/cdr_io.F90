module cdr_io
  ! CDR releases (UCLA cdr_frc) on the host: the CDR_FRC_SETTINGS namelist,
  ! the release cells and vertical profiles, the flux records and their time
  ! weights. Parameterized releases with tracer fluxes (cdr_trcflx) or with
  ! water (cdr_volume: cdr_volume [m3/s] and cdr_tracer concentrations,
  ! temperature and salinity included), depth profiles
  ! (cdr_forcing_depth_profiles) and 3D forcing (cdr_forcing_3d). Volume
  ! releases are also set at the predictor's forcing time (set_cdr_pred):
  ! the CPU's predictor and first corrector omega use that volume.
  !
  ! 3D forcing (CPU init_cdr_frc_3d, set_cdr_frc): cdr_trcflx_3d_ALK and
  ! cdr_trcflx_3d_DIC (cdr_time, s_rho, eta_rho, xi_rho) [mmol/s per cell]
  ! from the forcing files (frcfiles), cataloged in the CPU's file order;
  ! each tile reads its interior window of a record; MARBL's ALK and DIC.
  !
  ! Depth profiles (CPU init_cdr_frc_dp, create_cdr_vertical_structure):
  ! release r (cdr_lon, cdr_lat; ncdr_prof of them) goes to its nearest
  ! cell; cdr_trcflx_profile(ncdr_prof, tracer, s_rho, time) [mmol/s per
  ! source layer] on the source layers cdr_layer_thickness(ncdr_prof,
  ! s_rho, time), its tracers named by cdr_trc_name (or ALK and DIC). Every
  ! corrector the two are interpolated in time, and each mapped tracer's
  ! flux per metre is remapped conservatively (piecewise parabolic, the
  ! CPU's vertical_remapping) onto the cell's current layers Hz and
  ! multiplied by Hz; the remap runs on the host (a few columns).
  !
  ! Release r (cdr_lon, cdr_lat, cdr_dep, cdr_hsc, cdr_vsc) as the CPU
  ! (find_release_locations, create_cdr_vertical_structure):
  ! - the haversine distance from every interior cell, land at 1e8 m with
  !   cdr_relocate_to_wet_pts; the nearest cell is the first minimum;
  ! - cdr_hsc = 0: the nearest cell with weight 1; otherwise the wet cells
  !   with frac = exp(-(dist/hsc)^2) > 1e-3 (the nearest cell with weight 1
  !   if there are none);
  ! - in the vertical, on the rest-state grid (z_r0, Hz0 with zeta = 0):
  !   cdr_vsc = 0, all at the level nearest to -cdr_dep; otherwise
  !   frac*exp(-((z_r0+dep)/vsc)^2)*Hz0/(vsc*sqrt(pi));
  ! - normalized so the weights of the release sum to 1 over the domain.
  ! The CPU finds the cells across its MPI ranks and sums the weights in one
  ! fixed global order (each cell's column, then the cells in (j, i) order);
  ! here every rank does it on the global grid in that order, so the
  ! weights are the same for every GPU tiling and match the CPU's.
  !
  ! cdr_trcflx(ncdr, ntracers, time) [mmol/s] has the CPU's tracer axis
  ! (T, S, generic, CDR_LITE, MARBL; passive_mod cpu_tracer_index) and must
  ! carry no temperature or salinity flux (the CPU's check_no_ts_flux). The
  ! records are bracketed and weighted as the CPU's set_frc_data at the
  ! corrector's forcing time (n+1/2), the only one the tracer source uses.
  use kinds
  use nc_io, only: iomsg_text
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use passive_mod, only: ntp, cpu_tracer_index, passive_name
  use cdr_src_mod
  use comm_mod, only: comm_rank
  use frc_io, only: case_frcfiles, case_nfrc
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use cudafor
#endif
  use netcdf
  implicit none
  private
  public :: read_cdr_namelist, init_cdr, set_cdr, set_cdr_pred
  public :: remap_src_to_grid      ! the CPU's flux-conserving PPM remap (also extract_out)
  public :: cdr_source

  logical :: cdr_source = .false., cdr_forcing_depth_profiles = .false., cdr_forcing_3d = .false.
  logical :: cdr_forcing_parameterized = .false., cdr_time_interpolation = .false.
  logical :: cdr_relocate_to_wet_pts = .false., cdr_volume = .false.
  character(len=256) :: cdr_file = 'cdr_release.nc'
  integer :: cdr_ncdr_parm = 0, cdr_nz_chd = 0

  ! flux records
  integer :: ncid_cdr = -1, vid_flx = -1, nrec = 0, it1 = 0, it2 = 0, ntr_file = 0, ncdr = 0
  real(rp), allocatable :: rec_t(:)              ! cdr_time (days)
  real(rp), allocatable :: frec(:,:), fpas(:,:)  ! one record (ncdr, ntracers), mapped (ntp, ncdr)
  real(rp), allocatable :: vrec(:), tsrec(:,:)    ! volume releases: volume (ncdr), T/S (2, ncdr)
  integer :: vid_vol = -1
  integer(ip), allocatable :: tmap(:)            ! passive tracer -> CPU tracer index
  ! depth profiles
  integer :: n_src = 0, ntrc_dp = 0, ndpm = 0, vid_hz = -1, nq_loc = 0, ncc_loc = 0
  integer, allocatable :: dp_col(:)               ! file tracer column of mapped tracer a
  integer(ip), allocatable :: dp_itr(:)           ! its passive index
  integer, allocatable :: q_rel(:), q_cell(:)     ! local profile q: release, cell
  real(rp), allocatable :: dhz(:,:,:), dfx(:,:,:,:)   ! records (ncdr, n_src, slot), (ncdr, ntrc, n_src, slot)
  ! 3D forcing: catalog record -> forcing file, record in that file
  integer, allocatable :: c3_file(:), c3_rec(:)

  real(rp), parameter :: pi = 3.14159265358979323_rp, deg2rad = pi / 180.0_rp   ! the CPU's
  real(rp), parameter :: radius = 6.371e6_rp
  ! Release-cell threshold on frac: the CPU's single-precision literal 1e-3 (cdr_frc.F90:705,770),
  ! which promotes to 0.0010000000475, not 1.0e-3 in double.
  real(rp), parameter :: cdr_frac_min = real(1.0e-3, rp)

contains

  subroutine read_cdr_namelist(unit)
    ! &cdr_frc_settings as the CPU. A namelist without the group has no
    ! releases.
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    namelist /cdr_frc_settings/ cdr_file, cdr_ncdr_parm, cdr_nz_chd, cdr_source, &
         cdr_forcing_depth_profiles, cdr_forcing_3d, cdr_forcing_parameterized, &
         cdr_time_interpolation, cdr_relocate_to_wet_pts, cdr_volume
    cdr_source = .false.
    rewind(unit)
    read(unit, nml=cdr_frc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'CDR_FRC_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_cdr_namelist'
    endif
    if (ios < 0) cdr_source = .false.
    if (.not. cdr_source) return
    if (count([cdr_forcing_parameterized, cdr_forcing_depth_profiles, cdr_forcing_3d]) /= 1) &
      error stop 'read_cdr_namelist: cdr_source needs exactly one of cdr_forcing_parameterized, '// &
                 'cdr_forcing_depth_profiles and cdr_forcing_3d'
    if (cdr_volume .and. .not. cdr_forcing_parameterized) &
      error stop 'read_cdr_namelist: cdr_volume = .true. requires cdr_forcing_parameterized = .true.'
    if (cdr_forcing_parameterized .and. cdr_ncdr_parm <= 0) error stop 'read_cdr_namelist: cdr_ncdr_parm must be > 0'
  end subroutine read_cdr_namelist

  subroutine init_cdr(grdname)
    ! Release cells and profiles of this tile, the time catalog, and the
    ! device lists (cdr_set_lists). After the passive tracer list is final.
    character(len=*), intent(in) :: grdname
    integer :: ncg, gx, gy, i, j, k, r, nzl, ierr, vid, d(3), nd, ntr_d, ncdr_d, nrec_d
    integer :: imin, jmin, nq, ic, iarg, ntot, q0
    integer(ip) :: itp
    real(rp), allocatable :: lon(:,:), lat(:,:), mask(:,:), hh(:,:), dist(:,:), frac(:,:), fr_all(:,:,:)
    real(rp), allocatable :: rlon(:), rlat(:), rdep(:), rhsc(:), rvsc(:), tsf(:,:,:)
    real(rp), allocatable :: zr0(:), hz0(:), prf(:), w(:,:), wq(:,:)
    integer, allocatable :: ci(:), cj(:), cr(:), nqr(:), cellof(:,:), cnt(:)
    integer(ip), allocatable :: oc_i(:), oc_j(:), oc_col(:), oc_p0(:), oq_rel(:)
    real(rp) :: dmin, vint, arg, gsum, colw, amin
    integer :: nq_r, q, c, li, lj, nloc_c
    if (.not. cdr_source) return
    nzl = int(decomp%nz)
    if (cdr_forcing_3d) then
      call init_3d()
      return
    endif
    call cdr_check(nf90_open(trim(cdr_file), nf90_nowrite, ncid_cdr), 'open '//trim(cdr_file))
    if (cdr_forcing_depth_profiles) then
      call init_dp_header()
      allocate(rlon(ncdr), rlat(ncdr), rdep(ncdr), rhsc(ncdr), rvsc(ncdr))
      call get_1d('cdr_lon', rlon); call get_1d('cdr_lat', rlat)
      rdep = 0.0_rp; rhsc = 0.0_rp; rvsc = 0.0_rp     ! single-point releases
    else
    ncdr = cdr_ncdr_parm

    ! --- the release file: locations, scales, tracer axis, times
    allocate(rlon(ncdr), rlat(ncdr), rdep(ncdr), rhsc(ncdr), rvsc(ncdr))
    call get_1d('cdr_lon', rlon); call get_1d('cdr_lat', rlat); call get_1d('cdr_dep', rdep)
    call get_1d('cdr_hsc', rhsc); call get_1d('cdr_vsc', rvsc)
    if (cdr_volume) then
      ! water and its tracer concentrations (temperature and salinity too)
      call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_tracer', vid_flx), 'var cdr_tracer')
      call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_volume', vid_vol), 'var cdr_volume')
    else
      call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_trcflx', vid_flx), 'var cdr_trcflx')
    endif
    call cdr_check(nf90_inquire_variable(ncid_cdr, vid_flx, ndims=nd, dimids=d), 'inquire cdr_trcflx/cdr_tracer')
    if (nd /= 3) error stop 'init_cdr: cdr_trcflx/cdr_tracer must have dimensions (ncdr, ntracers, time)'
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(1), len=ncdr_d), 'dim 1 of cdr_trcflx')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(2), len=ntr_d), 'dim 2 of cdr_trcflx')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(3), len=nrec_d), 'dim 3 of cdr_trcflx')
    if (ncdr_d < ncdr) error stop 'init_cdr: cdr_trcflx has fewer releases than cdr_ncdr_parm'
    ntr_file = ntr_d; nrec = nrec_d
    if (ntr_file /= int(nt + ntp)) then
      write(*,'(a,i0,a,i0,a)') 'init_cdr: cdr_trcflx has ', ntr_file, ' tracers, the model ', int(nt + ntp), &
           ' (T, S, the passive tracers, in the CPU order)'
      error stop 'init_cdr: cdr_trcflx tracer axis'
    endif
    allocate(rec_t(nrec))
    call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_time', vid), 'var cdr_time')
    call cdr_check(nf90_get_var(ncid_cdr, vid, rec_t), 'get cdr_time')
    do i = 2, nrec
      if (rec_t(i) < rec_t(i - 1)) error stop 'init_cdr: cdr_time must increase'
    enddo
    ! the CPU's check_no_ts_flux: tracer-flux releases carry no temp/salt
    if (.not. cdr_volume) then
    allocate(tsf(ncdr, 2, nrec))
    call cdr_check(nf90_get_var(ncid_cdr, vid_flx, tsf, start=[1, 1, 1], count=[ncdr, 2, nrec]), 'get cdr_trcflx T/S')
    do r = 1, ncdr
      if (any(tsf(r, :, :) /= 0.0_rp)) then
        write(*,'(a,i0,a)') 'init_cdr: CDR release ', r, ' has a nonzero temp/salt flux in cdr_trcflx '// &
             '(temperature and salinity can only be released with water: cdr_volume)'
        error stop 'init_cdr: temp/salt flux'
      endif
    enddo
    deallocate(tsf)
    endif
    allocate(tmap(max(ntp, 1_ip)), frec(ncdr, ntr_file), fpas(max(ntp, 1_ip), ncdr))
    allocate(vrec(ncdr), tsrec(2, ncdr))
    do itp = 1, ntp
      tmap(itp) = cpu_tracer_index(itp, .true.)
    enddo
    endif   ! depth profiles / parameterized

    ! --- the global grid: lon, lat, mask, h (interior cells 1..gnx, 1..gny
    ! are array indices 2..gnx+1, 2..gny+1)
    gx = int(decomp%gnx) + 2; gy = int(decomp%gny) + 2
    allocate(lon(gx, gy), lat(gx, gy), mask(gx, gy), hh(gx, gy), dist(gx, gy), frac(gx, gy))
    call cdr_check(nf90_open(trim(grdname), nf90_nowrite, ncg), 'open grid '//trim(grdname))
    call get_2d(ncg, 'lon_rho', lon); call get_2d(ncg, 'lat_rho', lat)
    call get_2d(ncg, 'mask_rho', mask); call get_2d(ncg, 'h', hh)
    ierr = nf90_close(ncg)

    ! --- weights frac of every release on the global grid (pass 1)
    allocate(fr_all(gx, gy, ncdr), nqr(ncdr))
    do r = 1, ncdr
      call haversine(lon, lat, rlon(r), rlat(r), dist, gx, gy)
      if (cdr_relocate_to_wet_pts) then
        do j = 2, gy - 1
          do i = 2, gx - 1
            if (mask(i, j) == 0.0_rp) dist(i, j) = 1.0e8_rp
          enddo
        enddo
      endif
      ! first minimum in array order (the CPU's MINLOC within a rank)
      dmin = huge(dmin); imin = 2; jmin = 2
      do j = 2, gy - 1
        do i = 2, gx - 1
          if (dist(i, j) < dmin) then
            dmin = dist(i, j); imin = i; jmin = j
          endif
        enddo
      enddo
      frac = 0.0_rp
      nq_r = 0
      if (rhsc(r) == 0.0_rp) then
        frac(imin, jmin) = 1.0_rp
      else
        do j = 2, gy - 1
          do i = 2, gx - 1
            frac(i, j) = exp(-(dist(i, j) / rhsc(r))**2)
            if (frac(i, j) > cdr_frac_min .and. mask(i, j) > 0.0_rp) nq_r = nq_r + 1
          enddo
        enddo
        if (nq_r == 0) then
          frac = 0.0_rp
          frac(imin, jmin) = 1.0_rp
        endif
      endif
      if (.not. cdr_relocate_to_wet_pts .and. frac(imin, jmin) == 1.0_rp .and. mask(imin, jmin) == 0.0_rp) then
        write(*,'(a,i0,a)') 'init_cdr: CDR release ', r, ' is on land'
        error stop 'init_cdr: release on land'
      endif
      if (comm_rank() == 0) write(*,'(a,i0,a,f12.3,a,2i6,a,2f10.4,a,2f10.4)') 'cdr: release ', r, &
           ' nearest cell at ', dmin, ' m, global (i, j)', imin - 1, jmin - 1, &
           ' lon/lat', lon(imin, jmin), lat(imin, jmin), ' intended', rlon(r), rlat(r)
      nqr(r) = count(frac(2:gx-1, 2:gy-1) > cdr_frac_min .and. mask(2:gx-1, 2:gy-1) > 0.0_rp)
      fr_all(:, :, r) = frac
    enddo

    ! --- profiles (pass 2): release-major, then j outer, i inner as the
    ! CPU's loop within a rank; then each release normalized to sum 1
    ntot = sum(nqr)
    allocate(ci(ntot), cj(ntot), cr(ntot), w(nzl, ntot))
    allocate(zr0(nzl), hz0(nzl), prf(nzl))
    q = 0
    do r = 1, ncdr
      q0 = q
      do j = 2, gy - 1
        do i = 2, gx - 1
          if (.not. (fr_all(i, j, r) > cdr_frac_min .and. mask(i, j) > 0.0_rp)) cycle
          call rest_grid(hh(i, j), nzl, zr0, hz0)
          prf = 0.0_rp
          if (cdr_forcing_depth_profiles) then
            ! weights from the remap, every corrector (set_dp_weights)
            continue
          else if (rvsc(r) == 0.0_rp) then
            amin = huge(amin); iarg = 1
            do k = 1, nzl
              if (abs(zr0(k) + rdep(r)) < amin) then
                amin = abs(zr0(k) + rdep(r)); iarg = k
              endif
            enddo
            prf(iarg) = fr_all(i, j, r)
          else
            do k = 1, nzl
              arg = ((zr0(k) + rdep(r)) / rvsc(r))**2
              prf(k) = exp(-arg) * hz0(k)
            enddo
            vint = rvsc(r) * sqrt(pi)
            prf = fr_all(i, j, r) * prf / vint
          endif
          q = q + 1
          ci(q) = i; cj(q) = j; cr(q) = r; w(:, q) = prf
        enddo
      enddo
      ! the release's weights sum to 1: each cell's column sum (k ascending),
      ! then the cells in global (j, i) order, as the CPU's release_weight_sum
      ! (ucla-roms#TBD; before it, the CPU summed per-rank partial sums)
      gsum = 0.0_rp
      do c = q0 + 1, q
        colw = 0.0_rp
        do k = 1, nzl
          colw = colw + w(k, c)
        enddo
        gsum = gsum + colw
      enddo
      ! depth profiles are not normalized (the CPU normalizes only
      ! parameterized releases)
      if (gsum /= 0.0_rp .and. .not. cdr_forcing_depth_profiles) w(:, q0 + 1:q) = w(:, q0 + 1:q) / gsum
    enddo

    ! --- this tile's cells (interior), each with its profiles in release
    ! order: profiles come release-major, so appending per cell keeps it
    allocate(cellof(decomp%nx, decomp%ny), cnt(decomp%nx * decomp%ny + 1))
    cellof = 0; nloc_c = 0; nq = 0
    do c = 1, ntot
      li = ci(c) - 1 - int(decomp%i0) + 1; lj = cj(c) - 1 - int(decomp%j0) + 1
      if (li < 1 .or. li > decomp%nx .or. lj < 1 .or. lj > decomp%ny) cycle
      nq = nq + 1
      if (cellof(li, lj) == 0) then
        nloc_c = nloc_c + 1; cellof(li, lj) = nloc_c
      endif
    enddo
    allocate(oc_i(max(nloc_c, 1)), oc_j(max(nloc_c, 1)), oc_col(max(nloc_c, 1)), oc_p0(nloc_c + 1))
    allocate(oq_rel(max(nq, 1)), wq(nzl, max(nq, 1)))
    cnt = 0
    do c = 1, ntot
      li = ci(c) - 1 - int(decomp%i0) + 1; lj = cj(c) - 1 - int(decomp%j0) + 1
      if (li < 1 .or. li > decomp%nx .or. lj < 1 .or. lj > decomp%ny) cycle
      ic = cellof(li, lj)
      cnt(ic) = cnt(ic) + 1
      oc_i(ic) = li; oc_j(ic) = lj; oc_col(ic) = tpcol_h(li, lj)
    enddo
    oc_p0(1) = 1
    do ic = 1, nloc_c
      oc_p0(ic + 1) = oc_p0(ic) + cnt(ic)
    enddo
    cnt = 0
    do c = 1, ntot
      li = ci(c) - 1 - int(decomp%i0) + 1; lj = cj(c) - 1 - int(decomp%j0) + 1
      if (li < 1 .or. li > decomp%nx .or. lj < 1 .or. lj > decomp%ny) cycle
      ic = cellof(li, lj)
      q = oc_p0(ic) + cnt(ic)
      cnt(ic) = cnt(ic) + 1
      oq_rel(q) = cr(c); wq(:, q) = w(:, c)
    enddo
    call cdr_set_lists(int(nloc_c, ip), oc_i, oc_j, oc_col, oc_p0(1:nloc_c + 1), int(nq, ip), oq_rel, wq, &
                       int(ncdr, ip), ntp)
    if (cdr_volume) call set_vol_lists(nloc_c, nq, oc_i, oc_j, oc_p0, wq)
    if (cdr_forcing_depth_profiles) then
      ! local profiles: release and cell, for the remap
      nq_loc = nq; ncc_loc = nloc_c
      allocate(q_rel(max(nq, 1)), q_cell(max(nq, 1)))
      do ic = 1, nloc_c
        do q = oc_p0(ic), oc_p0(ic + 1) - 1
          q_rel(q) = oq_rel(q); q_cell(q) = ic
        enddo
      enddo
      call cdr_set_dp(int(ndpm, ip), dp_itr)
    endif
    write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a)') 'cdr: rank ', comm_rank(), '  releases ', ncdr, '  cells ', nloc_c, &
         '  profiles ', nq, ' (of ', ntot, ' in the domain)'
    deallocate(lon, lat, mask, hh, dist, frac, fr_all, nqr, ci, cj, cr, w, zr0, hz0, prf, wq, cellof, cnt)
    deallocate(oc_i, oc_j, oc_col, oc_p0, oq_rel, rlon, rlat, rdep, rhsc, rvsc)
  end subroutine init_cdr

  subroutine set_cdr()
    ! Flux records and weights at the corrector's forcing time (frc_slot 1,
    ! n+1/2), as the CPU's set_frc_data: the bracket moves on when the upper
    ! record is earlier than the model time; c1 = (t2-t)/(t2-t1),
    ! c2 = (t-t1)/(t2-t1), or the earlier record held without
    ! cdr_time_interpolation.
    real(rp) :: tneed, t1, t2, c1, c2
    integer :: rec, i, old1, old2
    if (.not. cdr_source) return
    tneed = frc_modtime()
    if (nrec >= 2) then
      if (it1 == 0 .or. .not. (rec_t(it2) >= tneed .and. rec_t(it1) <= tneed)) then
        rec = 1
        do i = 1, nrec - 1
          if (rec_t(i) <= tneed .and. rec_t(i + 1) >= tneed) then
            rec = i
            exit
          endif
          if (rec_t(i) <= tneed) rec = i
        enddo
        rec = max(1, min(rec, nrec - 1))
        if (rec /= it1 .or. rec + 1 /= it2) then
          old1 = it1; old2 = it2
          it1 = rec; it2 = rec + 1
          if (it1 /= old1) then
            if (it1 == old2) then
              if (cdr_forcing_depth_profiles) then
                dhz(:, :, 1) = dhz(:, :, 2); dfx(:, :, :, 1) = dfx(:, :, :, 2)
              else if (cdr_forcing_3d) then
                call copy_3d_upper()
              else
                call cdr_copy_upper()
              endif
            else
              call load_record(it1, 1)
            endif
          endif
          if (it2 /= old2) call load_record(it2, 2)
        endif
      endif
      t1 = rec_t(it1); t2 = rec_t(it2)
      if (t1 > tneed + dt * sec2day .or. t2 < tneed - dt * sec2day) then
        write(*,'(a,3es16.8)') 'set_cdr: CDR records do not cover the model time ', t1, t2, tneed
        error stop 'set_cdr: time interpolation error'
      endif
      if (cdr_time_interpolation) then
        c1 = (t2 - tneed) / (t2 - t1)
        c2 = (tneed - t1) / (t2 - t1)
      else
        c1 = 1.0_rp; c2 = 0.0_rp
      endif
    else
      if (it1 == 0) then
        it1 = 1; it2 = 1
        call load_record(1, 1)
        call load_record(1, 2)
      endif
      c1 = 1.0_rp; c2 = 0.0_rp
    endif
    call cdr_set_weights(c1, c2)
    if (cdr_forcing_depth_profiles) call set_dp_weights(c1, c2)
  end subroutine set_cdr

  subroutine init_dp_header()
    ! Depth-profile file (CPU init_cdr_frc_dp, map_cdr_dp_tracers): ncdr_prof
    ! releases on s_rho source layers; the tracer axis mapped to passive
    ! tracers by cdr_trc_name, or ALK and DIC for two unnamed columns.
    integer :: dimid, vid, nd, d(4), xt, nchar, nnam, i, n, vtime
    integer(ip) :: itp
    character(len=:), allocatable :: names
    character(len=64) :: padded, tname
    call cdr_check(nf90_inq_dimid(ncid_cdr, 'ncdr_prof', dimid), 'dimension ncdr_prof')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, dimid, len=ncdr), 'ncdr_prof')
    call cdr_check(nf90_inq_dimid(ncid_cdr, 's_rho', dimid), 'dimension s_rho')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, dimid, len=n_src), 's_rho')
    if (cdr_nz_chd > 0 .and. cdr_nz_chd /= n_src) then
      write(*,'(a,i0,a,i0,a)') 'init_cdr: cdr_nz_chd = ', cdr_nz_chd, ' does not match the ', n_src, &
           ' layers (s_rho) of the CDR forcing file'
      error stop 'init_cdr: cdr_nz_chd'
    endif
    call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_trcflx_profile', vid_flx), 'var cdr_trcflx_profile')
    call cdr_check(nf90_inquire_variable(ncid_cdr, vid_flx, ndims=nd, dimids=d), 'inquire cdr_trcflx_profile')
    if (nd /= 4) error stop 'init_cdr: cdr_trcflx_profile must have dimensions (cdr_time, s_rho, tracer, ncdr_prof)'
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(2), len=ntrc_dp), 'tracer axis')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(4), len=nrec), 'time axis')
    call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_layer_thickness', vid_hz), 'var cdr_layer_thickness')
    allocate(rec_t(nrec))
    call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_time', vtime), 'var cdr_time')
    call cdr_check(nf90_get_var(ncid_cdr, vtime, rec_t), 'get cdr_time')
    do i = 2, nrec
      if (rec_t(i) < rec_t(i - 1)) error stop 'init_cdr: cdr_time must increase'
    enddo
    allocate(dp_col(ntrc_dp), dp_itr(ntrc_dp))
    ndpm = 0
    if (nf90_inq_varid(ncid_cdr, 'cdr_trc_name', vid) /= nf90_noerr) then
      if (ntrc_dp /= 2) error stop 'init_cdr: cdr_trcflx_profile has other than 2 tracer columns but no cdr_trc_name'
      do i = 1, 2
        tname = merge('ALK', 'DIC', i == 1)
        itp = find_passive(tname)
        if (itp == 0) error stop 'init_cdr: depth profiles without cdr_trc_name are ALK and DIC, which need MARBL'
        ndpm = ndpm + 1; dp_col(ndpm) = i; dp_itr(ndpm) = itp
      enddo
      if (comm_rank() == 0) write(*,'(a)') 'cdr: no cdr_trc_name in the depth-profile file; its two columns '// &
           'are applied to ALK and DIC'
    else
      call cdr_check(nf90_inquire_variable(ncid_cdr, vid, xtype=xt, ndims=nd, dimids=d), 'inquire cdr_trc_name')
      if (xt /= nf90_char .or. nd /= 2) error stop 'init_cdr: cdr_trc_name must be NC_CHAR (tracer, nchar)'
      call cdr_check(nf90_inquire_dimension(ncid_cdr, d(1), len=nchar), 'nchar')
      call cdr_check(nf90_inquire_dimension(ncid_cdr, d(2), len=nnam), 'names')
      if (nnam /= ntrc_dp) error stop 'init_cdr: cdr_trc_name and cdr_trcflx_profile differ in tracers'
      allocate(character(len=nchar * ntrc_dp) :: names)
      call cdr_check(nf90_get_var(ncid_cdr, vid, names, start=[1, 1], count=[nchar, ntrc_dp]), 'get cdr_trc_name')
      do i = 1, ntrc_dp
        padded = names((i - 1) * nchar + 1:i * nchar)
        n = scan(padded//achar(0), achar(0)) - 1
        tname = padded(1:n)
        if (trim(tname) == 'temp' .or. trim(tname) == 'salt') then
          write(*,'(3a)') 'init_cdr: cdr_trc_name names ', trim(tname), &
               ', but temperature and salinity can only be released with water (cdr_volume)'
          error stop 'init_cdr: temp/salt in depth profiles'
        endif
        itp = find_passive(tname)
        if (itp == 0) then
          if (comm_rank() == 0) write(*,'(3a)') 'cdr: skipping ', trim(tname), ' fluxes (no such tracer in this run)'
          cycle
        endif
        if (any(dp_itr(1:ndpm) == itp)) then
          write(*,'(3a)') 'init_cdr: cdr_trc_name names ', trim(tname), ' more than once'
          error stop 'init_cdr: duplicate tracer'
        endif
        ndpm = ndpm + 1; dp_col(ndpm) = i; dp_itr(ndpm) = itp
        if (comm_rank() == 0) write(*,'(3a)') 'cdr: depth-profile fluxes for ', trim(tname), ' are applied'
      enddo
      if (ndpm == 0) error stop 'init_cdr: none of the tracers in cdr_trc_name exist in this run'
    endif
    allocate(dhz(ncdr, n_src, 2), dfx(ncdr, ntrc_dp, n_src, 2))
    dhz = 0.0_rp; dfx = 0.0_rp
  end subroutine init_dp_header

  subroutine init_3d()
    ! The record catalog of cdr_trcflx_3d_ALK/DIC over the forcing files in
    ! the CPU's order (files in list order; records later than the last one
    ! kept; the search ends at a file with the variable but no cdr_time).
    integer :: f, ncid, vid, vt, d(4), nd, n, k, dimlen(4), i
    real(rp), allocatable :: tl(:), tall(:)
    integer, allocatable :: fall(:), rall(:)
    integer(ip) :: ia, id
    ia = find_passive('ALK'); id = find_passive('DIC')
    if (ia == 0 .or. id == 0) error stop 'init_cdr: cdr_forcing_3d forces MARBL''s ALK and DIC and needs MARBL'
    allocate(tall(100000), fall(100000), rall(100000))
    nrec = 0
    do f = 1, case_nfrc
      call cdr_check(nf90_open(trim(case_frcfiles(f)), nf90_nowrite, ncid), 'open '//trim(case_frcfiles(f)))
      if (nf90_inq_varid(ncid, 'cdr_trcflx_3d_ALK', vid) == nf90_noerr) then
        if (nf90_inq_varid(ncid, 'cdr_trcflx_3d_DIC', vt) /= nf90_noerr) &
          error stop 'init_cdr: a forcing file has cdr_trcflx_3d_ALK but not cdr_trcflx_3d_DIC'
        call cdr_check(nf90_inquire_variable(ncid, vid, ndims=nd, dimids=d), 'inquire cdr_trcflx_3d_ALK')
        if (nd /= 4) error stop 'init_cdr: cdr_trcflx_3d_ALK must be (cdr_time, s_rho, eta_rho, xi_rho)'
        do k = 1, 4
          call cdr_check(nf90_inquire_dimension(ncid, d(k), len=dimlen(k)), 'dims of cdr_trcflx_3d_ALK')
        enddo
        if (dimlen(1) /= int(decomp%gnx) + 2 .or. dimlen(2) /= int(decomp%gny) + 2 .or. dimlen(3) /= int(decomp%nz)) then
          write(*,'(a,3i6)') 'init_cdr: cdr_trcflx_3d_ALK (xi_rho, eta_rho, s_rho) =', dimlen(1:3)
          error stop 'init_cdr: 3D CDR forcing must be on the model''s rho grid and levels'
        endif
        if (nf90_inq_varid(ncid, 'cdr_time', vt) /= nf90_noerr) then
          if (nrec == 0) error stop 'init_cdr: cdr_trcflx_3d_ALK first found without cdr_time'
          i = nf90_close(ncid)
          exit
        endif
        n = dimlen(4)
        allocate(tl(n))
        call cdr_check(nf90_get_var(ncid, vt, tl), 'get cdr_time')
        do k = 1, n
          if (nrec > 0) then
            if (.not. (tl(k) > tall(nrec))) cycle
          endif
          nrec = nrec + 1
          tall(nrec) = tl(k); fall(nrec) = f; rall(nrec) = k
        enddo
        deallocate(tl)
      endif
      i = nf90_close(ncid)
    enddo
    if (nrec == 0) error stop 'init_cdr: no cdr_trcflx_3d_ALK in the forcing files'
    allocate(rec_t(nrec), c3_file(nrec), c3_rec(nrec))
    rec_t = tall(1:nrec); c3_file = fall(1:nrec); c3_rec = rall(1:nrec)
    deallocate(tall, fall, rall)
    call cdr_set_3d(ia, id)
    if (comm_rank() == 0) write(*,'(a,i0,a,2es14.6)') 'cdr: 3D forcing (ALK, DIC), ', nrec, ' records, days ', &
         rec_t(1), rec_t(nrec)
  end subroutine init_3d

  subroutine load_3d(idx, slot)
    ! This tile's interior of record idx (ALK, DIC) into slot `slot`.
    integer, intent(in) :: idx, slot
    integer :: ncid, vid, m, ierr, nx, ny, nz
    real(rp), allocatable :: buf(:,:,:)
    character(len=24) :: vn
    nx = int(decomp%nx); ny = int(decomp%ny); nz = int(decomp%nz)
    allocate(buf(nx, ny, nz))
    call cdr_check(nf90_open(trim(case_frcfiles(c3_file(idx))), nf90_nowrite, ncid), 'open 3D CDR forcing')
    call sync_arch()                  ! queued kernels may still read the slot
    do m = 1, 2
      vn = merge('cdr_trcflx_3d_ALK', 'cdr_trcflx_3d_DIC', m == 1)
      call cdr_check(nf90_inq_varid(ncid, trim(vn), vid), 'var '//trim(vn))
      call cdr_check(nf90_get_var(ncid, vid, buf, start=[int(decomp%i0) + 1, int(decomp%j0) + 1, 1, c3_rec(idx)], &
                     count=[nx, ny, nz, 1]), 'get '//trim(vn))
      where (buf /= buf) buf = 0.0_rp           ! NaN (fill) -> 0
#ifdef USE_CUDA
      ierr = cudaMemcpy(cdr_3d_buf(:, :, :, m, slot), buf, nx * ny * nz, cudaMemcpyHostToDevice)
      if (ierr /= 0) error stop 'load_3d: cudaMemcpy'
#else
      cdr_3d_buf(:, :, :, m, slot) = buf
#endif
    enddo
    ierr = nf90_close(ncid)
    deallocate(buf)
  end subroutine load_3d

  subroutine copy_3d_upper()
    integer :: ierr
    call sync_arch()
#ifdef USE_CUDA
    ierr = cudaMemcpy(cdr_3d_buf(:, :, :, :, 1), cdr_3d_buf(:, :, :, :, 2), size(cdr_3d_buf(:, :, :, :, 1)), &
                      cudaMemcpyDeviceToDevice)
    if (ierr /= 0) error stop 'copy_3d_upper: cudaMemcpy'
#else
    cdr_3d_buf(:, :, :, :, 1) = cdr_3d_buf(:, :, :, :, 2)
#endif
  end subroutine copy_3d_upper

  integer(ip) function find_passive(name) result(itp)
    character(len=*), intent(in) :: name
    integer(ip) :: m
    itp = 0
    do m = 1, ntp
      if (trim(passive_name(m)) == trim(name)) then
        itp = m
        return
      endif
    enddo
  end function find_passive

  subroutine set_dp_weights(c1, c2)
    ! This corrector's depth-profile weights: the records interpolated in
    ! time (set_frc_data: c1*v1 + c2*v2, or v1 held), the flux per metre of
    ! each mapped tracer remapped onto the cell's layers and times Hz (0
    ! when the source column has no thickness).
    real(rp), intent(in) :: c1, c2
    real(rp), allocatable :: hzs(:), flx(:), fpm(:), rem(:), wt(:,:,:)
    integer :: q, a, r, k, c, nzl
    nzl = int(decomp%nz)
    allocate(hzs(n_src), flx(n_src), fpm(n_src), rem(nzl), wt(nzl, max(ndpm, 1), max(nq_loc, 1)))
    wt = 0.0_rp
    call cdr_dp_gather_hz()            ! drains the device: Hz of the release columns
    do q = 1, nq_loc
      r = q_rel(q); c = q_cell(q)
      if (cdr_time_interpolation) then
        hzs = c1 * dhz(r, :, 1) + c2 * dhz(r, :, 2)
      else
        hzs = dhz(r, :, 1)
      endif
      if (.not. (sum(hzs) > 0.0_rp)) cycle
      do a = 1, ndpm
        if (cdr_time_interpolation) then
          flx = c1 * dfx(r, dp_col(a), :, 1) + c2 * dfx(r, dp_col(a), :, 2)
        else
          flx = dfx(r, dp_col(a), :, 1)
        endif
        do k = 1, n_src
          if (hzs(k) > 0.0_rp) then
            fpm(k) = flx(k) / hzs(k)
          else
            fpm(k) = 0.0_rp
          endif
        enddo
        call remap_src_to_grid(n_src, hzs, fpm, nzl, cdr_dp_hzc(:, c), rem)
        wt(:, a, q) = rem(:) * cdr_dp_hzc(:, c)
      enddo
    enddo
    call cdr_set_dp_weights(wt)
    deallocate(hzs, flx, fpm, rem, wt)
  end subroutine set_dp_weights

  subroutine load_record(idx, slot)
    ! cdr_trcflx(:, :, idx), mapped to (passive tracer, release); depth
    ! profiles: the layer thicknesses and profiles of record idx (host).
    integer, intent(in) :: idx, slot
    integer(ip) :: itp
    integer :: r
    if (cdr_forcing_3d) then
      call load_3d(idx, slot)
      return
    endif
    if (cdr_forcing_depth_profiles) then
      call cdr_check(nf90_get_var(ncid_cdr, vid_hz, dhz(:, :, slot), start=[1, 1, idx], count=[ncdr, n_src, 1]), &
                     'get cdr_layer_thickness')
      call cdr_check(nf90_get_var(ncid_cdr, vid_flx, dfx(:, :, :, slot), start=[1, 1, 1, idx], &
                     count=[ncdr, ntrc_dp, n_src, 1]), 'get cdr_trcflx_profile')
      return
    endif
    call cdr_check(nf90_get_var(ncid_cdr, vid_flx, frec, start=[1, 1, idx], count=[ncdr, ntr_file, 1]), &
                   'get cdr_trcflx')
    do r = 1, ncdr
      do itp = 1, ntp
        fpas(itp, r) = frec(r, tmap(itp))
      enddo
    enddo
    if (cdr_volume) then
      call cdr_check(nf90_get_var(ncid_cdr, vid_vol, vrec, start=[1, idx], count=[ncdr, 1]), 'get cdr_volume')
      do r = 1, ncdr
        tsrec(1, r) = frec(r, 1); tsrec(2, r) = frec(r, 2)
      enddo
      call cdr_set_record_vol(slot, fpas, tsrec, vrec)
      return
    endif
    call cdr_set_record(slot, fpas)
  end subroutine load_record

  subroutine set_cdr_pred()
    ! Volume releases: the volume at the predictor's forcing time (the
    ! CPU's set_cdr_frc in its first set_forces), for the predictor and the
    ! first corrector omega. The other modes use only the corrector's.
    if (.not. (cdr_source .and. cdr_volume)) return
    call set_cdr()
  end subroutine set_cdr_pred

  subroutine set_vol_lists(nc, nq, ci, cj, cp0, wq)
    ! Volume releases: each profile's weight sum (for zeta) and the cell map
    ! (for omega).
    integer, intent(in) :: nc, nq
    integer(ip), intent(in) :: ci(:), cj(:), cp0(:)
    real(rp), intent(in) :: wq(:,:)
    real(rp), allocatable :: qs(:)
    integer(ip), allocatable :: cm(:,:)
    integer :: q, k, c
    allocate(qs(max(nq, 1)), cm(decomp%is:decomp%ie, decomp%js:decomp%je))
    qs = 0.0_rp; cm = 0
    do q = 1, nq
      do k = 1, size(wq, 1)
        qs(q) = qs(q) + wq(k, q)
      enddo
    enddo
    do c = 1, nc
      cm(ci(c), cj(c)) = c
    enddo
    call cdr_set_vol(qs, cm)
    deallocate(qs, cm)
  end subroutine set_vol_lists

  subroutine rest_grid(h, nz, zr0, hz0)
    ! z_r0, Hz0 of one column with zeta = 0, in the CPU's expressions
    ! (set_depth at iic = 0): z = h*(cff + Cs*h)*hinv, hinv = 1/(h+hc).
    real(rp), intent(in) :: h
    integer, intent(in) :: nz
    real(rp), intent(out) :: zr0(nz), hz0(nz)
    real(rp) :: hinv, ds, cff_w, cff_r, zw_prev, zw
    integer :: k
    hinv = 1.0_rp / (h + hc)
    ds = 1.0_rp / real(nz, rp)
    zw_prev = -h
    do k = 1, nz
      cff_w = hc * ds * real(k - nz, rp)
      cff_r = hc * ds * (real(k - nz, rp) - 0.5_rp)
      zw = h * (cff_w + Cs_w(k) * h) * hinv
      zr0(k) = h * (cff_r + Cs_r(k) * h) * hinv
      hz0(k) = zw - zw_prev
      zw_prev = zw
    enddo
  end subroutine rest_grid

  subroutine haversine(lon, lat, lon0, lat0, dist, gx, gy)
    ! The CPU's ll2dist, on the interior cells (the rest stay far away).
    integer, intent(in) :: gx, gy
    real(rp), intent(in) :: lon(gx, gy), lat(gx, gy), lon0, lat0
    real(rp), intent(out) :: dist(gx, gy)
    real(rp) :: dlon2, dlat2, a
    integer :: i, j
    dist = 1.0e6_rp
    do j = 2, gy - 1
      do i = 2, gx - 1
        dlon2 = 0.5_rp * (lon(i, j) - lon0) * deg2rad
        dlat2 = 0.5_rp * (lat(i, j) - lat0) * deg2rad
        a = sin(dlat2)**2 + cos(lat(i, j) * deg2rad) * cos(lat0 * deg2rad) * sin(dlon2)**2
        a = 2 * atan2(sqrt(a), sqrt(1 - a))
        dist(i, j) = radius * a
      enddo
    enddo
  end subroutine haversine

  subroutine get_1d(name, x)
    character(len=*), intent(in) :: name
    real(rp), intent(out) :: x(:)
    integer :: vid
    call cdr_check(nf90_inq_varid(ncid_cdr, name, vid), 'var '//name)
    call cdr_check(nf90_get_var(ncid_cdr, vid, x, start=[1], count=[size(x)]), 'get '//name)
  end subroutine get_1d

  subroutine get_2d(ncid, name, x)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(rp), intent(out) :: x(:,:)
    integer :: vid
    call cdr_check(nf90_inq_varid(ncid, name, vid), 'grid var '//name)
    call cdr_check(nf90_get_var(ncid, vid, x), 'get grid '//name)
  end subroutine get_2d

  subroutine cdr_check(ierr, where)
    integer,          intent(in) :: ierr
    character(len=*), intent(in) :: where
    if (ierr /= nf90_noerr) then
      write(*,'(a,a,a,a)') 'cdr_io: ', trim(where), ': ', trim(nf90_strerror(ierr))
      error stop 'cdr_io: netCDF error'
    endif
  end subroutine cdr_check

  ! ---- the CPU's vertical_remapping (conservative piecewise-parabolic
  ! remap, White and Adcroft 2008), verbatim but for the real kind
  subroutine remap_src_to_grid(N_src, H_src, t_src, N_tgt, H_tgt, t_tgt)

    ! Perform conservative remapping of a depth profile from the source grid to the
    ! target (current) grid using piecewise parabolic reconstruction.

    implicit none

    integer(kind=4), intent(in)                 :: N_src
    real(rp), dimension(N_src), intent(in)  :: H_src
    real(rp), dimension(N_src), intent(in)  :: t_src
    integer(kind=4), intent(in)                 :: N_tgt
    real(rp), dimension(N_tgt), intent(in)  :: H_tgt
    real(rp), dimension(N_tgt), intent(out) :: t_tgt

    ! local
    integer(kind=4) :: k,k_new,trc,idx
    real(rp), dimension(N_src+1) :: interface_values
    real(rp) :: total_depth_src, total_depth_tgt
    real(rp), dimension(N_src) :: a0, a1, a2
    real(rp), dimension(N_src+1) :: z_orig_interfaces
    real(rp), dimension(N_src) :: H_orig
    real(rp) :: H_src_tot

    integer(kind=4) :: curr_src_index
    integer(kind=4), dimension(N_tgt) :: tgt_index_start
    integer(kind=4), dimension(N_tgt) :: tgt_index_end

    ! Keep track of how far up we are in the water column
    real(rp) :: curr_tgt_height, curr_src_height

    ! Keep track of where in each H_orig cell the interfaces of H_tgt are
    real(rp), dimension(N_tgt) :: tgt_frac_start
    real(rp), dimension(N_tgt) :: tgt_frac_end

    ! This holds the integrated_values
    real(rp), dimension(N_tgt) :: definite_integral

    ! Temporary array to hold the remapped fluxes
    real(rp), dimension(N_tgt) :: t_tmp

    ! Scalars to hold the tracer mass in each cell
    real(rp) :: total_t_src
    real(rp) :: total_t_tgt
    real(rp) :: total_t_diff
    real(rp) :: cellwise_mass_frac


    tgt_frac_start(:) = 0.0_rp
    tgt_frac_end(:)   = 0.0_rp

    definite_integral(:) = 0.0_rp
    t_tmp(:) = 0.0_rp
    t_tgt(:) = 0.0_rp

    call calc_interface_values(N_src, H_src, t_src, interface_values)

    ! Compare total depths and expand/contract current grid to match original depth
    total_depth_src = 0
    total_depth_tgt = 0
    do k=1,N_tgt
      total_depth_tgt = total_depth_tgt + H_tgt(k)
    enddo

    do k=1,N_src
      total_depth_src = total_depth_src + H_src(k)
    enddo

    total_t_src = 0.0_rp
    do k=1,N_src
      a0(k) = interface_values(k)
      a1(k) = 6*t_src(k) - 4*interface_values(k) - 2*interface_values(k+1)
      a2(k) = 3*(interface_values(k) + interface_values(k+1) - 2*t_src(k))
      total_t_src = total_t_src + t_src(k)*H_src(k)
    enddo

    ! These are vector copies of the source and target column.
    do k = 1,N_src
      H_orig(k) = H_src(k)
    enddo

    ! Match the thicknesses of the source and target columns
    do k = 1,N_src
      H_orig(k) = H_orig(k) * (total_depth_tgt / total_depth_src)
    enddo
    !! The division and multiplication in the above step may not make the total column thicknesses
    !! match exactly, so one more really small correction will be added to the top layer.
    H_src_tot = 0
    do k = 1,N_src
      H_src_tot = H_src_tot + H_orig(k)
    enddo
    H_orig(N_src) = H_orig(N_src) + total_depth_tgt - H_src_tot

    !! Also keep track of the original interface depths.
    z_orig_interfaces(1) = 0
    do k = 1,N_src
      z_orig_interfaces(k+1) = z_orig_interfaces(k) + H_orig(k)
    enddo

    ! Keep track of how far up we are in the water column
    tgt_index_start = 1
    tgt_index_end = 1
    curr_tgt_height = H_tgt(1)
    curr_src_height = H_orig(1)
    curr_src_index = 1

    ! Sort the source and target cell heights so we know how to perform the integration
    do k_new = 1,N_tgt-1

      do while (curr_tgt_height > curr_src_height)
        curr_src_index = curr_src_index + 1
        curr_src_height = curr_src_height + H_orig(curr_src_index)
      enddo

      tgt_index_end(k_new) = curr_src_index
      tgt_index_start(k_new+1) = curr_src_index

      tgt_frac_end(k_new) = (curr_tgt_height - z_orig_interfaces(curr_src_index)) / H_orig(curr_src_index)
      tgt_frac_start(k_new+1) = tgt_frac_end(k_new)

      curr_tgt_height = curr_tgt_height + H_tgt(k_new+1)

    enddo

    ! Remap by integrating the piecewise parabolas from the source grid
    tgt_index_end(N_tgt) = N_src
    tgt_frac_end(N_tgt)  = 1

    total_t_tgt = 0
    do k = 1,N_tgt

      do idx = tgt_index_start(k),tgt_index_end(k)

        if (tgt_index_start(k) == tgt_index_end(k)) then
          definite_integral(k) = integrate(a0(idx), a1(idx), a2(idx), tgt_frac_start(k), tgt_frac_end(k))*H_src(idx)
        elseif ((idx == tgt_index_start(k)) .and. (idx < tgt_index_end(k))) then
          definite_integral(k) = integrate(a0(idx), a1(idx), a2(idx), tgt_frac_start(k), 1.0_rp)*H_src(idx)
        elseif ((idx < tgt_index_end(k)) .and. (idx > tgt_index_start(k))) then
          definite_integral(k) = definite_integral(k) + integrate(a0(idx), a1(idx), a2(idx), 0.0_rp, 1.0_rp)*H_src(idx)
        else
          definite_integral(k) = definite_integral(k) +&
          &integrate(a0(idx), a1(idx), a2(idx), 0.0_rp, tgt_frac_end(k))*H_src(idx)
        endif

      enddo
      t_tmp(k) = definite_integral(k) / H_tgt(k)
      total_t_tgt = total_t_tgt + t_tmp(k)*H_tgt(k)
    enddo

    ! Correction to ensure exact tracer conservation in the remapping
    ! We distribute the correction over the entire column, where the cellwise amount
    ! depends on how much of the column mass the cell contains
    total_t_diff = total_t_tgt - total_t_src

    if (total_t_tgt /= 0.0_rp) then
      do k = 1,N_tgt
        cellwise_mass_frac = t_tmp(k) / total_t_tgt
        t_tgt(k) = t_tmp(k) - total_t_diff*cellwise_mass_frac
      enddo
    endif

  end subroutine remap_src_to_grid
! ----------------------------------------------------------------------
  real(rp) function integrate(a0, a1, a2, z0, z1) ![
    ! Definite integration of a parabola

    implicit none

    real(rp) :: a0, a1, a2, z0, z1
    real(rp) :: one_third = 0.3333333333_rp

    integrate = a0 * (z1 - z0) + 0.5_rp*a1*(z1**2 - z0**2) +&
    &one_third*a2*(z1**3 - z0**3)

  end function integrate !]
! ----------------------------------------------------------------------
  subroutine calc_interface_values(N_src, H, arr, int_vals)  ![

    ! Interpolate/extrapolate values from cell centers to the cell interfaces

    implicit none

    integer(kind=4), intent(in) :: N_src
    real(rp), dimension(N_src), intent(in)    :: H
    real(rp), dimension(N_src), intent(in)    :: arr
    real(rp), dimension(N_src+1), intent(out) :: int_vals

    real(rp), dimension(0:3) :: B
    real(rp), dimension(0:3,0:3) :: M
    real(rp) :: h_b
    real(rp) :: h_t
    real(rp) :: iH
    real(rp) :: Ts_bot
    real(rp) :: Ts_top

    integer(kind=4) :: k, kk, kkp, kkp1, trc
    integer(kind=4) :: ord = 3

    !!                  Get tracer values at layer interfaces                !!
    !!  Calculated with Implicit Fourth-order scheme using Thomas algorithm  !!
    !!                   (White and Adcroft, 2008, Eq. 46)                   !!

    !! Bottom boundary extrapolation
    !! Calculated using a polynomial of order "ord" over the bottom (ord+1) cells

    h_b = 0
    h_t = H(1)
    iH  = 1.0_rp / (h_t - h_b)

    do k = 0,ord
      do kk = 0,ord
        kkp1 = kk+1
        M(k,kk) = (1.0_rp / kkp1) * iH * (h_t**(kkp1) - h_b**(kkp1))
      enddo

      h_b = h_b + H(k+1)
      h_t = h_t + H(k+2)
      iH = 1.0_rp/(h_t - h_b)
      B(k) = arr(k+1)
    enddo

    Ts_bot = gauss(M,B,ord)

    !! Top boundary extrapolation
    h_b = 0
    h_t = H(N_src)
    iH = 1.0_rp / (h_t - h_b)
    do k = 0,ord
      do kk = 0,ord
        kkp1 = kk+1
        M(k,kk) = (1.0_rp / kkp1) * iH * (h_t**kkp1 - h_b**kkp1)
      enddo

      h_b = h_b + H(N_src-k)
      h_t = h_t + H(N_src-1-k)
      iH = 1.0_rp / (h_t - h_b)
      B(k) = arr(N_src-k)
    enddo

    Ts_top = gauss(M,B,ord)
    !!!!!!!!!!!!!!!!!!!!!!

    call thomas_PPM(N_src, Ts_bot, Ts_top, arr, H, int_vals)

  end subroutine calc_interface_values  !]
! ----------------------------------------------------------------------
  real(rp) function gauss(M, b, ord) ![

    !! This routine uses Gaussian Elimination to reduce the matrix M to row-echelon form (lower triangular).
    !! It only returns the first element of the solution vector.

    implicit none

    real(rp), dimension(0:3,0:3), intent(inout) :: M
    real(rp), dimension(0:3), intent(inout) :: b
    integer(kind=4), intent(in) :: ord
    real(rp) :: ratio
    integer(kind=4) :: i,j,k

    !! Starting from the last row, we're going to work upwards to
    !! make M a lower triangular matrix
    do i = ord,1,-1
      !! Iterate over all rows above i
      do j = 0,(i-1)
        ratio = M(j,i) / M(i,i)

        !! Iterate over all columns up to i.
        !! This operation basically is multiplying the ith row by "ratio" and adding it to the jth row
        do k = 0,i
          M(j,k) = M(j,k) - ratio*M(i,k)
        enddo
        b(j) = b(j) - ratio*b(i)
      enddo
    enddo

    gauss = (b(0) / M(0,0))

  end function gauss !]
! ----------------------------------------------------------------------
  subroutine thomas_PPM(N_src, Ts_bot, Ts_top, arr, H, int_vals)  ![

    ! Extremely efficient tridiagonal matrix solver for cases where we
    ! have a diagonally dominant input matrix

    implicit none

    integer(kind=4), intent(in) :: N_src
    real(rp), intent(in) :: Ts_bot
    real(rp), intent(in) :: Ts_top
    real(rp), dimension(N_src),   intent(in)  :: arr
    real(rp), dimension(N_src),   intent(in)  :: H
    real(rp), dimension(N_src+1), intent(out) :: int_vals

    integer(kind=4) :: n,t,k
    real(rp), dimension(N_src+1) :: a, b, c, d, cp, dp, x
    real(rp) :: h0, h1, alpha, beta, d1, d2

    n = N_src+1

    ! Solution for top and bottom boundaries
    b(1) = 1.0_rp
    b(n) = 1.0_rp
    d(1) = Ts_bot
    d(n) = Ts_top
    a(n) = 0
    c(1) = 0

    ! Fill interior of tridiagonal matrix
    do k=2,(n-1)
      h0 = H(k-1)
      h1 = H(k)

      alpha = (h1**2) / ((h0 + h1)**2)
      beta  = (h0**2) / ((h0 + h1)**2)
      d1 = 2*(h1**2)*(h1**2 + 2*h0**2 + 3*h0*h1) / ((h0+h1)**4)
      d2 = 2*(h0**2)*(h0**2 + 2*h1**2 + 3*h0*h1) / ((h0+h1)**4)

      a(k) = alpha
      b(k) = 1.0_rp
      c(k) = beta

      d(k) = d1*arr(k-1) + d2*arr(k)
    enddo

    ! Solve using Thomas algorithm
    cp(1) = c(1) / b(1)
    dp(1) = d(1) / b(1)

    do k = 2,(n-1)
      cp(k) = c(k) / (b(k) - a(k)*cp(k-1))
      dp(k) = (d(k) - a(k)*dp(k-1)) / (b(k) - a(k)*cp(k-1))
    enddo
    dp(n) = (d(n) - a(n)*dp(n-1)) / (b(n) - a(n)*cp(n-1))

    int_vals(n) = d(n)
    do k = (n-1),1,-1
      int_vals(k) = dp(k) - cp(k)*int_vals(k+1)
    enddo

  end subroutine thomas_PPM !]
!-------------------------------------------------------------------

end module cdr_io

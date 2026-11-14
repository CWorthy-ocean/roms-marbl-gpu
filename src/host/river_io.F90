#include "mem_attr.inc"
module river_io
  ! River forcing input (UCLA river_frc.F90): the namelist, the river
  ! locations, and the river_volume / river_tracer records with their time
  ! interpolation weights. The device side is rivers_mod.
  !
  ! Locations, as the CPU's init_river_frc / calc_river_flux: river_index
  ! and river_fraction from the forcing file that holds river_volume, else
  ! river_flux (= index + fraction) from the grid file. A river cell must be
  ! land with at least one ocean neighbour; its fraction is shared equally by
  ! its faces to ocean cells, signed by direction, and each such face gets
  ! riv_umask/riv_vmask = 1. Each tile lists the faces it updates in each of
  ! the CPU's loop ranges and the ocean cells it owns next to river faces.
  !
  ! Records: one time catalog over all forcing files that hold river_volume,
  ! bracketed and weighted as frc_io advance_series / weights (the CPU's
  ! set_frc_data_1d/2d formula). Only the rivers this tile touches go to the
  ! device. A cycle_length attribute on river_time (of the first file with
  ! river records) makes the records a climatology, bracketed by the CPU's
  ! find_new_record_cycle rule (frc_io cyc_bracket), as for the other
  ! forcing fields.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use nc_io
  use frc_io, only: case_frcfiles, case_nfrc, frc_file_ncid, cyc_bracket
  use passive_mod, only: ntp, ncdr, cpu_tracer_index
  use rivers_mod
  use comm_mod, only: comm_rank
  use arch_mod, only: sync_arch
  use netcdf
  implicit none
  private
  public :: read_river_namelist, init_rivers, set_rivers
  public :: river_source

  logical :: river_source = .false., river_analytical = .false.
  integer :: nriv = 0

  ! time catalog of the river records (sorted by time)
  integer :: ncat = 0, it1 = 0, it2 = 0
  real(rp), allocatable :: cat_t(:)
  integer, allocatable :: cat_f(:), cat_r(:)
  real(rp) :: cat_cycle                       ! cycle_length of river_time (<= 0: not cyclic)
  integer, allocatable :: fncid(:)            ! open handle per forcing file (0: no rivers)
  integer :: nriver_f = 0, ntr_f = 0          ! dims of the river variables in the files
  integer(ip) :: nloc = 0                     ! rivers this tile touches
  integer, allocatable :: glob(:)             ! local river -> river index (1-based)
  real(rp), allocatable :: vol_g(:), trc_g(:,:), vol_l(:), trc_l(:,:)

contains

  subroutine read_river_namelist(unit)
    ! &river_frc_settings nriv, river_source, river_analytical (as the CPU).
    ! A namelist without the group has no rivers.
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    namelist /river_frc_settings/ nriv, river_source, river_analytical
    river_source = .false.; river_analytical = .false.; nriv = 0
    rewind(unit)
    read(unit, nml=river_frc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'RIVER_FRC_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_river_namelist'
    endif
    if (ios < 0) then
      river_source = .false.
      return
    endif
    if (river_source .and. river_analytical) &
      error stop 'read_river_namelist: analytical rivers are not available on the GPU model'
  end subroutine read_river_namelist

  subroutine init_rivers(grdname)
    character(len=*), intent(in) :: grdname
    integer :: i, j, k, nx, ny, nf, nc, m, ifile, ncid, varid, ierr, ntr_model, nrec, l
    integer :: nfmax, rfile, gi, gj, ir
    logical :: have_sep, riv_cdr
    integer(ip) :: itp
    integer(ip), allocatable :: rmap(:)
    real(rp) :: faces, uflx, frac
    real(rp), allocatable :: ridx_r(:,:), rfrc(:,:), rflx(:,:), rm(:,:), tloc(:)
    integer, allocatable :: ridx(:,:), loc_of(:)
    integer(ip), allocatable :: fi(:), fj(:), ft(:), fr(:), fs(:), ci(:), cj(:), ccol(:), cn(:), cf(:,:)
    real(rp), allocatable :: ffrac(:), csg(:,:)
    integer, allocatable :: fglob(:), cell_of(:,:)
    integer(ip) :: iu_s2d0, iu_s2d1, jv_s2d0, jv_s2d1, iu_pre0, jv_pre0
    integer :: nbad, maxidx
    character(len=16) :: what
    real(rp) :: cyc
    logical :: lcyc

    cat_cycle = -1.0_rp
    if (.not. river_source) return
    nx = int(decomp%nx); ny = int(decomp%ny)
    if (decomp%periodic_x .or. decomp%periodic_y) &
      error stop 'init_rivers: rivers on a periodic grid are not supported on the GPU model'

    ! --- the forcing files with river data; the first one holds the locations.
    ! Scanned through frc_io's handles (open during initialization); the
    ! river files get their own handles, kept open for the record reads.
    allocate(fncid(max(case_nfrc, 1)))
    fncid = 0
    rfile = 0
    do i = 1, case_nfrc
      ncid = frc_file_ncid(i)
      if (nc_has_var(ncid, 'river_volume')) then
        if (.not. nc_has_var(ncid, 'river_tracer')) then
          write(*,*) 'init_rivers: river_volume without river_tracer in ', trim(case_frcfiles(i))
          error stop 'init_rivers'
        endif
        call nc_open_read(case_frcfiles(i), fncid(i))
        if (rfile == 0) rfile = i
      endif
    enddo
    if (rfile == 0) error stop 'init_rivers: river_source = .true. but no forcing file holds river_volume'

    ! --- locations over the tile's cells 0:nx+1, 0:ny+1 (file window at the
    ! tile's offset, as the other rho fields)
    allocate(ridx(0:nx+1, 0:ny+1), rfrc(0:nx+1, 0:ny+1))
    ncid = fncid(rfile)
    have_sep = nc_has_var(ncid, 'river_index') .and. nc_has_var(ncid, 'river_fraction')
    if (have_sep) then
      allocate(ridx_r(0:nx+1, 0:ny+1))
      call read_window(ncid, 'river_index', ridx_r, nx, ny)
      call read_window(ncid, 'river_fraction', rfrc, nx, ny)
      where (ridx_r /= ridx_r) ridx_r = 0.0_rp       ! NaN fill off the rivers
      where (rfrc /= rfrc) rfrc = 0.0_rp
      ridx = int(ridx_r)
      if (any(abs(ridx_r - real(ridx, rp)) > 1.0e-6_rp)) error stop 'init_rivers: river_index contains non-integers'
      deallocate(ridx_r)
      what = 'river_index'
    else
      call nc_open_read(grdname, ncid)
      if (.not. nc_has_var(ncid, 'river_flux')) then
        write(*,*) 'init_rivers: no river_index/river_fraction in ', trim(case_frcfiles(rfile)), &
                   ' and no river_flux in ', trim(grdname)
        error stop 'init_rivers'
      endif
      allocate(rflx(0:nx+1, 0:ny+1))
      call read_window(ncid, 'river_flux', rflx, nx, ny)
      call nc_close(ncid)
      where (rflx /= rflx) rflx = 0.0_rp
      do j = 0, ny + 1
        do i = 0, nx + 1
          if (rflx(i, j) > 0.0_rp) then
            ridx(i, j) = floor(rflx(i, j) - 1e-5)
            rfrc(i, j) = rflx(i, j) - ridx(i, j)
          else
            ridx(i, j) = 0
            rfrc(i, j) = 0.0_rp
          endif
        enddo
      enddo
      deallocate(rflx)
      what = 'river_flux'
    endif

    ! --- dims of the river variables. As the CPU (init_river_trc_map),
    ! river_tracer holds every tracer in the CPU's order, or every tracer
    ! but the CDR_LITE block, whose tracers then get concentration 0.
    call river_dims(fncid(rfile), nriver_f, ntr_f)
    ntr_model = int(nt + ntp)
    riv_cdr = (ntr_f == ntr_model)
    if (.not. riv_cdr .and. (ncdr == 0 .or. ntr_f /= ntr_model - int(ncdr))) then
      write(*,'(a,i0,a,i0,a,i0,a)') 'init_rivers: ntracers = ', ntr_f, ' in the river forcing file, but must be ', &
           ntr_model, ' (all tracers) or ', ntr_model - int(ncdr), ' (all tracers but the CDR_LITE block)'
      error stop 'init_rivers'
    endif
    if (.not. riv_cdr .and. comm_rank() == 0) write(*,'(a,i0,a)') &
         'init_rivers: river_tracer has no CDR_LITE tracers; ', ncdr, ' CDR_LITE tracers get river concentration 0'
    maxidx = maxval(ridx)
    if (maxidx > nriver_f) then
      write(*,'(a,i0,a,i0)') 'init_rivers: river index ', maxidx, ' exceeds the nriver dimension ', nriver_f
      error stop 'init_rivers'
    endif
    if (nriv > 0 .and. maxidx > nriv) then
      write(*,'(a,i0,a,i0,a)') 'init_rivers: nriv = ', nriv, ' but index ', maxidx, ' found in the river input'
      error stop 'init_rivers'
    endif

    ! --- the faces (CPU calc_river_flux over the cells 0:nx+1, 0:ny+1)
    call sync_masks_to_host(rm, nx, ny)
    nfmax = 4 * count(rfrc > 0.0_rp) + 1
    allocate(fi(nfmax), fj(nfmax), ft(nfmax), fr(nfmax), fs(nfmax), ffrac(nfmax), fglob(nfmax))
    nf = 0; nbad = 0
    iu_s2d0 = merge(2, 1, decomp%edge_w); iu_s2d1 = merge(nx, nx + 1, decomp%edge_e)
    jv_s2d0 = merge(2, 1, decomp%edge_s); jv_s2d1 = merge(ny, ny + 1, decomp%edge_n)
    iu_pre0 = merge(2, 1, decomp%edge_w); jv_pre0 = merge(2, 1, decomp%edge_s)
    do j = 0, ny + 1
      do i = 0, nx + 1
        if (rfrc(i, j) <= 0.0_rp) cycle
        faces = rm(i - 1, j) + rm(i + 1, j) + rm(i, j - 1) + rm(i, j + 1)
        if ((faces == 0.0_rp .or. rm(i, j) > 0.0_rp) .and. i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) then
          write(*,'(a,i0,a,i0,a,i0)') 'init_rivers: river grid position error at global i,j = ', &
               int(decomp%i0) + i - 1, ',', int(decomp%j0) + j - 1, '  index ', ridx(i, j)
          nbad = nbad + 1
          cycle
        endif
        if (faces == 0.0_rp) cycle
        if (ridx(i, j) < 1) then
          write(*,'(a,i0,a,i0)') 'init_rivers: river fraction > 0 with river index < 1 at global i,j = ', &
               int(decomp%i0) + i - 1, ',', int(decomp%j0) + j - 1
          nbad = nbad + 1
          cycle
        endif
        if (rm(i - 1, j) > 0.0_rp) call add_face(1, i, j, -(rfrc(i, j)) / faces + 10 * ridx(i, j))
        if (rm(i + 1, j) > 0.0_rp) call add_face(1, i + 1, j, (rfrc(i, j)) / faces + 10 * ridx(i, j))
        if (rm(i, j - 1) > 0.0_rp) call add_face(2, i, j, -(rfrc(i, j)) / faces + 10 * ridx(i, j))
        if (rm(i, j + 1) > 0.0_rp) call add_face(2, i, j + 1, (rfrc(i, j)) / faces + 10 * ridx(i, j))
      enddo
    enddo
    if (nbad > 0) error stop 'init_rivers: invalid river locations (see above)'

    ! --- local rivers (the order of first appearance)
    allocate(loc_of(nriver_f), glob(max(nf, 1)))
    loc_of = 0; nloc = 0
    do m = 1, nf
      ir = fglob(m)
      if (loc_of(ir) == 0) then
        nloc = nloc + 1
        loc_of(ir) = int(nloc)
        glob(nloc) = ir
      endif
      fr(m) = loc_of(ir)
    enddo

    ! --- the owned ocean cells next to the faces of the tracer range
    ! (u faces 1:nx+1 x 1:ny, v faces 1:nx x 1:ny+1), up to 4 faces each
    allocate(cell_of(nx, ny)); cell_of = 0
    allocate(ci(max(nf, 1)), cj(max(nf, 1)), ccol(max(nf, 1)), cn(max(nf, 1)), cf(4, max(nf, 1)), csg(4, max(nf, 1)))
    nc = 0; cn = 0; cf = 0; csg = 0.0_rp
    do m = 1, nf
      i = fi(m); j = fj(m)
      if (ft(m) == 1) then
        if (i < 1 .or. i > nx + 1 .or. j < 1 .or. j > ny) cycle
        if (rm(i, j) > 0.0_rp) then
          gi = i; gj = j; frac = 1.0_rp     ! the face is the cell's west face
        else
          gi = i - 1; gj = j; frac = -1.0_rp
        endif
      else
        if (i < 1 .or. i > nx .or. j < 1 .or. j > ny + 1) cycle
        if (rm(i, j) > 0.0_rp) then
          gi = i; gj = j; frac = 1.0_rp     ! the cell's south face
        else
          gi = i; gj = j - 1; frac = -1.0_rp
        endif
      endif
      if (gi < 1 .or. gi > nx .or. gj < 1 .or. gj > ny) cycle
      if (cell_of(gi, gj) == 0) then
        nc = nc + 1
        cell_of(gi, gj) = nc
        ci(nc) = gi; cj(nc) = gj
        ccol(nc) = 0
        if (allocated(tpcol_h)) ccol(nc) = tpcol_h(gi, gj)
      endif
      k = cell_of(gi, gj)
      cn(k) = cn(k) + 1
      if (cn(k) > 4) error stop 'init_rivers: more than 4 river faces on one cell'
      cf(cn(k), k) = m
      csg(cn(k), k) = frac
    enddo

    ! --- time catalog
    ncat = 0
    do ifile = 1, case_nfrc
      if (fncid(ifile) /= 0) ncat = ncat + nc_dim_len(fncid(ifile), 'river_time')
    enddo
    if (ncat < 1) error stop 'init_rivers: no river records'
    allocate(cat_t(ncat), cat_f(ncat), cat_r(ncat))
    l = 0
    do ifile = 1, case_nfrc
      if (fncid(ifile) == 0) cycle
      nrec = nc_dim_len(fncid(ifile), 'river_time')
      allocate(tloc(nrec))
      call nc_get_1d(fncid(ifile), 'river_time', tloc)
      do k = 1, nrec
        l = l + 1
        cat_t(l) = tloc(k); cat_f(l) = ifile; cat_r(l) = k
      enddo
      deallocate(tloc)
      call river_dims(fncid(ifile), i, j)
      if (i /= nriver_f .or. j /= ntr_f) then
        write(*,*) 'init_rivers: river files disagree on nriver/ntracers: ', trim(case_frcfiles(ifile))
        error stop 'init_rivers'
      endif
    enddo
    call sort_catalog()
    it1 = 0; it2 = 0
    do ifile = 1, case_nfrc
      if (fncid(ifile) == 0) cycle
      call nc_get_att_real(fncid(ifile), 'river_time', 'cycle_length', cyc, lcyc)
      if (lcyc .and. cyc > 0.0_rp) cat_cycle = cyc
      exit
    enddo
    if (cat_cycle > 0.0_rp .and. comm_rank() == 0) &
      write(*,'(a,f10.3)') 'rivers: river_time is cyclic, cycle_length =', cat_cycle
    ! trc_l row ntr_f + 1 stays 0: the row of CDR_LITE tracers the file lacks
    allocate(vol_g(nriver_f), trc_g(nriver_f, ntr_f), vol_l(max(nloc, 1)), trc_l(ntr_f + 1, max(nloc, 1)))
    trc_l = 0.0_rp

    call rivers_set_lists(int(nf, ip), fi, fj, ft, fr, fs, ffrac, int(nc, ip), ci, cj, ccol, cn, cf, csg, &
                          nloc, int(ntr_f + 1, ip))
    allocate(rmap(max(ntp, 1_ip)))
    do itp = 1, ntp
      rmap(itp) = cpu_tracer_index(itp, riv_cdr)
      if (rmap(itp) == 0) rmap(itp) = int(ntr_f + 1, ip)
    enddo
    call rivers_set_map(rmap(1:ntp))
    deallocate(rmap)
    write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,a,a,i0,a)') 'rivers: rank ', comm_rank(), '  faces ', nf, &
         '  cells ', nc, '  rivers ', nloc, '  fast-loop faces next to an edge ', nrf_s2d, &
         '  (', trim(what), ', ', ncat, ' records)'
    deallocate(ridx, rfrc, rm, loc_of, cell_of)

  contains

    subroutine add_face(typ, ii, jj, flx)
      ! One face as the CPU's riv_uflx/riv_vflx: iriver = nint(flx/10),
      ! frac = flx - 10*iriver. Faces inside the arrays set the river mask;
      ! every face is listed, with the bits of the loop ranges it is in.
      integer,  intent(in) :: typ, ii, jj
      real(rp), intent(in) :: flx
      integer :: iriver
      integer(ip) :: sb, s2d
      logical :: far
      iriver = nint(flx / 10)
      if (ii >= decomp%is .and. ii <= decomp%ie .and. jj >= decomp%js .and. jj <= decomp%je) then
        if (typ == 1) then
          riv_umask(ii, jj) = 1.0_rp
        else
          riv_vmask(ii, jj) = 1.0_rp
        endif
      endif
      ! fast-loop faces two cells or more from every physical edge are done
      ! inside step2d_mid's kernel (RIV_S2DF), the others by rivers_ubar.
      ! Every face of the scan is listed with RIV_RNG: rivers_huv and
      ! rivers_duon check the ranges of set_HUV and step2d's DUon/DVom (up
      ! to the faces 0:nx+2 / 0:ny+2 of the seam ring) themselves.
      sb = int(RIV_RNG, ip)
      if (typ == 1) then
        far = (.not. decomp%edge_w .or. ii >= 3) .and. (.not. decomp%edge_e .or. ii <= nx - 1) .and. &
              (.not. decomp%edge_s .or. jj >= 3) .and. (.not. decomp%edge_n .or. jj <= ny - 2)
      else
        far = (.not. decomp%edge_w .or. ii >= 3) .and. (.not. decomp%edge_e .or. ii <= nx - 2) .and. &
              (.not. decomp%edge_s .or. jj >= 3) .and. (.not. decomp%edge_n .or. jj <= ny - 1)
      endif
      s2d = merge(int(RIV_S2DF, ip), int(RIV_S2D, ip), far)
      if (typ == 1) then
        if (ii >= iu_s2d0 .and. ii <= iu_s2d1 .and. jj >= 1 .and. jj <= ny) sb = ior(sb, s2d)
        if (ii >= iu_pre0 .and. ii <= nx .and. jj >= 1 .and. jj <= ny) sb = ior(sb, int(RIV_PRE, ip))
        if (ii >= 1 .and. ii <= nx .and. jj >= 1 .and. jj <= ny) sb = ior(sb, int(RIV_COR, ip))
      else
        if (ii >= 1 .and. ii <= nx .and. jj >= jv_s2d0 .and. jj <= jv_s2d1) sb = ior(sb, s2d)
        if (ii >= 1 .and. ii <= nx .and. jj >= jv_pre0 .and. jj <= ny) sb = ior(sb, int(RIV_PRE, ip))
        if (ii >= 1 .and. ii <= nx .and. jj >= 1 .and. jj <= ny) sb = ior(sb, int(RIV_COR, ip))
      endif
      nf = nf + 1
      fi(nf) = ii; fj(nf) = jj; ft(nf) = typ; fs(nf) = sb
      fglob(nf) = iriver
      ffrac(nf) = flx - 10 * iriver
    end subroutine add_face

  end subroutine init_rivers

  subroutine read_window(ncid, name, dest, nx, ny)
    ! dest(0:nx+1, 0:ny+1) = file(tile_fi0 + 0:nx+1, tile_fj0 + 0:ny+1), as
    ! the rho fields of the grid (file dims xi_rho x eta_rho).
    integer,          intent(in)  :: ncid, nx, ny
    character(len=*), intent(in)  :: name
    real(rp),         intent(out) :: dest(0:nx+1, 0:ny+1)
    integer :: varid
    call riv_check(nf90_inq_varid(ncid, trim(name), varid), 'var '//trim(name))
    call riv_check(nf90_get_var(ncid, varid, dest, start=(/int(decomp%i0), int(decomp%j0)/), &
                  count=(/nx + 2, ny + 2/)), 'get '//trim(name))
  end subroutine read_window

  subroutine riv_check(ierr, where)
    integer,          intent(in) :: ierr
    character(len=*), intent(in) :: where
    if (ierr /= nf90_noerr) then
      write(*,'(a,a,a,a)') 'river_io: ', trim(where), ': ', trim(nf90_strerror(ierr))
      error stop 'river_io: netCDF error'
    endif
  end subroutine riv_check

  subroutine river_dims(ncid, nriver, ntracers)
    integer, intent(in)  :: ncid
    integer, intent(out) :: nriver, ntracers
    nriver = nc_dim_len(ncid, 'nriver')
    ntracers = nc_dim_len(ncid, 'ntracers')
  end subroutine river_dims

  subroutine sync_masks_to_host(rm, nx, ny)
    ! Host copy of rmask over the cells -1:nx+2 (the stencil of the face
    ! count), and the river masks zeroed beyond what the initial file set.
    real(rp), allocatable, intent(out) :: rm(:,:)
    integer, intent(in) :: nx, ny
    integer :: i, j
    call sync_arch()
    allocate(rm(-1:nx+2, -1:ny+2))
    rm = 0.0_rp
    do j = max(-1, int(decomp%js)), min(ny + 2, int(decomp%je))
      do i = max(-1, int(decomp%is)), min(nx + 2, int(decomp%ie))
        rm(i, j) = rmask(i, j)
      enddo
    enddo
  end subroutine sync_masks_to_host

  subroutine sort_catalog()
    integer :: i, j, f, r
    real(rp) :: tt
    do i = 2, ncat
      tt = cat_t(i); f = cat_f(i); r = cat_r(i)
      j = i - 1
      do while (j >= 1)
        if (cat_t(j) <= tt) exit
        cat_t(j + 1) = cat_t(j); cat_f(j + 1) = cat_f(j); cat_r(j + 1) = cat_r(j)
        j = j - 1
      enddo
      cat_t(j + 1) = tt; cat_f(j + 1) = f; cat_r(j + 1) = r
    enddo
  end subroutine sort_catalog

  subroutine set_rivers(stage)
    ! River data at the forcing time of frc_slot (0: time n for the
    ! predictor, 1: n+1/2 for the fast loop and the corrector): move the
    ! bracket as frc_io advance_series, then the weights as frc_io weights.
    integer, intent(in) :: stage
    real(rp) :: tneed, t1, t2, dtm, c1, c2
    integer :: rec, i, old1, old2, i1, i2
    logical :: keep
    if (.not. river_on) return
    tneed = frc_modtime()
    if (ncat >= 2 .and. cat_cycle > 0.0_rp) then
      ! climatology: records shifted by whole cycles (t1, t2 the shifted times)
      call cyc_bracket(cat_t, ncat, cat_cycle, tneed, i1, i2, t1, t2)
      if (i1 /= it1) then
        old2 = it2
        it1 = i1
        if (i1 == old2) then
          call rivers_copy_upper()
        else
          call load_record(it1, 1)
        endif
      endif
      if (i2 /= it2) then
        it2 = i2
        call load_record(it2, 2)
      endif
      dtm = t2 - t1
      if (abs(dtm) < 1.0e-12_rp) then
        c1 = 1.0_rp; c2 = 0.0_rp
      else
        c1 = (t2 - tneed) / dtm
        c2 = (tneed - t1) / dtm
      endif
    elseif (ncat >= 2) then
      keep = .false.
      if (it1 >= 1) keep = (cat_t(it2) >= tneed .and. cat_t(it1) <= tneed)
      if (.not. keep) then
      rec = 1
      do i = 1, ncat - 1
        if (cat_t(i) <= tneed .and. cat_t(i + 1) >= tneed) then
          rec = i
          exit
        endif
        if (cat_t(i) <= tneed) rec = i
      enddo
      rec = max(1, min(rec, ncat - 1))
      if (rec /= it1 .or. rec + 1 /= it2) then
        old1 = it1; old2 = it2
        it1 = rec; it2 = rec + 1
        if (it1 /= old1) then
          if (it1 == old2) then
            call rivers_copy_upper()
          else
            call load_record(it1, 1)
          endif
        endif
        if (it2 /= old2) call load_record(it2, 2)
      endif
      endif
      t1 = cat_t(it1); t2 = cat_t(it2)
      if (t1 > tneed + dt * sec2day .or. t2 < tneed - dt * sec2day) then
        write(*,'(a,3es16.8)') 'set_rivers: river records do not cover the model time ', t1, t2, tneed
        error stop 'set_rivers: time interpolation error'
      endif
      dtm = t2 - t1
      if (abs(dtm) < 1.0e-12_rp) then
        c1 = 1.0_rp; c2 = 0.0_rp
      else
        c1 = (t2 - tneed) / dtm
        c2 = (tneed - t1) / dtm
      endif
    else
      if (it1 == 0) then
        it1 = 1; it2 = 1
        call load_record(1, 1)
        call load_record(1, 2)
      endif
      c1 = 1.0_rp; c2 = 0.0_rp
    endif
    if (stage == 0) then
      call rivers_set_weights(RIV_PRE, c1, c2)
    else
      call rivers_set_weights(RIV_COR, c1, c2)
    endif
  end subroutine set_rivers

  subroutine load_record(idx, slot)
    ! river_volume(:, rec) and river_tracer(:, :, rec) of catalog entry idx,
    ! gathered for this tile's rivers into slot `slot`.
    integer, intent(in) :: idx, slot
    integer :: ncid, varid, l, k
    ncid = fncid(cat_f(idx))
    call riv_check(nf90_inq_varid(ncid, 'river_volume', varid), 'var river_volume')
    call riv_check(nf90_get_var(ncid, varid, vol_g, start=(/1, cat_r(idx)/), count=(/nriver_f, 1/)), &
                  'get river_volume')
    call riv_check(nf90_inq_varid(ncid, 'river_tracer', varid), 'var river_tracer')
    call riv_check(nf90_get_var(ncid, varid, trc_g, start=(/1, 1, cat_r(idx)/), &
                  count=(/nriver_f, ntr_f, 1/)), 'get river_tracer')
    do l = 1, int(nloc)
      vol_l(l) = vol_g(glob(l))
      do k = 1, ntr_f
        trc_l(k, l) = trc_g(glob(l), k)
      enddo
    enddo
    call rivers_set_record(slot, vol_l, trc_l)
  end subroutine load_record

end module river_io

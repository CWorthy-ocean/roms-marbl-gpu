module case_io
  ! Namelist + grid/init NetCDF. Host-side; one rank reads the whole tile.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use geom, only: setup_metrics, set_depth, dump_vic, set_HUV, omega
  use rho_eos_mod, only: rho_eos
  use nc_io
  use host_buf, only: host_read_2d, host_read_3d
  use par_read, only: pr_begin, pr_flush, pr_take, pr_end, pr_taking, pr_store_2d, pr_store_3d, pr_store_3d_pt
  use frc_io, only: read_forcing_namelist, init_forcing, set_forces, close_frc_cache, init_cflx_corr
  use bry_io, only: init_bry, set_bry_all
  use tide_io, only: read_tide_namelist, init_tides, set_tides
  use river_io, only: read_river_namelist, init_rivers, set_rivers
  use cdr_io, only: read_cdr_namelist, init_cdr
  use cdr_lite_io, only: read_cdr_lite_namelist, init_cdr_lite
  use cdr_lite_out, only: read_cdr_lite_output_namelist, set_cdr_lite_output_root, init_cdr_lite_output
  use random_out, only: read_random_namelist, set_random_output_root, init_random_output
  use cdr_gas_out, only: read_cdr_gas_output_namelist, set_cdr_gas_output_root, init_cdr_gas_output
  use cdr_out, only: read_cdr_output_namelist, set_cdr_output_root, init_cdr_output
  use upscale_out, only: read_upscale_namelist, set_upscale_root, init_upscale_output
  use his_io, only: read_output_namelist, read_refdate_namelist, set_output_root, init_output
  use avg_out, only: read_avg_namelist, set_avg_root, init_avg_output
  use extract_out, only: read_extract_namelist, init_extract
  use comm_mod, only: comm_rank, comm_size, comm_min_real
  use arch_mod, only: sync_arch
  use tile_balance, only: balance_cuts, uniform_cuts
  use halo_mod, only: halo_fill_2d, halo_fill_2d_s, halo_fill_3d_s, halo_fill_3d_t, halo_fill_3d_tp
  use formulas, only: rho0_eos => rho0
  use pflx_mod, only: calc_pflx, pflx_timescale, init_pflx, pflx_fill_slow_halos, u_slow, v_slow, p_slow
  use passive_mod, only: ntp, tp, read_passive_namelist, alloc_passive, init_passive, passive_name, &
       passive_hold, set_cdr_lite, passive_finish_list, ncdr, tnu2_tp, n_marbl, cdr0, passive_mirror
  use marbl_gpu, only: marbl_gpu_init_diagnostics
  use marbl_gpu, only: marbl_on, read_marbl_namelist, marbl_gpu_setup_tracers, marbl_gpu_init_forcing, &
                       marbl_gpu_load_state
  implicit none
  private
  logical :: passive_loaded = .false.
  ! file start indices of this tile's local point 0 (rho) / 1 (u,v): the same
  ! pair for every stagger; 1 for a single tile
  integer :: tile_fi0 = 1, tile_fj0 = 1
  integer :: ld_phase = 0        ! 1 = queue reads, 2 = take them (par_read)

  character(len=512), public :: case_nml = ''
  character(len=512), public :: case_grdname = ''
  character(len=512), public :: case_inifile = ''
  character(len=256), public :: case_title = 'roms_gpu'
  character(len=512), public :: case_output_root = './output'
  integer(ip), public :: case_llm = 0
  integer(ip), public :: case_mmm = 0
  integer(ip), public :: case_nz = 0
  integer(ip), public :: case_npx = 1
  integer(ip), public :: case_npy = 1
  ! Tile balance: 'wet' (default) places the tile cuts from the land mask
  ! so every tile carries about the same cost (wet columns plus
  ! case_tile_land_weight per land cell), 'uniform' uses equal widths,
  ! 'explicit' takes case_tile_xcut / case_tile_ycut from the namelist.
  character(len=16), public :: case_tile_balance = 'wet'
  real(rp), public :: case_tile_land_weight = 0.10_rp   ! measured best on Pac with dry skip (2026-10-08)
  ! Largest tile area allowed, as a multiple of the mean (memory scales with
  ! cells, not wet cells): the land weight is raised until the cuts comply.
  real(rp), public :: case_tile_max_cells = 0.0_rp   ! area cap (x mean), 0 = off
  ! Memory cap of the 'wet' balance (2026-10-08): estimated tile memory
  ! cells * mem_cell + wet * mem_wet must stay below max_mem (GB). 0 = auto:
  ! 92% of the smallest device memory; 0.532*nz kB per cell (the fields of
  ! the full tile, measured on Pac); 3*ntp*nz*bytes per wet column (the
  ! passive tracers, stored over wet columns only).
  real(rp), public :: case_tile_max_mem_gb = 0.0_rp
  real(rp), public :: case_tile_mem_cell_kb = 0.0_rp, case_tile_mem_wet_kb = 0.0_rp
  integer(ip), public :: case_ntp_est = 0
  integer(ip), public :: case_tile_xcut(64) = 0, case_tile_ycut(64) = 0
  integer(ip), public :: case_record = 1
  integer(ip), public :: case_nrrec = -1
  logical,     public :: case_exact_restart = .false.
  logical :: ncid_has_uslow = .false., ncid_has_vslow = .false.   ! calc_pflx restart fields present
  logical :: case_have_prev = .false.
  logical,     public :: case_periodic_x = .false.
  logical,     public :: case_periodic_y = .false.
  real(rp),    public :: case_v_sponge = 0.0_rp
  ! UCLA ROMS reads akt_bak and tnu2 with one value per tracer (NT long), in
  ! its tracer order (T, S, nt_passive generic, CDR_LITE, MARBL; tracers.F90),
  ! and C-Star writes them so. A fixed, generous length accepts any such list
  ! however the tracers are counted. Entries not given are 0, as on the CPU.
  integer, parameter :: nml_ntrc_max = 512
  real(rp) :: case_tnu2(nml_ntrc_max) = 0.0_rp   ! &tracer_diff2 tnu2 (TS_DIF2 background)
  integer(ip) :: case_nt_passive = 0             ! param_settings nt_passive (the CPU's count)

  public :: read_namelist
  public :: load_case
  public :: report_loaded_case
  public :: write_toy_case
  public :: sanitize_land

contains

  subroutine read_namelist(path)
    character(len=*), intent(in) :: path
    integer :: unit, ios
    character(len=512) :: msg, nml_dir

    integer(ip) :: llm, mmm, nz, n, np_xi, np_eta, nt_passive, nt_bgc, ntrc_bio, nt_cdr_oae, nt_cdr_dor
    integer(ip) :: nsub_x, nsub_e, ndtfast_in, ntimes_in, ninfo, nrrec
    real(rp) :: dt_in, theta_s_in, theta_b_in, hc_in
    real(rp) :: zob_in, rdrg_in, rdrg2_in, visc2_in, akv_bak_in, rho0_in
    real(rp) :: akt_bak_in(2)
    character(len=512) :: grdname, inifile, ininame, title, output_root_name

    namelist /param_settings/ np_xi, np_eta, nsub_x, nsub_e, llm, mmm, nz, n, &
                              nt_passive, nt_bgc, ntrc_bio, nt_cdr_oae, nt_cdr_dor
    namelist /grid_settings/ grdname
    namelist /initial_conditions/ inifile, ininame, nrrec
    namelist /simulation_name_settings/ output_root_name, title
    namelist /ubind_settings/ ubind
    real(rp) :: v_sponge
    namelist /v_sponge_settings/ v_sponge
    namelist /calc_pflx_settings/ calc_pflx, pflx_timescale

    llm = 32; mmm = 32; nz = 0; n = 0
    np_xi = 1; np_eta = 1; nsub_x = 1; nsub_e = 1
    nt_passive = 0; nt_bgc = 0; ntrc_bio = 0; nt_cdr_oae = 0; nt_cdr_dor = 0; nrrec = -1
    ininame = ''
    dt_in = 100.0_rp; ndtfast_in = 60; ntimes_in = 1; ninfo = 1
    theta_s_in = 6.0_rp; theta_b_in = 6.0_rp; hc_in = 250.0_rp
    zob_in = 0.01_rp; rdrg_in = 0.0_rp; rdrg2_in = 0.0_rp; visc2_in = 0.0_rp
    akv_bak_in = 0.0_rp; akt_bak_in = 0.0_rp; rho0_in = 1025.0_rp
    grdname = ''; inifile = ''
    title = 'roms_gpu'; output_root_name = './output'

    case_nml = trim(path)
    nml_dir = dirname(path)

    open(newunit=unit, file=trim(path), status='old', action='read', iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'cannot open namelist ', trim(path), ': ', trim(iomsg_text(msg))
      error stop 'read_namelist'
    endif

    rewind(unit); read(unit, nml=param_settings, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'PARAM_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_namelist'
    endif
    call set_cdr_lite(nt_cdr_oae, nt_cdr_dor)
    case_nt_passive = max(nt_passive, 0_ip)

    rewind(unit); read(unit, nml=grid_settings, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'GRID_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_namelist'
    endif

    rewind(unit); read(unit, nml=initial_conditions, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'INITIAL_CONDITIONS: ', trim(iomsg_text(msg))
      error stop 'read_namelist'
    endif

    call read_s_coord(unit, theta_s_in, theta_b_in, hc_in)
    call read_time_stepping(unit, dt_in, ndtfast_in, ntimes_in, ninfo)
    ninfo_steps = max(ninfo, 1)
    call read_optional_scalars(unit, zob_in, rdrg_in, rdrg2_in, visc2_in, &
                               akv_bak_in, akt_bak_in, rho0_in)
    rewind(unit); read(unit, nml=ubind_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'UBIND_SETTINGS')
    v_sponge = 0.0_rp
    rewind(unit); read(unit, nml=v_sponge_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'V_SPONGE_SETTINGS')
    if (ios /= 0) v_sponge = 0.0_rp
    case_v_sponge = v_sponge
    ! pressure fluxes for nesting (optional group; absent: off)
    calc_pflx = .false.; pflx_timescale = 0.0_rp
    rewind(unit); read(unit, nml=calc_pflx_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'CALC_PFLX_SETTINGS')
    if (ios /= 0) calc_pflx = .false.
    call read_forcing_namelist(unit, nml_dir)
    call read_tide_namelist(unit)
    call read_refdate_namelist(unit)
    call read_output_namelist(unit)
    call read_avg_namelist(unit)
    call read_extract_namelist(unit)
    call read_passive_namelist(unit, nt_passive)
    call read_marbl_namelist(unit, nml_dir, nt_bgc)
    call read_river_namelist(unit)
    call read_cdr_namelist(unit)
    call read_cdr_lite_namelist(unit)
    call read_cdr_lite_output_namelist(unit)
    call read_random_namelist(unit)
    call read_cdr_gas_output_namelist(unit)
    call read_cdr_output_namelist(unit)
    call read_upscale_namelist(unit)
    case_ntp_est = nt_bgc + ntp + ncdr        ! MARBL + generic + CDR_LITE (ntp = nt_passive here)

    rewind(unit); read(unit, nml=simulation_name_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'SIMULATION_NAME_SETTINGS')
    call read_tile_settings(unit)   ! optional group; before the unit is closed
    close(unit)

    if (nz <= 0) nz = n
    if (nz <= 0) nz = 16
    if (len_trim(inifile) == 0) inifile = ininame
    case_llm = llm
    case_mmm = mmm
    case_nz = nz
    case_npx = np_xi
    case_npy = np_eta
    case_grdname = resolve_path(nml_dir, grdname)
    case_inifile = resolve_path(nml_dir, inifile)
    case_nrrec = nrrec
    rho0_case = rho0_in
    if (abs(rho0_in - rho0_eos) > 1.0e-6_rp) then
      write(*,'(a,f10.4,a,f10.4,a)') 'case_io: namelist rho0 =', rho0_in, &
           ' but build uses rho0 =', rho0_eos, '; rebuild with -DROMS_RHO0=<value>'
      error stop 1
    endif
    case_title = trim(title)
    case_output_root = resolve_path(nml_dir, output_root_name)
    call set_cdr_lite_output_root(case_output_root)
    call set_random_output_root(case_output_root)
    call set_cdr_gas_output_root(case_output_root)
    call set_cdr_output_root(case_output_root)
    call set_upscale_root(case_output_root)
    theta_s = theta_s_in
    theta_b = theta_b_in
    hc = hc_in
    zob = zob_in
    rdrg = rdrg_in
    rdrg2 = rdrg2_in
    visc2 = visc2_in
    akv_bak = akv_bak_in
    akt_bak = akt_bak_in

    call apply_decomp(case_periodic_x, case_periodic_y)
    call init_time_state(nz, ndtfast_in, dt_in, ntimes_in, 2_ip)
  end subroutine read_namelist

  subroutine read_tile_settings(unit)
    ! Optional &tile_settings group (absent = 'wet' balance, land weight 0.25).
    integer, intent(in) :: unit
    integer :: ios
    character(len=512) :: msg
    character(len=16) :: tile_balance
    real(rp) :: tile_land_weight, tile_max_cells_ratio, tile_max_mem_gb, tile_mem_cell_kb, tile_mem_wet_kb
    integer(ip) :: tile_xcut(64), tile_ycut(64)
    namelist /tile_settings/ tile_balance, tile_land_weight, tile_xcut, tile_ycut, tile_max_cells_ratio, &
         tile_max_mem_gb, tile_mem_cell_kb, tile_mem_wet_kb
    tile_balance = case_tile_balance; tile_land_weight = case_tile_land_weight
    tile_max_cells_ratio = case_tile_max_cells
    tile_max_mem_gb = case_tile_max_mem_gb
    tile_mem_cell_kb = case_tile_mem_cell_kb; tile_mem_wet_kb = case_tile_mem_wet_kb
    tile_xcut = 0; tile_ycut = 0
    rewind(unit)
    read(unit, nml=tile_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'TILE_SETTINGS')
    if (ios == 0) then
      case_tile_balance = tile_balance
      case_tile_land_weight = tile_land_weight
      case_tile_max_cells = tile_max_cells_ratio
      if (case_tile_max_cells > 0.0_rp) case_tile_max_cells = max(case_tile_max_cells, 1.0_rp)
      case_tile_max_mem_gb = tile_max_mem_gb
      case_tile_mem_cell_kb = tile_mem_cell_kb; case_tile_mem_wet_kb = tile_mem_wet_kb
      case_tile_xcut = tile_xcut; case_tile_ycut = tile_ycut
    endif
  end subroutine read_tile_settings

  subroutine set_tnu2()
    ! TS_DIF2 background diffusivity, CPU init_arrays_mixing: diff2 =
    ! tnu2(itrc) for every tracer; apply_sponge then adds the sponge ramp
    ! (CPU set_nudgcof). T and S: diff2(:, :, itrc). Passives: tnu2_tp(itp),
    ! the entry of the CPU tracer each one is (passive.F90 tp layout). The
    ! generic passives beyond param_settings' nt_passive (the GPU-only
    ! &passive_tracers n_passive) are not CPU tracers and take salinity's
    ! tnu2, as does tracer 1 of passive_init = 'mirror_salt', which mirrors
    ! salinity.
    integer(ip) :: itrc, itp, k, m
    do itrc = 1, int(size(diff2, 3), ip)
      diff2(:, :, itrc) = case_tnu2(itrc)
    enddo
    do itp = 1, ntp
      if (itp > cdr0) then                     ! CDR_LITE
        k = 2 + case_nt_passive + (itp - cdr0)
      elseif (itp > n_marbl) then              ! generic
        m = itp - n_marbl
        if (m <= case_nt_passive .and. .not. (passive_mirror .and. m == 1)) then
          k = 2 + m
        else
          k = isalt
        endif
      else                                     ! MARBL
        k = 2 + case_nt_passive + ncdr + itp
      endif
      if (k <= nml_ntrc_max) tnu2_tp(itp) = case_tnu2(k)
    enddo
    if (comm_rank() == 0 .and. (any(case_tnu2 /= 0.0_rp))) then
      write(*,'(a,2es12.4)') 'tnu2 (T, S) =', case_tnu2(1:2)
      do itp = 1, ntp
        write(*,'(a,i0,a,a,a,es12.4)') 'tnu2 passive ', itp, ' (', trim(passive_name(itp)), ') =', tnu2_tp(itp)
      enddo
    endif
  end subroutine set_tnu2

  subroutine apply_sponge()
    ! CPU set_nudgcof (SPONGE, UV_VIS2, TS_DIF2, not PACIFIC): linear ramp
    ! over isp = sponge_size+1 points from every open edge (the CPU's
    ! OBC_WEST/EAST/SOUTH/NORTH; here obc_*, set by init_bry and init_tides).
    ! A closed edge gets no sponge.
    integer(ip), parameter :: sponge_size = 15_ip
    integer(ip) :: i, j, ig, jg, isp, ibnd, itrc, i_lo, i_hi, j_lo, j_hi
    real(rp), allocatable :: wrk(:,:)

    if (case_v_sponge <= 0.0_rp) return
    isp = sponge_size + 1_ip
    i_lo = lbound(visc2_r, 1); i_hi = ubound(visc2_r, 1)
    j_lo = lbound(visc2_r, 2); j_hi = ubound(visc2_r, 2)
    allocate(wrk(i_lo:i_hi, j_lo:j_hi))
    do j = j_lo, j_hi
      jg = decomp%j0 + j - 1_ip
      do i = i_lo, i_hi
        ig = decomp%i0 + i - 1_ip
        ibnd = isp
        if (obc_west)  ibnd = min(ibnd, ig)
        if (obc_east)  ibnd = min(ibnd, decomp%gnx + 1_ip - ig)
        if (obc_south) ibnd = min(ibnd, jg)
        if (obc_north) ibnd = min(ibnd, decomp%gny + 1_ip - jg)
        wrk(i, j) = real(isp - ibnd, rp) / real(isp, rp)
      enddo
    enddo

    visc2_r = visc2_r + case_v_sponge * wrk
    do j = j_lo + 1, j_hi
      do i = i_lo + 1, i_hi
        visc2_p(i, j) = visc2_p(i, j) + 0.25_rp * case_v_sponge * &
             (wrk(i, j) + wrk(i - 1, j) + wrk(i, j - 1) + wrk(i - 1, j - 1))
      enddo
    enddo
    do itrc = 1, size(diff2, 3)
      diff2(:, :, itrc) = diff2(:, :, itrc) + case_v_sponge * wrk
    enddo
    diff2_sp = case_v_sponge * wrk    ! the passives add it to their tnu2
    deallocate(wrk)
    if (comm_rank() == 0) write(*,'(a,i3,a,es12.4)') 'Sponge layer = ', sponge_size, &
         ' interior grid points, v_sponge =', case_v_sponge
  end subroutine apply_sponge

  subroutine apply_decomp(periodic_x, periodic_y)
    logical, intent(in) :: periodic_x, periodic_y
    integer(ip) :: npx, npy, nproc
    integer(ip) :: xcut(0:64), ycut(0:64)
    nproc = comm_size()
    npx = case_npx
    npy = case_npy
    if (nproc == 1) then
      if (npx * npy > 1 .and. comm_rank() == 0) then
        write(*,'(a,2i4,a)') 'note: namelist np_xi,np_eta =', npx, npy, &
             ' but 1 MPI rank; using 1 x 1'
      endif
      npx = 1_ip
      npy = 1_ip
    elseif (nproc /= npx * npy) then
      write(*,*) 'MPI size', nproc, '/= np_xi*np_eta', npx * npy
      error stop 'apply_decomp'
    endif
    if (npx * npy == 1) then
      call init_decomp(case_llm, case_mmm, case_nz, 2_ip, npx, npy, comm_rank(), &
                       periodic_x, periodic_y)
    else
      call tile_cuts(npx, npy, xcut, ycut)
      call init_decomp(case_llm, case_mmm, case_nz, 2_ip, npx, npy, comm_rank(), &
                       periodic_x, periodic_y, xcut(0:npx), ycut(0:npy))
    endif
    tile_fi0 = int(decomp%i0); tile_fj0 = int(decomp%j0)
  end subroutine apply_decomp

  subroutine tile_cuts(npx, npy, xcut, ycut)
    ! Tile cut positions (global interior indices) for npx x npy tiles from
    ! the land mask of the grid file, equal widths, or the namelist.
    integer(ip), intent(in)  :: npx, npy
    integer(ip), intent(out) :: xcut(0:64), ycut(0:64)
    integer(ip), parameter :: minw = 2_ip * 2_ip + 2_ip   ! 2*halo + 2
    integer(ip) :: gnx, gny, i, j, k, l, ncells, nwet
    integer :: ncid
    real(rp), allocatable :: mask(:,:), cost(:,:)
    real(rp) :: wland, tcost, cmax, cmean
    real(rp) :: ratio, memlim, mcell, mwet, memmax, mem
    integer :: cellmax
    character(len=16) :: mode
    gnx = case_llm; gny = case_mmm
    mode = case_tile_balance
    xcut = 0; ycut = 0
    allocate(mask(gnx + 2, gny + 2), cost(gnx, gny))
    mask = 1.0_rp
    call nc_open_read(case_grdname, ncid)
    if (nc_has_var(ncid, 'mask_rho')) then
      call nc_read_2d(ncid, 'mask_rho', mask, int(gnx) + 2, int(gny) + 2)
    elseif (mode == 'wet') then
      mode = 'uniform'
    endif
    call nc_close(ncid)
    wland = case_tile_land_weight
    select case (trim(mode))
    case ('explicit')
      xcut(0) = 0; xcut(npx) = gnx; ycut(0) = 0; ycut(npy) = gny
      xcut(1:npx - 1) = case_tile_xcut(1:npx - 1)
      ycut(1:npy - 1) = case_tile_ycut(1:npy - 1)
      do k = 1, npx
        if (xcut(k) - xcut(k - 1) < minw) error stop 'tile_settings: tile_xcut gives a tile narrower than 2*halo+2'
      enddo
      do k = 1, npy
        if (ycut(k) - ycut(k - 1) < minw) error stop 'tile_settings: tile_ycut gives a tile lower than 2*halo+2'
      enddo
    case ('uniform')
      call uniform_cuts(gnx, npx, xcut(0:npx))
      call uniform_cuts(gny, npy, ycut(0:npy))
    case default   ! 'wet'
      ! Balance the wet-weighted cost, but keep every tile's estimated GPU
      ! memory below the limit (and, when case_tile_max_cells is set, its
      ! area within that multiple of the mean): raise the land weight
      ! until both hold.
      call tile_mem_model(memlim, mcell, mwet)
      do
        do j = 1, gny
          do i = 1, gnx
            if (mask(i + 1, j + 1) > 0.5_rp) then
              cost(i, j) = 1.0_rp
            else
              cost(i, j) = wland
            endif
          enddo
        enddo
        call balance_cuts(cost, gnx, gny, npx, npy, minw, xcut(0:npx), ycut(0:npy))
        cellmax = 0
        memmax = 0.0_rp
        do l = 1, npy
          do k = 1, npx
            ncells = (xcut(k) - xcut(k - 1)) * (ycut(l) - ycut(l - 1))
            cellmax = max(cellmax, int(ncells))
            nwet = 0
            do j = ycut(l - 1) + 1, ycut(l)
              do i = xcut(k - 1) + 1, xcut(k)
                if (mask(i + 1, j + 1) > 0.5_rp) nwet = nwet + 1
              enddo
            enddo
            mem = (mcell * real(ncells, rp) + mwet * real(nwet, rp)) * 1.0e-6_rp   ! GB
            memmax = max(memmax, mem)
          enddo
        enddo
        ratio = real(cellmax, rp) * real(npx * npy, rp) / real(gnx * gny, rp)
        if ((case_tile_max_cells <= 0.0_rp .or. ratio <= case_tile_max_cells) .and. &
            (memlim <= 0.0_rp .or. memmax <= memlim)) exit
        if (wland >= 1.0_rp) exit
        wland = min(1.0_rp, wland * 1.5_rp)
      enddo
      if (comm_rank() == 0) then
        write(*,'(a,f7.2,a,f7.2,a,f7.2,a,f7.2,a)') 'tiles: memory model ', mcell, ' kB per cell + ', mwet, &
             ' kB per wet cell; largest tile ', memmax, ' GB (limit ', memlim, ' GB)'
        if (wland /= case_tile_land_weight) &
             write(*,'(a,f6.3,a,f6.3,a)') 'tiles: land weight raised to ', wland, &
             ' to fit the memory / area limits (largest tile area ', ratio, ' x mean)'
        if (memlim > 0.0_rp .and. memmax > memlim) &
             write(*,'(a)') 'tiles: WARNING the estimated memory of the largest tile exceeds the limit'
      endif
    end select
    if (comm_rank() == 0) then
      write(*,'(a,a,a,f6.3)') 'tiles: balance = ', trim(mode), '  land weight = ', wland
      write(*,'(a,64i6)') 'tiles: x cuts =', xcut(0:npx)
      write(*,'(a,64i6)') 'tiles: y cuts =', ycut(0:npy)
      cmax = 0.0_rp; cmean = 0.0_rp
      do l = 1, npy
        do k = 1, npx
          ncells = (xcut(k) - xcut(k - 1)) * (ycut(l) - ycut(l - 1))
          nwet = 0
          do j = ycut(l - 1) + 1, ycut(l)
            do i = xcut(k - 1) + 1, xcut(k)
              if (mask(i + 1, j + 1) > 0.5_rp) nwet = nwet + 1
            enddo
          enddo
          tcost = real(nwet, rp) + wland * real(ncells - nwet, rp)
          cmax = max(cmax, tcost); cmean = cmean + tcost / real(npx * npy, rp)
          write(*,'(a,i3,a,i4,a,i4,a,i8,a,i8,a,f10.1)') 'tiles: rank', (l - 1) * npx + k - 1, &
               '  nx =', xcut(k) - xcut(k - 1), '  ny =', ycut(l) - ycut(l - 1), &
               '  cells =', ncells, '  wet =', nwet, '  cost =', tcost
        enddo
      enddo
      write(*,'(a,f8.4)') 'tiles: max/mean cost =', cmax / max(cmean, 1.0e-30_rp)
    endif
    deallocate(mask, cost)
  end subroutine tile_cuts

  subroutine tile_mem_model(memlim, mcell, mwet)
    ! Memory limit (GB) and model (kB per tile cell, kB per wet cell) of the
    ! 'wet' balance; namelist values when set, otherwise the defaults above.
    ! The limit is the same on every rank (minimum over the ranks).
#ifdef USE_CUDA
    use cudafor
    integer(kind=cuda_count_kind) :: freeb, totalb
    integer :: ierr
#endif
    real(rp), intent(out) :: memlim, mcell, mwet
    real(rp) :: dev
    mcell = case_tile_mem_cell_kb
    if (mcell <= 0.0_rp) mcell = 0.532_rp * real(case_nz, rp)
    mwet = case_tile_mem_wet_kb
    if (mwet <= 0.0_rp) mwet = 3.0_rp * real(case_ntp_est, rp) * real(case_nz, rp) * &
         real(storage_size(1.0_pp) / 8, rp) / 1000.0_rp
    memlim = case_tile_max_mem_gb
    if (memlim <= 0.0_rp) then
      dev = 0.0_rp
#ifdef USE_CUDA
      ierr = cudaMemGetInfo(freeb, totalb)
      if (ierr == 0) dev = real(totalb, rp) * 1.0e-9_rp
#endif
      memlim = 0.92_rp * dev
    endif
    memlim = comm_min_real(memlim)
  end subroutine tile_mem_model

  subroutine read_s_coord(unit, theta_s_in, theta_b_in, hc_in)
    integer, intent(in) :: unit
    real(rp), intent(inout) :: theta_s_in, theta_b_in, hc_in
    integer :: ios
    character(len=512) :: msg
    real(rp) :: theta_s, theta_b, hc
    namelist /s_coord/ theta_s, theta_b, hc
    theta_s = theta_s_in; theta_b = theta_b_in; hc = hc_in
    rewind(unit)
    read(unit, nml=s_coord, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'S_COORD')
    if (ios == 0) then
      theta_s_in = theta_s
      theta_b_in = theta_b
      hc_in = hc
    endif
  end subroutine read_s_coord

  subroutine read_time_stepping(unit, dt_in, ndtfast_in, ntimes_in, ninfo)
    integer, intent(in) :: unit
    real(rp), intent(inout) :: dt_in
    integer(ip), intent(inout) :: ndtfast_in, ntimes_in, ninfo
    integer :: ios
    character(len=512) :: msg
    real(rp) :: dt
    integer(ip) :: ndtfast, ntimes
    namelist /time_stepping/ dt, ndtfast, ntimes, ninfo
    dt = dt_in; ndtfast = ndtfast_in; ntimes = ntimes_in
    rewind(unit)
    read(unit, nml=time_stepping, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'TIME_STEPPING missing or invalid: ', trim(iomsg_text(msg))
      error stop 'read_namelist'
    endif
    dt_in = dt
    ndtfast_in = ndtfast
    ntimes_in = ntimes
  end subroutine read_time_stepping

  subroutine read_optional_scalars(unit, zob_in, rdrg_in, rdrg2_in, visc2_in, &
                                   akv_bak_in, akt_bak_in, rho0_in)
    integer, intent(in) :: unit
    real(rp), intent(inout) :: zob_in, rdrg_in, rdrg2_in, visc2_in, akv_bak_in
    real(rp), intent(inout) :: akt_bak_in(2), rho0_in
    ! akt_bak and tnu2: one value per tracer (see case_tnu2). Only
    ! akt_bak(1:2) (T, S) is kept: entries beyond T and S are ignored, as on
    ! the CPU, where Akt is allocated to isalt and the other tracers mix
    ! with salinity's diffusivity. tnu2 goes to case_tnu2 (set_tnu2).
    integer :: ios
    character(len=512) :: msg
    real(rp) :: rdrg, rdrg2, zob, visc2, akv_bak, rho0
    real(rp) :: akt_bak(nml_ntrc_max), tnu2(nml_ntrc_max)
    namelist /bottom_drag_settings/ rdrg, rdrg2, zob
    namelist /vertical_mixing_settings/ akv_bak, akt_bak
    namelist /lateral_visc_settings/ visc2
    namelist /tracer_diff2/ tnu2
    namelist /rho0_settings/ rho0
    rdrg = rdrg_in; rdrg2 = rdrg2_in; zob = zob_in
    visc2 = visc2_in; akv_bak = akv_bak_in
    akt_bak = 0.0_rp; akt_bak(1:2) = akt_bak_in
    tnu2 = 0.0_rp; rho0 = rho0_in
    rewind(unit); read(unit, nml=bottom_drag_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'BOTTOM_DRAG_SETTINGS')
    rewind(unit); read(unit, nml=vertical_mixing_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'VERTICAL_MIXING_SETTINGS')
    rewind(unit); read(unit, nml=lateral_visc_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'LATERAL_VISC_SETTINGS')
    rewind(unit); read(unit, nml=tracer_diff2, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'TRACER_DIFF2')
    rewind(unit); read(unit, nml=rho0_settings, iostat=ios, iomsg=msg)
    call nml_stop(ios, msg, 'RHO0_SETTINGS')
    zob_in = zob; rdrg_in = rdrg; rdrg2_in = rdrg2; visc2_in = visc2
    akv_bak_in = akv_bak; akt_bak_in = akt_bak(1:2)
    case_tnu2 = tnu2
    rho0_in = rho0
  end subroutine read_optional_scalars

  subroutine load_case(nml_path)
    character(len=*), intent(in) :: nml_path
    integer :: ncid, xi_rho, eta_rho, xi_u, eta_v, s_rho
    integer :: i0, j0, nxi, neta
    integer(ip) :: k
    logical :: had_f, had_lat, passive_bg
    real(rp), parameter :: deg2rad = 3.14159265358979323846_rp / 180.0_rp
    real(rp), parameter :: omega_earth = 4.0_rp * 3.14159265358979323846_rp / (23.9344699_rp * 3600.0_rp)

    call read_namelist(nml_path)
    call alloc_state(nt)
    call init_pflx()                    ! calc_pflx: running means (restart fields) and fluxes
    if (marbl_on) call marbl_gpu_setup_tracers()
    call passive_finish_list()          ! without MARBL: generic + CDR_LITE tracers

    call nc_open_read(case_grdname, ncid)
    xi_rho = nc_dim_len(ncid, 'xi_rho')
    eta_rho = nc_dim_len(ncid, 'eta_rho')
    call infer_layout(xi_rho, eta_rho, i0, j0, nxi, neta)
    write(*,'(a,a)') 'grid file: ', trim(case_grdname)
    write(*,'(a,2i6,a,2i4)') '  xi_rho, eta_rho =', xi_rho, eta_rho, '  mapped i0,j0 =', i0, j0

    ! The static fields through the collective reader (whole-variable reads
    ! spread over the ranks, windows redistributed): phase 1 queues, phase 2 stores.
    do ld_phase = 1, 2
      if (ld_phase == 1) then
        call pr_begin()
      else
        call pr_take()
      endif
      call pr_store_2d(ncid, 'h', h(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      call pr_store_2d(ncid, 'pm', pm(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      call pr_store_2d(ncid, 'pn', pn(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      if (nc_has_var(ncid, 'mask_rho')) then
        call pr_store_2d(ncid, 'mask_rho', rmask(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      else
        rmask = 1.0_rp
      endif
      ! As roms_bugfix with SPHERICAL: f comes from lat_rho and the sidereal
      ! day, not from the grid file (whose f may use a 24 h day).
      had_lat = nc_has_var(ncid, 'lat_rho')
      if (had_lat) then
        call pr_store_2d(ncid, 'lat_rho', latr(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      endif
      had_f = nc_has_var(ncid, 'f') .and. .not. had_lat
      if (had_f) then
        call pr_store_2d(ncid, 'f', f(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      endif
      if (nc_has_var(ncid, 'lon_rho')) then
        call pr_store_2d(ncid, 'lon_rho', lonr(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      endif
      if (nc_has_var(ncid, 'angle')) then
        call pr_store_2d(ncid, 'angle', angler(decomp%is, decomp%js), i0, j0, nxi, neta, i0f=tile_fi0, j0f=tile_fj0)
      endif
      if (ld_phase == 1) call pr_flush(ncid)
    enddo
    call pr_end()
    call nc_close(ncid)
    call wall_stamp('grid read')
    if (.not. had_f) then
      f(i0:i0+nxi-1, j0:j0+neta-1) = omega_earth * &
           sin(deg2rad * latr(i0:i0+nxi-1, j0:j0+neta-1))
    endif

    call sanitize_land()
    call pad_physical_halo()
    call fill_static_halos()
    ! setup_metrics, the mask builders and build_tp_columns are host loops
    ! over pm/pn/h/rmask: wait for the sanitize/pad kernels (on one tile
    ! without neighbours no halo exchange synchronizes them).
    call sync_arch()
    call setup_metrics()
    call fill_metric_halos()
    call build_wet_list()
    call build_dry_map()
    ! The passives are stored over wet columns (and the boundary rows) only,
    ! so they are allocated once the mask is final.
    call build_tp_columns()
    call alloc_passive()
    if (ntp > 0) write(*,'(a,i0,a,i0,a,i0,a)') 'passive storage: rank ', comm_rank(), '  columns ', ntpcol, &
         ' of ', (decomp%ie - decomp%is + 1) * (decomp%je - decomp%js + 1), ' (wet + boundary rows)'
    write(*,'(a,i0,a,i0,a,i0,a,i0)') 'dry skip: rank ', comm_rank(), '  wet ', nwet, &
         '  deep-land ', ndeep, '  radius ', dry_radius
    visc2_r = visc2
    visc2_p = visc2
    call set_tnu2()

    call nc_open_read(case_inifile, ncid)
    write(*,'(a,a)') 'init file: ', trim(case_inifile)
    ! tile extents of the staggered fields (the file dims are global)
    xi_u = merge(nxi, nxi - 1, case_periodic_x)
    eta_v = merge(neta, neta - 1, case_periodic_y)
    s_rho = nc_dim_len(ncid, 's_rho')
    if (s_rho /= decomp%nz) then
      write(*,*) 's_rho', s_rho, '/= namelist nz', decomp%nz
      error stop 'load_case'
    endif
    call load_init_records(ncid, i0, j0, nxi, neta, xi_u, eta_v)
    call nc_close(ncid)
    call mask_loaded_velocities()
    call fill_masked_velocity_halos()
    if (calc_pflx) call init_pflx_slow(ncid_has_uslow, ncid_has_vslow)
    call wall_stamp('initial state read')

    call sanitize_land()
    call pad_physical_halo()
    call fill_static_halos()
    ! Slot 3 is the nnew workspace. CPU leaves it at 0 after get_init;
    ! Orlanski vin on the first step is that zero, not a copy of nstp.
    ! Slot 2 is n-1 only when both records were restored; otherwise copy
    ! the current state into the AB/AM slots.
    call copy_initial_slots(.not. case_have_prev)
    if (.not. case_exact_restart) call init_barotropic_fluxes()
    knew = 1
    nstp = 1
    nrhs = 1
    tdays = start_time * sec2day
    call dump_vic('after load')
    if (.not. passive_loaded) call init_passive()
    if (marbl_on .and. .not. passive_loaded) error stop 'MARBL: initial file lacks the MARBL tracers'
    call passive_hold(0_ip)   ! use_constant_Fe (no-op otherwise): Fe at its value from the start
    call init_forcing()
    call wall_stamp('forcing init')
    if (marbl_on) call marbl_gpu_init_forcing()
    call wall_stamp('marbl forcing init')
    call init_bry()
    call wall_stamp('boundary init')
    call init_tides()
    call wall_stamp('tide init')
    ! the closed sides are known from here on (obc_*)
    call zero_closed_wall_velocities()
    ! the sponge follows the open sides (obc_*), known from here on
    call apply_sponge()
    passive_bg = .false.
    if (ntp > 0) passive_bg = any(tnu2_tp /= 0.0_rp)
    call build_mix_map(passive_bg)
    write(*,'(a,i0,a,i0,a,i0)') 'mixing skip: rank ', comm_rank(), '  active wet cells ', nmix, ' of ', nwet
    ! after the initial state: the river faces add to riv_umask/riv_vmask
    ! as read from the initial file (CPU: get_init, then calc_river_flux);
    ! before close_frc_cache: it scans the forcing files' open handles
    call init_rivers(case_grdname)
    call wall_stamp('river init')
    call init_cdr(case_grdname)       ! CDR releases (passive list and columns final)
    call wall_stamp('cdr init')
    call init_cdr_lite()              ! CDR_LITE surface forcing (binds forcing series)
    call init_cflx_corr()             ! CFLX_CORR tracers (after init_cdr_lite sizes cdrl_psf)
    call wall_stamp('cdr_lite init')
    call close_frc_cache()
    frc_slot = 0
    call set_forces()
    call set_rivers(0)                ! river volume at time n for set_HUV (CPU: set_forces)
    call set_bry_all()
    call set_tides()
    ! The CPU's start-up diagnostics before its initial history record
    ! (main.F90 after this set_forces: set_depth, set_HUV, omega,
    ! rho_eos): the initial record's rho, omega and w, and the depths the
    ! initial MARBL diagnostics need. Every step recomputes them first.
    iic = 0
    call set_depth()
    call set_HUV()
    call omega()
    call rho_eos(nstp)
    ! the CPU's initial MARBL diagnostics (its first wrt_bgc)
    if (marbl_on) call marbl_gpu_init_diagnostics()
    call set_output_root(case_output_root, case_title)
    call init_output()
    call set_avg_root(case_output_root)
    call init_avg_output()            ! before the outputs that write a double record at start-up
    call init_cdr_lite_output()       ! after the history/restart staging is sized
    call init_random_output()         ! surface passive tracers (random_output_settings)
    call init_cdr_gas_output()        ! writes its start (bracket) record
    call init_cdr_output()
    call init_upscale_output()        ! after init_bry (open sides)
    call init_extract()               ! child-grid boundary extraction
    call wall_stamp('output init')
  end subroutine load_case

  subroutine load_init_records(ncid, i0, j0, nxi, neta, xi_u, eta_v)
    ! UCLA EXACT_RESTART soft policy: if two consecutive records exist
    ! (ocean_time differs by dt, time_step consecutive when present) then
    ! restore 3-D u/v/t into slots 2 then 1. Coupling DU/DV_avg* from the
    ! last record are required to keep the exact (non-forward) first step.
    integer, intent(in) :: ncid, i0, j0, nxi, neta, xi_u, eta_v
    integer :: nrecs, rec, ibuf(8)
    real(rp) :: tbuf(1), time_bak, tscale
    logical :: times_ok, steps_ok, have_coup, has_tstep
    character(len=64) :: units

    case_exact_restart = .false.
    case_have_prev = .false.
    forw_start = 1
    ntstart = 1
    rec = 1
    nrecs = 1
    tscale = 1.0_rp
    have_coup = .false.
    has_tstep = .false.

    if (nc_has_var(ncid, 'ocean_time')) then
      nrecs = nc_var_nrecs(ncid, 'ocean_time')
      if (case_nrrec > 0) then
        rec = case_nrrec
        if (rec > nrecs) then
          write(*,*) 'requested restart record', rec, 'exceeds nrecs', nrecs
          error stop 'load_case'
        endif
      else
        rec = nrecs
      endif
      units = ''
      call nc_get_att_text(ncid, 'ocean_time', 'units', units)
      if (units(1:3) == 'day' .or. units(1:3) == 'Day') tscale = 86400.0_rp
      call nc_get_1d(ncid, 'ocean_time', tbuf, rec)
      start_time = tbuf(1) * tscale
      time = start_time

      times_ok = .false.
      steps_ok = .true.
      if (rec >= 2) then
        call nc_get_1d(ncid, 'ocean_time', tbuf, rec - 1)
        time_bak = tbuf(1) * tscale
        times_ok = (abs(start_time - time_bak - dt) < 0.01_rp * dt)
        if (.not. times_ok) then
          write(*,'(a,2i4,a)') 'WARNING: Exact restart is requested, but ', &
               rec - 1, rec, ' are not consecutive time steps; using forward start.'
        endif
      else
        write(*,'(a)') 'WARNING: Exact restart is requested, but the initial file ' // &
             'does not contain sufficient records.'
      endif

      has_tstep = nc_has_var(ncid, 'time_step')
      if (has_tstep) then
        call nc_get_ints(ncid, 'time_step', ibuf, rec)
        ntstart = ibuf(1) + 1
        if (times_ok) then
          ! CPU get_init: ntstart = time_step(n-1)+1, then require
          ! time_step(n) == ntstart, i.e. the two records differ by 1.
          call nc_get_ints(ncid, 'time_step', ibuf, rec - 1)
          steps_ok = (ibuf(1) + 1 == ntstart - 1)
          if (.not. steps_ok) then
            write(*,'(a,2i4,a)') 'WARNING: Exact restart is requested, but ', &
                 rec - 1, rec, ' are not consecutive time_step values; using forward start.'
          endif
        endif
      endif

      have_coup = has_coupling_fields(ncid)
      if (times_ok .and. steps_ok) then
        write(*,'(a,i0,a)') 'exact-restart: reading previous 3-D record ', rec - 1, ' into slot 2'
        case_have_prev = .true.
      endif
      if (times_ok .and. steps_ok .and. have_coup .and. has_tstep) then
        case_exact_restart = .true.
        forw_start = 1
      else
        forw_start = ntstart
        if (times_ok .and. steps_ok .and. .not. have_coup) then
          write(*,'(a)') 'WARNING: Exact restart cancelled: DU/DV_avg coupling fields missing.'
        endif
      endif
    else
      rec = 1
    endif
    case_record = rec

    write(*,'(a,i0,a,i0)') 'exact-restart: reading current record ', rec, ' into slot 1; nrecs=', nrecs
    ! All records through the collective reader: phase 1 queues every window,
    ! pr_flush reads whole variables in parallel and redistributes, phase 2
    ! stores the windows (same call sequence both times).
    do ld_phase = 1, 2
      if (ld_phase == 1) then
        call pr_begin()
      else
        call pr_take()
      endif
      if (case_have_prev) then
        call load_3d_record(ncid, rec - 1, 2, i0, j0, nxi, neta, xi_u, eta_v)
        call load_rufrc_slot(ncid, rec - 1, 2, i0, j0, nxi, neta, xi_u, eta_v)
      endif
      call load_2d_record(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
      call load_3d_record(ncid, rec, 1, i0, j0, nxi, neta, xi_u, eta_v)
      call load_rufrc_slot(ncid, rec, 1, i0, j0, nxi, neta, xi_u, eta_v)
      if (have_coup .or. has_coupling_fields(ncid)) then
        call load_coupling_fields(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
      endif
      if (calc_pflx) call load_pflx_slow(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
      if (ld_phase == 1) then
        call wall_stamp('ini queued')
        call pr_flush(ncid)
        call wall_stamp('ini read')
      endif
    enddo
    call pr_end()
    call wall_stamp('ini stored')
    if (case_exact_restart) then
      write(*,'(a,f12.4,a,i0,a,i0)') 'get_init :: Exact restart from day =', &
           start_time * sec2day, ' rec =', rec, ' ntstart=', ntstart
    elseif (nc_has_var(ncid, 'time_step')) then
      write(*,'(a,f12.4,a,i0,a,i0)') 'get_init :: Approximate, single-step restart from day =', &
           start_time * sec2day, ' rec =', rec, ' ntstart=', ntstart
    else
      write(*,'(a,f12.4,a,i0)') 'get_init :: Read initial conditions for day =', &
           start_time * sec2day, ' record =', rec
    endif
    call fill_loaded_halos()
    call wall_stamp('ini halos')
  end subroutine load_init_records

  logical function has_coupling_fields(ncid) result(ok)
    integer, intent(in) :: ncid
    ok = nc_has_var(ncid, 'DU_avg1') .and. nc_has_var(ncid, 'DV_avg1') .and. &
         nc_has_var(ncid, 'DU_avg2') .and. nc_has_var(ncid, 'DV_avg2') .and. &
         nc_has_var(ncid, 'DU_avg_bak') .and. nc_has_var(ncid, 'DV_avg_bak')
  end function has_coupling_fields

  subroutine load_2d_record(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, xi_u, eta_v
    call pr_store_2d(ncid, 'zeta', zeta(decomp%is, decomp%js, 1), i0, j0, nxi, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'ubar', ubar(decomp%is, decomp%js, 1), i0+1, j0, xi_u, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'vbar', vbar(decomp%is, decomp%js, 1), i0, j0+1, nxi, eta_v, rec, i0f=tile_fi0, j0f=tile_fj0)
    if (nc_has_var(ncid, 'hbls')) then
      call pr_store_2d(ncid, 'hbls', hbls(decomp%is, decomp%js), i0, j0, nxi, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    endif
    if (nc_has_var(ncid, 'hbbl')) then
      call pr_store_2d(ncid, 'hbbl', hbbl(decomp%is, decomp%js), i0, j0, nxi, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    endif
    ! River masks (CPU get_init: needed for exact restarts). Absent: 0.
    ! Clamped to [0, 1] as the CPU does (fill values on land), not masked
    ! with umask/vmask: river faces have umask = 0.
    if (nc_has_var(ncid, 'riv_umask')) then
      call pr_store_2d(ncid, 'riv_umask', riv_umask(decomp%is, decomp%js), i0+1, j0, xi_u, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
      if (pr_taking()) then
        where (riv_umask /= riv_umask .or. riv_umask < 0.0_rp .or. riv_umask > 1.0_rp) riv_umask = 0.0_rp
      endif
    endif
    if (nc_has_var(ncid, 'riv_vmask')) then
      call pr_store_2d(ncid, 'riv_vmask', riv_vmask(decomp%is, decomp%js), i0, j0+1, nxi, eta_v, rec, i0f=tile_fi0, j0f=tile_fj0)
      if (pr_taking()) then
        where (riv_vmask /= riv_vmask .or. riv_vmask < 0.0_rp .or. riv_vmask > 1.0_rp) riv_vmask = 0.0_rp
      endif
    endif
  end subroutine load_2d_record

  subroutine load_3d_record(ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v
    integer :: itp
    call pr_store_3d(ncid, 'u', u(decomp%is, decomp%js, 1, tindx), i0+1, j0, xi_u, neta, decomp%nz, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_3d(ncid, 'v', v(decomp%is, decomp%js, 1, tindx), i0, j0+1, nxi, eta_v, decomp%nz, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_3d(ncid, 'temp', t(decomp%is, decomp%js, 1, tindx, 1), i0, j0, nxi, neta, decomp%nz, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_3d(ncid, 'salt', t(decomp%is, decomp%js, 1, tindx, 2), i0, j0, nxi, neta, decomp%nz, rec, i0f=tile_fi0, j0f=tile_fj0)
    ! Passive tracers, when the file carries them (float32 in restarts).
    if (ntp > 0) then
      if (nc_has_var(ncid, trim(passive_name(1)))) then
        ! Gather all tracers in host memory in tp's (tracer, i, j, k) order and
        ! store them with one sweep: per-tracer stores into the managed array
        ! walk every page ntp times. A tracer the file lacks stays 0, as in
        ! the CPU model (e.g. CDR_LITE tracers in a file without them).
        do itp = 1, ntp
          if (.not. nc_has_var(ncid, trim(passive_name(itp)))) then
            if (pr_taking() .and. tindx == 1 .and. comm_rank() == 0) &
                 write(*,'(a,a,a)') '  --- ', trim(passive_name(itp)), ' not in initial file: 0'
            cycle
          endif
          call pr_store_3d_pt(ncid, trim(passive_name(itp)), tp(1, 0, 1, tindx), ntp, itp, &
                              i0, j0, nxi, neta, decomp%nz, rec, i0f=tile_fi0, j0f=tile_fj0)
        enddo
        if (pr_taking()) passive_loaded = .true.
      endif
    endif
    if (marbl_on .and. tindx == 1) call marbl_gpu_load_state(ncid, rec, i0, j0, nxi, neta, tile_fi0, tile_fj0)
  end subroutine load_3d_record

  subroutine load_coupling_fields(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, xi_u, eta_v
    call pr_store_2d(ncid, 'DU_avg1', DU_avg1(decomp%is, decomp%js), i0+1, j0, xi_u, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'DV_avg1', DV_avg1(decomp%is, decomp%js), i0, j0+1, nxi, eta_v, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'DU_avg2', DU_avg2(decomp%is, decomp%js), i0+1, j0, xi_u, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'DV_avg2', DV_avg2(decomp%is, decomp%js), i0, j0+1, nxi, eta_v, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'DU_avg_bak', DU_avg_bak(decomp%is, decomp%js), i0+1, j0, xi_u, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, 'DV_avg_bak', DV_avg_bak(decomp%is, decomp%js), i0, j0+1, nxi, eta_v, rec, i0f=tile_fi0, j0f=tile_fj0)
    ! CPU get_init: zero true land but keep river faces (umask = 0, riv_umask = 1)
    DU_avg1 = DU_avg1 * (umask + riv_umask)
    DV_avg1 = DV_avg1 * (vmask + riv_vmask)
    DU_avg2 = DU_avg2 * (umask + riv_umask)
    DV_avg2 = DV_avg2 * (vmask + riv_vmask)
    DU_avg_bak = DU_avg_bak * (umask + riv_umask)
    DV_avg_bak = DV_avg_bak * (vmask + riv_vmask)
  end subroutine load_coupling_fields

  subroutine load_pflx_slow(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    ! The CPU's get_init_slow: the running means of calc_pflx from the
    ! current record, masked; u_slow and v_slow absent: the velocity
    ! (init_pflx_slow, after the velocities are loaded and masked); p_slow
    ! absent: 0 (init_pflx).
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, xi_u, eta_v
    ncid_has_uslow = nc_has_var(ncid, 'u_slow')
    ncid_has_vslow = nc_has_var(ncid, 'v_slow')
    if (ncid_has_uslow) then
      call pr_store_3d(ncid, 'u_slow', u_slow(decomp%is, decomp%js, 1), i0+1, j0, xi_u, neta, decomp%nz, rec, &
                       i0f=tile_fi0, j0f=tile_fj0)
    endif
    if (ncid_has_vslow) then
      call pr_store_3d(ncid, 'v_slow', v_slow(decomp%is, decomp%js, 1), i0, j0+1, nxi, eta_v, decomp%nz, rec, &
                       i0f=tile_fi0, j0f=tile_fj0)
    endif
    if (nc_has_var(ncid, 'p_slow')) then
      call pr_store_3d(ncid, 'p_slow', p_slow(decomp%is, decomp%js, 1), i0, j0, nxi, neta, decomp%nz, rec, &
                       i0f=tile_fi0, j0f=tile_fj0)
    elseif (pr_taking() .and. comm_rank() == 0) then
      write(*,'(a)') ' --- WARNING: p_slow not in initial file.  Initialized to 0.0'
    endif
  end subroutine load_pflx_slow

  subroutine init_pflx_slow(have_u, have_v)
    ! After the restart read: mask the running means read from the file, or
    ! start them from the (masked) velocity of the current record as the
    ! CPU does (the full velocity, owned faces 1:i1 x j0:j1 / i0:i1 x 1:j1);
    ! exchange p_slow.
    logical, intent(in) :: have_u, have_v
    integer(ip) :: k, ia, ib, ja, jb
    call sync_arch()
    if (have_u) then
      do k = 1, decomp%nz
        u_slow(:, :, k) = u_slow(:, :, k) * umask
      enddo
    else
      ia = 1; ib = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
      ja = merge(0, 1, decomp%edge_s); jb = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
      u_slow(ia:ib, ja:jb, :) = u(ia:ib, ja:jb, :, 1)
      if (comm_rank() == 0) write(*,'(a)') ' --- WARNING: u_slow not in initial file.  Initialized to u'
    endif
    if (have_v) then
      do k = 1, decomp%nz
        v_slow(:, :, k) = v_slow(:, :, k) * vmask
      enddo
    else
      ia = merge(0, 1, decomp%edge_w); ib = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
      ja = 1; jb = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
      v_slow(ia:ib, ja:jb, :) = v(ia:ib, ja:jb, :, 1)
      if (comm_rank() == 0) write(*,'(a)') ' --- WARNING: v_slow not in initial file.  Initialized to v'
    endif
    do k = 1, decomp%nz
      p_slow(:, :, k) = p_slow(:, :, k) * rmask
    enddo
    call pflx_fill_slow_halos()
  end subroutine init_pflx_slow

  subroutine load_rufrc_slot(ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v
    character(len=16) :: uname, vname
    if (nc_has_var(ncid, 'rufrc_bak')) then
      uname = 'rufrc_bak'
    elseif (nc_has_var(ncid, 'rufrc')) then
      uname = 'rufrc'
    else
      return
    endif
    if (nc_has_var(ncid, 'rvfrc_bak')) then
      vname = 'rvfrc_bak'
    elseif (nc_has_var(ncid, 'rvfrc')) then
      vname = 'rvfrc'
    else
      return
    endif
    call pr_store_2d(ncid, trim(uname), rufrc_bak(decomp%is, decomp%js, tindx), i0+1, j0, xi_u, neta, rec, i0f=tile_fi0, j0f=tile_fj0)
    call pr_store_2d(ncid, trim(vname), rvfrc_bak(decomp%is, decomp%js, tindx), i0, j0+1, nxi, eta_v, rec, i0f=tile_fi0, j0f=tile_fj0)
    rufrc_bak(:, :, tindx) = rufrc_bak(:, :, tindx) * umask
    rvfrc_bak(:, :, tindx) = rvfrc_bak(:, :, tindx) * vmask
  end subroutine load_rufrc_slot

  subroutine infer_layout(xi_rho, eta_rho, i0, j0, nxi, neta)
    integer, intent(in)  :: xi_rho, eta_rho
    integer, intent(out) :: i0, j0, nxi, neta
    nxi = xi_rho
    neta = eta_rho
    if (xi_rho == case_llm + 2 .and. eta_rho == case_mmm + 2) then
      i0 = 0; j0 = 0
      case_periodic_x = .false.
      case_periodic_y = .false.
      nxi = int(decomp%nx) + 2; neta = int(decomp%ny) + 2   ! this tile's window of the global grid
    elseif (xi_rho == case_llm .and. eta_rho == case_mmm) then
      i0 = 1; j0 = 1
      case_periodic_x = .true.
      case_periodic_y = .true.
      nxi = int(decomp%nx); neta = int(decomp%ny)
    else
      write(*,*) 'grid dims', xi_rho, eta_rho, 'do not match LLm,MMm', case_llm, case_mmm
      error stop 'infer_layout'
    endif
    ! decomp already created as non-periodic default; re-init if periodic.
    if (case_periodic_x .or. case_periodic_y) then
      call free_state()
      call apply_decomp(case_periodic_x, case_periodic_y)
      call alloc_state(nt)
      nxi = int(decomp%nx); neta = int(decomp%ny)
    endif
  end subroutine infer_layout

  subroutine zero_closed_wall_velocities()
    ! CPU get_init (get_init_mod.F90:695-730): on a closed, non-periodic side the
    ! normal velocity on the boundary face is set to zero, for an initial file
    ! that was not made for a closed side: u and ubar on the west (i = 1) and
    ! east (i = nx+1) faces, v and vbar on the south (j = 1) and north (j = ny+1)
    ! faces. Here after init_bry/init_tides, which set obc_*, so every slot the
    ! load and copy_initial_slots filled is zeroed, and the cold-start barotropic
    ! fluxes are recomputed. A closed side whose boundary faces are land (CCS
    ! east) is already zero.
    ! The zeroing is per tile; the halo refill after it is collective, so every
    ! rank takes part whenever the domain has a closed side.
    integer(ip) :: j, k, m, nx, ny, nz, is, ie, js, je
    logical :: cw, ce, cs, cn, zw, ze, zs, zn
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    cw = .not. obc_west .and. .not. case_periodic_x
    ce = .not. obc_east .and. .not. case_periodic_x
    cs = .not. obc_south .and. .not. case_periodic_y
    cn = .not. obc_north .and. .not. case_periodic_y
    if (.not. (cw .or. ce .or. cs .or. cn)) return
    zw = decomp%edge_w .and. cw; ze = decomp%edge_e .and. ce
    zs = decomp%edge_s .and. cs; zn = decomp%edge_n .and. cn
    if (zw .or. ze) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 1, nz
        do j = js, je
          if (zw) then
            u(1, j, k, 1) = 0.0_rp; u(1, j, k, 2) = 0.0_rp
          endif
          if (ze) then
            u(nx + 1, j, k, 1) = 0.0_rp; u(nx + 1, j, k, 2) = 0.0_rp
          endif
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = js, je
        do m = 1, 4
          if (zw) ubar(1, j, m) = 0.0_rp
          if (ze) ubar(nx + 1, j, m) = 0.0_rp
        enddo
      enddo
    endif
    if (zs .or. zn) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do k = 1, nz
        do j = is, ie
          if (zs) then
            v(j, 1, k, 1) = 0.0_rp; v(j, 1, k, 2) = 0.0_rp
          endif
          if (zn) then
            v(j, ny + 1, k, 1) = 0.0_rp; v(j, ny + 1, k, 2) = 0.0_rp
          endif
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = is, ie
        do m = 1, 4
          if (zs) vbar(j, 1, m) = 0.0_rp
          if (zn) vbar(j, ny + 1, m) = 0.0_rp
        enddo
      enddo
    endif
    call fill_masked_velocity_halos()
    if (.not. case_exact_restart) call init_barotropic_fluxes()
    if (comm_rank() == 0) write(*,'(a,4l2)') 'closed sides W E S N: boundary-face velocity set to 0:', &
         cw, ce, cs, cn
  end subroutine zero_closed_wall_velocities

  subroutine init_barotropic_fluxes()
    ! CPU set_depth with iic == 0, which main.F90 calls after get_init on
    ! every start that is not an exact restart (get_init leaves iic = 1 only
    ! after a successful exact restart): the fast-time-averaged barotropic
    ! fluxes from the initial state,
    !   DU_avg1 = 0.5*(h(i-1)+h(i)+zeta(i)+zeta(i-1))*dn_u*ubar,
    ! which the first step's predictor coupling uses as its target. Without
    ! it a cold start couples to zero transport.
    integer(ip) :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        if (i > is) DU_avg1(i, j) = 0.5_rp * (h(i - 1, j) + h(i, j) + zeta(i, j, 1) + &
                                    zeta(i - 1, j, 1)) * dn_u(i, j) * (ubar(i, j, 1))
        if (j > js) DV_avg1(i, j) = 0.5_rp * (h(i, j) + h(i, j - 1) + zeta(i, j, 1) + &
                                    zeta(i, j - 1, 1)) * dm_v(i, j) * (vbar(i, j, 1))
      enddo
    enddo
  end subroutine init_barotropic_fluxes

  subroutine mask_loaded_velocities()
    ! CPU get_init: the loaded ubar/vbar (slot 1) and u/v (slots 1 and 2)
    ! times (umask + riv_umask) / (vmask + riv_vmask): zero on land, kept on
    ! river faces. Initial files may hold nonzero velocities on land
    ! (Iceland0: every land face), which the CPU never sees.
    integer(ip) :: i, j, k, n, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        ubar(i, j, 1) = ubar(i, j, 1) * (umask(i, j) + riv_umask(i, j))
        vbar(i, j, 1) = vbar(i, j, 1) * (vmask(i, j) + riv_vmask(i, j))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = js, je
        do i = is, ie
          do n = 1, 2
            u(i, j, k, n) = u(i, j, k, n) * (umask(i, j) + riv_umask(i, j))
            v(i, j, k, n) = v(i, j, k, n) * (vmask(i, j) + riv_vmask(i, j))
          enddo
        enddo
      enddo
    enddo
  end subroutine mask_loaded_velocities

  subroutine fill_masked_velocity_halos()
    ! Refill the tile halos of what mask_loaded_velocities changed. Before
    ! fill_metric_halos, umask/vmask were 1 on a tile's outermost halo column/row,
    ! so a land face there kept the file's velocity: on CCS the Nass river faces at
    ! the 2x4 tile cut kept the CPU's river velocity while their owner zeroed
    ! it, and the predictor's UP3 stencil read the stale copy. With the masks
    ! filled this refill changes nothing; it is kept as a guard.
    integer(ip) :: n
    call halo_fill_2d_s(ubar, STAGGER_U, 1_ip)
    call halo_fill_2d_s(vbar, STAGGER_V, 1_ip)
    do n = 1, 2
      call halo_fill_3d_s(u, STAGGER_U, n)
      call halo_fill_3d_s(v, STAGGER_V, n)
    enddo
  end subroutine fill_masked_velocity_halos

  subroutine sanitize_land()
    ! Elementwise on the device: the state may already be on the GPU (the
    ! restart is stored by device kernels), and host loops over these
    ! managed arrays migrated ~2 GB per rank to the host and back. The tests
    ! and assignments per element are those of the former host loops.
    integer(ip) :: i, j, k, n, m, is, ie, js, je, nz, ntr, c, nt3
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nz = decomp%nz; ntr = size(t, 5)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        if (rmask(i, j) /= rmask(i, j)) rmask(i, j) = 0.0_rp
        if (rmask(i, j) < 0.5_rp) then
          rmask(i, j) = 0.0_rp
          if (h(i, j) /= h(i, j) .or. h(i, j) <= 0.0_rp) h(i, j) = 1.0_rp
          if (pm(i, j) /= pm(i, j) .or. pm(i, j) <= 0.0_rp) pm(i, j) = 1.0_rp
          if (pn(i, j) /= pn(i, j) .or. pn(i, j) <= 0.0_rp) pn(i, j) = 1.0_rp
          ! CPU setup_grid writes f from lat_rho unmasked; Coriolis uses it on land.
          do m = 1, 4
            zeta(i, j, m) = 0.0_rp
          enddo
          hbls(i, j) = 0.0_rp
          hbbl(i, j) = 0.0_rp
          do n = 1, ntr
            do m = 1, 3
              do k = 1, nz
                t(i, j, k, m, n) = 0.0_rp
              enddo
            enddo
          enddo
        endif
        if (h(i, j) /= h(i, j) .or. h(i, j) <= 0.0_rp) h(i, j) = 1.0_rp
        if (pm(i, j) /= pm(i, j) .or. pm(i, j) <= 0.0_rp) pm(i, j) = 1.0e-4_rp
        if (pn(i, j) /= pn(i, j) .or. pn(i, j) <= 0.0_rp) pn(i, j) = 1.0e-4_rp
        if (f(i, j) /= f(i, j)) f(i, j) = 0.0_rp
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = js, je
        do i = is, ie
          do n = 1, 3
            if (u(i, j, k, n) /= u(i, j, k, n)) u(i, j, k, n) = 0.0_rp
            if (v(i, j, k, n) /= v(i, j, k, n)) v(i, j, k, n) = 0.0_rp
          enddo
          do n = 1, ntr
            do m = 1, 3
              if (t(i, j, k, m, n) /= t(i, j, k, m, n)) t(i, j, k, m, n) = 0.0_rp
            enddo
          enddo
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        do m = 1, 4
          if (ubar(i, j, m) /= ubar(i, j, m)) ubar(i, j, m) = 0.0_rp
          if (vbar(i, j, m) /= vbar(i, j, m)) vbar(i, j, m) = 0.0_rp
          if (zeta(i, j, m) /= zeta(i, j, m)) zeta(i, j, m) = 0.0_rp
        enddo
      enddo
    enddo
    ! The passive tracers at land as the CPU's (0): land cells inside the
    ! tile read the zero column, but the stored columns of the boundary ring
    ! held the initial file's land values until the first step's boundary
    ! conditions (seen in the initial history record and in averages over
    ! the first steps).
    if (ntp > 0 .and. allocated(tp)) then
      nt3 = int(ntp)
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = js, je
        do i = is, ie
          c = tpcol(i, j)
          if (c > 0 .and. rmask(i, j) < 0.5_rp) then
            do m = 1, 3
              do k = 1, nz
                do n = 1, nt3
                  tp(n, c, k, m) = 0.0_pp
                enddo
              enddo
            enddo
          endif
        enddo
      enddo
    endif
  end subroutine sanitize_land

  subroutine pad_physical_halo()
    ! Halo cells outside a physical edge: copy column 0 / nx+1 outward
    ! (west/east), then copy row 0 / ny+1 metrics and zero the state
    ! (south/north). Device kernels, one per edge in the former order (the
    ! south/north passes read the corner cells the west/east passes wrote).
    integer(ip) :: i, j, k, m, n, ms, is, ie, js, je, nx, ny, nz, ntr, isrc, jsrc, i0, i1, j0, j1
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = size(t, 5)
    do m = 1, 2
      if (m == 1) then
        if (.not. decomp%edge_w) cycle
        i0 = is; i1 = -1; isrc = 0
      else
        if (.not. decomp%edge_e) cycle
        i0 = nx + 2; i1 = ie; isrc = nx + 1
      endif
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = js, je
        do i = i0, i1
          h(i, j) = h(isrc, j); pm(i, j) = pm(isrc, j); pn(i, j) = pn(isrc, j)
          rmask(i, j) = rmask(isrc, j); f(i, j) = f(isrc, j)
          lonr(i, j) = lonr(isrc, j); latr(i, j) = latr(isrc, j); angler(i, j) = angler(isrc, j)
          do k = 1, 4
            zeta(i, j, k) = zeta(isrc, j, k)
            ubar(i, j, k) = ubar(isrc, j, k)
            vbar(i, j, k) = vbar(isrc, j, k)
          enddo
          do n = 1, 3
            do k = 1, nz
              u(i, j, k, n) = u(isrc, j, k, n)
              v(i, j, k, n) = v(isrc, j, k, n)
            enddo
          enddo
          do n = 1, ntr
            do ms = 1, 3
              do k = 1, nz
                t(i, j, k, ms, n) = t(isrc, j, k, ms, n)
              enddo
            enddo
          enddo
          hbls(i, j) = hbls(isrc, j); hbbl(i, j) = hbbl(isrc, j)
        enddo
      enddo
    enddo
    do m = 1, 2
      if (m == 1) then
        if (.not. decomp%edge_s) cycle
        j0 = js; j1 = -1; jsrc = 0
      else
        if (.not. decomp%edge_n) cycle
        j0 = ny + 2; j1 = je; jsrc = ny + 1
      endif
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = j0, j1
        do i = is, ie
          h(i, j) = h(i, jsrc); pm(i, j) = pm(i, jsrc); pn(i, j) = pn(i, jsrc)
          f(i, j) = f(i, jsrc)
          lonr(i, j) = lonr(i, jsrc); latr(i, j) = latr(i, jsrc); angler(i, j) = angler(i, jsrc)
          rmask(i, j) = 0.0_rp
          umask(i, j) = 0.0_rp
          vmask(i, j) = 0.0_rp
          do k = 1, 4
            zeta(i, j, k) = 0.0_rp
            ubar(i, j, k) = 0.0_rp
            vbar(i, j, k) = 0.0_rp
          enddo
          do n = 1, 3
            do k = 1, nz
              u(i, j, k, n) = 0.0_rp
              v(i, j, k, n) = 0.0_rp
            enddo
          enddo
          do n = 1, ntr
            do ms = 1, 3
              do k = 1, nz
                t(i, j, k, ms, n) = 0.0_rp
              enddo
            enddo
          enddo
          hbls(i, j) = 0.0_rp
          hbbl(i, j) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine pad_physical_halo

  subroutine copy_initial_slots(copy_prev)
    ! zeta/ubar/vbar slots 2-4 = slot 1; with copy_prev also u, v and the
    ! two active tracers slot 2 = slot 1 (device kernels, as sanitize_land).
    logical, intent(in) :: copy_prev
    integer(ip) :: i, j, k, m, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        do m = 2, 4
          zeta(i, j, m) = zeta(i, j, 1)
          ubar(i, j, m) = ubar(i, j, 1)
          vbar(i, j, m) = vbar(i, j, 1)
        enddo
      enddo
    enddo
    if (.not. copy_prev) return
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = js, je
        do i = is, ie
          u(i, j, k, 2) = u(i, j, k, 1)
          v(i, j, k, 2) = v(i, j, k, 1)
          t(i, j, k, 2, 1) = t(i, j, k, 1, 1)
          t(i, j, k, 2, 2) = t(i, j, k, 1, 2)
        enddo
      enddo
    enddo
  end subroutine copy_initial_slots

  subroutine fill_static_halos()
    ! Tile halos of the grid fields: the window read covers local 0..nx+1
    ! and pad_physical_halo the physical edges; the two halo points toward a
    ! neighbor tile come from that neighbor. No-op on a single tile.
    call halo_fill_2d(h, STAGGER_RHO)
    call halo_fill_2d(pm, STAGGER_RHO)
    call halo_fill_2d(pn, STAGGER_RHO)
    call halo_fill_2d(rmask, STAGGER_RHO)
    call halo_fill_2d(f, STAGGER_RHO)
    call halo_fill_2d(lonr, STAGGER_RHO)
    call halo_fill_2d(latr, STAGGER_RHO)
    call halo_fill_2d(angler, STAGGER_RHO)
  end subroutine fill_static_halos

  subroutine fill_metric_halos()
    ! setup_metrics sets the u-point fields from is+1 and the v-point fields from
    ! js+1, so on a tile's outermost halo column (u) or row (v) they kept their
    ! defaults (masks 1). Take them from the neighbor tile, as for the grid fields,
    ! before anything reads them. The p-point fields (pmask, dm_p, dn_p) have the
    ! same gap, but no kernel reads them on the outermost halo. No-op on a single tile.
    call halo_fill_2d(umask, STAGGER_U)
    call halo_fill_2d(dm_u, STAGGER_U)
    call halo_fill_2d(dn_u, STAGGER_U)
    call halo_fill_2d(pmon_u, STAGGER_U)
    call halo_fill_2d(vmask, STAGGER_V)
    call halo_fill_2d(dm_v, STAGGER_V)
    call halo_fill_2d(dn_v, STAGGER_V)
    call halo_fill_2d(pnom_v, STAGGER_V)
  end subroutine fill_metric_halos

  subroutine fill_loaded_halos()
    ! Tile halos of every state field read from the initial/restart file
    ! (slots 1 and 2; slot 2 only matters for an exact restart). No-op on a
    ! single tile.
    integer(ip) :: n, itrc
    do n = 1, 2
      call halo_fill_2d_s(zeta, STAGGER_RHO, n)
      call halo_fill_2d_s(ubar, STAGGER_U, n)
      call halo_fill_2d_s(vbar, STAGGER_V, n)
      call halo_fill_3d_s(u, STAGGER_U, n)
      call halo_fill_3d_s(v, STAGGER_V, n)
      do itrc = 1, nt
        call halo_fill_3d_t(t, STAGGER_RHO, n, itrc)
      enddo
      if (ntp > 0) call halo_fill_3d_tp(tp, ntp, n)
    enddo
    call halo_fill_2d(hbls, STAGGER_RHO)
    call halo_fill_2d(hbbl, STAGGER_RHO)
    call halo_fill_2d(DU_avg1, STAGGER_U)
    call halo_fill_2d(DV_avg1, STAGGER_V)
    call halo_fill_2d(DU_avg2, STAGGER_U)
    call halo_fill_2d(DV_avg2, STAGGER_V)
    call halo_fill_2d(DU_avg_bak, STAGGER_U)
    call halo_fill_2d(DV_avg_bak, STAGGER_V)
    call halo_fill_2d(rufrc, STAGGER_U)
    call halo_fill_2d(rvfrc, STAGGER_V)
  end subroutine fill_loaded_halos

  subroutine report_loaded_case()
    integer(ip) :: nwet, i, j, k
    real(rp) :: hmin, hmax, tmin, tmax, zmax, umax
    nwet = count(rmask(1:decomp%nx, 1:decomp%ny) > 0.5_rp)
    hmin = huge(1.0_rp); hmax = -huge(1.0_rp)
    tmin = huge(1.0_rp); tmax = -huge(1.0_rp)
    zmax = 0.0_rp; umax = 0.0_rp
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) > 0.5_rp) then
          hmin = min(hmin, h(i, j))
          hmax = max(hmax, h(i, j))
          zmax = max(zmax, abs(zeta(i, j, 1)))
          do k = 1, decomp%nz
            tmin = min(tmin, t(i, j, k, 1, 1))
            tmax = max(tmax, t(i, j, k, 1, 1))
          enddo
        endif
      enddo
    enddo
    umax = 0.0_rp
    ic_u_i = 0; ic_u_j = 0; ic_u_k = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx + 1
        if (umask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (abs(u(i, j, k, 1)) > umax) then
            umax = abs(u(i, j, k, 1))
            ic_u_i = i; ic_u_j = j; ic_u_k = k
          endif
        enddo
      enddo
    enddo
    ic_u_val = umax
    write(*,'(a)') 'loaded case'
    write(*,'(a,a)') '  title  = ', trim(case_title)
    write(*,'(a,3i6)') '  nx,ny,nz = ', decomp%nx, decomp%ny, decomp%nz
    write(*,'(a,i8)') '  wet rho = ', nwet
    write(*,'(a,2es14.6)') '  h min,max = ', hmin, hmax
    write(*,'(a,2es14.6)') '  T min,max = ', tmin, tmax
    write(*,'(a,es14.6)') '  max |zeta| = ', zmax
    write(*,'(a,es14.6,a,3i6)') '  max |u|    = ', umax, ' at i,j,k =', ic_u_i, ic_u_j, ic_u_k
    write(*,'(a,es14.6)') '  ocean_time = ', start_time
    write(*,'(a,2i8,a,l1)') '  ntstart, forw_start = ', ntstart, forw_start, &
         '  exact = ', case_exact_restart
    write(*,'(a,3es12.4)') '  dt, theta_s, hc = ', dt, theta_s, hc
    iic = 0
    knew = 1
    call set_depth()
    write(*,'(a,es14.6)') '  max |Hz|   = ', maxval(abs(Hz(1:decomp%nx, 1:decomp%ny, :)), &
         mask=spread(rmask(1:decomp%nx, 1:decomp%ny) > 0.5_rp, 3, decomp%nz))
  end subroutine report_loaded_case

  subroutine write_toy_case(dir, llm, mmm, nz)
    character(len=*), intent(in) :: dir
    integer(ip),      intent(in) :: llm, mmm, nz
    integer :: ncid, d_xi, d_eta, d_xiu, d_etav, d_s, d_t, nxi, neta
    integer :: i, j, k
    integer :: id_h, id_pm, id_pn, id_mask, id_f, id_z, id_ub, id_vb, id_u, id_v, id_t, id_s, id_ot
    real(rp), allocatable :: a2(:,:), au(:,:), av(:,:), a3(:,:,:), au3(:,:,:), av3(:,:,:)
    integer :: unit

    nxi = llm + 2
    neta = mmm + 2
    allocate(a2(nxi, neta), au(nxi - 1, neta), av(nxi, neta - 1))
    allocate(a3(nxi, neta, nz), au3(nxi - 1, neta, nz), av3(nxi, neta - 1, nz))

    do j = 1, neta
      do i = 1, nxi
        a2(i, j) = 200.0_rp + real(i + j, rp)
      enddo
    enddo
    call nc_create(trim(dir)//'/toy_grid.nc', ncid)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id_h = nc_def_var(ncid, 'h', (/d_xi, d_eta/))
    id_pm = nc_def_var(ncid, 'pm', (/d_xi, d_eta/))
    id_pn = nc_def_var(ncid, 'pn', (/d_xi, d_eta/))
    id_mask = nc_def_var(ncid, 'mask_rho', (/d_xi, d_eta/))
    id_f = nc_def_var(ncid, 'f', (/d_xi, d_eta/))
    call nc_enddef(ncid)
    call nc_put_2d(ncid, 'h', a2)
    a2 = 1.0_rp / 12000.0_rp
    call nc_put_2d(ncid, 'pm', a2)
    call nc_put_2d(ncid, 'pn', a2)
    a2 = 1.0_rp
    a2(3, 3) = 0.0_rp
    call nc_put_2d(ncid, 'mask_rho', a2)
    a2 = 1.0e-4_rp
    call nc_put_2d(ncid, 'f', a2)
    call nc_close(ncid)

    a2 = 0.01_rp
    a2(3, 3) = 0.0_rp
    au = 0.02_rp
    av = -0.03_rp
    do k = 1, nz
      a3(:, :, k) = 15.0_rp + 0.1_rp * real(k, rp)
      au3(:, :, k) = 0.05_rp
      av3(:, :, k) = -0.04_rp
    enddo
    call nc_create(trim(dir)//'/toy_init.nc', ncid)
    d_t = nc_def_dim(ncid, 'ocean_time', 1)
    d_s = nc_def_dim(ncid, 's_rho', nz)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id_ot = nc_def_var(ncid, 'ocean_time', (/d_t/))
    id_z = nc_def_var(ncid, 'zeta', (/d_xi, d_eta/))
    id_ub = nc_def_var(ncid, 'ubar', (/d_xiu, d_eta/))
    id_vb = nc_def_var(ncid, 'vbar', (/d_xi, d_etav/))
    id_u = nc_def_var(ncid, 'u', (/d_xiu, d_eta, d_s/))
    id_v = nc_def_var(ncid, 'v', (/d_xi, d_etav, d_s/))
    id_t = nc_def_var(ncid, 'temp', (/d_xi, d_eta, d_s/))
    id_s = nc_def_var(ncid, 'salt', (/d_xi, d_eta, d_s/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'ocean_time', (/ 86400.0_rp /))
    call nc_put_2d(ncid, 'zeta', a2)
    call nc_put_2d(ncid, 'ubar', au)
    call nc_put_2d(ncid, 'vbar', av)
    call nc_put_3d(ncid, 'u', au3)
    call nc_put_3d(ncid, 'v', av3)
    call nc_put_3d(ncid, 'temp', a3)
    a3 = 35.0_rp
    call nc_put_3d(ncid, 'salt', a3)
    call nc_close(ncid)

    open(newunit=unit, file=trim(dir)//'/toy.nml', status='replace')
    write(unit,'(a)') '&param_settings'
    write(unit,'(a,i0)') '    llm = ', llm
    write(unit,'(a,i0)') '    mmm = ', mmm
    write(unit,'(a)')    '    np_eta = 1'
    write(unit,'(a)')    '    np_xi = 1'
    write(unit,'(a)')    '    nt_bgc = 0'
    write(unit,'(a)')    '    nt_passive = 0'
    write(unit,'(a,i0)') '    nz = ', nz
    write(unit,'(a)')    '/'
    write(unit,'(a)') '&grid_settings'
    write(unit,'(a)') "    grdname = 'toy_grid.nc'"
    write(unit,'(a)') '/'
    write(unit,'(a)') '&initial_conditions'
    write(unit,'(a)') "    inifile = 'toy_init.nc'"
    write(unit,'(a)') '/'
    write(unit,'(a)') '&s_coord'
    write(unit,'(a)') '    hc = 250.0'
    write(unit,'(a)') '    theta_b = 6.0'
    write(unit,'(a)') '    theta_s = 6.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&time_stepping'
    write(unit,'(a)') '    dt = 900.0'
    write(unit,'(a)') '    ndtfast = 20'
    write(unit,'(a)') '    ninfo = 1'
    write(unit,'(a)') '    ntimes = 3'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&bottom_drag_settings'
    write(unit,'(a)') '    rdrg = 0.0'
    write(unit,'(a)') '    rdrg2 = 0.001'
    write(unit,'(a)') '    zob = 0.01'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&vertical_mixing_settings'
    write(unit,'(a)') '    akt_bak = 0.0, 0.0'
    write(unit,'(a)') '    akv_bak = 0.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&lateral_visc_settings'
    write(unit,'(a)') '    visc2 = 0.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&tracer_diff2'
    write(unit,'(a)') '    tnu2 = 0.0, 0.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&rho0_settings'
    write(unit,'(a)') '    rho0 = 1000.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&simulation_name_settings'
    write(unit,'(a)') "    output_root_name = './output'"
    write(unit,'(a)') "    title = 'toy_case'"
    write(unit,'(a)') '/'
    close(unit)
    deallocate(a2, au, av, a3, au3, av3)
  end subroutine write_toy_case

  function dirname(path) result(dir)
    character(len=*), intent(in) :: path
    character(len=512) :: dir
    integer :: n
    n = index(path, '/', back=.true.)
    if (n > 0) then
      dir = path(1:n-1)
    else
      dir = '.'
    endif
  end function dirname

  function resolve_path(dir, path) result(out)
    character(len=*), intent(in) :: dir, path
    character(len=512) :: out
    if (len_trim(path) == 0) then
      out = ''
    elseif (path(1:1) == '/') then
      out = trim(path)
    else
      out = trim(dir)//'/'//trim(path)
    endif
  end function resolve_path

  subroutine nml_stop(ios, msg, group)
    ! After an optional group's read: absent (end of file, ios < 0) is fine;
    ! present but unreadable (ios > 0, e.g. a key the GPU does not know)
    ! stops, since the rest of the group would silently keep its defaults.
    integer, intent(in) :: ios
    character(len=*), intent(in) :: msg, group
    if (ios <= 0) return
    write(*,*) group, ': ', trim(iomsg_text(msg))
    error stop 'read_namelist: unreadable namelist group (see above)'
  end subroutine nml_stop

end module case_io

module marbl_host
  ! Host-side MARBL instance for the GPU model.
  !
  ! MARBL's own settings/initialisation code runs unchanged here (put_setting
  ! from the config file, init with the GCM column count). From the
  ! initialised instance we take tracer indices/metadata, forcing indices,
  ! saved-state indices and the handful of scalar parameters the device
  ! kernels need, packed in plain derived types that CUF kernels receive by
  ! value. The instance is also kept as a per-column *reference*: the
  ! unchanged MARBL surface_flux_compute can be run on any column so that the
  ! device kernels are validated column by column against CPU MARBL.
  use kinds, only: rp, ip
  use marbl_interface, only: marbl_interface_class
  use marbl_kinds_mod, only: r8
  implicit none
  private

  type(marbl_interface_class), public, save :: mi
  integer(ip), public, save :: nt_marbl = 0
  character(len=64), allocatable, public, save :: marbl_tracer_names(:)
  character(len=64), allocatable, public, save :: marbl_tracer_units(:)
  character(len=128), allocatable, public, save :: marbl_tracer_lnames(:)   ! long names (averages file)
  logical, public, save :: marbl_host_ready = .false.

  ! Indices into mi%surface_flux_forcings (0 when MARBL did not request it).
  integer(ip), public, save :: sfi_u10sqr = 0, sfi_sss = 0, sfi_sst = 0, sfi_ifrac = 0, &
                               sfi_dust = 0, sfi_iron = 0, sfi_nox = 0, sfi_nhy = 0,   &
                               sfi_atmp = 0, sfi_xco2 = 0, sfi_xco2_alt = 0

  ! Surface-flux parameters for the device kernel. Scalars only: CUF kernels
  ! take the whole type by value at launch.
  integer, parameter, public :: MSF_MAX_AUTO = 8
  type, public :: marbl_sf_par_t
    integer(ip) :: ntr = 0
    integer(ip) :: po4 = 0, no3 = 0, sio3 = 0, nh4 = 0, fe = 0, o2 = 0
    integer(ip) :: dic = 0, dic_alt = 0, alk = 0, alk_alt = 0
    integer(ip) :: nauto = 0
    integer(ip) :: chl(MSF_MAX_AUTO) = 0
    integer(ip) :: flux_o2 = 1, flux_co2 = 1, nhx_emis = 1   ! 0/1 (no logicals in device structs)
    real(rp)    :: xkw_coeff = 0.0_rp
    real(rp)    :: mass_to_vol = 0.0_rp       ! mol/kg -> model conc units
    real(rp)    :: dust_po4_fac = 0.0_rp      ! dust (kg/m2/s) -> PO4 flux
    real(rp)    :: dust_sio3_fac = 0.0_rp     ! dust (kg/m2/s) -> SiO3 flux
    real(rp)    :: len2m = 1.0_rp, cm2len = 1.0_rp, m2len = 1.0_rp
    real(rp)    :: del_ph = 0.2_rp, phlo_init = 7.0_rp, phhi_init = 9.0_rp
    real(rp)    :: ifrac = 0.0_rp, atm_pres = 1.0_rp   ! the ROMS driver's constants
  end type marbl_sf_par_t
  type(marbl_sf_par_t), public, save :: msf

  ! Surface diagnostics the kernel and the host reference both produce.
  integer, parameter, public :: MSF_NDIAG = 15
  integer, parameter, public :: D_PH = 1, D_PCO2 = 2, D_DPCO2 = 3, D_FGCO2 = 4, &
       D_PH_ALT = 5, D_PCO2_ALT = 6, D_FGCO2_ALT = 7, D_PVO2 = 8, D_SCHO2 = 9, &
       D_O2SAT = 10, D_CO2STAR = 11, D_DCO2STAR = 12, D_PVCO2 = 13, D_SCHCO2 = 14, &
       D_NHX = 15
  character(len=16), parameter, public :: msf_diag_name(MSF_NDIAG) = [character(len=16) :: &
       'PH', 'pCO2SURF', 'DpCO2', 'FG_CO2', 'PH_ALT_CO2', 'pCO2SURF_ALT_CO2', &
       'FG_ALT_CO2', 'PV_O2', 'SCHMIDT_O2', 'O2SAT', 'CO2STAR', 'DCO2STAR', &
       'PV_CO2', 'SCHMIDT_CO2', 'NHx_SURF_EMIS']


  ! Parameters for the device interior kernels: a flat copy of the MARBL
  ! settings / PFT tables, scalars and fixed-size arrays only (lives in a
  ! module-level managed variable on the device side).
  integer, parameter, public :: MA = 4    ! max autotrophs
  integer, parameter, public :: MZ = 2    ! max zooplankton
  integer, parameter, public :: MP = 4    ! max prey classes
  integer, parameter, public :: NTRMAX = 32   ! base MARBL tracer set; marbl_host_fill_par errors above it
  integer, parameter, public :: ND_IT = 16 ! interior debug diagnostics
  type, public :: marbl_par_t
    integer(ip) :: ntr = 0, nauto = 0, nzoo = 0, nprey = 0
    integer(ip) :: po4 = 0, no3 = 0, sio3 = 0, nh4 = 0, fe = 0, lig = 0, o2 = 0
    integer(ip) :: dic = 0, dica = 0, alk = 0, alka = 0
    integer(ip) :: doc = 0, don = 0, dop = 0, dopr = 0, donr = 0, docr = 0
    integer(ip) :: a_chl(MA) = 0, a_c(MA) = 0, a_p(MA) = 0, a_fe(MA) = 0, a_si(MA) = 0, a_caco3(MA) = 0
    integer(ip) :: a_tfunc(MA) = 0, a_nfixer(MA) = 0, a_impcalc(MA) = 0, a_expcalc(MA) = 0
    integer(ip) :: a_silic(MA) = 0, a_climit(MA) = 0
    integer(ip) :: z_c(MZ) = 0, z_tfunc(MZ) = 0
    integer(ip) :: g_nauto(MP, MZ) = 0, g_nzoo(MP, MZ) = 0, g_func(MP, MZ) = 0
    integer(ip) :: g_auto(MA, MP, MZ) = 0, g_zoo(MZ, MP, MZ) = 0
    integer(ip) :: lvar_ptoc = 1, caco3_bury_iopt = 2    ! 1 fixed depth, 2 omega_calc
    ! autotroph parameters
    real(rp) :: a_kFe(MA) = 0, a_kCO2(MA) = 0, a_kPO4(MA) = 0, a_kDOP(MA) = 0, a_kNO3(MA) = 0
    real(rp) :: a_kNH4(MA) = 0, a_kSiO3(MA) = 0, a_Qp_fixed(MA) = 0, a_gQfe_0(MA) = 0, a_gQfe_min(MA) = 0
    real(rp) :: a_alphaPI(MA) = 0, a_PCref(MA) = 0, a_thetaN_max(MA) = 0, a_loss_thres(MA) = 0
    real(rp) :: a_loss_thres2(MA) = 0, a_temp_thres(MA) = 0, a_mort(MA) = 0, a_mort2(MA) = 0
    real(rp) :: a_agg_rate_max(MA) = 0, a_agg_rate_min(MA) = 0, a_loss_poc(MA) = 0, a_Tref(MA) = 0, a_Ea(MA) = 0
    ! zooplankton parameters
    real(rp) :: z_mort_0(MZ) = 0, z_mort2_0(MZ) = 0, z_basal(MZ) = 0, z_loss_thres(MZ) = 0, z_Tref(MZ) = 0, z_Ea(MZ) = 0
    ! grazing (prey, pred)
    real(rp) :: g_umax(MP, MZ) = 0, g_grz(MP, MZ) = 0, g_zoofrac(MP, MZ) = 0, g_poc(MP, MZ) = 0
    real(rp) :: g_doc(MP, MZ) = 0, g_fdetr(MP, MZ) = 0
    ! scalar settings
    real(rp) :: Q = 0, Qp_zoo = 0, Qfe_zoo = 0, Red_D_C_O2 = 0, Remin_D_C_O2 = 0, Red_D_C_O2_diaz = 0
    real(rp) :: Red_Fe_C = 0, dust_to_Fe = 0, denitrif_C_N = 0, f_toDON = 0, f_toDOP = 0
    real(rp) :: POCremin_refract = 0, PONremin_refract = 0, POPremin_refract = 0
    real(rp) :: DOCprod_refract = 0, DONprod_refract = 0, DOPprod_refract = 0
    real(rp) :: f_graze_CaCO3_remin = 0, f_graze_si_remin = 0, f_graze_sp_poc_lim = 0
    real(rp) :: caco3_poc_min = 0, spc_poc_fac = 0, labile_ratio = 0, o2_min = 0, o2_min_delta = 0
    real(rp) :: kappa_nitrif = 0, nitrif_par_lim = 0, scalelen_z(4) = 0, scalelen_vals(4) = 0
    real(rp) :: POC_diss = 0, CaCO3_diss = 0, CaCO3_gamma = 0, hPOC_CaCO3_ratio = 0, hPOC_SiO2_ratio = 0
    real(rp) :: hPOC_dust_ratio = 0, SiO2_diss = 0, SiO2_gamma = 0, Fe_desorption_rate0 = 0
    real(rp) :: sed_denitrif_coeff = 0, flux_ref_depth = 0, caco3_bury_depth = 0, caco3_bury_omega = 0
    real(rp) :: POM_bury_frac_max = 0, bSi_bury_frac_max = 0, o2_sf_hi = 0, o2_sf_lo = 0, o2_sf_val = 0
    real(rp) :: Fe_scav_yps = 0, Lig_scav_yps = 0, FeLig_scav_yps = 0, dust_Fe_scav_scale = 0
    real(rp) :: remin_to_Lig = 0, Lig_degrade_rate0 = 0
    real(rp) :: DOC_reminR_light = 0, DON_reminR_light = 0, DOP_reminR_light = 0
    real(rp) :: DOC_reminR_dark = 0, DON_reminR_dark = 0, DOP_reminR_dark = 0
    real(rp) :: DOCr_reminR0 = 0, DONr_reminR0 = 0, DOPr_reminR0 = 0, DOMr_reminR_photo = 0
    real(rp) :: f_qsw_par = 0, Q_10 = 0, thres_z1_auto = 0, thres_z2_auto = 0, thres_z1_zoo = 0, thres_z2_zoo = 0
    real(rp) :: auto_mort2_exp = 0, zoo_mort2_exp = 0, gQsi_0 = 0, gQsi_max = 0, gQsi_min = 0
    real(rp) :: gQ_Fe_kFe_thres = 0, gQ_Si_kSi_thres = 0, PquotaSlope = 0, PquotaIntercept = 0, PquotaMinNP = 0
    real(rp) :: QCaCO3_max = 0, f_prod_sp_CaCO3 = 0, CaCO3_sp_thres = 0, CaCO3_temp_thres1 = 0
    real(rp) :: CaCO3_temp_thres2 = 0, f_photosp_CaCO3 = 0, r_Nfix_photo = 0
    real(rp) :: POC_bury_coeff = 0, POP_bury_coeff = 0, bSi_bury_coeff = 0, PON_bury_coeff = 0
    real(rp) :: del_ph = 0, phlo_3d_init = 0, phhi_3d_init = 0, mass_to_vol = 0
    real(rp) :: len2m = 1, len2cm = 100, m2len = 1, mass2g = 1000, nmol2mol_prefix = 1.0e-6_rp
    real(rp) :: conc_flux2mmol_m2s = 1
  end type marbl_par_t

  public :: marbl_host_init
  public :: marbl_diag_info
  public :: marbl_host_fill_par
  public :: marbl_host_interior_column
  public :: marbl_host_print_log
  public :: marbl_host_sflux_column

contains

  subroutine marbl_host_init(nz, config_file, verbose)
    use marbl_settings_mod, only: xkw_coeff, del_ph, phlo_surf_init, phhi_surf_init, &
                                  lflux_gas_o2, lflux_gas_co2, lcompute_nhx_surface_emis, &
                                  autotroph_cnt, ladjust_bury_coeff
    use marbl_constants_mod, only: rho_sw, molw_P, molw_Si
    integer(ip), intent(in) :: nz
    character(len=*), intent(in) :: config_file
    logical, intent(in), optional :: verbose
    real(r8), allocatable :: dummy(:)
    character(len=256) :: line
    integer :: unit, ios, idx, m
    logical :: verb, exists
    ! mks unit system (the ROMS driver initialises MARBL with 'mks')
    real(rp), parameter :: mass2g = 1.0e3_rp, mass2kg = 1.0_rp, m2len = 1.0_rp, &
                           mol_prefix = 1.0e3_rp, len2m = 1.0_rp, cm2len = 0.01_rp

    verb = .true.
    if (present(verbose)) verb = verbose

    ! Settings: every line of the config file goes through put_setting,
    ! exactly as the CPU driver does. An empty name = MARBL defaults; a file
    ! that cannot be opened stops the run, as on the CPU.
    if (len_trim(config_file) == 0) then
      if (verb) write(*,'(a)') 'marbl_host: marbl_config_file = "": using MARBL defaults'
    else
      inquire(file=config_file, exist=exists)
      ios = 1
      if (exists) open(newunit=unit, file=config_file, status='old', action='read', iostat=ios)
      if (ios /= 0) then
        write(*,'(3a)') 'marbl_host: cannot open MARBL settings file ', trim(config_file), &
             ' (marbl_config_file); set marbl_config_file = "" to run MARBL with its default settings'
        error stop 'marbl_host_init: marbl_config_file'
      endif
      do
        read(unit, '(A)', iostat=ios) line
        if (ios /= 0) exit
        call mi%put_setting(line)
        if (verb .and. line(1:1) /= '!' .and. len_trim(line) > 0) &
             write(*,'(a,": ",a)') trim(config_file), trim(line)
      enddo
      close(unit)
    endif

    allocate(dummy(nz)); dummy = 1.0_r8
    call mi%init(gcm_num_levels = int(nz), gcm_num_PAR_subcols = 1, &
                 gcm_num_elements_surface_flux = 1, gcm_delta_z = dummy, &
                 gcm_zw = dummy, gcm_zt = dummy, lgcm_has_global_ops = .true., &
                 unit_system_opt = 'mks')
    if (mi%StatusLog%labort_marbl) then
      call marbl_host_print_log()
      error stop 'marbl_host_init: MARBL init failed'
    endif
    if (verb) call marbl_host_print_log()
    call mi%StatusLog%erase()

    nt_marbl = size(mi%tracer_metadata)
    allocate(marbl_tracer_names(nt_marbl), marbl_tracer_units(nt_marbl), marbl_tracer_lnames(nt_marbl))
    do m = 1, nt_marbl
      marbl_tracer_names(m) = trim(mi%tracer_metadata(m)%short_name)
      marbl_tracer_units(m) = trim(mi%tracer_metadata(m)%units)
      marbl_tracer_lnames(m) = trim(mi%tracer_metadata(m)%long_name)
    enddo

    do idx = 1, size(mi%surface_flux_forcings)
      select case (trim(mi%surface_flux_forcings(idx)%metadata%varname))
      case ('u10_sqr');              sfi_u10sqr = idx
      case ('sss');                  sfi_sss = idx
      case ('sst');                  sfi_sst = idx
      case ('Ice Fraction');         sfi_ifrac = idx
      case ('Dust Flux');            sfi_dust = idx
      case ('Iron Flux');            sfi_iron = idx
      case ('NOx Flux');             sfi_nox = idx
      case ('NHy Flux');             sfi_nhy = idx
      case ('Atmospheric Pressure'); sfi_atmp = idx
      case ('xco2');                 sfi_xco2 = idx
      case ('xco2_alt_co2');         sfi_xco2_alt = idx
      case default
        if (verb) write(*,'(2a)') 'marbl_host: surface forcing not indexed: ', &
             trim(mi%surface_flux_forcings(idx)%metadata%varname)
      end select
    enddo

    ! Kernel parameters
    msf%ntr     = nt_marbl
    msf%po4     = mi%tracer_indices%po4_ind
    msf%no3     = mi%tracer_indices%no3_ind
    msf%sio3    = mi%tracer_indices%sio3_ind
    msf%nh4     = mi%tracer_indices%nh4_ind
    msf%fe      = mi%tracer_indices%fe_ind
    msf%o2      = mi%tracer_indices%o2_ind
    msf%dic     = mi%tracer_indices%dic_ind
    msf%dic_alt = mi%tracer_indices%dic_alt_co2_ind
    msf%alk     = mi%tracer_indices%alk_ind
    msf%alk_alt = mi%tracer_indices%alk_alt_co2_ind
    msf%nauto   = autotroph_cnt
    if (autotroph_cnt > MSF_MAX_AUTO) error stop 'marbl_host_init: raise MSF_MAX_AUTO'
    do m = 1, autotroph_cnt
      msf%chl(m) = mi%tracer_indices%auto_inds(m)%Chl_ind
    enddo
    msf%flux_o2  = merge(1_ip, 0_ip, lflux_gas_o2)
    msf%flux_co2 = merge(1_ip, 0_ip, lflux_gas_co2)
    msf%nhx_emis = merge(1_ip, 0_ip, lcompute_nhx_surface_emis)
    msf%xkw_coeff = xkw_coeff
    msf%mass_to_vol = rho_sw * (1.0e3_rp * mass2kg * m2len**3)
    msf%dust_po4_fac  = (0.00105_rp * 0.15_rp / molw_P)  * mass2g * mol_prefix
    msf%dust_sio3_fac = (0.308_rp * 0.075_rp / molw_Si) * mass2g * mol_prefix
    msf%len2m = len2m; msf%cm2len = cm2len; msf%m2len = m2len
    msf%del_ph = del_ph; msf%phlo_init = phlo_surf_init; msf%phhi_init = phhi_surf_init

    if (ladjust_bury_coeff) error stop 'marbl_host_init: ladjust_bury_coeff not supported on the GPU path'
    if (sfi_xco2 == 0 .or. sfi_sst == 0 .or. sfi_sss == 0 .or. sfi_u10sqr == 0) &
         error stop 'marbl_host_init: unexpected MARBL surface forcing list'
    marbl_host_ready = .true.

    if (verb) then
      write(*,'(a,i0)') 'marbl_host: MARBL tracers = ', nt_marbl
      write(*,'(a,*(1x,a))') 'marbl_host: names =', (trim(marbl_tracer_names(m)), m = 1, nt_marbl)
      write(*,'(a,3i2,a,i0)') 'marbl_host: flux_o2/flux_co2/nhx =', msf%flux_o2, msf%flux_co2, &
           msf%nhx_emis, '  autotrophs = ', msf%nauto
      write(*,'(a,2i4)') 'marbl_host: saved-state counts (surf, interior) =', &
           mi%surface_flux_saved_state%saved_state_cnt, mi%interior_tendency_saved_state%saved_state_cnt
    endif
  end subroutine marbl_host_init

  subroutine marbl_diag_info(name, found, lname, units)
    ! Long name and units of a MARBL diagnostic; found = .false. if MARBL
    ! has none of that name.
    character(len=*), intent(in) :: name
    logical, intent(out) :: found
    character(len=*), intent(out) :: lname, units
    integer :: m
    found = .false.; lname = ''; units = ''
    do m = 1, size(mi%surface_flux_diags%diags)
      if (trim(mi%surface_flux_diags%diags(m)%short_name) /= trim(name)) cycle
      found = .true.
      lname = trim(mi%surface_flux_diags%diags(m)%long_name)
      units = trim(mi%surface_flux_diags%diags(m)%units)
      return
    enddo
    do m = 1, size(mi%interior_tendency_diags%diags)
      if (trim(mi%interior_tendency_diags%diags(m)%short_name) /= trim(name)) cycle
      found = .true.
      lname = trim(mi%interior_tendency_diags%diags(m)%long_name)
      units = trim(mi%interior_tendency_diags%diags(m)%units)
      return
    enddo
  end subroutine marbl_diag_info

  subroutine marbl_host_print_log()
    use marbl_logging, only: marbl_status_log_entry_type
    type(marbl_status_log_entry_type), pointer :: e
    e => mi%StatusLog%FullLog
    do while (associated(e))
      write(*,'(a)') trim(e%LogMessage)
      e => e%next
    enddo
  end subroutine marbl_host_print_log

  ! Reference: unchanged MARBL surface_flux_compute on one column, marshalled
  ! exactly as the CPU ROMS driver does (ifrac = 0, atm pressure = 1).
  !   trs      tracers at the surface (model conc units), size nt_marbl
  !   ph, ph_alt   saved state in/out
  !   flux     surface fluxes (conc units * m/s), size nt_marbl
  !   diag     MSF_NDIAG diagnostics
  subroutine marbl_host_sflux_column(trs, u10sqr, sst, sss, xco2, xco2_alt, dust, iron, &
                                     nox, nhy, ph, ph_alt, flux, diag, ierr)
    use marbl_diagnostics_share_mod, only: marbl_surface_flux_diag_ind
    real(rp), intent(in) :: trs(:)
    real(rp), intent(in) :: u10sqr, sst, sss, xco2, xco2_alt, dust, iron, nox, nhy
    real(rp), intent(inout) :: ph, ph_alt
    real(rp), intent(out) :: flux(:), diag(MSF_NDIAG)
    integer, intent(out) :: ierr
    integer :: m

    ierr = 0
    associate(f => mi%surface_flux_forcings, d => mi%surface_flux_diags%diags, &
              di => marbl_surface_flux_diag_ind)
    if (sfi_sss > 0)      f(sfi_sss)%field_0d(1) = sss
    if (sfi_sst > 0)      f(sfi_sst)%field_0d(1) = sst
    if (sfi_ifrac > 0)    f(sfi_ifrac)%field_0d(1) = msf%ifrac
    if (sfi_u10sqr > 0)   f(sfi_u10sqr)%field_0d(1) = u10sqr
    if (sfi_atmp > 0)     f(sfi_atmp)%field_0d(1) = msf%atm_pres
    if (sfi_xco2 > 0)     f(sfi_xco2)%field_0d(1) = xco2
    if (sfi_xco2_alt > 0) f(sfi_xco2_alt)%field_0d(1) = xco2_alt
    if (sfi_dust > 0)     f(sfi_dust)%field_0d(1) = dust
    if (sfi_iron > 0)     f(sfi_iron)%field_0d(1) = iron
    if (sfi_nox > 0)      f(sfi_nox)%field_0d(1) = nox
    if (sfi_nhy > 0)      f(sfi_nhy)%field_0d(1) = nhy
    do m = 1, nt_marbl
      mi%tracers_at_surface(1, m) = trs(m)
    enddo
    mi%surface_flux_saved_state%state(mi%surf_state_ind%base_bio_ph_surf)%field_2d(1) = ph
    mi%surface_flux_saved_state%state(mi%surf_state_ind%base_bio_ph_alt_co2_surf)%field_2d(1) = ph_alt

    call mi%surface_flux_compute()
    if (mi%StatusLog%labort_marbl) then
      ierr = 1
      call marbl_host_print_log()
    endif
    call mi%StatusLog%erase()

    ph     = mi%surface_flux_saved_state%state(mi%surf_state_ind%base_bio_ph_surf)%field_2d(1)
    ph_alt = mi%surface_flux_saved_state%state(mi%surf_state_ind%base_bio_ph_alt_co2_surf)%field_2d(1)
    do m = 1, nt_marbl
      flux(m) = mi%surface_fluxes(1, m)
    enddo
    diag = 0.0_rp
    if (msf%flux_co2 /= 0) then
      diag(D_PH)        = d(di%PH)%field_2d(1)
      diag(D_PCO2)      = d(di%pCO2SURF)%field_2d(1)
      diag(D_DPCO2)     = d(di%DpCO2)%field_2d(1)
      diag(D_FGCO2)     = d(di%FG_CO2)%field_2d(1)
      diag(D_PH_ALT)    = d(di%PH_ALT_CO2)%field_2d(1)
      diag(D_PCO2_ALT)  = d(di%pCO2SURF_ALT_CO2)%field_2d(1)
      diag(D_FGCO2_ALT) = d(di%FG_CO2_ALT_CO2)%field_2d(1)
      diag(D_CO2STAR)   = d(di%CO2STAR)%field_2d(1)
      diag(D_DCO2STAR)  = d(di%DCO2STAR)%field_2d(1)
      diag(D_PVCO2)     = d(di%PV_CO2)%field_2d(1)
      diag(D_SCHCO2)    = d(di%SCHMIDT_CO2)%field_2d(1)
    endif
    if (msf%flux_o2 /= 0) then
      diag(D_PVO2)  = d(di%PV_O2)%field_2d(1)
      diag(D_SCHO2) = d(di%SCHMIDT_O2)%field_2d(1)
      diag(D_O2SAT) = d(di%O2SAT)%field_2d(1)
    endif
    diag(D_NHX) = d(di%NHx_SURFACE_EMIS)%field_2d(1)
    end associate
  end subroutine marbl_host_sflux_column

  ! Copy the MARBL settings and PFT tables into the flat parameter type.
  subroutine marbl_host_fill_par(p)
    use marbl_settings_mod
    use marbl_pft_mod, only: Qp_zoo
    use marbl_constants_mod, only: rho_sw
    type(marbl_par_t), intent(out) :: p
    integer :: n, m, j, i
    real(rp), parameter :: mass2kg = 1.0_rp, m2len = 1.0_rp
    if (.not. marbl_host_ready) error stop 'marbl_host_fill_par: call marbl_host_init first'
    if (autotroph_cnt > MA .or. zooplankton_cnt > MZ .or. max_grazer_prey_cnt > MP .or. nt_marbl > NTRMAX) &
         error stop 'marbl_host_fill_par: raise MA/MZ/MP/NTRMAX'
    if (Lig_cnt /= 1) error stop 'marbl_host_fill_par: only Lig_cnt = 1 is ported'
    p%ntr = nt_marbl; p%nauto = autotroph_cnt; p%nzoo = zooplankton_cnt; p%nprey = max_grazer_prey_cnt
    associate(ti => mi%tracer_indices)
    p%po4 = ti%po4_ind; p%no3 = ti%no3_ind; p%sio3 = ti%sio3_ind; p%nh4 = ti%nh4_ind; p%fe = ti%fe_ind
    p%lig = ti%lig_ind; p%o2 = ti%o2_ind; p%dic = ti%dic_ind; p%dica = ti%dic_alt_co2_ind
    p%alk = ti%alk_ind; p%alka = ti%alk_alt_co2_ind; p%doc = ti%doc_ind; p%don = ti%don_ind
    p%dop = ti%dop_ind; p%dopr = ti%dopr_ind; p%donr = ti%donr_ind; p%docr = ti%docr_ind
    do n = 1, autotroph_cnt
      p%a_chl(n) = ti%auto_inds(n)%Chl_ind; p%a_c(n) = ti%auto_inds(n)%C_ind
      p%a_p(n) = ti%auto_inds(n)%P_ind; p%a_fe(n) = ti%auto_inds(n)%Fe_ind
      p%a_si(n) = ti%auto_inds(n)%Si_ind; p%a_caco3(n) = ti%auto_inds(n)%CaCO3_ind
    enddo
    do n = 1, zooplankton_cnt
      p%z_c(n) = ti%zoo_inds(n)%C_ind
    enddo
    end associate
    do n = 1, autotroph_cnt
      associate(a => autotroph_settings(n))
      p%a_tfunc(n) = a%temp_func_form_iopt
      p%a_nfixer(n) = merge(1, 0, a%Nfixer); p%a_impcalc(n) = merge(1, 0, a%imp_calcifier)
      p%a_expcalc(n) = merge(1, 0, a%exp_calcifier); p%a_silic(n) = merge(1, 0, a%silicifier)
      p%a_climit(n) = merge(1, 0, a%is_carbon_limited)
      p%a_kFe(n) = a%kFe; p%a_kCO2(n) = a%kCO2; p%a_kPO4(n) = a%kPO4; p%a_kDOP(n) = a%kDOP
      p%a_kNO3(n) = a%kNO3; p%a_kNH4(n) = a%kNH4; p%a_kSiO3(n) = a%kSiO3; p%a_Qp_fixed(n) = a%Qp_fixed
      p%a_gQfe_0(n) = a%gQfe_0; p%a_gQfe_min(n) = a%gQfe_min; p%a_alphaPI(n) = a%alphaPI
      p%a_PCref(n) = a%PCref; p%a_thetaN_max(n) = a%thetaN_max; p%a_loss_thres(n) = a%loss_thres
      p%a_loss_thres2(n) = a%loss_thres2; p%a_temp_thres(n) = a%temp_thres; p%a_mort(n) = a%mort
      p%a_mort2(n) = a%mort2; p%a_agg_rate_max(n) = a%agg_rate_max; p%a_agg_rate_min(n) = a%agg_rate_min
      p%a_loss_poc(n) = a%loss_poc; p%a_Tref(n) = a%Tref; p%a_Ea(n) = a%Ea
      end associate
    enddo
    do n = 1, zooplankton_cnt
      associate(z => zooplankton_settings(n))
      p%z_tfunc(n) = z%temp_func_form_iopt
      p%z_mort_0(n) = z%z_mort_0; p%z_mort2_0(n) = z%z_mort2_0; p%z_basal(n) = z%basal_respiration_rate
      p%z_loss_thres(n) = z%loss_thres; p%z_Tref(n) = z%Tref; p%z_Ea(n) = z%Ea
      end associate
    enddo
    do m = 1, zooplankton_cnt
      do j = 1, max_grazer_prey_cnt
        associate(g => grazing_relationship_settings(j, m))
        p%g_nauto(j, m) = g%auto_ind_cnt; p%g_nzoo(j, m) = g%zoo_ind_cnt; p%g_func(j, m) = g%grazing_function
        do i = 1, g%auto_ind_cnt
          p%g_auto(i, j, m) = g%auto_ind(i)
        enddo
        do i = 1, g%zoo_ind_cnt
          p%g_zoo(i, j, m) = g%zoo_ind(i)
        enddo
        p%g_umax(j, m) = g%z_umax_0; p%g_grz(j, m) = g%z_grz; p%g_zoofrac(j, m) = g%graze_zoo
        p%g_poc(j, m) = g%graze_poc; p%g_doc(j, m) = g%graze_doc; p%g_fdetr(j, m) = g%f_zoo_detr
        end associate
      enddo
    enddo
    p%lvar_ptoc = merge(1, 0, lvariable_PtoC)
    p%caco3_bury_iopt = merge(1, 2, caco3_bury_thres_iopt == caco3_bury_thres_iopt_fixed_depth)
    p%Q = Q; p%Qp_zoo = Qp_zoo; p%Qfe_zoo = Qfe_zoo
    p%Red_D_C_O2 = parm_Red_D_C_O2; p%Remin_D_C_O2 = parm_Remin_D_C_O2; p%Red_D_C_O2_diaz = parm_Red_D_C_O2_diaz
    p%Red_Fe_C = parm_Red_Fe_C; p%dust_to_Fe = dust_to_Fe; p%denitrif_C_N = denitrif_C_N
    p%f_toDON = f_toDON; p%f_toDOP = f_toDOP
    p%POCremin_refract = POCremin_refract; p%PONremin_refract = PONremin_refract; p%POPremin_refract = POPremin_refract
    p%DOCprod_refract = DOCprod_refract; p%DONprod_refract = DONprod_refract; p%DOPprod_refract = DOPprod_refract
    p%f_graze_CaCO3_remin = f_graze_CaCO3_remin; p%f_graze_si_remin = f_graze_si_remin
    p%f_graze_sp_poc_lim = f_graze_sp_poc_lim; p%caco3_poc_min = caco3_poc_min; p%spc_poc_fac = spc_poc_fac
    p%labile_ratio = parm_labile_ratio; p%o2_min = parm_o2_min; p%o2_min_delta = parm_o2_min_delta
    p%kappa_nitrif = parm_kappa_nitrif; p%nitrif_par_lim = parm_nitrif_par_lim
    p%scalelen_z = parm_scalelen_z; p%scalelen_vals = parm_scalelen_vals
    p%POC_diss = parm_POC_diss; p%CaCO3_diss = parm_CaCO3_diss; p%CaCO3_gamma = parm_CaCO3_gamma
    p%hPOC_CaCO3_ratio = parm_hPOC_CaCO3_ratio; p%hPOC_SiO2_ratio = parm_hPOC_SiO2_ratio
    p%hPOC_dust_ratio = parm_hPOC_dust_ratio; p%SiO2_diss = parm_SiO2_diss; p%SiO2_gamma = parm_SiO2_gamma
    p%Fe_desorption_rate0 = parm_Fe_desorption_rate0; p%sed_denitrif_coeff = parm_sed_denitrif_coeff
    p%flux_ref_depth = particulate_flux_ref_depth; p%caco3_bury_depth = caco3_bury_thres_depth
    p%caco3_bury_omega = caco3_bury_thres_omega_calc; p%POM_bury_frac_max = POM_bury_frac_max
    p%bSi_bury_frac_max = bSi_bury_frac_max; p%o2_sf_hi = o2_sf_o2_range_hi; p%o2_sf_lo = o2_sf_o2_range_lo
    p%o2_sf_val = o2_sf_val_lo_o2; p%Fe_scav_yps = parm_Fe_scavenge_rate0_yps
    p%Lig_scav_yps = parm_Lig_scavenge_rate0_yps; p%FeLig_scav_yps = parm_FeLig_scavenge_rate0_yps
    p%dust_Fe_scav_scale = dust_Fe_scavenge_scale; p%remin_to_Lig = remin_to_Lig
    p%Lig_degrade_rate0 = parm_Lig_degrade_rate0
    p%DOC_reminR_light = DOC_reminR_light; p%DON_reminR_light = DON_reminR_light; p%DOP_reminR_light = DOP_reminR_light
    p%DOC_reminR_dark = DOC_reminR_dark; p%DON_reminR_dark = DON_reminR_dark; p%DOP_reminR_dark = DOP_reminR_dark
    p%DOCr_reminR0 = DOCr_reminR0; p%DONr_reminR0 = DONr_reminR0; p%DOPr_reminR0 = DOPr_reminR0
    p%DOMr_reminR_photo = DOMr_reminR_photo; p%f_qsw_par = f_qsw_par; p%Q_10 = Q_10
    p%thres_z1_auto = thres_z1_auto; p%thres_z2_auto = thres_z2_auto
    p%thres_z1_zoo = thres_z1_zoo; p%thres_z2_zoo = thres_z2_zoo
    p%auto_mort2_exp = auto_mort2_exp; p%zoo_mort2_exp = zoo_mort2_exp
    p%gQsi_0 = gQsi_0; p%gQsi_max = gQsi_max; p%gQsi_min = gQsi_min
    p%gQ_Fe_kFe_thres = gQ_Fe_kFe_thres; p%gQ_Si_kSi_thres = gQ_Si_kSi_thres
    p%PquotaSlope = PquotaSlope; p%PquotaIntercept = PquotaIntercept; p%PquotaMinNP = PquotaMinNP
    p%QCaCO3_max = QCaCO3_max; p%f_prod_sp_CaCO3 = parm_f_prod_sp_CaCO3; p%CaCO3_sp_thres = CaCO3_sp_thres
    p%CaCO3_temp_thres1 = CaCO3_temp_thres1; p%CaCO3_temp_thres2 = CaCO3_temp_thres2
    p%f_photosp_CaCO3 = f_photosp_CaCO3; p%r_Nfix_photo = r_Nfix_photo
    p%POC_bury_coeff = parm_init_POC_bury_coeff; p%POP_bury_coeff = parm_init_POP_bury_coeff
    p%bSi_bury_coeff = parm_init_bSi_bury_coeff; p%PON_bury_coeff = PON_bury_coeff
    p%del_ph = del_ph; p%phlo_3d_init = phlo_3d_init; p%phhi_3d_init = phhi_3d_init
    p%mass_to_vol = rho_sw * (1.0e3_rp * mass2kg * m2len**3)
    if (init_bury_coeff_opt /= 'settings_file') error stop 'marbl_host_fill_par: init_bury_coeff_opt'
    if (lo2_consumption_scalef .or. lp_remin_scalef) error stop 'marbl_host_fill_par: scale-factor forcings not ported'
    if (.not. lsource_sink) error stop 'marbl_host_fill_par: lsource_sink = .false.'
  end subroutine marbl_host_fill_par

  ! Reference: unchanged MARBL interior_tendency_compute on one column,
  ! marshalled as the CPU ROMS driver does. MARBL level 1 is the surface.
  !   dz, zw, zt   layer thickness, bottom-interface depth, centre depth (m, positive)
  !   potemp, salt, press(bar), qsw (W/m2), dust (kg/m2/s)
  !   tr(ntr, nz) tracers; ph, ph_alt(nz) saved state in/out
  !   tend(ntr, nz) tendencies (conc/s); diag(ND_IT, nz) debug diagnostics
  subroutine marbl_host_interior_column(nz, dz, zw, zt, potemp, salt, press, qsw, dust, tr, &
                                        ph, ph_alt, tend, diag, ierr)
    use marbl_diagnostics_share_mod, only: marbl_interior_tendency_diag_ind
    integer(ip), intent(in) :: nz
    real(rp), intent(in) :: dz(nz), zw(nz), zt(nz), potemp(nz), salt(nz), press(nz), qsw, dust
    real(rp), intent(in) :: tr(:,:)
    real(rp), intent(inout) :: ph(nz), ph_alt(nz)
    real(rp), intent(out) :: tend(:,:), diag(ND_IT, nz)
    integer, intent(out) :: ierr
    integer :: m, k, idx
    ierr = 0
    mi%domain%zw(:) = zw; mi%domain%zt(:) = zt; mi%domain%delta_z(:) = dz; mi%domain%kmt = nz
    do idx = 1, size(mi%interior_tendency_forcings)
      select case (trim(mi%interior_tendency_forcings(idx)%metadata%varname))
      case ('Dust Flux');              mi%interior_tendency_forcings(idx)%field_0d(1) = dust
      case ('Surface Shortwave');      mi%interior_tendency_forcings(idx)%field_1d(1, 1) = qsw
      case ('Potential Temperature');  mi%interior_tendency_forcings(idx)%field_1d(1, :) = potemp
      case ('Salinity');               mi%interior_tendency_forcings(idx)%field_1d(1, :) = salt
      case ('Pressure');               mi%interior_tendency_forcings(idx)%field_1d(1, :) = press
      case ('Iron Sediment Flux');     mi%interior_tendency_forcings(idx)%field_1d(1, :) = 0.0_rp
      case default
        error stop 'marbl_host_interior_column: unexpected interior forcing'
      end select
    enddo
    do m = 1, nt_marbl
      mi%tracers(m, :) = tr(m, :)
    enddo
    mi%interior_tendency_saved_state%state(mi%interior_state_ind%ph_col)%field_3d(:, 1) = ph
    mi%interior_tendency_saved_state%state(mi%interior_state_ind%ph_alt_co2_col)%field_3d(:, 1) = ph_alt
    mi%bot_flux_to_tend(:) = 0.0_rp
    mi%bot_flux_to_tend(nz) = 1.0_rp / dz(nz)
    call mi%interior_tendency_compute()
    if (mi%StatusLog%labort_marbl) then
      ierr = 1
      call marbl_host_print_log()
    endif
    call mi%StatusLog%erase()
    do m = 1, nt_marbl
      tend(m, :) = mi%interior_tendencies(m, :)
    enddo
    ph = mi%interior_tendency_saved_state%state(mi%interior_state_ind%ph_col)%field_3d(:, 1)
    ph_alt = mi%interior_tendency_saved_state%state(mi%interior_state_ind%ph_alt_co2_col)%field_3d(:, 1)
    associate(d => mi%interior_tendency_diags%diags, di => marbl_interior_tendency_diag_ind)
    diag(1, :)  = d(di%PAR_avg)%field_3d(:, 1)
    diag(2, :)  = d(di%NITRIF)%field_3d(:, 1)
    diag(3, :)  = d(di%DENITRIF)%field_3d(:, 1)
    diag(4, :)  = d(di%photoC_TOT)%field_3d(:, 1)
    diag(5, :)  = d(di%POC_FLUX_IN)%field_3d(:, 1)
    diag(6, :)  = d(di%POC_REMIN_DIC)%field_3d(:, 1)
    diag(7, :)  = d(di%O2_PRODUCTION)%field_3d(:, 1)
    diag(8, :)  = d(di%O2_CONSUMPTION)%field_3d(:, 1)
    diag(9, :)  = d(di%Fe_scavenge)%field_3d(:, 1)
    diag(10, :) = d(di%insitu_temp)%field_3d(:, 1)
    diag(11, :) = d(di%CO3)%field_3d(:, 1)
    diag(12, :) = d(di%H2CO3)%field_3d(:, 1)
    diag(13, :) = d(di%co3_sat_calc)%field_3d(:, 1)
    diag(14, :) = d(di%pH_3D)%field_3d(:, 1)
    diag(15, :) = d(di%DOC_prod)%field_3d(:, 1)
    diag(16, :) = d(di%POC_PROD)%field_3d(:, 1)
    end associate
  end subroutine marbl_host_interior_column

end module marbl_host

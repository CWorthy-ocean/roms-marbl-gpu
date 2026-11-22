#include "mem_attr.inc"
module cdr_lite_io
  ! CDR_LITE surface forcing, host side: &cdr_lite_settings, the DIC
  ! tracers and their ALK pairs, the forcing series (frc_io), and the
  ! surface fluxes at the corrector's forcing time (cdr_lite_mod). As the
  ! CPU, the carbonate sensitivities ddic_dco2/ddic_dalk must be in the
  ! forcing files; <name>_flx of a DIC tracer may be absent (0).
  ! cdr_online_carbonate_sensitivity: beta/eta computed every corrector from
  ! the predicted ALT_CO2 surface state (carb_sens_mod, the CPU's
  ! set_online_carbonate_sensitivity) instead of read; the ddic files are
  ! then not needed.
  use kinds
  use nc_io, only: iomsg_text
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use comm_mod, only: comm_rank
  use passive_mod, only: ntp, tp, n_oae, n_dor, ncdr, cdr0, passive_name, n_marbl
  use carb_sens_mod, only: cs_surface
  use cdr_lite_mod
  use frc_io, only: init_cdr_lite_forcing, set_cdr_lite_forces, uwnd, vwnd
  implicit none
  private
  public :: read_cdr_lite_namelist, init_cdr_lite, set_cdr_lite, alt_co2_tracers

  logical :: cdr_online_carbonate_sensitivity = .false.
  ! ddic_dco2/ddic_dalk from the forcing files: .true. (default) linear in
  ! time between the bracketing records; .false. each record held until the
  ! model time passes the next one, as the CPU (frc_io hold_slot), without
  ! the missing-record rule (missing_sensitivity), as the CPU.
  logical :: cdr_carbonate_sensitivity_time_interpolation = .true.
  namelist /cdr_lite_settings/ cdr_online_carbonate_sensitivity, cdr_carbonate_sensitivity_time_interpolation
  integer(ip) :: ialk = 0, idic = 0, ipo4 = 0, isio3 = 0      ! ALT_CO2 system (online)
  integer :: ir0, ir1, jr0, jr1

contains

  subroutine read_cdr_lite_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=512) :: msg
    cdr_carbonate_sensitivity_time_interpolation = .true.
    rewind(unit)
    read(unit, nml=cdr_lite_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'CDR_LITE_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_cdr_lite_namelist'
    endif
    if (ios /= 0) cdr_online_carbonate_sensitivity = .false.
  end subroutine read_cdr_lite_namelist

  subroutine init_cdr_lite()
    ! After the passive tracer list and the wet columns are final.
    integer(ip) :: nd, m
    integer(ip), allocatable :: dic(:), alk(:)
    character(len=32), allocatable :: names(:)
    logical :: hd, ha
    logical, allocatable :: hf(:)
    if (ncdr == 0) then
      call cdrl_ensure()
      return
    endif
    if (cdr_online_carbonate_sensitivity) then
      call alt_co2_tracers(ialk, idic, ipo4, isio3)
      if (ialk == 0 .or. idic == 0 .or. ipo4 == 0 .or. isio3 == 0) &
        error stop 'init_cdr_lite: cdr_online_carbonate_sensitivity needs the MARBL tracers PO4, SiO3, '// &
                   'ALK_ALT_CO2 and DIC_ALT_CO2'
      ! interior and physical boundary points (the gas flux uses both)
      ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
      jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    endif
    nd = n_oae + n_dor
    allocate(dic(max(nd, 1)), alk(max(nd, 1)), names(max(nd, 1)), hf(max(nd, 1)))
    do m = 1, n_oae
      dic(m) = cdr0 + 2 * m; alk(m) = cdr0 + 2 * m - 1
    enddo
    do m = 1, n_dor
      dic(n_oae + m) = cdr0 + 2 * n_oae + m; alk(n_oae + m) = 0
    enddo
    call cdrl_setup(nd, dic, alk, ntp)
    do m = 1, nd
      names(m) = passive_name(dic(m))
    enddo
    call init_cdr_lite_forcing(int(nd), names, .not. cdr_online_carbonate_sensitivity, &
                               cdr_carbonate_sensitivity_time_interpolation, hd, ha, hf)
    if (.not. cdr_online_carbonate_sensitivity .and. .not. (hd .and. ha)) then
      write(*,'(a,2l2)') 'init_cdr_lite: ddic_dco2, ddic_dalk found =', hd, ha
      error stop 'init_cdr_lite: CDR_LITE needs ddic_dco2 and ddic_dalk in the forcing files'
    endif
    if (comm_rank() == 0) then
      do m = 1, nd
        write(*,'(a,a,a,i0,a,i0,a,l2)') 'cdr_lite: ', trim(names(m)), ' (tracer ', dic(m), ', ALK ', alk(m), &
             ')  _flx found =', hf(m)
      enddo
      if (.not. allocated(uwnd)) write(*,'(a)') 'cdr_lite: no bulk winds, k_gas = 0'
      if (cdr_online_carbonate_sensitivity) write(*,'(a)') 'cdr_lite: carbonate sensitivities computed online (ALT_CO2)'
    endif
    deallocate(dic, alk, names, hf)
  end subroutine init_cdr_lite

  subroutine set_cdr_lite()
    ! After the corrector's set_forces (frc_slot 1, n+1/2): the forcing at
    ! n+1/2 and the surface fluxes from the predicted state (slot nrhs).
    real(rp) :: tneed
    if (.not. cdrl_on) return
    tneed = frc_modtime()
    call set_cdr_lite_forces(tneed, int(cdrl_nd), cdrl_ddco2, cdrl_ddalk, cdrl_flx)
    if (cdr_online_carbonate_sensitivity) &
      call cs_surface(tp, ntp, nrhs, ialk, idic, ipo4, isio3, int(ir0, ip), int(ir1, ip), int(jr0, ip), int(jr1, ip), &
                      cdrl_ddco2, cdrl_ddalk)
    if (allocated(uwnd)) then
      call cdrl_sflx(tp, ntp, nrhs, uwnd, vwnd, .true.)
    else
      call cdrl_sflx(tp, ntp, nrhs, cdrl_ddco2, cdrl_ddco2, .false.)
    endif
  end subroutine set_cdr_lite

  subroutine alt_co2_tracers(ia, id, ipo, isi)
    ! Passive indices of ALK_ALT_CO2, DIC_ALT_CO2, PO4 and SiO3 (0: absent).
    integer(ip), intent(out) :: ia, id, ipo, isi
    integer(ip) :: m
    ia = 0; id = 0; ipo = 0; isi = 0
    do m = 1, n_marbl
      select case (trim(passive_name(m)))
      case ('ALK_ALT_CO2'); ia = m
      case ('DIC_ALT_CO2'); id = m
      case ('PO4'); ipo = m
      case ('SiO3'); isi = m
      end select
    enddo
  end subroutine alt_co2_tracers

end module cdr_lite_io

#include "mem_attr.inc"
module bulk_coare
  ! COARE 3.0 bulk fluxes (Fairall et al. 2003), ported from
  ! roms_bugfix/src/bulk_frc.F90, with SFLX_CORR and CFLX_CORR (namelist
  ! switches, frc_io) and the SEA_ICE_NOFLUX cap. No QCORR / TAU_CORR.
  ! All loops are device kernels (pointwise; the COARE iteration stays
  ! serial per point). Inputs must be managed/device arrays.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use formulas
  implicit none
  private

  real(rp), parameter :: blk_Rgas = 287.1_rp
  real(rp), parameter :: blk_ZW = 10.0_rp
  real(rp), parameter :: blk_ZT = 10.0_rp
  real(rp), parameter :: blk_ZQ = 10.0_rp
  real(rp), parameter :: blk_Zabl = 600.0_rp
  real(rp), parameter :: blk_beta = 1.2_rp
  real(rp), parameter :: blk_Cpa = 1004.67_rp
  real(rp), parameter :: emiss_lw = 0.985_rp
  real(rp), parameter :: SigmaSB = 5.6697e-8_rp
  real(rp), parameter :: patm = 1010.0_rp
  real(rp), parameter :: eps = 1.0e-20_rp
  real(rp), parameter :: r3 = 1.0_rp / 3.0_rp
  real(rp), parameter :: cfb_slope = -0.0029_rp
  real(rp), parameter :: cfb_offset = 0.008_rp
  real(rp), parameter :: Wspd_min = 3.0_rp
  real(rp), parameter :: stau_ref = -0.0027_rp
  real(rp), parameter :: cmday2ms = 0.01_rp / 86400.0_rp
  real(rp), parameter :: Cp = 3985.0_rp

  public :: calc_bulk_coare, calc_cflx_corr

contains

  subroutine calc_bulk_coare(uwnd, vwnd, tair, qair, prate, lwrad, do_sss, dsss, sss)
    real(rp), WORK_ATTR intent(in) :: uwnd(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: vwnd(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: tair(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: qair(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: prate(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: lwrad(decomp%is:decomp%ie, decomp%js:decomp%je)
    logical,  intent(in) :: do_sss                 ! SFLX_CORR salinity restoring
    real(rp), intent(in) :: dsss                   ! piston velocity, m/s
    real(rp), WORK_ATTR intent(in) :: sss(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, iter, itermax, nx, ny, nz, is, ie, js, je, n_rhs, i_fb, j_fb
    real(rp) :: rho0i, cpi, TseaC, TseaK, TairC, TairK, Qsea, rhoAir
    real(rp) :: VisAir, Hlv, delW, delT, delQ, u10, Zo10, Cd10, Ch10, Ct10, Cd
    real(rp) :: Ct, CC, Ri, Ribcu, Zetu, L10, Wstar, Tstar, Qstar
    real(rp) :: ZoW, ZoT, ZoQ, ZoL, L, Rr, Bff, Wpsi, Tpsi, Qpsi
    real(rp) :: wspd0, Wgus, charn, hfsen, hflat, hflw, upvel, radlw, evap
    real(rp) :: cff, wspd, r0, aer, cer

    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    n_rhs = nrhs
    r0 = rho0_case
    rho0i = 1.0_rp / r0
    cpi = 1.0_rp / Cp

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        sustr_r(i, j) = 0.0_rp
        svstr_r(i, j) = 0.0_rp
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        if (rmask(i, j) < 0.5_rp) then
          stflx(i, j, itemp) = 0.0_rp
          swflx(i, j) = 0.0_rp
        else

        wspd0 = sqrt(uwnd(i, j) * uwnd(i, j) + vwnd(i, j) * vwnd(i, j))
        radlw = lwrad(i, j) / (r0 * Cp)
        TairC = tair(i, j)
        TairK = TairC + 273.16_rp
        TseaC = t(i, j, nz, n_rhs, itemp)
        TseaK = TseaC + 273.16_rp
        hflw = radlw - emiss_lw * rho0i * cpi * SigmaSB * TseaK**4

        cff = (1.0007_rp + 3.46e-6_rp * patm) * 6.1121_rp * &
              exp(17.502_rp * TseaC / (240.97_rp + TseaC))
        cff = cff * 0.98_rp
        Qsea = 0.62197_rp * (cff / (patm - 0.378_rp * cff))

        rhoAir = patm * 100.0_rp / (blk_Rgas * TairK * (1.0_rp + 0.61_rp * qair(i, j)))
        VisAir = 1.326e-5_rp * (1.0_rp + TairC * (6.542e-3_rp + TairC * &
                 (8.301e-6_rp - 4.84e-9_rp * TairC)))
        Hlv = (2.501_rp - 0.00237_rp * TseaC) * 1.0e6_rp

        Wgus = 0.5_rp
        delW = sqrt(wspd0 * wspd0 + Wgus * Wgus)
        delQ = Qsea - qair(i, j)
        delT = TseaC - TairC

        ZoW = 0.0001_rp
        u10 = delW * log(10.0_rp / ZoW) / log(blk_ZW / ZoW)
        Wstar = 0.035_rp * u10
        Zo10 = 0.011_rp * Wstar * Wstar / g + 0.11_rp * VisAir / max(Wstar, eps)
        Cd10 = (vonkar / log(10.0_rp / Zo10))**2
        Ch10 = 0.00115_rp
        Ct10 = Ch10 / sqrt(Cd10)
        ZoT = 10.0_rp / exp(vonkar / Ct10)
        Cd = (vonkar / log(blk_ZW / Zo10))**2
        Ct = vonkar / log(blk_ZT / ZoT)
        CC = vonkar * Ct / Cd
        Ribcu = -blk_ZW / (blk_Zabl * 0.004_rp * blk_beta**3)
        Ri = -g * blk_ZW * (delT + 0.61_rp * TairK * delQ) / (TairK * delW * delW)
        if (Ri < 0.0_rp) then
          Zetu = CC * Ri / (1.0_rp + Ri / Ribcu)
        else
          Zetu = CC * Ri / (1.0_rp + 3.0_rp * Ri / CC)
        endif
        L10 = blk_ZW / (Zetu + sign(eps, Zetu + eps))
        if (Zetu > 50.0_rp) then
          itermax = 1
        else
          itermax = 3
        endif

        Wstar = delW * vonkar / (log(blk_ZW / Zo10) - bulk_psiu(blk_ZW / L10))
        Tstar = -delT * vonkar / (log(blk_ZT / ZoT) - bulk_psit(blk_ZT / L10))
        Qstar = -delQ * vonkar / (log(blk_ZQ / ZoT) - bulk_psit(blk_ZQ / L10))

        if (delW > 18.0_rp) then
          charn = 0.018_rp
        elseif (delW > 10.0_rp) then
          charn = 0.011_rp + 0.125_rp * (0.018_rp - 0.011_rp) * (delW - 10.0_rp)
        else
          charn = 0.011_rp
        endif

        do iter = 1, itermax
          ZoW = charn * Wstar * Wstar / g + 0.11_rp * VisAir / (Wstar + eps)
          Rr = ZoW * Wstar / VisAir
          ZoQ = min(1.15e-4_rp, 5.5e-5_rp / max(Rr, eps)**0.6_rp)
          ZoT = ZoQ
          ZoL = vonkar * g * blk_ZW * (Tstar * (1.0_rp + 0.61_rp * qair(i, j)) + &
                0.61_rp * TairK * Qstar) / &
                (TairK * Wstar * Wstar * (1.0_rp + 0.61_rp * qair(i, j)) + eps)
          L = blk_ZW / (ZoL + eps)
          Wpsi = bulk_psiu(ZoL)
          Tpsi = bulk_psit(blk_ZT / L)
          Qpsi = bulk_psit(blk_ZQ / L)
          Wstar = max(eps, delW * vonkar / (log(blk_ZW / ZoW) - Wpsi))
          Tstar = -delT * vonkar / (log(blk_ZT / ZoT) - Tpsi)
          Qstar = -delQ * vonkar / (log(blk_ZQ / ZoQ) - Qpsi)
          Bff = -g / TairK * Wstar * (Tstar + 0.61_rp * TairK * Qstar)
          if (Bff > 0.0_rp) then
            Wgus = blk_beta * (Bff * blk_Zabl)**r3
          else
            Wgus = 0.2_rp
          endif
          delW = sqrt(wspd0 * wspd0 + Wgus * Wgus)
        enddo

        wspd0 = sqrt(wspd0 * wspd0 + Wgus * Wgus)
        Cd = Wstar * Wstar / (wspd0 * wspd0 + eps)
        hfsen = -blk_Cpa * rhoAir * Wstar * Tstar
        hflat = -Hlv * rhoAir * Wstar * Qstar
        upvel = -1.61_rp * Wstar * Qstar - (1.0_rp + 1.61_rp * qair(i, j)) * Wstar * Tstar / TairK
        hflat = hflat + rhoAir * Hlv * upvel * qair(i, j)
        hflat = -hflat * rho0i * cpi
        hfsen = -hfsen * rho0i * cpi
        stflx(i, j, itemp) = apply_mask(srflx(i, j) + hflw + hflat + hfsen, rmask(i, j))
        evap = -Cp * hflat / Hlv
        swflx(i, j) = apply_mask(prate(i, j) * cmday2ms - evap, rmask(i, j))
        ! roms_bugfix SEA_ICE_NOFLUX (set_global_definitions.h, always on): no
        ! surface heat flux and no shortwave where the sea surface is at or
        ! below -1.8 C ("poor man's sea ice").
        if (TseaC <= -1.8_rp) then
          stflx(i, j, itemp) = 0.0_rp
          srflx(i, j) = 0.0_rp
        endif
        aer = rhoAir * wspd0 * rho0i
        cer = Cd
        sustr_r(i, j) = apply_mask(aer * cer * uwnd(i, j), rmask(i, j))
        svstr_r(i, j) = apply_mask(aer * cer * vwnd(i, j), rmask(i, j))
        endif
      enddo
    enddo

    ! Surface stress current feedback (the i = nx+1 / j = ny+1 points of the
    ! CPU loops only computed an unused cff, so the ranges stop at nx / ny at
    ! a physical edge). Toward a neighbor tile the point is a seam face that
    ! the neighbor computes as its own i = 1 / j = 1, so it is included there.
    i_fb = merge(nx, nx + 1, decomp%edge_e)
    j_fb = merge(ny, ny + 1, decomp%edge_n)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 0, i_fb
        wspd = sqrt(uwnd(i, j) * uwnd(i, j) + vwnd(i, j) * vwnd(i, j))
        if (wspd > Wspd_min) then
          cff = cfb_slope * wspd + cfb_offset
        else
          cff = stau_ref
        endif
        sustr_r(i, j) = sustr_r(i, j) + cff * 0.5_rp * (u(i, j, nz, n_rhs) + u(i + 1, j, nz, n_rhs)) / r0
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        sustr(i, j) = apply_mask(0.5_rp * (sustr_r(i - 1, j) + sustr_r(i, j)), umask(i, j))
      enddo
    enddo

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, j_fb
      do i = 0, nx + 1
        wspd = sqrt(uwnd(i, j) * uwnd(i, j) + vwnd(i, j) * vwnd(i, j))
        if (wspd > Wspd_min) then
          cff = cfb_slope * wspd + cfb_offset
        else
          cff = stau_ref
        endif
        svstr_r(i, j) = svstr_r(i, j) + cff * 0.5_rp * (v(i, j, nz, n_rhs) + v(i, j + 1, nz, n_rhs)) / r0
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        svstr(i, j) = apply_mask(0.5_rp * (svstr_r(i, j - 1) + svstr_r(i, j)), vmask(i, j))
      enddo
    enddo
    if (do_sss) then
      ! roms_bugfix bulk_frc.F90 SFLX_CORR: stflx(isalt) = -dSSSdt (S(nz,nrhs) - sss) rmask
      cff = dsss
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = js, je
        do i = is, ie
          stflx(i, j, isalt) = apply_mask(-cff * (t(i, j, nz, n_rhs, isalt) - sss(i, j)), rmask(i, j))
        enddo
      enddo
    else
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = js, je
        do i = is, ie
          stflx(i, j, isalt) = 0.0_rp
        enddo
      enddo
    endif
  end subroutine calc_bulk_coare

  subroutine calc_cflx_corr(tp, ntr, nrs, psf, sdic, salk, dc, idic, idica, ialk, ialka)
    ! roms_bugfix bulk_frc.F90 CFLX_CORR (MARBL): the surface fluxes of the
    ! DIC and ALK pairs, from slot nrs of the passive tracers, into the
    ! passive corrector's surface-flux array psf(itr, wet column):
    !   DIC, DIC_ALT_CO2: -dCdt (DIC_ALT_CO2(nz) - sDIC) rmask
    !   ALK, ALK_ALT_CO2: -dCdt (ALK_ALT_CO2(nz) - sALK) rmask
    ! (both tracers of a pair use the _ALT_CO2 tracer's value, as the CPU).
    ! One thread per interior wet column; the other tracers' entries are
    ! not touched.
    integer(ip), intent(in) :: ntr, nrs, idic, idica, ialk, ialka
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), WORK_ATTR intent(inout) :: psf(:,:)
    real(rp), WORK_ATTR intent(in) :: sdic(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: salk(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: dc
    integer(ip) :: c, i, j, nz, nr, a1, a2, b1, b2
    real(rp) :: d, g
    if (nwet <= 0) return
    nz = decomp%nz; nr = nrs; d = dc
    a1 = idic; a2 = idica; b1 = ialk; b2 = ialka
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*>>>
#endif
    do c = 1, nwet
      i = wet_i(c); j = wet_j(c)
      g = -(d * (real(tp(a2, c, nz, nr), rp) - sdic(i, j)) * rmask(i, j))
      psf(a2, c) = g
      psf(a1, c) = g
      g = -(d * (real(tp(b2, c, nz, nr), rp) - salk(i, j)) * rmask(i, j))
      psf(b2, c) = g
      psf(b1, c) = g
    enddo
  end subroutine calc_cflx_corr

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  real(rp) function bulk_psiu(ZoL)
    real(rp), intent(in) :: ZoL
    real(rp) :: Fw, cff, psic, psik, x, y
    if (ZoL < 0.0_rp) then
      x = (1.0_rp - 15.0_rp * ZoL)**0.25_rp
      psik = 2.0_rp * log(0.5_rp * (1.0_rp + x)) + log(0.5_rp * (1.0_rp + x * x)) - &
             2.0_rp * atan(x) + 0.5_rp * pi
      cff = sqrt(3.0_rp)
      y = (1.0_rp - 10.15_rp * ZoL)**r3
      psic = 1.5_rp * log(r3 * (1.0_rp + y + y * y)) - cff * atan((1.0_rp + 2.0_rp * y) / cff) + pi / cff
      cff = ZoL * ZoL
      Fw = cff / (1.0_rp + cff)
      bulk_psiu = (1.0_rp - Fw) * psik + Fw * psic
    else
      cff = min(50.0_rp, 0.35_rp * ZoL)
      bulk_psiu = -((1.0_rp + ZoL) + 0.6667_rp * (ZoL - 14.28_rp) / exp(cff) + 8.525_rp)
    endif
  end function bulk_psiu

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  real(rp) function bulk_psit(ZoL)
    real(rp), intent(in) :: ZoL
    real(rp) :: Fw, cff, psic, psik, x, y
    if (ZoL < 0.0_rp) then
      x = (1.0_rp - 15.0_rp * ZoL)**0.5_rp
      psik = 2.0_rp * log(0.5_rp * (1.0_rp + x))
      cff = sqrt(3.0_rp)
      y = (1.0_rp - 34.15_rp * ZoL)**r3
      psic = 1.5_rp * log(r3 * (1.0_rp + y + y * y)) - cff * atan((1.0_rp + 2.0_rp * y) / cff) + pi / cff
      cff = ZoL * ZoL
      Fw = cff / (1.0_rp + cff)
      bulk_psit = (1.0_rp - Fw) * psik + Fw * psic
    else
      cff = min(50.0_rp, 0.35_rp * ZoL)
      bulk_psit = -((1.0_rp + 2.0_rp * ZoL)**1.5_rp + &
                   0.6667_rp * (ZoL - 14.28_rp) / exp(cff) + 8.525_rp)
    endif
  end function bulk_psit

end module bulk_coare

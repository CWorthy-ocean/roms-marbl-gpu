module formulas
  ! Numerics as host+device functions. CPU oracle and GPU kernels call
  ! the same math. No compute_*.h includes.
  use kinds
  implicit none
  private

  real(rp), parameter, public :: g = 9.81_rp
  ! Boussinesq reference density [kg/m^3]: &rho0_settings rho0, set at run
  ! time by case_io (set_rho0), as UCLA ROMS reads it. The EOS and pressure
  ! functions below use it in device and host code, so it is managed memory.
  ! Host code outside these functions uses gpu_state's rho0_case (the same
  ! value), so that the host does not pull the page back from the GPU every
  ! step. No initializer (nvfortran 25.9's fort2 has crashed on real
  ! initializers in this code base).
#ifdef USE_CUDA
  real(rp), managed, public :: rho0
#else
  real(rp), public :: rho0
#endif
  real(rp), parameter, public :: AM3_crv = 1.0_rp / 6.0_rp

  ! PAC/MiniPac EXTRAP_BAR_FLUXES coefficients (set_HUV1 / step2d).
  real(rp), parameter, public :: EXTRAP_NOW = 3.63_rp
  real(rp), parameter, public :: EXTRAP_MID = 4.47_rp
  real(rp), parameter, public :: EXTRAP_BAK = 2.05_rp
  ! DU_avg_bak = DU_avg1 - this * DU_avg2  (0.21/2.05, PAC23).
  real(rp), parameter, public :: EXTRAP_AVG_BAK_FAC = 0.1024390243902439_rp

  ! omega We/Wi split (omega_mod.F90)
  real(rp), parameter, public :: cu_min = 0.6_rp
  real(rp), parameter, public :: cu_max = 1.0_rp

  ! Shchepetkin density Jacobian (prsgrd_mod.F90)
  real(rp), parameter, public :: OneFifth = 0.2_rp
  real(rp), parameter, public :: OneTwelfth = 1.0_rp / 12.0_rp
  real(rp), parameter, public :: qp2 = 0.0000172_rp
  ! The CPU's literals (compute_horiz_tracer_fluxes.h:112,197; pre_step3d_mod.F90:105 and
  ! step3d_uv_mod.F90:77), not 1/6, so that both codes use the same constants.
  real(rp), parameter, public :: coef_up3 = 0.1666666666666666_rp
  real(rp), parameter, public :: uv_delta = 0.1666666666666667_rp
  real(rp), parameter, public :: uv_gamma = 1.0_rp / 3.0_rp
  ! step3d_uv2 mass-flux filter (not TR).
  real(rp), parameter, public :: FLUX_EPSIL = 0.36_rp
  real(rp), parameter, public :: FLUX_DELTA = 0.28_rp

  real(rp), parameter, public :: pi = 3.14159265358979323846_rp
  real(rp), parameter, public :: vonkar = 0.41_rp

  ! LMD94 interior (lmd_vmix_mod.F90). DDMIX / CONVEC stay off.
  real(rp), parameter, public :: lmd_Ri0 = 0.7_rp
  real(rp), parameter, public :: lmd_nu0m = 1.0e-2_rp
  real(rp), parameter, public :: lmd_nu0s = 1.0e-2_rp
  real(rp), parameter, public :: lmd_nuwm = 1.0e-4_rp
  real(rp), parameter, public :: lmd_nuws = 0.1e-4_rp
  real(rp), parameter, public :: lmd_Lturb = 10.0_rp
  real(rp), parameter, public :: lmd_shear_floor = 1.0e-10_rp

  ! KPP / BKPP (lmd_kpp_mod.F90). INT_AT_RHO_POINTS, SMOOTH_HBL.
  real(rp), parameter, public :: lmd_Ricr = 0.15_rp
  real(rp), parameter, public :: lmd_Ri_inv = 1.0_rp / 0.15_rp
  real(rp), parameter, public :: lmd_epssfc = 0.1_rp
  real(rp), parameter, public :: lmd_betaT = -0.2_rp
  real(rp), parameter, public :: lmd_Cv = 1.8_rp
  real(rp), parameter, public :: lmd_C_Ek = 258.0_rp
  real(rp), parameter, public :: lmd_Cstar = 10.0_rp
  real(rp), parameter, public :: lmd_zeta_m = -0.2_rp
  real(rp), parameter, public :: lmd_a_m = 1.257_rp
  real(rp), parameter, public :: lmd_c_m = 8.360_rp
  real(rp), parameter, public :: lmd_zeta_s = -1.0_rp
  real(rp), parameter, public :: lmd_a_s = -28.86_rp
  real(rp), parameter, public :: lmd_c_s = 98.96_rp
  real(rp), parameter, public :: lmd_r2 = 0.5_rp
  real(rp), parameter, public :: lmd_r3 = 1.0_rp / 3.0_rp
  real(rp), parameter, public :: lmd_r4 = 0.25_rp
  real(rp), parameter, public :: lmd_eps = 1.0e-20_rp
  real(rp), parameter, public :: lmd_Vtsq_fix = 1.8_rp
  real(rp), parameter, public :: lmd_smooth_c = 1.0_rp / 12.0_rp
  real(rp), parameter, public :: lmd_smooth_c1 = 3.0_rp / 16.0_rp
  ! The CPU's literal (lmd_kpp_mod.F90:509), not 1/3.
  real(rp), parameter, public :: lmd_ustar_w = 0.333333333333_rp

  ! Paulson-Simpson Jerlov type 1 (Jwt=1).
  real(rp), parameter, public :: jwt1_mu1 = 0.35_rp
  real(rp), parameter, public :: jwt1_mu2 = 23.0_rp
  real(rp), parameter, public :: jwt1_r1 = 0.58_rp

  ! Jackett-McDougall 1995, DUKO_2001 reference T,S for K0_Duk.
  real(rp), parameter, public :: eos_r00 = 999.842594_rp
  real(rp), parameter, public :: eos_r01 = 6.793952e-2_rp
  real(rp), parameter, public :: eos_r02 = -9.095290e-3_rp
  real(rp), parameter, public :: eos_r03 = 1.001685e-4_rp
  real(rp), parameter, public :: eos_r04 = -1.120083e-6_rp
  real(rp), parameter, public :: eos_r05 = 6.536332e-9_rp
  real(rp), parameter, public :: eos_r10 = 0.824493_rp
  real(rp), parameter, public :: eos_r11 = -4.08990e-3_rp
  real(rp), parameter, public :: eos_r12 = 7.64380e-5_rp
  real(rp), parameter, public :: eos_r13 = -8.24670e-7_rp
  real(rp), parameter, public :: eos_r14 = 5.38750e-9_rp
  real(rp), parameter, public :: eos_rS0 = -5.72466e-3_rp
  real(rp), parameter, public :: eos_rS1 = 1.02270e-4_rp
  real(rp), parameter, public :: eos_rS2 = -1.65460e-6_rp
  real(rp), parameter, public :: eos_r20 = 4.8314e-4_rp
  real(rp), parameter, public :: eos_K00 = 19092.56_rp
  real(rp), parameter, public :: eos_K01 = 209.8925_rp
  real(rp), parameter, public :: eos_K02 = -3.041638_rp
  real(rp), parameter, public :: eos_K03 = -1.852732e-3_rp
  real(rp), parameter, public :: eos_K04 = -1.361629e-5_rp
  real(rp), parameter, public :: eos_K10 = 104.4077_rp
  real(rp), parameter, public :: eos_K11 = -6.500517_rp
  real(rp), parameter, public :: eos_K12 = 0.1553190_rp
  real(rp), parameter, public :: eos_K13 = 2.326469e-4_rp
  real(rp), parameter, public :: eos_KS0 = -5.587545_rp
  real(rp), parameter, public :: eos_KS1 = 0.7390729_rp
  real(rp), parameter, public :: eos_KS2 = -1.909078e-2_rp
  ! DUKO_2001: the reference temperature and salinity of K0_Duk (jackett_qp1; also IMEX_IGW's
  ! boundary density, imex_obm.inc obm_rho_d)
  real(rp), parameter, public :: eos_Ttr = 3.8_rp
  real(rp), parameter, public :: eos_Tsr = 34.5_rp

  public :: csf
  public :: sm09_z
  public :: thickness_aware_faces
  public :: thickness_tplus
  public :: thickness_tminus
  public :: fb_ab3_am4_weights
  public :: fb_time_indices
  public :: density_weighted_zeta
  public :: barotropic_pgf
  public :: flux_mismatch
  public :: apply_mask
  public :: harmonic_avg
  public :: split_eos_insitu
  public :: adiabatic_diff
  public :: dens_jacobian
  public :: surface_P
  public :: up3_flux
  public :: jackett_rho1
  public :: jackett_qp1
  public :: jackett_split_eos
  public :: lf_am3_weights
  public :: upstream_uv_face
  public :: centered_uv_face
  public :: jackett_alpha_beta
  public :: jackett_alpha
  public :: jackett_beta
  public :: lmd_gradient_ri
  public :: lmd_nu_sx
  public :: lmd_bottom_mult
  public :: lmd_wscale_ws
  public :: lmd_wscale_wm_ws
  public :: lmd_ws
  public :: lmd_wm
  public :: lmd_ws_pair
  public :: lmd_Cg
  public :: lmd_Vtc
  public :: swr_band_atten

contains

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function csf(sc, theta_s, theta_b) result(cs)
    ! SM2009 stretching, set_scoord_mod.F90 CSF.
    real(rp), intent(in) :: sc, theta_s, theta_b
    real(rp) :: cs, csrf

    if (theta_s > 0.0_rp) then
      csrf = (1.0_rp - cosh(theta_s * sc)) / (cosh(theta_s) - 1.0_rp)
    else
      csrf = -sc * sc
    endif
    if (theta_b > 0.0_rp) then
      cs = (exp(theta_b * csrf) - 1.0_rp) / (1.0_rp - exp(-theta_b))
    else
      cs = csrf
    endif
  end function csf

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function sm09_z(zeta, h, hc, s, Cs) result(z)
    ! SM09: z = zeta + (zeta+h) * (hc*s + h*Cs) / (h+hc)
    real(rp), intent(in) :: zeta, h, hc, s, Cs
    real(rp) :: z
    z = zeta + (zeta + h) * (hc * s + h * Cs) / (h + hc)
  end function sm09_z

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine thickness_aware_faces(t_km1, t_k, t_kp1, t_kp2, &
                                   Hz_km1, Hz_k, Hz_kp1, Hz_kp2, &
                                   at_bottom, at_surface, tplus, tminus)
    ! Faces between layers k and k+1. Bottom: Hz(0)=Hz(1), dlt(0)=0.
    ! Surface: Hz(nz+1)=Hz(nz), dlt(nz)=0 (zero gradient, as roms_bugfix).
    real(rp), intent(in)  :: t_km1, t_k, t_kp1, t_kp2
    real(rp), intent(in)  :: Hz_km1, Hz_k, Hz_kp1, Hz_kp2
    logical,  intent(in)  :: at_bottom, at_surface
    real(rp), intent(out) :: tplus, tminus
    real(rp) :: dlt_m, dlt, dlt_p, tlin, hkm, hkp2

    dlt = (t_kp1 - t_k) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
    if (at_bottom) then
      dlt_m = 0.0_rp
      hkm   = Hz_k
    else
      dlt_m = (t_k - t_km1) / max(Hz_km1 + Hz_k, 1.0e-30_rp)
      hkm   = Hz_km1
    endif
    if (at_surface) then
      hkp2 = Hz_kp1
      dlt_p = 0.0_rp
    else
      hkp2 = Hz_kp2
      dlt_p = (t_kp2 - t_kp1) / max(Hz_kp1 + Hz_kp2, 1.0e-30_rp)
    endif

    tlin = (Hz_kp1 * t_k + Hz_k * t_kp1) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
    tplus  = tlin - Hz_k * Hz_kp1 * (dlt - dlt_m) / max(hkm + Hz_k + Hz_kp1, 1.0e-30_rp)
    tminus = tlin - Hz_k * Hz_kp1 * (dlt_p - dlt) / max(Hz_k + Hz_kp1 + hkp2, 1.0e-30_rp)
  end subroutine thickness_aware_faces

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function thickness_tplus(t_km1, t_k, t_kp1, Hz_km1, Hz_k, Hz_kp1, at_bottom) result(tplus)
    real(rp), intent(in) :: t_km1, t_k, t_kp1, Hz_km1, Hz_k, Hz_kp1
    logical,  intent(in) :: at_bottom
    real(rp) :: tplus, dlt, dlt_m, tlin, hkm
    dlt = (t_kp1 - t_k) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
    if (at_bottom) then
      dlt_m = 0.0_rp
      hkm = Hz_k
    else
      dlt_m = (t_k - t_km1) / max(Hz_km1 + Hz_k, 1.0e-30_rp)
      hkm = Hz_km1
    endif
    tlin = (Hz_kp1 * t_k + Hz_k * t_kp1) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
    tplus = tlin - Hz_k * Hz_kp1 * (dlt - dlt_m) / max(hkm + Hz_k + Hz_kp1, 1.0e-30_rp)
  end function thickness_tplus

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function thickness_tminus(t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2, &
                            at_bottom, at_surface) result(tminus)
    real(rp), intent(in) :: t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2
    logical,  intent(in) :: at_bottom, at_surface
    real(rp) :: tminus, dlt, dlt_m, dlt_p, tlin, hkp2
    dlt = (t_kp1 - t_k) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
    if (at_bottom) then
      dlt_m = 0.0_rp
    else
      dlt_m = (t_k - t_km1) / max(Hz_km1 + Hz_k, 1.0e-30_rp)
    endif
    tlin = (Hz_kp1 * t_k + Hz_k * t_kp1) / max(Hz_k + Hz_kp1, 1.0e-30_rp)
    if (at_surface) then
      hkp2 = Hz_kp1
      dlt_p = 0.0_rp
    else
      hkp2 = Hz_kp2
      dlt_p = (t_kp2 - t_kp1) / max(Hz_kp1 + Hz_kp2, 1.0e-30_rp)
    endif
    tminus = tlin - Hz_k * Hz_kp1 * (dlt_p - dlt) / max(Hz_k + Hz_kp1 + hkp2, 1.0e-30_rp)
  end function thickness_tminus

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine fb_ab3_am4_weights(iif, fwd, fwd1, fwd2, bkw_new, bkw, bkw1, bkw2)
    ! Generalized FB AB3-AM4 coefficients from step2d_mod.F90.
    integer(ip), intent(in)  :: iif
    real(rp),    intent(out) :: fwd, fwd1, fwd2, bkw_new, bkw, bkw1, bkw2

    if (iif == 1) then
      fwd = 1.0_rp; fwd1 = 0.0_rp; fwd2 = 0.0_rp
      bkw_new = 0.0_rp; bkw = 1.0_rp; bkw1 = 0.0_rp; bkw2 = 0.0_rp
    elseif (iif == 2) then
      fwd = 1.0_rp; fwd1 = 0.0_rp; fwd2 = 0.0_rp
      bkw_new = 1.0833333333333_rp
      bkw     = -0.1666666666666_rp
      bkw1    =  0.0833333333333_rp
      bkw2    =  0.0_rp
    else
      fwd = 1.781105_rp; fwd1 = -1.06221_rp; fwd2 = 0.281105_rp
      bkw_new = 0.614_rp; bkw = 0.285_rp; bkw1 = 0.088_rp; bkw2 = 0.013_rp
    endif
  end subroutine fb_ab3_am4_weights

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine fb_time_indices(iif, kstp, kbak, kold)
    integer(ip), intent(in)  :: iif, kstp
    integer(ip), intent(out) :: kbak, kold

    if (iif == 1) then
      kbak = kstp
      kold = kstp
    elseif (iif == 2) then
      kbak = kstp - 1
      if (kbak < 1) kbak = 4
      kold = kbak
    else
      kbak = kstp - 1
      if (kbak < 1) kbak = 4
      kold = kbak - 1
      if (kold < 1) kold = 4
    endif
  end subroutine fb_time_indices

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine density_weighted_zeta(zwrk, rhoS, rhoA, rzeta, rzetaSA)
    ! VAR_RHO_2D weights. rhoA = rhoS = 0 recovers SWE.
    real(rp), intent(in)  :: zwrk, rhoS, rhoA
    real(rp), intent(out) :: rzeta, rzetaSA
    rzeta   = (1.0_rp + rhoS) * zwrk
    rzetaSA = zwrk * (rhoS - rhoA)
  end subroutine density_weighted_zeta

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function barotropic_pgf(h_m, h_p, rzeta_m, rzeta_p, rzeta2_m, rzeta2_p, &
                          rzetaSA_m, rzetaSA_p, rhoA_m, rhoA_p, zwrk_m, zwrk_p, &
                          metric) result(pgf)
    ! Shallow-water PGF with optional VAR_RHO_2D terms (step2d_mod.F90).
    ! Subscript _m is the up-gradient neighbor (i-1 or j-1), _p is (i,j).
    real(rp), intent(in) :: h_m, h_p, rzeta_m, rzeta_p, rzeta2_m, rzeta2_p
    real(rp), intent(in) :: rzetaSA_m, rzetaSA_p, rhoA_m, rhoA_p
    real(rp), intent(in) :: zwrk_m, zwrk_p, metric
    real(rp) :: pgf
    pgf = 0.5_rp * g * metric * ( &
          (h_m + h_p) * (rzeta_m - rzeta_p) + rzeta2_m - rzeta2_p + &
          (h_m - h_p) * (rzetaSA_m + rzetaSA_p + &
          0.333333333333_rp * (rhoA_m - rhoA_p) * (zwrk_m - zwrk_p)))  ! the CPU's literal, step2d_mod.F90:263
  end function barotropic_pgf

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function flux_mismatch(int_flux, D, DU_target) result(du)
    ! (vertical integral of Hz*u*dn - DU_target) / D, in m/s.
    real(rp), intent(in) :: int_flux, D, DU_target
    real(rp) :: du
    du = (int_flux - DU_target) / max(D, 1.0e-30_rp)
  end function flux_mismatch

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function apply_mask(val, mask) result(out)
    ! Always evaluate the stencil; mask the store. Do not use val*mask:
    ! Inf/NaN on land would survive (Inf*0 is NaN) and leak into wet points.
    real(rp), intent(in) :: val, mask
    real(rp) :: out
    out = 0.0_rp
    if (mask > 0.5_rp) out = val
  end function apply_mask

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function harmonic_avg(a, b) result(h)
    ! Harmonic mean of elementary differences; 0 if they differ in sign.
    real(rp), intent(in) :: a, b
    real(rp) :: h, cff
    cff = 2.0_rp * a * b
    if (cff > 0.0_rp) then
      h = cff / (a + b)
    else
      h = 0.0_rp
    endif
  end function harmonic_avg

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function split_eos_insitu(rho1, qp1, zr) result(r)
    ! rho = rho1 + qp1 * dpth * (1 - qp2 * dpth), dpth = -z_r
    real(rp), intent(in) :: rho1, qp1, zr
    real(rp) :: r, dpth
    dpth = -zr
    r = rho1 + qp1 * dpth * (1.0_rp - qp2 * dpth)
  end function split_eos_insitu

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function adiabatic_diff(rho1_p, rho1_m, qp1_p, qp1_m, zr_p, zr_m) result(dR)
    ! Match UCLA rho_eos SPLIT_EOS association exactly (no FMA contraction):
    !   (rho1(k+1)-rho1(k)) + (qp1(k+1)-qp1(k))*dpth*(1-qp2*dpth)
    ! with dpth = -0.5*(z_r(k+1)+z_r(k)). Tiny adr residuals (~1e-15) amplify
    ! into Riraw when sh2 is small and seed the Akv→Thomas→FuE→We loop.
    real(rp), intent(in) :: rho1_p, rho1_m, qp1_p, qp1_m, zr_p, zr_m
    real(rp) :: dR, dpth, drho, dqp, cff
    dpth = -0.5_rp * (zr_p + zr_m)
    drho = rho1_p - rho1_m
    dqp  = qp1_p - qp1_m
    cff  = 1.0_rp - qp2 * dpth
    dR   = drho + (dqp * dpth) * cff
  end function adiabatic_diff

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function dens_jacobian(rho_p, rho_m, z_p, z_m, dR_p, dR_m, dZ_p, dZ_m) result(jac)
    ! Cubic density Jacobian; OneFifth=0 recovers the POM Jacobian.
    real(rp), intent(in) :: rho_p, rho_m, z_p, z_m, dR_p, dR_m, dZ_p, dZ_m
    real(rp) :: jac
    jac = (rho_p + rho_m) * (z_p - z_m) - OneFifth * ( &
          (dR_p - dR_m) * (z_p - z_m - OneTwelfth * (dZ_p + dZ_m)) - &
          (dZ_p - dZ_m) * (rho_p - rho_m - OneTwelfth * (dR_p + dR_m)))
  end function dens_jacobian

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function surface_P(zw_n, zr_n, zr_nm1, rho_n, rho_nm1) result(Ptop)
    real(rp), intent(in) :: zw_n, zr_n, zr_nm1, rho_n, rho_nm1
    real(rp) :: Ptop, grho
    grho = g / rho0
    Ptop = g * zw_n + grho * (rho_n + 0.5_rp * (rho_n - rho_nm1) * &
           (zw_n - zr_n) / max(zr_n - zr_nm1, 1.0e-30_rp)) * (zw_n - zr_n)
  end function surface_P

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function up3_flux(t_l, t_r, curv_l, curv_r, uflux) result(fx)
    ! UPSTREAM_TS face flux. t_l is the left/south cell, t_r the right/north.
    real(rp), intent(in) :: t_l, t_r, curv_l, curv_r, uflux
    real(rp) :: fx
    fx = 0.5_rp * (t_l + t_r) * uflux - coef_up3 * ( &
         curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp))
  end function up3_flux

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function jackett_rho1(Tt, Ts) result(rho1_out)
    ! SPLIT_EOS + DUKO_2001 1-Atm density anomaly. A CUF kernel cannot
    ! call a subroutine with intent(out); use functions instead.
    real(rp), intent(in) :: Tt, Ts
    real(rp) :: rho1_out, sqrtTs, dr00
    dr00 = eos_r00 - rho0
    sqrtTs = sqrt(max(0.0_rp, Ts))
    rho1_out = dr00 + Tt * (eos_r01 + Tt * (eos_r02 + Tt * (eos_r03 + Tt * ( &
               eos_r04 + Tt * eos_r05)))) &
             + Ts * (eos_r10 + Tt * (eos_r11 + Tt * (eos_r12 + Tt * ( &
               eos_r13 + Tt * eos_r14))) &
             + sqrtTs * (eos_rS0 + Tt * (eos_rS1 + Tt * eos_rS2)) &
             + Ts * eos_r20)
  end function jackett_rho1

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function jackett_qp1(Tt, Ts, rho1_in) result(qp1_out)
    real(rp), intent(in) :: Tt, Ts, rho1_in
    real(rp) :: qp1_out, sqrtTs, K0, K0_Duk, Ttr, Tsr
    Ttr = eos_Ttr
    Tsr = eos_Tsr
    K0_Duk = Ttr * (eos_K01 + Ttr * (eos_K02 + Ttr * (eos_K03 + Ttr * eos_K04))) &
           + Tsr * (eos_K10 + Ttr * (eos_K11 + Ttr * (eos_K12 + Ttr * eos_K13)) &
           + sqrt(Tsr) * (eos_KS0 + Ttr * (eos_KS1 + Ttr * eos_KS2)))
    sqrtTs = sqrt(max(0.0_rp, Ts))
    K0 = Tt * (eos_K01 + Tt * (eos_K02 + Tt * (eos_K03 + Tt * eos_K04))) &
       + Ts * (eos_K10 + Tt * (eos_K11 + Tt * (eos_K12 + Tt * eos_K13)) &
       + sqrtTs * (eos_KS0 + Tt * (eos_KS1 + Tt * eos_KS2)))
    qp1_out = 0.1_rp * (rho0 + rho1_in) * (K0_Duk - K0) / &
              max((eos_K00 + K0) * (eos_K00 + K0_Duk), 1.0e-30_rp)
  end function jackett_qp1

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine jackett_split_eos(Tt, Ts, rho1_out, qp1_out)
    real(rp), intent(in)  :: Tt, Ts
    real(rp), intent(out) :: rho1_out, qp1_out
    rho1_out = jackett_rho1(Tt, Ts)
    qp1_out = jackett_qp1(Tt, Ts, rho1_out)
  end subroutine jackett_split_eos

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine lf_am3_weights(first, dt_in, dtau, cf_stp, cf_bak)
    logical,  intent(in)  :: first
    real(rp), intent(in)  :: dt_in
    real(rp), intent(out) :: dtau, cf_stp, cf_bak
    if (first) then
      dtau   = 0.5_rp * dt_in
      cf_stp = 1.0_rp
      cf_bak = 0.0_rp
    else
      dtau   = dt_in * (1.0_rp - AM3_crv)
      cf_stp = 0.5_rp + AM3_crv
      cf_bak = 0.5_rp - AM3_crv
    endif
  end subroutine lf_am3_weights

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function upstream_uv_face(u_l, u_r, uxx_l, uxx_r, Huxx_l, Huxx_r, flux_l, flux_r) result(ff)
    real(rp), intent(in) :: u_l, u_r, uxx_l, uxx_r, Huxx_l, Huxx_r, flux_l, flux_r
    real(rp) :: ff, cff
    cff = flux_l + flux_r - uv_delta * (Huxx_l + Huxx_r)
    ff = 0.25_rp * (cff * (u_l + u_r) - uv_gamma * ( &
         max(cff, 0.0_rp) * uxx_l + min(cff, 0.0_rp) * uxx_r))
  end function upstream_uv_face

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function centered_uv_face(u_l, u_r, uxx_l, uxx_r, Huxx_l, Huxx_r, flux_l, flux_r) result(ff)
    ! Non-UPSTREAM_UV branch of compute_horiz_rhs_uv_terms.h (pre_step3d).
    real(rp), intent(in) :: u_l, u_r, uxx_l, uxx_r, Huxx_l, Huxx_r, flux_l, flux_r
    real(rp) :: ff
    ff = 0.25_rp * (u_l + u_r - uv_delta * (uxx_l + uxx_r)) * &
         (flux_l + flux_r - uv_delta * (Huxx_l + Huxx_r))
  end function centered_uv_face

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine jackett_alpha_beta(Tt, Ts, alpha, beta)
    ! Surface Jackett-McDougall 1992 alpha, beta / rho0 (alphabeta_mod.F90).
    real(rp), intent(in)  :: Tt, Ts
    real(rp), intent(out) :: alpha, beta
    real(rp) :: cff, sqrtTs

    cff = 1.0_rp / rho0
    sqrtTs = sqrt(max(0.0_rp, Ts))
    alpha = -cff * (eos_r01 + Tt * (2.0_rp * eos_r02 + Tt * (3.0_rp * eos_r03 + Tt * ( &
            4.0_rp * eos_r04 + Tt * 5.0_rp * eos_r05))) &
          + Ts * (eos_r11 + Tt * (2.0_rp * eos_r12 + Tt * (3.0_rp * eos_r13 + Tt * 4.0_rp * eos_r14)) &
          + sqrtTs * (eos_rS1 + Tt * 2.0_rp * eos_rS2)))
    beta = cff * (eos_r10 + Tt * (eos_r11 + Tt * (eos_r12 + Tt * (eos_r13 + Tt * eos_r14))) &
         + 1.5_rp * (eos_rS0 + Tt * (eos_rS1 + Tt * eos_rS2)) * sqrtTs &
         + 2.0_rp * eos_r20 * Ts)
  end subroutine jackett_alpha_beta

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function jackett_alpha(Tt, Ts) result(alpha)
    real(rp), intent(in) :: Tt, Ts
    real(rp) :: alpha, cff, sqrtTs
    cff = 1.0_rp / rho0
    sqrtTs = sqrt(max(0.0_rp, Ts))
    alpha = -cff * (eos_r01 + Tt * (2.0_rp * eos_r02 + Tt * (3.0_rp * eos_r03 + Tt * ( &
            4.0_rp * eos_r04 + Tt * 5.0_rp * eos_r05))) &
          + Ts * (eos_r11 + Tt * (2.0_rp * eos_r12 + Tt * (3.0_rp * eos_r13 + Tt * 4.0_rp * eos_r14)) &
          + sqrtTs * (eos_rS1 + Tt * 2.0_rp * eos_rS2)))
  end function jackett_alpha

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function jackett_beta(Tt, Ts) result(beta)
    real(rp), intent(in) :: Tt, Ts
    real(rp) :: beta, cff, sqrtTs
    cff = 1.0_rp / rho0
    sqrtTs = sqrt(max(0.0_rp, Ts))
    beta = cff * (eos_r10 + Tt * (eos_r11 + Tt * (eos_r12 + Tt * (eos_r13 + Tt * eos_r14))) &
         + 1.5_rp * (eos_rS0 + Tt * (eos_rS1 + Tt * eos_rS2)) * sqrtTs &
         + 2.0_rp * eos_r20 * Ts)
  end function jackett_beta

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_gradient_ri(bvf, dudz, dvdz) result(rig)
    real(rp), intent(in) :: bvf, dudz, dvdz
    real(rp) :: rig
    rig = bvf / (lmd_Ri0 * max(dudz * dudz + dvdz * dvdz, lmd_shear_floor))
  end function lmd_gradient_ri

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_nu_sx(rig) result(nu)
    real(rp), intent(in) :: rig
    real(rp) :: nu, cff
    cff = min(1.0_rp, max(0.0_rp, rig))
    nu = 1.0_rp - cff * cff
    nu = nu * nu * nu
  end function lmd_nu_sx

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_bottom_mult(dist) result(mult)
    real(rp), intent(in) :: dist
    real(rp) :: mult
    if (dist < lmd_Lturb) then
      mult = sin(0.5_rp * pi * dist / lmd_Lturb)
    else
      mult = 1.0_rp
    endif
  end function lmd_bottom_mult

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine lmd_wscale_ws(zscale_in, Bfsfc, ustar, hbl, rmask_ij, ws)
    ! lmd_wscale_ws_only.h. LIMIT_UNSTABLE_ONLY is off: always cap zscale.
    real(rp), intent(in)  :: zscale_in, Bfsfc, ustar, hbl, rmask_ij
    real(rp), intent(out) :: ws
    real(rp) :: zscale, zetahat, ustar3

    zscale = min(zscale_in, hbl * lmd_epssfc) * rmask_ij
    zetahat = vonkar * zscale * Bfsfc
    ustar3 = ustar ** 3
    if (zetahat >= 0.0_rp) then
      ws = vonkar * ustar * ustar3 / max(ustar3 + 5.0_rp * zetahat, 1.0e-20_rp)
    elseif (zetahat > lmd_zeta_s * ustar3) then
      ws = vonkar * ((ustar3 - 16.0_rp * zetahat) / max(ustar, 1.0e-30_rp)) ** lmd_r2
    else
      ws = vonkar * (lmd_a_s * ustar3 - lmd_c_s * zetahat) ** lmd_r3
    endif
  end subroutine lmd_wscale_ws

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_ws(zscale_in, Bfsfc, ustar, hbl, rmask_ij) result(ws)
    real(rp), intent(in) :: zscale_in, Bfsfc, ustar, hbl, rmask_ij
    real(rp) :: ws, zscale, zetahat, ustar3
    zscale = min(zscale_in, hbl * lmd_epssfc) * rmask_ij
    zetahat = vonkar * zscale * Bfsfc
    ustar3 = ustar ** 3
    if (zetahat >= 0.0_rp) then
      ws = vonkar * ustar * ustar3 / max(ustar3 + 5.0_rp * zetahat, 1.0e-20_rp)
    elseif (zetahat > lmd_zeta_s * ustar3) then
      ws = vonkar * ((ustar3 - 16.0_rp * zetahat) / max(ustar, 1.0e-30_rp)) ** lmd_r2
    else
      ws = vonkar * (lmd_a_s * ustar3 - lmd_c_s * zetahat) ** lmd_r3
    endif
  end function lmd_ws

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  subroutine lmd_wscale_wm_ws(zscale_in, Bfsfc, ustar, hbl, rmask_ij, wm, ws)
    ! lmd_wscale_wm_and_ws.h
    real(rp), intent(in)  :: zscale_in, Bfsfc, ustar, hbl, rmask_ij
    real(rp), intent(out) :: wm, ws
    real(rp) :: zscale, zetahat, ustar3

    zscale = min(zscale_in, hbl * lmd_epssfc) * rmask_ij
    zetahat = vonkar * zscale * Bfsfc
    ustar3 = ustar ** 3
    if (zetahat >= 0.0_rp) then
      wm = vonkar * ustar * ustar3 / max(ustar3 + 5.0_rp * zetahat, 1.0e-20_rp)
      ws = wm
    else
      if (zetahat > lmd_zeta_m * ustar3) then
        wm = vonkar * (ustar * (ustar3 - 16.0_rp * zetahat)) ** lmd_r4
      else
        wm = vonkar * (lmd_a_m * ustar3 - lmd_c_m * zetahat) ** lmd_r3
      endif
      if (zetahat > lmd_zeta_s * ustar3) then
        ws = vonkar * ((ustar3 - 16.0_rp * zetahat) / max(ustar, 1.0e-30_rp)) ** lmd_r2
      else
        ws = vonkar * (lmd_a_s * ustar3 - lmd_c_s * zetahat) ** lmd_r3
      endif
    endif
  end subroutine lmd_wscale_wm_ws

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_wm(zscale_in, Bfsfc, ustar, hbl, rmask_ij) result(wm)
    real(rp), intent(in) :: zscale_in, Bfsfc, ustar, hbl, rmask_ij
    real(rp) :: wm, zscale, zetahat, ustar3
    zscale = min(zscale_in, hbl * lmd_epssfc) * rmask_ij
    zetahat = vonkar * zscale * Bfsfc
    ustar3 = ustar ** 3
    if (zetahat >= 0.0_rp) then
      wm = vonkar * ustar * ustar3 / max(ustar3 + 5.0_rp * zetahat, 1.0e-20_rp)
    elseif (zetahat > lmd_zeta_m * ustar3) then
      wm = vonkar * (ustar * (ustar3 - 16.0_rp * zetahat)) ** lmd_r4
    else
      wm = vonkar * (lmd_a_m * ustar3 - lmd_c_m * zetahat) ** lmd_r3
    endif
  end function lmd_wm

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_ws_pair(zscale_in, Bfsfc, ustar, hbl, rmask_ij) result(ws)
    real(rp), intent(in) :: zscale_in, Bfsfc, ustar, hbl, rmask_ij
    real(rp) :: ws
    ws = lmd_ws(zscale_in, Bfsfc, ustar, hbl, rmask_ij)
  end function lmd_ws_pair

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_Cg() result(cg)
    real(rp) :: cg
    cg = lmd_Cstar * vonkar * (lmd_c_s * vonkar * lmd_epssfc) ** lmd_r3
  end function lmd_Cg

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function lmd_Vtc() result(vtc)
    real(rp) :: vtc
    vtc = lmd_Cv * sqrt(-lmd_betaT / (lmd_c_s * lmd_epssfc)) / (lmd_Ricr * vonkar ** 2)
  end function lmd_Vtc

#ifdef USE_CUDA
  attributes(host, device) &
#endif
  function swr_band_atten(swdk, attn, hz_k) result(out)
    real(rp), intent(in) :: swdk, attn, hz_k
    real(rp) :: out, xi
    xi = attn * hz_k
    if (xi > -20.0_rp) then
      out = swdk * exp(xi)
    else
      out = 0.0_rp
    endif
  end function swr_band_atten

end module formulas

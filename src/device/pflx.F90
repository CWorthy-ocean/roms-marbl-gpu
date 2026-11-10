#include "mem_attr.inc"
module pflx_mod
  ! Depth-integrated baroclinic (internal-tide) energy flux for nesting:
  ! UCLA ROMS calc_pflx_mod (DIAGNOSTICS + &calc_pflx_settings calc_pflx),
  ! with the internal-tide pressure of ucla-roms eae883ee (Kelly, Nash and
  ! Kunze 2010). Divided by rho0, in m^4/s^3 (multiply by rho0 for W/m):
  !   up = int u' p' dz at u-points, vp = int v' p' dz at v-points,
  ! computed for the interior faces i = 1..nx, j = 1..ny of each tile, as
  ! the CPU, once per step after the corrector prsgrd.
  !
  ! u', v': velocity on the s-levels at n+1/2 (slot 3) minus its depth
  ! mean (the column transport sum FlxU over the depth of the face) minus
  ! its running mean u_slow / v_slow.
  ! p': hydrostatic pressure of the density on the s-levels, integrated
  ! over the levels' rest positions z0 = (z_r - zeta) h/(h + zeta) with the
  ! in-situ density at z0 (SPLIT_EOS: rho1 - qp1 z0 (1 + qp2 z0)),
  !   p(k) = g/rho0 int_{z0(k)}^0 rho dz0,
  ! minus its running mean p_slow, minus its depth mean (removed as
  ! int u'(p' - <p'>) dz = int u'p' dz - <p'> int u' dz).
  ! Running means: x_slow = (1 - alpha) x_slow + alpha x, alpha =
  ! dt/pflx_timescale, masked. Same operations in the same order as the
  ! CPU (top-down level sums, the CPU's operand order), so the two differ
  ! by rounding only.
  !
  ! The running means are restart fields (u_slow, v_slow, p_slow; his_io,
  ! case_io). The CPU computes the fluxes only when it is built with the
  ! DIAGNOSTICS cpp key; the GPU model has no cpp keys and computes them
  ! whenever calc_pflx is .true.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas, only: g, rho0, qp2
  use halo_mod, only: halo_fill_3d
  use time_state, only: dt
  implicit none
  private
  public :: calc_pflx, pflx_timescale, init_pflx, calc_pressure_flux, pflx_fill_slow_halos
  public :: up, vp, u_slow, v_slow, p_slow

  ! &calc_pflx_settings (the CPU's group)
  logical :: calc_pflx = .false.
  real(rp) :: pflx_timescale = 0.0_rp

  real(rp), WORK_ATTR allocatable :: up(:,:), vp(:,:)
  real(rp), WORK_ATTR allocatable :: u_slow(:,:,:), v_slow(:,:,:), p_slow(:,:,:)
  ! work: p' at every level (the up/vp kernel needs it at i-1 / j-1), the
  ! depth mean of p'
  real(rp), WORK_ATTR allocatable :: p_fast(:,:,:), p_int(:,:)
  real(rp) :: alpha = 0.0_rp, beta = 1.0_rp

contains

  subroutine init_pflx()
    ! After alloc_state (tile bounds) and the time-stepping namelist (dt).
    integer(ip) :: is, ie, js, je, nz
    if (.not. calc_pflx) return
    if (pflx_timescale <= 0.0_rp) &
      error stop 'calc_pflx: &calc_pflx_settings pflx_timescale must be > 0 when calc_pflx is on'
    alpha = dt / pflx_timescale
    beta = 1 - alpha
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    allocate(up(is:ie, js:je), vp(is:ie, js:je), p_int(is:ie, js:je))
    allocate(u_slow(is:ie, js:je, nz), v_slow(is:ie, js:je, nz), p_slow(is:ie, js:je, nz))
    allocate(p_fast(is:ie, js:je, nz))
    up = 0.0_rp; vp = 0.0_rp; p_int = 0.0_rp
    u_slow = 0.0_rp; v_slow = 0.0_rp; p_slow = 0.0_rp; p_fast = 0.0_rp
  end subroutine init_pflx

  subroutine pflx_fill_slow_halos()
    ! After the restart read: the CPU exchanges p_slow (get_init_slow), whose
    ! halo column/row i = 0, j = 0 the pressure kernel updates and the up/vp
    ! kernel reads.
    if (.not. calc_pflx) return
    call halo_fill_3d(p_slow, STAGGER_RHO)
  end subroutine pflx_fill_slow_halos

  subroutine calc_pressure_flux()
    ! The CPU's calc_pressure_flux (prsgrd, corrector stage). Geometry and
    ! density are those of the corrector prsgrd: z_r, z_w, Hz of time n,
    ! rho1/qp1 of the n+1/2 tracers (rho_eos stage 2, exchanged), FlxU/FlxV
    ! from set_HUV1, u/v in slot 3.
    integer(ip) :: i, j, k, nx, ny, nz, ir
    real(rp) :: a, b, grho, hgrho, zs, zsrf, z0, rz, rl, pl, ps, pf, pacc, rm
    real(rp) :: s, ub, vb, um, vm, us, uf, vs, vf, hzu, hzv, acu, acv, iu, iv
    if (.not. calc_pflx) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    ir = 3                       ! nrhs of the corrector
    a = alpha; b = beta
    grho = g / rho0
    hgrho = 0.5_rp * grho

    ! p' at rho points 0..nx x 0..ny (the CPU's range): running mean,
    ! deviation at every level, and its depth mean
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny
      do i = 0, nx
        zs = -z_w(i, j, 0) / (z_w(i, j, nz) - z_w(i, j, 0))       ! h/(h+zeta)
        zsrf = z_w(i, j, nz)
        rm = rmask(i, j)
        pl = 0.0_rp; rl = 0.0_rp; pacc = 0.0_rp
        do k = nz, 1, -1           ! top down: p is integrated from the surface
          z0 = (z_r(i, j, k) - zsrf) * zs
          rz = rho1(i, j, k) - (qp1(i, j, k) * z0) * (1.0_rp + qp2 * z0)
          if (k == nz) then
            pl = -((grho * rz) * z0)
          else
            pl = pl + ((hgrho * (rz + rl)) * (z_r(i, j, k + 1) - z_r(i, j, k))) * zs
          endif
          rl = rz
          ps = ((b * p_slow(i, j, k)) + (a * pl)) * rm
          p_slow(i, j, k) = ps
          pf = (pl - ps) * rm
          p_fast(i, j, k) = pf
          pacc = pacc + pf * Hz(i, j, k)
        enddo
        p_int(i, j) = pacc / (z_w(i, j, nz) - z_w(i, j, 0))
      enddo
    enddo

    ! u', v' and the fluxes: u faces 1..nx+1 x 1..ny, v faces 1..nx x
    ! 1..ny+1 (the running means), up/vp on 1..nx x 1..ny
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx + 1
        if (j <= ny) then
          s = 0.0_rp
          do k = 1, nz
            s = s + FlxU(i, j, k)
          enddo
          ! barotropic velocity [m/s]
          ub = (s * (pn(i, j) + pn(i - 1, j))) / &
               (((z_w(i, j, nz) - z_w(i, j, 0)) + z_w(i - 1, j, nz)) - z_w(i - 1, j, 0))
          um = umask(i, j)
          acu = 0.0_rp; iu = 0.0_rp
          do k = nz, 1, -1
            us = ((b * u_slow(i, j, k)) + (a * (u(i, j, k, ir) - ub))) * um
            u_slow(i, j, k) = us
            uf = ((u(i, j, k, ir) - ub) - us) * um
            if (i <= nx) then
              hzu = 0.5_rp * (Hz(i - 1, j, k) + Hz(i, j, k))
              acu = acu + ((uf * 0.5_rp) * (p_fast(i - 1, j, k) + p_fast(i, j, k))) * hzu
              iu = iu + uf * hzu
            endif
          enddo
          if (i <= nx) up(i, j) = acu - (0.5_rp * (p_int(i - 1, j) + p_int(i, j))) * iu
        endif
        if (i <= nx) then
          s = 0.0_rp
          do k = 1, nz
            s = s + FlxV(i, j, k)
          enddo
          vb = (s * (pm(i, j) + pm(i, j - 1))) / &
               (((z_w(i, j, nz) - z_w(i, j, 0)) + z_w(i, j - 1, nz)) - z_w(i, j - 1, 0))
          vm = vmask(i, j)
          acv = 0.0_rp; iv = 0.0_rp
          do k = nz, 1, -1
            vs = ((b * v_slow(i, j, k)) + (a * (v(i, j, k, ir) - vb))) * vm
            v_slow(i, j, k) = vs
            vf = ((v(i, j, k, ir) - vb) - vs) * vm
            if (j <= ny) then
              hzv = 0.5_rp * (Hz(i, j - 1, k) + Hz(i, j, k))
              acv = acv + ((vf * 0.5_rp) * (p_fast(i, j - 1, k) + p_fast(i, j, k))) * hzv
              iv = iv + vf * hzv
            endif
          enddo
          if (j <= ny) vp(i, j) = acv - (0.5_rp * (p_int(i, j - 1) + p_int(i, j))) * iv
        endif
      enddo
    enddo
  end subroutine calc_pressure_flux

end module pflx_mod

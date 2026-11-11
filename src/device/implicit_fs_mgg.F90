#include "mem_attr.inc"
#include "cuf_launch.inc"
#ifdef USE_CUDA
#define FSD device,
#else
#define FSD
#endif
module fs_mgg
  ! Implicit free surface (implicit_fs.F90), fs_gpu_coarse = 'device': the
  ! multigrid's global levels held redundantly on every GPU instead of the
  ! host (fs_host keeps only the layout and the coarsest level's banded
  ! Cholesky). The coarsest distributed level is gathered by point-to-point
  ! GPU-aware messages (comm_allgatherv_dev) and placed into the global grid
  ! by index maps that reproduce the CPU's assembly order (mg_gather_coef,
  ! mg_gather_vec); the global levels' smoothing, residual, restriction and
  ! prolongation are the CPU's expressions in device kernels (the host
  ! mode's arithmetic); halos are zero, or the periodic wrap before each
  ! sweep (mg_halo_glob). Global level 1 = the CPU's mg_ld+1.
  use kinds
  use decomp_mod, only: decomp
  use fs_data
  use comm_mod, only: comm_allgatherv_dev, comm_allgatherv_h8, comm_rank, comm_size
  use fs_host, only: fsh_get_layout, fsh_factor_from, fsh_direct_vec
  use fs_mgk, only: mgk_init, mgk_down, mgk_up
  use fs_ipc, only: ipc_gather_init, ipcg_on, ipc_gather
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: mgg_init, mgg_setup, mgg_vcycle, mgg_set_faces, mgg_act_gather, mgg_act_coarse

  integer :: ng = 0, gx1 = 0, gy1 = 0, np = 1, me = 0, nvd = 0, ncd = 0
  integer :: glx(20) = 0, gly(20) = 0
  integer, allocatable :: gi0(:), gj0(:), gnx(:), gny(:), gcnt(:), gdsp(:), ccnt(:), cdsp(:)
  logical :: perx = .false., pery = .false.
  real(rp), FSD allocatable, dimension(:,:,:) :: gx_, gb_, ge_, gw_, gs_, gd_, gt_
  ! coarse face factors of the global levels (fs_mg_faces; (ic, level), levels 2..ng)
  real(rp), FSD allocatable :: gfx_(:,:), gfy_(:,:)
  integer, FSD allocatable :: mapv(:,:), mape(:,:), mapw(:,:), maps(:,:)
  real(rp), FSD allocatable :: dvs(:), dva(:), dcs(:), dca(:), dcf(:)
  ! the coarsest level's vector: dco (written by the down leg), dcx (read by the up leg); host
  ! views hco_b, hco_x. fs_gpu_opt 4: two mapped pinned buffers; otherwise dco = dcx = dco0 on the
  ! device with copies to and from the pinned hco_b0, hco_x0
  real(rp), FSD pointer, contiguous :: dco(:) => null(), dcx(:) => null()
  real(rp), pointer, contiguous :: hco_b(:) => null(), hco_x(:) => null()
  real(rp), FSD allocatable, target :: dco0(:)
  ! fs_gpu_opt 8: the gathers into mapped pinned buffers (host views hva, hca; device views
  ! dvm, dcm: this rank's block is written there by the pack kernel), host MPI, one copy up
  real(rp), FSD pointer, contiguous :: dvm(:) => null(), dcm(:) => null()
  real(rp), pointer, contiguous :: hva(:) => null(), hca(:) => null()
#ifdef USE_CUDA
  type(cudaGraph) :: gr_dn, gr_up
  type(cudaGraphExec) :: ge_dn, ge_up
  logical :: gr_ready = .false.
  real(rp), pinned, allocatable, target :: hco_b0(:), hco_x0(:)
  real(rp), pinned, allocatable :: hcf(:)
#else
  real(rp), allocatable, target :: hco_b0(:), hco_x0(:)
  real(rp), allocatable :: hcf(:)
#endif

contains

  subroutine mgg_init(nxd, nyd)
    ! layout from fs_host; device planes and the assembly maps
    integer, intent(in) :: nxd, nyd
    integer :: lnx(20), lny(20), k, i, j, n, i0, j0, mx, my
    real(rp), pointer, contiguous :: hp(:)
    real(rp), FSD pointer, contiguous :: dp(:)
    integer, allocatable :: hv(:,:), he(:,:), hw(:,:), hs(:,:)
    np = int(comm_size()); me = int(comm_rank())
    allocate(gi0(np), gj0(np), gnx(np), gny(np), gcnt(np), gdsp(np), ccnt(np), cdsp(np))
    call fsh_get_layout(ng, lnx, lny, gi0, gj0, gnx, gny, gcnt, gdsp, ccnt, cdsp)
    glx = lnx; gly = lny; gx1 = lnx(1); gy1 = lny(1)
    perx = decomp%periodic_x; pery = decomp%periodic_y
    allocate(gx_(0:gx1 + 1, 0:gy1 + 1, ng), gb_(0:gx1 + 1, 0:gy1 + 1, ng), ge_(0:gx1 + 1, 0:gy1 + 1, ng), &
             gw_(0:gx1 + 1, 0:gy1 + 1, ng), gs_(0:gx1 + 1, 0:gy1 + 1, ng), gd_(0:gx1 + 1, 0:gy1 + 1, ng), &
             gt_(0:gx1 + 1, 0:gy1 + 1, ng))
    gx_ = 0; gb_ = 0; ge_ = 0; gw_ = 0; gs_ = 0; gd_ = 0; gt_ = 0
    ! maps: the receive-buffer position (1-based) of every global value, in
    ! the CPU's loop order (later ranks overwrite earlier ones on shared faces)
    allocate(hv(gx1, gy1), he(gx1, gy1), hw(gx1 + 1, gy1), hs(gx1, gy1 + 1))
    hv = 0; he = 0; hw = 0; hs = 0
    do k = 1, np
      i0 = gi0(k); j0 = gj0(k); mx = gnx(k); my = gny(k)
      n = gdsp(k)
      do j = 1, my
        do i = 1, mx
          n = n + 1; hv(i0 + i, j0 + j) = n
        enddo
      enddo
      n = cdsp(k)
      do j = 1, my
        do i = 1, mx
          n = n + 1; he(i0 + i, j0 + j) = n
        enddo
      enddo
      do j = 1, my
        do i = 1, mx + 1
          n = n + 1; hw(i0 + i, j0 + j) = n
        enddo
      enddo
      do j = 1, my + 1
        do i = 1, mx
          n = n + 1; hs(i0 + i, j0 + j) = n
        enddo
      enddo
    enddo
    allocate(mapv(gx1, gy1), mape(gx1, gy1), mapw(gx1 + 1, gy1), maps(gx1, gy1 + 1))
    mapv = hv; mape = he; mapw = hw; maps = hs
    deallocate(hv, he, hw, hs)
    nvd = nxd * nyd; ncd = 3 * nxd * nyd + nxd + nyd
    allocate(dvs(nvd), dva(sum(gcnt)), dcs(ncd), dca(sum(ccnt)), dcf(3 * (glx(ng) + 2) * (gly(ng) + 2)))
    dvs = 0; dva = 0; dcs = 0; dca = 0; dcf = 0
    allocate(hcf(3 * (glx(ng) + 2) * (gly(ng) + 2)))
    hcf = 0
    n = glx(ng) * gly(ng)
    ! (map_alloc returns local pointers, assigned here: a module device pointer associated through a
    ! dummy argument keeps a stale device-side descriptor for the CUF kernels that name it)
    if (o_cvec) then
      call map_alloc(n, hp, dp); hco_b => hp; dco => dp
      call map_alloc(n, hp, dp); hco_x => hp; dcx => dp
    else
      allocate(dco0(n), hco_b0(n), hco_x0(n)); dco0 = 0; hco_b0 = 0; hco_x0 = 0
      dco => dco0; dcx => dco0; hco_b => hco_b0; hco_x => hco_x0
    endif
    if (o_fleg) call mgk_init(glx, gly)
    if (o_ipc) then
      call ipc_gather_init(np, me, sum(gcnt), gdsp(me + 1))
      if (talk) write(*,'(1x,A,L1)') 'IMPLICIT_FS: fs_gpu_opt 256: multigrid gather by CUDA IPC: ', ipcg_on
    endif
    if (o_gath .and. np > 1) then
      call map_alloc(sum(gcnt), hp, dp); hva => hp; dvm => dp
      call map_alloc(sum(ccnt), hp, dp); hca => hp; dcm => dp
    endif
  end subroutine mgg_init

  subroutine mgg_set_faces(n1, fx, fy)
    ! the coarse face factors of the global levels (fs_mg.F90 mg_faces)
    integer, intent(in) :: n1
    real(rp), intent(in) :: fx(n1, 20), fy(n1, 20)
    if (allocated(gfx_)) deallocate(gfx_, gfy_)
    allocate(gfx_(n1, 20), gfy_(n1, 20))
    gfx_ = fx; gfy_ = fy
  end subroutine mgg_set_faces

  subroutine mgg_setup(l)
    ! global coefficients from the coarsest distributed level l: gather,
    ! assembly, global coarsening, the coarsest level's factor on the host
    integer, intent(in) :: l
    integer(ip) :: i, j, ic, jc, nxc, nyc, nxf, nyf, fi, fj, ll, k, o, a, b, gl
    integer :: ierr, n, m
    real(rp) :: s
    nxc = mg_nx(l); nyc = mg_ny(l); ll = l; o = cdsp(me + 1)
    if (o_gath .and. np > 1) then
      CUF_DO2
      do jc = 1, nyc + 1
        do ic = 1, nxc + 1
          if (ic <= nxc .and. jc <= nyc) dcm(o + ic + (jc - 1) * nxc) = me_(ic, jc, ll)
          if (jc <= nyc) dcm(o + nxc * nyc + ic + (jc - 1) * (nxc + 1)) = mw_(ic, jc, ll)
          if (ic <= nxc) dcm(o + nxc * nyc + (nxc + 1) * nyc + ic + (jc - 1) * nxc) = ms_(ic, jc, ll)
        enddo
      enddo
#ifdef USE_CUDA
      ierr = cudaStreamSynchronize(work_stream)
#endif
      call comm_allgatherv_h8(hca, ccnt, cdsp)
#ifdef USE_CUDA
      ierr = cudaMemcpyAsync(dca, hca, sum(ccnt), cudaMemcpyHostToDevice, work_stream)
      if (ierr /= 0) error stop 'implicit_fs: mgg_setup gather copy'
#endif
    else
    CUF_DO2
    do jc = 1, nyc + 1
      do ic = 1, nxc + 1
        if (ic <= nxc .and. jc <= nyc) then
          dcs(ic + (jc - 1) * nxc) = me_(ic, jc, ll); dca(o + ic + (jc - 1) * nxc) = me_(ic, jc, ll)
        endif
        if (jc <= nyc) then
          dcs(nxc * nyc + ic + (jc - 1) * (nxc + 1)) = mw_(ic, jc, ll)
          dca(o + nxc * nyc + ic + (jc - 1) * (nxc + 1)) = mw_(ic, jc, ll)
        endif
        if (ic <= nxc) then
          dcs(nxc * nyc + (nxc + 1) * nyc + ic + (jc - 1) * nxc) = ms_(ic, jc, ll)
          dca(o + nxc * nyc + (nxc + 1) * nyc + ic + (jc - 1) * nxc) = ms_(ic, jc, ll)
        endif
      enddo
    enddo
    endif
    if (np > 1 .and. .not. o_gath) then
#ifdef USE_CUDA
      ierr = cudaStreamSynchronize(work_stream)
#endif
      call comm_allgatherv_dev(ncd, dcs, dca, ccnt, cdsp)
    endif
    a = gx1; b = gy1
    CUF_DO2
    do j = 0, b + 1
      do i = 0, a + 1
        ge_(i, j, 1) = 0; gw_(i, j, 1) = 0; gs_(i, j, 1) = 0; gd_(i, j, 1) = 0
        if (i >= 1 .and. i <= a .and. j >= 1 .and. j <= b) ge_(i, j, 1) = dca(mape(i, j))
        if (i >= 1 .and. j >= 1 .and. j <= b) gw_(i, j, 1) = dca(mapw(i, j))
        if (i >= 1 .and. i <= a .and. j >= 1) gs_(i, j, 1) = dca(maps(i, j))
      enddo
    enddo
    call gdiag(1)
    do gl = 2, ng
      nxc = glx(gl); nyc = gly(gl); nxf = glx(gl - 1); nyf = gly(gl - 1); k = gl
      CUF_DO2
      do j = 0, b + 1
        do i = 0, a + 1
          ge_(i, j, k) = 0; gw_(i, j, k) = 0; gs_(i, j, k) = 0; gd_(i, j, k) = 0
        enddo
      enddo
      CUF_DO2
      do jc = 1, nyc + 1
        do ic = 1, nxc + 1
          if (ic <= nxc .and. jc <= nyc) then
            s = 0.0_rp
            do j = 2 * jc - 1, min(2 * jc, nyf)
              do i = 2 * ic - 1, min(2 * ic, nxf)
                s = s + ge_(i, j, k - 1)
              enddo
            enddo
            ge_(ic, jc, k) = s
          endif
          if (jc <= nyc) then
            fi = min(2 * ic - 1, nxf + 1)
            s = 0.0_rp
            do j = 2 * jc - 1, min(2 * jc, nyf)
              s = s + gfx_(ic, k) * gw_(fi, j, k - 1)
            enddo
            gw_(ic, jc, k) = s
          endif
          if (ic <= nxc) then
            fj = min(2 * jc - 1, nyf + 1)
            s = 0.0_rp
            do i = 2 * ic - 1, min(2 * ic, nxf)
              s = s + gfy_(jc, k) * gs_(i, fj, k - 1)
            enddo
            gs_(ic, jc, k) = s
          endif
        enddo
      enddo
      call gdiag(gl)
    enddo
    ! the coarsest level's e, cw, cs (with halos) to the host for the factor
    nxc = glx(ng); nyc = gly(ng); k = ng
    n = int((nxc + 2) * (nyc + 2))
    CUF_DO2
    do j = 0, nyc + 1
      do i = 0, nxc + 1
        dcf(i + 1 + j * (nxc + 2)) = ge_(i, j, k)
        dcf(n + i + 1 + j * (nxc + 2)) = gw_(i, j, k)
        dcf(2 * n + i + 1 + j * (nxc + 2)) = gs_(i, j, k)
      enddo
    enddo
#ifdef USE_CUDA
    ierr = cudaMemcpyAsync(hcf, dcf, 3 * n, cudaMemcpyDeviceToHost, work_stream)
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'implicit_fs: mgg_setup copy'
#else
    hcf(1:3 * n) = dcf(1:3 * n)
#endif
    call fsh_factor_from(int(nxc), int(nyc), hcf(1), hcf(n + 1), hcf(2 * n + 1))
  end subroutine mgg_setup

  subroutine gdiag(gl)
    integer, intent(in) :: gl
    integer(ip) :: i, j, nx, ny, k
    nx = glx(gl); ny = gly(gl); k = gl
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        if (ge_(i, j, k) > 0.0_rp) gd_(i, j, k) = 1.0_rp / (ge_(i, j, k) + gw_(i, j, k) + gw_(i + 1, j, k) &
             + gs_(i, j, k) + gs_(i, j + 1, k))
      enddo
    enddo
  end subroutine gdiag
  subroutine mgg_vcycle(l)
    ! the global levels of one V-cycle: the rhs of the coarsest distributed
    ! level l gathered into global level 1, down to the coarsest (direct
    ! solve on the host), up again, this rank's block of global level 1
    ! back into level l's x
    integer, intent(in) :: l
    integer(ip) :: i, j, nxd, nyd, ll, o, a, b, i0, j0, k
    integer :: ierr, gl, n, kk
    nxd = mg_nx(l); nyd = mg_ny(l); ll = l; o = gdsp(me + 1)
    if (ipcg_on) then
      ! fs_gpu_opt 256: device-initiated gather on one node (fs_ipc)
      call ipc_gather(mb_, mgx, mgy, int(ll), int(nxd), int(nyd), dva)
      n_ring = n_ring + 1
    else if (o_gath .and. np > 1) then
      CUF_DO2
      do j = 1, nyd
        do i = 1, nxd
          dvm(o + i + (j - 1) * nxd) = mb_(i, j, ll)
        enddo
      enddo
      if (vg_rec) then
        call vg_point_p(2, [0, 0, 0, 0, 0, 0, 0, 0])
      else
        call mgg_act_gather()
      endif
#ifdef USE_CUDA
      ierr = cudaMemcpyAsync(dva, hva, sum(gcnt), cudaMemcpyHostToDevice, work_stream)
      if (ierr /= 0) error stop 'implicit_fs: mgg_vcycle gather copy'
#endif
    else
      CUF_DO2
      do j = 1, nyd
        do i = 1, nxd
          dvs(i + (j - 1) * nxd) = mb_(i, j, ll); dva(o + i + (j - 1) * nxd) = mb_(i, j, ll)
        enddo
      enddo
      if (np > 1) then
        if (vg_rec) then
          call vg_point_p(2, [0, 0, 0, 0, 0, 0, 0, 0])
        else
          call mgg_act_gather()
        endif
      endif
    endif
    a = gx1; b = gy1
    CUF_DO2
    do j = 1, b
      do i = 1, a
        gb_(i, j, 1) = dva(mapv(i, j))
      enddo
    enddo
    call tq(T_MCO)
    a = glx(ng); b = gly(ng); k = ng; n = int(a * b)
#ifdef USE_CUDA
    if (o_fleg) then
      call mgk_down(gx_, gb_, ge_, gw_, gs_, gd_, gt_, dco, gx1, gy1, ng, fs_mg_nu, jac_sm, perx, pery, om)
    else if (fs_gpu_graph > 0 .and. .not. o_vgr) then
      if (.not. gr_ready) then
        fs_capt = .true.
        ierr = cudaStreamBeginCapture(work_stream, 0)
        if (ierr /= 0) error stop 'implicit_fs: graph capture (down)'
        call leg_down()
        ierr = cudaStreamEndCapture(work_stream, gr_dn)
        if (ierr /= 0) error stop 'implicit_fs: graph end (down)'
        ierr = cudaGraphInstantiate(ge_dn, gr_dn, 0)
        ierr = cudaStreamBeginCapture(work_stream, 0)
        if (ierr /= 0) error stop 'implicit_fs: graph capture (up)'
        call leg_up()
        ierr = cudaStreamEndCapture(work_stream, gr_up)
        if (ierr /= 0) error stop 'implicit_fs: graph end (up)'
        ierr = cudaGraphInstantiate(ge_up, gr_up, 0)
        fs_capt = .false.
        gr_ready = .true.
      endif
      ierr = cudaGraphLaunch(ge_dn, work_stream)
      if (ierr /= 0) error stop 'implicit_fs: graph launch (down)'
    else
      call leg_down()
    endif
#else
    call leg_down()
#endif
    ! coarsest: direct solve on the host
#ifdef USE_CUDA
    if (.not. o_cvec) ierr = cudaMemcpyAsync(hco_b, dco, n, cudaMemcpyDeviceToHost, work_stream)
#else
    if (.not. o_cvec) hco_b(1:n) = dco(1:n)
#endif
    if (vg_rec) then
      call vg_point_p(3, [0, 0, 0, 0, 0, 0, 0, 0])
    else
      call mgg_act_coarse()
    endif
#ifdef USE_CUDA
    if (.not. o_cvec) ierr = cudaMemcpyAsync(dco, hco_x, n, cudaMemcpyHostToDevice, work_stream)
    if (ierr /= 0) error stop 'implicit_fs: mgg coarse copy up'
    if (o_fleg) then
      call mgk_up(gx_, gb_, ge_, gw_, gs_, gd_, gt_, dcx, mx_, gx1, gy1, ng, fs_mg_nu, jac_sm, perx, pery, om, &
           gi0(me + 1), gj0(me + 1), int(nxd), int(nyd), mgx, mgy, int(ll))
    else if (fs_gpu_graph > 0 .and. .not. o_vgr) then
      ierr = cudaGraphLaunch(ge_up, work_stream)
      if (ierr /= 0) error stop 'implicit_fs: graph launch (up)'
    else
      call leg_up()
    endif
#else
    if (.not. o_cvec) dco(1:n) = hco_x(1:n)
    call leg_up()
#endif
    call tq(T_MCO)
  contains
    subroutine leg_down()
      integer :: gl2, kk2
      integer(ip) :: i2, j2, a2, b2, k2
      do gl2 = 1, ng - 1
        call gzero(gl2)
        do kk2 = 1, fs_mg_nu
          call ghalo(gl2)
          call gsmooth(gl2, 1)
        enddo
        call ghalo(gl2)
        call gresid_restrict(gl2)
      enddo
      a2 = glx(ng); b2 = gly(ng); k2 = ng
      CUF_DO2
      do j2 = 1, b2
        do i2 = 1, a2
          dco(i2 + (j2 - 1) * a2) = gb_(i2, j2, k2)
        enddo
      enddo
    end subroutine leg_down
    subroutine leg_up()
      integer :: gl2, kk2
      integer(ip) :: i2, j2, a2, b2, k2, i02, j02, nx2, ny2, l2
      a2 = glx(ng); b2 = gly(ng); k2 = ng
      call gzero(ng)
      CUF_DO2
      do j2 = 1, b2
        do i2 = 1, a2
          gx_(i2, j2, k2) = dcx(i2 + (j2 - 1) * a2)
        enddo
      enddo
      do gl2 = ng - 1, 1, -1
        call gprolong(gl2)
        do kk2 = 1, fs_mg_nu
          call ghalo(gl2)
          call gsmooth(gl2, 2)
        enddo
      enddo
      ! this rank's block into the coarsest distributed level (zero halo)
      i02 = gi0(me + 1); j02 = gj0(me + 1); a2 = mgx; b2 = mgy; nx2 = nxd; ny2 = nyd; l2 = ll
      CUF_DO2
      do j2 = 0, b2 + 1
        do i2 = 0, a2 + 1
          mx_(i2, j2, l2) = 0.0_rp
          if (i2 >= 1 .and. i2 <= nx2 .and. j2 >= 1 .and. j2 <= ny2) mx_(i2, j2, l2) = gx_(i02 + i2, j02 + j2, 1)
        enddo
      enddo
    end subroutine leg_up
  end subroutine mgg_vcycle

  subroutine mgg_act_gather()
    ! the host part of the vector gather: the pack kernel's data complete, then MPI
    integer :: ierr
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
#endif
    if (o_gath) then
      call comm_allgatherv_h8(hva, gcnt, gdsp)
    else
      call comm_allgatherv_dev(nvd, dvs, dva, gcnt, gdsp)
    endif
    n_ring = n_ring + 1
  end subroutine mgg_act_gather

  subroutine mgg_act_coarse()
    ! the host part of the coarsest level: the down leg complete, the banded Cholesky solve
    integer :: ierr
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'implicit_fs: mgg coarse copy down'
#endif
    call fsh_direct_vec(glx(ng) * gly(ng), hco_b, hco_x)
  end subroutine mgg_act_coarse

  subroutine gzero(gl)
    integer, intent(in) :: gl
    integer(ip) :: i, j, a, b, k
    a = gx1; b = gy1; k = gl
    CUF_DO2
    do j = 0, b + 1
      do i = 0, a + 1
        gx_(i, j, k) = 0.0_rp
      enddo
    enddo
  end subroutine gzero

  subroutine ghalo(gl)
    ! mg_halo_glob: periodic wrap; zero halos stay zero (only interiors are written)
    integer, intent(in) :: gl
    integer(ip) :: m, nx, ny, mm, k
    logical :: px, py
    if (.not. (perx .or. pery)) return
    nx = glx(gl); ny = gly(gl); k = gl; px = perx; py = pery; mm = max(nx, ny) + 2
    CUF_DO1
    do m = 0, mm
      if (px .and. m >= 1 .and. m <= ny) then
        gx_(0, m, k) = gx_(nx, m, k); gx_(nx + 1, m, k) = gx_(1, m, k)
      endif
      if (py .and. m >= 1 .and. m <= nx) then
        gx_(m, 0, k) = gx_(m, ny, k); gx_(m, ny + 1, k) = gx_(m, 1, k)
      endif
    enddo
  end subroutine ghalo

  subroutine gsmooth(gl, mode)
    integer, intent(in) :: gl, mode
    integer(ip) :: i, j, nx, ny, c, col, k, p, hx, kc
    real(rp) :: w
    nx = glx(gl); ny = gly(gl); k = gl
    if (jac_sm) then
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          gt_(i, j, k) = gd_(i, j, k) * (gb_(i, j, k) + gw_(i, j, k) * gx_(i - 1, j, k) &
               + gw_(i + 1, j, k) * gx_(i + 1, j, k) + gs_(i, j, k) * gx_(i, j - 1, k) &
               + gs_(i, j + 1, k) * gx_(i, j + 1, k))
        enddo
      enddo
      w = om
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          gx_(i, j, k) = (1.0_rp - w) * gx_(i, j, k) + w * gt_(i, j, k)
        enddo
      enddo
      call tq(T_MSM)
      return
    endif
    do c = 1, 2
      col = mod(c + mode, 2)
      if (o_cc) then
        p = mod(1 + col, 2); hx = (nx + 1) / 2
        CUF_DO2
        do j = 1, ny
          do kc = 0, hx - 1
            i = 1 + mod(1 + j + p, 2) + 2 * kc
            if (i <= nx) &
               gx_(i, j, k) = gd_(i, j, k) * (gb_(i, j, k) + gw_(i, j, k) * gx_(i - 1, j, k) &
               + gw_(i + 1, j, k) * gx_(i + 1, j, k) + gs_(i, j, k) * gx_(i, j - 1, k) &
               + gs_(i, j + 1, k) * gx_(i, j + 1, k))
          enddo
        enddo
        cycle
      endif
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          if (mod(i + j + col, 2) == 1) &
               gx_(i, j, k) = gd_(i, j, k) * (gb_(i, j, k) + gw_(i, j, k) * gx_(i - 1, j, k) &
               + gw_(i + 1, j, k) * gx_(i + 1, j, k) + gs_(i, j, k) * gx_(i, j - 1, k) &
               + gs_(i, j + 1, k) * gx_(i, j + 1, k))
        enddo
      enddo
    enddo
    call tq(T_MSM)
  end subroutine gsmooth

  subroutine gresid_restrict(gl)
    integer, intent(in) :: gl
    integer(ip) :: i, j, ic, jc, nxc, nyc, nxf, nyf, k
    real(rp) :: s, r
    nxf = glx(gl); nyf = gly(gl); nxc = glx(gl + 1); nyc = gly(gl + 1); k = gl
    CUF_DO2
    do jc = 1, nyc
      do ic = 1, nxc
        s = 0.0_rp
        do j = 2 * jc - 1, min(2 * jc, nyf)
          do i = 2 * ic - 1, min(2 * ic, nxf)
            r = 0.0_rp
            if (ge_(i, j, k) > 0.0_rp) r = gb_(i, j, k) - (ge_(i, j, k) * gx_(i, j, k) &
                 + gw_(i, j, k) * (gx_(i, j, k) - gx_(i - 1, j, k)) &
                 + gw_(i + 1, j, k) * (gx_(i, j, k) - gx_(i + 1, j, k)) &
                 + gs_(i, j, k) * (gx_(i, j, k) - gx_(i, j - 1, k)) &
                 + gs_(i, j + 1, k) * (gx_(i, j, k) - gx_(i, j + 1, k)))
            s = s + r
          enddo
        enddo
        gb_(ic, jc, k + 1) = s
      enddo
    enddo
    call tq(T_MTR)
  end subroutine gresid_restrict

  subroutine gprolong(gl)
    integer, intent(in) :: gl
    integer(ip) :: i, j, nxf, nyf, k
    nxf = glx(gl); nyf = gly(gl); k = gl
    CUF_DO2
    do j = 1, nyf
      do i = 1, nxf
        if (ge_(i, j, k) > 0.0_rp) gx_(i, j, k) = gx_(i, j, k) + gx_((i + 1) / 2, (j + 1) / 2, k + 1)
      enddo
    enddo
    call tq(T_MTR)
  end subroutine gprolong

end module fs_mgg

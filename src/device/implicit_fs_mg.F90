#include "mem_attr.inc"
#include "cuf_launch.inc"
#ifdef USE_CUDA
#define FSD device,
#else
#define FSD
#endif
module fs_mg
  ! Implicit free surface (implicit_fs.F90): the multigrid preconditioner's
  ! device part (FS_SPEC 5): level layout, coarse operators, the V-cycle on
  ! the fine level and the distributed coarse levels; the global levels are
  ! on the host (fs_host).
  use kinds
  use decomp_mod, only: decomp
  use fs_data
  use comm_mod, only: comm_allreduce_sum, comm_allreduce_min, comm_rank, comm_size
  use run_timers, only: wall_time
  use fs_host, only: fsh_layout, fsh_setup, fsh_vcycle, fsh_print, fsh_levels, fsh_set_faces, fsh_colord
  use fs_mgg, only: mgg_init, mgg_setup, mgg_vcycle, mgg_set_faces, mgg_act_gather, mgg_act_coarse
  use comm_mod, only: comm_allgather_i4
  use fs_ipc, only: ipc_on, ipc_ring_plane
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: mg_build_layout, mg_setup, mg_vcycle

  ! fs_gpu_opt 128: the recorded V-cycle, vg_n stretches, each a graph followed by a host
  ! action (vg_kind 1 ring MPI with vg_par = nb(4), nw, ns; 2 vector gather; 3 coarsest solve on
  ! the host after the device down leg; 4 the host global levels; 0 none: the last stretch)
  integer, parameter :: VG_MAX = 128
  integer :: vg_n = 0
  logical :: vg_ready = .false.
  integer :: vg_kind(VG_MAX) = 0, vg_par(8, VG_MAX) = 0
  ! FS_STAT: the ring rounds inside the recorded graphs (IPC rings and the IPC gather, which are
  ! kernels, not host actions); a replay adds them (the host actions count themselves in vg_act)
  integer(8) :: vg_ring0 = 0, vg_ring_g = 0
  ! fs_gpu_opt 8192 (even fs_mg_nu, red-black): one kernel per sweep of the fine and the distributed
  ! levels, reading one array and writing the other (zz <-> zzb, mx_ <-> mxb_); the down leg's first
  ! sweep reads zz0 / mz_, zeros that are never written (instead of zz, mx_ zeroed)
  real(rp), FSD allocatable :: zzb(:,:), zz0(:,:), mxb_(:,:,:), mz_(:,:,:)
  logical :: g2f = .false.
#ifdef USE_CUDA
  type(cudaGraphExec) :: vg_exec(VG_MAX)
#endif

contains

  subroutine mg_build_layout()
    integer :: l, ng, np, rk
    real(rp) :: mn(1), gmn(1), cl(1), gcl(1)
    mg_nx(0) = decomp%nx; mg_ny(0) = decomp%ny
    fsh_colord = o_chol
    l = 0
    do
      mn(1) = real(min(mg_nx(l), mg_ny(l)), rp)
      call comm_allreduce_min(1, mn, gmn)
      if (l >= 1 .and. (gmn(1) < 4.0_rp .or. (fs_mg_levels > 0 .and. l >= fs_mg_levels))) exit
      if (l >= 1 .and. fs_gpu_gather > 0) then
        cl(1) = real(mg_nx(l), rp) * real(mg_ny(l), rp)
        call comm_allreduce_sum(1, cl, gcl)
        if (gcl(1) <= real(fs_gpu_gather, rp)) exit
      endif
      if (l + 1 > mg_maxl - 6) exit
      l = l + 1
      mg_nx(l) = (mg_nx(l - 1) + 1) / 2; mg_ny(l) = (mg_ny(l - 1) + 1) / 2
    enddo
    mg_ld = l
    mgx = mg_nx(1); mgy = mg_ny(1)
    allocate(mx_(0:mgx + 1, 0:mgy + 1, mg_ld), mb_(0:mgx + 1, 0:mgy + 1, mg_ld), me_(0:mgx + 1, 0:mgy + 1, mg_ld), &
             mw_(0:mgx + 1, 0:mgy + 1, mg_ld), ms_(0:mgx + 1, 0:mgy + 1, mg_ld), md_(0:mgx + 1, 0:mgy + 1, mg_ld), &
             mt_(0:mgx + 1, 0:mgy + 1, mg_ld))
    mx_ = 0; mb_ = 0; me_ = 0; mw_ = 0; ms_ = 0; md_ = 0; mt_ = 0
    g2f = o_g2f .and. .not. jac_sm .and. mod(fs_mg_nu, 2) == 0
    if (g2f) then
      allocate(zzb(decomp%is:decomp%ie, decomp%js:decomp%je), zz0(decomp%is:decomp%ie, decomp%js:decomp%je), &
               mxb_(0:mgx + 1, 0:mgy + 1, mg_ld), mz_(0:mgx + 1, 0:mgy + 1, 1))
      zzb = 0; zz0 = 0; mxb_ = 0; mz_ = 0
    endif
    l = mg_ld
    allocate(db_v(mg_nx(l) * mg_ny(l)), db_c(3 * mg_nx(l) * mg_ny(l) + mg_nx(l) + mg_ny(l)))
    allocate(hb_b(mg_nx(l) * mg_ny(l)), hb_x(mg_nx(l) * mg_ny(l)), hb_c(3 * mg_nx(l) * mg_ny(l) + mg_nx(l) + mg_ny(l)))
    db_v = 0; db_c = 0; hb_b = 0; hb_x = 0; hb_c = 0
    np = int(comm_size()); rk = int(comm_rank())
    call fsh_layout(np, rk, int(decomp%inode), int(decomp%jnode), int(mg_nx(l)), int(mg_ny(l)), fs_mg_nu, jac_sm, &
                    decomp%periodic_x, decomp%periodic_y, ng)
    mg_ng = ng
    if (dev_glob) call mgg_init(int(mg_nx(l)), int(mg_ny(l)))
    call mg_faces()
    if (talk) then
      write(*,'(1x,A,I2,A,I2,A,A)') 'IMPLICIT_FS mg: ', mg_ld + mg_ng, ' levels (', mg_ld, &
           ' distributed coarse, then global on the ', trim(merge('device', 'host  ', dev_glob)) // &
           ', the coarsest solved on the host)'
      do l = 0, mg_ld
        write(*,'(1x,A,I2,A,I5,A,I5,A,L1)') 'IMPLICIT_FS mg level', l, ' nx', mg_nx(l), ' ny', mg_ny(l), ' global ', .false.
      enddo
      call fsh_print(mg_ld + 1)
    endif
  end subroutine mg_build_layout

  subroutine mg_faces()
    ! Coarse face factors of every coarsened level (fs_mg_faces; CPU 36a863a8 mg_build_faces, FS_SPEC
    ! 5, 9b-E item 2). 'width': for the coarse face between cells I-1 and I (fine face fi = min(2I-1,
    ! nf+1) between fine cells fi-1 and fi), (w_f(fi-1) + w_f(fi)) / (w_c(I-1) + w_c(I)), the cell
    ! widths in fine-grid cells on the level's global index, from all tiles' sizes (the neighbours'
    ! halo cells have their true widths, so both sides of a tile seam get the same factor; beyond a
    ! physical edge the edge cell's width, a periodic wrap with periodic_x/y). Distributed levels
    ! 1..mg_ld: 2x2 aggregation inside each tile; the gathered level (global level 1) has the cells
    ! of level mg_ld; global levels 2..ng: aggregation of the global grid. 'half': 0.5 everywhere.
    ! Identical to 0.5 for full aggregates of equal cells; differs at partial aggregates (odd tile
    ! sizes, tiles one cell wide).
    integer :: np, k, ngl, lnx(20), lny(20), n1, l
    integer, allocatable :: info(:,:)
    integer, allocatable :: cxs(:), cys(:)
    real(rp), allocatable :: fd(:,:), fe(:,:), gx(:,:), gy(:,:)
    integer :: npx, npy
    logical :: ok
    np = int(comm_size())
    call fsh_levels(ngl, lnx, lny)
    allocate(mfx(mgx + 1, mg_ld), mfy(mgy + 1, mg_ld))
    n1 = max(lnx(1), lny(1)) + 1
    allocate(fd(mgx + 1, mg_ld), fe(mgy + 1, mg_ld), gx(n1, 20), gy(n1, 20))
    fd = 0.5_rp; fe = 0.5_rp; gx = 0.5_rp; gy = 0.5_rp
    if (trim(fs_mg_faces) == 'width') then
      allocate(info(4, np))
      call comm_allgather_i4(4, [int(decomp%inode), int(decomp%jnode), int(decomp%nx), int(decomp%ny)], info)
      npx = maxval(info(1, :)) + 1; npy = maxval(info(2, :)) + 1
      allocate(cxs(0:npx - 1), cys(0:npy - 1))
      cxs = -1; cys = -1
      ok = .true.
      do k = 1, np            ! the tile sizes per column / row (a tensor-product tiling)
        if (cxs(info(1, k)) < 0) cxs(info(1, k)) = info(3, k)
        if (cys(info(2, k)) < 0) cys(info(2, k)) = info(4, k)
        ok = ok .and. cxs(info(1, k)) == info(3, k) .and. cys(info(2, k)) == info(4, k)
      enddo
      if (.not. ok .or. any(cxs < 0) .or. any(cys < 0)) &
           error stop "IMPLICIT_FS: fs_mg_faces = 'width' needs a tensor-product tiling (use 'half')"
      call faces_dir(cxs, npx, int(decomp%inode), mg_nx, lnx, decomp%periodic_x, mgx + 1, fd, gx)
      call faces_dir(cys, npy, int(decomp%jnode), mg_ny, lny, decomp%periodic_y, mgy + 1, fe, gy)
      deallocate(info, cxs, cys)
    endif
    mfx = fd; mfy = fe
    call fsh_set_faces(n1, gx, gy)
    if (dev_glob) call mgg_set_faces(n1, gx, gy)
    if (talk) then
      ! (rank 0's distributed factors)
      do l = 1, mg_ld
        write(*,'(1x,A,I2,A,4F8.4)') 'IMPLICIT_FS mg faces level', l, ' min, max x; min, max y', &
             minval(fd(1:mg_nx(l) + 1, l)), maxval(fd(1:mg_nx(l) + 1, l)), &
             minval(fe(1:mg_ny(l) + 1, l)), maxval(fe(1:mg_ny(l) + 1, l))
      enddo
      do l = 2, ngl
        write(*,'(1x,A,I2,A,4F8.4)') 'IMPLICIT_FS mg faces level', mg_ld + l - 1, ' min, max x; min, max y', &
             minval(gx(1:lnx(l) + 1, l)), maxval(gx(1:lnx(l) + 1, l)), minval(gy(1:lny(l) + 1, l)), &
             maxval(gy(1:lny(l) + 1, l))
      enddo
    endif
    deallocate(fd, fe, gx, gy)
  contains
    subroutine faces_dir(c0, npn, me, nl, lg, per, nd, ffd, ffg)
      integer, intent(in) :: npn, me, nd, lg(20)
      integer, intent(in) :: c0(0:npn - 1)
      integer, intent(in) :: nl(0:mg_maxl)
      logical, intent(in) :: per
      real(rp), intent(inout) :: ffd(nd, mg_ld), ffg(n1, 20)
      integer :: c(0:npn - 1), cp(0:npn - 1), t, ic, k0f, k0c, g0f, g0c, nf, nc, fi, ll, lev
      real(rp), allocatable :: wf(:), wc(:)
      c = c0
      allocate(wf(sum(c))); wf = 1.0_rp
      do ll = 1, mg_ld + ngl
        if (ll <= mg_ld) then                      ! aggregation inside each tile
          cp = c; c = (c + 1) / 2
          allocate(wc(sum(c)))
          k0f = 0; k0c = 0
          do t = 0, npn - 1
            do ic = 1, c(t)
              wc(k0c + ic) = wf(k0f + 2 * ic - 1)
              if (2 * ic <= cp(t)) wc(k0c + ic) = wc(k0c + ic) + wf(k0f + 2 * ic)
            enddo
            k0f = k0f + cp(t); k0c = k0c + c(t)
          enddo
          g0f = sum(cp(0:me - 1)); g0c = sum(c(0:me - 1)); nf = cp(me); nc = c(me)
          if (nc /= nl(ll)) error stop 'IMPLICIT_FS mg_faces: distributed level size'
        else if (ll == mg_ld + 1) then             ! the gathered level: the same cells
          cycle
        else                                       ! aggregation of the global grid
          nf = size(wf); nc = (nf + 1) / 2
          allocate(wc(nc))
          do ic = 1, nc
            wc(ic) = wf(2 * ic - 1)
            if (2 * ic <= nf) wc(ic) = wc(ic) + wf(2 * ic)
          enddo
          g0f = 0; g0c = 0
          if (nc /= lg(ll - mg_ld)) error stop 'IMPLICIT_FS mg_faces: global level size'
        endif
        do ic = 1, nc + 1
          fi = min(2 * ic - 1, nf + 1)
          if (ll <= mg_ld) then
            ffd(ic, ll) = (wget(wf, g0f + fi - 1, per) + wget(wf, g0f + fi, per)) &
                 / (wget(wc, g0c + ic - 1, per) + wget(wc, g0c + ic, per))
          else
            lev = ll - mg_ld
            ffg(ic, lev) = (wget(wf, g0f + fi - 1, per) + wget(wf, g0f + fi, per)) &
                 / (wget(wc, g0c + ic - 1, per) + wget(wc, g0c + ic, per))
          endif
        enddo
        call move_alloc(wc, wf)
      enddo
      deallocate(wf)
    end subroutine faces_dir
    real(rp) function wget(w, k, per)
      ! width of cell k of a level (beyond the ends: periodic wrap or the end cell)
      real(rp), intent(in) :: w(:)
      integer, intent(in) :: k
      logical, intent(in) :: per
      integer :: n
      n = size(w)
      if (k < 1) then
        wget = w(1)
        if (per) wget = w(n)
      else if (k > n) then
        wget = w(n)
        if (per) wget = w(1)
      else
        wget = w(k)
      endif
    end function wget
  end subroutine mg_faces

  subroutine mg_setup()
    ! coarse operators of all levels from the step's fine operator
    integer(ip) :: i, j, ic, jc, l, nxc, nyc, nxf, nyf, fi, fj, n, mxx, myy
    real(rp) :: s
    integer :: ierr
    real(8) :: t0
    t0 = wall_time()
    mxx = mgx; myy = mgy
    do l = 1, mg_ld
      nxc = mg_nx(l); nyc = mg_ny(l); nxf = mg_nx(l - 1); nyf = mg_ny(l - 1)
      CUF_DO2
      do j = 0, myy + 1
        do i = 0, mxx + 1
          me_(i, j, l) = 0; mw_(i, j, l) = 0; ms_(i, j, l) = 0; md_(i, j, l) = 0
        enddo
      enddo
      if (l == 1) then
        CUF_DO2
        do jc = 1, nyc + 1
          do ic = 1, nxc + 1
            if (ic <= nxc .and. jc <= nyc) then
              s = 0.0_rp
              do j = 2 * jc - 1, min(2 * jc, nyf)
                do i = 2 * ic - 1, min(2 * ic, nxf)
                  s = s + act(i, j) * (ar(i, j) + rb(i, j))
                enddo
              enddo
              me_(ic, jc, 1) = s
            endif
            if (jc <= nyc) then
              fi = min(2 * ic - 1, nxf + 1)
              s = 0.0_rp
              do j = 2 * jc - 1, min(2 * jc, nyf)
                s = s + mfx(ic, 1) * cu(fi, j)
              enddo
              mw_(ic, jc, 1) = s
            endif
            if (ic <= nxc) then
              fj = min(2 * jc - 1, nyf + 1)
              s = 0.0_rp
              do i = 2 * ic - 1, min(2 * ic, nxf)
                s = s + mfy(jc, 1) * cv(i, fj)
              enddo
              ms_(ic, jc, 1) = s
            endif
          enddo
        enddo
      else
        CUF_DO2
        do jc = 1, nyc + 1
          do ic = 1, nxc + 1
            if (ic <= nxc .and. jc <= nyc) then
              s = 0.0_rp
              do j = 2 * jc - 1, min(2 * jc, nyf)
                do i = 2 * ic - 1, min(2 * ic, nxf)
                  s = s + me_(i, j, l - 1)
                enddo
              enddo
              me_(ic, jc, l) = s
            endif
            if (jc <= nyc) then
              fi = min(2 * ic - 1, nxf + 1)
              s = 0.0_rp
              do j = 2 * jc - 1, min(2 * jc, nyf)
                s = s + mfx(ic, l) * mw_(fi, j, l - 1)
              enddo
              mw_(ic, jc, l) = s
            endif
            if (ic <= nxc) then
              fj = min(2 * jc - 1, nyf + 1)
              s = 0.0_rp
              do i = 2 * ic - 1, min(2 * ic, nxf)
                s = s + mfy(jc, l) * ms_(i, fj, l - 1)
              enddo
              ms_(ic, jc, l) = s
            endif
          enddo
        enddo
      endif
      CUF_DO2
      do jc = 1, nyc
        do ic = 1, nxc
          if (me_(ic, jc, l) > 0.0_rp) md_(ic, jc, l) = 1.0_rp / (me_(ic, jc, l) + mw_(ic, jc, l) + mw_(ic + 1, jc, l) &
               + ms_(ic, jc, l) + ms_(ic, jc + 1, l))
        enddo
      enddo
    enddo
    if (dev_glob) then
      call mgg_setup(mg_ld)
    else
      ! the coarsest distributed level to the host: e(1:nx,1:ny), cw(1:nx+1,1:ny), cs(1:nx,1:ny+1)
      l = mg_ld; nxc = mg_nx(l); nyc = mg_ny(l)
      CUF_DO2
      do jc = 1, nyc + 1
        do ic = 1, nxc + 1
          if (ic <= nxc .and. jc <= nyc) db_c(ic + (jc - 1) * nxc) = me_(ic, jc, l)
          if (jc <= nyc) db_c(nxc * nyc + ic + (jc - 1) * (nxc + 1)) = mw_(ic, jc, l)
          if (ic <= nxc) db_c(nxc * nyc + (nxc + 1) * nyc + ic + (jc - 1) * nxc) = ms_(ic, jc, l)
        enddo
      enddo
      n = 3 * nxc * nyc + nxc + nyc
#ifdef USE_CUDA
      ierr = cudaMemcpyAsync(hb_c, db_c, int(n), cudaMemcpyDeviceToHost, work_stream)
      ierr = cudaStreamSynchronize(work_stream)
      if (ierr /= 0) error stop 'implicit_fs: mg_setup copy'
#else
      hb_c(1:n) = db_c(1:n)
#endif
      call fsh_setup(int(nxc), int(nyc), hb_c)
    endif
    tm_mgsetup = tm_mgsetup + (wall_time() - t0)
    call tq0()
  end subroutine mg_setup

  subroutine vg_point(kind, p)
    ! a host action inside the V-cycle being recorded: close the stretch's graph, launch it,
    ! do the action, open the next stretch
    integer, intent(in) :: kind, p(8)
#ifdef USE_CUDA
    type(cudaGraph) :: g
    integer :: ierr
    if (vg_n >= VG_MAX) error stop 'implicit_fs: V-cycle recording: too many stretches'
    ierr = cudaStreamEndCapture(work_stream, g)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: end capture'
    vg_n = vg_n + 1
    ierr = cudaGraphInstantiate(vg_exec(vg_n), g, 0)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: instantiate'
    vg_kind(vg_n) = kind; vg_par(:, vg_n) = p
    ierr = cudaGraphLaunch(vg_exec(vg_n), work_stream)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: launch'
    call vg_act(kind, p)
    ierr = cudaStreamBeginCapture(work_stream, 0)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: begin capture'
#endif
  end subroutine vg_point

  subroutine vg_act(kind, p)
    integer, intent(in) :: kind, p(8)
    integer(ip) :: nb(4)
    select case (kind)
    case (1)
      nb = int(p(1:4), ip)
      call ring_mpi_act(nb, p(5), p(6))
    case (2)
      call mgg_act_gather()
    case (3)
      call mgg_act_coarse()
    case (4)
      call mg_act_host()
    end select
  end subroutine vg_act

  subroutine vg_replay()
    integer :: s
#ifdef USE_CUDA
    integer :: ierr
    do s = 1, vg_n
      ierr = cudaGraphLaunch(vg_exec(s), work_stream)
      if (ierr /= 0) error stop 'implicit_fs: V-cycle replay: launch'
      if (vg_kind(s) > 0) call vg_act(vg_kind(s), vg_par(:, s))
    enddo
    n_ring = n_ring + vg_ring_g
#endif
  end subroutine vg_replay

  subroutine mg_act_host()
    ! the host global levels (fs_gpu_coarse 'host'): the rhs copy complete, the host V-cycle
    integer :: ierr, l
    l = mg_ld
#ifdef USE_CUDA
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) error stop 'implicit_fs: vcycle copy down'
#endif
    call tq(T_MTR)
    call fsh_vcycle(int(mg_nx(l)), int(mg_ny(l)), hb_b, hb_x)
  end subroutine mg_act_host

  subroutine mg_vcycle()
    ! zz = one V-cycle applied to rr, from zz = 0 (mg_vcycle of the CPU)
    integer(ip) :: i, j, l, k, nxl, nyl, nx, ny, ic, jc
    integer :: ierr, n
    logical :: rec
    n_vc = n_vc + 1
    rec = .false.
#ifdef USE_CUDA
    if (o_vgr) then
      if (vg_ready) then
        call vg_replay()
        n_gath = n_gath + 1
        return
      endif
      ! record this V-cycle (its kernels run as the stretches' graphs)
      rec = .true.; vg_n = 0; vg_ring0 = n_ring
      vg_point_p => vg_point
      fs_capt = .true.; vg_rec = .true.
      ierr = cudaStreamBeginCapture(work_stream, 0)
      if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: begin capture'
    endif
#endif
    nx = decomp%nx; ny = decomp%ny
    if (g2f) then
      ! fs_gpu_opt 8192: sweep 1 zz0 -> zzb (no exchange: zz0's halo is zero, as zz's after zero2),
      ! then alternately zzb -> zz, zz -> zzb, each after the ring of the array it reads (ends in zz)
      do k = 1, fs_mg_nu
        if (k == 1) then
          call sweep_fine2(1, zz0, zzb)
        else if (mod(k, 2) == 0) then
          call ring_fine_a(zzb)
          call sweep_fine2(1, zzb, zz)
        else
          call ring_mg(0)
          call sweep_fine2(1, zz, zzb)
        endif
      enddo
    else
    call zero2(zz)
    do k = 1, fs_mg_nu
      if (k > 1) call ring_mg(0)
      call smooth_fine(1)
    enddo
    endif
    call ring_mg(0)
    call resid_restrict(0)
    do l = 1, mg_ld - 1
      if (g2f) then
        do k = 1, fs_mg_nu
          if (k == 1) then
            call sweep_c2(l, 1, mz_, 1, mxb_, int(l))
          else if (mod(k, 2) == 0) then
            call ring_mg_a(l, mxb_)
            call sweep_c2(l, 1, mxb_, int(l), mx_, int(l))
          else
            call ring_mg(l)
            call sweep_c2(l, 1, mx_, int(l), mxb_, int(l))
          endif
        enddo
      else
      call zero_plane(l)
      do k = 1, fs_mg_nu
        if (k > 1) call ring_mg(l)
        call smooth_c(l, 1)
      enddo
      endif
      call ring_mg(l)
      call resid_restrict(l)
    enddo
    if (dev_glob) then
      call mgg_vcycle(mg_ld)
      n_gath = n_gath + 1
    else
      ! the global levels on the host
      l = mg_ld; nxl = mg_nx(l); nyl = mg_ny(l); n = int(nxl * nyl)
      CUF_DO2
      do j = 1, nyl
        do i = 1, nxl
          db_v(i + (j - 1) * nxl) = mb_(i, j, l)
        enddo
      enddo
#ifdef USE_CUDA
      ierr = cudaMemcpyAsync(hb_b, db_v, n, cudaMemcpyDeviceToHost, work_stream)
#else
      hb_b(1:n) = db_v(1:n)
#endif
      if (vg_rec) then
        call vg_point(4, [0, 0, 0, 0, 0, 0, 0, 0])
      else
        call mg_act_host()
      endif
      n_gath = n_gath + 1
#ifdef USE_CUDA
      ierr = cudaMemcpyAsync(db_v, hb_x, n, cudaMemcpyHostToDevice, work_stream)
      if (ierr /= 0) error stop 'implicit_fs: vcycle copy up'
#else
      db_v(1:n) = hb_x(1:n)
#endif
      call tq(T_MCO)
      CUF_DO2
      do j = 0, mgy + 1
        do i = 0, mgx + 1
          mx_(i, j, l) = 0.0_rp
          if (i >= 1 .and. i <= nxl .and. j >= 1 .and. j <= nyl) mx_(i, j, l) = db_v(i + (j - 1) * nxl)
        enddo
      enddo
    endif
    do l = mg_ld - 1, 1, -1
      call prolong(l)
      if (g2f) then
        do k = 1, fs_mg_nu                ! alternately mx_ -> mxb_, mxb_ -> mx_ (ends in mx_)
          if (mod(k, 2) == 1) then
            call ring_mg(l)
            call sweep_c2(l, 2, mx_, int(l), mxb_, int(l))
          else
            call ring_mg_a(l, mxb_)
            call sweep_c2(l, 2, mxb_, int(l), mx_, int(l))
          endif
        enddo
        cycle
      endif
      do k = 1, fs_mg_nu
        call ring_mg(l)
        call smooth_c(l, 2)
      enddo
    enddo
    call prolong(0)
    if (g2f) then
      do k = 1, fs_mg_nu                  ! alternately zz -> zzb, zzb -> zz (ends in zz)
        if (mod(k, 2) == 1) then
          call ring_mg(0)
          call sweep_fine2(2, zz, zzb)
        else
          call ring_fine_a(zzb)
          call sweep_fine2(2, zzb, zz)
        endif
      enddo
    else
    do k = 1, fs_mg_nu
      call ring_mg(0)
      call smooth_fine(2)
    enddo
    endif
#ifdef USE_CUDA
    if (rec) then
      ! the last stretch (no action after it)
      call vg_end()
      vg_rec = .false.; fs_capt = .false.; vg_ready = .true.
      ! the recording V-cycle counted every ring once (host actions in vg_act, graph rings where
      ! they were issued); a replay counts only the host actions, so it adds the graph rings
      vg_ring_g = n_ring - vg_ring0 - count(vg_kind(1:vg_n) == 1 .or. vg_kind(1:vg_n) == 2)
      if (talk) write(*,'(1x,A,I4,A)') 'IMPLICIT_FS: V-cycle recorded as', vg_n, ' graphs (fs_gpu_opt 128)'
    endif
#endif
  end subroutine mg_vcycle

  subroutine vg_end()
#ifdef USE_CUDA
    type(cudaGraph) :: g
    integer :: ierr
    ierr = cudaStreamEndCapture(work_stream, g)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: end capture (last)'
    vg_n = vg_n + 1
    ierr = cudaGraphInstantiate(vg_exec(vg_n), g, 0)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: instantiate (last)'
    vg_kind(vg_n) = 0; vg_par(:, vg_n) = 0
    ierr = cudaGraphLaunch(vg_exec(vg_n), work_stream)
    if (ierr /= 0) error stop 'implicit_fs: V-cycle recording: launch (last)'
#endif
  end subroutine vg_end

  subroutine zero_plane(l)
    integer, intent(in) :: l
    integer(ip) :: i, j, mxx, myy, ll
    mxx = mgx; myy = mgy; ll = l
    CUF_DO2
    do j = 0, myy + 1
      do i = 0, mxx + 1
        mx_(i, j, ll) = 0.0_rp
      enddo
    enddo
  end subroutine zero_plane

  subroutine smooth_fine(mode)
    ! one sweep on the fine level (halo current): 'rbgs' red then black
    ! (mode 1) or black then red (mode 2), colours tile-local; 'jacobi' damped
    integer, intent(in) :: mode
    integer(ip) :: i, j, nx, ny, c, col, p, hx, k
    real(rp) :: w
    nx = decomp%nx; ny = decomp%ny
    if (jac_sm) then
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          wk2(i, j) = dinv(i, j) * (rr(i, j) + cu(i, j) * zz(i - 1, j) + cu(i + 1, j) * zz(i + 1, j) &
               + cv(i, j) * zz(i, j - 1) + cv(i, j + 1) * zz(i, j + 1))
        enddo
      enddo
      w = om
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          zz(i, j) = (1.0_rp - w) * zz(i, j) + w * wk2(i, j)
        enddo
      enddo
      call tq(T_MSM)
      return
    endif
    do c = 1, 2
      col = mod(c + mode, 2)
      if (o_cc) then
        ! fs_gpu_opt 64: only the colour's cells (mod(i + j, 2) = p), i = i0(j) + 2 k
        p = mod(1 + col, 2); hx = (nx + 1) / 2
        CUF_DO2
        do j = 1, ny
          do k = 0, hx - 1
            i = 1 + mod(1 + j + p, 2) + 2 * k
            if (i <= nx) &
                 zz(i, j) = dinv(i, j) * (rr(i, j) + cu(i, j) * zz(i - 1, j) + cu(i + 1, j) * zz(i + 1, j) &
                 + cv(i, j) * zz(i, j - 1) + cv(i, j + 1) * zz(i, j + 1))
          enddo
        enddo
        cycle
      endif
      CUF_DO2
      do j = 1, ny
        do i = 1, nx
          if (mod(i + j + col, 2) == 1) &
               zz(i, j) = dinv(i, j) * (rr(i, j) + cu(i, j) * zz(i - 1, j) + cu(i + 1, j) * zz(i + 1, j) &
               + cv(i, j) * zz(i, j - 1) + cv(i, j + 1) * zz(i, j + 1))
        enddo
      enddo
    enddo
    call tq(T_MSM)
  end subroutine smooth_fine

  subroutine smooth_c(l, mode)
    integer, intent(in) :: l, mode
    integer(ip) :: i, j, nxl, nyl, c, col, ll, p, hx, k
    real(rp) :: w
    nxl = mg_nx(l); nyl = mg_ny(l); ll = l
    if (jac_sm) then
      CUF_DO2
      do j = 1, nyl
        do i = 1, nxl
          mt_(i, j, ll) = md_(i, j, ll) * (mb_(i, j, ll) + mw_(i, j, ll) * mx_(i - 1, j, ll) &
               + mw_(i + 1, j, ll) * mx_(i + 1, j, ll) + ms_(i, j, ll) * mx_(i, j - 1, ll) &
               + ms_(i, j + 1, ll) * mx_(i, j + 1, ll))
        enddo
      enddo
      w = om
      CUF_DO2
      do j = 1, nyl
        do i = 1, nxl
          mx_(i, j, ll) = (1.0_rp - w) * mx_(i, j, ll) + w * mt_(i, j, ll)
        enddo
      enddo
      call tq(T_MSM)
      return
    endif
    do c = 1, 2
      col = mod(c + mode, 2)
      if (o_cc) then
        p = mod(1 + col, 2); hx = (nxl + 1) / 2
        CUF_DO2
        do j = 1, nyl
          do k = 0, hx - 1
            i = 1 + mod(1 + j + p, 2) + 2 * k
            if (i <= nxl) &
                 mx_(i, j, ll) = md_(i, j, ll) * (mb_(i, j, ll) + mw_(i, j, ll) * mx_(i - 1, j, ll) &
                 + mw_(i + 1, j, ll) * mx_(i + 1, j, ll) + ms_(i, j, ll) * mx_(i, j - 1, ll) &
                 + ms_(i, j + 1, ll) * mx_(i, j + 1, ll))
          enddo
        enddo
        cycle
      endif
      CUF_DO2
      do j = 1, nyl
        do i = 1, nxl
          if (mod(i + j + col, 2) == 1) &
               mx_(i, j, ll) = md_(i, j, ll) * (mb_(i, j, ll) + mw_(i, j, ll) * mx_(i - 1, j, ll) &
               + mw_(i + 1, j, ll) * mx_(i + 1, j, ll) + ms_(i, j, ll) * mx_(i, j - 1, ll) &
               + ms_(i, j + 1, ll) * mx_(i, j + 1, ll))
        enddo
      enddo
    enddo
    call tq(T_MSM)
  end subroutine smooth_c

  subroutine resid_restrict(l)
    ! residual of level l (0: fine) summed over each aggregate into b of l+1
    integer, intent(in) :: l
    integer(ip) :: i, j, ic, jc, nxc, nyc, nxf, nyf, ll
    real(rp) :: s, r
    ll = l
    nxf = mg_nx(l); nyf = mg_ny(l); nxc = mg_nx(l + 1); nyc = mg_ny(l + 1)
    if (l == 0) then
      CUF_DO2
      do jc = 1, nyc
        do ic = 1, nxc
          s = 0.0_rp
          do j = 2 * jc - 1, min(2 * jc, nyf)
            do i = 2 * ic - 1, min(2 * ic, nxf)
              r = act(i, j) * ((ar(i, j) + rb(i, j)) * zz(i, j) &
                   + cu(i + 1, j) * (zz(i, j) - zz(i + 1, j)) + cu(i, j) * (zz(i, j) - zz(i - 1, j)) &
                   + cv(i, j + 1) * (zz(i, j) - zz(i, j + 1)) + cv(i, j) * (zz(i, j) - zz(i, j - 1)))
              r = act(i, j) * (rr(i, j) - r)
              s = s + r
            enddo
          enddo
          mb_(ic, jc, 1) = s
        enddo
      enddo
    else
      CUF_DO2
      do jc = 1, nyc
        do ic = 1, nxc
          s = 0.0_rp
          do j = 2 * jc - 1, min(2 * jc, nyf)
            do i = 2 * ic - 1, min(2 * ic, nxf)
              r = 0.0_rp
              if (me_(i, j, ll) > 0.0_rp) r = mb_(i, j, ll) - (me_(i, j, ll) * mx_(i, j, ll) &
                   + mw_(i, j, ll) * (mx_(i, j, ll) - mx_(i - 1, j, ll)) &
                   + mw_(i + 1, j, ll) * (mx_(i, j, ll) - mx_(i + 1, j, ll)) &
                   + ms_(i, j, ll) * (mx_(i, j, ll) - mx_(i, j - 1, ll)) &
                   + ms_(i, j + 1, ll) * (mx_(i, j, ll) - mx_(i, j + 1, ll)))
              s = s + r
            enddo
          enddo
          mb_(ic, jc, ll + 1) = s
        enddo
      enddo
    endif
    call tq(T_MTR)
  end subroutine resid_restrict

  subroutine prolong(l)
    ! x of level l (0: zz) += x of level l+1 on its active cells
    integer, intent(in) :: l
    integer(ip) :: i, j, nxf, nyf, ll
    ll = l
    nxf = mg_nx(l); nyf = mg_ny(l)
    if (l == 0) then
      CUF_DO2
      do j = 1, nyf
        do i = 1, nxf
          zz(i, j) = zz(i, j) + act(i, j) * mx_((i + 1) / 2, (j + 1) / 2, 1)
        enddo
      enddo
    else
      CUF_DO2
      do j = 1, nyf
        do i = 1, nxf
          if (me_(i, j, ll) > 0.0_rp) mx_(i, j, ll) = mx_(i, j, ll) + mx_((i + 1) / 2, (j + 1) / 2, ll + 1)
        enddo
      enddo
    endif
    call tq(T_MTR)
  end subroutine prolong

  subroutine ring_mg(l)
    ! one-cell halo of the smoother's x on level l (0: zz, the fine level
    ! as ring; >= 1: the coarse plane, rows 1..ny, columns 1..nx, a missing
    ! neighbour gives a zero halo: the CPU's mg_exch)
    integer, intent(in) :: l
    if (l == 0) then
      call ring(1, zz, fdum, fdum, fdum, fdum, fdum, .true.)
      call tq(T_MEX)
      return
    endif
    call ring_mg_a(l, mx_)
  end subroutine ring_mg

  subroutine ring_fine_a(x)
    ! ring_mg(0) of the array x (zzb)
    real(rp), FSD intent(inout) :: x(decomp%is:decomp%ie, decomp%js:decomp%je)
    call ring(1, x, fdum, fdum, fdum, fdum, fdum, .true.)
    call tq(T_MEX)
  end subroutine ring_fine_a

  subroutine sweep_fine2(mode, xs, xd)
    ! fs_gpu_opt 8192: one red-black sweep of the fine level in one kernel, from xs (halo current) into
    ! xd, as gsweep2 (fs_mgg) for the global levels: a second-colour cell forms the new values of its
    ! four first-colour neighbours with smooth_fine's expression (xs's halo value where the neighbour
    ! is a halo cell), then its own; bit-identical to smooth_fine
    integer, intent(in) :: mode
    real(rp), FSD intent(in) :: xs(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), FSD intent(inout) :: xd(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, nx, ny, p1
    real(rp) :: vw, ve, vs, vn
    nx = decomp%nx; ny = decomp%ny; p1 = mod(1 + mode, 2)
    CUF_DO2
    do j = 1, ny
      do i = 1, nx
        if (mod(i + j + p1, 2) == 1) then
          xd(i, j) = dinv(i, j) * (rr(i, j) + cu(i, j) * xs(i - 1, j) &
                 + cu(i + 1, j) * xs(i + 1, j) + cv(i, j) * xs(i, j - 1) &
                 + cv(i, j + 1) * xs(i, j + 1))
        else
          if (i > 1) then
            vw = dinv(i - 1, j) * (rr(i - 1, j) + cu(i - 1, j) * xs(i - 2, j) &
                 + cu(i, j) * xs(i, j) + cv(i - 1, j) * xs(i - 1, j - 1) &
                 + cv(i - 1, j + 1) * xs(i - 1, j + 1))
          else
            vw = xs(i - 1, j)
          endif
          if (i < nx) then
            ve = dinv(i + 1, j) * (rr(i + 1, j) + cu(i + 1, j) * xs(i, j) &
                 + cu(i + 2, j) * xs(i + 2, j) + cv(i + 1, j) * xs(i + 1, j - 1) &
                 + cv(i + 1, j + 1) * xs(i + 1, j + 1))
          else
            ve = xs(i + 1, j)
          endif
          if (j > 1) then
            vs = dinv(i, j - 1) * (rr(i, j - 1) + cu(i, j - 1) * xs(i - 1, j - 1) &
                 + cu(i + 1, j - 1) * xs(i + 1, j - 1) + cv(i, j - 1) * xs(i, j - 2) &
                 + cv(i, j) * xs(i, j))
          else
            vs = xs(i, j - 1)
          endif
          if (j < ny) then
            vn = dinv(i, j + 1) * (rr(i, j + 1) + cu(i, j + 1) * xs(i - 1, j + 1) &
                 + cu(i + 1, j + 1) * xs(i + 1, j + 1) + cv(i, j + 1) * xs(i, j) &
                 + cv(i, j + 2) * xs(i, j + 2))
          else
            vn = xs(i, j + 1)
          endif
          xd(i, j) = dinv(i, j) * (rr(i, j) + cu(i, j) * vw + cu(i + 1, j) * ve &
               + cv(i, j) * vs + cv(i, j + 1) * vn)
        endif
      enddo
    enddo
    call tq(T_MSM)
  end subroutine sweep_fine2

  subroutine sweep_c2(l, mode, xs, ks, xd, kd)
    ! fs_gpu_opt 8192: sweep_fine2 for the distributed level l (smooth_c's expression), from
    ! xs(:, :, ks) into xd(:, :, kd)
    integer, intent(in) :: l, mode, ks, kd
    real(rp), FSD intent(in) :: xs(0:mgx + 1, 0:mgy + 1, *)
    real(rp), FSD intent(inout) :: xd(0:mgx + 1, 0:mgy + 1, *)
    integer(ip) :: i, j, nxl, nyl, ll, p1, s1, d1
    real(rp) :: vw, ve, vs, vn
    nxl = mg_nx(l); nyl = mg_ny(l); ll = l; p1 = mod(1 + mode, 2); s1 = ks; d1 = kd
    CUF_DO2
    do j = 1, nyl
      do i = 1, nxl
        if (mod(i + j + p1, 2) == 1) then
          xd(i, j, d1) = md_(i, j, ll) * (mb_(i, j, ll) + mw_(i, j, ll) * xs(i - 1, j, s1) &
                 + mw_(i + 1, j, ll) * xs(i + 1, j, s1) + ms_(i, j, ll) * xs(i, j - 1, s1) &
                 + ms_(i, j + 1, ll) * xs(i, j + 1, s1))
        else
          if (i > 1) then
            vw = md_(i - 1, j, ll) * (mb_(i - 1, j, ll) + mw_(i - 1, j, ll) * xs(i - 2, j, s1) &
                 + mw_(i, j, ll) * xs(i, j, s1) + ms_(i - 1, j, ll) * xs(i - 1, j - 1, s1) &
                 + ms_(i - 1, j + 1, ll) * xs(i - 1, j + 1, s1))
          else
            vw = xs(i - 1, j, s1)
          endif
          if (i < nxl) then
            ve = md_(i + 1, j, ll) * (mb_(i + 1, j, ll) + mw_(i + 1, j, ll) * xs(i, j, s1) &
                 + mw_(i + 2, j, ll) * xs(i + 2, j, s1) + ms_(i + 1, j, ll) * xs(i + 1, j - 1, s1) &
                 + ms_(i + 1, j + 1, ll) * xs(i + 1, j + 1, s1))
          else
            ve = xs(i + 1, j, s1)
          endif
          if (j > 1) then
            vs = md_(i, j - 1, ll) * (mb_(i, j - 1, ll) + mw_(i, j - 1, ll) * xs(i - 1, j - 1, s1) &
                 + mw_(i + 1, j - 1, ll) * xs(i + 1, j - 1, s1) + ms_(i, j - 1, ll) * xs(i, j - 2, s1) &
                 + ms_(i, j, ll) * xs(i, j, s1))
          else
            vs = xs(i, j - 1, s1)
          endif
          if (j < nyl) then
            vn = md_(i, j + 1, ll) * (mb_(i, j + 1, ll) + mw_(i, j + 1, ll) * xs(i - 1, j + 1, s1) &
                 + mw_(i + 1, j + 1, ll) * xs(i + 1, j + 1, s1) + ms_(i, j + 1, ll) * xs(i, j, s1) &
                 + ms_(i, j + 2, ll) * xs(i, j + 2, s1))
          else
            vn = xs(i, j + 1, s1)
          endif
          xd(i, j, d1) = md_(i, j, ll) * (mb_(i, j, ll) + mw_(i, j, ll) * vw + mw_(i + 1, j, ll) * ve &
               + ms_(i, j, ll) * vs + ms_(i, j + 1, ll) * vn)
        endif
      enddo
    enddo
    call tq(T_MSM)
  end subroutine sweep_c2

  subroutine ring_mg_a(l, x)
    ! ring_mg of level l >= 1 of the plane array x (mx_ or mxb_)
    integer, intent(in) :: l
    real(rp), FSD intent(inout) :: x(0:mgx + 1, 0:mgy + 1, *)
    integer(ip) :: nxl, nyl, m, mx, w, e, s, n, nb(4), ll
    nxl = mg_nx(l); nyl = mg_ny(l); ll = l
    nb = [decomp%nbr_w, decomp%nbr_e, decomp%nbr_s, decomp%nbr_n]
    mx = max(nxl, nyl)
    if (all(nb == NBR_NONE)) then
      CUF_DO1
      do m = 1, mx
        if (m <= nyl) then
          x(0, m, ll) = 0.0_rp; x(nxl + 1, m, ll) = 0.0_rp
        endif
        if (m <= nxl) then
          x(m, 0, ll) = 0.0_rp; x(m, nyl + 1, ll) = 0.0_rp
        endif
      enddo
      call tq(T_MEX)
      return
    endif
    if (ipc_on) then
      call ipc_ring_plane(x, mgx, mgy, int(ll), int(nxl), int(nyl))
      n_ring = n_ring + 1
      call tq(T_MEX)
      return
    endif
    CUF_DO1
    do m = 1, mx
      if (m <= nyl) then
        rg_sw(m) = x(1, m, ll); rg_se(m) = x(nxl, m, ll)
      endif
      if (m <= nxl) then
        rg_ss(m) = x(m, 1, ll); rg_sn(m) = x(m, nyl, ll)
      endif
    enddo
    call ring_mpi(nb, int(nyl), int(nxl))
    w = nb(1); e = nb(2); s = nb(3); n = nb(4)
    CUF_DO1
    do m = 1, mx
      if (m <= nyl) then
        if (w == NBR_NONE) then
          x(0, m, ll) = 0.0_rp
        else if (w == NBR_SELF) then
          x(0, m, ll) = rg_se(m)
        else
          x(0, m, ll) = rg_rw(m)
        endif
        if (e == NBR_NONE) then
          x(nxl + 1, m, ll) = 0.0_rp
        else if (e == NBR_SELF) then
          x(nxl + 1, m, ll) = rg_sw(m)
        else
          x(nxl + 1, m, ll) = rg_re(m)
        endif
      endif
      if (m <= nxl) then
        if (s == NBR_NONE) then
          x(m, 0, ll) = 0.0_rp
        else if (s == NBR_SELF) then
          x(m, 0, ll) = rg_sn(m)
        else
          x(m, 0, ll) = rg_rs(m)
        endif
        if (n == NBR_NONE) then
          x(m, nyl + 1, ll) = 0.0_rp
        else if (n == NBR_SELF) then
          x(m, nyl + 1, ll) = rg_ss(m)
        else
          x(m, nyl + 1, ll) = rg_rn(m)
        endif
      endif
    enddo
    call tq(T_MEX)
  end subroutine ring_mg_a

end module fs_mg

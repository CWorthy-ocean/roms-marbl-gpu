#include "mem_attr.inc"
module cdr_out
  ! CDR output (CPU cdr_output.F90, the C-Star fields): &cdr_output_settings;
  ! files <root>_cdr.<date>.nc (the CPU naming, his_io output_fname) of
  ! nrpf_cdr records every output_period_cdr, in double, with the CPU's
  ! variables and attributes: zeta, T, S, ALK, DIC, their ALT_CO2 twins,
  ! PO4, SiO3, the ALK/DIC vertical integrals, 3-D pH, the air-sea CO2
  ! fluxes, surface pCO2, carbonate ion and saturation fields, saturation
  ! depths, surface chlorophyll, phytoplankton carbon over the top 100 m,
  ! the instantaneous thickness-weighted ALK/DIC, the CDR sources (with
  ! cdr_source) and the averaged thickness-weighted ALK/DIC.
  ! With wrt_cdr_avg the CPU's sums: every step after the step c0*sum + x
  ! (c0 = 0 on the window's first step); the fields MARBL changes only on
  ! its steps are added once, weighted by the steps they were held, before
  ! each MARBL step and at the write; at the write sum/navg. Not on the GPU
  ! yet: cdr_monthly_averages (stops at start-up).
  use kinds
  use his_io, only: output_fname, refdatestr
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use implicit_fs_gpu, only: fs_global_atts
  use io_server, only: IO_KIND_CDROUT, io_init, io_init8, io_stage8_begin, io_stage8_2d, io_stage8_3d, &
                       io_stage8_3d_pt, io_stage8_end
  use comm_mod, only: comm_rank, compute_barrier
  use passive_mod, only: ntp, tp, passive_name
  use marbl_host, only: D_FGCO2, D_FGCO2_ALT, D_PCO2, D_PCO2_ALT
  use marbl_sflux_mod, only: sf_diag
  use marbl_interior_mod, only: it_ph, it_pha, it_co3, it_co3a, it_co3sat, it_co3sata, it_zsatc, it_zsata
  use marbl_gpu, only: marbl_on, marbl_timestep_ratio, marbl_need_zsat
  use cdr_src_mod, only: cdr_on, cdr_src_out, cdr_src_out3d, cdr_mode, CDR_MODE_3D
  use cdr_io, only: cdr_source
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: read_cdr_output_namelist, set_cdr_output_root, init_cdr_output, wrt_cdr

  real(rp) :: output_period_cdr = 3600.0_rp
  integer  :: nrpf_cdr = 4
  logical  :: wrt_cdr_avg = .false., cdr_monthly_averages = .false., do_cdr_output = .false.
  namelist /cdr_output_settings/ output_period_cdr, nrpf_cdr, wrt_cdr_avg, cdr_monthly_averages, do_cdr_output

  ! 2-D sums a2(:, :, n) and 3-D sums a3(:, :, :, n)
  integer, parameter :: Z2 = 1, IZA = 2, IZD = 3, IZAA = 4, IZDA = 5, CHL = 6, C100 = 7, FG = 8, FGA = 9, &
                        PC = 10, PCA = 11, ZSA = 12, ZSC = 13, N2 = 13
  integer, parameter :: TT = 1, TS = 2, AA = 3, DD = 4, AAL = 5, DAL = 6, P4 = 7, SI = 8, HA = 9, HD = 10, &
                        HAA = 11, HDA = 12, PHT = 13, PHA = 14, C3 = 15, C3A = 16, CSA = 17, CSC = 18, &
                        SA = 19, SAA = 20, SD = 21, SDA = 22, N3 = 22
  logical :: on = .false., wsrc = .false., marbl_first = .true.
  character(len=512) :: root = './output', fname = ''
  integer :: record = 0, nfile = 0, navg = 0, nheld = 0
  real(rp) :: output_time = 0.0_rp, avg_begin_time = 0.0_rp
  integer(ip) :: ialk, idic, ialka, idica, ipo4, isio3, ispchl, idiatchl, idiazchl, ispc, idiatc, idiazc
  integer :: ir0, ir1, jr0, jr1, gxi, geta, gxi_u, geta_v
  real(rp), WORK_ATTR allocatable :: a2(:,:,:), a3(:,:,:,:), w2(:,:), w3(:,:,:)

contains

  subroutine read_cdr_output_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    rewind(unit)
    read(unit, nml=cdr_output_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'CDR_OUTPUT_SETTINGS: ', trim(iomsg_text(msg))
      error stop 'read_cdr_output_namelist'
    endif
    if (ios < 0) do_cdr_output = .false.
  end subroutine read_cdr_output_namelist

  subroutine set_cdr_output_root(out_root)
    character(len=*), intent(in) :: out_root
    root = trim(out_root)
    if (len_trim(root) == 0) root = './output'
  end subroutine set_cdr_output_root

  integer(ip) function pidx(name) result(itp)
    character(len=*), intent(in) :: name
    integer(ip) :: m
    itp = 0
    do m = 1, ntp
      if (trim(passive_name(m)) == trim(name)) itp = m
    enddo
    if (itp == 0) then
      write(*,'(3a)') 'init_cdr_output: CDR output needs the MARBL tracer ', trim(name), ' (not in this run)'
      error stop 'init_cdr_output: missing MARBL tracer'
    endif
  end function pidx

  subroutine init_cdr_output()
    integer :: nxy, is, ie, js, je, nz
    on = .false.
    if (.not. do_cdr_output) return
    if (.not. marbl_on) error stop 'init_cdr_output: the CDR output needs MARBL'
    if (cdr_monthly_averages) error stop 'init_cdr_output: cdr_monthly_averages is not on the GPU model yet'
    if (nrpf_cdr < 1) error stop 'init_cdr_output: nrpf_cdr must be >= 1'
    ialk = pidx('ALK'); idic = pidx('DIC'); ialka = pidx('ALK_ALT_CO2'); idica = pidx('DIC_ALT_CO2')
    ipo4 = pidx('PO4'); isio3 = pidx('SiO3')
    ispchl = pidx('spChl'); idiatchl = pidx('diatChl'); idiazchl = pidx('diazChl')
    ispc = pidx('spC'); idiatc = pidx('diatC'); idiazc = pidx('diazC')
    wsrc = cdr_source
    marbl_need_zsat = .true.
    ir0 = merge(0, 1, decomp%edge_w); ir1 = merge(decomp%nx + 1, decomp%nx, decomp%edge_e)
    jr0 = merge(0, 1, decomp%edge_s); jr1 = merge(decomp%ny + 1, decomp%ny, decomp%edge_n)
    gxi = merge(int(decomp%gnx), int(decomp%gnx) + 2, decomp%periodic_x)
    gxi_u = merge(int(decomp%gnx), int(decomp%gnx) + 1, decomp%periodic_x)
    geta = merge(int(decomp%gny), int(decomp%gny) + 2, decomp%periodic_y)
    geta_v = merge(int(decomp%gny), int(decomp%gny) + 1, decomp%periodic_y)
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    allocate(w2(is:ie, js:je), w3(is:ie, js:je, nz))
    if (wrt_cdr_avg) then
      allocate(a2(is:ie, js:je, N2), a3(is:ie, js:je, nz, N3))
    else
      allocate(a2(is:ie, js:je, 1), a3(is:ie, js:je, nz, 1))
    endif
    call zero_all(.true.)
    nxy = (ir1 - ir0 + 1) * (jr1 - jr0 + 1)
    call io_init(1, 1)
    call io_init8((N3 + 4) * nxy * nz + N2 * nxy, nxy * nz)
    record = nrpf_cdr; nfile = 0; navg = 0; nheld = 0; output_time = 0.0_rp; marbl_first = .true.
    on = .true.
    if (comm_rank() == 0) write(*,'(a,es12.4,a,i0,a,l2,a,l2)') 'cdr output: period ', output_period_cdr, ' s, ', &
         nrpf_cdr, ' records per file, averages', wrt_cdr_avg, ', CDR sources', wsrc
  end subroutine init_cdr_output

  subroutine zero_all(with_scratch)
    ! The sums (and the scratch fields) to 0, on the device.
    logical, intent(in) :: with_scratch
    integer :: i, j, k, n, is, ie, js, je, nz, n2, n3, fs
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    n2 = size(a2, 3); n3 = size(a3, 4); fs = merge(1, 0, with_scratch)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = js, je
        do i = is, ie
          do n = 1, n3
            a3(i, j, k, n) = 0.0_rp
          enddo
          if (k == 1) then
            do n = 1, n2
              a2(i, j, n) = 0.0_rp
            enddo
          endif
          if (fs == 1) then
            w3(i, j, k) = 0.0_rp
            if (k == 1) w2(i, j) = 0.0_rp
          endif
        enddo
      enddo
    enddo
  end subroutine zero_all

  subroutine sum_step(c0)
    ! The CPU's calc_average, every step: c0*sum + x over the written points
    ! (C_TOT_100m on the interior).
    real(rp), intent(in) :: c0
    integer :: i, j, k, nz, i0, i1, j0, j1, nn, kn, col, nx, ny
    real(rp) :: alk, dic, alka, dica, h, sa, sd, saa, sda
    integer(ip) :: ia, id, iaa, ida, ip4, isi, ic1, ic2, ic3
    nz = decomp%nz; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1; nn = nnew; kn = knew; nx = decomp%nx; ny = decomp%ny
    ia = ialk; id = idic; iaa = ialka; ida = idica; ip4 = ipo4; isi = isio3
    ic1 = ispchl; ic2 = idiatchl; ic3 = idiazchl
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          col = tpcol(i, j)
          alk = real(tp(ia, col, k, nn), rp); dic = real(tp(id, col, k, nn), rp)
          alka = real(tp(iaa, col, k, nn), rp); dica = real(tp(ida, col, k, nn), rp)
          h = Hz(i, j, k)
          a3(i, j, k, TT) = c0 * a3(i, j, k, TT) + t(i, j, k, nn, itemp)
          a3(i, j, k, TS) = c0 * a3(i, j, k, TS) + t(i, j, k, nn, isalt)
          a3(i, j, k, P4) = c0 * a3(i, j, k, P4) + real(tp(ip4, col, k, nn), rp)
          a3(i, j, k, SI) = c0 * a3(i, j, k, SI) + real(tp(isi, col, k, nn), rp)
          a3(i, j, k, AA) = c0 * a3(i, j, k, AA) + alk
          a3(i, j, k, DD) = c0 * a3(i, j, k, DD) + dic
          a3(i, j, k, AAL) = c0 * a3(i, j, k, AAL) + alka
          a3(i, j, k, DAL) = c0 * a3(i, j, k, DAL) + dica
          a3(i, j, k, HA) = c0 * a3(i, j, k, HA) + alk * h
          a3(i, j, k, HD) = c0 * a3(i, j, k, HD) + dic * h
          a3(i, j, k, HAA) = c0 * a3(i, j, k, HAA) + alka * h
          a3(i, j, k, HDA) = c0 * a3(i, j, k, HDA) + dica * h
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        col = tpcol(i, j)
        a2(i, j, Z2) = c0 * a2(i, j, Z2) + zeta(i, j, kn)
        a2(i, j, CHL) = c0 * a2(i, j, CHL) + real(tp(ic1, col, nz, nn), rp) + real(tp(ic2, col, nz, nn), rp) &
                        + real(tp(ic3, col, nz, nn), rp)
        sa = 0.0_rp; sd = 0.0_rp; saa = 0.0_rp; sda = 0.0_rp
        do k = 1, nz
          h = Hz(i, j, k)
          sa = sa + real(tp(ia, col, k, nn), rp) * h
          sd = sd + real(tp(id, col, k, nn), rp) * h
          saa = saa + real(tp(iaa, col, k, nn), rp) * h
          sda = sda + real(tp(ida, col, k, nn), rp) * h
        enddo
        a2(i, j, IZA) = c0 * a2(i, j, IZA) + sa
        a2(i, j, IZD) = c0 * a2(i, j, IZD) + sd
        a2(i, j, IZAA) = c0 * a2(i, j, IZAA) + saa
        a2(i, j, IZDA) = c0 * a2(i, j, IZDA) + sda
      enddo
    enddo
    call ctot100(.false., c0)
  end subroutine sum_step

  subroutine ctot100(inst, c0)
    ! Phytoplankton carbon (sp + diat + diaz) over the top 100 m of each
    ! interior column (the CPU's c_tot_100m): into w2 (inst) or the sum.
    logical, intent(in) :: inst
    real(rp), intent(in) :: c0
    integer :: i, j, k, nz, nx, ny, nn, col, md
    real(rp) :: tot, cs, fr
    integer(ip) :: i1, i2, i3
    nz = decomp%nz; nx = decomp%nx; ny = decomp%ny; nn = nnew; md = merge(1, 0, inst)
    i1 = ispc; i2 = idiatc; i3 = idiazc
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = 1, ny
      do i = 1, nx
        col = tpcol(i, j)
        tot = 0.0_rp; cs = 0.0_rp
        k = nz
        do while ((tot < 100.0_rp) .and. (k >= 1))
          cs = cs + (real(tp(i1, col, k, nn), rp) + real(tp(i2, col, k, nn), rp) + real(tp(i3, col, k, nn), rp)) &
                    * Hz(i, j, k)
          tot = tot + Hz(i, j, k)
          k = k - 1
        enddo
        if (tot >= 100.0_rp) then
          k = k + 1
          fr = (tot - 100.0_rp) / Hz(i, j, k)
          cs = cs - fr * (real(tp(i1, col, k, nn), rp) + real(tp(i2, col, k, nn), rp) + real(tp(i3, col, k, nn), rp)) &
                    * Hz(i, j, k)
        endif
        if (md == 1) then
          w2(i, j) = cs
        else
          a2(i, j, C100) = c0 * a2(i, j, C100) + cs
        endif
      enddo
    enddo
  end subroutine ctot100

  subroutine sum_sources()
    ! The CPU's add_cdr_source_sums: this step's ALK, ALT ALK, DIC and ALT
    ! DIC sources added to their sums.
    integer :: is, js
    is = decomp%is; js = decomp%js
    if (cdr_mode == CDR_MODE_3D) then
      call cdr_src_out3d(a3(is, js, 1, SA), 1_ip, 1.0_rp, 1.0_rp, .false.)
      call cdr_src_out3d(a3(is, js, 1, SD), 2_ip, 1.0_rp, 1.0_rp, .false.)
    else
      call cdr_src_out(a3(is, js, 1, SA), ialk, 1.0_rp, 1.0_rp, .false.)
      call cdr_src_out(a3(is, js, 1, SAA), ialka, 1.0_rp, 1.0_rp, .false.)
      call cdr_src_out(a3(is, js, 1, SD), idic, 1.0_rp, 1.0_rp, .false.)
      call cdr_src_out(a3(is, js, 1, SDA), idica, 1.0_rp, 1.0_rp, .false.)
    endif
  end subroutine sum_sources

  subroutine zero_sources()
    integer :: i, j, k, n, is, ie, js, je, nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = js, je
        do i = is, ie
          do n = SA, SDA
            a3(i, j, k, n) = 0.0_rp
          enddo
        enddo
      enddo
    enddo
  end subroutine zero_sources

  subroutine sum_marbl()
    ! The CPU's accumulate_marbl_fields: the MARBL-held fields, weighted by
    ! the steps they were held (c0 = 0 on the window's first addition).
    integer :: i, j, k, nz, i0, i1, j0, j1
    real(rp) :: c0, w
    if (nheld == 0) return
    w = real(nheld, rp)
    c0 = merge(0.0_rp, 1.0_rp, marbl_first)
    nz = decomp%nz; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          a3(i, j, k, PHT) = c0 * a3(i, j, k, PHT) + w * it_ph(i, j, k)
          a3(i, j, k, PHA) = c0 * a3(i, j, k, PHA) + w * it_pha(i, j, k)
          a3(i, j, k, C3) = c0 * a3(i, j, k, C3) + w * it_co3(i, j, k)
          a3(i, j, k, C3A) = c0 * a3(i, j, k, C3A) + w * it_co3a(i, j, k)
          a3(i, j, k, CSA) = c0 * a3(i, j, k, CSA) + w * it_co3sata(i, j, k)
          a3(i, j, k, CSC) = c0 * a3(i, j, k, CSC) + w * it_co3sat(i, j, k)
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        a2(i, j, FG) = c0 * a2(i, j, FG) + w * sf_diag(i, j, D_FGCO2)
        a2(i, j, FGA) = c0 * a2(i, j, FGA) + w * sf_diag(i, j, D_FGCO2_ALT)
        a2(i, j, PC) = c0 * a2(i, j, PC) + w * sf_diag(i, j, D_PCO2)
        a2(i, j, PCA) = c0 * a2(i, j, PCA) + w * sf_diag(i, j, D_PCO2_ALT)
        a2(i, j, ZSA) = c0 * a2(i, j, ZSA) + w * it_zsata(i, j)
        a2(i, j, ZSC) = c0 * a2(i, j, ZSC) + w * it_zsatc(i, j)
      enddo
    enddo
    nheld = 0
    marbl_first = .false.
  end subroutine sum_marbl

  subroutine finalize()
    ! The CPU's finalize_average: sum/navg.
    integer :: i, j, k, n, nz, i0, i1, j0, j1, n3
    real(rp) :: r
    call sum_marbl()
    r = 1.0_rp / real(navg, rp)
    nz = decomp%nz; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1
    n3 = merge(SDA, CSC, wsrc)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          do n = 1, n3
            a3(i, j, k, n) = r * a3(i, j, k, n)
          enddo
          if (k == 1) then
            do n = 1, N2
              a2(i, j, n) = r * a2(i, j, n)
            enddo
          endif
        enddo
      enddo
    enddo
  end subroutine finalize

  subroutine wrt_cdr()
    ! Every step, after the step (the CPU's wrt_cdr).
    real(rp) :: tnow, c0
    logical :: writing
    if (.not. on) return
    tnow = start_time + dt * real(iic - ntstart + 1, rp)
    output_time = output_time + dt
    writing = (output_time >= output_period_cdr)
    if (wrt_cdr_avg) then
      if (navg == 0) avg_begin_time = tnow - dt
      navg = navg + 1
      if (navg == 1) then
        c0 = 0.0_rp
        nheld = 0
        marbl_first = .true.
        if (wsrc) call zero_sources()
      else
        c0 = 1.0_rp
      endif
      call sum_step(c0)
      if (wsrc) call sum_sources()
      nheld = nheld + 1
      ! the MARBL-held fields before the next MARBL step changes them
      if (modulo(iic + 1, marbl_timestep_ratio) == 0) call sum_marbl()
      if (writing) call finalize()
    endif
    if (writing) then
      call write_record(tnow)
      output_time = 0.0_rp
    endif
  end subroutine wrt_cdr

  subroutine inst_h(itr)
    ! w3 = t*Hz of passive tracer itr (the CPU's multiply_by_thickness).
    integer(ip), intent(in) :: itr
    integer :: i, j, k, nz, i0, i1, j0, j1, nn
    integer(ip) :: it
    nz = decomp%nz; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1; nn = nnew; it = itr
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = j0, j1
        do i = i0, i1
          w3(i, j, k) = real(tp(it, tpcol(i, j), k, nn), rp) * Hz(i, j, k)
        enddo
      enddo
    enddo
  end subroutine inst_h

  subroutine inst_iz(itr)
    ! w2 = sum over k of t*Hz of passive tracer itr.
    integer(ip), intent(in) :: itr
    integer :: i, j, k, nz, i0, i1, j0, j1, nn, col
    integer(ip) :: it
    real(rp) :: sm
    nz = decomp%nz; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1; nn = nnew; it = itr
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        col = tpcol(i, j)
        sm = 0.0_rp
        do k = 1, nz
          sm = sm + real(tp(it, col, k, nn), rp) * Hz(i, j, k)
        enddo
        w2(i, j) = sm
      enddo
    enddo
  end subroutine inst_iz

  subroutine inst_chl()
    integer :: i, j, nz, i0, i1, j0, j1, nn, col
    integer(ip) :: c1, c2, c3
    nz = decomp%nz; i0 = ir0; i1 = ir1; j0 = jr0; j1 = jr1; nn = nnew
    c1 = ispchl; c2 = idiatchl; c3 = idiazchl
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, *, 0, work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        col = tpcol(i, j)
        w2(i, j) = real(tp(c1, col, nz, nn), rp) + real(tp(c2, col, nz, nn), rp) + real(tp(c3, col, nz, nn), rp)
      enddo
    enddo
  end subroutine inst_chl

  subroutine zero_w(which)
    integer, intent(in) :: which
    integer :: i, j, k, is, ie, js, je, nz, wh
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz; wh = which
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*, *, 0, work_stream>>>
#endif
    do k = 1, nz
      do j = js, je
        do i = is, ie
          if (wh == 3) w3(i, j, k) = 0.0_rp
          if (wh == 2 .and. k == 1) w2(i, j) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine zero_w

  subroutine write_record(tnow)
    real(rp), intent(in) :: tnow
    character(len=512) :: open_path
    integer :: is, js, nz, kn, nn
    is = decomp%is; js = decomp%js; nz = decomp%nz; kn = knew; nn = nnew
    open_path = ''
    if (record >= nrpf_cdr) then
      nfile = nfile + 1
      record = 0
      call output_fname(root, 'cdr', tnow, fname)
      if (comm_rank() == 0) call create_file(fname)
      call compute_barrier()
      open_path = fname
    endif
    record = record + 1
    if (wrt_cdr_avg) then
      call io_stage8_begin(IO_KIND_CDROUT, record, nfile, tnow, open_path, tavg=(/avg_begin_time, tnow/))
      call s2('zeta', a2(is, js, Z2))
      call s3('temp', a3(is, js, 1, TT)); call s3('salt', a3(is, js, 1, TS))
      call s3('ALK', a3(is, js, 1, AA)); call s3('DIC', a3(is, js, 1, DD))
      call s3('ALK_ALT_CO2', a3(is, js, 1, AAL)); call s3('DIC_ALT_CO2', a3(is, js, 1, DAL))
      call s3('PO4', a3(is, js, 1, P4)); call s3('SiO3', a3(is, js, 1, SI))
      call inst_h(ialk); call s3('hALK', w3)
      call inst_h(ialka); call s3('hALK_ALT_CO2', w3)
      call inst_h(idic); call s3('hDIC', w3)
      call inst_h(idica); call s3('hDIC_ALT_CO2', w3)
      call s3('hALK_avg', a3(is, js, 1, HA)); call s3('hALK_ALT_CO2_avg', a3(is, js, 1, HAA))
      call s3('hDIC_avg', a3(is, js, 1, HD)); call s3('hDIC_ALT_CO2_avg', a3(is, js, 1, HDA))
      call s3('pH', a3(is, js, 1, PHT)); call s3('pH_ALT_CO2', a3(is, js, 1, PHA))
      call s2('FG_CO2', a2(is, js, FG)); call s2('FG_ALT_CO2', a2(is, js, FGA))
      call s2('int_z_ALK', a2(is, js, IZA)); call s2('int_z_DIC', a2(is, js, IZD))
      call s2('int_z_ALK_ALT_CO2', a2(is, js, IZAA)); call s2('int_z_DIC_ALT_CO2', a2(is, js, IZDA))
      call s2('pCO2SURF', a2(is, js, PC)); call s2('pCO2SURF_ALT_CO2', a2(is, js, PCA))
      call s2('zsatarag', a2(is, js, ZSA)); call s2('zsatcalc', a2(is, js, ZSC))
      call s3('CO3', a3(is, js, 1, C3)); call s3('CO3_ALT_CO2', a3(is, js, 1, C3A))
      call s3('co3_sat_arag', a3(is, js, 1, CSA)); call s3('co3_sat_calc', a3(is, js, 1, CSC))
      call s2('Chl_TOT_surf', a2(is, js, CHL)); call s2('C_TOT_100m', a2(is, js, C100))
      if (wsrc) then
        call s3('ALK_source', a3(is, js, 1, SA)); call s3('ALK_ALT_source', a3(is, js, 1, SAA))
        call s3('DIC_source', a3(is, js, 1, SD)); call s3('DIC_ALT_source', a3(is, js, 1, SDA))
      endif
      call io_stage8_end()
      call zero_all(.false.)           ! stream-ordered after the gathers
    else
      call io_stage8_begin(IO_KIND_CDROUT, record, nfile, tnow, open_path)
      call s2('zeta', zeta(is, js, kn))
      call s3('temp', t(is, js, 1, nn, itemp)); call s3('salt', t(is, js, 1, nn, isalt))
      call sp('ALK', ialk); call sp('DIC', idic); call sp('ALK_ALT_CO2', ialka); call sp('DIC_ALT_CO2', idica)
      call sp('PO4', ipo4); call sp('SiO3', isio3)
      call inst_h(ialk); call s3('hALK', w3)
      call inst_h(ialka); call s3('hALK_ALT_CO2', w3)
      call inst_h(idic); call s3('hDIC', w3)
      call inst_h(idica); call s3('hDIC_ALT_CO2', w3)
      call s3('pH', it_ph(is, js, 1)); call s3('pH_ALT_CO2', it_pha(is, js, 1))
      call s2('FG_CO2', sf_diag(is, js, D_FGCO2)); call s2('FG_ALT_CO2', sf_diag(is, js, D_FGCO2_ALT))
      call inst_iz(ialk); call s2('int_z_ALK', w2)
      call inst_iz(ialka); call s2('int_z_ALK_ALT_CO2', w2)
      call inst_iz(idic); call s2('int_z_DIC', w2)
      call inst_iz(idica); call s2('int_z_DIC_ALT_CO2', w2)
      call s2('pCO2SURF', sf_diag(is, js, D_PCO2)); call s2('pCO2SURF_ALT_CO2', sf_diag(is, js, D_PCO2_ALT))
      call s2('zsatarag', it_zsata(is, js)); call s2('zsatcalc', it_zsatc(is, js))
      call s3('CO3', it_co3(is, js, 1)); call s3('CO3_ALT_CO2', it_co3a(is, js, 1))
      call s3('co3_sat_arag', it_co3sata(is, js, 1)); call s3('co3_sat_calc', it_co3sat(is, js, 1))
      call inst_chl(); call s2('Chl_TOT_surf', w2)
      call zero_w(2); call ctot100(.true., 0.0_rp); call s2('C_TOT_100m', w2)
      if (wsrc) then
        call inst_src(ialk, 1); call s3('ALK_source', w3)
        call inst_src(ialka, 0); call s3('ALK_ALT_source', w3)
        call inst_src(idic, 2); call s3('DIC_source', w3)
        call inst_src(idica, 0); call s3('DIC_ALT_source', w3)
      endif
      call io_stage8_end()
    endif
    navg = 0
    write(*,'(a,es14.6,a,i0,a,i0)') 'staged cdr t = ', tnow, '  rec = ', record, '  file ', nfile
  contains
    subroutine s2(name, a)
      character(len=*), intent(in) :: name
      real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
      call io_stage8_2d(name, a, ir0, ir1, jr0, jr1, ir0, jr0)
    end subroutine s2
    subroutine s3(name, a)
      character(len=*), intent(in) :: name
      real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz)
      call io_stage8_3d(name, a, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    end subroutine s3
    subroutine sp(name, itr)
      character(len=*), intent(in) :: name
      integer(ip), intent(in) :: itr
      call io_stage8_3d_pt(name, tp(1, 0, 1, nn), ntp, itr, 1, nz, ir0, ir1, jr0, jr1, ir0, jr0)
    end subroutine sp
    subroutine inst_src(itr, m3)
      ! the CPU's calc_cdr_source: releases (any tracer), 3D forcing (ALK m3 =
      ! 1, DIC m3 = 2; the ALT_CO2 sources stay 0)
      integer(ip), intent(in) :: itr
      integer, intent(in) :: m3
      call zero_w(3)
      if (cdr_mode == CDR_MODE_3D) then
        if (m3 > 0) call cdr_src_out3d(w3, int(m3, ip), 0.0_rp, 0.0_rp, .true.)
      else
        call cdr_src_out(w3, itr, 0.0_rp, 0.0_rp, .true.)
      endif
    end subroutine inst_src
  end subroutine write_record

  subroutine create_file(path)
    ! The CPU's dimensions, variables (double), attributes and order.
    character(len=*), intent(in) :: path
    integer :: ncid, d_t, d_xi, d_xiu, d_eta, d_etav, d_s, id, n
    integer :: d3(4), d2(3)
    n = index(path, '/', back=.true.)
    if (n > 1) call execute_command_line('mkdir -p "'//path(1:n-1)//'"', wait=.true.)
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 0)
    d_xi = nc_def_dim(ncid, 'xi_rho', gxi)
    d_xiu = nc_def_dim(ncid, 'xi_u', gxi_u)
    d_eta = nc_def_dim(ncid, 'eta_rho', geta)
    d_etav = nc_def_dim(ncid, 'eta_v', geta_v)
    d_s = nc_def_dim(ncid, 's_rho', decomp%nz)
    d3 = (/d_xi, d_eta, d_s, d_t/); d2 = (/d_xi, d_eta, d_t/)
    id = nc_def_var(ncid, 'ocean_time', (/d_t/))
    call nc_put_att_text(ncid, 'ocean_time', 'long_name', refdatestr)
    call nc_put_att_text(ncid, 'ocean_time', 'units', 'second')
    call def1('avg_begin_time', (/d_t/), 'Time at beginning of averaging period', 'seconds')
    call def1('avg_end_time', (/d_t/), 'Time at end of averaging period', 'seconds')
    call def1('zeta', d2, 'free-surface elevation', 'meters')
    call def1('temp', d3, 'potential temperature', 'degrees Celsius')
    call def1('salt', d3, 'salinity', 'PSU')
    call def1('ALK', d3, 'Alkalinity', 'meq/m^3')
    call def1('DIC', d3, 'Dissolved Inorganic Carbon', 'mmol/m^3')
    call def1('ALK_ALT_CO2', d3, 'Alkalinity, Alternative CO2', 'meq/m^3')
    call def1('DIC_ALT_CO2', d3, 'Dissolved Inorganic Carbon, Alternative CO2', 'mmol/m^3')
    call def1('PO4', d3, 'Dissolved Inorganic Phosphate', 'mmol/m^3')
    call def1('SiO3', d3, 'Dissolved Inorganic Silicate', 'mmol/m^3')
    call def1('int_z_ALK', d2, 'depth-integrated Alkalinity', 'meters * meq/m^3')
    call def1('int_z_DIC', d2, 'instantaneous depth-integrated Dissolved Inorganic Carbon', 'meters * mmol/m^3')
    call def1('int_z_ALK_ALT_CO2', d2, 'depth-integrated Alkalinity, Alternative CO2', 'meters * meq/m^3')
    call def1('int_z_DIC_ALT_CO2', d2, 'instantaneous depth-integrated Dissolved Inorganic Carbon, Alternative CO2', &
              'meters * mmol/m^3')
    call def1('pH', d3, '3D pH', 'pH')
    call def1('pH_ALT_CO2', d3, '3D pH (alternate CO2)', 'pH')
    call def1('FG_CO2', d2, 'DIC Surface Gas Flux', 'mmol/m^2/s')
    call def1('FG_ALT_CO2', d2, 'DIC Surface Gas Flux, Alternative CO2', 'mmol/m^2/s')
    call def1('pCO2SURF', d2, 'surface pCO2', 'ppmv')
    call def1('pCO2SURF_ALT_CO2', d2, 'surface pCO2, Alternative CO2', 'ppmv')
    call def1('CO3', d3, 'Carbonate Ion Concentration', 'mmol/m^3')
    call def1('CO3_ALT_CO2', d3, 'Carbonate Ion Concentration, Alternative CO2', 'mmol/m^3')
    call def1('co3_sat_arag', d3, 'CO3 concentration at aragonite saturation', 'mmol/m^3')
    call def1('co3_sat_calc', d3, 'CO3 concentration at calcite saturation', 'mmol/m^3')
    call def1('zsatarag', d2, 'Aragonite Saturation Depth', 'm')
    call def1('zsatcalc', d2, 'Calcite Saturation Depth', 'm')
    call def1('Chl_TOT_surf', d2, 'Total surface chlorophyll', 'mg/m3')
    call def1('C_TOT_100m', d2, 'Total biomass over top 100m', 'mmol/m2')
    call def1('hALK', d3, 'instantaneous thickness-weighted Alkalinity', 'meters meq/m^3')
    call def1('hALK_ALT_CO2', d3, 'instantaneous thickness-weighted Alkalinity, Alternative CO2', 'meters meq/m^3')
    call def1('hDIC', d3, 'instantaneous thickness-weighted Dissolved Inorganic Carbon', 'meters mmol/m^3')
    call def1('hDIC_ALT_CO2', d3, 'instantaneous thickness-weighted Dissolved Inorganic Carbon, Alternative CO2', &
              'meters mmol/m^3')
    if (wsrc) then
      call def1('ALK_source', d3, 'ALK source from CDR module', 'meq/s')
      call def1('ALK_ALT_source', d3, 'alt ALK source from CDR module', 'meq/s')
      call def1('DIC_source', d3, 'DIC source from CDR module', 'mmol/s')
      call def1('DIC_ALT_source', d3, 'alt DIC source from CDR module', 'mmol/s')
    endif
    if (wrt_cdr_avg) then
      call def1('hALK_avg', d3, 'time-averaged thickness-weighted Alkalinity', 'meters meq/m^3')
      call def1('hALK_ALT_CO2_avg', d3, 'time-averaged thickness-weighted Alkalinity, Alternative CO2', 'meters meq/m^3')
      call def1('hDIC_avg', d3, 'time-averaged thickness-weighted Dissolved Inorganic Carbon', 'meters mmol/m^3')
      call def1('hDIC_ALT_CO2_avg', d3, 'time-averaged thickness-weighted Dissolved Inorganic Carbon, Alternative CO2', &
                'meters mmol/m^3')
    endif
    call nc_put_att_text(ncid, '', 'title', 'roms_gpu CDR output')
    call fs_global_atts(ncid)   ! barotropic scheme (&implicit_fs)
    call nc_enddef(ncid)
    call nc_close(ncid)
    write(*,'(a,a)') 'created cdr file: ', trim(path)
  contains
    subroutine def1(name, dims, ln, un)
      character(len=*), intent(in) :: name, ln, un
      integer, intent(in) :: dims(:)
      integer :: vid
      vid = nc_def_var(ncid, name, dims)
      call nc_put_att_text(ncid, name, 'long_name', ln)
      call nc_put_att_text(ncid, name, 'units', un)
    end subroutine def1
  end subroutine create_file

end module cdr_out

module feature_guard
  ! Stops the run when the namelist switches on a UCLA ROMS feature that the
  ! GPU model does not have. The GPU reads only the namelist groups it knows,
  ! so without this check a CPU namelist with, say, cdr_source = .true.
  ! would run without CDR forcing and without a warning.
  !
  ! The namelist is scanned as text, not read with namelist I/O: the CPU
  ! groups hold keys the GPU does not declare, which a namelist read would
  ! reject. Every rank scans it (a few kB), so all ranks, writers included,
  ! stop together; world rank 0 prints the report.
  !
  ! Switches that act on the CPU only together with a cpp key the GPU cannot
  ! see (DIAGNOSTICS) print a warning instead of stopping.
  ! (&calc_pflx_settings calc_pflx is on the GPU: pflx_mod; &sponge_tune_settings
  ! ub_tune: sponge_io.)
  use comm_mod, only: comm_world_rank
  implicit none
  private
  public :: check_namelist_features

  ! The namelist text: comments removed, lower case, the contents of quoted
  ! strings replaced by '_' so that a '/' or '&' in a file path cannot end
  ! or start a group.
  character(len=:), allocatable :: txt

contains

  subroutine check_namelist_features(path)
    character(len=*), intent(in) :: path
    integer :: nerr, nwarn
    logical :: talk

    if (.not. load_text(path)) return   ! a missing namelist is reported by load_case
    talk = (comm_world_rank() == 0)
    nerr = 0; nwarn = 0

    ! Not on the GPU: stop.
    if (is_true('river_frc_settings', 'river_source')) &
      call need_off('river_frc_settings', 'river_analytical', 'analytical river forcing')
    call need_off('pipe_frc_settings', 'pipe_source', 'pipe (point source) forcing')
    ! every CDR forcing mode (cdr_frc_settings) is on the GPU
    if (is_true('cdr_output_settings', 'do_cdr_output')) &
      call need_off('cdr_output_settings', 'cdr_monthly_averages', 'monthly CDR averages')
    call need_off('cdr_tracer_output_settings', 'do_cdr_tracer_output', 'CDR tracer output')
    if (is_true('cdr_gas_exch_output_settings', 'do_cdr_gas_exch_output')) &
      call need_off('cdr_gas_exch_output_settings', 'cdr_gas_monthly_averages', 'monthly CDR gas-exchange averages')
    if (is_true('cdr_lite_output_settings', 'do_cdr_lite_output')) &
      call need_off('cdr_lite_output_settings', 'cdr_lite_monthly_averages', 'monthly CDR_LITE averages')
    call need_off('zslice_settings', 'do_zslice', 'z-slice output')
    call need_off('particles_settings', 'floats', 'particles (floats)')
    ! random_output_settings (do_random) is on the GPU: random_out, surface passive tracers
    call need_off('frc_output_settings', 'wrt_frc', 'forcing output')
    call need_off('surf_flx_output_settings', 'wrt_smflx', 'surface momentum flux output')
    call need_off('surf_flx_output_settings', 'wrt_stflx', 'surface tracer flux output')
    call need_off('surf_flx_output_settings', 'wrt_rstflx', 'surface restoring flux output')
    call need_off('basic_output_settings', 'monthly_restarts', 'monthly restarts')
    call need_off('bgc_settings', 'wrt_bgc_dia_avg', 'BGC diagnostic averages output')
    call need_off('tidal_frc_settings', 'ana_tides', 'analytical tides')
    call need_off('tides_settings', 'ana_tides', 'analytical tides')

    ! Inert on the CPU without a cpp key, or nesting-only: warn.
    call warn_on('diagnostics_settings', 'diag_uv', &
                 'momentum budget diagnostics (DIAGNOSTICS cpp key on the CPU) are not on the GPU')
    call warn_on('diagnostics_settings', 'diag_trc', &
                 'tracer budget diagnostics (DIAGNOSTICS cpp key on the CPU) are not on the GPU')
    ! sponge tuning (ub_tune) is on the GPU (sponge_io); its output is not
    call need_off('sponge_tune_settings', 'wrt_sponge', 'sponge tuning output (_spn file)')

    if (talk .and. nerr + nwarn > 0) flush(6)
    deallocate(txt)
    if (nerr > 0) then
      if (talk) then
        write(*,'(a,i0,a)') 'feature_guard: ', nerr, &
             ' namelist setting(s) request features the GPU model does not have.'
        write(*,'(a)') 'feature_guard: set them to .false. to run on the GPU; stopping.'
        flush(6)
      endif
      error stop 'feature_guard: unsupported features requested (see above)'
    endif

  contains

    subroutine need_off(group, key, what)
      character(len=*), intent(in) :: group, key, what
      if (.not. is_true(group, key)) return
      nerr = nerr + 1
      if (talk) write(*,'(7a)') 'ERROR: &', group, ' ', key, ' = .true.: ', what, &
                                ' is not available on the GPU model.'
    end subroutine need_off

    subroutine warn_on(group, key, what)
      character(len=*), intent(in) :: group, key, what
      if (.not. is_true(group, key)) return
      nwarn = nwarn + 1
      if (talk) write(*,'(6a)') 'WARNING: &', group, ' ', key, ' = .true.: ', what
    end subroutine warn_on

  end subroutine check_namelist_features

  logical function load_text(path) result(ok)
    ! Reads the namelist into txt, one blank between lines.
    character(len=*), intent(in) :: path
    character(len=4096) :: line
    character(len=1) :: q
    integer :: unit, ios, i, n
    logical :: inq
    ok = .false.
    open(newunit=unit, file=trim(path), status='old', action='read', iostat=ios)
    if (ios /= 0) return
    txt = ''
    inq = .false.; q = ' '
    do
      read(unit, '(a)', iostat=ios) line
      if (ios /= 0) exit
      n = len_trim(line)
      do i = 1, n
        if (inq) then
          if (line(i:i) == q) then
            inq = .false.
          else
            line(i:i) = '_'
          endif
        elseif (line(i:i) == '''' .or. line(i:i) == '"') then
          inq = .true.; q = line(i:i)
        elseif (line(i:i) == '!') then
          line(i:n) = ' '
          exit
        elseif (line(i:i) >= 'A' .and. line(i:i) <= 'Z') then
          line(i:i) = achar(iachar(line(i:i)) + 32)
        endif
      enddo
      txt = txt // trim(line) // ' '
    enddo
    close(unit)
    ok = .true.
  end function load_text

  logical function is_true(group, key) result(v)
    ! .true. when &group sets key to a true logical value.
    character(len=*), intent(in) :: group, key
    integer :: g0, g1, p, q
    v = .false.
    call find_group(group, g0, g1)
    if (g0 == 0) return
    p = find_key(key, g0, g1)
    if (p == 0) return
    ! value: skip blanks, optional leading '.', then t or f
    q = p
    do while (q <= g1)
      if (txt(q:q) /= ' ') exit
      q = q + 1
    enddo
    if (q > g1) return
    if (txt(q:q) == '.' .and. q < g1) q = q + 1
    v = (txt(q:q) == 't')
  end function is_true

  subroutine find_group(group, g0, g1)
    ! g0..g1: the body of the first &group (after its name, up to the '/'
    ! or the next '&'); g0 = 0 when the group is absent.
    character(len=*), intent(in) :: group
    integer, intent(out) :: g0, g1
    integer :: s, k, e
    g0 = 0; g1 = 0
    s = 1
    do
      k = index(txt(s:), '&' // group)
      if (k == 0) return
      k = s + k - 1
      e = k + 1 + len(group)
      if (e > len(txt)) return
      if (.not. ident_char(txt(e:e))) exit
      s = k + 1
    enddo
    g0 = e
    g1 = len(txt)
    k = index(txt(g0:), '/')
    if (k > 0) g1 = g0 + k - 2
    k = index(txt(g0:g1), '&')
    if (k > 0) g1 = g0 + k - 2
  end subroutine find_group

  integer function find_key(key, g0, g1) result(p)
    ! Position just after the '=' of "key =" inside g0..g1, 0 if absent.
    character(len=*), intent(in) :: key
    integer, intent(in) :: g0, g1
    integer :: s, k, e
    p = 0
    s = g0
    do
      if (s > g1) return
      k = index(txt(s:g1), key)
      if (k == 0) return
      k = s + k - 1
      s = k + 1
      if (k > 1) then
        if (ident_char(txt(k-1:k-1))) cycle
      endif
      e = k + len(key)
      do while (e <= g1)
        if (txt(e:e) /= ' ') exit
        e = e + 1
      enddo
      if (e > g1) return
      if (txt(e:e) /= '=') cycle
      p = e + 1
      return
    enddo
  end function find_key

  logical function ident_char(c)
    character(len=1), intent(in) :: c
    ident_char = (c >= 'a' .and. c <= 'z') .or. (c >= '0' .and. c <= '9') .or. c == '_' .or. c == '%'
  end function ident_char

end module feature_guard

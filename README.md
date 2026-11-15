# roms-marbl-gpu: UCLA ROMS with MARBL on NVIDIA GPUs

roms-marbl-gpu is a GPU version of the [C]Worthy fork of the [UCLA ROMS](https://github.com/CWorthy-ocean/ucla-roms)
regional ocean model, coupled to the [MARBL](https://github.com/marbl-ecosys/MARBL)
ocean biogeochemistry library and to the CDR tracers that [C]Worthy uses to
model marine carbon dioxide removal, such as ocean alkalinity enhancement. It
is written in CUDA Fortran for NVIDIA GPUs. Each time step follows the CPU
model routine by routine, and the GPU model reads the same namelists and
input files as [C]Worthy UCLA ROMS fork (v0.11.0). It is suitable for any standard application
of the ROMS modeling system, as well as for the CDR applications.

> **Experimental.** This is a beta release (v0.1.3-beta, November 2026). It
> has been checked carefully against the CPU model (see Validation below),
> but it has had few users so far. It is not yet part of
> [C-Star](https://github.com/CWorthy-ocean/C-Star), so it cannot yet be
> set up from a C-Star blueprint or run from a C-Star workplan; it is built
> and run by hand, as described below. C-Star support is planned for late
> 2026.

## How this code was written

Nearly all of the code and documentation in this repository was written by
Claude Code, Anthropic's AI coding assistant, in October 2026, working under
the direction of Scott Bachman ([C]Worthy). Scott set the goals, the design
choices and the tests each stage had to pass, and checked the results
against CPU UCLA ROMS; he did not review every line of code. Confidence in
the code therefore rests on the validation below: the GPU model is compared
with the CPU model field by field and step by step, and in long runs against
the CPU model's own run-to-run variability. Please report anything that
looks wrong as an issue.

## What it covers

The GPU model runs the full ROMS time step (dynamics, vertical mixing,
tracer advection and mixing, tides, surface forcing, rivers and open
boundaries), MARBL, passive tracers, and the CDR_LITE alkalinity and DIC
tracers with all CDR release types, on one GPU or on many GPUs across
nodes, optionally with UCLA ROMS's BGC tracer subcycling (see "BGC
tracer subcycling"). It writes the same output files as the CPU model. A
namelist that switches on a CPU feature the GPU model does not have yet
stops the run at start-up and lists the switches.

## Getting started

### Requirements

- NVIDIA GPUs and the NVIDIA HPC SDK (`nvfortran`; developed with 25.9).
- MPI with Fortran bindings. A CUDA-aware MPI is optional: it lets halo
  exchanges go straight from GPU to GPU; otherwise they go through pinned
  host memory.
- NetCDF-C and NetCDF-Fortran built with parallel I/O (parallel HDF5);
  `nf-config` and `nc-config` on the `PATH`.
- Python 3 for `tools/gpu_layout.py` (standard library only); `numpy` and
  `netCDF4` for the comparison tools.

### Get the code

MARBL is a git submodule:

```
git clone --recurse-submodules https://github.com/CWorthy-ocean/gpu-ucla-roms.git roms-marbl-gpu
cd roms-marbl-gpu
# or, in an existing clone:  git submodule update --init
```

### Build

`make gpu` compiles MARBL's Fortran source from the submodule together with the
model and writes `build_gpu/roms_gpu_cuda`. The general form:

```
make gpu FC_GPU=nvfortran \
  FFLAGS_GPU="-O2 -g -cuda -gpu=cc80 -DUSE_CUDA -DPASSIVE_FP64 -Mpreprocess -Mfree -Mnofma -Isrc/shared $(nf-config --fflags)" \
  FFLAGS_MPI="-DUSE_MPI -I<MPI include directory>" \
  LIBS="$(nf-config --flibs) -lnetcdf <MPI Fortran libraries> <rpath flags>"
```

- `-gpu=ccXX` selects the GPU architecture: `cc80` for A100, `cc90` for
  H100 and H200, `cc100` for B200.
- `-DPASSIVE_FP64` stores the MARBL and passive tracers in double
  precision. This is the validated configuration for MARBL runs: with MARBL,
  single precision moves the carbonate saturation depths by far more than
  the CPU model's own run-to-run variability.
- Without the flag, the passive tracers are stored in float32. This suits
  CDR_LITE runs without MARBL:
  - it halves the tracer memory;
  - over 325 days on a Pacific domain, it changed the domain-integrated CDR DIC uptake
    by at most 2e-6 relative (MARBL_GPU_PORT.md);
  - the CDR_LITE output fields are then written as float; with the flag
    they are double, as on the CPU.
- `-Mnofma` turns off fused multiply-add, for reproducible rounding; it
  costs no measurable speed.
- Run one `make` at a time. `make gpu` recompiles the model every time; a
  change of compiler or flags also rebuilds the cached MARBL objects
  (`build_gpu/flags.txt` records the ones they were built with). The model
  prints its version (`git describe`) at start-up.

On NCAR's Derecho (Cray MPICH, which needs `libmpi_gtl_cuda` for GPU-to-GPU
halo exchange):

```
module --force purge
module load ncarenv nvhpc cuda cray-mpich netcdf-mpi
NFF=$(nf-config --fflags); NCP=$(nc-config --prefix)
GTL=/opt/cray/pe/mpich/8.1.32/gtl/lib
make gpu FC_GPU=nvfortran \
  FFLAGS_GPU="-O2 -g -cuda -gpu=cc80 -DUSE_CUDA -DPASSIVE_FP64 -Mpreprocess -Mfree -Mnofma -Isrc/shared ${NFF}" \
  FFLAGS_MPI="-DUSE_MPI -I${CRAY_MPICH_DIR}/include" \
  LIBS="$(nf-config --flibs) -L${NCP}/lib64 -lnetcdf -L${CRAY_MPICH_DIR}/lib -lmpifort -lmpi -L${GTL} -lmpi_gtl_cuda \
        -Wl,-rpath,$(nf-config --prefix)/lib -Wl,-rpath,${CRAY_MPICH_DIR}/lib -Wl,-rpath,${NCP}/lib64 -Wl,-rpath,${GTL}"
```

### Run

```
mpiexec -n <ranks> build_gpu/roms_gpu_cuda run namelist.nml
```

- **Ranks:** `np_xi x np_eta` compute ranks (from the namelist), each
  driving one GPU, plus writer ranks that write the output files. Two ways
  to lay them out:
  - `ROMS_GPU_NWRITERS=w` (writers per node): on every node the last `w`
    ranks are writers and the others compute ranks, so every node gets the
    same number of ranks (SLURM `--nodes=N --ntasks-per-node=g+w
    --gpus-per-node=g`, PBS `select=N:...:mpiprocs=g+w:ngpus=g`) and the
    writers are spread over the nodes. If the compute ranks do not number
    `np_xi x np_eta` the run stops and prints the layout it expected.
  - `ROMS_GPU_NWRITERS` unset: the first `np_xi x np_eta` ranks compute and
    every extra rank is a writer, so the writers sit on the last node.
  - Forcing reader (optional, `ROMS_GPU_FRC_READER=1`): the last rank of
    every node reads the surface forcing and boundary records ahead of the
    model, each record whole and in file order, so that the compute ranks'
    own reads come from the node's page cache instead of waiting on the
    file system at every new record. Results are unchanged (the compute
    ranks read as before). Launch one more rank on every node; the other
    ranks follow the two rules above. Without a spare rank on every node
    the run has no reader and says so.
  `python3 tools/gpu_layout.py namelist.nml [--per-node] [--reader]` prints
  the suggested numbers (compute ranks, writers, total, a PBS select line,
  and with `--per-node` the writers per node); `tools/submit_gpu.sh <job
  script> [PERNODE=1] [READER=0]` does the same and submits a PBS job with
  them, with the forcing reader unless `READER=0`.
- **GPUs:** a compute rank uses GPU (its rank on the node mod the number of
  visible GPUs), so place ranks node by node and make all of a node's GPUs
  visible to every rank. `tools/numa_bind.sh` (launch with `mpiexec
  --cpu-bind none ... tools/numa_bind.sh <exe> run <namelist>`) pins each
  GPU rank to the CPU cores next to its GPU.
- **Environment:** set `MPICH_GPU_SUPPORT_ENABLED=1` to use GPU-to-GPU halo
  exchange (CUDA-aware MPI required); leave it unset otherwise.
- **Namelist:** the same groups and keys as UCLA ROMS, plus `np_xi` and
  `np_eta` (GPU tiles), `nt_bgc` (32 with MARBL) and `nt_passive`. Paths are
  relative to the namelist file. A group with a key the GPU model does not
  know stops the run at start-up, naming the group.
- **Surface forcing:** bulk (COARE 3.0) when the forcing files hold
  `uwnd`, `vwnd`, `Tair` and `qair`, otherwise flux forcing (`sustr`,
  `svstr`, `shflux`, `swflux`, `swrad`). Each field is read either on the
  model grid in the UCLA ROMS layout (rho points `xi_rho x eta_rho`, u
  points `xi_u x eta_rho`, v points `xi_rho x eta_v`) or at half
  resolution, as the CPU model's `interp_bulk_frc` / `interp_flux_frc` /
  `interp_bgc_frc` inputs; the GPU model tells the two apart by the file
  shape and interpolates half-resolution fields as the CPU model does. Any
  other shape stops the run at start-up. Surface restoring (SSS, SST,
  MARBL DIC/ALK), selected on the CPU with cpp keys, is switched on with
  `&surf_frc_settings sflx_corr / qcorrection / cflx_corr` (see
  `NAMELIST_COMPAT`).
- **MARBL output:** with `nt_bgc = 32` the history file carries every
  MARBL tracer, the surface diagnostics PH, pCO2SURF, DpCO2, FG_CO2 and
  their ALT_CO2 twins (PH_ALT_CO2, pCO2SURF_ALT_CO2, FG_ALT_CO2), the
  saturation depths zsatcalc and zsatarag, and the 3-D CO3, CO3_ALT_CO2,
  co3_sat_calc and co3_sat_arag; `_bgc_avg` averages the tracers named in
  `marbl_tracers_to_write`. There is no separate `_bgc` or `_bgc_dia`
  stream.
- **Namelist versions:** `NAMELIST_COMPAT` gives the UCLA ROMS namelist
  each release reads (0.11.0 so far) and the later keys it does not.
- **Output files** are named as UCLA ROMS names them,
  `<output_root_name>_<stream>.<YYYYMMDDHHMMSS>.nc` (streams `his`, `rst`,
  `avg`, `bgc_avg`, `cdr`, `cdrtrc`, `cdrgas`, `rnd`, `uscl`, and
  `<extract_root_name>_bry`), where the date is the model date at which the
  file is started, counted from `&reference_date_settings reference_date`
  (default 2000/1/1). `ocean_time` is in seconds, with `long_name = "Time
  since YYYY/MM/DD"`. Files are whole (not split by rank) NetCDF-4.

`examples/minipac/` is a complete small case (namelist, MARBL settings, PBS
script): 3 days of MiniPac, a small coarse Pacific test domain, with MARBL on one 4-GPU node. The input data are
not included; its README lists the files.

## Changes in v0.1.3-beta (since v0.1.2-beta)

Ports of recent UCLA ROMS fixes and features, and fixes of GPU-only
defects. Runs that use none of the items below give the same results as
v0.1.2-beta, bit for bit.

- Boundary tides are evaluated at the middle of the step for the
  barotropic loop (tidal runs change).
- Rivers deliver exactly the volume of the river file in the split
  barotropic step (predictor transport from the current river volume,
  river faces carry the river transport in the fast loop).
- CDR volume releases conserve volume (S stays constant for a release at
  the ambient salinity) and no longer depend on the tiling.
- Surface restoring switches `sflx_corr`, `qcorrection` (new) and
  `cflx_corr` (new; DIC/ALK restoring applies from the first step, as in
  current UCLA ROMS).
- Flux forcing: the unit conversions of the forcing fields raced with the
  GPU kernels, so flux-forced runs were not reproducible; wind stress in
  the standard u/v layout was misread as half-resolution data; and
  half-resolution inputs were interpolated without the tile offset (wrong
  on more than one GPU) and not as the CPU model does. All fixed; a field
  of unknown shape now stops the run.
- The namelist values `zob` (`&bottom_drag_settings`), `akv_bak` and
  `akt_bak` (`&vertical_mixing_settings`) take effect; before, the GPU
  model used 0.01, 0 and 0 whatever the namelist said. Runs with other
  values change and now match UCLA ROMS.
- `tnu2` (`&tracer_diff2`) is used: each tracer's harmonic diffusivity is
  its `tnu2` entry plus the sponge ramp, per tracer as in UCLA ROMS (one
  value per tracer, in UCLA ROMS's tracer order). Before, the GPU model
  ignored `tnu2` and diffused with the sponge ramp only. Runs with
  non-zero `tnu2` change and now match UCLA ROMS.
- Open-boundary inflow: tracers entering through an open boundary carry
  the boundary data. The horizontal tracer flux of an inflow face on an
  open side gets the term 0.5*(t_boundary - t_interior)*inflow (before,
  only upscaling runs had it), which makes the corrector's flux
  first-order upwind from the boundary value; every tracer, predictor and
  corrector, never at a periodic seam or a wall. This matches the same
  change in UCLA ROMS (upwind-inflow-default). Answers change in every run
  with an open boundary (first next to the boundary); closed and periodic
  runs are unchanged.

## Changes after v0.1.3-beta

- Forcing reader (`ROMS_GPU_FRC_READER=1`; `tools/submit_gpu.sh` adds the
  rank by default, `READER=0` leaves it out, and `tools/gpu_layout.py
  --reader` prints the layout with it): one more rank on every
  node reads each surface forcing and boundary record ahead of the model,
  whole and in file order, into the node's page cache. The compute ranks
  read as before, so results are unchanged, but their reads no longer wait
  on the file system at every new record. On Derecho (A100, forcing files
  not yet read by the node, as in any production run) this removed the
  whole cold-read cost: MiniPac physics-only 52.97 to 40.61 ms per step
  (40.67 with the files already cached), MiniPac with MARBL 74.79 to 68.59
  (68.85), Pac physics-only 109.71 to 102.48 (103.01); the first read of
  each monthly boundary file fell from 157-415 ms to 7 ms. No cost when the
  files are already cached. Without a spare rank on every node the run has
  no reader and says so.
- `tools/numa_bind.sh` binds every compute rank of a multi-node job with
  the forcing reader. Before, on N nodes in the default layout, the last
  N-1 compute ranks (world rank 8 of the 2-node 4 + reader | 4 + writer +
  reader layout) ran unbound: the readers shift the world ranks, and the
  script compared them with the compute rank count `submit_gpu.sh`
  passes. It now counts the readers out as the model does, from the node
  index (`PALS_NODEID` or `SLURM_NODEID`). Other layouts bind as before;
  results are unchanged, bit for bit (binding only).
- The 'wet' tile balance weights land by run: without MARBL a land cell
  counts 0.35 of a wet cell (was 0.10), with MARBL it stays 0.10
  (`&tile_settings tile_land_weight`; a value in the namelist is used as
  given, and a start-up line names the weight and where it came from).
  Kernels that also run over land weigh more without MARBL; MARBL's work
  is per wet column. Median step on Derecho A100s, 0.10 to 0.35: Pac
  physics-only split 106.0 to 102.6 ms (2 nodes, 8x1) and 72.5 to 69.7
  ms (4 nodes, 16x1); Pac with IMEX 156.7 to 153.4 and 110.0 to 107.2
  ms; the N2 nest with IMEX 18.9 to 17.9 ms (1 node, 2x2). MiniPac with
  MARBL would have been 0.4 % slower at 0.35, Pac with MARBL 3 % slower
  at 0.25, hence 0.10 there. Memory per GPU falls with 0.35 (Pac 8x1
  IMEX 29.6 to 26.3 GB on the fullest GPU). A multi-tile 'wet' run
  without MARBL that does not set the key gets new tile cuts. With the
  implicit free surface or IMEX its results change at round-off; with
  the split free surface they can too (Pac: 2e-10 relative after 6
  steps, around a new tile edge that crosses islands).
  `tile_land_weight = 0.10` keeps the old cuts. MARBL runs are
  unchanged, bit for bit.

- Averaged MARBL output as UCLA ROMS writes it from 0.13.0 (ucla-roms
  #400). `_bgc_avg` with `wrt_bgc_his = .false.`: the start-up state is
  no longer a sample of the first window, so each record averages the
  steps of the period that ends at its time. Before, the initial state
  was averaged in and each record was written one step early.
  `_cdr` averages (`wrt_cdr_avg`) with `marbl_timestep` > `dt`: the
  fields MARBL updates only on its steps (pH, CO3, the saturation fields
  and depths, FG_CO2, pCO2SURF and their ALT_CO2 twins) are added right
  after the MARBL step that computed them, for the steps since the
  previous MARBL step. Before, they were added for the steps after it,
  and the first record after a restart differed from an uninterrupted
  run's. `output_period_cdr` must now be a whole multiple of the MARBL
  timestep, or the run stops at start-up, as in UCLA ROMS. Runs without
  these outputs, `_bgc_avg` with `wrt_bgc_his = .true.`, and `_cdr` with
  `marbl_timestep` = `dt` are unchanged, bit for bit.
- IMEX step faster on the GPU, results unchanged bit for bit except for one
  rounding-level lever. On the default fused path the IMEX kernels skip
  deep land as the stock kernels do, the mixing split forms its velocity
  only where the stress kernels read it, each PCG iteration exchanges its
  FP32 search direction in one round through the solution's one-cell
  ring, the solution's halo is kept current inside the solve (no exchange
  at its end), and the deterministic reduction spreads its strided sums
  over 16 blocks on tiles of 65536 columns or more. Restart files hold
  zeros in `imex_pgu`, `imex_pgv` at deep land instead of
  history-dependent values (from the third step of a run, once the
  dry-column skip applies). With `fs_gpu_opt` bit 512 (in the default)
  the PCG's reductions of every fused IMEX solve also use the free
  surface's shared-memory sum when all compute ranks share one node: the
  same rounding-level change as for the free surface on four ranks
  (iteration counts unchanged), none on one or two ranks or several
  nodes. New GPU-only key `imex_gpu_timers` (default 0): 1 and 2 print
  `IMEX_WIN` lines with synchronised windows of the IMEX step (the stage
  windows on the fused path). Median step against the previous code
  (A B B A in one job after a warm-up run, Derecho A100; sargasso and the
  nest with equal tiles, Pac with the default 'wet' tiles and land weight
  0.35): sargasso 2x2 -4.4 %, the California Current nest 2x2 -3.1 %,
  Pac 8x1 on 2 nodes and 16x1 on 4 nodes -2.7 % each.
- Namelists written from UCLA ROMS's current template (after 0.14.0) run:
  `&bgc_settings` `bgc_physics_subcycle` and `bgc_physics_timestep` (and the
  testing keys `bgc_subcycle_pos`, `bgc_subcycle_diag`) and
  `&marbl_biogeochemistry_settings` `marbl_balance` are read. Before, any
  of them stopped the run at the namelist read. `marbl_balance = .true.`
  is ignored with a note (see `NAMELIST_COMPAT`).
- Periodic grids (a grid file of LLm x MMm points): history, average and
  restart files are written, and a restart from them is exact. Before,
  the first record stopped the run with "NetCDF: Start+count exceeds
  dimension bound" (rho-point slabs started one point too far in a
  periodic direction, `io_server.F90` `add_var`), and initial and
  restart files were read with u and v points one point off
  (`case_io.F90`). Non-periodic runs are unchanged, bit for bit.
- Periodic grids can be set in the namelist: the new optional group
  `&periodic_settings` with `ew_periodic` and `ns_periodic`, UCLA ROMS's
  `EW_PERIODIC` and `NS_PERIODIC` (a key left out is `.false.`; one
  direction may be periodic alone). UCLA ROMS ignores the group, so one
  namelist serves both models. With the group, the grid, initial and
  restart files may each use UCLA ROMS's layout (LLm+2 x MMm+2 whatever the
  periodicity; the interior is read) or roms_gpu's (no boundary points in a periodic direction), and
  a file that fits neither stops the run with the sizes expected. Forcing
  fields on a periodic grid are checked per direction and also accept
  roms_gpu's periodic u (v) faces (LLm, MMm). Before, the GPU model only
  inferred the periodicity from the grid file's size, so a UCLA ROMS
  periodic grid ran as a closed basin without a message. Without the group
  the run is unchanged, bit for bit, and a start-up line names the layout
  inferred. Output keeps roms_gpu's layout. Also fixed: `tile_balance =
  'wet'` (the default) stopped the run on a roms_gpu-layout periodic grid
  file with `mask_rho`.
- Surface forcing on a periodic grid: after each record the forcing
  fields' halo in a periodic direction is the periodic copy of the
  interior, as UCLA ROMS's exchange after each forcing read makes it.
  Before, the halo kept the file's boundary points (UCLA ROMS layout) or
  was never set (roms_gpu layout); either way the stress, heat and
  freshwater fluxes next to the periodic seam were wrong (on a doubly
  periodic seamount with flux forcing, 1e8 times the CPU's strict spread
  at the first step; now 1.8 times). Periodic runs with surface forcing
  change; all other runs are unchanged, bit for bit.
- 2x-coarse surface forcing (`interp_flux_frc`, `interp_bulk_frc`,
  `interp_bgc_frc`) on a grid with an odd LLm or MMm: the last fine rho
  column (row) at the east (north) edge and the u (v) end face take the
  last coarse value, as UCLA ROMS does since ucla-roms #439. Before, the
  GPU model reproduced UCLA ROMS's earlier result there, which mixed in a
  zero from beyond the file: 0.75 times the forcing on that boundary
  ring, 0.5625 times at the corner and half the stress on the u (v) end
  face. Grids with even LLm and MMm, periodic directions and forcing on
  the model grid are unchanged, bit for bit.
- 2x-coarse surface forcing on a periodic grid, and the CDR_LITE
  carbonate sensitivities (`ddic_dco2`, `ddic_dalk`) where one of their
  two records is missing: the forcing halo in a periodic direction is the
  periodic copy of the interior, and the 2x-coarse field is periodic too,
  as UCLA ROMS's exchanges make them. Before, the file's coarse boundary
  points were used in the halo and next to it (on a doubly periodic
  seamount whose coarse file does not hold periodic copies there, 4e13
  times the CPU's strict spread at the first step; now 1.7 times). A file
  whose coarse boundary points are periodic copies gives the same results
  as before, and closed grids are unchanged, bit for bit.
- Forcing on the model grid in the GPU-only interior shape (rho LLm x
  MMm, u (LLm+1) x MMm, v LLm x (MMm+1), no boundary points) on a grid
  with more than one tile: the halo between tiles is now filled from the
  neighbours after each record, as UCLA ROMS's exchange after each
  forcing read does (UCLA ROMS reads LLm+2 only). Before, it was never
  set, so the wind stress from bulk forcing and the tidal potential were
  wrong next to the tile seams and the result depended on the tiling (on
  the 39 x 19 closed basin, 1x1 against 2x2: bulk forcing 7e-4 m/s in u
  at the seam after one step, tidal potential 0.37 m in zeta). Tidal
  fields read in that shape get the same fill. Forcing in the LLm+2 or
  2x-coarse shape is unchanged, bit for bit.
- Rivers next to a tile seam: the split step no longer depends on the
  tiling. On a seam the fast loop forms the free surface on the first
  halo ring itself, from the transports one face beyond it, and sets the
  river transport on the river faces there. A river face whose river cell
  lies two cells beyond the seam (outside the tile's river scan) was not
  listed, so the ring took the transport from the extrapolated ubar
  instead. These faces now take the neighbour's river code at start-up.
  Before, moving a tile cut next to a river cell changed the results at
  round-off (Pac 8x1 with `tile_land_weight` 0.10 against 0.35: up to
  2e-10 relative after 6 steps, around a Kuril island with river cells
  next to the cut at xi 527); now they are bit-identical. Runs without
  rivers, and seams with no such face, are unchanged, bit for bit.
- `rho0` (`&rho0_settings`) is read at run time, as UCLA ROMS reads it:
  one binary runs any reference density. Before, it was a build constant
  (`-DROMS_RHO0`, default 1000), and a namelist with another value
  stopped the run with a request to rebuild; `-DROMS_RHO0` is now
  ignored. A namelist without `&rho0_settings` stops the run, as in UCLA
  ROMS. Runs at rho0 = 1000 are unchanged, bit for bit.
- Cyclic river forcing: a `cycle_length` attribute on `river_time` makes
  the river records a climatology, bracketed across the end of the cycle
  as UCLA ROMS does (and as the GPU model already did for the other
  forcing fields). Before, the GPU model ignored the attribute and
  stopped once the model time left the records. Runs whose river file has
  no `cycle_length` are unchanged, bit for bit.
- The `diag: uv2 couple1 IC-jet` line of `step3d_uv2` prints on probe
  steps only (none with `ROMS_GPU_PROBES=0`). Before, it printed on every
  step on every rank whose initial u is non-zero, and each print
  synchronised the device and read a managed array on the host (5 % of
  the step on the N2 nest with IMEX 'modal'). Results are unchanged, bit
  for bit.
- Diagnostic probes are opt-in: they print only when `ROMS_GPU_PROBES` is
  set to a non-zero value, and a start-up line says so. Before, they were
  on unless `ROMS_GPU_PROBES=0`. The fixed probe points are now global
  indices: each rank prints a point only if the point lies inside the
  grid and inside the rank's own tile, and rank 0 names each point
  outside the grid once at start-up. Before, the points were used as
  local tile indices, so with probes on several probes read outside their
  arrays and the sargasso and N2 cases stopped at the first step with a
  segmentation fault. The probes' single-purpose points (the coastal
  `huv`/`flux` cell and the `rim` sites) follow the same rule, and probes
  that print fixed levels (50, 99) print 0 for a level above `nz` instead
  of reading past the array. Runs with `ROMS_GPU_PROBES=0` are unchanged, bit for
  bit, and so are runs with probes on (except for the probe lines). A run
  that left the variable unset now uses the faster kernels of non-probe
  steps in its first 24 steps too; on the closed basin with rivers and
  CDR that changes results at round-off (1e-13 in `DU_avg1`), on sargasso
  and the N2 nest not at all.
- BGC tracer subcycling (`&bgc_settings bgc_physics_subcycle`,
  `bgc_physics_timestep`; the port of UCLA ROMS #428 and #430): the BGC
  tracers are transported, mixed and forced once per window from the
  transports summed over the window, as UCLA ROMS does it, with its checks,
  restart rule, messages and `BGCSUB` window diagnostics. Rivers, CDR
  releases, CDR_LITE, CFLX_CORR and passive tracers are supported;
  `bgc_subcycle_pos` and `bgc_subcycle_diag >= 2` stop the run at
  start-up. Measured: Pac (8 A100) 1.37x faster at k = 4 (window
  of 4 steps with MARBL every 4) and 1.20x at k = 2, MiniPac 1.28x and
  1.18x. Off by default; runs without it are unchanged, bit for bit. See
  "BGC tracer subcycling" below.
- BGC tracer subcycling together with the IMEX internal-wave step
  (`&imex_igw_settings`): the start-up refusal is gone, as UCLA ROMS dev
  809aa24a runs the combination. The window sums IMEX's corrector
  transports, its 'modal' open faces included. The IMEX tracer predictor
  no longer updates the subcycled tracers, as the stock predictor already
  did not; it would have overwritten one of the time levels that hold the
  BGC state between windows. Runs without subcycling are unchanged, bit
  for bit.
- Pressure fluxes for nesting (`&calc_pflx_settings calc_pflx`): the
  depth-integrated internal-tide energy flux `up`/`vp` of the parent,
  written to the child boundary file (`up_<side>`, `vp_<side>`) as UCLA ROMS
  writes it, with the running means `u_slow`, `v_slow`, `p_slow` in the
  restart. The pressure is the internal-tide pressure of Kelly, Nash and
  Kunze (2010) (ucla-roms eae883ee), and the boundary values are taken by
  index (ucla-roms 4b07d30e).
- Sponge tuning for nested children (`&sponge_tune_settings ub_tune`, the
  CPU's SPONGE_TUNE): the inflow nudging of the 3-D open boundary
  conditions (velocity and every tracer) is tuned every step so that the
  child's internal-tide energy flux next to each side follows the parent's
  `up`/`vp` from the boundary files; needs `calc_pflx`. The coefficients
  `ub_west`, `ub_east`, `ub_south`, `ub_north` are kept in the restart,
  and `wrt_sponge` writes the fluxes and coefficients of every open side to
  the `_spn` file.
- CDR_LITE: where one of two bracketing carbonate-sensitivity records is
  missing (0), the gas flux now ramps linearly to zero instead of spiking.
- Implicit free surface (`&implicit_fs fs_mode = 'implicit'`, the port of
  ucla-roms' implicit free surface with its defaults): one generalized-alpha
  barotropic step per baroclinic step (two sub-steps of dt/2 by default),
  reduced to a symmetric positive-definite Helmholtz problem for the
  sea-level change and solved by multigrid-preconditioned conjugate
  gradients, replaces the split-explicit fast loop. Without the group, or
  with `fs_mode = 'split'`, the model is unchanged. The settings are those
  of UCLA ROMS's namelist template: `fs_mode`, `fs_nsub` (2), `fs_rho_inf`
  (0.5, in [0, 1)), `fs_tol` (1e-8) and `fs_verbose` (0). UCLA ROMS's
  solver and development settings are read too, and the GPU adds tuning
  settings (`fs_gpu_gather`, `fs_gpu_coarse`, `fs_gpu_graph`, `fs_gpu_opt`,
  `fs_gpu_timers`); see `NAMELIST_COMPAT`. A solve that does not converge
  stops the run. The restart file holds the scheme's state, so a restart
  is exact, and the output files record the scheme (`fs_mode`,
  `fs_rho_inf`, `fs_nsub`) as global attributes.
  - Cost (Derecho, A100, default settings, no output; measured with this
    branch's implicit free surface before its optimization, 64e3bd2, whose
    results this version reproduces bit for bit): mean time per step,
    implicit against split, sargasso 2x2 (4 GPUs on 1 node, dt 480 s)
    16.5 against 36-39 ms (0.44), Iceland0 2x2 (dt 120 s) 26.0 against
    34-36 ms (0.75), Pac 8x1 (8 GPUs on 2 nodes, dt 900 s, tides, rivers)
    115.6 against 121-135 ms (median 118-119; 0.86-0.96). The free-surface
    window is 5.0 ms per step on the 1-node cases and 16.3 ms on Pac: two
    solves of 4.6-6 iterations each, bound by latency (halo exchanges,
    reductions, V-cycles), so the second sub-step adds a fixed 2.4 ms
    (1 node) to 5.5 ms (Pac) per step. With this branch's latency levers
    (`fs_gpu_opt = 65495`, below; measured at 212c26f, this branch before
    its rebase onto the ucla-roms review fixes, same settings): window 3.4,
    3.6 and 14.8 ms, step 15.0, 24.7 and 114.3 ms (median 14.8, 24.5,
    112.0), 0.41, 0.75 and 0.94 of the split step's median. In 2-day runs
    of the same binary (median of three runs per configuration) the
    implicit step takes 0.42 (sargasso), 0.45 (hawaii), 0.44 (warmpool),
    0.74 (Iceland0, 1 node), 0.91 (MiniPac with MARBL, 1 node) and 0.95
    (Pac, 2 nodes) of the split step.
  - Tiling: the implicit step is not bitwise independent of the tiling.
    Its results differ between tilings at about the level of `fs_tol`:
    sargasso, 20 steps, the rms differences between 1x1, 2x2 and 4x1 GPU
    tilings are 0.3-1.3 times the CPU model's own 2x2 against 4x1
    difference (hbbl 1.8 times), and the GPU against the CPU at the same
    tiling 0.6-1.05 times (hbbl 1.8).
  - Not tested on the GPU: periodic domains, `fs_drag_implicit`, sponge tuning
    (`ub_tune`) together with the implicit free surface, more than two
    nodes. A closed basin and river and CDR volume sources were checked
    during the port's development, not on this branch's gate cases, of
    which only Pac has rivers. Pipe sources and MPI_MASKING are not
    available in the GPU model.
  `fs_gpu_opt` is a bit mask of latency levers (documented in
  `src/device/implicit_fs_data.F90`); the default 65495 turns on all that
  were faster. All but one leave the result bit-identical to
  `fs_gpu_opt = 0`; bit 512 (the solver's sums over the four ranks of one
  node through shared memory) adds them in a different order than
  `MPI_Allreduce`, a rounding-level difference (iteration counts unchanged;
  fields in rms at most 2.1 times the CPU's regular-to-strict build spread,
  at single points up to 3.6 times). A
  solve that meets a NaN or Inf stops the run with a message (the same
  test and message as the implicit free surface of UCLA ROMS).
- IMEX internal-wave step (`&imex_igw_settings`; without the group the
  model is unchanged): the port of UCLA ROMS's IMEX_IGW (ucla-roms
  #433, with the 'modal' open boundary of #434), which treats the
  internal-wave terms of the
  LF-AM3 baroclinic step implicitly (an elliptic solve per stage, Jacobi
  PCG with FP32 storage) so that the baroclinic step can be several times
  longer than the internal-wave limit. It runs on the implicit free
  surface with its defaults. The open boundary is the CPU's 'modal' one
  (the characteristic, per-mode Flather condition of the internal modes,
  implicit in the stage solves), which the default `imex_obc_mode =
  'auto'` selects on a domain whose open sides have 3-D boundary data, as
  on the CPU; 'fixed' keeps the boundary condition's explicit values. Its
  mode build runs on the device at every b_ref rebuild (one thread per
  face column), and the boundary density is the model's EOS. Lateral mixing is a split step from the
  corrected state (`imex_mix_split = 1`). The reference stratification,
  and with it the operator, is rebuilt every 24 model hours
  (`imex_bref_hours`). Restarts are exact (`imex_rst_state = 1`). The
  settings are those of UCLA ROMS's namelist template (`imex_verbose`,
  `imex_bref_hours`, `imex_rst_state`, `imex_obc_mode`,
  `imex_obc_tref`); keys removed from UCLA ROMS on
  2026-11-13 stop the run with a message, and `NAMELIST_COMPAT` gives
  the sed line that deletes them.
  - Checked against UCLA ROMS #433 and #434 (strict build, FP64 solver
    storage, tolerances 1e-12) on sargasso (dt 1440 s, sponge, 20
    steps; 'fixed' and 'modal'), a closed basin with walls and lateral
    viscosity ('fixed', 20 steps) and a California Current nest
    ('modal', 36 steps of 600 s, three boundary settings): identical
    IMEX and free-surface iteration counts. With 'fixed', prognostic
    fields are within 2.1 times the CPU's own regular-to-strict build
    difference and every restart variable within 5.3 times (the
    largest: the surface boundary-layer depth, at a mixing threshold).
    With 'modal', prognostic fields are within 2.9 times on sargasso
    (the stored surface water flux, `fs_qd`, reaches 19 times at one
    step at four interior points around one column whose boundary-layer
    depth sits at a mixing threshold; 2.7 with `fs_gpu_opt` bit 512 off)
    and every restart variable
    within 4.1 times on the nest, except in its last steps with all
    boundary settings at non-defaults: up to 9.5 times, and 36 times at
    single points next to the San Francisco Bay estuary, where the
    CPU's own build difference is largest.
  - Cost (Derecho, A100, no output, from the mean step after the first
    10, reference stratification rebuilt every 24 hours, every scheme
    from the same commit; sargasso and the nest with equal tiles,
    `tile_balance = 'uniform'`; the Pacific basin with the default
    `tile_balance = 'wet'` and land weight 0.35): sargasso 2x2 (1 node)
    1.51 s per simulated day at dt 1440 s against 6.42 for the split
    step at 480 s (0.23) and 2.65 for the implicit free surface alone at
    480 s; the California Current nest (2x2, 1 node, tides, rivers) 2.38
    at 600 s against 11.16 (split, 150 s; 0.21) and 6.61 (implicit,
    150 s); Pac 8x1 (2 nodes, tides, rivers) 7.18 at 1800 s against 10.05
    (split, 900 s; 0.71) and 9.64 (implicit, 900 s). 'modal' adds 0.1 %
    per step to 'fixed' on Pac and about 2 % on the nest.
- Tides (`&tidal_frc_settings`) cost less per step. The per-constituent
  cos/sin of each call go to the device by one asynchronous copy instead
  of through managed arrays that the host wrote and the first tide kernel
  then faulted back (on the California Current nest that kernel took
  0.26-0.29 ms; it now takes 0.01 ms). The second call of a step keeps
  the tidal potential of the first, evaluated at the same time.
  California Current nest with
  IMEX (2x2, 1 node, 15 constituents, potential only): 17.72 to 16.68 ms
  per step (-5.9 %); Pac (8x1, 2 nodes, boundary and potential tides):
  156.4 to 155.9 ms (-0.3 %, within the run-to-run spread). Results are
  unchanged, bit for bit.

## Numerics

The GPU model implements one configuration of UCLA ROMS, the one in common
use with MARBL; other CPU options are not available. In ROMS terms:

- SM09 s-coordinate, Shchepetkin pressure-gradient scheme
- LF-AM3 predictor / corrector time stepping
- Split-explicit barotropic mode (FB AB3-AM4) with the `weight(1:2,1:nfast)` filter, or the implicit free surface (`&implicit_fs`)
- Thickness-aware C4 predictor / U3 corrector vertical advection
- UPSTREAM_TS horizontal tracer advection (`UPSTREAM_TS_LAND_CURV` always on)
- LMD94 interior mixing and KPP surface and bottom boundary layers (SMOOTH_RIG, INT_AT_RHO_POINTS, SMOOTH_HBL, Jwt=1)
- Not available: NHMG, isoneutral mixing, splines, LMD_DDMIX, LMD_CONVEC

## BGC tracer subcycling

`&bgc_settings bgc_physics_subcycle = .true.` transports, mixes and forces
the BGC tracers (every tracer after temperature and salinity: MARBL, the
CDR tracers, passive tracers) once per window of `bgc_physics_timestep`
instead of every time step, as UCLA ROMS does since #428 and #430. The
physics still steps every `dt`; the window uses the corrector transports
and the vertical diffusivity summed over the window, so it conserves each
tracer's content and keeps a uniform tracer uniform. MARBL runs at the end
of a window, at its own `marbl_timestep`. Physics results are unchanged,
bit for bit.

- **Settings.** `bgc_physics_timestep` is the window length in seconds, a
  whole multiple of `dt` that divides `marbl_timestep` (0, the default:
  `marbl_timestep`; without MARBL it must be set). The run stops at
  start-up, naming the nearest valid lengths, if it is not. With MARBL every
  4 steps, a window of 4 steps (k = 4) is the fastest setting, and a window
  of 2 steps (k = 2) is in between; UCLA ROMS's documentation of the feature
  discusses the accuracy of each. Restarts must fall on window boundaries
  (`output_period_rst` a multiple of the window; the run starts on a window
  boundary).
- **Speed** (Derecho A100, 130 steps, no output, forcing reader on,
  against today's setting: MARBL every 4 steps, BGC tracers transported
  every step):

  | case (ms per step, steady mean) | today | k = 2 | k = 4 | speed-up k = 2 / k = 4 |
  |---|---:|---:|---:|---:|
  | MiniPac, 4 A100 (1 node) | 67.89 | 57.36 | 53.04 | 1.18x / 1.28x |
  | MiniPac, implicit free surface | 61.61 | 51.08 | 46.82 | 1.21x / 1.32x |
  | Pac, 8 A100 (2 nodes) | 182.31 | 152.18 | 133.42 | 1.20x / 1.37x |
  | Pac, implicit free surface | 178.12 | | 129.29 | - / 1.38x |
  | Pac with rivers (51 729) | 184.31 | | 135.39 | - / 1.36x |

  The gain is smaller than UCLA ROMS's (1.6-1.9x at k = 4) because the
  BGC transport is a smaller part of a GPU step (about 40 %, against 60 %
  on the CPU). Without the forcing reader the gains are the same (MiniPac
  k = 4: 1.28x).
- **Supported:** rivers, CDR releases in all four modes, the CDR_LITE and
  CFLX_CORR surface fluxes, any number of passive tracers with or without
  MARBL, split and implicit free surface, the IMEX internal-wave step
  (`&imex_igw_settings`, 'modal' or 'fixed' open boundary), any tiling.
- **Not on the GPU yet** (the run stops at start-up): the positivity
  limiter `bgc_subcycle_pos` and its test `bgc_subcycle_diag >= 2`.
  UPSCALING output and `ub_tune` are refused, as in UCLA ROMS.
- **Diagnostics:** `bgc_subcycle_diag = 1` prints UCLA ROMS's `BGCSUB` and
  `BGCSUBNEG` lines for every window (Courant numbers, sub-passes,
  continuity, the shadow uniform tracer, the worst per-tracer budget
  residual, new negatives). The run timers report the window's parts.
- **Differences from UCLA ROMS:** the window is ported routine by routine
  with the CPU's operand order, so the results differ from the CPU's only by
  rounding, as the stock tracer step does: the per-window sub-pass counts
  and Courant numbers equal the CPU's to the printed digits, and GPU minus
  CPU is of the size of the CPU's own strict-minus-regular rounding spread
  (next item). Two GPU-only details: the budget residual is summed in a
  fixed order over columns, so it prints about 10 times smaller than the
  CPU's (both are rounding); and river faces in a tile's outer halo row are
  summed by the rank that needs them, where the CPU exchanges the sums (the
  same values). `marbl_balance` has no effect on the GPU.
- **Validation:** against UCLA ROMS dev 5c8034bf on its `bgc_real_marbl`
  test domain, a closed basin with a river and CDR releases, MiniPac and
  Pac: the per-window sub-pass counts and Courant numbers equal the CPU's to
  the printed digits (on MiniPac over the first 20 hours, after which the
  two models' physics drift apart chaotically; the sub-pass counts stay
  equal over 3 days), GPU minus CPU of the BGC tracers stays within 5 times
  the CPU's strict-minus-regular spread at the maximum and 4 times in rms
  over the first 16 steps (MiniPac: outside the rounding-sensitive spot
  described in AGENTS.md), and the subcycling effect after 1 and 3 days of
  MiniPac (k = 4 minus stock) matches the CPU's: ratio 0.985-1.000,
  correlation 0.977-0.9999 at day 1, relative rms within 0.2 % of the CPU's
  at day 3. Restarts on window boundaries are exact, and results do not
  depend on the tiling (split free surface).
- **With the IMEX internal-wave step:** against UCLA ROMS dev 809aa24a
  (mbg, k = 2, 4, k = 2 with MARBL every 4, implicit free surface,
  'fixed' boundary, forced sub-passes; MiniPac k = 4, 'modal' boundary):
  physics bit-identical with subcycling on and off, the sub-pass counts
  and Courant numbers equal to the CPU's in every window, the effect of
  subcycling equal to the CPU's (ratio 1.0000, correlation 1.000000), and
  an exact restart inside an IMEX operator-rebuild period. The GPU's IMEX
  step at its default settings differs from the CPU's by more than the
  CPU's strict-minus-regular spread, with or without subcycling;
  subcycling does not widen that difference. Like the IMEX step itself,
  the combination is not bitwise tiling-independent (rounding-level
  differences).
- **Testing:** `tools/bgcsub_gates.py` runs the fast gate set on a small
  MARBL case (AGENTS.md, "Testing").

## Validation

A GPU model and a CPU model cannot produce bit-for-bit identical results: the
compilers round arithmetic differently (the CPU build, for example, fuses
multiplies and adds into one rounding step), so the two models differ in the
last digit from the first time step. The question is whether the differences
are only rounding, or whether the GPU model solves slightly different
equations. All the tests below were run on the California Current System
(CCS) at 4 km resolution, 672 x 1344 x 100 grid cells, with rivers, a closed
boundary and coastal alkalinity releases, against UCLA ROMS 0.11.0.
Detailed results of the comparisons are available on request.

**After one time step the differences are rounding.** As a yardstick, we
built the CPU model twice, once with its usual fast floating-point settings
and once with strict ones; the two CPU builds differ only by rounding. GPU
minus CPU is the same size as that difference, everywhere in the domain
(maps below). It stays within a factor of 3 of the yardstick over the first eight
steps, and at every stage inside the time step (checked on a smaller
Iceland domain), in physics, MARBL and
the CDR tracers, including near river mouths and the alkalinity releases. A
line-by-line comparison of the two source codes found the same discrete
equations; the few small differences it found were fixed in v0.1.1-beta.

![Two maps of the CCS domain coloured by the size of the surface-temperature difference after one 450-second time step: GPU minus CPU (left) and CPU strict minus CPU standard floating point (right). Both show the same speckle at 1e-15 to 1e-12 degC.](docs/validation_roundoff.png)

**Over months, the GPU model gives the same CDR outcomes as the CPU model.**
Ocean eddies are chaotic: a rounding-sized difference grows within weeks
into a visibly different eddy field, so after a few weeks any two runs
differ in detail, whichever model made them. We therefore ran each model 20
times, each run starting from the same 1 January 2013 ocean state plus its
own random 1e-6 degC temperature noise, and asked whether the 20 GPU runs
and the 20 CPU runs are drawn from the same distribution of outcomes. Each
run released alkalinity at ten coastal sites from Sitka, Alaska, to Baja California
from 1 February to 1 May.

- The fraction of released alkalinity that becomes CO2 uptake (the uptake
  efficiency a CDR study reports) agrees: for all ten sites together, the
  GPU and CPU means differ by 0.15% on 1 May. At single sites the means
  were shown to be within 1% of each other in 31 of 33 site and date
  combinations; at the other two, the run-to-run spread at those sites is
  too large for 20 runs to settle a 1% question either way. Splitting the
  40 runs into two random groups of 20 gives such cases about as often.
- On 264 measures of uptake, plume size, position and depth, and physics,
  the GPU runs fall among the CPU runs the way further CPU runs would, with
  no sign of a bias or of a different amount of variability.
- Maps of the plumes, sea-surface temperature and sea level show no region
  where the GPU runs depart from the CPU runs.

![Depth-integrated alkalinity plumes on 30 April from four of the ten release sites. Rows: a CPU run and a GPU run from the same initial state, and a CPU run started 1e-6 degC away. At each site the three plumes share their large eddies and differ in detail.](docs/validation_plumes.png)

The plumes on 30 April, 88 days into the release, at four of the ten sites.
The top two rows are a CPU run and a GPU run from exactly the same initial
state; the bottom row is a CPU run started 1e-6 degC away. The GPU run
differs from the CPU run by no more than the second CPU run does.

## Performance

The CCS 4 km runs above, 120 days each, on NCAR's Derecho. The CPU model
ran on 512 AMD EPYC cores (4 CPU nodes); the GPU model on 8 NVIDIA A100
GPUs (40 GB) on 2 GPU nodes. "MARBL" is the full biogeochemistry with the CDR
tracers; "CDR_LITE" is the CDR tracers alone, without MARBL. Times are for
the whole job, including start-up and all output.

| | CPU, MARBL | GPU, MARBL | CPU, CDR_LITE | GPU, CDR_LITE |
|---|---:|---:|---:|---:|
| Hardware | 512 cores | 8 A100 | 512 cores | 8 A100 |
| Nodes | 4 | 2 | 4 | 2 |
| Seconds per time step | 1.010 | 0.177 | 0.355 | 0.078 |
| Wall-clock time for 120 days | 6.49 h | 1.14 h | 2.28 h | 0.51 h |
| Node-hours for 120 days | 26.0 | 2.3 | 9.1 | 1.0 |
| Simulated years per day | 1.2 | 6.9 | 3.5 | 15.6 |
| GPU memory used | | 14-18 of 40 GB | | 6-8 of 40 GB |

Per time step the GPU model is 5.7 times faster with MARBL and 4.6 times
faster without, on half as many nodes; per node, 11 and 9 times. The run
uses each GPU's memory less than half, so it would also fit on fewer GPUs.

## Acknowledgements

roms-marbl-gpu is derived from the ROMS model developed by the CESR-lab group at
UCLA ([CESR-lab/ucla-roms](https://github.com/CESR-lab/ucla-roms)). We thank
them for their collaboration and their contributions to this work.

## License

roms-marbl-gpu is derived from UCLA ROMS (https://github.com/CWorthy-ocean/ucla-roms)
and is distributed under the GNU General Public License, version 3 (see
`LICENSE`). MARBL is distributed under its own license
(`MARBL/LICENSE.txt`).

## Contact

For questions or help, contact scott@cworthy.org.

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
nodes. It writes the same output files as the CPU model. A namelist that
switches on a CPU feature the GPU model does not have yet stops the run at
start-up and lists the switches.

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
  `python3 tools/gpu_layout.py namelist.nml [--per-node]` prints the
  suggested numbers (compute ranks, writers, total, a PBS select line, and
  with `--per-node` the writers per node); `tools/submit_gpu.sh <job script>
  [PERNODE=1]` does the same and submits a PBS job with them.
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

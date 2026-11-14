# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

roms-marbl-gpu is a GPU version of [C]Worthy's fork of UCLA ROMS, with MARBL
biogeochemistry and the CDR tracers, in CUDA Fortran for NVIDIA GPUs
(`nvfortran`). Each time step follows the CPU model (ucla-roms) routine by
routine. The GPU model reads the same namelists and input files and writes the
same output files. It implements one ROMS configuration (README, "Numerics").
There is no `cppdefs.opt`: features the CPU selects with cpp keys are namelist
switches here. One binary serves every case:
`mpiexec -n <ranks> build_gpu/roms_gpu_cuda run namelist.nml`.

Each MPI compute rank drives one GPU and tile (`np_xi x np_eta`); extra writer
ranks run the I/O server (`src/host/io_server.F90`). Arrays are CUDA managed
memory (`src/shared/mem_attr.inc`), loops are CUF kernels on one stream
(`src/shared/cuf_launch.inc`) or explicit kernels, and the host only launches
kernels and does I/O.

## Environment and build

You need the NVIDIA HPC SDK, MPI with Fortran bindings, and parallel
NetCDF-C/Fortran (`nf-config` and `nc-config` on `PATH`). The README's Build
section has the general command and a worked Derecho example:

```bash
git submodule update --init          # MARBL; needed in every new worktree
make gpu FC_GPU=nvfortran \
  FFLAGS_GPU="-O2 -g -cuda -gpu=cc80 -DUSE_CUDA -DPASSIVE_FP64 -Mpreprocess -Mfree -Mnofma -Isrc/shared $(nf-config --fflags)" \
  FFLAGS_MPI="-DUSE_MPI -I<MPI include dir>" \
  LIBS="$(nf-config --flibs) -lnetcdf <MPI Fortran libs> <rpath flags>"
```

- Always pass `FFLAGS_GPU`. The Makefile default lacks `-Mnofma` (reproducible
  rounding, CPU parity) and `-DPASSIVE_FP64` (the validated MARBL setting).
- The binary is `build_gpu/roms_gpu_cuda`; it prints `git describe --dirty` at
  start-up. `make gpu` recompiles the model every time (minutes) and the cached
  MARBL objects when the flags change. Run one `make` at a time per tree.
- Plain `make`, `make tests` and `make run` are the old gfortran host build,
  which does not build (the I/O modules need `marbl_host` and MARBL).
- Some files are compiled on their own or at other levels (`comm.F90`,
  three `implicit_fs_*` files, MARBL's interior module) to avoid nvfortran
  inlining losses, miscompiles and crashes. Read the Makefile comments first.
  Compile a new module with external MPI calls on its own, like `comm.F90`.
- Rank layout: `python3 tools/gpu_layout.py namelist.nml`, or submit with
  `tools/submit_gpu.sh examples/minipac/run_minipac.pbs` (PBS). Set
  `MPICH_GPU_SUPPORT_ENABLED=1` only with a CUDA-aware MPI.

## Testing

There is no CI, no pre-commit configuration and no pytest suite. Changes are
gated by hand with GPU runs. Run the cheapest check that covers your change
first, on the smallest case that exercises it.

- `make gpu` is the only static check. The unit programs in `tests/` and their
  targets (`tests`, `build_gpu/test_marbl_*`) lack modules they use and do not
  build; fix them only if asked.
- **Bit identity.** With a new feature off, the run must be bit-identical to
  the base binary: `python3 tools/cmp_nc.py A B`. Compare restart files with a
  record every step (history is float32, restarts are double). In split mode
  the state is bit-identical on 1x1, 2x2 and 4x1 tiles, and a restart is
  exact (N steps = N/2 + N/2). The implicit free surface is not bitwise
  tiling-independent.
- **CPU parity.** Run the same namelist on ucla-roms built regular and
  `BUILD_MODE=strict` (Intel), with `UPSTREAM_TS_LAND_CURV`, which the GPU
  always has. Compare GPU minus CPU with CPU strict minus CPU regular at the
  same records (max and rms over wet points), on the same tiling: set
  `&tile_settings tile_balance = 'uniform'`, since the default `'wet'` gives
  unequal tiles. For a ported change that alters results, also compare
  dG = GPU(new) - GPU(old) with dC = CPU(new) - CPU(old).
- **Long runs** diverge chaotically: compare GPU minus CPU with a CPU twin
  perturbed by 1e-6 degC (`tools/noise_floor_eval.py` and its MARBL twin).
- **Timing.** With `ROMS_GPU_PROBES=0` and no output, run base and new as
  A B B A in one job on the same nodes, and report the median steady step.
  Do not claim a gain you did not measure.

## Repository layout

`src/shared/` holds kinds, decomposition, MPI (`comm.F90`), streams
(`arch.F90`) and formulas. `src/device/` holds the kernels. `src/host/` holds
the driver, case set-up (`case_io.F90`), input readers, output writers, the I/O
server, the MARBL instance and `feature_guard.F90`. `MARBL/` is a submodule
built unchanged as host code. `tools/` holds layout, comparison and nsys
scripts. `MARBL_GPU_PORT.md` is a dated development log that names a `cases/`
tree not in this repository; check what you take from it against the source.

## Namelist handling

`NAMELIST_COMPAT` lists the ucla-roms namelist each release reads, the later
keys it does not read, and the GPU-only groups and keys. Read it first.

- Each group is read in the module that uses it, with a fixed key list
  (`grep -rn "namelist */<group>/" src`). A group that is present but
  unreadable, for example with an unknown key, stops the run and names the
  group. An absent group keeps its defaults, and groups the GPU does not read
  are ignored. `feature_guard.F90` scans the text and stops on switches for
  CPU features the GPU lacks.
- When ucla-roms adds a key, the GPU must read it or stop clearly. Declare it
  with the CPU default in the GPU reader. Then implement it, or accept its
  "off" value and stop on any other value, naming the key. Update
  `NAMELIST_COMPAT`. Example: GPU `dev` stops on ucla-roms `dev` namelists,
  which add `bgc_physics_subcycle`, `bgc_physics_timestep` (`&bgc_settings`,
  `src/host/avg_out.F90`) and `marbl_balance` (`src/host/marbl_step.F90`).
- A CPU cpp key becomes a GPU-only logical, such as `&surf_frc_settings
  sflx_corr`. A CPU case built with SFLX_CORR needs `sflx_corr = .true.` on the
  GPU, which warns when the forcing holds `sss` and the switch is off.

## Porting from ucla-roms

- Port a named ucla-roms commit, and cite its hash in the commit message.
- Mirror the CPU routine line by line, with the same expressions, operand
  order, loop ranges and update order, so that with `-Mnofma` the GPU rounds
  as the CPU does. `src/device/passive_column.F90`'s header shows the practice.
- Keep runs that do not use the feature bit-identical. Add a new term as its
  own guarded statement, not inside an existing expression.
- Reproduce a CPU quirk instead of fixing it on the GPU alone, and report it
  upstream. Document deliberate differences in the commit and in
  `NAMELIST_COMPAT`.
- Judging parity: the README's yardstick is within 3x of the CPU strict spread
  over the first eight steps. A point at a KPP threshold can reach 10-100x
  while barotropic fields stay at 0.5-2x. Locate any excess first. At an open
  boundary, river mouth, tile edge or masked neighbour, treat it as a scheme
  error until shown otherwise. Accept it only where the CPU's own spread is
  large there, and leave the decision to the user.
- Put each GPU-only speed lever behind its own bit (as `fs_gpu_opt` does),
  bit-identical or documented as rounding-level, so it can be bisected.

## Gotchas

- Kernels run asynchronously. A host loop over a managed array right after a
  kernel races with it unless `sync_arch()` comes first. A kernel is better,
  since host access also migrates pages. A run that is not reproducible but is
  under `CUDA_LAUNCH_BLOCKING=1` has such a race.
- `rho0` is a build constant (`-DROMS_RHO0`, default 1000). A namelist with
  another value stops at start-up.
- Diagnostic probes print in the first steps at fixed points of the
  development grid: set `ROMS_GPU_PROBES=0` (`tools/submit_gpu.sh` does). CUDA
  graphs (`ROMS_GPU_GRAPH=1`) damp zeta; keep them off for validation.
- nvfortran pitfalls (`value` arguments in device code, the inliner, stack
  size) are in `MARBL_GPU_PORT.md`, "Toolchain lessons".
- If a run differs from its own bit-identical repeats, rerun it on other nodes
  before you debug: a faulty GPU node has given silently wrong results. Wrap
  job runs in `timeout`.

## Related repositories

ucla-roms (CWorthy-ocean/ucla-roms) is the CPU reference: its namelist sets the
vocabulary, and ports and parity follow it. MARBL (CWorthy-ocean/MARBL) is a
pinned submodule. roms-tools makes the inputs both models read. C-Star does
not run the GPU model yet (README), so namelist changes here need no C-Star
work.

## Data safety

Never delete or overwrite model output, restart files, scratch directories, or
another user's files without the user's explicit OK, even to start fresh.

## Release notes

There are no release-note files or tooling. Releases are `vX.Y.Z-beta` tags. A
user-visible change adds a bullet to the README's "Changes after <latest tag>"
section, and a namelist change updates `NAMELIST_COMPAT`, whose table maps
each release to the ucla-roms namelist it reads.

## Posting on GitHub

This project follows the [C]Worthy AI policy:
https://github.com/CWorthy-ocean/.github/blob/main/AI_POLICY.md

- You may draft pull request descriptions, bug reports, and summaries of test
  results for the user.
- Do not post replies, review comments, answers, or discussion comments on the
  user's behalf, and do not join a discussion on your own. If asked to, tell
  the user that the project asks for replies in their own words, and give them
  notes to write from.
- There is no pull request template. Recent pull requests use ucla-roms's
  sections (Summary, Breaking Changes, New Features, Bug Fixes, Improvements,
  Miscellaneous, Notes for Reviewers). Add the validation and what was not
  run, and an `AI disclosure:` line: `none`, or `<tool>; <what it did>`.
- Do not add `Co-authored-by`, `Assisted-by`, `Generated-by`, or similar AI
  trailers to commits. Commit subjects are imperative sentences with no prefix.
- The user answers for what you open under their account. Tell them plainly
  what you did not run or could not verify.

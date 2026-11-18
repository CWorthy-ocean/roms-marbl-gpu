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
  `MPICH_GPU_SUPPORT_ENABLED=1` only with a CUDA-aware MPI. Ranks other than
  compute ranks: writers (`src/host/io_server.F90`) and, with
  `ROMS_GPU_FRC_READER=1`, one forcing reader per node
  (`src/host/frc_reader.F90`, the last rank of each node); roles are set in
  `comm.F90` `init_comm`. On more than one node the default layout puts
  writers on every node (`ROMS_GPU_NWRITERS`), and `init_comm` numbers them
  round-robin over the nodes so that each file's writer group spans all
  nodes (a node's buffered Lustre client is the limit for one shared
  file). Keep that order if you touch the writer tables: `io_server` builds
  its groups from consecutive writer indices. Writer and reader ranks hold
  a CUDA context (~0.4 GB, from `MPI_Init` with GPU support) on their first
  visible GPU; `tools/numa_bind.sh` shows each one GPU so they spread. A
  multi-node timing that slows down with more writers per node is most
  likely this (check the `gpumem` lines of the compute ranks).

## Testing

There is no CI, no pre-commit configuration and no pytest suite. Changes are
gated by hand with GPU runs. Run the cheapest check that covers your change
first, on the smallest case that exercises it.

- `make gpu` is the only static check. The unit programs in `tests/` and their
  targets (`tests`, `build_gpu/test_marbl_*`) lack modules they use and do not
  build; fix them only if asked. The exception is
  `build_gpu/test_cdf_direct` (netCDF only, no GPU, about 1 s): it checks the
  O_DIRECT reader `src/host/cdf_direct.F90` against `nf90_get_var` bit for
  bit on files it writes; run it after any change to that module
  (`build_gpu/test_cdf_direct <scratch dir>`). With `--file <path> <vars>` it
  checks a real file. Likewise `build_gpu/test_cdf_write <dir on Lustre>`
  (about 15 s) for the direct writes of the same module: header copy,
  page ownership of 1-4 writers, packing, record count, length, layout
  read-back, all against `nf90_get_var`; run it in a default-layout
  directory and in one with `lfs setstripe -c 48 -S 16M`.
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
- **Timing.** With probes off (the default) and no output, run base and new as
  A B B A in one job on the same nodes, and report the median steady step.
  Do not claim a gain you did not measure. The first run of a job is often
  slower (input files not yet cached): leave it out, or start with a warm-up
  run of the same length (a short one does not read the later forcing
  records).
  `fs_gpu_timers` and `imex_gpu_timers` (1, 2) give synchronised windows of
  the free surface and of the IMEX step; results are unchanged.
- **BGC tracer subcycling** (`src/device/bgc_subcycle.F90`,
  `bgc_window.F90`): `tools/bgcsub_gates.py` runs the fast gate set on a
  small MARBL case, as UCLA ROMS's pytest cases do (built on its
  `bgc_real_marbl` test domain, 39 x 19 x 10, where each run takes seconds):
  k = 2, k = 4, k = 4 implicit, k = 2 with MARBL every 4 steps, an exact
  restart, the three rejected window lengths, a restart period that is not a
  multiple of the window and a start inside a window, the IMEX internal-wave
  step (`xoff`, `xk4`, `xk2m4`, `xk4i` and a restart at step 8 inside an
  operator rebuild period, 'modal' boundary), and 2x2 against 1x1 with
  `--tiles2x2` (with and without IMEX; the IMEX solve is not bitwise
  tiling-independent, so that pair is held to 1e-12). `setup --nml <GPU namelist of the case> --exe <binary>
  --dir <dir>` writes the run directories, `run --dir <dir>` runs them inside
  a GPU job, `check --dir <dir> [--ref <dir of the base binary or an earlier
  commit>]` prints one PASS/FAIL line per gate (G-RUN, G-PHYS, G-BUD, G-RST,
  G-TILE, G-MSG, and G-REF bit identity with the reference runs). After a
  change to the window kernels, also run MiniPac k = 4 (two sub-passes per
  window on some levels) and a case with rivers and CDR releases, and compare
  with UCLA ROMS as above (the `BGCSUB` lines' m and Courant numbers must
  equal the CPU's).
- **IMEX restart files** hold zeros in `imex_pgu`, `imex_pgv` at deep land
  (the fused path skips it). Coastal land faces there still depend on the
  run's history, so a restart comparison lists land-only differences of
  these two variables apart.

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
  `NAMELIST_COMPAT`. Examples: `bgc_subcycle_pos` (`&bgc_settings`,
  `src/host/avg_out.F90`) is read, and `.true.` stops in
  `src/device/bgc_subcycle.F90` with "not on the GPU yet"; `marbl_balance`
  (`src/host/marbl_step.F90`) has no effect on the GPU, so `.true.` is
  ignored with a start-up note.
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
  - MiniPac has one such spot, on the shelf slope off New Guinea at j 58-66,
    i 428-436 (0-based rho-grid indices). N^2 is near zero in the upper
    levels and the shear is weak, so a 1e-16 difference in `bvf` moves the LMD
    Richardson-number mixing, and the difference grows about 10x per step:
    salt reaches 800x the strict spread by step 4 while the rest of the domain
    stays at 0.5-3x. The CPU does the same from a rounding-level perturbation
    (see `adiabatic_diff` in `src/shared/formulas.F90`), so it is not a GPU
    error. Judge MiniPac outside these columns, at every level (the
    difference spreads below the upper 20 levels), against the strict spread.
    Inside them, compare with a CPU run whose initial temperature is perturbed
    by 1e-15 relative (pass at 3x), or exclude the columns and say so.
- Put each GPU-only speed lever behind its own bit (as `fs_gpu_opt` does),
  bit-identical or documented as rounding-level, so it can be bisected. A
  bit-identical lever may become plain code once its gates pass, if the
  history keeps it in a commit of its own (the IMEX levers: one commit per
  lever, then one that removes their key).

## Gotchas

- Kernels run asynchronously. A host loop over a managed array right after a
  kernel races with it unless `sync_arch()` comes first. A kernel is better,
  since host access also migrates pages. A run that is not reproducible but is
  under `CUDA_LAUNCH_BLOCKING=1` has such a race.
- `rho0` (`&rho0_settings`) is read at run time into `formulas`' managed
  `rho0` (`case_io` `set_rho0`), which the EOS and pressure functions read
  in kernels. It is not a `parameter`: do not use it in constant
  expressions.
- Diagnostic probes are off unless `ROMS_GPU_PROBES` is non-zero; then they
  print in the first 24 steps at fixed global points of the development grid
  (`time_state.F90`), on the rank that owns each point. A new probe reads the
  rank-local `iprb_*`/`jprb_*` (or `iprb_x`/`jprb_x`), checks them against the
  tile range, and checks any fixed level against nz.
  CUDA graphs (`ROMS_GPU_GRAPH=1`) damp zeta; keep them off for validation.
- nvfortran pitfalls (`value` arguments in device code, the inliner, stack
  size) are in `MARBL_GPU_PORT.md`, "Toolchain lessons".
  Two more from `imex_igw.F90` (nvfortran 25.9): module device functions
  called from its CUF kernels stopped NVVM with "invalid forward reference
  to function ... with wrong type" (defined before or after the kernel),
  so the `imex_corr_adv` kernels write the arithmetic out; and editing one
  format string there made `fort2` abort with a malloc assertion in the
  full `make gpu`, while the file alone compiled. Build every commit.
  From `imex_pcm.inc` (the `imex_pc = 'modes'` port): a global kernel there
  that read `imex_red`'s device array `red_part` crashed the host inside
  `cudaLaunchKernel` (SIGSEGV, no CUDA error); the same pass as a CUF kernel
  runs. Small edits of the file twice left a runtime routine undeclared in
  the LLVM module (`pgf90_man_dealloc03_i8` for one `deallocate` of three
  managed arrays, `pgf90_dev_memset4v` for an integer device-array
  assignment): one array per `deallocate`, and a zeroing kernel, compiled.
  A compile of `imex_igw.F90` alone against the modules of a full build
  (about 1 min on a build node) catches these before the 12-min build.
  It does not catch the `fort2` abort ("munmap_chunk(): invalid pointer"
  in the full `make gpu`): the `imex_corr_adv_clip` port met it again with
  an internal write (`write(cbuf, fmt) ...`) in `check_settings`; with a
  plain message in its place the build passed, and so did the same code
  with more added elsewhere. It depends on the whole file, not on one
  construct: try small variants of the new code in separate builds.
- nvfortran 25.9 at -O2 (host code) dropped the sign extension of
  `int(x, int32)` for an `integer(int8)` x inside
  `ior(ishft(int(b0, int32), 8), ...)`: shorts read by `cdf_direct` came out
  65536 too large. `cdf_unpack` builds the unsigned value and subtracts
  65536; keep byte-level integer code under a bit-for-bit unit test.
  nvfortran 25.9 also has no `c_ssize_t` (use `c_long` on LP64 Linux).
- `comm.F90` (compiled on its own; the main command reads `comm_mod.mod`):
  adding module variables, a `public` name and two procedures in the middle
  of the module broke the full `make gpu` in `imex_igw.F90` (LLVM `opt`:
  "expected function name" on a `declare` with a garbage name; three
  builds, with and without `findloc`). The same code with every addition
  at the end of its section (variables after the last variable, a new
  `public` line after the last one, procedures before `end module`)
  built. Append to `comm_mod`; do not insert.
- `io_server.F90`: moving every internal procedure of `io_server_run`
  (`announce`, `open_for`) out to module level made `fort2` abort with
  "munmap_chunk(): invalid pointer" in `imex_igw.F90` of the full `make
  gpu` (five builds; module variables appended at the end of their
  section, the new module procedures and a new `write_direct` each built
  on their own). Keeping `open_for`, `write_ready` and `evict` internal
  to `io_server_run` built. Bisect such aborts with throwaway branches
  built in parallel from `git archive` copies.
- The writers take in later records while they write one (`io_progress`,
  `ROMS_GPU_IO_PROGRESS=0` turns it off): a command handled there may only
  queue a completed record (`ready_q`), never write one, and only when a
  record slot is free; the main loop writes the queue in order and is the
  only place that evicts. Keep it that way: a write started from inside
  another write would break the group's barrier order.
- CDF5 output is written by the writer ranks with `O_DIRECT`, each 4-KiB
  page by one writer (`cdf_direct`, `io_server` `write_direct`). Rules
  that the benchmarks behind it gave (do not relax them without measuring
  on Lustre with several writers per node on two nodes):
  1. While the writers write a file nothing else touches it: no netCDF
     handle (its classic layer rewrites whole blocks and takes client
     locks), no buffered `pwrite`/`pread`, no `stat` or `lseek` to the
     end, no `ftruncate`, no Lustre group lock. Direct writes on Lustre
     take no client lock; anything that does stalls them. Buffered edge
     pages next to direct writes hung a benchmark; a truncate while a
     writer held a group lock deadlocked; a group lock turned the lockless
     path off (half the rate). The header comes from netCDF through a
     scratch file `<file>.hdr` that `nc_close` copies with `O_DIRECT`; the
     length is set by writer 0 after every writer has closed.
  2. Shared files need a wide stripe: 32-48 OSTs of 4-16 MiB. 1-MiB
     stripes (what `lfs setstripe -c N` gives without `-S`) cut every
     write into 1-MiB RPCs (a third of the rate); Derecho's default
     progressive layout puts a file's first 16 GiB on 4 OSTs.
  3. No fill, and the header is never redefined: unwritten bytes read as 0.
  4. Every page has exactly one writer and is written whole; a page
     partly covered by a record (the previous record's tail, a variable
     the record does not write) is read with `O_DIRECT` first. A change to
     the segment or ownership logic needs `test_cdf_write` and a
     comparison by value with the output of the code before the change
     (or of a 1x1 run, whose single compute rank writes serially).
  A `cdf_direct: WATCHDOG` line on stderr means a `pwrite` had not returned
  after `ROMS_GPU_IO_WATCHDOG` seconds (60): keep the log and the node
  names; one benchmark run was 60 times slower on one node for a reason
  not found.
- `nc_io`'s `nc_close` chooses between the header copy and a rename by
  whether data were put through `nc_io`'s put routines after `nc_create`:
  a creation routine that wrote data with `nf90_put_var` directly would
  lose it. Use the `nc_io` routines.
- Start-up reads use `O_DIRECT` (`ROMS_GPU_PAR_READ`, README "Run"). When a
  start-up read misbehaves, compare with `ROMS_GPU_PAR_READ=whole`, and run
  with `ROMS_GPU_PAR_READ_VERIFY=1`, which re-reads every variable through
  netCDF and names any that differ.
- Forcing start-up (`frc_io` `frc_warm_*`): the cost on cold nodes is the
  latency of many small reads (a time axis of a classic file is one value
  per record, a record apart), not bandwidth. Measured on Derecho: buffered
  8-KiB reads 0.3-0.5 ms each on a CPU node, 6-8 ms inside the model;
  `O_DIRECT` 4-KiB reads 18-20 ms each (do not use `O_DIRECT` for small
  reads); `posix_fadvise(WILLNEED)` is ignored by the Lustre client
  (nothing becomes resident). Hence the read-ahead threads (cdf_direct
  `cdf_warm_*`, plain `pread`s, data discarded). `wall:` stamps name each
  start-up stage (first forcing, river, boundary and tide records, the
  initial diagnostics, the output set-up); `ROMS_GPU_FRC_WARM=0` gives the
  old start-up for an A/B.
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

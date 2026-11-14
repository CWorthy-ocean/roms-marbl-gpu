# MARBL on the GPU model: port notes (2026-10-02)

Status: surface-flux and interior paths ported to CUDA Fortran, validated
column by column against unchanged CPU MARBL, integrated into the model
(MiniPac, 32 tracers, hourly MARBL) and validated over a day against CPU
ROMS-MARBL with the noise-floor method (all tracers within the CPU model's
own perturbation sensitivity).

## Layout

| Piece | File | Notes |
|---|---|---|
| MARBL sources | `$(MARBL_SRC)/*.F90`, built into `build_gpu/marbl/*.o` | unchanged upstream code, compiled with nvfortran as host code; `marbl_interior_tendency_mod.F90` at -O0 (nvfortran -O2 miscompiles its WHERE constructs) |
| Host instance | `src/host/marbl_host.F90` | `put_setting` from `marbl_in`, `init(nz, mks)`; exports tracer names/indices, forcing indices, the flat parameter table `marbl_par_t`; per-column reference calls (`marbl_host_sflux_column`, `marbl_host_interior_column`) for tests |
| Surface fluxes | `src/device/marbl_sflux.F90` | one thread per point: gas exchange (Schmidt, O2 saturation), carbonate solver (coefficients incl. optional pressure correction, safeguarded Newton `drtsafe1`), NHx emission, dust/iron/NOx/NHy deposition; arrays `sf_*` (forcing, pH saved state, fluxes, 15 diagnostics) |
| Interior | `src/device/marbl_interior.F90` | K1 `carb_kernel` (3-D pointwise: in-situ T, two carbonate solves, CO3 saturation) and K2 `interior_kernel` (one thread per column, surface to bottom, PAR + full ecosystem + sinking recurrence streamed level by level with scalar state `sink_t`); tendencies applied in place to slot nnew; parameters in the static device table `prd` |
| Model driver | `src/host/marbl_step.F90` (`marbl_gpu`) | namelist, setup, forcing, the per-call sequence (CPU order: fluxes applied to the top layer, then interior), restart/history hooks; history carries the surface set (PH, pCO2SURF, DpCO2, FG_CO2, FG_ALT_CO2, PH_ALT_CO2, pCO2SURF_ALT_CO2), the saturation depths zsatcalc/zsatarag (`marbl_zsat_compute`, MARBL's `compute_saturation_depth` on the K1 fields) and the 3-D CO3, CO3_ALT_CO2, co3_sat_calc, co3_sat_arag (float32, CPU names and units) |
| Tracers | `src/device/passive.F90` MARBL mode | the 32 MARBL tracers are the passive tracers (names from MARBL, IC by name, boundary data `tb_*` from the bgc boundary file); build with `-DPASSIVE_FP64` for double storage |
| Forcing | `src/host/frc_io.F90`, `src/host/bry_io.F90` | climatology cycling (`cycle_length`, same bracket as the CPU `find_new_record_cycle`), bgc surface series (dust, iron, nox, nhy, xco2_air, xco2_air_alt) and per-tracer boundary series |

Namelist: `nt_bgc = 32` turns MARBL on; `&marbl_biogeochemistry_settings`
(`marbl_config_file`, `marbl_timestep`); the bgc files go in `frcfiles`.
Example: `cases/GPU/namelist_roms_gpu_marbl.nml`, job `submit_roms_gpu_marbl.sh`.

## Validation

Column tests (`tests/test_marbl_sflux.F90`, `tests/test_marbl_interior.F90`,
jobs `cases/GPU/submit_test_marbl_*.sh`) feed identical MiniPac columns from
the UP3 restart to CPU MARBL and to the kernels:

- surface flux, 4096 columns, two pH-bracket scenarios: deposition terms,
  Schmidt numbers, piston velocities, NHx emission exact; pH 4e-15
  relative; CO2 fluxes 1e-9 relative worst case (round-off of the air-sea
  difference near equilibrium).
- interior, 2304 full columns (230,400 levels): all 32 tendencies within
  4e-16 relative, pH within 4e-15, PAR/nitrification within 2e-15; no
  solver failures. The host-compiled copy of the same source is bit-exact
  against MARBL.
- MARBL off: day run restart bit-identical and history identical to the
  pre-port reference.

Timing (A100, warm, 25,600 columns = 15% of MiniPac): carbonate 59 ms,
interior 93 ms per call, i.e. about 1 s per hourly call for the full
domain before any tuning (255 registers, 3.3 kB stack per thread in K2).

## Day run against CPU ROMS-MARBL (MiniPac, 2010-03-30, 32 tracers, hourly MARBL)

Cases: GPU `cases/GPU/namelist_roms_gpu_marbl.nml`; CPU reference and
perturbed twin (initial T + 1e-6 degC pattern) in `cases/CPU_marbl`
(same dycore sources as the matched CPU case, MARBL/NOX/NHY/PCO2AIR
enabled). Evaluation `tools/marbl_noise_floor_eval.py`: RMS over wet
points of (GPU - ref) and (twin - ref), ratio = GPU/twin.

Result (first full run): every one of the 32 tracers stays inside the CPU
model's own perturbation sensitivity at hours 1, 6, 12 and 24; the worst
ratio over the day is 0.90 (diatC at hour 24), most are 0.3-0.6, and the
surface pH and pCO2 ratios are 0.35 and 0.32. 24 MARBL calls, no
carbonate-solver failures, 27.5 GB device memory. The GPU-CPU difference
grows from ~1e-8 of the tracer scale at hour 1 (round-off) to the
twin's level by day end, as it does for the physics. FG_CO2 at hour 1 had
a relative difference of ~1e-5 before the twin's signal had grown (ratio
33): the bgc forcing was interpolated to the n+3/2 boundary time instead
of the CPU's n+1/2 set_forces time; fixed in `marbl_gpu_step`. After the
fix the hour-1 FG_CO2 difference is 2e-17 (round-off) and its ratio at
hours 6 and 24 is 0.00 and 0.31; the worst tracer ratio stays 0.90.

Three-day run (2026-10-05, same setup, 6-hourly output; table in
`MARBL_NOISE_FLOOR_3D_2026-10-05.txt`): hour 24 reproduces the one-day
numbers exactly (both models are deterministic across runs). The worst
ratio over all 32 tracers and the surface diagnostics is 0.90 at hour 24,
0.97 at hour 48 and 1.04 at hour 72 (spCaCO3); in the double-precision
day-3 restarts every field, including temperature (0.66), salinity (0.98)
and the saved 3-D pH (0.84), is at or below the twin's level. The drift of
the ratios toward 1 is the expected saturation: by day 3 both the GPU-CPU
difference and the twin's 1e-6 degC perturbation have grown to the model's
own divergence level, so the two are statistically indistinguishable.
72 hourly MARBL calls, no solver failures, 27.5 GB.

Cost: a step with 32 FP64 tracers takes 0.42 s (0.15 s for the dycore
with T/S only), and the step that includes the hourly MARBL call 0.72 s,
so MARBL itself costs about 0.3 s per hour of simulation.

## Passive-tracer transport performance (2026-10-05)

Profile of the 32-FP64-tracer step before this work (nsys, steady state,
per step): passive kernels 268 ms (predictor horizontal 41, vertical 40,
implicit 38; corrector horizontal 42, vertical 42, implicit 38; t3dmix 27),
dycore + T/S 153 ms, MARBL 118 ms on the hourly step. All passive kernels
are fused over the 32 tracers already; each ran at ~1.2 ms per tracer,
20-70% of memory bandwidth, at 25% occupancy (128 registers).

Changes (all bit-identical to the previous output in the FP64 build,
verified on the 3-day run `minipac_gpu_marbl3dv3` vs `3dv2`):

- `src/device/passive_column.F90` (new module): vertical advection and
  the implicit solve fused into one column kernel, one thread per
  (block of NTB = 2 tracers, column); the Thomas intermediate is kept in
  place in `tp(nn)`, the column factors come from a separate small
  factorization kernel (`passive_factor`, 1 ms). NTB = 2 (50 ms per call)
  beat 1 (55), 4 (59) and 8 (81): occupancy matters more than reuse.
  Replaces `passive_vertical` + `passive_implicit` (78 ms per call).
- Corrector horizontal kernel: masks and fluxes loaded once per cell and
  the shared face differences reused (42 -> 38 ms). The same change made
  the C4 predictor slower (41 -> 45 ms) and was reverted there.
- Net: passive kernels 268 -> 209 ms per step; mean step 0.500 -> 0.441 s
  on the 3-day MiniPac run (pre_step3d 154 -> 128 ms, step3d_t bracket
  201 -> 167 ms).

Not done / next candidates: float32 storage for the passives would halve
their traffic (a precision decision for the carbonate tracers); the C4/UP3
horizontal kernels (~1.3 ms per tracer) and t3dmix (27 ms) are at the
same per-tracer cost as the T/S path; MARBL's K2 (255 registers) costs
~100 ms per hourly call; the per-step passive boundary fills are small
(kernels < 1 ms, `set_bry` 1.7 ms) and not worth moving.

Second round (2026-10-05, evening), all checked bit-identical on the 3-day
run unless noted:

- Exit hang fixed: with `wrt_file_his = .false.` the I/O rank was never
  told to quit (`io_finalize` returned early when nothing had been staged)
  and the run hung after its last step.
- Alternative-CO2 carbonate solve skipped when its inputs and bracket equal
  the base solve's (true wherever ALT_CO2 runs the same forcing, as in
  MiniPac): carbonate kernel 18.5 -> 13.7 ms per call, surface flux
  0.53 -> 0.37 ms. Interior kernel per-thread arrays sized to the 32
  tracers in use (NTRMAX 64 -> 32): 96.9 -> 92.7 ms. Hourly MARBL call
  117 -> 107 ms.
- Tried and reverted: fusing the horizontal C4/UP3 step into the passive
  column kernel (bit-identical, but 165-180 ms per call against 50 + 40 ms
  for the separate kernels; a per-column kernel cannot hide the latency of
  the eight neighbour loads per level that the 3-D kernel hides with 100x
  more threads). Source kept as `passive*.F90.hfuse_slow`.
- Measured and declined: float32 storage for the passives
  (build without `-DPASSIVE_FP64`): passive kernels 210 -> 206 ms per
  step, mean step 0.441 -> 0.429 s, device memory 27.5 -> 19.2 GB. The
  transport kernels are not bandwidth-bound; their cost is instruction
  issue and latency. Science of the float32 run (`minipac_gpu_marbl3df32`,
  `MARBL_NOISE_FLOOR_3D_2026-10-05.txt`): all 32 tracers stay inside the
  CPU noise floor (worst ratio 1.04, as FP64; float32 vs FP64 GPU
  differences 1e-7 to 3e-6 relative at hour 72), but the saturation-depth
  diagnostics break: zsatcalc/zsatarag RMS differences 2,000-4,000 times
  the twin's, because the root of CO3 - CO3_sat at depth flips level under
  float32 rounding of DIC/ALK. FP64 storage is kept (decision 2026-10-05).
- Not done: splitting the interior kernel into light / pointwise
  physiology / particle-recurrence kernels. The level routine is a strict
  per-level chain (PAR feeds the physiology, Fe scavenging depends on the
  particle state and feeds it back, remineralisation feeds denitrification,
  ligands, O2 and the tendencies), so a split needs three kernels and ~40
  intermediate 3-D fields for an estimated ~25 ms per hourly call (< 2% of
  a step), with loss of bit-identity.

Third round (2026-10-05, late): tracer-innermost passive layout. `tp` is
now `(ntp, is:ie, js:je, nz, 3)`, so the tracers of a point are one
contiguous 256-byte line. The transport kernels (`passive_column.F90`)
are one thread per (tracer, cell) for the C4 predictor, the UP3 corrector
and the lateral diffusion, and one thread per (tracer, column) for the
fused vertical advection + implicit solve; consecutive threads hold
consecutive tracers, so every tracer load of a warp is one coalesced
line, and per-thread state is one tracer (80-91 registers, no stack).
The per-element arithmetic is unchanged and the 3-day run is
bit-identical to the previous layout (history and restart). The boundary
kernels loop over the tracer innermost; the MARBL kernels take the
tracer index first; history staging gathers one tracer from the
interleaved slab; the restart write and the initial read use array
sections. Result on the 4-step profile: passive kernels 209 -> 179 ms
per step (horizontal 2 x 40 -> 2 x 40, column 2 x 50 -> 2 x 36, t3dmix
27 -> 23), step 367 -> 351 ms; 3-day mean step 0.441 -> 0.402 s. The
column tests (`tests/test_marbl_*`) were re-mapped to the new layout and
pass with the same accuracy as before.

Tried and rejected on top of it: shared-memory tile kernels for the
horizontal steps and the lateral diffusion (a 4 x 4-cell block computes
each face once into shared memory, 2.5 face evaluations per cell instead
of 4, bit-identical by construction). Slower: horizontal 2 x 40 -> 2 x 55
ms, diffusion 23 -> 31 ms; the shared-memory round trip, the block
synchronisation and the half-empty second face round cost more than the
repeated face arithmetic. Source kept as `passive_column.F90.tile_slow`.
With one tracer per thread the horizontal kernels did not change at all
(2 x 40 ms before and after the layout change), so their cost is the
per-cell face arithmetic itself (four C4 faces with a division each, or
the UP3 faces), which bit-identity forbids reducing algebraically.

Dycore round (2026-10-06): kernel fusion, all bit-identical on the 3-day
run. Tracer lateral diffusion: the two flux temporaries and the divergence
kernel became one kernel per tracer with the faces inline (5.1 -> 3.3 ms
per step). Horizontal pressure gradient: per direction, the face
differences, the two edge-copy kernels, the harmonic averages and the
update became one kernel, with the edge copies folded into index clamps
(14.8 -> 6.1 ms). Horizontal momentum right-hand side: the Coriolis/
curvature term, the four flux families (curvature temporaries with edge
copies, then faces) and the divergence became one kernel per component
(15.0 -> 9.8 ms; 21 launches per call -> 1). The same per-point expressions
in the same accumulation order make these bit-identical; the price is that
each face is evaluated by both neighbouring cells, which here costs less
than writing and re-reading the temporaries. Dycore 152.7 -> 134.6 ms per
step, 3-day mean step 0.402 -> 0.389 s. Not pursued: raising the occupancy
of the column kernels above 128 registers (vertical momentum advection,
KPP, momentum corrector) by block-size caps; at most ~2 ms at stake.

Where the remaining time goes (per plain step, 0.351 s on the GPU
timeline after the layout change; the numbers below are from before it): passive kernels 206-210 ms, dycore + T/S 153 ms spread over
~1,700 launches with no kernel above 6 ms, GPU idle 4 ms; the hourly MARBL
call adds 107 ms; a history record adds 70-100 ms.

Pitfalls met: nvfortran inlined the explicit `<<<>>>` launch into
`passive_predictor` and the host segfaulted in `cuLaunchKernel`
(compute-sanitizer clean); keeping the kernel + launcher in their own
module fixed it. CUF kernel bodies cannot hold local arrays (per-thread
tracer blocks need an explicit kernel). Never run two makes concurrently
in `build_gpu`.

## Multi-GPU tiling (2026-10-06)

MiniPac on four A100s of one node: `np_xi = np_eta = 2` compute ranks,
one GPU each (`select_device(rank)` before `init_arch`), plus the I/O
server on the last world rank (`mpiexec -n 5 -ppn 5`, qsub
`select=1:ncpus=64:mpiprocs=5:ngpus=4`). Output stays in single files:
every tile reads its window of the global grid, forcing, boundary and
tide files (`decomp%i0/j0` is the file index of local point 0, the same
pair for every stagger), and writes its owned slab at the matching global
offset, either through the server (history) or rank by rank into one
NetCDF file (restart). The 4-step and the 3-day 2x2 runs are bit-identical
to the 1x1 runs in history and restart (`tools/gpu_bitwise_pair.py 4r 4q`,
`3dv3 3dq`), and the 1x1 run is unchanged against the committed reference.

Seam bugs that had to be fixed (none of them visible on one tile, which
is why the earlier single-rank validation never caught them):

1. Static grid fields and every field read from the initial/restart file
   were read over the tile window only; their two halo points toward a
   neighbor were never filled (`fill_static_halos`, `fill_loaded_halos`
   in case_io.F90).
2. The u/v exchange never carried the seam face. Convention: the halo
   exchange uses the rho-cell range for every stagger, so the seam face
   u(nx+1)/v(ny+1) is the east/north tile's point 1 and arrives on the
   west/south tile as a halo (plus one more point). uv_rhs and prsgrd
   compute the face on both sides, step3d_uv and rufrc only on the
   east/north side; the exchange makes them consistent. The old sends
   were off by one for the staggered direction (also for periodic wrap).
3. The fast barotropic loop exchanged zeta only. ubar/vbar are now
   exchanged every fast step after the OBC, zeta again after its OBC, and
   the step2d_mid kernel was split so Dnew and the PGF state at the halo
   points use the exchanged zeta(knew) (the east/north tile's u(1)/v(1)
   update reads them).
4. Loop ranges that were the CPU single-domain ranges: the boundary-column
   accumulation of DU/DV_avg1 (`jstrV:jend` = 2:ny only at a physical
   south edge) and the bulk_coare current-feedback loops (stop at nx/ny
   only at a physical east/north edge).
5. Writers wrote each tile's full local range including halos; the last
   arrival won, which showed up only in arrays that are never exchanged
   (DU/DV_avg*, MARBL pH state, MARBL diagnostics). `set_write_bounds`
   now writes owned points only and each slab carries its file start
   shift (rho axes +ir0/+jr0, u/v axes 0) through `io_stage_*(…, di, dj)`
   and the restart `put_*` helpers.

How they were found: per-step history runs of both cases
(`output_period_his = 900`, cases `4r1`/`4q1`) and
`tools/seam_map.py <rec>` (development repository only), which counts differing columns per field
by signed offset from the seams (file i = 257, j = 217 for MiniPac). The
first record that differs, the fields in it and the offset pattern
(e.g. hbls differing at W nx-1, nx and E 1 only; vbar zero at the
boundary column on the seam row) pointed at each bug directly.

Exchange cost: the original delivery (`comm_exchange2/3`: managed send
and receive buffers read and written on the host with allocate+reshape
and a cudaDeviceSynchronize per exchange) costs ~19 ms per 3-D exchange
and ~1.2 ms per 2-D one; with 60 fast steps the 2x2 step took 7.5 s
against 0.69 s on one GPU. `deliver_dev` (halo.F90) now packs on the
device, copies the packed buffers with cudaMemcpy into pinned host
buffers, exchanges them with `comm_sendrecv_raw` (contiguous, no
allocation) and copies back before the unpack kernels; pure copies, so
bit-identity is unaffected: 7.3 -> 1.43 s per step. The rest was two
kernels (rho_eos column kernel, the two fused prsgrd kernels) that the
CUF runtime launched with a single block on tiles only (161 / 117 / 85 ms
instead of 1.5 / 1.8 / 1.2 ms; the strided loop still covered the tile, so
results were unaffected). Cause: loop scalars passed by reference to the
formulas device functions (`-Minfo=accel`: "implicit copyin",
"Reference argument passing prevents parallelization", launch reported as
`<<< (1,1), (32,4) >>>`); the binary holds two variants of such a kernel
and the runtime picked the single-block one on tiles. Parenthesizing the
scalar call arguments (`jackett_rho1((Tt), (Ts))`) passes temporaries,
restores a runtime grid and is bit-identical. 2x2 four-step last step
(with the hourly MARBL call): 0.536 s against 0.580 s on one GPU; the
3-day mean step: 0.151 s on 2x2 tiles against 0.389 s on one GPU (2.6x
on 4 GPUs; 288 steps in 43.5 s of step time, history every 6 h), with the
3-day 2x2 run bit-identical to the 1x1 run. The remaining gap to 4x is
per-exchange latency: ~1800 exchanges per step (the fast loop alone does
60 x 4 fills), each with four synchronous staging copies and two
MPI_Sendrecv calls; GPU busy time is ~0.27 s of a 0.8 s MARBL step.
GPU-aware MPI on the device buffers (`comm_sendrecv_dev`, used when the
job sets MPICH_GPU_SUPPORT_ENABLED=1 and the binary is linked with
`-L/opt/cray/pe/mpich/8.1.32/gtl/lib -lmpi_gtl_cuda`; `deliver_dev` falls
back to pinned staging otherwise) and one 3-D exchange for the smoothed
Richardson number in lmd (was nz-1 2-D exchanges) bring the 3-day mean
step to 0.144 s (2.7x on 4 GPUs; 9.7 min of walltime for the 3-day case),
still bit-identical in history and restart. The per-rank step totals are
now unequal (28 to 41.5 s over 288 steps): the slowest tile sets the
pace, so load balance (wet columns per tile) and the remaining ~1800
exchanges per step are the next levers, followed by multi-node runs and
the CPU-vs-GPU benchmark at equal node counts.

Case files (cases/GPU): `namelist_roms_gpu_marbl4q.nml` /
`submit_roms_gpu_marbl4q.sh` (4 steps, restart after 4 steps),
`…4q1` (history every step), `…3dq` (3 days, 6-hourly history, 50 min
walltime with the old exchange); 1x1 counterparts `4r`, `4r1`, `3dv3`.

## Multi-GPU round 2: balance, exchange count, output path (2026-10-06)

Measured with the corrected per-step metric (the `cumul` of a
`step: baroclinic step` line is that step's wall time; the earlier "0.144 s"
figure in this file's first multi-GPU section was the largest single step
divided by 288 and is wrong). Non-output steps of the 3-day MiniPac case:

| configuration | non-output step | output steps / 3 d | 3-day walltime |
|---|---|---|---|
| 1 GPU | 0.424 s | 11 s | 3:59 |
| 2x2, commit 57690cc | 0.209 s | 190 s | 10:28 |
| 2x2 + mask-balanced cuts (lever 1) | 0.212 s | 183 s | 10:28 |
| 2x2 + server assembles records | 0.215 s | 6 s | 3:39 |
| 2x2 + fused fast-loop tail, non-blocking exchange | 0.192 s | 6 s | 3:35 |

- Lever 1, `src/shared/tile_balance.F90` + `&tile_settings`: tensor-product
  cuts from the land mask (cost = wet columns + land_weight x land cells,
  2-D prefix sums, quantile seeds, alternating exact min-max partition with
  the 2*halo+2 width floor; `uniform` and `explicit` modes; cuts and
  per-tile costs logged as `tiles:` lines). On MiniPac the cost ratio goes
  1.144 -> 1.091 but the step does not move: the tiled step is bound by the
  ~1400 exchanges per step (GPU busy 0.22 s of a 0.80 s MARBL step), not by
  the slowest tile's work. Keep it; it costs nothing and helps land-heavy
  domains once the exchange overhead is down.
- Output path: the server received one tile's record, wrote it (a partial-
  chunk rewrite), then received the next tile's, so the other tiles waited
  behind every write (10-25 s per history step). `write_assembled` now
  collects all tiles' parts of a record and writes each variable once at
  full extent: history steps 183 s -> 6 s per 3 days. The restart is still
  written tile by tile (24 s for 4 tiles); routing it through the server
  in double precision is the next output item.
- Lever 2 so far: one 3-D exchange for the smoothed Richardson number
  (lmd), two redundant passive fills dropped, the fast-loop tail doing one
  fused ubar/vbar exchange with zeta exchanged only on the last fast step
  (`step2d_tail`; the driver uses `step2d_with_halo`, not `step2d_fused`),
  and non-blocking exchanges (both directions in flight, one wait) in
  `comm_sendrecv_raw/dev`. 0.215 -> 0.192 s. The remaining idle time is the
  per-exchange latency; the next step is batching several fields per
  exchange at the heavy call sites (rho_eos, prsgrd, lmd, pre_step3d).
- Mask-aware tiling beyond this (the CPU MPI_MASKING idea: more subdomains
  than ranks, drop land-only subdomains; and wet-column compaction of the
  column kernels) is discussed in the resume notes.

## Multi-GPU round 3: exchange batching, wet columns, restart, init, two nodes (2026-10-06)

Every run below is bit-identical to the single-GPU reference over 4 steps
and over 3 days (history and restart). Non-output step = mean of the
`step: baroclinic step` cumul values over the 3-day run.

| configuration | non-output step | init | restart write | 3-day walltime |
|---|---|---|---|---|
| 1 GPU (reference) | 0.424 s | 97 s | 24 s | 3:59 |
| 2x2, round 2 | 0.192 s | 93 s | 24 s (rank-ordered) | 3:35 |
| 2x2 + A: group exchanges | 0.169 s | | | 3:40 |
| 2x2 + B: wet-column kernels | 0.155 s | | | 3:37 |
| 2x2 + C: restart via server | 0.169 s (run-to-run noise 0.155-0.169) | | 10.5 s on the server, overlapped | 3:30 |
| 4x2 on two nodes (D) | 0.126 s | 76 s | | 3:06 |
| final main binary (round 3), 1 GPU | 0.424 s | 67 s | 10 s on the server | |
| final, 2x2 | 0.171 s | 35 s | 10 s on the server | 2:27 |
| final, 4x2 on two nodes | 0.124 s | 22 s | 10 s on the server | 2:11 |

Pac CPU reference (`cases/CPU_Pac`, 512 ranks on 4 CPU nodes, daily
history + MARBL diagnostics): 3.55 s per step over the last 280 steps,
init 226 s, walltime 25:55, 365 GB of output (rst 118 GB).

- A, `halo_mod` group exchanges (`hg_begin / hg_add(id, slot, itr) / hg_end`):
  the consecutive fills at each call site (rho_eos, prsgrd, geom, lmd,
  pre_step3d, step3d_t/uv, visc3d, step2d, advect, the driver) become one
  EW and one NS round over packed multi-field buffers (ids F_HZ..F_ZETA
  resolved by a dispatcher; wrk3a/wrk3b keep their nz-from-level-0 quirk).
- B, wet columns: `build_wet_list()` (gpu_state) after `setup_metrics`
  gives `nwet, wet_i, wet_j`; the MARBL carb/interior/sflux kernels and the
  passive horiz/t3dmix/column kernels run over wet columns only (1-D grids,
  block 128). KPP is untouched because hbls carries nonzero land values.
  Land values of tracers/zeta/temp are exact zeros in the restart, so the
  skipped stores change nothing.
- C, restart through the I/O server in double precision: `io_stage8_*`
  stage real(8) windows through pinned/device buffers, the record header
  carries a precision flag (hdr(7)), the server receives MPI_REAL8 into
  real(8) tile buffers and `write_assembled` writes each variable once with
  the explicit-shape double putters `nc_put_2d/3d_rec_d`. Compute ranks no
  longer wait (24 s -> 0). Bug found on the way: the server's variable
  names were 16 characters (`NAMELEN`), truncating
  `MARBL_PH_SURF_ALT_CO2`; now 32 with a length check in `add_var`.
- D, two nodes: PBS `select=1:ncpus=64:mpiprocs=4:ngpus=4+1:ncpus=64:mpiprocs=5:ngpus=4`
  and `mpiexec -n 9` (no -ppn) place ranks 0-3 on node 1 and 4-8 on node 2,
  so the server (last rank) shares node 2 and every compute rank has its own
  GPU (`select_device` = rank mod device count). 0.169 -> 0.126 s per step
  (1.34x for 2x the GPUs: still exchange-latency bound). The history output
  steps stall ~4 s each: one server rank writes a 3.8 GB record in 6.5 s
  while the eight tiles compute the next 24 steps in 3 s, so the run is now
  bound by the single writer (13 records + restart = 95 s of writing per 3
  days on both 2x2 and 4x2). Several writer ranks or parallel NetCDF is the
  next output item.
- Initialization (97 -> 36 s on 2x2, bit-identical on 1x1, 2x2 and 4x2):
  - forcing file handles cached for the whole init (`frc_file_ncid`,
    `close_frc_cache`): the series binders opened every file three times
    per variable (174 series x 22 files) - 42 + 23 s -> 20 + 2 s;
  - time axes prefetched once in parallel (`prefetch_time_axes`: files
    assigned to ranks by size, all local reads first, then broadcasts;
    `cached_times` in the binders): a classic-format record axis costs one
    seek per record, 4348 records for `uwnd` - 20 s -> 5-9 s. A first
    version interleaved read and broadcast per file and serialized the
    reads (no gain): keep reads and broadcasts in separate loops;
  - `src/host/par_read.F90`: collective reader for the grid and the initial
    records. Phase 1 queues every window (`pr_store_2d/3d/3d_pt` with the
    same sequence on all ranks), `pr_flush` reads whole variables
    contiguously (one per rank per batch, windows allgathered) and
    redistributes the windows over MPI, phase 2 stores them on the device
    through a staging buffer and small kernels (`stage_to_device`). The
    tile-window reads straight from the contiguous file were the cost
    (0.29 s per tracer per tile on MiniPac, 1.5 s on Pac; 0.01 GB/s for an
    8x8 Pac tile): 25 s -> 19 s on MiniPac (bound by the 4-stripe file,
    reads 7-11 s per rank at ~0.43 GB/s) and the host-side stores into
    managed memory 8 s -> 1 s. Pac's restart has 24 stripes.
- SFLX_CORR (sea-surface salinity restoring, `roms_bugfix bulk_frc.F90`)
  ported for the Pac case: `&surf_frc_settings sflx_corr = .true.` and
  `&sss_correction dsssdt` (cm/day, stored as m/s), series `sss`/`sss_time`
  through the usual bulk interpolation, kernel at the end of
  `calc_bulk_coare`: `stflx(isalt) = -dSSSdt (S(nz, nrhs) - sss) rmask`
  (bulk forcing only, as the CPU). Default off keeps MiniPac bit-identical.
  Smoke test (4 steps, 2x2): mean surface salt change 2.7e-4 PSU, hbls
  responds everywhere, isolated KPP-depth outliers up to 0.08 PSU.
  Validation against the CPU comes with the Pac reference.
- Pac CPU reference: `cases/CPU_Pac` (repo sources at HEAD, CPU_marbl's
  cppdefs plus SFLX_CORR, CDR off, 32x16 tiles on 4 nodes, exact restart
  2010-02-08, daily history and MARBL diagnostics, restart at day 3). The
  CPU_marbl local source overrides (KPP, probe headers) must not be copied:
  they need a removed probe module, and they equal the repo files when
  LMD_KPP and LMD_BKPP are both defined.

## Pac on the GPU (2026-10-06): first run and CPU comparison

Case: `cases/GPU/namelist_roms_gpu_pac3d8.nml` + `submit_roms_gpu_pac3d8.sh`
(1856x960x100, 32 MARBL tracers, hourly MARBL, SFLX_CORR on, exact restart
2010-02-08, 4x2 wet-balanced tiles on 2 nodes = 8 GPUs + server rank);
`namelist_roms_gpu_pac8.nml` is the 4-step test. CPU reference:
`cases/CPU_Pac` (512 ranks on 4 CPU nodes).

| | GPU, 8 GPUs on 2 nodes | CPU, 512 cores on 4 nodes |
|---|---|---|
| mean step (3 days, with output) | 1.15 s | 3.55 s |
| initialization | 103 s (restart read 60 s) | 226 s |
| 3-day walltime | 9:54 - 11:03 | 25:55 |
| GPU memory per tile | 30 GB of 40 | |
| tile cost imbalance | 1.11 | |

- Correctness: `tools/gpu_cpu_day_cmp.py` (RMS of gpu-cpu over wet points
  per field, ratio to the CPU field's RMS). Day-3 ratios on Pac vs the
  MiniPac GPU/CPU pair: temp 1.3e-4 vs 1.1e-4, salt 3.2e-5 vs 2.4e-5, zeta
  4.7e-5 vs 2.2e-5, DIC 1.6e-5 vs 2.2e-5, PH 3.5e-6 vs 3.6e-6, O2 1.7e-4 vs
  1.6e-4, phytoplankton carbon 1.0-1.3e-2 vs 3-6e-3: the Pac run sits at the
  same noise-floor level as MiniPac.
- The first Pac run had temp RMS 0.17 C at day 3 (ratio 1.3e-2, 100x
  MiniPac), confined to sea-surface temperatures at -1.8 C: the CPU code
  always applies `SEA_ICE_NOFLUX` (set_global_definitions.h): where
  SST <= -1.8 C it zeroes stflx(itemp) and srflx. Ported into the bulk kernel
  (`calc_bulk_coare`); MiniPac stays bit-identical (never that cold). The
  CDR_LITE k_gas and DAILYPAR variants of the rule are not in the GPU model
  (those options are off).
- Step time is bimodal on Pac (p10 0.45 s, p50 0.90 s, p90 2.3 s, max 8.7 s):
  step3d_t (0.43 s mean, p10 0.14 / p90 1.22) and pre_step3d (0.30 mean,
  p50 0.15 / p80 0.67) vary 5-10x from step to step, which they do not on
  MiniPac tiles (flat 0.042 / 0.034 s). The pure dycore steps (p10) are 2.9x a
  MiniPac 2x2 step for 3.7x the cells, as expected; the rest looks like
  managed-memory migration (30 GB resident of 40, the MARBL surface host
  loop touching tp pages, 2 x 11.5 GB of tracers per tile). Profile a few Pac
  steps with nsys (UM traffic vs kernel time) before anything else.
- Output: the server writes a 30 GB history record in 50-64 s and the 55 GB
  restart in 90 s; three daily records plus the restart are ~4 minutes of a
  10-minute run, so the single writer is now the second item after the
  step-time variability.

## Pac optimization round (2026-10-06 night)

- The bimodal 8-GPU step was memory pressure, not kernels: the wet-balanced
  cuts gave rank 4 a 310k-cell tile at 39.4 of 39.5 GB (`gpumem:` report
  printed by `wall_stamp` at 'init done' / 'restart staged'; ~0.12 GB per
  1000 tile cells including halos), unified memory paged in a 16-step
  cycle and every rank waited at the exchanges. On 16 GPUs (18 GB tiles)
  the step was flat at 0.46 s. Fix: `&tile_settings tile_max_cells_ratio`
  (default 1.2): the land weight is raised in x1.5 steps until the largest
  tile area is within that ratio of the mean (Pac: 0.844, max tile 238k
  cells, 31.7 GB, cost imbalance 1.016).

| Pac 3 days | 8 GPUs before | 8 GPUs, cap (final) | 16 GPUs, cap (final) | CPU, 512 cores |
|---|---|---|---|---|
| step p10 / p50 / p90 | 0.45 / 0.90 / 2.3 s | 0.42 / 0.53 / 0.62 s | 0.25 / 0.31 / 0.38 s | 3.55 s mean |
| stepping per rank | 330 s | 166-172 s | 119 s | 1020 s |
| walltime | 9:54 | 7:14 - 8:42 | 6:36 | 25:55 |

With four writer ranks (2026-10-07, "Next levers"): 8 GPUs 5:16 walltime at
0.51 s per step (p50).

Final-binary verification (edge-row exchange + cap): MiniPac 4r/4ref,
4q/4r, 8q/4r, 3dq/3dv3, 3d8q/3dv3 bit-identical; Pac 8-tile and 16-tile
3-day histories bit-identical to each other; Pac 8-tile vs the CPU
reference at day 3: zeta 1.7e-5 (was 4.7e-5 before the seam fix), temp
1.3e-4, salt 3.8e-5, DIC 1.7e-5, PH 3.8e-6, O2 1.8e-4, diatC 1.1e-2 - the
MiniPac noise-floor levels.

- nsys on rank 0 (12 steps, `tools/nsys_timeline.py` on the sqlite export):
  steady fast steps are kernel-bound (0.47 s of kernels per step); passive
  tracer kernels (horiz 62 ms, column 48, pre-advection 26, t3dmix 18 ms
  per step) are a third and run near memory bandwidth with 32 FP64
  tracers; MARBL interior 74 ms per hour; exchange pack/unpack 42 ms over
  ~350 launches; per-step memsets 17 ms (step3d_uv2); UM traffic during
  steady steps ~170 MB/step. The output path dominates the fixed cost:
  one server rank writes a 30 GB history record in 50-64 s and the 55 GB
  restart in 90 s (4 of the 7 minutes of the 3-day run); compute ranks
  stall ~20 s at a history step when both staging slots are busy.
- Seam bug 6, visible only on Pac: `interior_bounds` exchanged rows 1..ny,
  so the physical boundary rows (j = 0 south, ny+1 north) were never
  exchanged east-west while the boundary routines (u2dbc, v2dbc, u3dbc,
  v3dbc, t3dbc radiation) use tangential neighbours on them. 8- and
  16-tile Pac runs diverged from step 1 at the south edge, exactly at the
  cuts (per-step history pair `pac8s`/`pac16s`, offsets from the cuts).
  MiniPac never showed it: its seams meet masked edge points. Fix: j1 = 0
  on a south edge, j2 = ny+1 on a north edge (the north-south round already
  packs columns i1-h..i2+h). MiniPac 2x2 stays bit-identical; the Pac
  8- and 16-tile per-step runs agree on every completed record.
- I/O server slots: with output faster than the writer (per-step 30 GB
  records on Pac) the server evicted slot 1 rather than the oldest record
  and dropped tiles still assembling at quit (records 2 and 4 of 5 had
  holes). Now `NSLOT = 4`, the oldest record is evicted with a WARNING,
  and incomplete records are flushed at quit: all five per-step records
  complete, MiniPac 2x2 bit-identical. A writer slower than the output
  cadence still costs partial-chunk writes; several writer ranks or
  parallel NetCDF remain the real fix.

## Next levers (2026-10-07): writer ranks, exchange batching, initialization

Build now links the parallel NetCDF module (`module load netcdf-mpi` AFTER
`cray-mpich`; `LIBS` adds `-L$(nc-config --prefix)/lib64 -lnetcdf`); the
serial paths are unchanged. All results bit-identical to the references.

- Several writer ranks (`ROMS_GPU_IO_SERVERS=n`, default 1; the last n
  world ranks; `comm_writers()`). Each compute rank sends every record
  header to every writer and variable iv's slab to writer mod(iv-1, n);
  each writer assembles and writes its own variables. Parallel HDF5 only
  extends a record variable collectively, so `write_assembled` first does
  a collective round (ocean_time, then one element per variable at
  x = writer index + 1, then a barrier) and the owners put their variables
  independently in parallel. Files are opened lazily when the first record
  for them is written, so that writers reach the collective open in record
  order (an eager open at CMD_OPEN deadlocked against another writer's
  record write). Restart (double) variables are now chunked one record per
  chunk (`nc_chunk_one_record`); with default chunking the independent
  writes had to allocate chunks and the sync failed with "HDF error".
  Slot overrun with several writers is an error (collective order) and an
  incomplete record at quit is dropped with a warning.
  MiniPac 2x2, 2 writers: history record 6.1 -> 3.7 s, restart 10.5 -> 5.5 s
  per writer. Pac 8 GPUs, 4 writers (ROMS_GPU_IO_SERVERS=4, node 2 with
  mpiprocs=8): 30 GB history record 60 -> 20 s per writer, 55 GB restart
  90 -> 31 s, 3-day walltime 7:14-8:42 -> 5:16, history bit-identical to
  the single-writer run. A deadlock found on the way: compute ranks sent
  record headers with blocking MPI_Send; a writer in the collective round
  does not receive, the compute rank blocks on that header, and another
  writer waits for that rank's data (posted after the headers). Now the
  data Isends are posted first and the headers are Isends too (buffers in
  the staging set, `nreq = nvars + up to 5 per writer`).
  A single process writes at 0.6 GB/s whatever the striping (24 stripes
  tested), so writers are the lever, not Lustre layout.
  Writer-count rule (no manual setting): the model takes the compute rank
  count from the namelist (np_xi x np_eta, scanned in `init_comm`) and every
  extra launched rank becomes a writer (`ROMS_GPU_IO_SERVERS` overrides).
  `tools/gpu_layout.py <namelist>` sizes the history record from the grid and
  the write flags and picks the fewest writers that write one record in
  about 15 s at 0.6 GB/s each (1..8), printing the PBS select line and the
  rank count; `cases/submit_gpu.sh <submit_script> [NWRITERS=n] [EXE=...]`
  submits with them (the submit scripts launch `mpiexec -n ${NRANKS:-...}`).
  MiniPac -> 1 writer, Pac at daily output -> 4.
- Exchange batching: `hg_end` resolves every registered field to a device
  address (`c_devloc`, kept in `hg_ptr`), and `hg_run_phase` packs/unpacks
  up to 4 fields with the same level range per launch (`hg_slab4`: device
  pointers from `c_f_pointer`, CONTIGUOUS so the explicit-shape device
  dummies get the base address). Default on; `ROMS_GPU_HG_BATCH=0` restores
  one launch per field, `=n` sets the group size. Lesson: the kernels must
  run on `work_stream` (the CUF_DO macros); on the default stream they
  raced the model kernels and filled the halos with garbage, with 1 field
  per launch as well. Gain: 2x2 0.170 -> 0.168 s per step, 4x2 two-node
  0.125 -> 0.123 s (launch count 354 -> ~120 per step; the exchange is
  latency-bound, not launch-bound).
- Initialization: the double-precision restart staging is pageable and
  allocated at the first restart write (two pinned 12 GB sets cost ~10 s
  per rank at init on Pac: output init 15.7 -> 6.2 s; the restart stage
  itself now takes 3 s). Overlapping the whole-variable reads with the
  window redistribution (`pr_flush` posts all receives, keeps one batch of
  sends in flight) changed nothing: the reads are the limit (~2.4 GB/s
  aggregate with 8 streams, 3.6 with 16). A collective MPI-IO read mode
  (`ROMS_GPU_PAR_READ=mpiio`, `flush_mpiio`) exists but needs pnetcdf for
  classic-format files, which netcdf-mpi lacks, so it falls back.

## Restart read through collective MPI-IO (2026-10-07): measured, not adopted

The question was whether every rank reading its own windows collectively
(parallel HDF5 through netcdf-mpi, `ROMS_GPU_PAR_READ=mpiio`) beats the
whole-variable reader (each rank reads whole variables and redistributes
the windows). Collective reads need NetCDF-4 files, so the CPU restarts
were converted once with `nccopy -k 4` (one record per chunk, 24 Lustre
stripes, 118 GB Pac file in 5 minutes) and the GPU-written restart was
used as is. All runs 4 steps, "ini read" segment of the init stamps, 8
GPUs for Pac, 4 for MiniPac; every pair restart bit-identical.

| restart file | whole-variable reads | collective MPI-IO reads |
|---|---|---|
| Pac, CPU restart, 87 variables, classic -> converted NetCDF-4 | 51.5 s | 86.7 s |
| Pac, GPU-written NetCDF-4 restart, 53 variables | 52.8 s | 38.7 s |
| MiniPac, CPU restart, classic -> converted NetCDF-4 | 13.7 s | 23.5 s |

Collective reads win only on the GPU-written file and lose elsewhere, so
whole-variable reads stay the default and `mpiio` stays opt-in. Two
observations for later: HDF5 whole-variable reads run at about half the
rate of classic-format reads (0.64 vs 0.33 s per 1.4 GB variable), so the
GPU-written restart reads slower than the CPU one; and the collective read
is a strided window read (one short row per (j, k)) that HDF5/ROMIO
aggregate poorly. Tile-friendly chunking of the restart (e.g. nz x 128 x
128 chunks, independent of the tiling) and MPI-IO hints are the untested
knobs; neither is geometry-free enough to adopt blind.

## Pipeline hygiene (2026-10-07): what a steady Pac step is made of

Fresh nsys profile of rank 0 (8 GPUs, 4 writers, batching on;
`tools/nsys_step_breakdown.py <report.sqlite> [step]`), steady step:

| item | per step |
|---|---|
| step wall time | 0.42-0.45 s |
| GPU busy | 0.30 s (71%) |
| kernel launches | 3161, of which 2666 under 20 us (~25 ms of GPU time) |
| GPU idle at the all-tracer exchange (`halo_fill_3d_tp`, 5 per step) | 62 ms |
| GPU idle at the group exchanges | 13-35 ms |
| memsets | none |
| end-of-step diagnostics (`check_mode_coupling`, `step_maxima`) | ~2 ms |

So hygiene is worth 2-3% at most; the all-tracer exchange (five ~50 MB
GPU-direct exchanges per step with nothing queued on the GPU meanwhile) is
the largest remaining lever at ~15% of the step (exchange overlap).

Changes (bit-identical by construction):
- `step_mark`: the 19 per-phase stamps (each a cudaDeviceSynchronize + print
  + flush) are off by default; only the `baroclinic step` stamp remains
  (its cumul is the step time the tools read). `ROMS_GPU_STEP_MARKS=1`
  restores them.
- `set_tides`: one kernel per target set (west/east, south/north, potential)
  with the constituent loop inside, host cos/sin in `tide_c/tide_s`
  (device cos/sin would differ in the last bit); the per-element sum runs
  in the original constituent order with the same rounded products.
  92 launches -> 6 per step.
- Redundant all-tracer exchanges removed (3 of the 5 `halo_fill_3d_tp` per
  step). Slot rotation: predictor reads nstp, writes 3; corrector reads 3,
  writes 3-nstp; the next step's nstp is that slot. So (1) the exchange of
  nstp at the predictor start repeats the previous step's post-t3dmix
  exchange (nothing writes tp in between; the load exchanges slots 1 and
  2); (2) the exchange of slot 3 right after the predictor kernels is
  repeated by pre_step3d after passive_t3dbc, and nothing reads the slot-3
  halo in between (passive_t3dbc reads slot nstp at tangential neighbours
  and writes slot nn; its closed-boundary copy writes halo rows of column
  0 that the following NS exchange overwrites either way); (3) the
  exchange of nnew after the corrector feeds only pointwise code (MARBL
  columns, t3dmix's stencil is slot 3) before passive_t3dmix exchanges it.
  Left: pre_step3d (slot 3 after t3dbc) and passive_t3dmix (nnew).
  The T/S exchanges have the same pattern (per-tracer after the predictor,
  then the group after t3dbc; after the corrector, then after t3dmix) but
  carry 2 fields instead of 32; left for the exchange-overlap round.

Results (every pair bit-identical: MiniPac 2x2 vs 1x1 4r history+restart;
Pac 8 GPUs 4-step restart vs pac8; Pac 3-day history vs pac3d8w4m):

| run | MiniPac non-output step | Pac 3-day: p50 step, per-rank stepping, walltime |
|---|---|---|
| main (b6f7621 + writers) | 0.1224 s | 0.511 s, 142.2 s, 5:29 |
| stamps off + fused tides | 0.1216 s | 0.510 s, 141.9 s, 5:03 |
| + 3 tracer exchanges removed | 0.1168 s | 0.488 s, 136.7 s, 5:04 |

Net 4.5% per step on Pac, 4.6% on MiniPac; the walltime difference between
the first two rows is init/queue noise (init 86 vs 83-93 s).

## Exchange overlap (2026-10-07): non-blocking tracer exchange, fewer exchanges, and what the step is really bound by

- The two remaining all-tracer exchanges per step are non-blocking in three
  pieces (`halo_tp_start` packs and posts the east-west round on device
  buffers through GPU-direct MPI with its own tags 321-324; `halo_tp_mid`
  waits, unpacks, packs and posts the north-south round; `halo_tp_finish`
  waits and unpacks). The slot-3 exchange is posted at the end of
  pre_step3d, advanced after visc3d and finished in passive_corrector; the
  end-of-step exchange is posted after passive_t3dmix, advanced after the
  next step's lmd_vmix and finished in passive_predictor
  (`passive_exchange_finish` before finalize). Between post and finish the
  slot's interior is unchanged and its halo unread (see the hygiene
  section for the slot-rotation argument). Without GPU-direct MPI the
  start does the blocking exchange. Debug knobs: `ROMS_GPU_TP_OVERLAP`
  (bitmask, 1 = slot-3 window, 2 = end-of-step window, default 3),
  `ROMS_GPU_TP_CHECK=1` (redo each overlapped exchange blockingly and
  count differing halo values), `ROMS_GPU_TP_DEVSYNC=1`.
- Eleven redundant single-field 3-D exchanges per step removed by the same
  argument: u and v (nrhs) and temp (nrhs) at the step start (exchanged by
  step3d_uv2's and rho_eos(nnew)'s groups after the last writes), the
  per-tracer exchanges after the T/S predictor (the group after t3dbc
  follows) and after the corrector (MARBL and t3dmix read nnew at the
  point; rho_eos(nnew) exchanges it), and t3dmix's exchanges of nrhs
  before and nnew after (slot 3 unchanged since the groups; nnew exchanged
  by rho_eos).
- Results, all bit-identical (MiniPac 4-step and 3-day vs 4r/3dv3, Pac
  4-step restart vs pac8, Pac 3-day history+restart vs pac3d8w4m):
  MiniPac non-output step 0.1168 -> 0.1168 (overlap) -> 0.1128 s (fewer
  exchanges); Pac 3-day p50 0.488 -> 0.494 (overlap) -> 0.478 s (fewer exchanges; per-rank stepping 136.7 -> 132.8 s).
  On rank 0 of the Pac profile the overlap removed the 62 ms of idle at the
  tracer exchanges, but the 3-day step did not move: an all-rank profile
  (`tools/nsys_rank_balance.py`, `nsys_allranks.sh`) shows GPU busy per
  step from 0.198 s (rank 7, 28k wet of 232k cells) to 0.340 s (rank 2,
  all wet); mean 0.29 s. The step is paced by the wettest tile, and the
  other ranks' exchange "waits" were waiting for it. The busy times fit
  cost = wet + 0.47 x land per cell, but the memory cap
  (`tile_max_cells_ratio` 1.2, ~0.12 GB per 1000 cells, 3 FP64 slots of 32
  tracers) forced the balancer to land weight 0.844, i.e. near-equal
  areas. So at 8 GPUs the remaining lever is balance, not latency: either
  memory per cell must drop (wet-column storage of the passive tracers,
  which would let the cap be set on wet cells) or the run needs 16 GPUs,
  where the cap is loose enough for the 0.47 weight. Consequence for the
  wet-column dycore item: it makes land cheaper still, which at the 1.2
  cap mostly speeds up tiles that already wait.
- Rebalancing alone does not help: a 3-day run with the fitted land weight
  (0.47) and the cap at 1.3 (largest tile 282k cells, model max/mean cost
  1.07, 30 GB on the sampled node) gave p50 0.487 s vs 0.478 s, still
  bit-identical. Reason: the ~30 exchanges per step synchronize neighbours
  phase by phase, so the step is the sum over phases of the slowest rank
  in each phase. The passive phases are paced by the wettest tile and the
  dycore phases (full-tile kernels) by the largest tile; a cost-balanced
  cut moves work between the two without shortening the sum. Phase times
  with stamps on (4-step Pac, median over ranks): pre_step3d 0.113 s,
  step3d_t 0.122 s (the passive kernels), step2d fast loop 0.039 s,
  corrector forces 0.028, step3d_uv1+visc 0.024, lmd_vmix 0.020, prsgrd
  0.013, the rest under 0.01; sum 0.39 s. So the fast loop (wider-halo
  item) is 8-10% of the step, and making every phase scale with wet cells
  (wet-column dycore + wet-column storage of the passive tracers so the
  memory cap can be set on wet cells) is what would let a wet-balanced cut
  balance all phases at once (busy 0.34 -> ~0.29 s on the pacing rank).
- A false alarm cost half a day: two runs diverged mid-run at last-digit
  level. The known-good blocking binary diverged at the same step on the
  same node (deg0072); 20+ runs on other nodes were bit-identical. Check
  `exec_host` before chasing a race.

## Wet-column dycore, compressed passive storage, memory-aware tiles, fast-loop exchange (2026-10-08)

Starting point (all-rank Pac profile, 8 GPUs): the passive kernels cost
nothing on land, every dycore kernel cost the same per land cell as per wet
cell (~76 ms per 1e5 cells per step), and the memory cap forced near-equal
tile areas, so the all-wet tile paced every phase.

- **Dry-column skip** (`gpu_state`: `build_dry_map`, `colact`, `dry_lim`,
  groups G_GEOM..G_2D). "Deep land" = land with no wet point within 3
  cells (judged inside the tile arrays; columns whose neighbourhood leaves
  the arrays count as active). Every stored field is constant there
  (history and restart: all zero on deep land at every record, the
  barotropic forcing and transports zero on every land face; hbls/hbbl
  vary only on coastal land), so the ~55 dycore kernels (depths, transports,
  omega, EOS, KPP, pressure gradient, momentum advection and viscosity,
  T/S advection, implicit solves, the momentum predictor/corrector, the
  fast loop) skip deep land after the first two steps of a run (which
  establish the values in every time slot). Test in the kernels:
  `if (lim > 0) then; if (colact(i, j) == 0) cycle; end if` (no mask load
  on tiles without deep land). `ROMS_GPU_SKIP_DRY` = bitmask of groups
  (default all), `ROMS_GPU_DRY_RADIUS` (default 3). Land cost 76 -> 40 ms
  per 1e5 land cells (the rest: coastal band, exchanges).
- **Compressed passive storage** (`tpcol`, `tpnb`, `ntpcol`): tp(ntr,
  0:ntpcol, nz, 3) over stored columns only: the interior wet columns
  first in wet-list order (so MARBL and the passive kernels use their wet
  index directly), then the other wet points and the physical boundary
  rows; column 0 is a shared zero column for every other point (what land
  held before). Stencil neighbours: x neighbours arithmetically
  (ci +- d inside the tile row: exact whenever the value is used, since the
  faces are masked with apply_mask otherwise), y neighbours and halo
  neighbours from a per-column table (tpnb, 8 ints). Exchange
  pack/unpack, restart load, output gather and init go through tpcol.
  Memory: rank 7 (12% wet) 27.2 -> 12.4 GB. Kernel cost: horiz kernel
  82 -> 85 ms on the all-wet rank (a first version with map lookups was
  107 ms, the table alone 95), t3dmix 24 -> 26.5 ms (table; arithmetic x
  was slower there, 33.6).
- **Memory-aware balance**: the 'wet' balance raises the land weight until
  the estimated memory of every tile (0.532*nz kB per cell + 3*ntp*nz*8 B
  per wet cell, fitted on Pac within ~2 GB) is below 92% of the smallest
  device memory (`tile_max_mem_gb`, `tile_mem_cell_kb`, `tile_mem_wet_kb`;
  `tile_max_cells_ratio` is now off by default). Default land weight 0.10
  (measured best).
- **Fast loop: one exchange per substep** instead of two: zeta_new is formed
  on the first halo ring of MPI seams (DUon/DVom one face further there)
  with the neighbour's arithmetic, so the mid-substep zeta exchange is gone
  and zeta(knew) travels with ubar/vbar in one exchange
  (`halo_fill_zuv_n`). The rufrc/rvfrc exchange of every substep is now
  done on the first substep only (the only one that changes them).
- **Layout**: tensor-product 4x2 cuts cannot balance Pac better than
  max/mean cost ~1.08-1.14; 8x1 strips reach 1.002 and drop every
  north-south (inter-node) exchange round. With strips the fast loop
  spends 46 ms per step (14 ms kernels) instead of 111 ms.

Pac 3-day, 8 GPUs + 4 writers, all bit-identical to the reference
(history and restart; different cuts give identical results):

| configuration | median step | mean step |
|---|---|---|
| start of this round (4x2, forced land weight 0.84) | 0.478 s | 0.462 s |
| + dry skip (same cuts) | 0.475 s | 0.459 s |
| + compressed storage, 4x2 rebalanced (0.40) | 0.448 s | 0.429 s |
| + one exchange per substep, 4x2 (0.25) | 0.449 s | 0.428 s |
| 8x1 strips, land weight 0.25 | 0.416 s | 0.418 s |
| 8x1 strips, land weight 0.10 | 0.404 s | 0.404 s |

Final verification with the rebuilt main binary (all bit-identical):
MiniPac 1x1 4-step vs 4ref; 2x2 4-step vs 4r; 4x2 two-node 4-step vs 4r;
2x2, 4x2 and 4x1 3-day vs 3dv3 (median step 0.133, 0.087, 0.137 s; 2x2
was 0.144 before this round, so strips do not help MiniPac on one node);
Pac 4x2 4-step vs pac8; Pac 8x1 3-day (default land weight 0.10) vs
pac3d8w4m, median 0.404 s, peak GPU memory 27.3 GB.

Not done: a 4-cell halo to exchange every second fast step. With a 2-cell
halo one exchange per substep is the minimum (ubar(knew) at a face depends
on fields 2 cells away within a substep); a 4-cell halo means widening every
array while keeping the 3-D exchanges at width 2. Remaining fast-loop idle
is ~32 ms per step, of which perhaps 10-15 ms is the exchange rounds this
would halve (~2-3%).

## Mixing skip, exchange audit, and where a Pac step goes (2026-10-08)

Lateral mixing only near nonzero coefficients: Pac has visc2 = 0 and
diff2 nonzero only in the sponge band. `build_mix_map` (gpu_state) marks
the cells within 2 (`mixact`) and 3 (`mixact3`, visc3d stress points) of
a nonzero visc2_r/visc2_p/diff2. A neighbourhood that leaves the array
counts as active. `mixc` lists the active wet columns for the passive
t3dmix launch. t3dmix and visc3d test the mask after the dry test.
`ROMS_GPU_SKIP_MIX=0` turns the skip off. Active wet cells per Pac strip:
0-12%. Pac 3-day mean step 0.404 -> 0.381 s, bit-identical.

Exchange audit: `ROMS_GPU_HG_AUDIT=1` makes `hg_end` copy every field's
slab before the exchange and count the halo values the exchange changed,
keyed by the exchange's position in the step (`hg audit rank R seq S`
lines at the end). Positions differ on the first run step, so read the
table from the second step on. Over MiniPac 2x2/4x2 3-day and Pac
4x2/strips 24-step runs, 16 group exchanges per step never changed a
value, 2 changed only on the first step, and 3 groups carried
never-changing fields. Each removal below also has a reason in a comment
at the call site:

- set_depth: z_w, z_r, Hz are formed over the whole array from zeta(knew),
  whose halo is current at every call: no exchange.
- rho_eos(tidx, stage): stage 1 (predictor), 2 (corrector), 3 (end of
  step). T is exchanged only at stage 3, plus stage 1 on the first run
  step. The geometry is never exchanged here. The results (rho1, qp1,
  rho, bvf, rhoA, rhoS) are not exchanged at stage 1 except on the first
  run step (`is_first_run_step`), because they equal the halo stage 3 of
  the previous step left.
- lmd_vmix: only the surface forcing (sustr, svstr, stflx t/s, srflx).
- Removed outright: the opening exchanges of prsgrd, pre_step3d,
  set_bottom_drag, visc3d and step3d_t.
- step3d_uv1: only u, v(nnew).
- tracer_halos: first run step only.

Pac 3-day mean step 0.381 -> 0.348 s, walltime 4:57 -> 4:40 (both items).
Bit-identical: MiniPac 1x1/2x2 4-step, 2x2/4x2 3-day; Pac 4x2 4-step;
strips 3-day history and restart.

Where the time goes now. Do not read idle time from an all-rank nsys
profile. It showed ~80 ms of idle per step, including 20 ms gaps at the
same wall time on all 8 ranks at random fast substeps: profiler stalls on
one rank, spread by the exchange chain. Without nsys the step times
repeat with a 4-step (hourly) cycle:

| step in the cycle | time (s) | extra |
|---|---|---|
| 1, 2 | 0.270 | - |
| 3 (MARBL call) | 0.360 | MARBL interior 73 ms + carbonate 11 ms of GPU |
| 4 (forcing read) | 0.433 | ~150 ms in the corrector's set_forces (host) |

Clean steps take 0.267-0.272 s against 0.248-0.261 s of GPU busy on the
slowest rank. So overlapping the 3-D exchanges and a wide fast-loop halo
can recover at most ~10 ms per step together. The wide halo would need
halo 3 (not 4) plus two time levels exchanged every second substep,
because AB3 reads the level formed by the unexchanged substep. It would
also need the OBC and volume sources applied in the seam halos at
physical edges. The phase table comes from `ROMS_GPU_STEP_MARKS=1`
(lines come in groups of 8, one per rank).

The forcing read (`read_catalog_rec`) opens the file, reads one record
of the tile window, closes, and copies to the device. When a series
advances by one record it rereads both brackets (`advance_series`),
although the new lower bracket is the old upper one.

A GPU-aware MPI benchmark (8 ranks on 2 nodes, chain exchange of device
buffers):

| setting | 52 MB message |
|---|---|
| default (CUDA IPC) | 3-4 ms (13-18 GB/s per direction, inter-node included) |
| MPICH_GPU_IPC_ENABLED=0 | 36 ms |
| MPICH_ASYNC_PROGRESS=1 | 2x slower |

The all-tracer exchange (~50 MB per side on the strips) costs ~4 ms of
transfer.

## Forcing reads while stepping (2026-10-08)

The hourly surface forcing cost ~150 ms per Pac hour (every 4th step,
inside the corrector's set_forces).

- The Pac files are CDF5 classic: 33 GB per month, 7 float32 record
  variables of 962x1858 (7 MB per variable and hour).
- Every rank read a strided window of every variable. For the strips
  that spans nearly every row, so each rank pulled each whole slab.
- When a series moved on by one record, `advance_series` reread both
  brackets.
- The file was opened and closed per record.
- A cold 7 MB slab read on Lustre takes 35-180 ms; a warm one ~14 ms.

`ROMS_GPU_FRC_BATCH` (bitmask, default 7; 0 = the former path):

1. Slot reuse: the new lower bracket is the old upper one, so slot 2 is
   copied into slot 1 on the device (`copy_upper_to_lower`, a D2D
   cudaMemcpy) instead of reading that record again.
2. One read per record: `set_forces` and `set_bgc_forces` advance every
   series first with the reads deferred (`frc_batch_begin`).
   `frc_batch_flush` then reads each needed record whole, once, on rank
   mod(r-1, nranks), through a handle kept open for the run
   (`frc_run_ncid`), and sends every tile its window. Files with time as
   the first Fortran dimension keep the window reads.
3. Read-ahead: `frc_prefetch`, called right after step3d_t, reads the
   record after each series' upper bracket.
   - It runs while the passive-corrector kernels are queued.
     marbl_gpu_step drains the device on some calls, so the call sits
     before it.
   - Each rank reads at most one record per call. The windows go out
     with non-blocking sends, one fixed tag per series.
   - The flush that needs the record only waits for its messages. A
     prefetched record that turns out not to be the needed one is dropped
     and read as in 2.
   - `frc_prefetch_drain` completes leftovers before finalize.

The bytes that land in a tile are the window read's (float32 -> double
is exact), so results are unchanged.

`ROMS_GPU_FRC_TIMING=1` prints every flush and prefetch. Pac hourly flush:
60-115 ms with 1+2 (`records 7 ... whole T`), 4 ms with 1+2+3 (`prefetched
7`). The prefetch reads cost the owner 15-70 ms of host time, mostly
hidden behind the queued kernels.

| Pac strips 3-day | mean step | 4-step cycle (s) | walltime |
|---|---|---|---|
| before | 0.348 s | 0.283 0.272 0.383 0.455 | 4:40 |
| slot reuse + one read per record | 0.321 s | 0.287 0.272 0.398 0.327 | 4:36 |
| + read-ahead (final) | 0.308 s | 0.291 0.273 0.378 0.290 | 4:33 |

Bit-identical for every build:

- MiniPac 1x1 and 2x2 4-step.
- MiniPac 2x2 and 4x2 3-day. These cross into April; one flush there
  needed 12 records with only 4 prefetched, which exercised the fallback.
- Pac 4x2 4-step.
- Pac strips 3-day history and restart.

What is left of a Pac hour beyond the clean steps: the MARBL call (~105 ms,
of which ~85 ms MARBL kernels) and ~35 ms around the forcing step.

## Kernel round after the full-code review (2026-10-08)

A read-only review ranked the remaining bit-identical work by measured
kernel time. Four agents then each worked in their own copy of the tree on
disjoint files, under a common written brief, while I
did the exchange and fast-loop items. A change was kept only if it passed
four tests: MiniPac 1x1 and 2x2 4-step, Pac 4x2 4-step, and MiniPac 2x2
3-day. The merged build was then retested on the full set.

**Passive tracers** (`passive_column.F90`, `halo.F90`):
- The all-tracer halo pack/unpack had the 2-wide halo index innermost: 97%
  of lanes were idle and 8 B were used per 32 B sector. Buffers are now
  tracer-innermost, `(h*ntr, j, nz)` and `(h*ntr, i, nz)`, and the W/E and
  S/N unpacks run in one launch each. 10.2 ms -> 0.5 ms per step.
- `horiz_kernel` became `c4_kernel` (predictor) and `up3_kernel`
  (corrector):
  - a 2-D grid replaces the runtime div/mod;
  - each thread does 20 levels, with the per-column stencil and mask work
    hoisted out of the level loop;
  - time slots are passed as separate arrays by sequence association, so
    loads take the read-only path;
  - every kernel in the module carries `launch_bounds`. With nvfortran
    25.9 one bound in a module changes the register limit of the module's
    unbounded kernels.
  - 56.5 -> 26.3 ms.
- `column_col` uses a sliding window (dlt_m(k) = dlt(k-1), dlt_p(k) =
  dlt(k+1): same operands and order), 4 divides per level instead of 6,
  and a peeled interior. 41.5 -> 33.9 ms.

**KPP** (`lmd.F90`):
- One column kernel computes the 3x3-smoothed Rig inline, runs the interior
  mixing recurrence, and stores Akv/Akt/ghat once. It replaces the 5
  smoothing kernels, lmd_interior and the per-level KPP kernel.
- The Vtsq and Rig work is folded into the KPP column sweep. lmd_wm/ws
  are called only when ssgm < 1, and their last result is reused when the
  inputs repeat. Dead stores and zero_mix_coefs are gone (proof in the
  comment of `lmd_zero_deep_frame`).
- lmd_swr_frac runs once per step: Hz is unchanged between the two
  lmd_vmix calls, and every reader is interior.
- Probe steps keep the old path. 30.2 -> 10.2 ms.

**Dycore** (uv_rhs, prsgrd, advect, geom/omega, step3d_t, pre_step3d,
step3d_uv, column_solve):
- `fomn` is a static 2-D array.
- The horizontal momentum kernels are split into upstream and centered
  versions with point pairs and `(32,16,1)` blocks: 13.4 -> 5.7 ms.
- prsgrd: a single-sweep column kernel, a frame-only zero kernel and point
  pairs: 12.2 -> 6.6 ms.
- Vertical advection carries the thickness slopes up the column (4
  divides per level instead of 9): 8.5 -> 4.0 ms.
- add_vert_uv_rhs keeps its recurrences in registers.
- The omega zero kernel zeroes only Wi(0) and the halo frame.
- The step3d_t forcing is fused into a new `implicit_tracer_columns_frc`,
  and the AM3 work is moved into the pre_step3d Thomas sweeps.
- 26.6 ms per step in total.

**MARBL** (`marbl_interior.F90`, `marbl_sflux.F90`, `marbl_step.F90`,
new `marbl_carb_helpers.inc`):
- The PFT counts are compile-time parameters (NAUTO=3, NZOO=1, NPREY=3;
  set_marbl_par stops if the instance differs). The per-PFT loops are
  fused and unrolled, tracers are read in place, and tendencies are
  applied by `emit_tend` with the same expression. The kernel stack went
  from 2528 B to 280 B, still at 255 registers.
- `prd` lives in constant memory.
- The temperature function is computed once per level and shared by PFTs
  with the same parameters.
- The carbonate helpers are included into the module, so they inline.
- zsat is computed only once history staging starts.
- The solver-failure report is a device reduction.
- interior 71.5 -> 29.8 ms and carb 11.0 -> 6.3 ms per hourly call.

**Fast loop** (`obc.F90`, `step2d.F90`, `halo.F90`):
- u2dbc and v2dbc run as one launch (`uv2dbc`). They write different
  arrays and read each other only at kstp.
- The zeta/ubar/vbar exchange waits on an event recorded after its pack
  instead of on the whole stream.
- accumulate_boundary_avg1 is queued behind the pack, so it runs during
  the MPI transfer. It reads only physical-edge cells, which no exchange
  writes.
- The W/E unpacks run in one launch.
- Fast-loop phase 22.7 -> 17.4 ms per step.

Pac 8x1 strips, 3-day run:

| | mean step | median | p10 | 4-step cycle (s) |
|---|---|---|---|---|
| before (098e783 + forcing work) | 0.308 s | 0.275 s | 0.271 s | 0.291 0.273 0.378 0.290 |
| after | 0.201 s | 0.178 s | 0.172 s | 0.186 0.196 0.221 0.202 |

- GPU busy on a clean step: 257.5 -> 163.3 ms. On the MARBL step: 342.8 ->
  200.0 ms.
- Stepping is 91 -> 64 s, but the walltime stays ~4:30. Init (~97 s) and
  the end-of-run writer backlog (~107 s after the last step) are now 75% of
  a 3-day run.
- `ninfo = 24` instead of 1 saves ~1 ms per step. It is now set in
  `cases/pac/gpu/namelist_roms_gpu_pac3dS10.nml`.

Notes:
- **Pre-existing race, fixed 2026-10-28.** u2dbc's closed-edge Neumann copy
  `ubar(m, 0, kn) = ubar(m, 1, kn)` at m = nx+1 read a cell that thread 1
  writes in the same launch. The closed-edge code now follows the CPU
  (gamma2 times the masked inside value), and its end faces are set in the
  serial uv2dbc_corners launch after uv2dbc, so no thread reads a cell
  another writes.
- **Latent, non-default case.** With a non-default `ROMS_GPU_SKIP_DRY`
  that skips G_MOM but not G_LMD, visc3d reads stale scratch at deep
  columns. Those values now differ from the old code's.
- **Builds share TMPDIR.** Concurrent builds in the shared TMPDIR (set by
  ncarenv) collide with "IILM file version error". Give each build its
  own TMPDIR.

## Per-run round: init, output and placement (2026-10-08 evening)

Pac 8x1 strips, 3-day, starting from ed156ed: walltime 275 s = init 97 s +
stepping ~64 s + ~107 s after the last step (writer backlog plus shutdown).

**Kept (bit-identical):**

- **C, forcing time axes.** `prefetch_time_axes` reads only the axes the
  binders request (`TAXES` in frc_io; any other axis is read by its
  binder). They are read through a handle opened with an 8 KB netCDF block.
  The hourly files carried two unused 744-record axes, and each value of a
  record variable cost a 4 MB block read. Forcing init 10.7 -> ~8 s.
- **D, init leftovers:**
  - bind_series, bind1 and bindz no longer read records 1-2 of
    multi-record series (the first advance reads the bracket it needs);
  - the boundary series reuse the upper slot and keep their files open
    (`frc_run_ncid_path`);
  - output staging set 2 is pinned only when a record finds set 1 still in
    flight (`set_free`);
  - sanitize_land, pad_physical_halo and the initial slot copies run as
    device kernels (the host loops migrated ~2 GB of managed state per rank).

  Boundary init 2.6 -> ~0 s, output init 12 -> ~8 s, cheaper daily
  boundary steps.
- **E, NUMA placement** (`tools/numa_bind.sh`, `--cpu-bind none`; used in
  cases/pac/gpu pac3dS10; `submit_gpu.sh` passes `NUMA_BIND_NRANKS`).
  - PALS pins every rank to one core of NUMA 0 (cores 0-4), while GPU 0..3
    sit on NUMA 3..0. The wrapper binds each GPU rank to its GPU's domain
    (read from nvidia-smi and /sys) and leaves the writers unbound. Binding
    the writers too slowed their file writes ~20%.
  - The step mean drops 0.196 -> 0.186 s (the MARBL and forcing steps gain
    most).
  - The end-of-run output is ~8 s slower on average: the restart and
    history parts now cross NUMA domains on their way to the writers.
  - Roughly neutral for a 3-day run; ~5% for long runs.
- **Writer timing:** each writer prints extend / assemble / put / sync per
  record.

**Measured and dropped:**

| Item | Result |
|---|---|
| B: size-balanced, non-lockstep restart reader | No gain. Reads are file-system bound (33-40 s per rank, ~3 GB/s aggregate); every rank still finishes at ~56 s. |
| 64 MB netCDF block for the restart file | Read 59 -> 93 s. |
| A1: 8 writers | Same put time per record. |
| A2: NOFILL | No change in the extension round. Its run also landed on deg0072. |
| A4: non-blocking part receives | No measurable gain. |
| A5: fixed 24-stripe output directory | Worse: record 3 took 34 s put + 22 s sync. |

**Write benchmark** (a standalone Fortran program, not in the repository): netCDF-4 parallel,
writers' pattern, 14 GB per file.

| Case | Aggregate |
|---|---|
| 1 file, 4 ranks | 0.75 GB/s |
| 2 files at once, one group per node | 1.23 GB/s |
| 2 files at once, both groups on one node | 1.00 GB/s |
| 1 file, 8 ranks | 0.50 GB/s |

Bandwidth is limited per file, and more writers on one file hurt. History
record 3, which crosses the 64 GB PFL component boundary, spends 17-18 s in
the collective extension round in the default directory.

**Result:** 3-day Pac walltime 275 -> ~250 s unbound (260-270 s bound; file
system variance is +-20 s).

**Open:**
- **A3:** write the final restart concurrently with the last history record
  (another file). From the benchmark, ~15-20 s per run.
- **Long runs are output-bound.** A daily Pac history record takes 4
  writers ~23-25 s, while a model day is ~18 s of compute. After about a week
  the compute ranks wait on the writers. The fix is to write different files
  concurrently, e.g. writer groups with one history file per record.

## Writer groups: different files written at the same time (2026-10-08 night)

From the per-run round's write benchmark, one file's bandwidth is limited
(~1-2 GB/s; 8 writers on one file are slower than 4), while two files at
once write ~1.3-1.6x faster. io_server therefore splits the writer ranks
into groups.

How it works:

- **Group size.** `ROMS_GPU_IO_GROUP` writers per group; the default,
  min(4, max(1, nwriters/2)), is lowered until it divides the writer count.
- **File routing.** File `nfile` of kind `fkind` (history 1, restart 2)
  belongs to group mod(nfile-1 + fkind-1, ngroups). So the restart and the
  history file of the same number, and consecutive history files when
  nrpf_his splits the history, go to different groups.
- **Collectives.** Each group has its own communicator for the parallel
  open, the record extension and the sync.
- **Sends.** Compute ranks send a record (data, headers, open command) only
  to its group's writers.
- **Dispatch order.** Records are dispatched in staging order (a sequence
  number per record), not by staging-set index. The lazy second set
  introduced in the per-run round let ranks place the same record in
  different sets, so with a history record and a restart staged in the
  same step, two ranks could send them in opposite orders. The writers'
  collective calls need one completion order per group, and staging order
  guarantees it.
- **Layout.** `tools/gpu_layout.py` takes writers per group from the record
  size (at most 4), and 2 groups when the run writes more than one file at
  a time (history and restart, or nrpf_his smaller than the number of
  history records); `--groups` overrides. Pac: 8 writers (2 x 4); MiniPac: 2
  (2 x 1); no output: 1.

Pac 8x1 strips, 3-day, 8 GPUs:

| output | writers | walltime | stepping | after the last step | worst step |
|---|---|---|---|---|---|
| daily history + restart | 1 group of 4 (before) | 244 s | 68 s | 95 s | 3.4 s |
| daily history + restart | 2 groups of 2 | 245 s | 70 s | 96 s | 0.7 s |
| daily history + restart | 2 groups of 4 | 223 s | 68 s | 66 s | 3.8 s |
| 12-hourly, one file per record | 1 group of 4 | 297 s | 126 s | 81 s | 12.4 s |
| 12-hourly, one file per record | 2 groups of 2 | 270 s | 96 s | 95 s | 9.7 s |
| 12-hourly, one file per record | 2 groups of 4 | 226 s | 75 s | 58 s | 3.8 s |

- **Groups of 2 are too small.** Two writers make one file writer-bound:
  a history record takes 28 s instead of 20, the restart 51 s instead of 33.
- **Groups of 4 deliver both gains.** They overlap the restart with the last
  history record (-29 s at the end of the run). Under heavy output they keep
  stepping nearly unthrottled: before, a daily-or-faster Pac history made the
  model wait on the writers.
- **One run had a hiccup.** One 2 x 4 run had a 40 s put on one writer, which
  stalled the model for 54 s; the repeat was clean.
- **Bit-identical:** MiniPac 2x2 4-step and 3-day (1 and 2 writers), Pac
  4x2 4-step, Pac strips 3-day history and restart for every layout, and all
  seven per-record history files of the heavy-output run (2 x 2 against
  1 x 4).

**For long runs.** If the writers cannot keep up with the output, split the
history (small nrpf_his, e.g. 1-5) so consecutive records go to different
groups, and add groups (`--groups 3` or 4 in the layout tool). A single
history file can only be written as fast as one file allows.

## Feature check and rivers (2026-10-16)

### Stop on CPU features the GPU does not have

The GPU reads only the namelist groups it knows. Before this change, a CPU
namelist that switched on, say, CDR forcing ran without it and without a
warning. `src/host/feature_guard.F90` now scans the namelist text on every
rank before the writer split.

- **Stops:** a switch that is on for a feature the GPU lacks stops all ranks
  with `error stop`, and world rank 0 lists the switches. These are pipes,
  CDR forcing and the four CDR outputs, analytical rivers, analytical tides,
  upscale, extract_data, z-slices, particles, random-point output, forcing
  and surface-flux output, averages (physical and BGC), monthly restarts,
  and `wrt_w` with history on.
- **Warnings only:** diag_uv, diag_trc, ub_tune, wrt_sponge and calc_pflx.
  These act on the CPU only with a cpp key the GPU cannot see, or they only
  add nesting fields to the restart.
- **Checked:** all 475 GPU namelists in the case directories pass. MiniPac
  output is byte-identical with the check in place.

### Rivers

UCLA `river_frc`: a river cell is a land cell with river_index > 0. Each of
its faces to an ocean cell carries river_fraction / (number of such faces)
of the river's volume flux. The CPU model treats these faces in five
places, and the GPU repeats its expressions there:

| CPU routine | what the river face gets | GPU |
|---|---|---|
| step2d | ubar/vbar(knew) = flux·2/(dn·(Dnew(a)+Dnew(b))), DU/DV_avg1 = flux, every fast step | fused into step2d_mid's last kernel; `rivers_ubar` for faces within 2 cells of a physical edge |
| pre_step3d | u/v(nnew) = q·frac/(dn·riv_depth), before u3dbc | `rivers_uv(RIV_PRE)` |
| step3d_uv2 | u/v(nnew) as above, before the exchange | `rivers_uv(RIV_COR)` |
| tracer advection, predictor and corrector | FX = riv_trc·½(Hz(a)+Hz(b))·q·frac/riv_depth | `rivers_ts` (T, S), `rivers_tp` (passive) |
| set_HUV, couple2, get_init | river faces count as wet (umask + riv_umask) | same masks |

- **Time interpolation.** River values are at time n in the predictor and
  at n+1/2 in the fast loop and the corrector, with the CPU's set_frc_data
  weights. `river_io` reads two records only when the bracket moves on
  (daily), so the per-step cost is kernels only.
- **Masks and restarts.** riv_umask/riv_vmask come from the initial file,
  as in the CPU model, and the restart writes them back.
- **Tracer flux.** The time-stepping kernels mask face fluxes with umask,
  which is 0 on a river face. So rivers_ts/_tp add the river-flux
  divergence to the ocean cell next to each face after the horizontal
  pass. Each thread handles one (level, ocean cell) and sums the cell's
  faces in a fixed order. There are no atomics, so every run and every
  tiling gives the same result.
- **Tiles.** Faces are listed over cells 0..nx+1 of each tile, as the CPU
  does after its halo-index fix (cb137a9b), so a river cell in a tile's
  halo still feeds the tile's ocean cells next to it.

**Optimization** (nsys, per time step; the rivers' kernel time and the
fast-loop override's extra time):

| version | Iceland0 rank 3 (1033 river cells in the domain) | MiniPac rank 2 (2181 river cells on the tile) | change |
|---|---|---|---|
| riv1 | 0.93 ms | | separate kernels; river tracers one thread per tracer |
| riv4 | 0.10 ms passive + 3.3 us per fast step | | override fused into step2d_mid; geometry shared with the passive kernel; river init through the cached forcing handles (it had cost 7 s) |
| riv6/7 | 0.38 ms | 0.85 ms (passive 0.62) | flux q*frac formed once per step for the fast loop and the corrector |
| riv10 (final) | 0.20 ms (0.45% of the step) | 0.50 ms (passive 0.27) | passive river kernel with 4 level chunks per thread |

- **Fused override.** step2d_mid's last kernel has a river copy, 96
  registers against 74, which runs only on tiles with river faces away from
  the edges. It costs 1.1 us per fast step. Runs without rivers use the
  plain copy and are untouched.
- **Passive kernel.** One thread per (cell, tracer, chunk of levels). The
  river tracer, sign and q/dep of each face are the same at every level, so
  they load once per chunk. MiniPac rivers_tp (nz = 100) per call: 309 us
  with a thread per level, 144 us with chunks of 10 levels, 137 us with 25
  (the final 4 chunks), 158 us with 100. The FP64 tracer read and write is
  then at about 55% of memory bandwidth.
- **T/S kernel.** The same chunking made rivers_ts slower (105 us against
  64 us per call): with two tracers there is too little work per thread to
  cover the serial level loop. It stays one thread per (level, cell).
- **Step time.** MiniPac 2x2, warm forcing cache: 64.47 ms per plain step
  with rivers against 64.30 ms without (riv10, same job). Part of the
  river work lands on ranks that wait for others anyway.
- **A false alarm.** River runs once looked 14 ms per step slower. The
  river run had gone first in its job and read the forcing files cold
  (prefetch reads 8.6 s in total against 0.3 s warm). With the order
  swapped, the extra time follows whichever run goes first.

Every optimization reproduced the previous version bit for bit. River
runs are bit-identical on 2x2, 1x1 and 4x1 tiles. A river restart is
bit-exact, except for DU_avg_bak on land, which comes from the land-velocity
bug below.

### Two cold-start bugs (found on Iceland0, fixed)

MiniPac and Pac start from exact CPU restarts, so these never showed
before. Iceland0 starts from an initial file.

1. **Land velocities.** The loaded u, v, ubar and vbar were not multiplied
   by umask + riv_umask, as the CPU get_init does. The Iceland0 initial
   file has nonzero ubar on every land face.
2. **Barotropic fluxes.** When the run is not an exact restart, the CPU's
   set_depth (iic = 0) forms DU_avg1/DV_avg1 from the initial ubar and
   zeta. The GPU left them 0, which gave 5 cm of zeta difference across the
   domain after one step. The fix is `init_barotropic_fluxes` in case_io.

### CPU commit 69a23382 ported

CPU commit 69a23382 (2026-10-08, "Changes to model results") came after
the GPU's validation build (the 2026-10-03 CPU). It changes results: on
MiniPac, zeta differs by 3.7 cm RMS after three days between the two CPU
builds. The GPU now has its three changes:

- t3dbc and passive_t3dbc use the interior tracer at nstp on all four sides
  (east and north had used nnew);
- the corrector's boundary data and the boundary tidal harmonics are at
  n+1 (frc_slot 2, `set_tides(1.0)`), while the tidal potential stays at
  n+1/2;
- the KPP 1-2-1 vertical smoothing works from unsmoothed neighbours, and
  the background is added once.

### Validation against the CPU noise floor

Reference: the CPU builds CMP/bld5_sflx_marbl (Iceland0) and
bld5_nosflx_marbl (MiniPac), which include 69a23382. GPU: riv7. riv8 and
riv10 reproduce it bit for bit.

**Per step** (history every step, `cases/RIV/dbg_cmp.py`). Iceland0 with
and without rivers, 7-11 steps: every physics field and every MARBL tracer
differs from the CPU by at most one float32 unit (relative 6e-8, worst
6.9e-8). MiniPac without rivers, 8 steps with 2 MARBL calls: the same on
every boundary row and ring and, for the large values, everywhere. Larger
relative differences occur only at tiny tracer values, where large terms
cancel, and in one column where a mixing instability grows from step 5.

**Three days, against CPU twins** (1e-6 degC random initial temperature
perturbation). Hour 6: GPU-CPU is 2-60% of twin-CPU for every field. By
hour 24-72 both saturate near 1. A second twin (seed 7) gives the CPU's own
spread, and the twin-2/twin-1 ratios at the last hour reach 1.3-1.4 (Fe,
Lig, zeta and PH in the river zone). Distances from the GPU to each of the
three CPU members, against the largest CPU-to-CPU distance:

| case, hour | physics, most tracers | outside the CPU spread |
|---|---|---|
| MiniPac, no rivers, 24 | 0.09-1.14 | none |
| MiniPac, no rivers, 72 | 0.70-1.12 | diatChl/C/P/Si 1.08-1.27 |
| MiniPac, rivers, 66 | 0.27-1.14 | diatoms 1.17-1.39, DOC/DON/DOP 1.18-1.33, O2 1.21-1.34 |
| Iceland0, rivers, 72 | 0.65-1.12 | temp 2.2, PH/pCO2SURF 1.5 |

The excesses are local, and none is reproducible from a common state:

- **Iceland0 temperature.** One 5 m column at -2 C next to river 107 (j366,
  i280) warmed by 0.27 K in the GPU run from hour 55. Continued from one
  CPU restart at hour 48, the GPU, the CPU and a CPU twin follow the same
  path there (-1.913 C at hour 60 in all three), and the GPU's PH
  difference stays below the twin's (2.0e-6 against 3.4e-6 RMS).
- **MiniPac diatoms.** Most of the excess is one interior cluster (j268-273,
  i232-237). The rest sits on the open boundaries, where the GPU's
  difference is 5-8 times the twins'. The boundary rows agree to one
  float32 unit step by step. A velocity sign change at a boundary point
  switches between radiating the interior value and nudging to the
  climatology. That moves diatoms, whose climatology differs from the
  interior, but not temperature. GPU round-off enters at the boundary on
  every step, whereas the twins' seed starts in the interior.
- **The seed matters, the biogeochemistry alone is not chaotic.** A CPU twin
  with every MARBL tracer multiplied by (1 +- 1e-7) stays within 1e-7 of
  the reference after three days (diatC RMS 5e-7 against 9e-3 for the
  temperature twin). Tracer divergence comes from the physics.

**MiniPac river 4.** River 4 delivers 10,080 m3/s into one 12 km cell.
Both models drive salinity there below zero by hour 72 (CPU -1.136, GPU
-1.137 at the surface), and MARBL's carbonate solver then fails. Both CPU
river twins stopped there. The riv7 GPU run counted the failures and ran
on; since riv11 the GPU stops too, on the same cell. MiniPac river
comparisons therefore stop at hour 66.

**Output difference.** Before the first MARBL call of a cold start
(Iceland0: the first hour), the GPU writes 0 for the MARBL surface
diagnostics; the CPU writes the initial-state values.


MiniPac's river test file (`cases/RIV/input/MiniPac_river.nc`) is cut from
the Pac river file, with tracer concentrations capped. The Pac file has
DIC and ALK up to 2e7 mmol/m3, which MARBL's carbonate solver cannot
handle.

## CDR port (2026-10-17): CPU reference and the CDR_LITE tracers

### CPU reference

- **Source and keys.** CWorthy-ocean/ucla-roms main 1c9fcfae, in a clone of
  its own. Built in cases/CDR/bld_cpu_ref with the validated Iceland0 keys
  (cases/CMP/bld5_sflx_marbl) plus CDR_FORCING and CDR_LITE.
- **CDR off.** It matches the validated build bit for bit (Iceland0, 12
  steps, every history, BGC and diagnostic file).
- **Changes since the validated base (accb74d9).** Only CDR code, the
  use_constant_Fe hooks (no-ops by default) and an I/O sync.
- **UPSCALING left out.** Compiled in (with MARBL and MARBL_DIAGS), it
  changes results even with do_upscale off: compute_horiz_tracer_fluxes.h
  (89893a7c) then uses upwind inflow fluxes for every tracer at the
  open-boundary faces. With CDR off this gave 1.5e-3 K in temperature after
  one step. Key bisection showed CDR_FORCING and CDR_LITE alone are
  bit-identical; UPSCALING alone reproduces the whole difference. Decision:
  this boundary treatment follows do_upscale, so it is on only in runs that
  write upscaling output (GPU, and our CPU reference when upscaling is
  ported).
- **CPU issues met while setting up:**
  - With CDR_LITE compiled in, ddic_dco2/ddic_dalk are read every step even
    with no CDR_LITE tracers, and the run stops if they are absent.
  - cdr_lite_output and cdr_gas_exch_output hold file names in
    character(len=99), so longer paths are cut. Our runs write through a
    short symlink to the output directory.
  - The ulla surface-physics files carry zero CDR _flx variables, with
    CDR_time only in 202212, which stops the CPU's record search in
    202301. Our runs list a CDR_time file first.
- **Test case.** cases/CDR/gen_cdr.py builds Iceland0 (ulla) variants from
  the validated CPU/GPU namelist pair: off, full, online, and lite_io (the
  transport test below).

### CDR_LITE tracers on the GPU (step 2: tracers and their inputs)

- **Layout.** CDR_LITE tracers are linear ALK/DIC anomalies: nt_cdr_oae
  ALK/DIC pairs and nt_cdr_dor DIC. MARBL never sees them. They follow the
  MARBL tracers in tp, so the MARBL kernels still address their tracers as
  1..32. The names are the CPU's (CDR_OAE_ALK<n>, CDR_OAE_DIC<n>,
  CDR_DOR_DIC<n>). The CPU orders all tracers T, S, generic, CDR_LITE,
  MARBL; inputs indexed by that order map through cpu_tracer_index.
- **Rivers.** river_tracer may hold every tracer or every tracer but the
  CDR block, as on the CPU. Each passive tracer maps to its row (rv_map);
  a CDR tracer the file lacks maps to an extra zero row.
- **Initial file.** A missing tracer stays 0, tracer by tracer.
- **Boundaries.** Boundary data are bound for the MARBL and CDR_LITE tracers
  (passive_bry); missing ones are 0 on inflow.
- **Restarts and history.** They carry the CDR tracers by name.
- **MARBL.** MARBL's surface fluxes go to its own tracers only.
- **Transport.** As for MARBL tracers: the salinity diffusivity, no KPP
  nonlocal term, harmonic mixing with salinity's coefficient. The CPU uses
  each tracer's tnu2; the two agree while tnu2 is 0 for every tracer, as in
  every case here.
- **Surface forcing.** Ported in step 4 (below). Until then a run with
  CDR_LITE tracers stopped unless ROMS_GPU_CDR_LITE_TRANSPORT_ONLY=1; that
  switch no longer exists.
- **Regression.** With no CDR tracers the build (cdr1) is bit-identical to
  riv11 (MiniPac smoke, rivers and MARBL).

**Transport test (gen_cdr.py lite_io).** Nonzero CDR_LITE initial,
boundary and river data (tools/make_lite_inputs.py; the ulla CDR inputs are
all zeros). No releases. No gas exchange: ddic = 0, which the CPU skips. No
surface flux. 12 steps on Iceland0, CPU against GPU every step:

- **Physics and MARBL tracers.** Within one float32 unit, as before.
- **CDR tracers.** The largest absolute difference is one float32 unit of
  each tracer's largest values.
- **Relative differences.** Only 11 of 14 million points differ by more
  than 1e-7 relative, and the largest (2.5e-6) is one column beside a river
  (j354, i259). There, at step 5, as river water arrives, a vertical
  redistribution of 2e-6 appears at levels 22-25 and then decays. This is
  consistent with the upwind vertical advection switching direction on a
  near-zero vertical velocity: round-off-triggered, of a size set by that
  velocity.

**Three days** (cpu_lite_ref, the 1e-6 degC twin cpu_lite_pert, and
gpu_lite). The CDR_LITE tracers stay within the CPU's own sensitivity:

| hour | GPU/twin ratio range, six CDR tracers (whole domain) |
|---|---|
| 6 | 0.26-0.47 |
| 24 | 0.75-0.96 |
| 48 | 0.81-0.90 |
| 72 | 0.92-0.99 |

The river zone (0.28-0.99) is similar. The physics and MARBL ratios are
identical to the Iceland0 river test, as they must be: the CDR tracers feed
back on nothing, and the physics is bit-identical to riv7. That includes the
chaotic temperature hotspot at hour 72 (see "Validation against the CPU
noise floor" above).

### CDR releases on the GPU (step 3: parameterized cdr_frc)

- **Scope.** The CPU's parameterized releases (cdr_frc_settings with
  cdr_source and cdr_forcing_parameterized, without cdr_volume):
  cdr_ncdr_parm releases, each a location, depth and horizontal/vertical
  scale (cdr_lon/lat/dep/hsc/vsc), with a flux in mmol/s per record and
  tracer in cdr_trcflx (the CPU's tracer order). They go into MARBL and
  CDR_LITE tracers alike. Volume releases, depth profiles and 3-D forcing
  stop at start-up (read_cdr_namelist and the feature check). As on the CPU
  (check_no_ts_flux), a nonzero temperature or salinity flux stops the run.
- **Placement (src/host/cdr_io.F90, once at start-up).** The CPU's
  algorithm on the global grid: haversine distance from every cell, land at
  1e8 m with cdr_relocate_to_wet_pts, the nearest cell the first minimum.
  hsc = 0 gives a point release, otherwise exp(-(d/hsc)^2) on wet cells
  above 1e-3. vsc = 0 puts the release on the level nearest -dep, otherwise
  a Gaussian in z times Hz/(vsc*sqrt(pi)). Both use the rest-state grid
  (zeta = 0) in the CPU's expressions. Each release is normalized by its sum
  over the domain, summed in the CPU's order. Rank 0 prints each release's
  nearest cell; the distances are the CPU's to the last digit.
- **Time.** The records are taken at the corrector's forcing time (n+1/2)
  with the CPU's bracket rule. Weights are c1 = (t2-t)/(t2-t1),
  c2 = (t-t1)/(t2-t1), or the earlier record held without
  cdr_time_interpolation.
- **Device (src/device/cdr_src.F90).** Each tile lists its release cells,
  with each cell's profiles in release order (CSR), plus two flux records.
  The corrector adds dt*pm*pn*w(k)*flux to the thickness-weighted tracer
  after the horizontal pass and the rivers, release by release, as the
  CPU's cidx loop does.

**Optimization.** The CPU adds to every tracer at every level of a release
cell. Where a tracer's flux is 0 in both records, or a level's weight is 0
in all of a cell's profiles, the addition is an exact zero. cdr3 therefore
visits only the active tracers (nonzero flux in either record, rebuilt when
a record loads) and each cell's range of nonzero levels. One thread handles
each (tracer, cell) pair and one of 4 chunks of levels, with the pairs
across the warp. With the 8 test releases, compute rank 2 has 16,014
release cells:

| version | cdr_apply_tp per call (rank 2) | layout |
|---|---|---|
| cdr2 | 2116 us | one thread per (level, cell, tracer), all 38 tracers; 61.5M threads |
| cdr3 | 296 us | active tracers (7 of 38) x cells, 4 level chunks; 448k threads |

cdr3 reads and writes about 55% of the memory bandwidth bound, as rivers_tp
does. Over 3 days, the median step is 58.6 ms with releases (cdr3) against
59.0 ms without (cdr1). The difference is within node-to-node noise, so the
releases cost about 0.5% of a step.

**Test (gen_cdr.py rel; tools/make_release_test.py).** Eight releases on
Iceland0 that cover the CPU's paths: point and single-level releases, a deep
release (200 m), two tracers in one release, two releases on the same cells
and tracer, and three records (8400.5, 8402.5, 9131 days, scales 1, 2 and
0.5), so a 3-day run crosses a record change. The releases go into DIC,
ALK and five CDR_LITE tracers, on top of the lite_io transport test.

- **Every step (12 steps, with and without time interpolation).** Physics
  and tracers are within one float32 unit, as in the transport test. The
  release signal (with releases minus without, tools/rel_signal.py) matches
  the CPU's. For ALK, a signal of 7.2 mmol/m3 differs by 2.4e-4, one float32
  unit of ALK (~2300). For the CDR tracers it differs by 1-2 units of
  their values.
- **Three days (cpu_rel_ref, the 1e-6 degC twin cpu_rel_pert, gpu_rel).**
  GPU/twin ratios at hour 72 are 0.92-0.99 for the six CDR tracers, 0.989
  for DIC and 0.983 for ALK. The release zone (within 3*hsc + 50 km of a
  release, at least 100 km; compare.py --release) gives the same. The
  physics and MARBL ratios are identical to the transport test's, hotspot
  included, since the releases feed back on nothing.
- **Release response (tools/rel_signal3d.py).** The GPU's error in the
  release signal, (gpu_rel - gpu_lite) - (cpu_rel_ref - cpu_lite_ref), is
  compared with the twin pair's. The CPU writes double and the GPU float32,
  so 'f32 floor' is the CPU signal formed from float32-rounded fields, the
  smallest error float32 can show. Zone RMS:

  | tracer | hour | signal | GPU error | twin error | f32 floor |
  |---|---|---|---|---|---|
  | DIC | 6 | 0.21 | 2.22e-5 | 7.6e-8 | 2.22e-5 |
  | DIC | 72 | 0.80 | 3.75e-5 | 4.22e-5 | 3.17e-5 |
  | ALK | 6 | 0.20 | 1.84e-5 | 2.0e-7 | 1.84e-5 |
  | ALK | 72 | 0.45 | 2.86e-5 | 1.63e-5 | 2.39e-5 |
  | CDR_OAE_ALK1 | 72 | 0.11 | 1.1e-7 | 1.3e-7 | 4.1e-9 |
  | CDR_DOR_DIC2 | 6 | 0.012 | 2.83e-8 | 7.3e-9 | 2.83e-8 |
  | CDR_DOR_DIC2 | 72 | 0.19 | 3.4e-6 | 2.4e-6 | 3.6e-8 |

  Wherever the GPU error is larger than the twin's (DIC and ALK through hour
  48, two CDR tracers at hour 6, the ALT_CO2 tracers), it equals the
  float32 floor. The GPU therefore reproduces the release response to the
  rounding of its history. Otherwise it is within the twin's spread, and
  the derived pH, pCO2SURF, FG_CO2 and CO3 signal errors are at most 0.63
  of it.
- **Regression.** With no releases, cdr2 is bit-identical to cdr1 (MiniPac
  smoke; Iceland0 lite_io). cdr3 is bit-identical to cdr2 (rel and relni
  over 12 steps, and the 3-day rel run, which crosses a record change).

### CDR_LITE surface forcing and output (step 4)

**Surface forcing (src/device/cdr_lite.F90, src/host/cdr_lite_io.F90).** The
CPU's tracers.F90 (set_surf_tracer_flx with CDR_LITE). Each DIC tracer
(CDR_OAE_DIC<n>, CDR_DOR_DIC<n>) takes the surface flux

    stflx = <name>_flx + FG,   FG = -(k_gas/beta) * (DIC' - eta*ALK') * rmask

- **Inputs.** beta = ddic_dco2 and eta = ddic_dalk, read from the forcing
  files; both are required, as on the CPU. ALK' is the paired
  CDR_OAE_ALK<n>, 0 for the DOR tracers.
- **No exchange** where beta is not > 0, NaN included.
- **k_gas** = 6.97e-7 * (uwnd^2 + vwnd^2) * sqrt(660/Sc). The Schmidt
  number Sc is a polynomial in SST, and k_gas = 0 where SST <= -1.8 C.
- **Precision.** Sc, beta, eta and ALK' are single precision, as in the CPU
  build (default real), and the rest is double.
- **ALK tracers** have no surface flux.
- **Timing.** The CPU sets the flux in its set_forces at n+1/2, from the
  predicted state (slot 3). Only the corrector applies it, as dt*stflx on
  the top layer after the vertical advection and before the implicit
  mixing; pre_step3d applies none.
- **On the GPU.** set_cdr_lite runs after the corrector's set_forces. A
  kernel with one thread per wet column fills psf(tracer, column), and the
  passive column kernel adds dtp*psf at the top (sfo = 1, corrector only).
  The column kernel's registers and stack are unchanged, and runs without
  CDR_LITE take the old path. The gas flux costs no measurable step time
  (3-day median 58.8 ms, against 58.6 ms with releases only).

Three CPU details that matter:

- **SEA_ICE_NOFLUX is always on.** set_global_definitions.h defines it
  unconditionally, not cppdefs.opt. The first GPU version missed it: k_gas
  must be 0 where SST <= -1.8 C. On Iceland0 that is a cold patch off east
  Greenland (j 358-381, i 251-281 in the test), where the four DIC tracers
  differed by 1e-3 relative after one step. The GPU's bulk heat flux
  already had the rule.
- **CPU file order.** The forcing series are cataloged in the order the
  CPU's find_new_record meets them (bind_series cpu_order). Files go in
  list order, a record is kept only if it is later than the last one kept,
  and the catalog ends at a file that has the variable but not its time
  axis. The ulla surface-physics files carry zero CDR _flx: 202212 has one
  CDR_time record (8400.958), 202301/02 have none. A time-sorted catalog
  would bracket the test file's 8410 record with that zero record, where
  the CPU uses the test file's 8400 and 8410 records.
- **NaN in the sensitivities stays NaN** (fill_r keep_nan), as the CPU's
  set_frc_data leaves it, so no exchange happens there. Other forcing
  still turns NaN into 0. Not on the GPU yet:
  cdr_online_carbonate_sensitivity, which stops at start-up.

**Output (src/host/cdr_lite_out.F90).** The CPU's cdr_lite_output:
&cdr_lite_output_settings with every key, including wrt_cdr_lite_avg,
wrt_tracers, wrt_vertical_integrals, wrt_thickness_weighted, wrt_sources,
wrt_alk, wrt_dic and wrt_gas_exchange.

- **Files.** <root>_cdrtrc.nc, <root>_cdrtrc_0002.nc and so on, the GPU's
  naming, with nrpf_cdr_lite records each, every output_period_cdr_lite.
- **Contents.** In double, with the CPU's dimensions, variables, attributes
  and order: tracers, int_z_*, h* (the instantaneous thickness-weighted
  field, also in averaging mode), h*_avg, *_source, FG_*, and
  avg_begin_time/avg_end_time.
- **Averages.** As the CPU's calc_average: every step after the step,
  avg = avg*(1-coef) + x*coef with the single-precision coef = 1./navg,
  over the points the tile writes. Release sources come from the
  corrector's flux, summed in release order (cdr_src_out). The gas flux
  also covers the east boundary column, which the CPU's loop (i to iend+1)
  includes.
- **Writer.** A third io_server kind (IO_KIND_CDR) on the double staging
  path. Records carry avg_begin_time/avg_end_time (hdr(8), three doubles in
  the time message), and io_init8 takes the largest capacity requested.
- **Not on the GPU yet:** cdr_lite_monthly_averages, which stops at
  start-up (in the feature check too).
- **Cost.** Measured on Iceland0 over 3 days with every field averaged and
  a record every 6 h (nsys on compute rank 2):
  - **Averaging layout.** The first version (cdr4e) kept the averages by
    (i, j, k) over every written point, land included. Its 3-D and 2-D
    (vertical integral) kernels took 1.2 ms each per step, because they
    read tp at the column stride and the integral had one thread per
    column.
  - **Now.** The averages live in tp's layout (tracer innermost) over the
    wet columns the tile writes. One kernel per step, a thread per
    (tracer, column) over the levels, updates all three; records gather
    them back to (i, j, k). That takes 1.1 ms per step for 6 tracers and
    reproduces the earlier output bit for bit.
  - **Step time.** Median 59.9 ms against 58.7 ms without the output
    (cdr4h, same node). The mean, which includes the record writes, is
    67.5 against 63.8 ms.
  - **Writers.** With two writer groups, file_group put the CDR_LITE files
    in the history's group. One writer then took about 5 s per history
    record and 4 s per CDR_LITE record, and fell behind; steps waiting for
    it took 2.3-2.7 s. CDR_LITE files now share the restarts' group when
    there are fewer than 3 groups, and the slowest step is 0.29 s.
  - **Per record.** A record still costs the compute ranks about 0.4 s:
    about 2.8 GB of doubles for the domain, staged through the pageable
    double buffers of the restart path (copies at 9 GB/s, then the sends).
    For daily output that is under 1 ms per step; for hourly output pinned
    staging would be the next step.
  - **Start-up.** The averages are zeroed on the device; zeroing about
    0.5 GB of managed memory on the host cost start-up seconds.

**Test (gen_cdr.py gas; --lite-out for every output field).** The release
test (step 3) plus the real carbonate sensitivities (roms_surf_flux_2023.nc)
and the nonzero DIC _flx of Iceland0_cdr_surface_flx_test.nc.

- **Every step (12 steps).** Every field is within one float32 unit,
  including the four DIC tracers (6e-8 relative).
- **Three days (cpu_gas_ref, cpu_gas_pert, gpu_gas).** GPU/twin ratios at
  hour 72 are 0.930-0.988 for the four DIC tracers (zone 0.954-1.001) and
  0.983-0.991 for the ALK tracers. The other ratios equal the release
  test's, hotspot included.
- **Surface-forcing response.** The GPU's error in the response, (gas -
  rel), is compared with the twin's (tools/rel_signal3d.py --case gas
  --base rel --gpu-base gpu_rel3). Domain-wide ratios are 0.13-1.25 over
  hours 6-72 and 0.59-0.94 at hour 72, for a signal of 0.03-0.27 mmol/m3
  RMS in the zone at hour 72.
- **Output (12 steps, a record every 2 steps, 4 per file).** Compared with
  the CPU's _cdrtrc file by file and record by record
  (tools/cmp_cdrtrc.py): the same variables, attributes, shapes, times and
  averaging times. Largest relative differences in the averaged run:
  tracers 4e-9, thickness-weighted 6e-10, integrals 9e-13, sources 2e-14,
  gas fluxes 1e-7 (its single-precision part). The instantaneous run is
  similar, up to 8e-8 for the tracers.
- **Output does not touch the solution.** The instantaneous _cdrtrc
  tracers, rounded to float32, equal the history of a run without CDR
  output (36 of 36 fields). A run writing history, restarts and _cdrtrc
  together reproduces both bit for bit. A 3-day run with averaged output
  of every field has the same history, bit for bit, as one without
  (final binary cdr4h).
- **Regression.** With ddic = 0 and zero _flx (the step 2/3 tests) the
  build is bit-identical to cdr3. MiniPac without CDR is bit-identical to
  cdr2, which covers the io_server changes.

### CDR gas-exchange output and online carbonate sensitivities (step 5)

**Carbonate sensitivities (src/device/carb_sens.F90).** A transliteration
of the CPU's carbonate_sensitivity.F90 (compute_surface_beta_eta), with the
same arithmetic per cell. It computes beta = dDIC/dCO2 and eta = dDIC/dALK
of the ALT_CO2 surface state:

- **Constants and solve.** Lueker et al. (2000) K1/K2 on the total scale,
  the rest the MARBL set. The pH solve is a safeguarded Newton-Raphson in
  pH 2-13, with a relative test (1e-10) and a residual check.
- **Buffer factor.** The Humphreys et al. (2018) isocapnic quotient.
- **Failures.** Land, the MARBL floors and failed solves give 0.
- **Kernel.** An explicit global kernel with one thread per point; device
  helpers take arguments by reference. A CUF kernel cannot pass its
  loop-local scalars to a device subroutine (S-0528, device attribute
  mismatch). 133 registers, no spill.
- **Online.** cdr_online_carbonate_sensitivity (cdr_lite_io) computes beta
  and eta after the corrector's set_forces, from the predicted state
  (slot nrhs), over the interior and physical boundary points, in place of
  the ddic files. As on the CPU, the files are then neither read nor
  needed.

**Output (src/host/cdr_gas_out.F90).** The CPU's cdr_gas_exch_output, with
&cdr_gas_exch_output_settings:

- **Files.** <root>_cdrgas.nc, _cdrgas_0002.nc and so on, in double, with
  the CPU's dimensions, variables and attributes. The long names differ
  with wrt_cdr_gas_avg, as on the CPU.
- **Records.** An instantaneous bracket record at the start, from the
  initial state. Every output_period_cdr_gas a record follows: either
  instantaneous, or the window means beta = n/sum(1/beta) and eta =
  sum(eta/beta)/sum(1/beta) over the steps with a valid solve, stamped at
  the middle of the window. A bracket record at the end follows unless
  the last record is already there. Bracket records do not count toward
  nrpf_cdr_gas.
- **Record times.** ocean_time, ddic_dco2_time = ddic_dalk_time (the
  forcing time in days), and avg_begin_time/avg_end_time with averages.
  The files can therefore force a later run as ddic files.
- **Writer.** A fourth io_server kind (IO_KIND_CDRGAS). The record's time
  message now carries four doubles (time, the averaging window, the
  forcing time), with flags in hdr(8).
- **Not on the GPU yet:** cdr_gas_monthly_averages, which stops at
  start-up.

**Validation (12 steps, a record every 240 s, 4 per file; gen_cdr.py gas
--lite-out --gas-out [--inst] [--online]).**

- **Structure.** Instantaneous, averaged and online runs give the CPU's
  records: the same count, the same file split, and the bracket records
  at both ends. Variables, attributes and every time variable are
  identical.
- **Values.** beta agrees to 1.5e-10 relative and eta to 1.2e-11, already
  in the start record from the identical initial state. That is the
  solver's relative tolerance plus libm differences.
- **Online sensitivities.** The CDR_LITE tracers and gas fluxes agree with
  the CPU as closely as with file sensitivities (tracers 8e-8, FG 2e-7).
- **Regression.** With neither feature on, the build reproduces cdr4h bit
  for bit (Iceland0 CDR_LITE output, MiniPac).
- **Three days, online (cpu_onl_ref, the 1e-6 degC twin cpu_onl_pert,
  gpu_onl; _cdrgas every 6 h).** GPU/twin ratios at hour 72 are 0.93-0.99
  for the four DIC tracers (zone 0.95-1.00), 0.96 for ddic_dco2 and 1.02
  for ddic_dalk (zone 1.01 and 1.03); at hour 6 they are 0.10 and 0.58.
  The physics and MARBL ratios are identical to the earlier tests,
  hotspot included.
- **Cost.** The 3-day online run with instantaneous _cdrgas output has a
  median step of 59.6 ms, against 58.7 ms with file sensitivities.

### The remaining CDR pieces (stage 6, 2026-10-18)

With these, every CDR forcing mode and every CDR output of the CPU model
runs on the GPU, apart from the monthly averages of the outputs, which
still stop at start-up.

**Depth profiles (cdr_forcing_depth_profiles; src/host/cdr_io.F90,
src/device/cdr_src.F90).**

- **CPU semantics.** Each release (cdr_lon/cdr_lat) goes to its single
  nearest wet cell. Its profiles cdr_trcflx_profile (mmol/s per source
  layer) on the source layers cdr_layer_thickness are named by
  cdr_trc_name; names that are not tracers are skipped. Every set_forces
  interpolates both in time, remaps each tracer's flux per metre onto the
  cell's current layers with the CPU's flux-conserving PPM remap
  (remap_src_to_grid), and multiplies by Hz; the release flux is 1.
- **On the GPU.** The release columns are few, so the remap runs on the
  host: the CPU's vertical_remapping routines are carried over verbatim in
  cdr_io. Every corrector gathers Hz of those columns (one device sync),
  remaps, and uploads the weights cq_wt(k, profile, cell). The kernel
  cdr_apply_dp_tp adds them in the CPU's release order.
- **CPU bug (report to Scott).** A source layer of zero thickness inside a
  column gives NaN for the whole column: thomas_PPM is singular when two
  interfaces coincide (vertical_remapping.F90:347). The test input
  (tools/make_dp_test.py) avoids it. A release whose whole source column
  has no thickness is handled: it has no source.
- **Validation (gen_cdr.py dp).** Sources agree to 7e-14 relative and
  tracers to one float32 unit.

**3-D forcing (cdr_forcing_3d).**

- **Inputs.** cdr_trcflx_3d_ALK and cdr_trcflx_3d_DIC (mmol/s per cell)
  come from the forcing files, cataloged in the CPU's file order. Each
  tile reads the interior window of two records.
- **Applied** to MARBL ALK and DIC in the corrector, as
  t += dt*pm*pn*(c1*f1 + c2*f2) on the wet interior columns.
- **Validation (gen_cdr.py d3).** Within one float32 unit; the
  forcing signal matches the CPU's.

**Volume releases (cdr_volume).**

- **CPU semantics.** cdr_volume (m3/s) of water with tracer
  concentrations cdr_tracer, temperature and salinity included. The
  tracer source is volume x concentration with the release's normalized
  weights. In addition:
  - zeta(kstp) += dtfast*pm*pn*vol*sum_k(w) at every fast step (step2d);
  - Wi takes the cumulative source from the bottom up in omega, before
    the rain term;
  - T and S get their own rows (cdr_apply_ts).
- **Time levels.** The CPU's predictor omega and its first corrector omega
  use the volume of its first set_forces (time n); set_cdr_pred sets that
  volume after the predictor's forcing.
- **Omega kernel.** The CDR term raised the column kernel from 89 to 114
  registers. omega therefore has two copies of the kernel; the one without
  the term runs whenever there are no volume releases (bit-identical to
  before).
- **Validation (gen_cdr.py vol).**
  - Every 2 steps, zeta, u, T, S and the tracers are within one float32
    unit.
  - The release signal matches: zeta 2.5657e-4 against the CPU's
    2.5660e-4, temperature 6.6038e-3 against 6.6037e-3.
  - A 3-day noise-floor run with volume releases has not been done yet.

**CDR output (cdr_output; src/host/cdr_out.F90).** The CPU's
cdr_output.F90, with &cdr_output_settings:

- **Files.** <root>_cdr.nc, _cdr_0002.nc and so on, nrpf_cdr records every
  output_period_cdr, in double, with the CPU's variables, attributes and
  order:
  - zeta, T, S, ALK, DIC and their ALT_CO2 twins, PO4, SiO3;
  - the ALK/DIC vertical integrals;
  - 3-D pH, FG_CO2 and FG_ALT_CO2, surface pCO2;
  - CO3, co3_sat_arag/calc, zsatarag/zsatcalc;
  - Chl_TOT_surf and C_TOT_100m;
  - the instantaneous thickness-weighted ALK/DIC;
  - with cdr_source, the ALK/DIC sources: releases, depth profiles, volume
    or 3-D forcing;
  - with wrt_cdr_avg, the averaged thickness-weighted ALK/DIC.
- **Averaging (wrt_cdr_avg).** These are the CPU's sums:
  - every step, c0*sum + x, with c0 = 0 on the window's first step;
  - the fields that only MARBL changes are added once, weighted by the
    steps they were held, before each MARBL step and at the write;
  - at the write, sum/navg.
- **Writer.** A fifth io_server kind (IO_KIND_CDROUT). The saturation
  depths are computed at every MARBL call when this output is on
  (marbl_need_zsat).
- **Not on the GPU yet:** cdr_monthly_averages, which stops at start-up.
- **Initial MARBL diagnostics.** Found with this output. At start-up the
  CPU calls marbldrv_compute_init_diagnostics (from its first wrt_bgc):
  MARBL's surface fluxes and interior carbonate chemistry on the initial
  state, with the start-time forcing and depths. These fill the
  diagnostic arrays without touching the tracers or ROMS's copy of the
  saved state. Until the first MARBL call (an hour into Iceland0), the
  CPU's history diagnostics and its CDR output hold these values. The GPU
  had zeros there. marbl_gpu_init_diagnostics (src/host/marbl_step.F90)
  now does the same:
  - after load_case's set_forces and a set_depth, the surface-flux,
    carbonate and saturation-depth kernels on the initial state;
  - sf_ph/sf_ph_alt/it_ph/it_pha are saved and restored, so the first
    MARBL call starts from the loaded pH, as on the CPU;
  - the 3-D pH output stays at the loaded saved state (0 in a cold start)
    until then, also as on the CPU.
- **Validation (12 steps, a record every 2 steps, 4 per file; gen_cdr.py rel
  --cdr-out [--inst]).**
  - Structure: the CPU's records, file split, averaging times, variables
    and attributes.
  - Values: zeta 2e-12 relative, ALK/DIC 1.4e-9, temperature 2e-8,
    PO4/SiO3 2e-7 (the state's own per-step differences); sources 2e-15;
    C_TOT_100m 1e-10; Chl 7e-12.
  - MARBL fields: all 12 steps fall before the first (hourly) MARBL call,
    so the fields hold the initial diagnostics. CO3, co3_sat, zsat, FG
    and pCO2 agree to 5e-8 relative; pH is 0 in both (cold start).
    Before marbl_gpu_init_diagnostics, the GPU wrote zeros there. Without
    the depths, the bottom co3_sat came out at its surface value.
  - The same initial values now fill the GPU history's MARBL diagnostics
    (PH, pCO2SURF and zsat identical to the CPU's _bgc_dia; FG 1.3e-7,
    CO3 7e-8, co3_sat 1e-7). Every other field and file is bit-identical
    to cdr6e (Iceland0 gas and upscaling, MiniPac).
  - Averaging across MARBL steps (co3_*_avg: marbl_timestep 240 s, MARBL
    every 2 steps, a record every 3 steps, so the windows straddle MARBL
    steps at both offsets). Same structure as the CPU's. The 3-D pH
    agrees to 3e-10 relative: the first window's 5.485 is 0 held one step
    and the step-2 value held two. CO3, co3_sat, zsat, FG and pCO2 agree
    to 5e-8; T, S, ALK, DIC, PO4 and SiO3 to 1.2e-7 at most.

**Upscaling output (upscale_output; src/host/upscale_out.F90).** It writes
the open-boundary transects from which a parent domain's CDR forcing is
made.

- **The boundary term.** With UPSCALING compiled, the CPU adds
  0.5*(t_out - t_in)*inflow to the horizontal tracer flux through every
  face on a physical edge (compute_horiz_tracer_fluxes.h): upwind on
  inflow, for every tracer, in the predictor (C4) and the corrector (UP3).
  Main gates it by the key alone, so it changes runs that write no
  upscaling output. Scott decided (2026-10-17) that it follows do_upscale.
  The GPU applies it with do_upscale (gpu_state uscl_obc):
  - edge code 2 in advect (per-tracer and fused T/S kernels) and in the
    passive C4/UP3 face routines;
  - masked like the face, so a river face keeps only its river flux, as
    the CPU's later override.
- **CPU reference change (roms_cdr_ref working tree, uncommitted).** The
  four boundary blocks in compute_horiz_tracer_fluxes.h test
  `do_upscale .and.` the edge. pre_step3d_mod uses do_upscale, and
  upscale_output gives it a default of .false. The diff is
  cases/CDR/bld_cpu_uscl/upscale_gate.patch; the build is
  cases/CDR/bld_cpu_uscl (the reference keys plus UPSCALING).
- **Output.**
  - Every step after step3d_t: the outward advective flux of ALK, DIC,
    their ALT_CO2 twins and the CDR_LITE tracers through each open-boundary
    face. It is the corrector's UP3 face with the term, recomputed from the
    same operands (slot nrhs).
  - The running means (the CPU's calc_average, avg*(1 - 1/n) + x/n) of:
    ALK - ALK_ALT_CO2, DIC - DIC_ALT_CO2, each CDR_LITE flux, and the
    boundary cells' layer thickness, corners included.
  - Records as the CPU's wrt_upscale: one at the first step, then every
    output_period_uscl, nrpf_uscl per file, stamped with the step's start
    time. The final call after the loop adds the last step's fluxes once
    more and writes if a record is due.
  - Files <root>_uscl.nc, _uscl_0002.nc and so on, in double, sides north,
    south, east, west, with the boundary lat/lon. Compute rank 0 writes
    them from a sum of the boundary lines over the ranks
    (comm_reduce_sum).
- **Validation (12 steps, a record every 2 steps, 4 per file; gen_cdr.py rel
  --upscale, CPU bld_cpu_uscl).**
  - CPU gating: with do_upscale off, the UPSCALING build is bit-identical
    to the reference build (history, BGC, diagnostics, CDR_LITE).
  - Size of the term: with it on, temperature moves by up to 1.3e-2 C at
    the open edges after 10 steps, and the CDR tracers next to the
    boundary by 10-16 %.
  - GPU against CPU with the term: every physics field and MARBL tracer is
    within one float32 unit every 2 steps, edges included (zeta 6e-8,
    temperature 4.8e-7, DIC/ALK 6e-8 relative).
  - _uscl structure: the CPU's 7 records in 2 files, attributes, and
    identical boundary lat/lon.
  - _uscl values: layer thickness 1.5e-14 relative, CDR_LITE fluxes 6e-11.
  - ALK_add and DIC_add are zero in that case (ALK = ALK_ALT_CO2). With
    releases next to each open boundary (tools/make_uscl_test.py,
    gen_cdr.py --rel-file) they are nonzero on all four sides and agree to
    2.6e-9.
- **Regression.** With do_upscale off, the build reproduces cdr6c bit for
  bit (volume releases, gas, MiniPac). The UP3 passive kernel went from 90
  to 95 registers (its launch bound allows 102); the fastest steps without
  output are unchanged.

### CDR_LITE without MARBL (2026-10-19)

The CDR_LITE tracers run on their own (nt_bgc = 0): the passive list is then
the CDR_LITE block alone. They get transport, releases, the <tracer>_flx
surface flux and gas exchange.

- **Inputs.**
  - The carbonate sensitivities come from files (ddic_dco2, ddic_dalk).
    The online computation needs MARBL's ALT_CO2, PO4 and SiO3 tracers.
  - The tracer axes of the river file and of the release file
    (cdr_trcflx) must match the run's tracers: T, S and the CDR_LITE
    block. The river file may also have T and S only.
  - The CPU also needs tnu2 with one value per tracer.
  - No BGC forcing files are needed.
- **Validation (Iceland0, 12 steps; CPU cases/CDR/bld_cpu_nomarbl: the
  reference keys without MARBL, MARBL_DIAGS and the NHY, NOX and PCO2AIR
  forcings).**
  - Every physics field and all six CDR_LITE tracers are within one
    float32 unit every 2 steps.
  - The instantaneous _cdrtrc fields agree to 2e-7 relative (FG_CDR_*
    included).
  - An averaged _cdrtrc with only FG_CDR_* and the DIC vertical integrals
    has the CPU's records and times; FG agrees to 1.8e-7, the integrals to
    3e-12.

**Capacity on Pac, 8 A100-40GB (8x1 tiles, 2 nodes, 4 writer ranks).**
Measured with `gpumem` at 'loop done', which matches the nvidia-smi
maximum (cases/GPU namelist_roms_gpu_pacnm{40,70,70x,78x}.nml).

- **Fit**, within 0.07 GiB over 13 ranks:
  - per GPU: 2.37 GiB + 40.9 kB per tile cell - 5.2 kB per wet cell
    + 2.515 kB per wet cell per tracer;
  - the bare FP64 tp (3 time levels x 100 levels) is 2.40 of the
    2.515 kB;
  - Pac has 1.78M cells and 1.21M wet, so a tracer costs 0.353 GiB per
    GPU on balanced tiles.
- **Stray contexts.** Rank 0 carries 1.33 GiB extra and rank 4 (first on
  node 2) 2.94 GiB, about 0.43 GiB for each other process on that node:
  - MPI_Init with GPU support opens a context on device 0 before
    `select_device`;
  - on node 2 the four writer ranks add theirs to rank 4's GPU.
- **The 'wet' planner goes the wrong way when tracers dominate.**
  - At 70 tracers the land weight 0.40 gave a modelled largest tile of
    39.40 GB (limit 39.01).
  - The loop raised the land weight to 1.0 (equal area), which puts the
    most wet cells on the open-ocean tiles.
  - Ranks 2-4 then needed 43-46 GiB. They sat at 39.49 GiB, oversubscribed
    through managed memory, at 10.4 s per step (3x).
  - Memory balance needs the land weight lowered toward
    mcell/(mcell + mwet) (0.19 here).
  - The default model is also off: 53.2 kB per cell (measured 41),
    2.40 kB per wet cell per tracer (2.515), no stray contexts.
- **Explicit cuts that balance the fitted memory** (stray contexts
  included):
  - 70 tracers, tile_xcut 291 542 727 907 1076 1284 1534: 35.2-35.8 GiB
    per GPU;
  - 78 tracers, tile_xcut 294 544 728 906 1075 1282 1532: 38.0-38.5 GiB,
    with no slowdown;
  - 80 would model at 39.0 GiB, too close to the 39.49 GiB device;
  - without the stray contexts the limit is 80.
  - A restart write adds one double tile field of device staging (about
    0.3 GB).
- **Step time.** 70 tracers run at 0.41 s per step after the 24-step
  probe window (3.7 s per step inside it). The window was then on by
  default; probes are now off unless `ROMS_GPU_PROBES` is non-zero.
- **Output size.** int_z_CDR_OAE_DIC* is double: 35 fields are 0.5 GB per
  record.
- **Against the CPU code.**
  - The CPU's per-tracer cost is its t(GLOBAL_2D_ARRAY,nz,3,nt) array, on
    every cell including land and the 2-cell halo.
  - Its fixed cost is measured as each rank's peak resident memory:
    Iceland0 no-MARBL with 8x8 and 4x4 tiles (cases/CDR
    submit_mem_cpu.sh) gives 70 MB per rank + 35 kB per cell at 60
    levels, scaled to 100 levels.
  - Pac on the 512-rank CPU layout (58x60 tiles, 14% halo):
    - per tracer, 4.9 GB on the CPU and 3.0 GB on the GPU (1.6x);
    - fixed, 155 GB on the CPU and 92 GB on the GPU (1.7x).
  - In the memory of 8 A100s (335 GB): 36 tracers on the CPU, 80 on the
    GPU (2.2x). With 1 TB the ratio falls to 1.75x.

**Float32 storage of the CDR_LITE tracers (Pac, 2026-10-19).** The build
without `-DPASSIVE_FP64` stores every passive tracer in float32, so it
suits only runs without MARBL (see the 2026-10-05 saturation-depth
finding). Test files are in cases/GPU: `tools_fp32/`,
`namelist_roms_gpu_fp32test_*`, `submit_fp32test.sh`, output in
`output_fp32test/`.

- **Runs.** Each covers 2010-02-08 to 2010-12-30 (325 days, 31,200 steps;
  the boundary data end on Dec 31):
  - FP64;
  - float32;
  - an FP64 twin with temperature +1e-6 C at wet points.
- **Releases and forcing.**
  - 12 OAE pairs at 6 sites, 2,000-4,900 km from the open boundaries:
    ALOHA, California shelf, Station Papa, equator 140W, Kuroshio
    Extension, Peru shelf.
  - Pairs 1-6 are 30-day pulses, 7-12 continuous; 1000 mol/s ALK each.
  - Static sensitivities from PyCO2SYS on the restart surface:
    - beta = gamma_DIC/CO2 (gamma_DIC is in mol/kg), median 19;
    - eta = 1/isocapnic quotient, 0.79-0.90.
- **Output.** Daily-mean int_z of DIC and ALK and FG for every pair, and a
  5-day 2-D history.
- **Physics.** The float32 and FP64 runs are bit-identical (zeta, ubar,
  vbar and hbls differences are 0 over all 325 days). The passives do not
  feed back, so their difference is storage rounding alone. The twin
  diverges by up to 1.5 m in zeta and 270 m in hbls.
- **Domain-integrated CDR_OAE_DIC (the uptake).** Float32 against FP64,
  relative: at most 1.9e-6 (Station Papa continuous), 6.4e-7 over all
  pairs.
  - The error is set in the first ~50 days, then stays flat to day 325.
  - The twin's chaotic spread at day 325 is 5e-6 to 2.7e-2 per pair
    (2.8e-3 over all pairs), mostly 1e-4 to 1e-2.
  - Uptake efficiency DIC/ALK at day 325 is 0.20-0.64.
- **Domain-integrated ALK.**
  - Float32 drift at most 2.4e-6, flat after ~60 days.
  - FP64 is exact to 1e-15 until the equator tracer reaches an open
    boundary (twin difference from day ~285).
- **Flux closure (FP64).** Daily-mean DIC against cumulative FG, with the
  half-day offset removed, agrees within 1e-3 after day 60. The few
  percent in the first weeks is daily-mean sampling of a variable flux,
  not a leak.
- **Memory and speed.** Float32 needs 1,286 B per wet cell per tracer
  (FP64: 2,515), 0.18 GiB per tracer per GPU on balanced Pac tiles; the
  128-tracer cap is about 34 GiB per GPU. The step is 2% faster (0.189 vs
  0.193 s median).
- **Output precision.** The CDR_LITE output fields (`_cdrtrc`) are now
  written in the passive storage precision (`cdr_lite_out.F90`,
  `out_float`):
  - double with `-DPASSIVE_FP64`, as the CPU;
  - float in the float32 build: computed and averaged in double, rounded
    once by netCDF at the write. The time variables stay double.
  - Check on Pac, 2 days, 24 tracers (`tools_fp32/check_outprec.py`):
    - the FP64 build's `_cdrtrc` is bit-identical to the earlier binary;
    - the float32 build's fields equal the earlier double values rounded
      to float32. Values below FLT_MIN (1.2e-38, 0.02% of them, plume
      tails) are written as 0. The file is half the size.
- **Wall-clock timers.** The 325-day runs reported a 1,660 s stepping
  loop for ~5,950 s of real time, and one step of -2147 s.
  - Cause: nvfortran's default-integer `system_clock` wraps every 2^31
    microseconds (2,147 s), and when it does it re-bases the integer(8)
    clock as well. The model mixed default-integer calls (step_driver,
    io_server) with integer(8) ones (run_timers), so the integer(8)
    intervals that spanned a wrap came out short.
  - Test over 2,200 s: with integer(8) calls only, the interval was
    correct; with a default-integer call every 20 s, the same integer(8)
    interval read 20.0 s.
  - Fix: `wall_time()` in run_timers (`clock_gettime`, CLOCK_MONOTONIC)
    replaces every `system_clock` call: run_timers, main (wall stamps),
    step_driver, par_read, frc_io, io_server, extract_out.

## Averages, true vertical velocity and child-grid extraction (2026-10-18)

**Averaged history and BGC averages (src/host/avg_out.F90).** These are
the CPU's basic_output (wrt_file_avg) and bgc_io (wrt_bgc_avg), in
double:

- **Running means.** Every step after the step, avg*(1 - 1/n) + x/n.
  - zeta, ubar and vbar are taken at knew (time n+1).
  - u, v and the tracers come from slot nstp (time n, the CPU's choice).
  - rho is rho1 - qp1*z_r; omega is We + Wi, times pm*pn at the write.
  - Also averaged: the true w, Akv, Akt, AKs (the CPU's name), hbls and
    hbbl.
- **Records.** _avg.nc carries a record every output_period_avg, stamped
  with the mean model time. _bgc_avg.nc carries the MARBL tracers every
  output_period_bgc_avg, stamped with the time at the write.
- **Which tracers.** As the CPU's wrt_t_avg: with MARBL, the names in
  marbl_tracers_to_write, or every tracer when the list is empty.
  Temperature, salinity and CDR_LITE go to _avg, MARBL to _bgc_avg. So with
  the usual MARBL list, the CPU's _avg file has no temperature or salinity
  (the likely CPU oversight, kept for now).
- **CPU behaviours kept.** Without BGC history output (wrt_bgc_his off),
  the CPU's start-up call already adds the initial state to the BGC
  average and advances its clock by a step.
- **Not kept.** The CPU resets its _bgc_avg record counter on every call,
  so it writes one file per record. Here nrpf_bgc_avg records go in each
  file.

**True vertical velocity (src/device/wvlcty.F90).** The CPU's wvlcty_tile:
the bottom-up mass-flux divergence times pm*pn, interpolated to rho
levels, plus u, v (slot nstp) along the s-surface slopes, and the gradient
condition at the physical edges. It feeds wrt_W in the history (no longer
stopped) and the w average.

**Start-up fixes, found with these outputs.**

- **Passive tracers at land are zeroed at load (sanitize_land).** Land
  cells of the boundary ring are stored columns, so they kept the initial
  file's values until the first step's boundary conditions. The CPU has 0
  there, and the values showed up in the initial history record and in
  averages that include the first steps.
- **The CPU's start-up diagnostics now run before the initial history
  record:** set_depth, set_HUV, omega and rho_eos. Before, the GPU's
  initial record had rho, omega and w from unset fields. Every step
  recomputes them first, so the run itself is unchanged: bit-identical to
  cdr6g on Iceland0 and MiniPac.

**Child-grid extraction (src/host/extract_out.F90).** The CPU's
extract_data, its PARALLEL_IO path:

- **Objects.** An objects file has one variable per object: the points of
  a child boundary in the parent's absolute index space, plus the child
  angle for u/v points, and the fields in output_vars (matched as
  substrings, as the CPU's findstr). Each tile takes the first contiguous
  run of points inside it.
- **Interpolation.** Bilinear with the mask in the weights, normalized;
  u/v are rotated by the parent angle minus the child angle. When the
  child's vertical grid differs (N_chd, theta_s/b_chd, hc_chd), every
  column is remapped with the CPU's PPM remap from the interpolated parent
  layers (at u/v points, the mean of two rho columns) to the child layers
  of the same depth.
- **Output.** _bry.nc has the CPU's dimensions and variables (per side:
  zeta, ubar, vbar, every tracer, u, v) and bry_time in days. A record
  comes every output_period_extract, nrpf_extract per file.
- **Where it runs.** The interpolation runs on the device, rotation and
  remap on the host, and compute rank 0 writes the sum over ranks.
- **Not supported:** up/vp (they need calc_pflx, not on the GPU) and w
  (the CPU's parallel-I/O path does not write it). Both stop at start-up.

**Validation (Iceland0, 12 steps; gen_cdr.py rel with --avg, --bgc-avg,
--his-all, --all-tracers, --extract; objects from
tools/make_extract_test.py).** The test child box is 2x refined over
parent cells 200-330 and crosses land on two sides.

- **Averages (av1: the MARBL list; av2: an empty list, with temperature,
  salinity and CDR_LITE).** Records, file split and variables are the
  CPU's.
  - Every field agrees to 6e-8 relative.
  - BGC averages agree to 1.3e-7.
  - Akv, Akt and AKs are within 1-3e-5. That is the KPP mixing
    coefficients' own GPU-CPU difference (3.7e-6 absolute after 10 steps).
    The CPU's 1e-6 degC twin differs by 1e-2 after 2 steps.
- **w and the initial record.** The history's w agrees to float32
  precision, and the initial record's w, rho and tracers now match.
- **Extraction (ex1 with a 40-level child grid and remap, ex2 without).**
  Records, files and variables are the CPU's, with no finite/NaN
  mismatch.
  - zeta, ubar and vbar agree to 4e-13.
  - u and v agree to 3e-10.
  - Temperature, salinity and the tracers agree to 7e-10.
- **Regression.** With none of these on, the build reproduces cdr6g bit
  for bit (Iceland0 gas, MiniPac).
- **Cost (Iceland0, 4 A100s, 36-step runs, median step 58.7 ms).**
  - Accumulating both averages adds 3.1 ms per step (5 %), mostly the 32
    MARBL tracers.
    - The BGC kernel takes one thread per (tracer, column, level), tracers
      fastest. A thread per (column, level) looping over the tracers cost
      4.3 ms.
    - The ocean fields and w add the rest.
  - An extraction record adds about 60 ms, nearly all the rank-0 netCDF
    write of ~170 variables. That is 2 ms per step at hourly records.
    - The interpolations are queued on the device into one buffer, copied
      once: 1 ms. Fetching each field through managed memory cost 0.2 s.
    - The remap takes 15 ms on the host and the reduction over the ranks
      10-20 ms.
    - netCDF-4 with a chunk per record beat CDF5 (150 ms) for this file.

## Generic passive tracers with MARBL, and the random output (2026-10-21)

For the CCS dye-tracer run (`ccs_4km_passive`): MARBL plus generic passive
tracers released by CDR Gaussian patches, with their surface fields written
every few hours.

- **Tracer layout.** `tp` holds the MARBL tracers (1 .. `n_marbl`), then the
  `nt_passive` generic tracers, then the CDR_LITE block (`cdr0` =
  `ntp_base` = `n_marbl` + `n_generic`).
  - Before, MARBL mode set the generic count to 0.
  - The MARBL kernels and `marbl_step` address `1 .. n_marbl` (they used
    `ntp_base`); `avg_out` and `extract_out` follow the same layout.
  - `cpu_tracer_index` maps tp to the CPU's order: T, S, generic, CDR_LITE,
    MARBL.
- **Inputs.** The river and CDR release tracer axes are read positionally in
  that CPU order, so a river file needs the generic rows (zero
  concentrations), as on the CPU.
  - roms-tools writes T, S and MARBL only; the run's
    `scripts/make_rivers.py` inserts the rows.
  - The roms-tools `CDRForcing` with `tracer_set="passive"` and
    `include_marbl_bgc=True` already has them.
  - Initial and boundary values come by name (`passive_tracer<n>`), and 0
    when absent.
- **Names.** Generic tracers are now `passive_tracer<n>` in every mode, as on
  the CPU; they were `tpNN` (`tools/mirror_eval.py` updated).
- **Random output** (`src/host/random_out.F90`, `&random_output_settings`:
  `output_period_random`, `nrpf_random`, `do_random`). The CPU module is a
  template that writes Hz, u and v.
  - Here it writes the surface (top-level) field of every generic passive
    tracer every `output_period_random` seconds.
  - Files are `<root>_rnd.nc`, `<root>_rnd_0002.nc`, ..., with `nrpf_random`
    records each.
  - Fields are in the passive storage precision; the time is double.
  - I/O kind 8.
- **Validation (cases/CDR).** Iceland0, MARBL + 4 passive tracers fed by 4
  passive releases, 12 steps (`gen_passive.py`, `tools/make_passive_test.py`).
  - CPU reference: `bld_cpu_passive`, i.e. `bld_cpu_ref` with a local
    `random_output.F90` writing the same surface fields.
  - Random output: within 1.5e-10 relative of the CPU, every record.
  - Physics and MARBL tracers: within one float32 unit every 2 steps.
- **Regressions (bit-identical, `tools/cmp_bitid.py`).**
  - A MARBL run with history, averages, BGC averages and child-grid
    extraction, old binary against new.
  - The no-MARBL CDR_LITE runs nm1 and nm2.

## C-Star compatibility (2026-11-02)

C-Star drives UCLA ROMS through its namelist, its output file names and
the rank layout of the job. These changes make the GPU model look like
UCLA ROMS at those points (C-Star's handoff: claude-docs
`plans/gpu-ucla-roms-pr0-handoff.md`).

- **Unreadable namelist groups stop the run.** A key the GPU does not
  declare makes a namelist read fail part way (nvfortran iostat 239, the
  keys after it keep their defaults). Most optional groups ignored the
  status; now every group stops when present but unreadable and keeps its
  defaults only when absent. Every group the GPU reads declares all of
  UCLA ROMS 0.11.0's keys for it, so 0.11.0 namelists read as before (none
  of 225 GPU and 174 CPU namelists of the validation runs changes). The
  keys added since (0.12.0 `cdr_carbonate_sensitivity_time_interpolation`
  in `&cdr_lite_settings`, 0.13.0 `v_sponge_trc` in `&v_sponge_settings`)
  stop the run instead of silently resetting the rest of their group (with
  `v_sponge_trc` first, the sponge itself). `NAMELIST_COMPAT` lists the
  vocabulary per release.
- **Output names and time metadata as UCLA ROMS.** Files are
  `<root>_<stream>.<YYYYMMDDHHMMSS>.nc` (no `_0002` counter), with the
  date from the CPU's `sec2date` and `&reference_date_settings` (read in
  `his_io`, default 2000/1/1; checked against Python's datetime for
  1970-2099). The date is the CPU's model time when it creates the file:
  every stream writes after the step, at time n+1, except the upscaling
  means (time n, inside the step); the restart is named at its second
  record's time (`EXACT_RESTART`: time + dt). So the history, `cdr`,
  `cdrtrc`, `cdrgas`, `rnd`, `uscl` and `bry` stamps equal the first
  record's time, the average files carry the end of their first window
  (the records are stamped mid-window), and restarts the time of the
  later of their two records, as in the CPU output. `ocean_time` has
  `long_name = refdatestr` in every stream (before: units only in
  history and restart, a fixed "Time since 2000/01/01" elsewhere), and
  the history and restart record dimension is `time` (was `ocean_time`).
  The reference date does not enter the arithmetic: model time comes from
  the initial file, as on the CPU.
- **Rank roles by node (`ROMS_GPU_NWRITERS`).** SLURM gives every node the
  same number of tasks, which the old rule (writers = the world ranks after
  np_xi x np_eta, so all on the last node) cannot use. With
  `ROMS_GPU_NWRITERS=w` the last w ranks of every node are writers
  (`MPI_Comm_split_type` gives the node-local rank); compute ranks keep
  world order, so the tiles are the same. A layout whose compute ranks do
  not number np_xi x np_eta stops at start-up with the layout it expected
  (before: a mis-sized job ran with a different writer count). The io_server
  addresses writers and indexes tile buffers through tables of world ranks
  (`comm_writer_world`, `comm_compute_index`) instead of assuming
  contiguous ranks, and a compute rank takes GPU (node rank mod visible
  GPUs) instead of (compute rank mod visible GPUs), the same on every
  layout used so far. Without the variable the old rule applies unchanged.
  Writer groups are consecutive writers, so with w writers per node and
  group size w each node writes its own files (two files from two nodes
  ran at 1.23 GB/s against 1.00 from one, "Per-run round").
- **Version line and stale objects.** The model prints `roms_gpu version
  <git describe>` at start-up (passed to the main compile command only).
  `build_gpu/flags.txt` records the compiler and flags the cached objects
  (MARBL, `kinds_pre`, `marbl_host`) were built with; a change rebuilds
  them (before, a precision or architecture change left them stale unless
  one ran `make clean`). The model itself is recompiled on every `make gpu`.
- **Checked.** MiniPac (2x2 and 4x2 tiles, 6 h, split history, averages, a
  restart) and CCS 4 km with MARBL, CDR and a 2010/06/15 reference date
  (24 steps, every stream). Output contents are bit-identical to the
  previous binary in both. The per-node layouts (one and two writers per
  node, with and without `tools/numa_bind.sh`) are bit-identical to the
  old layout; on CCS their files are byte-identical. The CCS file names
  and `ocean_time` values equal the CPU's for history, restart, averages,
  `cdrtrc` and `cdrgas` (averages within 6e-8 s, as before these
  changes). A namelist with an undeclared key stops on every rank with
  the group and nvfortran's message, and a job whose ranks do not fit
  the layout stops at start-up. Two stream differences remain: the GPU
  writes the MARBL tracers to `_his` and has no `_bgc` stream, and the
  CPU writes one `_bgc_avg` record per file whatever `nrpf_bgc_avg` says
  (the GPU honours it).

## Toolchain lessons (nvfortran 25.9)

- A by-value derived type as a global-kernel argument does not compile;
  parameters are scalar `value` args or a module-level `device` struct
  (a `managed` struct gave illegal-address errors at run time).
- Device subroutines must take every dummy by reference; mixing `value`
  inputs with by-reference outputs returned undefined inputs, and even
  pure device functions with `value` args (NHx emission) returned NaN in
  the FP64 model build although they passed the float32 column test.
- The inliner miscompiled callers of a device/host *function* with
  `value` args (`potemp_fn`): wrong pH saved state and a zero diagnostic
  with correct tendencies. Rewritten as a subroutine. Use
  `-Minline=except:<name>` to bisect such cases.
- MARBL's own interior module segfaults at -O2 (built at -O0 for the
  host reference).
- Never set `cudaLimitStackSize` large: the reservation is stack size x
  resident threads (64 kB cost 14 GB of device memory and slowed every
  kernel). The runtime sizes the stack from each kernel's requirement.
- The I/O server caps variables per record (`MAXV` in `io_server.F90`,
  now 96) and the staging capacity counts the MARBL 2-D diagnostics.

## Open items

- Restart precision: with `-DPASSIVE_FP64` the tracers are written to the
  restart in double; the history stays float32.
- History diagnostics match the CPU `_bgc_dia` set used for MiniPac
  (surface set + zsatcalc/zsatarag + 3-D CO3, CO3_ALT_CO2, co3_sat_calc,
  co3_sat_arag; added 2026-10-05). Over the three-day run they are at the
  noise floor like the tracers: CO3 ratio 0.10 (h6) to 0.72 (h72),
  co3_sat 0.82 worst, zsat 0.07 worst. The 3-D pH is in the restart only
  (`MARBL_PH_3D`), as in the CPU model. Other MARBL diagnostics
  (production, remineralisation, fluxes per PFT) are not written.
- Performance: K2 register pressure, carbonate solver cost, the 128
  per-step boundary fills for the tracers (could move to the hourly call).
- Carbonate solver failures (no pH bracket, or no convergence) stop the
  run, as MARBL's error does on the CPU (since 2026-10-16; before, the GPU
  counted them and ran on). After every MARBL call a device reduction
  counts sf_err/it_err (no measurable cost on MiniPac); a failing rank
  lists up to 10 columns (global i, j, surface T, S, DIC, ALK, PO4, SiO3,
  the column's lowest salinity) and calls comm_abort (MPI_Abort on the
  world communicator). MiniPac with rivers stops at iic 13760 on the cell
  (366, 99) where the perturbed CPU run stopped.
- Not ported (not used by MiniPac): ciso, abio_dic, Lig_cnt = 2, restoring,
  O2/remin scale-factor forcings, ladjust_bury_coeff, PAR subcolumns > 1.

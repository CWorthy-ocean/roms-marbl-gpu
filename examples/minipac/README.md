# MiniPac example

MiniPac is a 12 km tropical Pacific grid (512 x 432 x 100, dt = 900 s). This
example runs it for 3 days with MARBL (32 biogeochemical tracers) on one
node with 4 GPUs, 2 x 2 tiles. On Derecho's A100s a step takes about 75 ms;
the whole job takes under 3 minutes.

| File | |
|---|---|
| `namelist_minipac_marbl.nml` | the run; paths are relative to this file |
| `marbl_in` | MARBL settings |
| `run_minipac.pbs` | PBS job script; edit the settings block at the top |

## Input data (not included)

Put the MiniPac input files in `input_data/` here, or make `input_data` a
link to wherever they live:

- grid: `PACMED12km_v2ni_grid.nc`
- initial state: `minipac_rst.20100330000000.nc` (a spun-up restart from the
  CPU model), or `PACMED12km_v2ni_initial_conditions.nc` with `inifile`
  changed to match
- surface forcing: `PACMED12km_v2ni_surface-physics_YYYYMM.nc` (200912 to
  201007), `PACMED12km_v2ni_surface-restoring_clim.nc`,
  `PACMED12km_v2ni_surface-bgc-unified_clim.nc`,
  `PACMED12km_v2ni_surface-bgc-mbl_co2_2009.nc`, `..._2010.nc`
- boundary forcing: `PACMED12km_v2ni_boundary-physics_YYYYMM.nc` (200912 to
  201007), `PACMED12km_v2ni_boundary-bgc-unified_clim.nc`
- tides: `PACMED12km_v2ni_tidal.nc`

The namelist uses the same groups and keys as UCLA ROMS, plus `np_xi` and
`np_eta` (the GPU tiles), `nt_bgc` (MARBL tracers) and `nt_passive`.

## Run

```
tools/submit_gpu.sh examples/minipac/run_minipac.pbs
```

Output goes to `output/`: `minipac_marbl_his.20100330000000.nc` (6-hourly
history, single precision), `minipac_marbl_bgc.20100330000000.nc` and
`minipac_marbl_bgc_dia.20100330000000.nc` (6-hourly MARBL tracers and
diagnostics, double precision) and `minipac_marbl_rst.20100402000000.nc`
(double precision, two records at day 3 for an exact restart). As in UCLA
ROMS, each file is named by the model date at which it is started, counted
from `reference_date`. To restart, set `inifile` to the restart file.

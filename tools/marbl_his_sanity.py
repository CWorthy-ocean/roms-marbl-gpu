"""Sanity check of a MARBL output file (_bgc or _bgc_dia): ranges, NaNs and the change over
the run for every MARBL tracer and surface diagnostic.

usage: python marbl_his_sanity.py <bgc.nc> [surface-level-only]
"""
import os
import sys
import netCDF4 as nc
import numpy as np

f = nc.Dataset(sys.argv[1])
surf_only = len(sys.argv) > 2
# ---- settings: edit here, or set the environment variable ----
GRID = os.environ.get("ROMS_GPU_GRID", "/path/to/grid.nc")   # the run's grid file (mask_rho)
grd = nc.Dataset(GRID)
wet = grd["mask_rho"][:] > 0.5
tdim = 'ocean_time' if 'ocean_time' in f.dimensions else 'time'
nrec = len(f.dimensions[tdim])
print("records:", nrec)
names = ['PO4', 'NO3', 'SiO3', 'NH4', 'Fe', 'Lig', 'O2', 'DIC', 'DIC_ALT_CO2', 'ALK', 'ALK_ALT_CO2',
         'DOC', 'DON', 'DOP', 'DOPr', 'DONr', 'DOCr', 'zooC', 'spChl', 'spC', 'spP', 'spFe', 'spCaCO3',
         'diatChl', 'diatC', 'diatP', 'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazP', 'diazFe',
         'PH', 'pCO2SURF', 'DpCO2', 'FG_CO2', 'FG_ALT_CO2', 'PH_ALT_CO2', 'pCO2SURF_ALT_CO2']
print(f"{'var':16s} {'min0':>11s} {'max0':>11s} {'minN':>11s} {'maxN':>11s} {'rms(N-0)':>11s} {'nan0':>6s} {'nanN':>6s}")
for v in names:
    if v not in f.variables:
        continue
    var = f.variables[v]
    def rec(n):
        a = var[n]
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 3:
            a = a[-1] if surf_only else a
            w = wet if a.ndim == 2 else np.broadcast_to(wet, a.shape)
        else:
            w = wet
        return a, w
    a0, w = rec(0)
    aN, _ = rec(nrec - 1)
    m0 = a0[w]; mN = aN[w]
    d = (aN - a0)[w]
    fin = np.isfinite(d)
    print(f"{v:16s} {np.nanmin(m0):11.4e} {np.nanmax(m0):11.4e} {np.nanmin(mN):11.4e} {np.nanmax(mN):11.4e} "
          f"{np.sqrt(np.mean(d[fin]**2)):11.4e} {int(np.sum(~np.isfinite(m0))):6d} {int(np.sum(~np.isfinite(mN))):6d}")

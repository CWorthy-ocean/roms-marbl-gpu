"""Noise-floor test of the GPU MARBL port against CPU ROMS-MARBL.

For selected hours and every MARBL tracer (plus the surface diagnostics
PH, pCO2SURF, FG_CO2 where present), RMS differences over wet points of
  gpu  : GPU MiniPac day run with MARBL           vs CPU reference
  pert : CPU twin with initial T perturbed (1e-6) vs CPU reference
and the ratio gpu/pert. Ratios near or below 1 mean the GPU-CPU
difference is within the CPU model's own sensitivity.

Files (set below): GPU history (float32, tracers by MARBL name), CPU bgc
history (tracers) and bgc diagnostics history. Restart files at day end
(double) are compared when present.

usage: python marbl_noise_floor_eval.py [--tag TAG] [--every H] [hours...]
  --tag   run name suffix: files minipac_gpu_marbl<TAG>_his.nc and
          minipac_cpu_marbl<TAG>{,_pert}_*  (default '' = the 1-day runs;
          '3d' = the 3-day runs)
  --every hours between history records (default 1; 6 for the 3-day runs).
          Requested hours must be multiples of it.
  --gpu-tag  GPU run name suffix when it differs from --tag (e.g. a rerun
          written as minipac_gpu_marbl3dv2 compared with the 3d CPU runs).
"""
import glob
import os
import sys

import netCDF4 as nc
import numpy as np

# ---- settings: edit here, or set the environment variable ----
CASES = os.environ.get("ROMS_GPU_CASES", "/path/to/cases")             # root of the case directories
GRID = os.environ.get("ROMS_GPU_GRID", os.path.join(CASES, "MiniPac/input_data/PACMED12km_v2ni_grid.nc"))
GPU_OUT = os.path.join(CASES, "GPU/output/minipac_gpu_marbl")          # GPU file prefix (before <tag>)
CPU_OUT = os.path.join(CASES, "CPU_marbl/output/minipac_cpu_marbl")    # CPU file prefix (before <tag>)
grd = nc.Dataset(GRID)
mr = grd["mask_rho"][:] > 0.5
S = (slice(1, -1), slice(1, -1))
wet = mr[S]

TRACERS = ['PO4', 'NO3', 'SiO3', 'NH4', 'Fe', 'Lig', 'O2', 'DIC', 'DIC_ALT_CO2', 'ALK', 'ALK_ALT_CO2',
           'DOC', 'DON', 'DOP', 'DOPr', 'DONr', 'DOCr', 'zooC', 'spChl', 'spC', 'spP', 'spFe', 'spCaCO3',
           'diatChl', 'diatC', 'diatP', 'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazP', 'diazFe']
DIAGS = ['PH', 'pCO2SURF', 'FG_CO2', 'zsatcalc', 'zsatarag', 'CO3', 'CO3_ALT_CO2', 'co3_sat_calc', 'co3_sat_arag']


args = sys.argv[1:]
TAG = ''
GTAG = None
EVERY = 1
while args and args[0].startswith('--'):
    k = args.pop(0)
    if k == '--tag':
        TAG = args.pop(0)
    elif k == '--gpu-tag':
        GTAG = args.pop(0)
    elif k == '--every':
        EVERY = int(args.pop(0))
    else:
        sys.exit('unknown option ' + k)


class Source:
    """Records spread over one or more files; record n = hour n*EVERY."""

    def __init__(self, patterns):
        self.files = []
        for p in patterns:
            self.files += sorted(glob.glob(p))
        self.ds = [nc.Dataset(f) for f in self.files]
        self.index = {}   # var -> list of (dataset, local record)
        for d in self.ds:
            for v in d.variables:
                if 'time' in d.variables[v].dimensions or d.variables[v].dimensions[0] in ('time', 'ocean_time'):
                    nrec = d.variables[v].shape[0]
                    self.index.setdefault(v, [])
                    for r in range(nrec):
                        self.index[v].append((d, r))

    def has(self, v):
        return v in self.index

    def get(self, v, n):
        d, r = self.index[v][n]
        a = d.variables[v][r]
        return np.asarray(a, dtype=np.float64)

    def nrec(self, v):
        return len(self.index.get(v, []))


G = GPU_OUT + (TAG if GTAG is None else GTAG)
C = CPU_OUT + TAG
gpu = Source([G + "_bgc.*.nc", G + "_bgc_dia.*.nc", G + "_his.*.nc", G + "_his.nc", G + "_his_[0-9][0-9][0-9][0-9].nc"])
ref = Source([C + "_bgc.*.nc", C + "_bgc_dia.*.nc", C + "_his.*.nc"])
pert = Source([C + "_pert_bgc.*.nc", C + "_pert_bgc_dia.*.nc", C + "_pert_his.*.nc"])
print("gpu files:", [os.path.basename(f) for f in gpu.files])
print("ref files:", [os.path.basename(f) for f in ref.files])
print("pert files:", [os.path.basename(f) for f in pert.files])

hours = [int(h) for h in args] or ([1, 6, 12, 24] if EVERY == 1 else [6, 24, 48, 72])
for h in hours:
    if h % EVERY:
        sys.exit(f"hour {h} is not a multiple of the record interval {EVERY}")


def rms(d, w):
    d = d[..., w] if d.ndim == 3 else d[w]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else float('nan')


def amax(d, w):
    d = d[..., w] if d.ndim == 3 else d[w]
    d = d[np.isfinite(d)]
    return float(np.max(np.abs(d))) if d.size else float('nan')


def interior(a):
    return a[..., 1:-1, 1:-1]


worst = {}
for h in hours:
    n = h // EVERY
    print(f"\n=== hour {h} (record {n})")
    print(f"{'var':14s} {'scale':>11s} {'rms gpu-ref':>12s} {'rms pert-ref':>13s} {'ratio':>8s} {'max gpu-ref':>12s} {'max pert-ref':>13s}")
    for v in TRACERS + DIAGS:
        if not (gpu.has(v) and ref.has(v)):
            continue
        if gpu.nrec(v) <= n or ref.nrec(v) <= n:
            continue
        g = interior(gpu.get(v, n))
        r = interior(ref.get(v, n))
        if g.shape != r.shape:
            print(f"{v:14s} shape mismatch gpu {g.shape} ref {r.shape}")
            continue
        scale = rms(r, wet)
        dg = rms(g - r, wet)
        mg = amax(g - r, wet)
        if pert.has(v) and pert.nrec(v) > n:
            p = interior(pert.get(v, n))
            dp = rms(p - r, wet)
            mp = amax(p - r, wet)
            ratio = dg / dp if dp > 0 else float('inf')
        else:
            dp = float('nan'); mp = float('nan'); ratio = float('nan')
        worst[v] = max(worst.get(v, 0.0), ratio if np.isfinite(ratio) else 0.0)
        print(f"{v:14s} {scale:11.4e} {dg:12.4e} {dp:13.4e} {ratio:8.3f} {mg:12.4e} {mp:13.4e}")

print("\n=== worst ratio over the evaluated hours")
for v in TRACERS + DIAGS:
    if v in worst:
        print(f"{v:14s} {worst[v]:8.3f}")

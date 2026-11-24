"""GPU-vs-CPU comparison of one history day, per variable, on any domain.

For the physics fields (zeta, temp, salt, u, v) and every MARBL tracer and
surface diagnostic present in both runs: RMS of (gpu - cpu) over wet
points, the CPU field's RMS (scale), the ratio, and the max |diff|.
The CPU records may be spread over several files (his, bgc, bgc_dia);
GPU history holds everything in one file.

usage: python gpu_cpu_day_cmp.py --grid G --gpu GPU_HIS --cpu 'PAT1' ['PAT2' ...]
                                 --every HOURS day [day ...]
  --every  hours between records in the files (24 for daily output)
  day      model day since the start of the run (record = day*24/every)
  --level  compare only that level (1-based, nz = surface); default: all
"""
import glob
import sys

import netCDF4 as nc
import numpy as np

args = sys.argv[1:]
opts = {'--cpu': []}
days = []
while args:
    k = args.pop(0)
    if k == '--cpu':
        while args and not args[0].startswith('--') and not args[0].isdigit():
            opts['--cpu'].append(args.pop(0))
    elif k.startswith('--'):
        opts[k] = args.pop(0)
    else:
        days.append(int(k))
grd = nc.Dataset(opts['--grid'])
wet_full = grd['mask_rho'][:] > 0.5
EVERY = int(opts.get('--every', '24'))
LEVEL = int(opts['--level']) if '--level' in opts else None

PHYS = ['zeta', 'temp', 'salt', 'u', 'v']
TRACERS = ['PO4', 'NO3', 'SiO3', 'NH4', 'Fe', 'Lig', 'O2', 'DIC', 'DIC_ALT_CO2', 'ALK', 'ALK_ALT_CO2',
           'DOC', 'DON', 'DOP', 'DOPr', 'DONr', 'DOCr', 'zooC', 'spChl', 'spC', 'spP', 'spFe', 'spCaCO3',
           'diatChl', 'diatC', 'diatP', 'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazP', 'diazFe']
DIAGS = ['PH', 'pCO2SURF', 'FG_CO2', 'zsatcalc', 'zsatarag']


class Source:
    def __init__(self, patterns):
        self.files = []
        for p in patterns:
            self.files += sorted(glob.glob(p))
        if not self.files:
            sys.exit('no files for ' + str(patterns))
        self.ds = [nc.Dataset(f) for f in self.files]
        self.index = {}
        for d in self.ds:
            tname = 'ocean_time' if 'ocean_time' in d.variables else 'time'
            for v in d.variables:
                dims = d.variables[v].dimensions
                if len(dims) >= 3 and dims[0] in ('time', 'ocean_time'):
                    self.index.setdefault(v, [])
                    for r in range(d.variables[v].shape[0]):
                        self.index[v].append((d, r, float(d.variables[tname][r])))

    def has(self, v):
        return v in self.index

    def get(self, v, n):
        d, r, t = self.index[v][n]
        return np.array(d.variables[v][r], dtype=np.float64), t


# the MARBL fields are in the _bgc and _bgc_dia files next to _his
gpu = Source([opts['--gpu']] + [opts['--gpu'].replace('_his', s) for s in ('_bgc', '_bgc_dia') if '_his' in opts['--gpu']])
cpu = Source(opts['--cpu'])


def wet_for(a):
    # mask matching the field's horizontal shape (rho / u / v staggers)
    ny, nx = a.shape[-2:]
    m = wet_full
    if nx == wet_full.shape[1] - 1:
        m = m[:, 1:] & m[:, :-1]
    if ny == wet_full.shape[0] - 1:
        m = m[1:, :] & m[:-1, :]
    if (ny, nx) == (wet_full.shape[0] - 2, wet_full.shape[1] - 2):
        m = m[1:-1, 1:-1]
    return m[:ny, :nx]


def stats(g, c):
    if g.shape != c.shape:
        # the GPU writes the full rho extent; the CPU may omit the boundary rows
        if g.ndim == c.ndim and all(gs >= cs for gs, cs in zip(g.shape, c.shape)):
            off = [(gs - cs) // 2 for gs, cs in zip(g.shape, c.shape)]
            sl = tuple(slice(o, o + cs) for o, cs in zip(off, c.shape))
            g = g[sl]
        else:
            return None
    m = wet_for(c)
    if g.ndim == 3:
        if LEVEL is not None:
            g = g[LEVEL - 1:LEVEL]; c = c[LEVEL - 1:LEVEL]
        m = np.broadcast_to(m, g.shape)
    d = (g - c)[m]
    cc = c[m]
    ok = np.isfinite(d) & np.isfinite(cc)
    d = d[ok]; cc = cc[ok]
    if d.size == 0:
        return None
    rms_d = np.sqrt(np.mean(d * d))
    rms_c = np.sqrt(np.mean(cc * cc))
    return rms_d, rms_c, np.abs(d).max(), d.size


for day in days:
    n = day * 24 // EVERY
    print('== day %d (record %d)' % (day, n))
    print('  %-12s %12s %12s %10s %12s' % ('field', 'rms(g-c)', 'rms(c)', 'ratio', 'max|g-c|'))
    for v in PHYS + TRACERS + DIAGS:
        if not (gpu.has(v) and cpu.has(v)):
            continue
        if n >= len(gpu.index[v]) or n >= len(cpu.index[v]):
            print('  %-12s record %d missing (gpu %d, cpu %d records)' % (v, n, len(gpu.index[v]), len(cpu.index[v])))
            continue
        g, tg = gpu.get(v, n)
        c, tc = cpu.get(v, n)
        if abs(tg - tc) > 1.0:
            print('  %-12s time mismatch gpu %.1f cpu %.1f' % (v, tg, tc))
            continue
        s = stats(g, c)
        if s is None:
            print('  %-12s shapes differ: gpu %s cpu %s' % (v, g.shape, c.shape))
            continue
        rms_d, rms_c, mx, cnt = s
        ratio = rms_d / rms_c if rms_c > 0 else float('nan')
        print('  %-12s %12.4e %12.4e %10.2e %12.4e' % (v, rms_d, rms_c, ratio, mx))

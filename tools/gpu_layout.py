"""Rank layout for a roms_gpu case: compute ranks from np_xi x np_eta, writer
ranks from the history record size, and the PBS select line to launch it.

usage: python gpu_layout.py <namelist> [--gpus-per-node 4] [--target-seconds 15]
                                       [--rate-gbps 0.6] [--max-writers 16] [--groups G]
                                       [--per-node] [--reader]
Prints:  NTILE NWRITERS NRANKS SELECT   (one line, space separated)
         NTILE NWRITERS NRANKS SELECT NWPN   with --per-node
NRANKS counts every rank, forcing readers included.

Rule:
- Writers per group: one writer streams about 0.6 GB/s (measured, Derecho
  Lustre). Take the smallest number that writes one history record within
  --target-seconds, at most 4 (one file's bandwidth is limited, and more
  writers on one file were slower).
- Groups: 2 when the run writes more than one file at a time (history and
  restart, or history split over files by nrpf_his), else 1; --groups
  overrides.
- The model's io_server writes different files with different groups at
  the same time. Its default group size, min(4, nwriters/2), matches this
  layout.

Layouts:
- default: the writers are the extra ranks on the last node; the model
  derives their count from the namelist (every rank after np_xi x np_eta).
- --per-node: the same number of ranks on every node, NWPN writers per
  node (the total rounded up to a multiple of the node count), for the
  model's ROMS_GPU_NWRITERS=NWPN mode. This is the layout a SLURM job
  (--ntasks-per-node) can express, and it spreads the writer groups over
  the nodes. np_xi x np_eta must divide evenly over the nodes.
- --reader (with either layout): one more rank on every node, the forcing
  reader that reads the forcing records ahead into the node's page cache
  (the model's ROMS_GPU_FRC_READER=1 mode; the reader is the last rank of
  each node).
"""
import math
import re
import sys

args = sys.argv[1:]
opts = {'--gpus-per-node': 4, '--target-seconds': 15.0, '--rate-gbps': 0.6, '--max-writers': 16, '--groups': 0}
path = None
per_node = False
reader = 0
while args:
    a = args.pop(0)
    if a == '--per-node':
        per_node = True
    elif a == '--reader':
        reader = 1
    elif a in opts:
        opts[a] = type(opts[a])(args.pop(0))
    else:
        path = a
if path is None:
    sys.exit(__doc__)
txt = open(path).read()
txt = re.sub(r'!.*', '', txt)


def val(key, default=None):
    m = re.search(r'(?<![\w])%s\s*=\s*([^,\n/]+)' % re.escape(key), txt)
    if not m:
        return default
    v = m.group(1).strip().strip("'\"")
    if v.lower() in ('.true.', 't', 'true'):
        return True
    if v.lower() in ('.false.', 'f', 'false'):
        return False
    try:
        return float(v) if '.' in v or 'e' in v.lower() else int(v)
    except ValueError:
        return v


npx = int(val('np_xi', 1)); npy = int(val('np_eta', 1))
llm = int(val('llm', 0)); mmm = int(val('mmm', 0)); nz = int(val('nz', 0))
nt_bgc = int(val('nt_bgc', 0))
cells = (llm + 2) * (mmm + 2)
n2d = sum(1 for k in ('wrt_z', 'wrt_ub', 'wrt_vb', 'wrt_hbls', 'wrt_hbbl') if val(k, False))
n3d = sum(1 for k in ('wrt_u', 'wrt_v', 'wrt_temp', 'wrt_salt', 'wrt_akv', 'wrt_akt', 'wrt_r') if val(k, False))
if val('write_passive', True) and nt_bgc > 0:
    n3d += nt_bgc
if nt_bgc > 0:
    n2d += 8      # MARBL surface diagnostics (PH, pCO2, FG_CO2, zsat, ...)
    n3d += 4      # MARBL 3-D carbonate diagnostics
record_gb = 4.0 * cells * (n3d * nz + n2d) / 1e9
his = val('wrt_file_his', True)
rst = val('wrt_file_rst', True)
per_group = 1
if his:
    per_group = max(1, min(4, math.ceil(record_gb / (opts['--rate-gbps'] * opts['--target-seconds']))))
# files in flight at once: history + restart, or several history files
dt = float(val('dt', 1.0)); ntimes = int(val('ntimes', 1))
period = float(val('output_period_his', 0.0) or 0.0)
nrec_his = int(ntimes * dt / period) + 1 if (his and period > 0) else (1 if his else 0)
split_his = his and int(val('nrpf_his', 20)) < nrec_his
groups = opts['--groups'] if opts['--groups'] > 0 else (2 if (his and (rst or split_his)) else 1)
nw = max(1, min(opts['--max-writers'], groups * per_group))
ntile = npx * npy
g = opts['--gpus-per-node']
nodes = max(1, math.ceil(ntile / g))
if per_node:
    if ntile % nodes:
        sys.exit('gpu_layout.py: %d tiles do not divide over %d nodes of %d GPUs; choose np_xi x np_eta '
                 'as a multiple of the node count' % (ntile, nodes, g))
    wpn = math.ceil(nw / nodes)
    nw = wpn * nodes
    select = '%d:ncpus=64:mpiprocs=%d:ngpus=%d' % (nodes, ntile // nodes + wpn + reader, g)
    print('%d %d %d %s %d' % (ntile, nw, ntile + nw + reader * nodes, select, wpn))
    print('# per node: %d compute + %d writer rank(s)%s; run with ROMS_GPU_NWRITERS=%d%s. SLURM: --nodes=%d '
          '--ntasks-per-node=%d --gpus-per-node=%d' % (ntile // nodes, wpn, ' + 1 forcing reader' if reader else '',
                                                      wpn, ' ROMS_GPU_FRC_READER=1' if reader else '', nodes,
                                                      ntile // nodes + wpn + reader, g), file=sys.stderr)
else:
    if nodes == 1:
        select = '1:ncpus=64:mpiprocs=%d:ngpus=%d' % (ntile + nw + reader, g)
    else:
        last = ntile - g * (nodes - 1)
        select = '%d:ncpus=64:mpiprocs=%d:ngpus=%d+1:ncpus=64:mpiprocs=%d:ngpus=%d' % (nodes - 1, g + reader, g,
                                                                                     last + nw + reader, g)
    print('%d %d %d %s' % (ntile, nw, ntile + nw + reader * nodes, select))
    if reader:
        print('# + 1 forcing reader rank on each of %d node(s); run with ROMS_GPU_FRC_READER=1' % nodes,
              file=sys.stderr)
print('# history record ~%.1f GB -> %d writer(s) per group for <= %.0f s per record at %.1f GB/s each; '
      '%d group(s) -> %d writer(s)' % (record_gb, per_group, opts['--target-seconds'], opts['--rate-gbps'],
                                      groups, nw), file=sys.stderr)

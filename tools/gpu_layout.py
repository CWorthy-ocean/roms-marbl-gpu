"""Rank layout for a roms_gpu case: compute ranks from np_xi x np_eta, writer
ranks from the history record size, and the PBS select line to launch it.

usage: python gpu_layout.py <namelist> [--gpus-per-node 4] [--target-seconds 15]
                                       [--rate-gbps R] [--max-writers 64]
                                       [--per-node | --last-node] [--reader] [--wide-gb 4]
       python gpu_layout.py <namelist> --stripe [--wide-gb 4]
Prints:  NTILE NWRITERS NRANKS SELECT NWPN   per-node layout (NWPN = writers per node)
         NTILE NWRITERS NRANKS SELECT        last-node layout
NRANKS counts every rank, forcing readers included.
--stripe prints the output directory (from output_root_name, relative to the
current directory) and, when the largest output record (history or restart)
is at least --wide-gb GiB, the Lustre layout for it: `<dir> -c 48 -S 16M`
(tools/submit_gpu.sh passes it to `lfs setstripe`); smaller runs keep the
directory's default layout (only `<dir>` is printed).

Rule (CDF5 output):
- The writers write each shared file with O_DIRECT, one group of all
  writers. A writer streams about 1.3 GB/s (--rate-gbps default); one
  writer per node when one writes a history record within --target-seconds,
  else 8 per node (Pacific basin, 2 A100 nodes, 31-GB history and 63-GB
  restart records on -c 48 -S 16M: 16 writers wrote 18-22 GB/s in the
  benchmark, GPU_IO_STEP4_PLAN.md); last-node layout: up to 8 on that node.
- Large records need a wide stripe (--stripe above); on Derecho's default
  progressive layout the first 16 GiB of a file sit on 4 OSTs and a shared
  file takes 6-8 GB/s however many writers write it.
- The model's io_server writes every file with one group of all writers
  (ROMS_GPU_IO_GROUP splits them into groups that write different files at
  the same time).

Layouts:
- per-node (the default on more than one node, when np_xi x np_eta divides
  evenly over the nodes; --per-node forces it on one node too): the same
  number of ranks on every node, NWPN writers per node, for the model's
  ROMS_GPU_NWRITERS=NWPN mode. The model numbers the writers round-robin
  over the nodes, so every file's writer group spans all nodes and each
  file is written through every node's Lustre client. This is also the
  layout a SLURM job (--ntasks-per-node) can express.
- last-node (the default on one node, or when the tiles do not divide over
  the nodes; --last-node forces it): the writers are the extra ranks on the
  last node; the model derives their count from the namelist (every rank
  after np_xi x np_eta). On one node both layouts place the same ranks.
- --reader (with either layout): one more rank on every node, the forcing
  reader that reads the forcing records ahead into the node's page cache
  (the model's ROMS_GPU_FRC_READER=1 mode; the reader is the last rank of
  each node).
"""
import math
import re
import sys

args = sys.argv[1:]
opts = {'--gpus-per-node': 4, '--target-seconds': 15.0, '--rate-gbps': 1.3, '--max-writers': 64, '--wide-gb': 4.0}
path = None
layout = None          # None: automatic; 'per-node' or 'last-node'
reader = 0
stripe = False
while args:
    a = args.pop(0)
    if a == '--per-node':
        layout = 'per-node'
    elif a == '--last-node':
        layout = 'last-node'
    elif a == '--reader':
        reader = 1
    elif a == '--stripe':
        stripe = True
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
# a restart record (double): u, v, T, S and the BGC tracers at least
rst_gb = 8.0 * cells * ((4 + nt_bgc) * nz + 10) / 1e9 if rst else 0.0
if stripe:
    import os
    m = re.search(r'(?<![\w])output_root_name\s*=\s*[\'"]([^\'"]*)[\'"]', txt)
    root = m.group(1).strip() if m else 'output'
    outdir = os.path.dirname(os.path.abspath(root))
    big = max(record_gb if his else 0.0, rst_gb)
    print(outdir + (' -c 48 -S 16M' if big * 1e9 >= opts['--wide-gb'] * 2**30 else ''))
    print('# largest output record ~%.1f GB; wide Lustre stripe from %.1f GiB' % (big, opts['--wide-gb']),
          file=sys.stderr)
    sys.exit(0)
ntile = npx * npy
g = opts['--gpus-per-node']
nodes = max(1, math.ceil(ntile / g))
if layout is None:
    layout = 'per-node' if (nodes > 1 and ntile % nodes == 0) else 'last-node'
    if nodes > 1 and layout == 'last-node':
        print('# %d tiles do not divide over %d nodes: last-node layout (every writer on the last node); choose '
              'np_xi x np_eta as a multiple of the node count to spread the writers' % (ntile, nodes), file=sys.stderr)
elif layout == 'per-node' and ntile % nodes:
    sys.exit('gpu_layout.py: %d tiles do not divide over %d nodes of %d GPUs; choose np_xi x np_eta '
             'as a multiple of the node count' % (ntile, nodes, g))
wnodes = nodes if layout == 'per-node' else 1    # nodes that hold writers
per_group = 1
if his:
    need = math.ceil(record_gb / (opts['--rate-gbps'] * opts['--target-seconds']))   # writers at --rate-gbps each
    # one group of all writers, O_DIRECT
    per_group = max(1, min(8, need)) if wnodes == 1 else wnodes * (1 if need <= 1 else 8)
groups = 1                       # the model's default: one group of all writers
nw = max(1, min(opts['--max-writers'], groups * per_group))
if layout == 'per-node':
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
print('# history record ~%.1f GB -> %d writer(s) per file (%d per node on %d node(s)) for <= %.0f s per record; '
      '%d group(s) -> %d writer(s)' % (record_gb, per_group, per_group // wnodes, wnodes, opts['--target-seconds'],
                                      groups, nw), file=sys.stderr)

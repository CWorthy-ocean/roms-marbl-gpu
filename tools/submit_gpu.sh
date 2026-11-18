#!/bin/bash
# Submit a roms_gpu case with the rank layout derived from its namelist:
# compute ranks = np_xi x np_eta, writer ranks from the history record size
# (tools/gpu_layout.py), PBS select line and mpiexec count set accordingly.
#   tools/submit_gpu.sh <submit_script> [NWRITERS=n] [PERNODE=0|1] [READER=0] [STRIPE=0] [extra qsub -v settings, e.g. EXE=...]
# Layout (gpu_layout.py): on more than one node, by default the same ranks on
# every node with writers on each node (the model's ROMS_GPU_NWRITERS mode),
# so every output file is written through all the nodes; on one node, or
# when np_xi x np_eta does not divide over the nodes, the writers are the
# extra ranks on the last node. PERNODE=1 forces the per-node layout
# (gpu_layout.py --per-node), PERNODE=0 the last-node layout (--last-node).
# With the per-node layout NWRITERS is the total, rounded up to a multiple
# of the node count.
# READER=1 (the default): one more rank on every node, the forcing reader
# (gpu_layout.py --reader; the model's ROMS_GPU_FRC_READER=1 mode), which
# reads the forcing records ahead into the node's page cache; same results,
# no cost when the files are already cached. READER=0: no reader.
# The submit script must name its namelist on an NML= line (it may use
# ${CASE} from a CASE= line) and launch with "mpiexec -n ${NRANKS:-...}".
# Output layout: when the largest output record is 4 GiB or more (gpu_layout.py
# --stripe; ROMS_GPU_IO_WIDE_GB changes the threshold), the output directory
# (from output_root_name, made if missing) gets the Lustre layout
# `lfs setstripe -c 48 -S 16M`, which new files in it inherit: the writers
# write each shared CDF5 file with O_DIRECT, and on the default progressive
# layout a large file's first 16 GiB sit on 4 OSTs (6-8 GB/s however many
# writers; 18-22 GB/s on 48). Smaller runs keep the default layout. STRIPE=0
# leaves the directory as it is; the model warns about a narrow layout.
# PBS (qsub) version; for Slurm, use the printed counts with sbatch/srun.
# ROMS_GPU_PROBES=0 is passed for older binaries, where diagnostic probes were
# on by default; current binaries have them off unless it is non-zero.
set -euo pipefail
# ---- settings: edit here, or set the environment variable ----
ROOT=${ROMS_GPU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # roms_gpu checkout
PYTHON=${PYTHON:-python3}                                                 # gpu_layout.py needs only the standard library
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SCRIPT=$1; shift
# value of a VAR= line in the job script, without a trailing comment or quotes
value() { grep -E "^$1=" "$SCRIPT" | head -1 | sed -E "s/^$1=//; s/[[:space:]]+#.*\$//; s/^\"//; s/\"\$//" || true; }
CASE=$(value CASE)
NML=$(value NML)
NML=${NML//'${CASE}'/$CASE}
[ -f "$NML" ] || { echo "submit_gpu.sh: namelist not found: '$NML' (from the NML= line of $SCRIPT)" >&2; exit 1; }
EXTRA=""
NWOPT=""
PNOPT=""
READER=1
STRIPE=1
for a in "$@"; do
  case $a in
    NWRITERS=*) NWOPT="--max-writers ${a#NWRITERS=} --target-seconds 0.001" ;;
    PERNODE=1) PNOPT="--per-node" ;;
    PERNODE=0) PNOPT="--last-node" ;;
    READER=[01]) READER=${a#READER=} ;;
    STRIPE=[01]) STRIPE=${a#STRIPE=} ;;
    *) EXTRA="${EXTRA},${a}" ;;
  esac
done
if [ "$STRIPE" = 1 ]; then
  read OUTDIR LFSOPT <<< "$($PYTHON "$ROOT/tools/gpu_layout.py" "$NML" --stripe --wide-gb "${ROMS_GPU_IO_WIDE_GB:-4}")"
  if [ -n "$LFSOPT" ]; then
    mkdir -p "$OUTDIR"
    if command -v lfs > /dev/null && lfs setstripe $LFSOPT "$OUTDIR"; then
      echo "output directory $OUTDIR: lfs setstripe $LFSOPT (large records)"
    else
      echo "submit_gpu.sh: could not set the Lustre layout of $OUTDIR (lfs setstripe $LFSOPT); the model will warn" >&2
    fi
  fi
fi
RDOPT=""
if [ "$READER" = 1 ]; then RDOPT="--reader"; EXTRA="${EXTRA},ROMS_GPU_FRC_READER=1"; fi
LAYOUT=$($PYTHON "$ROOT/tools/gpu_layout.py" "$NML" --gpus-per-node $GPUS_PER_NODE $NWOPT $PNOPT $RDOPT)
read NTILE NW NRANKS SELECT WPN <<< "$LAYOUT"
[ -n "$SELECT" ] || { echo "submit_gpu.sh: gpu_layout.py gave no layout for $NML" >&2; exit 1; }
echo "layout: $NTILE compute ranks + $NW writer(s)${RDOPT:+ + forcing readers} = $NRANKS ranks; select=$SELECT"
if [ -n "$WPN" ]; then
  EXTRA=",ROMS_GPU_NWRITERS=${WPN}${EXTRA}"
else
  EXTRA=",NUMA_BIND_NRANKS=${NTILE}${EXTRA}"
fi
qsub -l select=$SELECT -v "ROMS_GPU_PROBES=0,NRANKS=${NRANKS}${EXTRA}" "$SCRIPT"

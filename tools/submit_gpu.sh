#!/bin/bash
# Submit a roms_gpu case with the rank layout derived from its namelist:
# compute ranks = np_xi x np_eta, writer ranks from the history record size
# (tools/gpu_layout.py), PBS select line and mpiexec count set accordingly.
#   tools/submit_gpu.sh <submit_script> [NWRITERS=n] [PERNODE=1] [extra qsub -v settings, e.g. EXE=...]
# PERNODE=1: the same ranks on every node with writers on each node
# (gpu_layout.py --per-node; the model's ROMS_GPU_NWRITERS mode). NWRITERS
# is then the total, rounded up to a multiple of the node count.
# The submit script must name its namelist on an NML= line (it may use
# ${CASE} from a CASE= line) and launch with "mpiexec -n ${NRANKS:-...}".
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
for a in "$@"; do
  case $a in
    NWRITERS=*) NWOPT="--max-writers ${a#NWRITERS=} --target-seconds 0.001" ;;
    PERNODE=1) PNOPT="--per-node" ;;
    *) EXTRA="${EXTRA},${a}" ;;
  esac
done
LAYOUT=$($PYTHON "$ROOT/tools/gpu_layout.py" "$NML" --gpus-per-node $GPUS_PER_NODE $NWOPT $PNOPT)
read NTILE NW NRANKS SELECT WPN <<< "$LAYOUT"
[ -n "$SELECT" ] || { echo "submit_gpu.sh: gpu_layout.py gave no layout for $NML" >&2; exit 1; }
echo "layout: $NTILE compute ranks + $NW writer(s) = $NRANKS ranks; select=$SELECT"
if [ -n "$WPN" ]; then
  EXTRA=",ROMS_GPU_NWRITERS=${WPN}${EXTRA}"
else
  EXTRA=",NUMA_BIND_NRANKS=${NTILE}${EXTRA}"
fi
qsub -l select=$SELECT -v "ROMS_GPU_PROBES=0,NRANKS=${NRANKS}${EXTRA}" "$SCRIPT"

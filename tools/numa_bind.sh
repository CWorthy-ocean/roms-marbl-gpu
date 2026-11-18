#!/bin/bash
# numa_bind.sh <cmd...>: run <cmd> on the CPU cores of the NUMA domain that
# holds this rank's GPU. Launch with "mpiexec --cpu-bind none ... numa_bind.sh exe args".
# GPU = the rank's node-local rank mod the number of GPUs, as roms_gpu's
# select_device does for compute ranks (the world rank where the launcher gives
# no local rank). CUDA ordinals are put in PCI bus order, so ordinal g is
# nvidia-smi index g. Everything is read from the node (nvidia-smi, /sys), so no
# node layout is assumed. If any lookup fails, the command runs unbound.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
r=${PALS_RANKID:-${PMI_RANK:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}}}
lr=${PALS_LOCAL_RANKID:-${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-}}}}
ls=${PALS_LOCAL_SIZE:-${OMPI_COMM_WORLD_LOCAL_SIZE:-${MPI_LOCALNRANKS:-${SLURM_NTASKS_PER_NODE:-}}}}
nd=${PALS_NODEID:-${SLURM_NODEID:-}}
# Writer ranks run unbound (binding them next to the GPU ranks slowed their
# file writes ~20%). With ROMS_GPU_NWRITERS=w they are the last w ranks of each
# node (before the forcing reader). Otherwise they are the ranks after the
# first NUMA_BIND_NRANKS (the GPU compute rank count; submit_gpu.sh passes it)
# in world order, counting the forcing readers out, as the model does
# (src/shared/comm.F90): with a reader on every node, the readers before this
# rank are one per earlier node, so the node index (PALS_NODEID or
# SLURM_NODEID) is taken off the world rank. Without a node index the world
# rank is used as it is, which leaves the last compute ranks of a multi-node
# job unbound. A forcing reader (ROMS_GPU_FRC_READER=1: the last rank of
# every node) runs unbound too.
# Writer and reader ranks never use a GPU, but with MPICH_GPU_SUPPORT_ENABLED=1
# MPI_Init gives each of them a CUDA context (~0.4 GB) on the first visible
# GPU, so with writers on every node they all landed next to compute rank 0
# (8 writers + a reader: 3.7 GB, which pushed Pac's largest tile into managed
# memory paging, 9x slower steps). io_exec shows each of them one GPU, its
# node-local rank mod the visible GPUs, which spreads the contexts over the
# GPUs. (Hiding every GPU is not possible: Cray's GTL then fails in
# cudaHostRegister, and MPICH_GPU_SUPPORT_ENABLED=0 on these ranks alone
# hangs MPI_Init.)
io_exec() {
  local devs n
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -r -a devs <<< "$CUDA_VISIBLE_DEVICES"
  else
    n=$(nvidia-smi -L 2>/dev/null | wc -l)
    devs=(); for (( i = 0; i < n; i++ )); do devs+=("$i"); done
  fi
  [ ${#devs[@]} -ge 1 ] && export CUDA_VISIBLE_DEVICES=${devs[$(( ${lr:-$r} % ${#devs[@]} ))]}
  exec "$@"
}
nrd=0
if [ "${ROMS_GPU_FRC_READER:-0}" = 1 ]; then
  nrd=1
  if [ -n "$lr" ] && [[ "$ls" =~ ^[0-9]+$ ]] && [ "$lr" -eq $(( ls - 1 )) ]; then io_exec "$@"; fi
fi
if [ -n "$ROMS_GPU_NWRITERS" ]; then
  if [ -n "$lr" ] && [[ "$ls" =~ ^[0-9]+$ ]] && [ "$lr" -ge $(( ls - ROMS_GPU_NWRITERS - nrd )) ]; then io_exec "$@"; fi
elif [ -n "$NUMA_BIND_NRANKS" ]; then
  below=$r                          # ranks before this one, readers not counted
  if [ "$nrd" = 1 ] && [[ "$nd" =~ ^[0-9]+$ ]]; then below=$(( r - nd )); fi
  if [ "$below" -ge "$NUMA_BIND_NRANKS" ]; then io_exec "$@"; fi
fi
ng=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ng" -ge 1 ]; then
  g=$(( ${lr:-$r} % ng ))
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$g" 2>/dev/null | tr 'A-Z' 'a-z')
  dev=${bus:4}                      # 00000000:03:00.0 -> 0000:03:00.0 (sysfs form)
  node=$(cat "/sys/bus/pci/devices/$dev/numa_node" 2>/dev/null)
  if [ -n "$node" ] && [ "$node" -ge 0 ] 2>/dev/null && [ -r "/sys/devices/system/node/node$node/cpulist" ]; then
    cpus=$(cat "/sys/devices/system/node/node$node/cpulist")
    [ "${NUMA_BIND_VERBOSE:-0}" = 1 ] && echo "numa_bind: rank $r -> GPU $g ($dev) NUMA $node cpus $cpus" >&2
    exec taskset -c "$cpus" "$@"
  fi
fi
exec "$@"

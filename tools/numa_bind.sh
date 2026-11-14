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
# Writer ranks run unbound (binding them next to the GPU ranks slowed their
# file writes ~20%). With ROMS_GPU_NWRITERS=w they are the last w ranks of each
# node (before the forcing reader); otherwise every world rank from
# NUMA_BIND_NRANKS on (the GPU compute rank count; submit_gpu.sh passes it).
# A forcing reader (ROMS_GPU_FRC_READER=1: the last rank of every node) runs
# unbound too.
nrd=0
if [ "${ROMS_GPU_FRC_READER:-0}" = 1 ]; then
  nrd=1
  if [ -n "$lr" ] && [[ "$ls" =~ ^[0-9]+$ ]] && [ "$lr" -eq $(( ls - 1 )) ]; then exec "$@"; fi
fi
if [ -n "$ROMS_GPU_NWRITERS" ]; then
  if [ -n "$lr" ] && [[ "$ls" =~ ^[0-9]+$ ]] && [ "$lr" -ge $(( ls - ROMS_GPU_NWRITERS - nrd )) ]; then exec "$@"; fi
elif [ -n "$NUMA_BIND_NRANKS" ] && [ "$r" -ge "$NUMA_BIND_NRANKS" ]; then exec "$@"; fi
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

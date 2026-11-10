# CPU backend: gfortran. GPU-first build: make gpu (nvfortran + CUDA).
# Working state is managed under USE_CUDA. One rank / one GPU / one tile.

FC_HOST   ?= gfortran
FFLAGS_HOST ?= -O2 -g -cpp -ffree-line-length-none -fcheck=bounds -Wall -Wextra -Wno-unused-dummy-argument -Wno-unused-parameter -Isrc/shared

FC_GPU    ?= nvfortran
FFLAGS_GPU ?= -O2 -g -cuda -gpu=cc80 -DUSE_CUDA -Mpreprocess -Mfree -Isrc/shared
FFLAGS_MPI ?= -DUSE_MPI

# conda nf-config --prefix is empty; take the first -L from --flibs.
NETCDF_FFLAGS ?= $(shell nf-config --fflags 2>/dev/null)
NETCDF_FLIBS  ?= $(shell nf-config --flibs 2>/dev/null)
NETCDF_LIBDIR := $(firstword $(patsubst -L%,%,$(filter -L%,$(NETCDF_FLIBS))))
ifneq ($(NETCDF_LIBDIR),)
  NETCDF_FLIBS += -Wl,-rpath,$(NETCDF_LIBDIR)
endif
# Parallel HDF5 NetCDF needs these even for serial Fortran I/O
# (--no-copy-dt-needed-entries does not pull them in via libnetcdf).
ifneq ($(wildcard $(NETCDF_LIBDIR)/libhdf5.so),)
  NETCDF_FLIBS += -lhdf5_hl -lhdf5
endif
ifneq ($(wildcard $(NETCDF_LIBDIR)/libmpi.so),)
  NETCDF_FLIBS += -lmpi
endif
# Fortran MPI bindings live in libmpifort (MPICH / Intel MPI).
ifneq ($(wildcard $(NETCDF_LIBDIR)/libmpifort.so),)
  NETCDF_FLIBS += -lmpifort
endif

# comm.F90 is always built with USE_MPI so every target shares one comm.mod.
# 'override' so the MPI/netCDF flags are appended even when FFLAGS_* is
# given on the make command line (a plain += is ignored in that case).
override FFLAGS_HOST += $(FFLAGS_MPI) $(NETCDF_FFLAGS)
override FFLAGS_GPU  += $(FFLAGS_MPI) $(NETCDF_FFLAGS)
LIBS        ?= $(NETCDF_FLIBS)

# Version printed at start-up: git describe of this checkout. It is passed to
# the main command only, so a new commit does not rebuild MARBL.
ROMS_GPU_VERSION ?= $(shell git -C $(CURDIR) describe --tags --always --dirty 2>/dev/null || echo unknown)
VERSION_DEF = -DROMS_GPU_VERSION=\"$(ROMS_GPU_VERSION)\"

SRC_SHARED = src/shared/kinds.F90 src/shared/decomp.F90 src/shared/tile_balance.F90 src/shared/arch.F90 \
             src/shared/comm.F90 src/shared/formulas.F90 src/shared/time_state.F90 \
             src/shared/run_timers.F90
SRC_DEV    = src/device/gpu_state.F90 src/device/column_solve.F90 src/device/halo.F90 src/device/rivers.F90 src/device/cdr_src.F90 src/device/cdr_lite.F90 src/device/carb_sens.F90 \
             src/device/geom.F90 src/device/obc.F90 src/device/step2d.F90 src/device/prsgrd.F90 src/device/pflx.F90 \
             src/device/advect.F90 src/device/rho_eos.F90 src/device/uv_rhs.F90 \
             src/device/passive_column.F90 src/device/passive.F90 \
             src/device/pre_step3d.F90 src/device/step3d_uv.F90 src/device/visc3d.F90 \
             src/device/step3d_t.F90 src/device/t3dmix.F90 src/device/lmd.F90 \
             src/device/bulk_coare.F90 src/device/wvlcty.F90
SRC_HOST   = src/host/step_driver.F90 src/host/feature_guard.F90
SRC_IO     = src/host/nc_io.F90 src/host/host_buf.F90 src/host/par_read.F90 src/host/frc_io.F90 \
             src/host/river_io.F90 src/host/cdr_io.F90 src/host/cdr_lite_io.F90 src/host/bry_io.F90 src/host/tide_io.F90 src/host/io_server.F90 src/host/marbl_step.F90 src/host/his_io.F90 src/host/cdr_lite_out.F90 src/host/random_out.F90 src/host/cdr_gas_out.F90 src/host/cdr_out.F90 src/host/upscale_out.F90 src/host/avg_out.F90 src/host/extract_out.F90 \
             src/host/case_io.F90
MODDIR     = build
GPUDIR     = build_gpu

# MARBL (upstream sources, compiled into the GPU build as host code). The
# list is in module-dependency order (from MARBL's makedep.py); each object
# is also chained to the one before it so a parallel make keeps the order.
# git submodule MARBL (CWorthy-ocean/MARBL @ 6e6b2f7); override with MARBL_SRC=<path>
MARBL_SRC ?= $(CURDIR)/MARBL/src
MARBL_NAMES = marbl_kinds_mod marbl_constants_mod marbl_logging marbl_pft_mod \
  marbl_settings_mod marbl_interface_public_types marbl_timing_mod \
  marbl_interface_private_types marbl_diagnostics_share_mod \
  marbl_abio_dic_diagnostics_mod marbl_abio_dic_interior_tendency_mod \
  marbl_co2calc_mod marbl_schmidt_number_mod marbl_abio_dic_surface_flux_mod \
  marbl_ciso_diagnostics_mod marbl_interior_tendency_share_mod \
  marbl_ciso_interior_tendency_mod marbl_ciso_surface_flux_mod marbl_debug_mod \
  marbl_oxygen marbl_utils_mod marbl_diagnostics_mod marbl_glo_avg_mod \
  marbl_init_tracer_metadata_mod marbl_init_mod marbl_restore_mod \
  marbl_temperature marbl_interior_tendency_mod marbl_saved_state_mod \
  marbl_nhx_surface_emis_mod marbl_surface_flux_share_mod marbl_surface_flux_mod \
  marbl_interface marbl_interface_constants
MARBL_OBJS = $(addprefix $(GPUDIR)/marbl/,$(addsuffix .o,$(MARBL_NAMES)))
# Host-side MARBL glue (instance, settings, per-column reference). Compiled
# on its own like comm.F90: its external calls must not disable nvfortran's
# cross-file inlining in the main command.
SRC_MARBL_HOST = src/host/marbl_host.F90
# Device MARBL kernels (part of the main command).
SRC_MARBL_DEV  = src/device/marbl_sflux.F90 src/device/marbl_interior.F90

.PHONY: all host tests gpu clean run

all: host

host: tests $(MODDIR)/roms_gpu

$(MODDIR):
	mkdir -p $(MODDIR)

$(GPUDIR):
	mkdir -p $(GPUDIR)

$(MODDIR)/roms_gpu: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) src/host/main.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_formulas: $(SRC_SHARED) tests/test_formulas.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_halo: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 tests/test_halo.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_mpi_halo: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 tests/test_mpi_halo.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_set_depth: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 tests/test_set_depth.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_step2d: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/rivers.F90 src/device/geom.F90 src/device/obc.F90 src/device/step2d.F90 tests/test_step2d.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_prsgrd: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 src/device/prsgrd.F90 tests/test_prsgrd.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_advect: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/advect.F90 tests/test_advect.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_rho_eos: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 src/device/rho_eos.F90 tests/test_rho_eos.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_dyn_core: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_dyn_core.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_obc: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_obc.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_bulk: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_bulk.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_tides: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_tides.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_his: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_his.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_lmd: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 \
                    src/device/geom.F90 src/device/rho_eos.F90 src/device/lmd.F90 \
                    tests/test_lmd.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_io: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 \
                   src/device/geom.F90 src/device/bulk_coare.F90 $(SRC_IO) tests/test_io.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

tests: $(MODDIR)/test_formulas $(MODDIR)/test_halo $(MODDIR)/test_mpi_halo $(MODDIR)/test_set_depth $(MODDIR)/test_step2d $(MODDIR)/test_prsgrd $(MODDIR)/test_advect $(MODDIR)/test_rho_eos $(MODDIR)/test_lmd $(MODDIR)/test_io $(MODDIR)/test_obc $(MODDIR)/test_bulk $(MODDIR)/test_tides $(MODDIR)/test_his $(MODDIR)/test_dyn_core
	$(MODDIR)/test_formulas
	$(MODDIR)/test_halo
	mpiexec -n 2 $(MODDIR)/test_mpi_halo
	$(MODDIR)/test_set_depth
	$(MODDIR)/test_step2d
	$(MODDIR)/test_prsgrd
	$(MODDIR)/test_advect
	$(MODDIR)/test_rho_eos
	$(MODDIR)/test_lmd
	$(MODDIR)/test_io
	$(MODDIR)/test_obc
	$(MODDIR)/test_bulk
	$(MODDIR)/test_tides
	$(MODDIR)/test_his
	$(MODDIR)/test_dyn_core

run: $(MODDIR)/roms_gpu
	$(MODDIR)/roms_gpu geom

# comm.F90 is compiled on its own: with USE_MPI its external MPI calls
# stop nvfortran's cross-file inlining for every file that follows it in
# the same command, and the device kernels then lose the inlined
# formulas functions (rho_eos went from 96 registers / no stack to 154 /
# 72-byte stack and a 1x1x1 launch grid, 265x slower). Its object is
# linked in; the main command sees only comm_mod.mod.
SRC_SHARED_NOCOMM = $(filter-out src/shared/comm.F90,$(SRC_SHARED))

$(GPUDIR)/marbl:
	mkdir -p $(GPUDIR)/marbl
$(GPUDIR)/marbl/%.o: $(MARBL_SRC)/%.F90 | $(GPUDIR)/marbl
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c $< -o $@
# nvfortran 25.9 -O2 miscompiles the array-section WHERE constructs in
# MARBL's interior module (segfault in compute_autotroph_nutrient_uptake for
# the second autotroph); this host reference code is built at -O0.
$(GPUDIR)/marbl/marbl_interior_tendency_mod.o: $(MARBL_SRC)/marbl_interior_tendency_mod.F90 | $(GPUDIR)/marbl
	$(FC_GPU) $(filter-out -O2 -O3,$(FFLAGS_GPU)) -O0 -module $(GPUDIR) -c $< -o $@
# chain: object n depends (order-only) on object n-1
define MARBL_CHAIN
$(GPUDIR)/marbl/$(1).o: | $(GPUDIR)/marbl/$(2).o
endef
$(foreach i,$(shell seq 2 $(words $(MARBL_NAMES))),\
  $(eval $(call MARBL_CHAIN,$(word $(i),$(MARBL_NAMES)),$(word $(shell expr $(i) - 1),$(MARBL_NAMES)))))
.PHONY: marbl
marbl: $(MARBL_OBJS)

# The cached objects (MARBL, kinds_pre, marbl_host) do not depend on the
# flags by themselves: build_gpu/flags.txt holds the compiler and flags they
# were built with, and a change rewrites it, which rebuilds them.
GPU_FLAGS = $(GPUDIR)/flags.txt
ifneq ($(filter gpu $(GPUDIR)/%,$(MAKECMDGOALS)),)
  $(shell mkdir -p $(GPUDIR); printf '%s\n' '$(FC_GPU) $(FFLAGS_GPU)' | cmp -s - $(GPU_FLAGS) || \
    printf '%s\n' '$(FC_GPU) $(FFLAGS_GPU)' > $(GPU_FLAGS))
endif
$(MARBL_OBJS) $(GPUDIR)/kinds_pre.o: $(GPU_FLAGS)
$(GPU_FLAGS):
	mkdir -p $(GPUDIR); printf '%s\n' '$(FC_GPU) $(FFLAGS_GPU)' > $@

$(GPUDIR)/kinds_pre.o: src/shared/kinds.F90 | $(GPUDIR)
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c src/shared/kinds.F90 -o $(GPUDIR)/kinds_pre.o
$(GPUDIR)/marbl_host.o: $(SRC_MARBL_HOST) $(GPUDIR)/kinds_pre.o $(MARBL_OBJS)
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c $(SRC_MARBL_HOST) -o $(GPUDIR)/marbl_host.o

gpu: $(GPUDIR)/kinds_pre.o $(GPUDIR)/marbl_host.o | $(GPUDIR)
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c src/shared/comm.F90 -o $(GPUDIR)/comm.o
	$(FC_GPU) $(FFLAGS_GPU) $(VERSION_DEF) -module $(GPUDIR) -o $(GPUDIR)/roms_gpu_cuda \
	  $(SRC_SHARED_NOCOMM) $(SRC_DEV) $(SRC_MARBL_DEV) $(SRC_IO) $(SRC_HOST) src/host/main.F90 \
	  $(GPUDIR)/comm.o $(GPUDIR)/marbl_host.o $(MARBL_OBJS) $(LIBS)

# Column-by-column check of the device MARBL interior kernels against the
# unchanged MARBL code (host).
$(GPUDIR)/test_marbl_interior: $(GPUDIR)/kinds_pre.o $(GPUDIR)/marbl_host.o tests/test_marbl_interior.F90 \
                               $(SRC_MARBL_DEV) | $(GPUDIR)
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c src/shared/comm.F90 -o $(GPUDIR)/comm.o
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -o $@ \
	  $(SRC_SHARED_NOCOMM) $(SRC_MARBL_DEV) tests/test_marbl_interior.F90 \
	  $(GPUDIR)/comm.o $(GPUDIR)/marbl_host.o $(MARBL_OBJS) $(LIBS)

# Column-by-column check of the device MARBL surface-flux kernel against
# the unchanged MARBL code (host). Built the same way as the model.
$(GPUDIR)/test_marbl_sflux: $(GPUDIR)/kinds_pre.o $(GPUDIR)/marbl_host.o tests/test_marbl_sflux.F90 \
                            $(SRC_MARBL_DEV) | $(GPUDIR)
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c src/shared/comm.F90 -o $(GPUDIR)/comm.o
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -o $@ \
	  $(SRC_SHARED_NOCOMM) $(SRC_MARBL_DEV) tests/test_marbl_sflux.F90 \
	  $(GPUDIR)/comm.o $(GPUDIR)/marbl_host.o $(MARBL_OBJS) $(LIBS)

clean:
	rm -rf $(MODDIR) $(GPUDIR)

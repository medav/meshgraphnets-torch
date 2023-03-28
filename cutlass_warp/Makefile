
CUTLASS_PATH=/nobackup/medavies/cutlass
CUTLASS_INC=$(CUTLASS_PATH)/include
CUTLASS_LIB=$(CUTLASS_PATH)/build/tools/library

LIBS?=
LIBS+=-L$(CUTLASS_LIB) -lcutlass

INCS?=
INCS+=-I$(CUTLASS_INC)
INCS+=-I$(CUTLASS_PATH)/tools/util/include


.PHONY: default run objdump

default: run

APP=cutlass_test

$(APP): $(APP).cu
	/usr/local/cuda/bin/nvcc \
		-DCUTLASS_ENABLE_L2_PREFETCH \
		--expt-relaxed-constexpr \
		-O3 \
		-std=c++17 \
		-o $@ \
		-arch=sm_80 \
		$(LIBS) \
		$(INCS) \
		$<

clean:
	rm -f $(APP)

run: $(APP)
	LD_LIBRARY_PATH=$(CUTLASS_LIB) ./$(APP)

objdump: $(APP)
	/usr/local/cuda/bin/cuobjdump --dump-ptx $(APP)

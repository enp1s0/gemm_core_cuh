NVCC=nvcc
NVCCFLAGS=-std=c++11 -I./cutf -arch=sm_61 -Xcompiler=-fopenmp
TARGET=gemm_core_test.out

$(TARGET):test.cu gemm_core.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

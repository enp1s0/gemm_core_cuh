NVCC=nvcc
NVCCFLAGS=-std=c++11 -I./cutf -arch=sm_61
TARGET=gemm_core_test.out

$(TARGET):test.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../include -arch=sm_60 -Xcompiler=-fopenmp --ptxas-options=-v
TARGET=gevm_test.out

$(TARGET):test.cu ../../include/gemm_core/gemm_core.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

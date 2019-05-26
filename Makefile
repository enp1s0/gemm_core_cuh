NVCC=nvcc
NVCCFLAGS=-std=c++11 -I./cutf -arch=sm_60 -Xcompiler=-fopenmp --ptxas-options=-v -lcublas
TARGET=gemm_core_test.out

$(TARGET):test.cu gemm_core.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

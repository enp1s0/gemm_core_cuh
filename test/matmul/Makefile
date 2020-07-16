NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../cutf -I../../ -arch=sm_60 -Xcompiler=-fopenmp --ptxas-options=-v
TARGET=matmul_text.out

$(TARGET):test.cu ../../gemm_core.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

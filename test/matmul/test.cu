#include <iostream>
#include <random>
#include <gemm_core.cuh>

constexpr unsigned N = 16;
constexpr unsigned K = 32;

template <class T>
std::string get_type_name();
template <> std::string get_type_name<float>(){return "float";}
template <> std::string get_type_name<half>(){return "half";}

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <class T, unsigned K>
__global__ void test_matmul_16x16_kernel(T* const c, const T* const a, const T* const b){
	mtk::matmul_core16x16<K>(c, N, a, N, b, K, threadIdx.x & 0x1f);
}

template <class T>
void test_matmul(){
	T* a;
	T* b;
	T* c;

	std::printf("%s\n", get_type_name<T>().c_str());

	cudaMallocHost(&a, N * K * sizeof(T));
	cudaMallocHost(&b, K * N * sizeof(T));
	cudaMallocHost(&c, N * N * sizeof(T));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for(unsigned i = 0; i < K * N; i++){
		a[i] = convert<T>(dist(mt));
	}
	for(unsigned i = 0; i < K * N; i++){
		b[i] = convert<T>(dist(mt));
	}
	for(unsigned i = 0; i < N * N; i++){
		c[i] = convert<T>(0.0f);
	}

	cudaDeviceSynchronize();
	test_matmul_16x16_kernel<T, K><<<1, 32>>>(c, a, b);
	cudaDeviceSynchronize();

	float error = 0.0f;
	for(unsigned i = 0; i < N; i++){
		for(unsigned j = 0; j < N; j++){
			float sum = 0.0f;
			for(unsigned k = 0; k < K; k++){
				sum += convert<float>(a[k * N + i]) * convert<float>(b[j * K + k]);
			}
			error = std::max(error, std::abs(convert<float>(c[i + j * N]) - sum));
		}
	}
	std::printf("error = %e\n", error);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}

int main() {
	test_matmul<float>();
	test_matmul<half >();
}
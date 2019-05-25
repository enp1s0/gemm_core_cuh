#include <iostream>
#include <random>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/error.hpp>
#include "gemm_core.cuh"

constexpr std::size_t m = (1 << 14) - 1;
constexpr std::size_t n = (1 << 14) - 1;
constexpr std::size_t k = (1 << 14) - 1;

constexpr std::size_t warp_size = 32;
constexpr std::size_t block_size = 256;

template <class T>
void print_gemm_info(const std::size_t m, const std::size_t n, const std::size_t k, const std::size_t grid_size, std::size_t block_size, double elapsed_time){
	std::cout<<"Matrix size : "<<m<<", "<<n<<", "<<k<<std::endl;
	std::cout<<"Memory      : "<<((m * n + n * k + k * m) * sizeof(T) / (1024.0 * 1024.0))<<" MB"<<std::endl;
	std::cout<<"Grid size   : "<<grid_size<<std::endl;
	std::cout<<"Block size  : "<<block_size<<std::endl;
	std::cout<<"Performance : "<<(m * n * k * 2 / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0)) <<" TFLOPS"<<std::endl;
}

template <class T, unsigned num_warps>
__device__ void load64x64(
		T* const dst,
		const T* const src, const std::size_t m, const std::size_t n,
		const std::size_t start_m, const std::size_t start_n,
		const unsigned unique_id
		){}

template <>
__device__ void load64x64<float, 1>(
		float* const dst,
		const float* const src, const std::size_t m, const std::size_t n,
		const std::size_t start_m, const std::size_t start_n,
		const unsigned unique_id
		){
	constexpr std::size_t dim = 64;
	for(unsigned i = 0; i < dim; i++){
		const auto load_n = start_n + i;

		for(unsigned j = 0; j < (dim / warp_size); j++){
			const auto load_m = start_m + j * warp_size + unique_id;
			float tmp = 0.0f;
			if(load_m < m && load_n < n){
				tmp = src[load_m + load_n * m];
			}

			dst[j + i * dim] = tmp;
		}
	}
}

template <class T, unsigned num_warps>
__device__ void store64x64(
		T* const dst,const std::size_t m, const std::size_t n,
		const std::size_t start_m, const std::size_t start_n,
		const T* const src, 
		const unsigned unique_id
		){}

template <>
__device__ void store64x64<float, 1>(
		float* const dst, const std::size_t m, const std::size_t n,
		const std::size_t start_m, const std::size_t start_n,
		const float* const src, 
		const unsigned unique_id
		){
	constexpr std::size_t dim = 64;
	for(unsigned i = 0; i < dim; i++){
		const auto load_n = start_n + i;
		if(load_n >= n) return;

		for(unsigned j = 0; j < (dim / warp_size); j++){
			const auto load_m = start_m + j * warp_size + unique_id;
			if(load_m >= m) break;

			dst[load_m + load_n * m] = src[j + i * dim];
		}
	}
}

template <class T, unsigned num_warps>
__global__ void test_gemm_16x16_kernel(T* const c, const T* const a, const T* const b, const std::size_t m, const std::size_t n, const std::size_t k){}

template <>
__global__ void test_gemm_16x16_kernel<float, 1>(float* const c, const float* const a, const float* const b, const std::size_t m, const std::size_t n, const std::size_t k){
	constexpr std::size_t num_blocks_per_grid = block_size / warp_size;
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	constexpr std::size_t dim = 64;
	const auto num_m_blocks = (m + dim - 1) / dim;
	const auto num_n_blocks = (n + dim - 1) / dim;
	const auto num_k_blocks = (k + dim - 1) / dim;
	const auto matrix_id = tid / warp_size;
	const unsigned unique_id = tid & (warp_size - 1); 

	const std::size_t block_m = matrix_id / (num_n_blocks * num_k_blocks);
	const std::size_t block_n = (matrix_id % (num_n_blocks * num_k_blocks)) / num_k_blocks;
	//const std::size_t block_k = (matrix_id % (num_n_blocks * num_k_blocks)) % num_k_blocks;

	__shared__ float shared_a[16 * 16 * 4 * 4];
	__shared__ float shared_b[16 * 16 * 4 * 4];
	__shared__ float shared_c[16 * 16 * 4 * 4];


	for(std::size_t ik = 0; ik < num_k_blocks; ik++){
		// Load C
		load64x64<float, 1>(shared_c,
				c, m, n,
				block_m * dim, block_n * dim,
				unique_id);
		// Load A
		load64x64<float, 1>(shared_a,
				a, m, k,
				ik * dim, block_n * dim,
				unique_id);
		// Load B
		load64x64<float, 1>(shared_b,
				b, k, n,
				block_m * dim, ik * dim,
				unique_id);
		__syncthreads();

		for(unsigned i = 0; i < 16 / (block_size/warp_size); i++){
			const auto sub_block_m = 2 * i + (matrix_id & 0x3) / 4;
			const auto sub_block_n = matrix_id & 0x3;
			for(unsigned j = 0; j < (64/16); j++){
				gemm_core16x16<float, 1>(
						shared_c + sub_block_n * dim * 16 + sub_block_m * 16,
						shared_a + sub_block_m * 16 + j * (dim * 16),
						shared_b + j * 16 + sub_block_n * (dim * 16),
						dim, tid & 0x1f);
			}
		}

		__syncthreads();

		// Store C
		store64x64<float, 1>(
				c, m, n,
				block_m * dim, block_n * dim,
				shared_c,
				unique_id
				);
	}
}

template <class Func>
double get_elapsed_time(Func func){
	const auto start = std::chrono::system_clock::now();
	func();
	const auto end = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
}


template <class T, unsigned num_warps>
void test_gemm_16x16(T* const c, const T* const a, const T* const b, const std::size_t m, const std::size_t n, const std::size_t k){}

template <>
void test_gemm_16x16<float, 1>(float* const c, const float* const a, const float* const b, const std::size_t m, const std::size_t n, const std::size_t k){
	constexpr std::size_t dim = 64;
	constexpr std::size_t C = 2;
	const auto num_m_blocks = (m + dim - 1) / dim;
	const auto num_n_blocks = (n + dim - 1) / dim;
	const auto num_k_blocks = (k + dim - 1) / dim;

	const auto grid_size = num_n_blocks * num_m_blocks;

	const auto elapsed_time = get_elapsed_time(
			[&a, &b, &c, &m, &n, &k, &grid_size](){
			for(std::size_t i = 0;i < C; i++)
			test_gemm_16x16_kernel<float, 1><<<grid_size, block_size>>>(c, a, b, m, n, k);
			CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
			});

	print_gemm_info<float>(m, n, k, grid_size, block_size, elapsed_time / C);
}

int main(){
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * k);
	auto d_b = cutf::memory::get_device_unique_ptr<float>(k * n);
	auto d_c = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * k);
	auto h_b = cutf::memory::get_host_unique_ptr<float>(k * n);
	auto h_c = cutf::memory::get_host_unique_ptr<float>(m * n);

#pragma omp parallel for
	for(std::size_t i = 0; i < m * k; i++) h_a.get()[i] = dist(mt);
#pragma omp parallel for
	for(std::size_t i = 0; i < k * n; i++) h_b.get()[i] = dist(mt);
#pragma omp parallel for
	for(std::size_t i = 0; i < m * n; i++) h_c.get()[i] = 0.0f;

	cutf::memory::copy(d_a.get(), h_a.get(), m * k);
	cutf::memory::copy(d_b.get(), h_b.get(), k * n);
	cutf::memory::copy(d_c.get(), h_c.get(), m * n);

	test_gemm_16x16<float, 1>(d_c.get(), d_a.get(), d_b.get(), m, n, k);

	cutf::memory::copy(d_c.get(), h_c.get(), m * n);
}

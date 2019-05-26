#include <iostream>
#include <random>
#include <memory>
#include <chrono>
#include <omp.h>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <cutf/error.hpp>
#include <cutf/type.hpp>
#include "gemm_core.cuh"

constexpr std::size_t max_m = 1 << 14;
constexpr std::size_t max_n = 1 << 14;
constexpr std::size_t max_k = 1 << 14;

constexpr std::size_t warp_size = 32;
constexpr std::size_t block_size = 512;

using test_t = half;

template <class T>
std::string get_type_name();
template <> std::string get_type_name<float>(){return "float";}
template <> std::string get_type_name<half>(){return "half";}

template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr){
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			const auto val = cutf::type::cast<float>(ptr[j * m + i]);
			if(val < 0.0f){
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}


template <class T>
void print_gemm_info(const std::size_t m, const std::size_t n, const std::size_t k, const std::size_t grid_size, std::size_t block_size, double elapsed_time){
	std::cout<<"Type        : "<<get_type_name<T>()<<std::endl;
	std::cout<<"Matrix size : "<<m<<", "<<n<<", "<<k<<std::endl;
	std::cout<<"Memory      : "<<((m * n + n * k + k * m) * sizeof(T) / (1024.0 * 1024.0))<<" MB"<<std::endl;
	std::cout<<"Grid size   : "<<grid_size<<std::endl;
	std::cout<<"Block size  : "<<block_size<<std::endl;
	std::cout<<"Elapsed time: "<<elapsed_time<<" [s]"<<std::endl;
	std::cout<<"Performance : "<<(m * n * k * 2 / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0)) <<" TFLOPS"<<std::endl;
}

template <class T, unsigned num_warps>
__device__ void load64x64(
		T* const dst,
		const T* const src, const std::size_t m, const std::size_t n,
		const std::size_t start_m, const std::size_t start_n,
		const unsigned unique_id, const unsigned warp_id
		){
	constexpr std::size_t dim = 64;
	if(start_m + dim >= m || start_n + dim >= n){
		for(unsigned i = warp_id; i < dim; i+=num_warps){
			const auto load_n = start_n + i;

			for(unsigned j = 0; j < dim; j += warp_size){
				const auto load_m = start_m + j + unique_id;
				T tmp = cutf::type::cast<T>(0.0f);
				if(load_m < m && load_n < n){
					tmp = __ldg( &src[load_m + load_n * m] );
				}

				dst[j + unique_id + i * dim] = tmp;
			}
		}
	}else{
#pragma unroll
		for(unsigned i = warp_id; i < dim; i+=num_warps){
			const auto load_n = start_n + i;

#pragma unroll
			for(unsigned j = 0; j < dim; j += warp_size){
				const auto load_m = start_m + j + unique_id;
				
				dst[j + unique_id + i * dim] = __ldg( &src[load_m + load_n * m]);
			}
		}
	}
}

template <class T, unsigned num_warps>
__device__ void store64x64(
		T* const dst,const std::size_t m, const std::size_t n,
		const std::size_t start_m, const std::size_t start_n,
		const T* const src, 
		const unsigned unique_id, const unsigned warp_id
		){
	constexpr std::size_t dim = 64;
	if(start_m + dim >= m || start_n + dim >= n){
		for(unsigned i = warp_id; i < dim; i+=num_warps){
			const auto load_n = start_n + i;

			for(unsigned j = 0; j < dim; j += warp_size){
				const auto load_m = start_m + j + unique_id;
				if(load_m < m && load_n < n){
					dst[load_m + load_n * m] = src[j + unique_id + i * dim];
				}

			}
		}
	}else{
#pragma unroll
		for(unsigned i = warp_id; i < dim; i+=num_warps){
			const auto load_n = start_n + i;

#pragma unroll
			for(unsigned j = 0; j < dim; j += warp_size){
				const auto load_m = start_m + j + unique_id;
				
				dst[load_m + load_n * m] = src[j + unique_id + i * dim];
			}
		}
	}
}

template <class T>
__global__ void test_gemm_16x16_kernel(T* const c, const T* const a, const T* const b, const std::size_t m, const std::size_t n, const std::size_t k){
	constexpr std::size_t dim = 64;
	const auto num_m_blocks = (m + dim - 1) / dim;
	const auto num_k_blocks = (k + dim - 1) / dim;
	const auto matrix_id = blockIdx.x;
	const unsigned unique_id = threadIdx.x & (warp_size - 1); 
	const unsigned warp_id = threadIdx.x >> 5;

	const std::size_t block_m = matrix_id % num_m_blocks;
	const std::size_t block_n = matrix_id / num_m_blocks;

	__shared__ T shared_a[16 * 16 * 4 * 4];
	__shared__ T shared_b[16 * 16 * 4 * 4];
	__shared__ T shared_c[16 * 16 * 4 * 4];


	for(std::size_t ik = 0; ik < num_k_blocks; ik++){
		// Load C
		const auto block_m_start = block_m * dim;
		const auto block_n_start = block_n * dim;
		const auto block_k_start = ik * dim;
		load64x64<T, (block_size/warp_size)>(shared_c,
				c, m, n,
				block_m_start, block_n_start,
				unique_id, warp_id);
		// Load A
		load64x64<T, (block_size/warp_size)>(shared_a,
				a, m, k,
				block_m_start, block_k_start,
				unique_id, warp_id);
		// Load B
		load64x64<T, (block_size/warp_size)>(shared_b,
				b, k, n,
				block_k_start, block_n_start,
				unique_id, warp_id);

		__syncthreads();

		constexpr unsigned num_blocks_per_grid = block_size / warp_size;
		for(unsigned i = 0; i < 16 / num_blocks_per_grid; i++){
			const auto sub_block_m = 2 * i + (warp_id / 4);
			const auto sub_block_n = warp_id & (dim/16 - 1);
			for(unsigned j = 0; j < (dim/16); j++){
				gemm_core16x16<T, 1>(
						shared_c + sub_block_n * dim * 16 + sub_block_m * 16,
						shared_a + sub_block_m * 16 + j * (dim * 16),
						shared_b + j * 16 + sub_block_n * (dim * 16),
						dim, unique_id);
			}
		}

		__syncthreads();

		// Store C
		store64x64<T, (block_size/warp_size)>(
				c, m, n,
				block_m * dim, block_n * dim,
				shared_c,
				unique_id, warp_id
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

template <class T>
float get_norm(const T* const ptr, std::size_t size){
	const auto num_threads = 100;
	std::unique_ptr<float[]> sums(new float [num_threads]);
	for(std::size_t i = 0; i < num_threads; i++){
		sums.get()[i] = 0.0f;
	}
#pragma omp parallel for
	for(std::size_t i = 0; i < size; i++){
		const auto val = cutf::type::cast<float>(ptr[i]); 
		sums.get()[omp_get_thread_num()] += val * val;
	}
	float sum = 0.0f;
	for(std::size_t i = 0; i < num_threads; i++){
		sum += sums.get()[i];
	}
	return std::sqrt(sum);
}


template <class T, unsigned num_warps>
void test_gemm_16x16(T* const c, const T* const a, const T* const b, const std::size_t m, const std::size_t n, const std::size_t k, unsigned int print_mode = 0){
	constexpr std::size_t dim = 64;
	constexpr std::size_t C = 8;
	const auto num_m_blocks = (m + dim - 1) / dim;
	const auto num_n_blocks = (n + dim - 1) / dim;

	const auto grid_size = num_n_blocks * num_m_blocks;

	const auto elapsed_time = get_elapsed_time(
			[&a, &b, &c, &m, &n, &k, &grid_size](){
			for(std::size_t i = 0;i < C; i++)
			test_gemm_16x16_kernel<T><<<grid_size, block_size>>>(c, a, b, m, n, k);
			CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
			});

	if(print_mode == 0)
		print_gemm_info<T>(m, n, k, grid_size, block_size, elapsed_time / C);
	else
		std::printf("%lu,%lu,%lu,%.5f,%.5f\n", m, n, k, elapsed_time / C, 2 * (m * n * k) / (elapsed_time / C) / (1024.0 * 1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char** argv){
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	if(argc == 4){
		const auto m = std::stoul(argv[1]);
		const auto n = std::stoul(argv[2]);
		const auto k = std::stoul(argv[3]);
		auto d_a = cutf::memory::get_device_unique_ptr<test_t>(m * k);
		auto d_b = cutf::memory::get_device_unique_ptr<test_t>(k * n);
		auto d_c = cutf::memory::get_device_unique_ptr<test_t>(m * n);
		auto h_a = cutf::memory::get_host_unique_ptr<test_t>(m * k);
		auto h_b = cutf::memory::get_host_unique_ptr<test_t>(k * n);
		auto h_c = cutf::memory::get_host_unique_ptr<test_t>(m * n);

#pragma omp parallel for
		for(std::size_t i = 0; i < m * k; i++) h_a.get()[i] = cutf::type::cast<test_t>(dist(mt));
#pragma omp parallel for
		for(std::size_t i = 0; i < k * n; i++) h_b.get()[i] = cutf::type::cast<test_t>(dist(mt));
#pragma omp parallel for
		for(std::size_t i = 0; i < m * n; i++) h_c.get()[i] = cutf::type::cast<test_t>(0.0f);

		cutf::memory::copy(d_a.get(), h_a.get(), m * k);
		cutf::memory::copy(d_b.get(), h_b.get(), k * n);
		cutf::memory::copy(d_c.get(), h_c.get(), m * n);

		test_gemm_16x16<test_t, 1>(d_c.get(), d_a.get(), d_b.get(), m, n, k);

		cutf::memory::copy(h_c.get(), d_c.get(), m * n);
		const auto c_norm = get_norm(h_c.get(), m * n);

		// Validation
		auto cublas = cutf::cublas::get_cublas_unique_ptr();
		test_t alpha = cutf::type::cast<test_t>(1.0f), beta = cutf::type::cast<test_t>(-1.0f);
		CUTF_HANDLE_ERROR(
				cutf::cublas::gemm(*cublas.get(),
					CUBLAS_OP_N, CUBLAS_OP_N,
					m, n, k,
					&alpha,
					d_a.get(), m,
					d_b.get(), k,
					&beta,
					d_c.get(), m
					));
		cutf::memory::copy(h_c.get(), d_c.get(), m * n);

		const auto error = get_norm(h_c.get(), m * n);

		std::cout<<"Error    : "<<(error/c_norm)<<std::endl;
	} else {
		std::cout<<"m,n,k,time,tflops,("<<get_type_name<test_t>()<<")"<<std::endl;
		auto h_a = cutf::memory::get_host_unique_ptr<test_t>((max_m + 1) * (max_k + 1));
		auto h_b = cutf::memory::get_host_unique_ptr<test_t>((max_k + 1) * (max_n + 1));
		auto h_c = cutf::memory::get_host_unique_ptr<test_t>((max_m + 1) * (max_n + 1));

#pragma omp parallel for
		for(std::size_t i = 0; i < max_m * max_k; i++) h_a.get()[i] = cutf::type::cast<test_t>(dist(mt));
#pragma omp parallel for
		for(std::size_t i = 0; i < max_k * max_n; i++) h_b.get()[i] = cutf::type::cast<test_t>(dist(mt));
#pragma omp parallel for
		for(std::size_t i = 0; i < max_m * max_n; i++) h_c.get()[i] = cutf::type::cast<test_t>(0.0f);

		for(std::size_t m = (1 << 8); m <= max_m; m<<=1){
			for(int i = -1; i <= 1; i++){
				for(std::size_t n = (1 << 8); n < max_n; n<<=1){
					for(int j = -1; j <= 1; j++){
						for(std::size_t k = (1 << 8); k <= max_k; k<<=1){
							for(int l = -1; l <= 1; l++){
								auto d_a = cutf::memory::get_device_unique_ptr<test_t>((m + i) * (k + l));
								auto d_b = cutf::memory::get_device_unique_ptr<test_t>((k + l) * (n + j));
								auto d_c = cutf::memory::get_device_unique_ptr<test_t>((m + i) * (n + j));
								cutf::memory::copy(d_a.get(), h_a.get(), (m + i) * (k + l));
								cutf::memory::copy(d_b.get(), h_b.get(), (k + l) * (n + j));
								cutf::memory::copy(d_c.get(), h_c.get(), (m + i) * (n + j));
								test_gemm_16x16<test_t, 1>(d_c.get(), d_a.get(), d_b.get(), m + i, n + j, k + l, 1);
							}
						}
					}
				}
			}
		}
	}
}

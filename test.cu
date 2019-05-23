#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include "gemm_core.cuh"

constexpr std::size_t m = (1 << 14) - 1;
constexpr std::size_t n = (1 << 14) - 1;
constexpr std::size_t k = (1 << 14) - 1;

constexpr std::size_t warp_size = 32;
constexpr std::size_t block_size = 256;

void print_gemm_info(const std::size_t m, const std::size_t n, const std::size_t k, const std::size_t grid_size, std::size_t block_size){
	std::cout<<"Matrix size : "<<m<<", "<<n<<", "<<k<<std::endl;
	std::cout<<"Grid size   : "<<grid_size<<std::endl;
	std::cout<<"Block size  : "<<block_size<<std::endl;
}

template <class T, unsigned num_warps>
__global__ void test_gemm_16x16_kernel(T* const c, const T* const a, const T* const b, const std::size_t m, const std::size_t n, const std::size_t k){}

template <>
__global__ void test_gemm_16x16_kernel<float, 1>(float* const c, const float* const a, const float* const b, const std::size_t m, const std::size_t n, const std::size_t k){
}


template <class T, unsigned num_warps>
void test_gemm_16x16(T* const c, const T* const a, const T* const b, const std::size_t m, const std::size_t n, const std::size_t k){}

template <>
void test_gemm_16x16<float, 1>(float* const c, const float* const a, const float* const b, const std::size_t m, const std::size_t n, const std::size_t k){
	const auto num_m_blocks = (m + 15) / 16;
	const auto num_n_blocks = (n + 15) / 16;
	const auto num_k_blocks = (k + 15) / 16;

	const auto num_threads = (num_m_blocks * num_n_blocks * num_k_blocks) * warp_size;
	const auto grid_size = num_threads / block_size;

	print_gemm_info(m, n, k, grid_size, block_size);
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

#include <iostream>
#include <random>
#include <cutf/memory.hpp>

constexpr std::size_t m = (1 << 20) - 1;
constexpr std::size_t n = (1 << 20) - 1;
constexpr std::size_t k = (1 << 20) - 1;

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

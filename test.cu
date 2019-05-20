#include <iostream>
#include <cutf/memory.hpp>

constexpr std::size_t m = (1 << 20) - 1;
constexpr std::size_t n = (1 << 20) - 1;
constexpr std::size_t k = (1 << 20) - 1;

int main(){
	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * k);
	auto d_b = cutf::memory::get_device_unique_ptr<float>(k * n);
	auto d_c = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * k);
	auto h_b = cutf::memory::get_host_unique_ptr<float>(k * n);
	auto h_c = cutf::memory::get_host_unique_ptr<float>(m * n);
}

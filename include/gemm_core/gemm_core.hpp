#ifndef __GEMM_CORE_HPP__
#define __GEMM_CORE_HPP__
#include <cuda_fp16.h>

namespace mtk {
namespace gemm_core {
template <unsigned K = 16, bool sync_before_storing = false>
__device__ inline void gemm_core16x16(double* const c, const unsigned ldm_c, const double* const a, const unsigned ldm_a, const double* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	double tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		double sum = 0.0;
		for (unsigned k = 0; k < K; k += 1) {
			sum = fma(a[y + ldm_a * k], b[x * ldm_b + k], sum);
		}
		tmp_c[i / 2] = sum;
	}

	if (sync_before_storing) {
		__syncthreads();
	}

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		c[x * ldm_c + y] += tmp_c[i / 2];
	}
}

template <unsigned K = 16, bool sync_before_storing = false>
__device__ inline void gemm_core16x16(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		float sum = 0.0f;
		for (unsigned k = 0; k < K; k += 1) {
			sum = fmaf(a[y + ldm_a * k], b[x * ldm_b + k], sum);
		}
		tmp_c[i / 2] = sum;
	}

	if (sync_before_storing) {
		__syncthreads();
	}

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		c[x * ldm_c + y] += tmp_c[i / 2];
	}
}

template <unsigned K = 16, bool sync_before_storing = false>
__device__ inline void gemm_core16x16(half* const c, const unsigned ldm_c, const half* const a, const unsigned ldm_a, const half* const b, const unsigned ldm_b, const unsigned unique_id) {
	const auto y = unique_id & 0xf;
	const auto x = (unique_id >> 4) << 3;
	unsigned i, k;
	half2 sums[8];

#pragma unroll
	for (unsigned i = 0; i < 8; i++)
		sums[i] = __float2half2_rn(0.0);

#pragma unroll
	for (k = 0; k < K; k += 2) {
		const auto a2 = __halves2half2(a[k * ldm_a + y], a[(k + 1) * ldm_a + y]);

		const half2 *b2 = (half2*)(b + x * ldm_b + k);
		for (i = 0; i < 8; i++) {
			sums[i] = __hfma2(a2, *(b2), sums[i]);
			b2 += ldm_b / 2;
		}
	}

	if (sync_before_storing) {
		__syncthreads();
	}

	for (i = 0; i < 8; i++) {
		const auto sum = sums[i];
		c[(x + i) * ldm_c + y] += __low2half(sum) + __high2half(sum);
	}
}

template <unsigned K = 16, bool sync_before_storing = false>
__device__ inline void matmul_core16x16(double* const c, const unsigned ldm_c, const double* const a, const unsigned ldm_a, const double* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	double tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		double sum = 0.0f;
		for (unsigned k = 0; k < K; k += 1) {
			sum = fma(a[y + ldm_a * k], b[x * ldm_b + k], sum);
		}
		tmp_c[i / 2] = sum;
	}

	if (sync_before_storing) {
		__syncthreads();
	}

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <unsigned K = 16, bool sync_before_storing = false>
__device__ inline void matmul_core16x16(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		float sum = 0.0f;
		for (unsigned k = 0; k < K; k += 1) {
			sum = fmaf(a[y + ldm_a * k], b[x * ldm_b + k], sum);
		}
		tmp_c[i / 2] = sum;
	}

	if (sync_before_storing) {
		__syncthreads();
	}

	for (auto i = 0; i < 16; i += 2) {
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <unsigned K = 16, bool sync_before_storing = false>
__device__ inline void matmul_core16x16(half* const c, const unsigned ldm_c, const half* const a, const unsigned ldm_a, const half* const b, const unsigned ldm_b, const unsigned unique_id) {
	const auto y = unique_id & 0xf;
	const auto x = (unique_id >> 4) << 3;
	unsigned i, k;
	half2 sums[8];

#pragma unroll
	for (unsigned i = 0; i < 8; i++)
		sums[i] = __float2half2_rn(0.0);

#pragma unroll
	for (k = 0; k < K; k += 2) {
		const auto a2 = __halves2half2(a[k * ldm_a + y], a[(k + 1) * ldm_a + y]);

		const half2 *b2 = (half2*)(b + x * ldm_b + k);
		for (i = 0; i < 8; i++) {
			sums[i] = __hfma2(a2, *(b2), sums[i]);
			b2 += ldm_b / 2;
		}
	}

	if (sync_before_storing) {
		__syncthreads();
	}

	for (i = 0; i < 8; i++) {
		const auto sum = sums[i];
		c[(x + i) * ldm_c + y] = __low2half(sum) + __high2half(sum);
	}
}

template <unsigned K = 16>
__device__ inline void gemv_core16x16(float* const c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned unique_id) {
	const unsigned lane = unique_id >> 4;
	const unsigned m = unique_id & 0xf;
	const unsigned n = lane * K / 2;

	float sum = 0;
	for (unsigned i = 0; i < K / 2; i++) {
		sum += a[m + (n + i) * ldm_a] * b[n + i];
	}

	sum += __shfl_xor_sync(0xffffffff, sum, 16);

	if(lane == 0) {
		c[m] += sum;
	}
}

template <unsigned K = 16>
__device__ inline void gemv_core16x16(half* const c, const half* const a, const unsigned ldm_a, const half* const b, const unsigned unique_id) {
	const unsigned lane = unique_id >> 4;
	const unsigned m = unique_id & 0xf;
	const unsigned n = lane * K / 2;

	half2 sum = __float2half2_rn(0.0f);
	for (unsigned i = 0; i < (K / 2) / 2; i++) {
		half2 a2;
		a2.x = a[m + (n + 2 * i + 0) * ldm_a];
		a2.y = a[m + (n + 2 * i + 1) * ldm_a];
		const half2 b2 = *reinterpret_cast<const half2*>(b + n + i * 2);
		sum = __hfma2(a2, b2, sum);
	}

	half s = sum.x + sum.y;

	s += __shfl_xor_sync(0xffffffff, s, 16);

	if(lane == 0) {
		c[m] += s;
	}
}

template <unsigned K = 16>
__device__ inline void gevm_core16x16(float* const c, const float* const a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	const unsigned lane = unique_id >> 4;
	const unsigned m = lane * (K / 2);
	const unsigned n = unique_id & 0xf;

	float sum = 0;
	for (unsigned i = 0; i < K / 2; i++) {
		sum += a[m + i] * b[n * ldm_b + m + i];
	}

	sum += __shfl_xor_sync(0xffffffff, sum, 16);

	if(lane == 0) {
		c[n] += sum;
	}
}

template <unsigned K = 16>
__device__ inline void gevm_core16x16(half* const c, const half* const a, const half* const b, const unsigned ldm_b, const unsigned unique_id) {
	const unsigned lane = unique_id >> 4;
	const unsigned m = lane * (K / 2);
	const unsigned n = unique_id & 0xf;

	half2 sum = __float2half2_rn(0.0f);
	for (unsigned i = 0; i < (K / 2) / 2; i++) {
		const half2 a2 = *reinterpret_cast<const half2*>(a + m + i * 2);
		const half2 b2 = *reinterpret_cast<const half2*>(b + n * ldm_b + m + 2 * i);
		sum = __hfma2(a2, b2, sum);
	}

	half s = sum.x + sum.y;

	s += __shfl_xor_sync(0xffffffff, s, 16);

	if(lane == 0) {
		c[n] += s;
	}
}

template <unsigned K = 16>
__device__ inline void ger_core16x16(float* const c, const unsigned ldm_c, const float* const a, const float* const b, const unsigned unique_id) {
	const unsigned lane = unique_id >> 4;
	const unsigned m = unique_id & 0xf;
	const unsigned n = lane * (K / 2);

	const auto a1 = a[m];
	for (unsigned i = 0; i < (K / 2); i++) {
		const auto b1 = b[n + i];
		const auto c1 = a1 * b1;
		c[ldm_c * (n + i) + m] += c1;
	}
}

template <unsigned K = 16>
__device__ inline void ger_core16x16(half* const c, const unsigned ldm_c, const half* const a, const half* const b, const unsigned unique_id) {
	const unsigned lane = unique_id >> 4;
	const unsigned m = unique_id & 0xf;
	const unsigned n = lane * (K / 2);

	const auto a2 = __halves2half2(a[m], a[m]);

	for (unsigned i = 0; i < (K / 2) / 2; i++) {
		const auto b2 = *reinterpret_cast<const half2*>(b + n + i * 2);
		const auto c2 = __hmul2(a2, b2);
		c[ldm_c * (n + i * 2 + 0) + m] += c2.x;
		c[ldm_c * (n + i * 2 + 1) + m] += c2.y;
	}
}
} // namespace gemm_core
} // namespace mtk

#endif /* end of include guard */

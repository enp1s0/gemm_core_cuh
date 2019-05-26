#ifndef __GEMM_CORE_CUH__
#define __GEMM_CORE_CUH__
#include <cuda_fp16.h>

template<class T, std::size_t num_warps> 
__device__ void gemm_core16x16(T* const c, const T* const a, const T* const b, const unsigned ldm, const unsigned unique_id);

template<>
__device__ void gemm_core16x16<float, 1lu>(float* const c, const float* const a, const float* const b, const unsigned ldm, const unsigned unique_id){
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	for(auto i = 0; i < 16; i+= 2){
		const auto x = i + lane;

		float sum = 0.0f;
		for(unsigned k = 0; k < 16; k += 4){
			const float4 b4 = *reinterpret_cast<const float4*>(b  + x * ldm + k);
			sum += a[(k + 0) * ldm + y] * b4.x;
			sum += a[(k + 1) * ldm + y] * b4.y;
			sum += a[(k + 2) * ldm + y] * b4.z;
			sum += a[(k + 3) * ldm + y] * b4.w;
		}
		c[x * ldm + y] += sum;
	}
}

template<>
__device__ void gemm_core16x16<half, 1lu>(half* const c, const half* const a, const half* const b, const unsigned ldm, const unsigned unique_id){
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	for(auto i = 0; i < 16; i+= 2){
		const auto x = i + lane;
		const half2 *b2_ptr = reinterpret_cast<const half2*>(b + x * ldm);

		half2 sum = __float2half2_rn(0.0f);
		for(unsigned k = 0; k < 16; k += 2){
			const auto b2 = b2_ptr[(k >> 1)];
			const auto a2 = __halves2half2(a[k * ldm + y], a[(k+1) * ldm + y]);
			sum = __hfma2(a2, b2, sum);
		}
		c[x * ldm + y] += __high2half(sum) + __low2half(sum);
	}
}
#endif /* end of include guard */

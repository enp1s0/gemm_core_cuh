#ifndef __GEMM_CORE_CUH__
#define __GEMM_CORE_CUH__
#include <cuda_fp16.h>

template<class T, std::size_t num_warps> 
__device__ void gemm_core16x16(T* const c, const T* const a, const T* const b, const unsigned ldm, const unsigned unique_id) {}

template<>
__device__ void gemm_core16x16<float, 1lu>(float* const c, const float* const a, const float* const b, const unsigned ldm, const unsigned unique_id){
	const auto lane = unique_id >> 4;
	for(auto i = 0; i < 16; i+= 2){
		const auto x = i + lane;
		const auto y = unique_id & 0xf;

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
#endif /* end of include guard */

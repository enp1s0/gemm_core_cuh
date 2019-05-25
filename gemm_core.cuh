#ifndef __GEMM_CORE_CUH__
#define __GEMM_CORE_CUH__
#include <cuda_fp16.h>

template<class T, std::size_t num_warps> 
__device__ void gemm_core16x16(T* const c, const T* const a, const T* const b, const unsigned ldm, const unsigned unique_id) {}

template<>
__device__ void gemm_core16x16<float, 1lu>(float* const c, const float* const a, const float* const b, const unsigned ldm, const unsigned unique_id){
	const unsigned x = unique_id >> 1;
	const unsigned y_start = (unique_id & 0x1) << 3;
#pragma unroll
	for(unsigned p = 0; p < 8; p++){
		const auto y = y_start + p;
		float sum = 0.0f;
#pragma unroll
		for(unsigned i = 0; i < 4; i++){
			const float4 b4 = *reinterpret_cast<const float4*>(b  + x * ldm + i * 4);
			sum += b4.x * a[y + (i * 4 + 0) * ldm];
			sum += b4.y * a[y + (i * 4 + 1) * ldm];
			sum += b4.z * a[y + (i * 4 + 2) * ldm];
			sum += b4.w * a[y + (i * 4 + 3) * ldm];
		}
		c[y + x * ldm] += sum;
	}
}

template<class T, std::size_t num_warps> 
__device__ void gemm_tn_core16x16(T* const c, const T* const a, const T* const b, const unsigned ldm, const unsigned unique_id) {}

template<>
__device__ void gemm_tn_core16x16<float, 1lu>(float* const c, const float* const a, const float* const b, const unsigned ldm, const unsigned unique_id){
	const unsigned x = unique_id >> 1;
	const unsigned y_start = (unique_id & 0x1) << 3;
#pragma unroll
	for(unsigned p = 0; p < 8; p++){
		const auto y = y_start + p;
		float sum = 0.0f;
#pragma unroll
		for(unsigned i = 0; i < 4; i++){
			const float4 b4 = *reinterpret_cast<const float4*>(b  + x * ldm + i * 4);
			const float4 a4 = *reinterpret_cast<const float4*>(a  + y * ldm + i * 4);
			sum += b4.x * a4.x;
			sum += b4.y * a4.y;
			sum += b4.z * a4.z;
			sum += b4.w * a4.w;
			c[y + x * ldm] += sum;
		}
	}
}

#endif /* end of include guard */

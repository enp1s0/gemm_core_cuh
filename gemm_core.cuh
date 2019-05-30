#ifndef __GEMM_CORE_CUH__
#define __GEMM_CORE_CUH__
#include <cuda_fp16.h>

template<class T, std::size_t num_warps> 
__device__ void gemm_core16x16(T* const c, const T* const a, const T* const b, const unsigned ldm, const unsigned unique_id);

template<>
__device__ void gemm_core16x16<float, 1lu>(float* const c, const float* const a, const float* const b, const unsigned ldm, const unsigned unique_id){
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	// unrollするとレジスタを1つ多く確保する
	// 実行すると遅い
//#pragma unroll
	for(auto i = 0; i < 16; i+= 2){
		const auto x = i + lane;

		// はじめにcを読んでおくと誤差が小さくなる
		// cuBLASの内部と計算順序が合うのかも
		float sum = c[x * ldm + y];
		for(unsigned k = 0; k < 16; k+=1){
			sum = fmaf(a[y + ldm * k], b[x * ldm + k], sum);
		}
		// float4による128bitアクセスはSharedメモリに対しては意味がないみたい
		/*for(unsigned k = 0; k < 16; k += 4){
			const float4 b4 = *reinterpret_cast<const float4*>(b + x * ldm + k);
			sum = fmaf(a[(k + 0) * ldm + y], b4.x, sum);
			sum = fmaf(a[(k + 1) * ldm + y], b4.y, sum);
			sum = fmaf(a[(k + 2) * ldm + y], b4.z, sum);
			sum = fmaf(a[(k + 3) * ldm + y], b4.w, sum);
		}*/
		c[x * ldm + y] = sum;
	}
}

template<>
__device__ void gemm_core16x16<half, 1lu>(half* const c, const half* const a, const half* const b, const unsigned ldm, const unsigned unique_id){
	const auto y = unique_id & 0xf;
	const auto x = (unique_id >> 4) << 3;
	unsigned i, k;
	half2 sums[8];

#pragma unroll
	for(unsigned i = 0; i < 8; i++)
		sums[i] = __float2half2_rn(0.0);

#pragma unroll
	for(k = 0; k < 16; k+= 2){
		const auto a2 = __halves2half2(a[k * ldm + y], a[(k + 1) * ldm + y]);

		const half2 *b2 = (half2*)(b + x * ldm + k);
		for(i = 0; i < 8; i++){
			sums[i] = __hfma2(a2, *(b2), sums[i]);
			b2 += ldm / 2;
		}
	}
	for(i = 0; i < 8; i++){
		const auto sum = sums[i];
		c[(x + i) * ldm + y] = __low2half(sum) + __high2half(sum) + (c[(x + i) * ldm + y]);
	}
}
#endif /* end of include guard */

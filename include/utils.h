#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cassert>

#define half __half
#define WARP_SIZE 32
#define NUMWARPPERGROUP 4
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ __forceinline__ void mma16816(float* d, const uint32_t* A, const uint32_t* B) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%0, %1, %2, %3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1])
  );
}

__device__ __forceinline__ void ldmatrix_x2_T(uint32_t* A, const void* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
  asm volatile(
    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(A[0]), "=r"(A[1])
    : "r"(addr)
  );
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t* A, const void* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
  asm volatile(
    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(A[0]), "=r"(A[1])
    : "r"(addr)
  );
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* A, const void* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "r"(addr));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ static __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
  asm volatile (
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(bar_ptr), "r"(thread_count+transaction_count)
  );
}

__device__ static __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
  asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(bar_ptr), "r"(bytes));
}

__device__ static inline void load_async(half *dst, void const* src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
  uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));;

  asm volatile (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5}], [%2];"
    :
    : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
      "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
    : "memory"
  );
}

__device__ static inline void store_async(void const* dst_tma_map, half *src, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile (
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :
        : "l"(tma_ptr), "r"(src_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64)
        : "memory"
    );
}

__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    // Call mbarrier.try_wait in a while loop till it returns true.
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}


typedef struct mykernelParamType
{
    half*  __restrict__ Q; 
    half*  __restrict__ K;
    half*  __restrict__ V;
    half*  __restrict__ O;  
    float* __restrict__ L;      
    float* __restrict__ M;
    int      B;
    int      N;
    int      S;
    int      d;
    int      Tr;
    int      Tc;
    float    softmax_scale;
    float    dropout_prob;
    unsigned long long seed;
    curandStatePhilox4_32_10_t* states; 
    int      window_size_left;
    int      window_size_right;
    float*   alibi_slopes_ptr;
    int      split_num;
}mykernelParamType;


template <int BlockMajorSize, int BlockMinorSize, bool swizzle=false, bool padding=false>
__host__ static inline CUtensorMap create_tensor_map(half* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    static_assert(BlockMinorSize >= 64);
    assert(global_width % 64 == 0);
    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width/64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(half) * global_width, 64*sizeof(half), 0, 0, 0};
    uint32_t smem_box_shape[5] = {padding ? 72 : 64, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize/64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, gmem_address, gmem_prob_shape,
        gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}


// 用于打包数据
__device__ __forceinline__ uint32_t pack_2float_to_1uint32(float num1, float num2) {
    half a = __float2half(num1);
    half b = __float2half(num2);

    uint16_t a_bits = __half_as_ushort(a);
    uint16_t b_bits = __half_as_ushort(b);

    return (static_cast<uint32_t>(b_bits) << 16u) | a_bits;
}
#endif // UTILS_H


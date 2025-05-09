#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <curand_kernel.h>
#include <cuda_fp16.h>


#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
// MMA
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "+f"(RD0), "+f"(RD1), "+f"(RD2), "+f"(RD3) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))
// LDMATRIX
#define LDMATRIX_X2(R0, R1, addr) asm volatile( "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"  : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst), "l"(src), "n"(bytes))

typedef struct mykernelParamType
{
    half*  __restrict__ Q; 
    half*  __restrict__ K;
    half*  __restrict__ V;
    float* __restrict__ O;
    float* __restrict__ O_tmp;  
    float* __restrict__ L;      
    float* __restrict__ M; 
    int      N;
    int      d;
    int      Br;
    int      Bc;
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

// 用于打包数据
__device__ inline uint32_t pack_float_to_uint32(float num1, float num2) {
    half a = __float2half(num1);
    half b = __float2half(num2);

    uint16_t a_bits = __half_as_ushort(a);
    uint16_t b_bits = __half_as_ushort(b);

    return (static_cast<uint32_t>(b_bits) << 16u) | a_bits;
}

// 用于从全局内存加载Q, K, V
__device__ inline void load_data(half* dst, half* src, int dst_row_length, int src_row_length, int count, int Br, int bx, int bdx, int tx, int seq_len, bool Is_even_MN, bool Is_even_K){
    if (Is_even_MN && Is_even_K){
        int offset = tx * 8;
        #pragma unroll
        for (int i = 0; i < count / 8; i++){
            CP_ASYNC_CG(__cvta_generic_to_shared(&dst[offset]), &src[offset], 16);
            offset += bdx * 8;
        }
    }
    else if (Is_even_K && !Is_even_MN){
        int offset = tx * 8;
        #pragma unroll
        for (int i = 0; i < count / 8; i++){
            int row = offset / dst_row_length;
            if(row + Br * bx < seq_len){
                CP_ASYNC_CG(__cvta_generic_to_shared(&dst[offset]), &src[offset], 16);
            }
            else{
                #pragma unroll
                for(int i = 0; i < 8; i++){
                    dst[offset+i] = 0.0f;
                }
            }
            offset += bdx * 8;
        }
        
    }
    else if (Is_even_MN && !Is_even_K){
        int offset = tx;
        #pragma unroll
        for(int i = 0; i < count; i++){
            int row = offset / dst_row_length;
            int col = offset % dst_row_length;
            if(col < src_row_length){
                dst[offset] = src[row * src_row_length + col];
            }
            else{
                dst[offset] = 0.0f;
            }
            offset += bdx;
        }
    }
    else{
        int offset = tx;
        #pragma unroll
        for(int i = 0; i < count; i++){
            int row = offset / dst_row_length;
            int col = offset % dst_row_length;
            if(row + Br * bx < seq_len && col < src_row_length){
                dst[offset] = src[row * src_row_length + col];
            }
            else{
                dst[offset] = 0.0f;
            }
            offset += bdx;
        }
    }
}   

// 用于把O矩阵从共享内存加载到全局内存
__device__ inline void load_data(float* dst, float* src, int dst_row_length, int src_row_length, int count, int Br, int bx, int bdx, int tx, int seq_len, bool Is_even_MN, bool Is_even_K){
    if (Is_even_MN && Is_even_K){
        int offset = tx * 4;
        #pragma unroll
        for (int i = 0; i < count / 4; i++){
            LDST128BITS(dst[offset]) = LDST128BITS(src[offset]);
            offset += bdx * 4;
        }
    }
    else if (Is_even_K && !Is_even_MN){
        int offset = tx * 4;
        #pragma unroll
        for (int i = 0; i < count / 4; i++){
            int row = offset / src_row_length;
            if(row + Br * bx < seq_len){
                LDST128BITS(dst[offset]) = LDST128BITS(src[offset]);
            }
            offset += bdx * 4;
        }
    }
    else if (Is_even_MN && !Is_even_K){
        int offset = tx;
        #pragma unroll
        for(int i = 0; i < count; i++){
            int row = offset / src_row_length;
            int col = offset % dst_row_length;
            if(col < dst_row_length){
                dst[row * dst_row_length + col] = src[row * src_row_length + col];
            }
            offset += bdx;
        }
    }
    else{
        int offset = tx;
        #pragma unroll
        for(int i = 0; i < count; i++){
            int row = offset / src_row_length;
            int col = offset % src_row_length;
            if(row + Br * bx < seq_len && col < dst_row_length){
                dst[row * dst_row_length + col] = src[row * src_row_length + col];
            }
            offset += bdx;
        }
    }
}
#endif // UTILS_H

#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <cstdint>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#include "utils.h"

template<int Num_consumers, int Br, int Bc>
__device__ __forceinline__ void apply_softmax(
    float data[Br/Num_consumers/NUMWARPPERGROUP/MMA_M][Bc/MMA_N][4], 
    float row_m_prev[Br/Num_consumers/NUMWARPPERGROUP/MMA_M][2], 
    float row_m[Br/Num_consumers/NUMWARPPERGROUP/MMA_M][2], 
    float row_l[Br/Num_consumers/NUMWARPPERGROUP/MMA_M][2], 
    float row_l_prev[Br/Num_consumers/NUMWARPPERGROUP/MMA_M][2]
) {
    // 线程中规约出最大值
    #pragma unroll
    for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++){
        #pragma unroll
        for (int j = 0; j < Bc / MMA_N; j++) {
            row_m[i][0] = fmaxf(row_m[i][0], data[i][j][0]);
            row_m[i][0] = fmaxf(row_m[i][0], data[i][j][1]);

            row_m[i][1] = fmaxf(row_m[i][1], data[i][j][2]);
            row_m[i][1] = fmaxf(row_m[i][1], data[i][j][3]);
        }
    }

    // 线程束中规约出最大值
    #pragma unroll
    for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
        #pragma unroll
        for (int j = 3; j >= 1; j /= 2){
            float row_m_other;
            row_m_other = __shfl_xor_sync(0xffffffff, row_m[i][0], j, 32);
            row_m[i][0] = fmaxf(row_m[i][0], row_m_other);
            row_m_other = __shfl_xor_sync(0xffffffff, row_m[i][1], j, 32);
            row_m[i][1] = fmaxf(row_m[i][1], row_m_other);
        }
    }

    // 更新最大值
    #pragma unroll
    for (int i = 0; i < Br/Num_consumers/NUMWARPPERGROUP/MMA_M; i++) {
        row_m[i][0] = fmaxf(row_m[i][0], row_m_prev[i][0]);
        row_m[i][1] = fmaxf(row_m[i][1], row_m_prev[i][1]);
    }

    // 计算每个线程每一行的和
    #pragma unroll
    for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < Bc / MMA_N; j++){
            data[i][j][0] = __expf(data[i][j][0] - row_m[i][0]);        
            row_l[i][0] += data[i][j][0];
            data[i][j][1] = __expf(data[i][j][1] - row_m[i][0]);        
            row_l[i][0] += data[i][j][1];

            data[i][j][2] = __expf(data[i][j][2] - row_m[i][1]);        
            row_l[i][1] += data[i][j][2];
            data[i][j][3] = __expf(data[i][j][3] - row_m[i][1]);        
            row_l[i][1] += data[i][j][3];
        }
    }

    // 规约求和
    #pragma unroll
    for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
        #pragma unroll
        for (int j = 3; j >= 1; j /= 2){
            float row_l_other;
            row_l_other = __shfl_xor_sync(0xffffffff, row_l[i][0], j, 32);
            row_l[i][0] += row_l_other;
            row_l_other = __shfl_xor_sync(0xffffffff, row_l[i][1], j, 32);
            row_l[i][1] += row_l_other;
        }
    }
    
    // 更新和
    #pragma unroll
    for (int i = 0; i < Br/Num_consumers/NUMWARPPERGROUP/MMA_M; i++) {
        row_l[i][0] += __expf(row_m_prev[i][0] - row_m[i][0]) * row_l_prev[i][0];
        row_l[i][1] += __expf(row_m_prev[i][1] - row_m[i][1]) * row_l_prev[i][1];
    }
}
#endif // SOFTMAX_H
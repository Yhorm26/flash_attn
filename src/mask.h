#ifndef MASK_H
#define MASK_H
#include <cstdint>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#include "utils.h"


template<bool Is_causal, bool Is_local, bool Has_alibi, int Num_consumers, int Br, int Bc>
__device__ __forceinline__ void apply_mask(float data[Br/Num_consumers/NUMWARPPERGROUP/MMA_M][Bc/MMA_N][4], int row, int col, int tid, int window_size_left, float alibi_slope) {
    row += tid % WARP_SIZE / 4;
    col += tid % WARP_SIZE % 4 * 2;
   
    if constexpr (Is_causal) {
        #pragma unroll
        for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
            #pragma unroll
            for (int j = 0; j < Bc / MMA_N; j++) {
                int diff = row + i * MMA_M - (col + j * MMA_N);
                if constexpr (Has_alibi) {
                    data[i][j][0] -= alibi_slope * abs(diff);
                    data[i][j][1] -= alibi_slope * abs(diff - 1);
                    data[i][j][2] -= alibi_slope * abs(diff + 8);
                    data[i][j][3] -= alibi_slope * abs(diff + 7);
                }
                if (diff < 0){
                    data[i][j][0] = -INFINITY;
                }
                if (diff < 1){
                    data[i][j][1] = -INFINITY;
                }
                if (diff + 8 < 0){
                    data[i][j][2] = -INFINITY;
                }
                if (diff + 8 < 1){
                    data[i][j][3] = -INFINITY;
                }
            }
        }
    }

    if constexpr (Is_local) {
        #pragma unroll
        for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
            #pragma unroll
            for (int j = 0; j < Bc / MMA_N; j++) {
                int diff = row + i * MMA_M - (col + j * MMA_N);
                if constexpr (Has_alibi) {
                    data[i][j][0] -= alibi_slope * abs(diff);
                    data[i][j][1] -= alibi_slope * abs(diff - 1);
                    data[i][j][2] -= alibi_slope * abs(diff + 8);
                    data[i][j][3] -= alibi_slope * abs(diff + 7);
                }
                if (diff > window_size_left || diff < -window_size_left){
                    data[i][j][0] = -INFINITY;
                }
                if (diff - 1 > window_size_left || diff - 1 < -window_size_left){
                    data[i][j][1] = -INFINITY;
                }
                if (diff + 8 > window_size_left || diff + 8 < -window_size_left){
                    data[i][j][2] = -INFINITY;
                }
                if (diff + 7 > window_size_left || diff + 7 < -window_size_left){
                    data[i][j][3] = -INFINITY;
                }
            }
        }
    }

    if constexpr (Has_alibi && !Is_causal && !Is_local) {
        #pragma unroll
        for (int i = 0; i < Br / Num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
            #pragma unroll
            for (int j = 0; j < Bc / MMA_N; j++) {
                int diff = row + i * MMA_M - (col + j * MMA_N);
                data[i][j][0] -= alibi_slope * abs(diff);
                data[i][j][1] -= alibi_slope * abs(diff - 1);
                data[i][j][2] -= alibi_slope * abs(diff + 8);
                data[i][j][3] -= alibi_slope * abs(diff + 7);
            }
        }
    }
}
#endif // MASK_H
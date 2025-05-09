#ifndef FLASH_FORWARD_SPLITKV_H
#define FLASH_FORWARD_SPLITKV_H
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <curand_kernel.h>
#include <cstdint>
#include "utils.h"
# include <cmath>

using namespace nvcuda;

template<bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K>
__global__
void forward_kernel_splitkv(mykernelParamType param){
    const int tx = threadIdx.x;
    const int warp_id = tx / 32; const int lane_id = tx % 32;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    int align_d = (param.d + 31) / 32 * 32;
    extern __shared__ char shared_mem[];
    half*  Qj = reinterpret_cast<half*>(shared_mem);
    half*  Kj = reinterpret_cast<half*>(shared_mem + param.Br * align_d * sizeof(half));
    half*  Vj = reinterpret_cast<half*>(shared_mem + (param.Br + param.Bc) * align_d * sizeof(half));
    float* Oj = reinterpret_cast<float*>(shared_mem + (param.Br + param.Bc * 2) * align_d * sizeof(half));

    float row_l_prev1 = 0;
    float row_l_prev2 = 0;
    float row_m_prev1 = -INFINITY;
    float row_m_prev2 = -INFINITY;
    const int tile_size = param.Bc * param.d;
    const float alibi_slope = !Has_alibi ? 0.0f : param.alibi_slopes_ptr[by];

    uint32_t a_frag[4];
    uint32_t b_frag[4];
    float    c_frag[8][8];
    uint32_t d_frag[8][4];

    int total_kv_blocks = (param.N + param.Bc - 1) / param.Bc; // 总K/V块数
    int blocks_per_split = (total_kv_blocks + param.split_num - 1) / param.split_num;
    int n_block_min = blocks_per_split * (bx % param.split_num);
    int n_block_max = min(n_block_min + blocks_per_split, total_kv_blocks);
    n_block_min = !Is_local? n_block_min : max(n_block_min, (((bx / param.split_num) * param.Br - param.window_size_left) / param.Bc));

    if (Is_local){
        n_block_max = min(n_block_max, (((bx / param.split_num) + 1) * param.Br + param.window_size_right + param.Bc - 1) / param.Bc);
    }

    if (Is_causal){
        n_block_max = min(n_block_max, (((bx / param.split_num) + 1) * param.Br + param.Bc - 1) / param.Bc);
    }

    curandStatePhilox4_32_10_t local_state;
    if (Is_dropout) {
        const int state_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
        local_state = param.states[state_idx];
    }

    // 计算K,V的全局内存地址
    int kv_offset = (bz * gridDim.y + by)* param.N * param.d + param.d * param.Bc * n_block_min;
    // // 计算Q的全局内存地址
    int q_offset = (bz * gridDim.y + by)* param.N * param.d + param.d * param.Br * (bx / param.split_num);
    int o_offset;
    if (Is_even_MN || bx / param.split_num != param.Tr - 1){
        o_offset = (bz * gridDim.y + by) * param.N * param.d * param.split_num + param.d * param.Br * bx;
    }
    else{
        o_offset = (bz * gridDim.y + by) * param.N * param.d * param.split_num + param.d * param.Br * bx \
                    - (param.Tr * param.Br - param.N) * (bx % param.split_num) * param.d;
    }
    
    half*  Q = param.Q + q_offset;
    half*  K = param.K + kv_offset;
    half*  V = param.V + kv_offset;
    float* O = param.O_tmp + o_offset;

    // 初始化结果
    #pragma unroll
    for (int i = tx; i < param.Br * align_d; i += blockDim.x) {
        Oj[i] = 0.0f;
    }
    __syncthreads();

    const int load_QO_num = param.Br * align_d / blockDim.x;   // 每个线程需要搬运多少个Q矩阵的元素
    const int load_KV_num = param.Bc * align_d / blockDim.x;   // 每个线程需要搬运多少个K, V矩阵的元素

    if (n_block_min < n_block_max){
        // 把Q从全局内存加载到共享内存
        load_data(Qj, Q, align_d, param.d, load_QO_num, param.Br, bx / param.split_num, blockDim.x, tx, param.N, Is_even_MN, Is_even_K);
        // 把K的第一块从全局内存加载到共享内存
        load_data(Kj, K, align_d, param.d, load_KV_num, param.Bc, n_block_min, blockDim.x, tx, param.N, Is_even_MN, Is_even_K);
        if(Is_even_K){
            CP_ASYNC_COMMIT_GROUP();
        }
    }
    
    for (int iter = n_block_min; iter < n_block_max; iter++){
        // 确保Q,K均加载到共享内存
        if(Is_even_MN && Is_even_K){
            CP_ASYNC_WAIT_ALL();
        }
        else{
            __syncthreads();
        }
        // 异步加载V到共享内存
        load_data(Vj, V, align_d, param.d, load_KV_num, param.Bc, iter, blockDim.x, tx, param.N, Is_even_MN, Is_even_K);

        // K, V指针指向下一块K, V要读取的数据的地址
        K += tile_size;
        V += tile_size;

        memset(c_frag, 0, sizeof(c_frag));
        __syncthreads();
        // S = Q * K.T
        #pragma unroll
        for (int x = 0; x < align_d / 16; x++){
            uint32_t aOffsetPtr = __cvta_generic_to_shared(&Qj[(warp_id*16+lane_id%16)*align_d+x*16+(lane_id/16)*8]);
            LDMATRIX_X4(a_frag[0], a_frag[1], a_frag[2], a_frag[3], aOffsetPtr);
            #pragma unroll
            for (int y = 0; y < param.Bc / 16; y++){
                uint32_t bOffsetPtr = __cvta_generic_to_shared(&Kj[(y*16+lane_id%16)*align_d+x*16+(lane_id/16)*8]);
                LDMATRIX_X4(b_frag[0], b_frag[2], b_frag[1], b_frag[3], bOffsetPtr);

                HMMA16816F32(c_frag[y][0], c_frag[y][1], c_frag[y][4], c_frag[y][5], a_frag[0], a_frag[1], a_frag[2], a_frag[3], \
                             b_frag[0], b_frag[1], c_frag[y][0], c_frag[y][1], c_frag[y][4], c_frag[y][5]);

                HMMA16816F32(c_frag[y][2], c_frag[y][3], c_frag[y][6], c_frag[y][7], a_frag[0], a_frag[1], a_frag[2], a_frag[3], \
                             b_frag[2], b_frag[3], c_frag[y][2], c_frag[y][3], c_frag[y][6], c_frag[y][7]);
            }
        }

        // causal mask 处理
        if (Is_causal && iter == bx / param.split_num){
            #pragma unroll
            for(int i = warp_id + 1; i < 8; i++){
                #pragma unroll
                for (int j = 0; j < 8; j++){
                    c_frag[i][j] = -INFINITY;
                }
            }
            
            #pragma unroll
            for (int j = 0; j < 8; j++){
                int row = lane_id / 4 + (j / 4) * 8;
                int col = (lane_id % 4) * 2 + (j % 4) / 2 * 8 + j % 2;
                if(col > row){
                    c_frag[warp_id][j] = -INFINITY;
                }
            }       
        }

        // window attention处理
        if (Is_local && ((bx / param.split_num + 1) * param.Br - 1 - ((iter - 1) * param.Bc + 1) > param.window_size_left || \
                         (iter + 1) * param.Bc - 1 - ((bx / param.split_num - 1) * param.Br + 1) > param.window_size_right)){
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++){
                    int row = (bx / param.split_num) * param.Br + warp_id * 16 + lane_id / 4 + (j / 4) * 8;
                    int col = iter * param.Bc + i * 16 + (lane_id % 4) * 2 + (j % 4) / 2 * 8 + j % 2;
                    if(row - col > param.window_size_left || col - row > param.window_size_right){
                        c_frag[i][j] = -INFINITY;
                    }
                }
            }
        }

        // 确保V已经加载到共享内存, 同时也避免有些线程束没有计算完成,而后续读取K会导致结果错误
        __syncthreads();
        CP_ASYNC_WAIT_ALL();
        // 异步加载下一次迭代需要的K到共享内存
        if (iter < n_block_max - 1){
            load_data(Kj, K, align_d, param.d, load_KV_num, param.Bc, iter + 1, blockDim.x, tx, param.N, Is_even_MN, Is_even_K);
        }

        // 先计算alibi(可选),然后求每个线程每行的最大值
        float row_m1 = -INFINITY;
        float row_m2 = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 8; i++){
            #pragma unroll
            for (int j = 0; j < 4; j++){
                c_frag[i][j]     *= param.softmax_scale;
                c_frag[i][j + 4] *= param.softmax_scale;

                if (Has_alibi){
                    int row = (bx / param.split_num) * param.Br + warp_id * 16 + lane_id / 4;
                    int col = iter * param.Bc + i * 16 + (lane_id % 4) * 2 + j / 2 * 8 + j % 2;
                    c_frag[i][j]     -= alibi_slope * abs(row - col);
                    c_frag[i][j + 4] -= alibi_slope * abs(row + 8 - col);
                }

                if (c_frag[i][j]     > row_m1)    row_m1 = c_frag[i][j]    ;
                if (c_frag[i][j + 4] > row_m2)    row_m2 = c_frag[i][j + 4];
            }
        }

        // 规约出最大值
        #pragma unroll
        for (int i = 3; i >= 1; i /= 2){
            float row_m_other = __shfl_xor_sync(0xffffffff, row_m1, i, 32);
            row_m1 = fmaxf(row_m1, row_m_other);
            row_m_other = __shfl_xor_sync(0xffffffff, row_m2, i, 32);
            row_m2 = fmaxf(row_m2, row_m_other);
        }
        
        // 计算每个线程每一行的和
        float row_l1 = 0;
        float row_l2 = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++){
                c_frag[i][j] = __expf(c_frag[i][j] - row_m1);        
                row_l1 += c_frag[i][j];
                c_frag[i][j + 4] = __expf(c_frag[i][j + 4] - row_m2);    
                row_l2 += c_frag[i][j + 4];
            }
        }

        // 规约求和
        #pragma unroll
        for (int i = 3; i >= 1; i /= 2){
            float row_l_other;
            row_l_other = __shfl_xor_sync(0xffffffff, row_l1, i, 32);
            row_l1 += row_l_other;
            row_l_other = __shfl_xor_sync(0xffffffff, row_l2, i, 32);
            row_l2 += row_l_other;
        }

        // 计算最大值与和
        float row_m_new1 = fmaxf(row_m1, row_m_prev1);
        float row_m_new2 = fmaxf(row_m2, row_m_prev2);
        float row_l_new1 = (__expf(row_m_prev1 - row_m_new1) * row_l_prev1) + (__expf(row_m1 - row_m_new1) * row_l1);
        float row_l_new2 = (__expf(row_m_prev2 - row_m_new2) * row_l_prev2) + (__expf(row_m2 - row_m_new2) * row_l2);

        if (Is_dropout) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    // 生成随机数
                    float rand_val = curand_uniform(&local_state);
                    
                    // 应用mask
                    if (rand_val < param.dropout_prob) {
                        c_frag[i][j] = 0.0f;
                    } else {
                        c_frag[i][j] *= 1.0f / (1.0f - param.dropout_prob);
                    }
                }
            }
        }

        // 打包数据,为了使用tensor core
        #pragma unroll
        for (int i = 0; i < 8; i++){
            #pragma unroll
            for (int j = 0; j < 4; j++){
                d_frag[i][j] = pack_float_to_uint32(c_frag[i][2*j], c_frag[i][2*j+1]);
            }
        }

        // 计算O需要的系数
        float factor1 = (row_l_new1 != 0.0f) ? (1.0f / row_l_new1) : 0.0f;;
        float factor2 = row_l_prev1 * __expf(row_m_prev1 - row_m_new1);
        float factor3 = __expf(row_m1 - row_m_new1);

        float factor4 = (row_l_new2 != 0.0f) ? (1.0f / row_l_new2) : 0.0f;;
        float factor5 = row_l_prev2 * __expf(row_m_prev2 - row_m_new2);
        float factor6 = __expf(row_m2 - row_m_new2);

        // S = S * V
        #pragma unroll 
        for (int i = 0; i < align_d / 16; i++){
            memset(c_frag, 0, sizeof(c_frag));
            #pragma unroll
            for (int j = 0; j < param.Bc / 16; j++){
                uint32_t aOffsetPtr;
                #pragma unroll
                for (int x = 0; x < 2; x++){
                    aOffsetPtr = __cvta_generic_to_shared(&Vj[(j*16+lane_id%16)*align_d+i*16+x*8]);
                    LDMATRIX_X2_T(a_frag[x*2], a_frag[x*2+1], aOffsetPtr);

                    HMMA16816F32(c_frag[x][0], c_frag[x][1], c_frag[x][2], c_frag[x][3],  \
                                 d_frag[j][0], d_frag[j][2], d_frag[j][1], d_frag[j][3],  \
                                 a_frag[x*2], a_frag[x*2+1],   \
                                 c_frag[x][0], c_frag[x][1], c_frag[x][2], c_frag[x][3]);
                }
            }

            // 更新输出
            int O_row = warp_id * 16 + lane_id / 4;
            int O_col = i * 16 + (lane_id % 4) * 2;
            #pragma unroll
            for (int y = 0; y < 2; y++){
                #pragma unroll
                for(int x = 0; x < 2; x++){
                    Oj[(O_row  )*align_d+O_col+y*8+x] = factor1 * (factor2 * Oj[(O_row  )*align_d+O_col+y*8+x] + factor3 * c_frag[y][x  ]);
                    Oj[(O_row+8)*align_d+O_col+y*8+x] = factor4 * (factor5 * Oj[(O_row+8)*align_d+O_col+y*8+x] + factor6 * c_frag[y][x+2]);
                }
            }

            __syncthreads();
        }

        // 更新最大值与和
        row_l_prev1 = row_l_new1;
        row_l_prev2 = row_l_new2;
        row_m_prev1 = row_m_new1;
        row_m_prev2 = row_m_new2;
    }

    // 把Q从共享内存写到全局内存
    load_data(O, Oj, param.d, align_d, load_QO_num, param.Br, bx / param.split_num, blockDim.x, tx, param.N, Is_even_MN, Is_even_K);

    // 将每行的最大值以及和写回到全局内存
    if(lane_id % 4 == 0){
        if(Is_even_MN || bx / param.split_num < param.Tr - 1){
            int lm_offset = ((bz * gridDim.y + by) * param.N + (bx / param.split_num) * param.Br + warp_id * 16 + lane_id / 4) * param.split_num \
                             + bx % param.split_num;
            param.L[lm_offset] = row_l_prev1;
            param.M[lm_offset] = row_m_prev1;
            param.L[lm_offset+8*param.split_num] = row_l_prev2;
            param.M[lm_offset+8*param.split_num] = row_m_prev2;
        }
        else{
            int lm_offset = ((bz * gridDim.y + by) * param.N) * param.split_num + bx % param.split_num;
            int row = (bx / param.split_num) * param.Br + warp_id * 16 + lane_id / 4;
            if(row < param.N){
                param.L[lm_offset+row*param.split_num] = row_l_prev1;
                param.M[lm_offset+row*param.split_num] = row_m_prev1;
            }
            if(row + 8 < param.N){
                param.L[lm_offset+(row+8)*param.split_num] = row_l_prev2;
                param.M[lm_offset+(row+8)*param.split_num] = row_m_prev2;
            }
        }   
    }

    if (Is_dropout) {
        const int state_idx = bz * gridDim.y * gridDim.x + by * gridDim.x + bx;
        param.states[state_idx] = local_state;
    }
}


__global__
void forward_kernel_splitkv_combine(mykernelParamType param){
    const int tx = threadIdx.x;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    int O_tmp_offset = (bz * gridDim.y + by) * param.d * param.N * param.split_num + bx * param.Br * param.d * param.split_num + tx * param.d;
    int O_offset     = (bz * gridDim.y + by) * param.d * param.N + (bx * param.Br + tx) * param.d;
    int lm_offset    = ((bz * gridDim.y + by) * param.N + bx * param.Br + tx) * param.split_num;

    float* O_tmp = param.O_tmp + O_tmp_offset;
    float* O     = param.O + O_offset;

    float row_m = -INFINITY;
    #pragma unroll
    for(int i = 0; i < param.split_num; i++){
        row_m = fmaxf(row_m, param.M[lm_offset+i]);
    }

    float row_l = 0;
    #pragma unroll
    for(int i = 0; i < param.split_num; i++){
        row_l += __expf(param.M[lm_offset+i] - row_m) * param.L[lm_offset+i];
    }

    int row_spacing;
    if(param.N % param.Br == 0 || bx < param.Tr - 1){
        row_spacing = param.Br;
    }
    else{
        row_spacing = param.N - (param.Tr - 1) * param.Br;
    }
    
    if(bx < param.Tr - 1 || (bx == param.Tr - 1 && tx < param.N - (param.Tr - 1) * param.Br)){
        float O_result[32];
        #pragma unroll
        for(int i = 0; i < (param.d + 31) / 32; i++){
            memset(O_result, 0, sizeof(O_result));
            #pragma unroll
            for(int j = 0; j < param.split_num; j++){
                float factor = (row_l != 0.0f) ? (1.0f / row_l) : 0.0f;
                factor *= param.L[lm_offset+j] * __expf(param.M[lm_offset+j] - row_m);
                #pragma unroll
                for(int k = 0; k + 32 * i < param.d; k++){
                    O_result[k] += factor * O_tmp[j*row_spacing*param.d+i*32+k];
                }
            }
            #pragma unroll
            for(int k = 0; k + 32 * i < param.d; k++){
                O[i*32+k] = O_result[k];
            }
        }
    }
}


#endif
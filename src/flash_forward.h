#ifndef FLASH_FORWARD_H
#define FLASH_FORWARD_H
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <curand_kernel.h>
#include <cstdint>
#include "utils.h"
#include "mask.h"
#include "softmax.h"

using namespace nvcuda;

CUtensorMap d_tma_map_Q;
CUtensorMap d_tma_map_K;
CUtensorMap d_tma_map_V;
CUtensorMap d_tma_map_O;

template<int VERSION, int Br, int Bc, bool Is_causal, bool Is_local>
struct Schedule;

template<int Br, int Bc, bool Is_causal, bool Is_local>
struct Schedule<0, Br, Bc, Is_causal, Is_local> {
    int index;
    int num_blocks_per_head;
    int seq_len;
    int window_size;
    int num_iter;
    int num_sm;

    __device__ __forceinline__ Schedule(int num_iter, int num_sm, int blockIdx_x, int seq_len, int num_blocks_per_head, int window_size) {
        index = blockIdx_x;
        this->num_iter = num_iter;
        this->num_sm = num_sm;
        this->seq_len = seq_len;
        this->num_blocks_per_head = num_blocks_per_head;
        this->window_size = window_size;
    }

    __device__ __forceinline__ bool next(int &blockId, int &n_block_min, int &n_block_max, int &seq_id) {
        if (index >= num_iter) {
            return false;
        }
        blockId = index;
        index += num_sm;
        seq_id = blockId / num_blocks_per_head;
        int bx = blockId % num_blocks_per_head;

        n_block_min = !Is_local? 0 : max(0, ((bx * Br - window_size) / Bc));
        n_block_max = (seq_len + Bc - 1) / Bc;

        if constexpr (Is_local) {
            n_block_max = min(n_block_max, ((bx + 1) * Br + window_size + Bc - 1) / Bc);
        }

        if constexpr (Is_causal) {
            n_block_max = min(n_block_max, ((bx + 1) * Br + Bc - 1) / Bc);
        }

        return true;
    }
};

template <int Br, int Bc, int d, int QSIZE>
struct alignas(128) SMem {
    alignas(128) half Q[Br*d];
    alignas(128) half K[d*Bc*QSIZE];
    alignas(128) half V[d*Bc*QSIZE];
    alignas(128) half O[Br*d];
    alignas(8) uint64_t fullK[QSIZE], emptyK[QSIZE];
    alignas(8) uint64_t fullV[QSIZE], emptyV[QSIZE];
    alignas(8) uint64_t fullQ, emptyQ;
};

template<bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, int Br, int Bc, int d, int QSIZE, int NUM_THREADS>
__global__ 
void __launch_bounds__(NUM_THREADS) forward_kernel(const __grid_constant__ CUtensorMap tensorMapQ, const __grid_constant__ CUtensorMap tensorMapK, const __grid_constant__ CUtensorMap tensorMapV, const __grid_constant__ CUtensorMap tensorMapO, mykernelParamType param){
    constexpr int num_consumers = NUM_THREADS / 128 - 1;
    const int tid = threadIdx.x % 128;
    int warp_group_role = threadIdx.x / 128;
    int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ alignas(128) uint8_t shared_mem[];
    SMem<Br, Bc, d, QSIZE> &s = *reinterpret_cast<SMem<Br, Bc, d, QSIZE>*>(shared_mem);
    
    half *sQ = s.Q, *sK = s.K, *sV = s.V, *sO = s.O;

    uint64_t *fullQ = &s.fullQ, *emptyQ = &s.emptyQ;
    uint64_t *fullK = s.fullK, *emptyK = s.emptyK;
    uint64_t *fullV = s.fullV, *emptyV = s.emptyV;

    if (threadIdx.x == 0) {
        init_barrier(fullQ, 0, 1);
        init_barrier(emptyQ, 0, num_consumers*NUMWARPPERGROUP);
        #pragma unroll
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier(&fullK[i], 0, 1);
            init_barrier(&emptyK[i], 0, num_consumers*NUMWARPPERGROUP);
            init_barrier(&fullV[i], 0, 1);
            init_barrier(&emptyV[i], 0, num_consumers*NUMWARPPERGROUP);
        }
        asm volatile("fence.proxy.async.shared::cta;\n" : :);
    }

    __syncthreads();

    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    Schedule<0, Br, Bc, Is_causal, Is_local> schedule(param.B*param.N*param.Tr, gridDim.x, blockIdx.x, param.S, param.Tr, param.window_size_left);

    // producer
    if (warp_group_role == 0) {
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroup_reg_dealloc<num_regs>();
        if (tid == 0){
            int q = 0, p = 0;
            int qidx = 0;

            int n_block_min, n_block_max, blockId, seq_id;
            while (schedule.next(blockId, n_block_min, n_block_max, seq_id)) {
                wait(emptyQ, q);
                q ^= 1;
                expect_bytes(fullQ, (Br * d)*sizeof(half));
                load_async(&sQ[0], &tensorMapQ, fullQ, /*col=*/0, /*row=*/blockId * Br);

                for (int iter = n_block_min; iter < n_block_max; iter++, qidx++) {
                    if (qidx == QSIZE) { qidx = 0;  p ^= 1; }
                    wait(&emptyK[qidx], p);
                    expect_bytes(&fullK[qidx], (Bc * d)*sizeof(half)); 
                    load_async(&sK[qidx*Bc*d], &tensorMapK, &fullK[qidx], /*col=*/0, /*row=*/(seq_id * param.Tc + iter) * Bc);

                    wait(&emptyV[qidx], p);
                    expect_bytes(&fullV[qidx], (Bc * d)*sizeof(half)); 
                    load_async(&sV[qidx*Bc*d], &tensorMapV, &fullV[qidx], /*col=*/0, /*row=*/(seq_id * param.Tc + iter) * Bc);
                }
            }
        }
    }
    // consumer
    else{
        constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
        warpgroup_reg_alloc<num_regs>();

        --warp_group_role;
        warp_id -= 4;
        if (lane_id == 0) {
            arrive(emptyQ);
            for (int qidx = 0; qidx < QSIZE; ++qidx) {
                arrive(&emptyK[qidx]);
                arrive(&emptyV[qidx]);
            }
        }
        int q = 0, p = 0;
        int qidx = 0;

        float row_l_prev[Br/num_consumers/NUMWARPPERGROUP/MMA_M][2];
        float row_m_prev[Br/num_consumers/NUMWARPPERGROUP/MMA_M][2];
        float row_l[Br/num_consumers/NUMWARPPERGROUP/MMA_M][2];
        float row_m[Br/num_consumers/NUMWARPPERGROUP/MMA_M][2];

        uint32_t a_frag[MMA_M*MMA_K/WARP_SIZE/2]; // 除2是因为2个half打包在一个uint32_t里
        uint32_t b_frag[MMA_K*MMA_N/WARP_SIZE/2]; 
        float    c_frag[Br/num_consumers/NUMWARPPERGROUP/MMA_M][Bc/MMA_N][4];
        uint32_t d_frag[Br/num_consumers/NUMWARPPERGROUP/MMA_M][Bc/MMA_N/2][4];
        float    o_frag[Br/num_consumers/NUMWARPPERGROUP/MMA_M][d/MMA_N][4];

        int n_block_min, n_block_max, blockId, seq_id;
        while (schedule.next(blockId, n_block_min, n_block_max, seq_id)) {
            // 初始化结果
            #pragma unroll
            for (int m = 0; m < Br/num_consumers/NUMWARPPERGROUP/MMA_M; m++) {
                #pragma unroll
                for (int n = 0; n < d/MMA_N; n++) {
                    o_frag[m][n][0] = 0.0f;
                    o_frag[m][n][1] = 0.0f;
                    o_frag[m][n][2] = 0.0f;
                    o_frag[m][n][3] = 0.0f;
                }
            }

            #pragma unroll
            for (int i = 0; i < Br/num_consumers/NUMWARPPERGROUP/MMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++){
                    row_l_prev[i][j] = 0;
                    row_m_prev[i][j] = -INFINITY;
                }
            }

            curandStatePhilox4_32_10_t local_state;
            if constexpr (Is_dropout) { local_state = param.states[blockId]; }

            wait(fullQ, q);
            q ^= 1;

            const float alibi_slope = !Has_alibi ? 0.0f : param.alibi_slopes_ptr[seq_id%param.N];
            
            for (int iter = n_block_min; iter < n_block_max; iter++, qidx++) {
                if (qidx == QSIZE) {qidx = 0; p ^= 1; }
                wait(&fullK[qidx], p);

                // 初始化矩阵乘结果
                memset(&c_frag[0][0][0], 0, sizeof(c_frag));

                #pragma unroll
                for (int i = 0; i < Br/num_consumers/NUMWARPPERGROUP/MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < 2; j++){
                        row_l[i][j] = 0;
                        row_m[i][j] = -INFINITY;
                    }
                }

                // S = Q * K
                #pragma unroll
                for (int k = 0; k < d / MMA_K; k++){
                    half* mma_sQ = sQ + warp_id * Br / num_consumers / NUMWARPPERGROUP * d + k * MMA_K + lane_id % 16 * d + lane_id / 16 * 8;
                    half* mma_sK = sK + qidx * Bc * d + k * MMA_K + lane_id % 8 * d + lane_id / 8 * 8;
                    #pragma unroll
                    for (int m = 0; m < Br / num_consumers / NUMWARPPERGROUP / MMA_M; m++){
                        ldmatrix_x4(&a_frag[0], &mma_sQ[m*MMA_M*d]);
                        #pragma unroll
                        for (int n = 0; n < Bc / MMA_N; n++){
                            ldmatrix_x2(&b_frag[0], &mma_sK[n*MMA_N*d]);
                            mma16816(&c_frag[m][n][0], &a_frag[0], &b_frag[0]);
                        }
                    }
                }

                if (lane_id == 0){
                    if (iter == n_block_max - 1)    arrive(emptyQ);
                    arrive(&emptyK[qidx]);
                }

                #pragma unroll
                for (int i = 0; i < Br/num_consumers/NUMWARPPERGROUP/MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Bc/MMA_N; j++) {
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            c_frag[i][j][k] *= param.softmax_scale;
                        }
                    }
                }

                apply_mask<Is_causal, Is_local, Has_alibi, num_consumers, Br, Bc>(c_frag, 
                    (blockId % param.Tr) * Br + warp_id * Br / num_consumers / NUMWARPPERGROUP,
                    iter * Bc, tid, param.window_size_left, alibi_slope
                );

                apply_softmax<num_consumers, Br, Bc>(c_frag, row_m_prev, row_m, row_l, row_l_prev);

                if constexpr (Is_dropout) {
                    #pragma unroll
                    for (int i = 0; i < Br/num_consumers/NUMWARPPERGROUP/MMA_M; i++) {
                        #pragma unroll
                        for (int j = 0; j < Bc/MMA_N; j++) {
                            #pragma unroll
                            for (int k = 0; k < 4; k++) {
                                float rand_val = curand_uniform(&local_state);

                                if (rand_val < param.dropout_prob) {
                                    c_frag[i][j][k] = 0.0f;
                                } else {
                                    c_frag[i][j][k] *= 1.0f / (1.0f - param.dropout_prob);
                                }
                            }
                        }
                    }
                }


                // 转换矩阵 P 的类型，为了后续的 tensor core 矩阵乘
                #pragma unroll
                for (int i = 0; i < Br / num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Bc / MMA_N / 2; j++) {
                        d_frag[i][j][0] = pack_2float_to_1uint32(c_frag[i][2*j][0],   c_frag[i][2*j][1]);
                        d_frag[i][j][1] = pack_2float_to_1uint32(c_frag[i][2*j][2],   c_frag[i][2*j][3]);
                        d_frag[i][j][2] = pack_2float_to_1uint32(c_frag[i][2*j+1][0], c_frag[i][2*j+1][1]);
                        d_frag[i][j][3] = pack_2float_to_1uint32(c_frag[i][2*j+1][2], c_frag[i][2*j+1][3]);
                    }
                }

                wait(&fullV[qidx], p);

                // O = P * V
                #pragma unroll
                for (int m = 0; m < Br / num_consumers / NUMWARPPERGROUP / MMA_M; m++) {
                    half* mma_sV = sV + qidx * Bc * d + lane_id % 16 * d;
                    #pragma unroll
                    for (int n = 0; n < d / MMA_N; n++) {
                        memset(&c_frag[0][0][0], 0, 4 * sizeof(float));   // 初始化结果
                        #pragma unroll
                        for (int k = 0; k < Bc / MMA_K; k++) {
                            ldmatrix_x2_T(&b_frag[0], &mma_sV[n*MMA_N + k*MMA_K*d]);
                            mma16816(&c_frag[0][0][0], &d_frag[m][k][0], &b_frag[0]);
                        }

                        // 更新 O 值
                        o_frag[m][n][0] = o_frag[m][n][0] * __expf(row_m_prev[m][0] - row_m[m][0]) + c_frag[0][0][0];
                        o_frag[m][n][1] = o_frag[m][n][1] * __expf(row_m_prev[m][0] - row_m[m][0]) + c_frag[0][0][1];
                        o_frag[m][n][2] = o_frag[m][n][2] * __expf(row_m_prev[m][1] - row_m[m][1]) + c_frag[0][0][2];
                        o_frag[m][n][3] = o_frag[m][n][3] * __expf(row_m_prev[m][1] - row_m[m][1]) + c_frag[0][0][3];
                    }
                }

                if (lane_id == 0) {
                    arrive(&emptyV[qidx]);
                }

                // 更新最大值与和
                #pragma unroll
                for (int i = 0; i < Br / num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
                    row_m_prev[i][0] = row_m[i][0];
                    row_m_prev[i][1] = row_m[i][1];
                    row_l_prev[i][0] = row_l[i][0];
                    row_l_prev[i][1] = row_l[i][1];
                }
            }

            // O 除以一行之和
            #pragma unroll
            for (int i = 0; i < Br / num_consumers / NUMWARPPERGROUP / MMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < d / MMA_N; j++) {
                    o_frag[i][j][0] /= row_l[i][0];
                    o_frag[i][j][1] /= row_l[i][0];
                    o_frag[i][j][2] /= row_l[i][1];
                    o_frag[i][j][3] /= row_l[i][1];
                }
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            // 将 O 加载至共享内存
            half* block_sO = sO + warp_group_role * Br / num_consumers * d;
            uint32_t tid_offset = warp_id % NUMWARPPERGROUP * Br / num_consumers / NUMWARPPERGROUP * d + lane_id % 16 * d;
            uint32_t base_addr = static_cast<uint32_t>(__cvta_generic_to_shared(block_sO)) + tid_offset * sizeof(half);

            half o_frag_half[4];
            int* data_ptr = (int*)o_frag_half;
            #pragma unroll
            for (int m = 0; m < Br/num_consumers/NUMWARPPERGROUP/MMA_M; m++) {
                #pragma unroll
                for (int n = 0; n < d / MMA_N; n++) {
                    uint32_t addr = base_addr + (m * MMA_M * d + n * MMA_N) * sizeof(half);
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        o_frag_half[k] = __float2half(o_frag[m][n][k]);
                    }
                    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared::cta.b16 [%0], {%1, %2};"
                                :: "r"(addr), "r"(data_ptr[0]), "r"(data_ptr[1]));
                }
            }

            // 将 O 加载至全局内存
            asm volatile("bar.sync %0, 128;\n" ::"r"(warp_group_role + 2) : "memory");
            if (tid == 0) {
                store_async(&tensorMapO, block_sO, /*col=*/0, /*row=*/blockId * Br + warp_group_role * Br / num_consumers);
                asm volatile("cp.async.bulk.commit_group;");
            }

            if constexpr (Is_dropout) {
                param.states[blockId] = local_state;
            }
        }
    }
}


template<int Br, int Bc, int NUM_THREADS, int NUM_CONSUMERS, int QSIZE>
void run_kernel(mykernelParamType param, int num_sm,bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi) {
    dim3 grid(num_sm);
    dim3 block(NUM_THREADS);

    if (param.d == 64) {
        d_tma_map_Q = create_tensor_map<Br, 64>(param.Q, param.B*param.N*param.S, 64);
        d_tma_map_K = create_tensor_map<Bc, 64>(param.K, param.B*param.N*param.S, 64);
        d_tma_map_V = create_tensor_map<Bc, 64>(param.V, param.B*param.N*param.S, 64);
        d_tma_map_O = create_tensor_map<Br / NUM_CONSUMERS, 64, false, false>(param.O, param.B*param.N*param.S, 64);

        constexpr size_t sMemSize = sizeof(SMem<Br, Bc, 64, QSIZE>);
        static_assert(sMemSize < 256 * 1024);

        if (Is_dropout) {
            if (Is_causal) {
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, true, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, true, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, true, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, true, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, false, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, false, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, false, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, false, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
            else{
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, true, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, true, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, true, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, true, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, false, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, false, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, false, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, false, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
        }
        else{
            if (Is_causal) {
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, true, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, true, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, true, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, true, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, false, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, false, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, false, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, false, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
            else{
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, true, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, true, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, true, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, true, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, false, true, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, false, true, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, false, false, Br, Bc, 64, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, false, false, Br, Bc, 64, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
        }
    }
    else if (param.d == 128) {
        d_tma_map_Q = create_tensor_map<Br, 128>(param.Q, param.B*param.N*param.S, 128);
        d_tma_map_K = create_tensor_map<Bc, 128>(param.K, param.B*param.N*param.S, 128);
        d_tma_map_V = create_tensor_map<Bc, 128>(param.V, param.B*param.N*param.S, 128);
        d_tma_map_O = create_tensor_map<Br / NUM_CONSUMERS, 128, false, false>(param.O, param.B*param.N*param.S, 128);

        constexpr size_t sMemSize = sizeof(SMem<Br, Bc, 64, QSIZE>);
        static_assert(sMemSize < 256 * 1024);

        if (Is_dropout) {
            if (Is_causal) {
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, true, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, true, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, true, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, true, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, false, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, false, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, true, false, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, true, false, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
            else{
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, true, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, true, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, true, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, true, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, false, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, false, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<true, false, false, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<true, false, false, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
        }
        else{
            if (Is_causal) {
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, true, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, true, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, true, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, true, false, Br, Bc, 128, QSIZE,NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, false, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, false, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, true, false, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, true, false, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
            else{
                if (Is_local) {
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, true, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, true, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, true, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, true, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
                else{
                    if (Has_alibi) {
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, false, true, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, false, true, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                    else{
                        CUDA_CHECK(cudaFuncSetAttribute(forward_kernel<false, false, false, false, Br, Bc, 128, QSIZE, NUM_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));
                        forward_kernel<false, false, false, false, Br, Bc, 128, QSIZE, NUM_THREADS><<<grid, block, sMemSize>>>(d_tma_map_Q, d_tma_map_K, d_tma_map_V, d_tma_map_O, param);
                    }
                }
            }
        }
    }
    else {
        throw std::runtime_error("Only d=64 and d=128 are supported in this implementation.");
    }
}


void run_flash_attention(
    const int B, 
    const int N, 
    const int S, 
    const int d,
    half *Q, 
    half *K, 
    half *V, 
    half *O, 
    half *L = nullptr, 
    half *M = nullptr,
    bool Is_dropout = false,
    bool Is_causal = false,
    bool Is_local = false,
    bool Has_alibi = false,
    int window_size = 0,
    float *alibi_slopes = nullptr,
    float dropout_prob = 0,
    curandStatePhilox4_32_10_t* states = nullptr
) {
    cudaError_t error;
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed for device 0 :"
                    << cudaGetErrorString(error) << std::endl;
    }

    constexpr int NUM_SM = 1;
    constexpr int NUM_THREADS = 128*3;
    constexpr int Br = 128;
    constexpr int Bc = 256;
    constexpr int NUM_CONSUMERS = (NUM_THREADS / 128) - 1;
    constexpr int QSIZE = 1;

    mykernelParamType param;
    param.Q                 = Q;
    param.K                 = K;
    param.V                 = V;
    param.O                 = O;
    param.B                 = B;
    param.N                 = N;
    param.S                 = S;
    param.d                 = d;
    param.Tc                = ceil((float)S / Bc);
    param.Tr                = ceil((float)S / Br);
    param.softmax_scale     = 1.0 / sqrt(d);
    param.window_size_right = window_size;
    param.window_size_left  = window_size;
    param.alibi_slopes_ptr  = alibi_slopes;
    param.dropout_prob      = dropout_prob;
    param.states            = states;

    float time_elapsed=0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    run_kernel<Br, Bc, NUM_THREADS, NUM_CONSUMERS, QSIZE>(param, prop.multiProcessorCount, Is_dropout, Is_causal, Is_local, Has_alibi);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时结束
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("TIME: %f\n", time_elapsed);
}

#endif



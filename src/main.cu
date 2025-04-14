#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "init_curand_states.h"
#include "flash_forward.h"
#include "flash_forward_splitkv.h"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

std::default_random_engine generator(26);
std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

// 核函数指针类型定义
template<bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi>
using KernelFunctionPtr = void (*)(mykernelParamType);

void verfiy(float* O, float* O_host, const int batch_size, const int n_heads, const int seq_len, const int head_dim, float range_of_error);

void attention_forward_cpu(float* Q, float* K, float* V, float softmax_scale, const int batch_size, const int n_heads, const int seq_len, 
    const int head_dim, float* output, const bool use_causal_mask = false, int window_size = -1, float* alibi_slopes = nullptr);

int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits, int seq_len);


int main(){
    const int  batch_size       = 1;
    const int  n_heads          = 8;
    const int  seq_len          = 1024;
    const int  head_dim         = 32;

    const bool dropout          = false;      // 一旦启用dropout,那核函数的结果和没有使用dropout的cpu端结果必然不同,因此便不再验证结果正确性
    const bool causal_mask      = true;     // 一般来说， causal_mask不会和window_attention同时启用 
    const bool window_attention = false;
    const bool alibi            = true;
    const bool Is_even_K        = !(head_dim % 32);
    const bool Is_even_MN       = !(seq_len % 128);

    float dropout_prob = 0.0f;
    curandStatePhilox4_32_10_t* d_states;
    if(dropout){
        dropout_prob = 0.1f;
    }

    int window_size = -1;
    if(window_attention){
        window_size = 128;
    }
    float *alibi_slopes = nullptr;
    float *alibi_slopes_device = nullptr;
    if(alibi){
        alibi_slopes = (float*)malloc(n_heads*sizeof(float));
        for (int i = 0; i < n_heads; i++){
            alibi_slopes[i] = -std::pow(2, -8.0 / n_heads * (i + 1));
        }
        cudaMalloc((void**)&alibi_slopes_device, n_heads*sizeof(float));
        cudaMemcpy(alibi_slopes_device, alibi_slopes, n_heads*sizeof(float),cudaMemcpyHostToDevice);
    }
      
    float *Q      = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));
    float *K      = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));
    float *V      = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));
    float *O      = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));
    float *O_host = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));

    half *Q_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));
    half *K_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));
    half *V_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));

    float *O_device;
    half  *Q_device_half,*K_device_half,*V_device_half;
    cudaMalloc((void**)&O_device, batch_size*n_heads*seq_len*head_dim*sizeof(float));
    cudaMalloc((void**)&Q_device_half, batch_size*n_heads*seq_len*head_dim*sizeof(half));
    cudaMalloc((void**)&K_device_half, batch_size*n_heads*seq_len*head_dim*sizeof(half));
    cudaMalloc((void**)&V_device_half, batch_size*n_heads*seq_len*head_dim*sizeof(half));

    float* O_tmp; float* L; float* M;

    for(int i = 0; i < batch_size*n_heads*seq_len*head_dim; i++)
    {
        Q[i] = distribution(generator);
        K[i] = distribution(generator);
        V[i] = distribution(generator);
        O[i] = 0.0f;

        Q_half[i] = __float2half(Q[i]);
        K_half[i] = __float2half(K[i]);
        V_half[i] = __float2half(V[i]);
    }

    cudaMemcpy(Q_device_half, Q_half, batch_size*n_heads*seq_len*head_dim*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(K_device_half, K_half, batch_size*n_heads*seq_len*head_dim*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(V_device_half, V_half, batch_size*n_heads*seq_len*head_dim*sizeof(half),cudaMemcpyHostToDevice);
    
    mykernelParamType param;
    param.Q                 = Q_device_half;
    param.K                 = K_device_half;
    param.V                 = V_device_half;
    param.O                 = O_device;
    param.N                 = seq_len;
    param.d                 = head_dim;
    param.Br                = 128;
    param.Bc                = 128;
    param.Tc                = ceil(seq_len / param.Bc);
    param.Tr                = ceil(seq_len / param.Br);
    param.softmax_scale     = 1.0 / sqrt(head_dim);
    param.window_size_right = window_size;
    param.window_size_left  = window_size;
    param.alibi_slopes_ptr  = alibi_slopes_device;

    int split_num = num_splits_heuristic(param.Tr * n_heads * batch_size, 108, param.Tr, seq_len / 128, seq_len);

    if(dropout){
        // 分配状态内存
        int num_blocks = param.Tr * n_heads * batch_size * split_num * 256;
        cudaMalloc(&d_states, num_blocks * sizeof(curandStatePhilox4_32_10_t));

        // 初始化状态
        dim3 grid((num_blocks + 255)/256, 1, 1);
        int seed = 48;
        init_curand_states<<<grid, 256>>>(d_states, seed, num_blocks);
        param.dropout_prob      = dropout_prob;
        param.states            = d_states;
    }

    // CPU端计算正确结果
    attention_forward_cpu(Q, K, V, param.softmax_scale, batch_size, n_heads, seq_len, head_dim, O, causal_mask, window_size, alibi_slopes);

    KernelFunctionPtr<dropout, causal_mask, window_attention, alibi> selectedKernel = nullptr;
    int griddim_x = 1, griddim_y = 1, griddim_z = 1;       // 初始化网格大小
    int blockdim_x = 1, blockdim_y = 1, blockdim_z = 1;    // 初始化线程块大小
    int sram_size;

    // 开始根据数据维度来选择核函数
    if(split_num == 1){
        // GPU网格尺寸
        griddim_x = param.Tr;
        griddim_y = n_heads;
        griddim_z = batch_size;
        // GPU线程块尺寸
        blockdim_x = 256;
        if(Is_even_K && Is_even_MN){
            // 共享内存大小
            sram_size = (param.Br + param.Bc * 2) * param.d * sizeof(half) + param.Br * param.d * sizeof(float);
            selectedKernel = forward_kernel<dropout, causal_mask, window_attention, alibi>;
        }
        else{
            int d_align32 = (param.d + 31) / 32 * 32;
            sram_size = (param.Br + param.Bc * 2) * d_align32 * sizeof(half) + param.Br * d_align32 * sizeof(float);
            // selectedKernel = forward_kernel_general<dropout, causal_mask, window_attention, alibi>;
        }
    }
    else{
        // GPU网格尺寸
        griddim_x = param.Tr * split_num;
        griddim_y = n_heads;
        griddim_z = batch_size; 
        // GPU线程块尺寸
        blockdim_x = 256;

        cudaMalloc((void**)&O_tmp, batch_size*n_heads*seq_len*head_dim*split_num*sizeof(float));
        cudaMalloc((void**)&L, batch_size*n_heads*seq_len*split_num*sizeof(float));
        cudaMalloc((void**)&M, batch_size*n_heads*seq_len*split_num*sizeof(float));

        param.O_tmp = O_tmp;
        param.split_num = split_num;
        param.L = L;
        param.M = M;

        if(Is_even_K && Is_even_MN){
            // 共享内存大小
            sram_size = (param.Br + param.Bc * 2) * param.d * sizeof(half) + param.Br * param.d * sizeof(float);
            selectedKernel = forward_kernel_splitkv<dropout, causal_mask, window_attention, alibi>;
        }
        else{
            int d_align32 = (param.d + 31) / 32 * 32;
            sram_size = (param.Br + param.Bc * 2) * d_align32 * sizeof(half) + param.Br * d_align32 * sizeof(float);
            // selectedKernel = forward_kernel_splitkv_general<dropout, causal_mask, window_attention, alibi>;
        }
    }

    dim3 grid_dim(griddim_x, griddim_y, griddim_z);
    dim3 block_dim(blockdim_x, blockdim_y, blockdim_z);
    
    // 计时
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    float time_elapsed=0.0;
    // 核函数启动
    selectedKernel<<<grid_dim, block_dim, sram_size>>>(param);
    if(split_num > 1){
        dim3 combine_grid_dim(param.Tr, n_heads, batch_size);
        forward_kernel_splitkv_combine<<<combine_grid_dim, 128>>>(param);
    }    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时结束
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    // 将GPU结果拷贝回主机端
    cudaMemcpy(O_host, O_device, batch_size*n_heads*seq_len*head_dim*sizeof(float), cudaMemcpyDeviceToHost);
    printf("kernel time: %f us\n", time_elapsed*1000);
    // 检验结果正确性
    if(!dropout){
        printf("Verify the result of kernel function\n");
        verfiy(O, O_host, batch_size, n_heads, seq_len, head_dim, 0.05);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放显存
    cudaFree(O_device);
    cudaFree(Q_device_half);
    cudaFree(K_device_half);
    cudaFree(V_device_half);
    cudaFree(d_states);
    
    cudaFree(L);
    cudaFree(M);
    cudaFree(O_tmp);
    
    // 释放内存
    free(Q);
    free(K);
    free(V);
    free(O);
    free(O_host);
    free(Q_half);
    free(K_half);
    free(V_half);
    
    return 0;
}


void verfiy(
    float* O, 
    float* O_host,
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    float range_of_error)
{
    int error=0;
    printf("===================start verfiy===================\n");
    for(int i=0;i<batch_size*n_heads*seq_len*head_dim;i++)
    {
        float device_out = O_host[i];
        if((fabs(O_host[i] - O[i]))/O_host[i] > range_of_error || std::isnan(device_out) || std::isinf(device_out))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, O_host[i], O[i]);
            error++;
            break;
        }        
    }
    printf("==================finish,error:%d==================\n",error);
}


void attention_forward_cpu(
    float* Q, 
    float* K, 
    float* V, 
    float softmax_scale,
    const int batch_size,
    const int n_heads,
    const int seq_len, 
    const int head_dim, 
    float* output,
    const bool use_causal_mask,
    int window_size,
    float* alibi_slopes)
{
    const int head_size = seq_len * head_dim;
    const int seq_sq = seq_len * seq_len;

    // 临时存储注意力分数
    float* scores = new float[seq_sq];

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            // 获取当前head的指针偏移量
            const int base_offset = b * n_heads * head_size + h * head_size;
            const float* Q_ptr = Q + base_offset;
            const float* K_ptr = K + base_offset;
            const float* V_ptr = V + base_offset;
            float* out_ptr = output + base_offset;

            // 1. 计算QK^T
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < head_dim; ++k) {
                        sum += Q_ptr[i * head_dim + k] * K_ptr[j * head_dim + k];
                    }
                    scores[i * seq_len + j] = sum * softmax_scale;
                }
            }

            // 2. 应用ALiBi偏置
            if (alibi_slopes != nullptr) {
                const float slope = alibi_slopes[h];
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        scores[i * seq_len + j] -= slope * std::abs(i - j);
                    }
                }
            }

            // 3. 应用注意力掩码
            if (use_causal_mask) {
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        if (j > i) {
                            scores[i * seq_len + j] = -INFINITY;
                        }
                    }
                }
            }

            if (window_size >= 0) {
                const int w = window_size;
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < seq_len; ++j) {
                        if (std::abs(i - j) > w) {
                            scores[i * seq_len + j] = -INFINITY;
                        }
                    }
                }
            }

            // 4. Softmax计算
            for (int i = 0; i < seq_len; ++i) {
                float max_val = -INFINITY;
                float* row = scores + i * seq_len;
                
                // 计算行最大值
                for (int j = 0; j < seq_len; ++j) {
                    max_val = std::max(max_val, row[j]);
                }

                // 计算指数和
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    row[j] = expf(row[j] - max_val);
                    sum += row[j];
                }

                // 归一化
                for (int j = 0; j < seq_len; ++j) {
                    row[j] /= sum;
                }
            }

            // 5. 计算加权和
            for (int i = 0; i < seq_len; ++i) {
                for (int k = 0; k < head_dim; ++k) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        sum += scores[i * seq_len + j] * V_ptr[j * head_dim + k];
                    }
                    out_ptr[i * head_dim + k] = sum;
                }
            }
        }
    }

    delete[] scores;
}


int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits, int seq_len) {
    if (batch_nheads_mblocks >= 4.0f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&ceildiv, &num_n_blocks, &seq_len](int num_splits) {
        return (num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)) && (seq_len % (128 * num_splits) == 0);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            // 计算拆分后的平均efficiency
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    // 选择满足85%利用率最小的拆分
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

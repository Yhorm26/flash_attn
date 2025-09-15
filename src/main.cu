#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "init_curand_states.h"
#include "flash_forward.h"


void verify(half* O, half* O_host, const int batch_size, const int n_heads, const int seq_len, const int head_dim, float range_of_error);

void attention_forward_cpu(const half* Q, const half* K, const half* V, float softmax_scale, const int batch_size, const int n_heads, const int seq_len, 
    const int head_dim, half* output, const bool use_causal_mask = false, int window_size = -1, const float* alibi_slopes = nullptr);

int main(){
    int  batch_size       = 2;
    int  n_heads          = 8;
    int  seq_len          = 2048;
    int  head_dim         = 64;

    bool dropout          = false;
    bool causal_mask      = true;
    bool window_attention = false;
    bool alibi            = true;
    float dropout_prob    = 0.0f;
    int window_size       = -1;

    curandStatePhilox4_32_10_t* d_states;

    float *alibi_slopes = nullptr;
    float *alibi_slopes_device = nullptr;
    if (alibi) {
        alibi_slopes = (float*)malloc(n_heads*sizeof(float));
        for (int i = 0; i < n_heads; i++){
            alibi_slopes[i] = -std::pow(2, -8.0 / n_heads * (i + 1));
        }
        cudaMalloc((void**)&alibi_slopes_device, n_heads*sizeof(float));
        cudaMemcpy(alibi_slopes_device, alibi_slopes, n_heads*sizeof(float),cudaMemcpyHostToDevice);
    }

    if (window_attention) {
        window_size = 128;
    }
      
    float *Q = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));
    float *K = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));
    float *V = (float*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(float));

    half *Q_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));
    half *K_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));
    half *V_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));
    half *O_half = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));
    half *O_host = (half*)malloc(batch_size*n_heads*seq_len*head_dim*sizeof(half));

    half  *Q_device,*K_device,*V_device, *O_device;
    cudaMalloc((void**)&Q_device, batch_size*n_heads*seq_len*head_dim*sizeof(half));
    cudaMalloc((void**)&K_device, batch_size*n_heads*seq_len*head_dim*sizeof(half));
    cudaMalloc((void**)&V_device, batch_size*n_heads*seq_len*head_dim*sizeof(half));
    cudaMalloc((void**)&O_device, batch_size*n_heads*seq_len*head_dim*sizeof(half));

    float* O_tmp; float* L; float* M;

    std::default_random_engine generator(26);
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);
    for(int i = 0; i < batch_size*n_heads*seq_len*head_dim; i++)
    {
        Q[i] = distribution(generator);
        K[i] = distribution(generator);
        V[i] = distribution(generator);

        Q_half[i] = __float2half(Q[i]);
        K_half[i] = __float2half(K[i]);
        V_half[i] = __float2half(V[i]);
        O_half[i] = 0;
    }

    cudaMemcpy(Q_device, Q_half, batch_size*n_heads*seq_len*head_dim*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(K_device, K_half, batch_size*n_heads*seq_len*head_dim*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(V_device, V_half, batch_size*n_heads*seq_len*head_dim*sizeof(half),cudaMemcpyHostToDevice);

    if (dropout) {
        // 分配状态内存
        int num_blocks = ceil((float)seq_len / 128) * n_heads * batch_size * 256;
        cudaMalloc(&d_states, num_blocks * sizeof(curandStatePhilox4_32_10_t));

        // 初始化状态
        dim3 grid((num_blocks + 255)/256, 1, 1);
        int seed = 48;
        init_curand_states<<<grid, 256>>>(d_states, seed, num_blocks);
    }

    // GPU端计算结果
    run_flash_attention(batch_size, n_heads, seq_len, head_dim, Q_device, K_device, V_device, O_device, nullptr, nullptr, dropout, causal_mask, window_attention, alibi, window_size, alibi_slopes_device, dropout_prob);

    cudaMemcpy(O_host, O_device, batch_size*n_heads*seq_len*head_dim*sizeof(half), cudaMemcpyDeviceToHost);
    // 检验结果正确性
    if(!dropout){
        printf("Verify the result of kernel function\n");
        // CPU端计算正确结果
        attention_forward_cpu(Q_half, K_half, V_half, 1.0 / sqrt(head_dim), batch_size, n_heads, seq_len, head_dim, O_half, causal_mask, window_size, alibi_slopes);
        verify(O_half, O_host, batch_size, n_heads, seq_len, head_dim, 0.06);
    }

    // 释放显存
    cudaFree(O_device);
    cudaFree(Q_device);
    cudaFree(K_device);
    cudaFree(V_device);
    cudaFree(d_states);
    
    cudaFree(L);
    cudaFree(M);
    cudaFree(O_tmp);
    
    // 释放内存
    free(Q);
    free(K);
    free(V);
    free(O_half);
    free(O_host);
    free(Q_half);
    free(K_half);
    free(V_half);
    
    return 0;
}

void verify(
    half* O, 
    half* O_host,
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    float range_of_error)
{
    int error=0;
    printf("===================start verify===================\n");
    for(int i=0;i<batch_size*n_heads*seq_len*head_dim;i++)
    {
        float device_out = __float2half(O_host[i]);
        float host_out = __float2half(O[i]);
        if((fabs(device_out - host_out))/host_out > range_of_error || std::isnan(device_out) || std::isinf(device_out))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, device_out, host_out);
            error++;
            break;
        }        
    }
    printf("==================finish,error:%d==================\n",error);
}

void attention_forward_cpu(
    const half* Q,
    const half* K,
    const half* V,
    float softmax_scale,
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    half* output,
    const bool use_causal_mask,
    int window_size,
    const float* alibi_slopes)
{
    const int head_size = seq_len * head_dim;
    const int seq_sq = seq_len * seq_len;

    // 临时存储注意力分数
    float* scores = new float[seq_sq];

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            // 获取当前head的指针偏移量
            const int base_offset = b * n_heads * head_size + h * head_size;
            const half* Q_ptr = Q + base_offset;
            const half* K_ptr = K + base_offset;
            const half* V_ptr = V + base_offset;
            half* out_ptr = output + base_offset;

            // 1. 计算QK^T
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < head_dim; ++k) {
                        sum += __half2float(Q_ptr[i * head_dim + k] * K_ptr[j * head_dim + k]);
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
                        sum += __half2float(__float2half(scores[i * seq_len + j]) * V_ptr[j * head_dim + k]);
                    }
                    out_ptr[i * head_dim + k] = __float2half(sum);
                }
            }
        }
    }

    delete[] scores;
}
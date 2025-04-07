#ifndef INIT_CURAND_STATES_H
#define INIT_CURAND_STATES_H
#include <cuda_runtime.h>
#include <curand_kernel.h>


__global__ void init_curand_states(curandStatePhilox4_32_10_t* states, 
                                 unsigned long long seed,
                                 int total_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}
#endif

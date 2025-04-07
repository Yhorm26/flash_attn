#ifndef FLASH_H
#define FLASH_H
#include "utils.h"

template<bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi>
__global__ void forward_kernel(mykernelParamType param);

#endif

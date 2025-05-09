# flash_attn
不使用cuTe和cutlass来实现一个功能全面的flash attention。

编译指令：

```nvcc -I ../include *.cu -o flash  -arch=sm_80 -std=c++11```

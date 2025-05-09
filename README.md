# flash_attn
不使用cuTe和cutlass来实现一个功能全面的flash attention。

在src目录下进行编译，编译指令：

```nvcc -I ../include *.cu -o flash  -arch=sm_80 -std=c++11```

支持任意维度的attention，只要显存够就行。

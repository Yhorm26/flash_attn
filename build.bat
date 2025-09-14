@echo off
REM 如果编译失败的话，可以参考 : https://blog.csdn.net/qq_42147816/article/details/127160601
REM 编译 CUDA 程序

echo Start compiling, this may take a few minutes

nvcc -O2 -I include src/main.cu -o flash -arch=sm_120a -std=c++17 -w -Xptxas --disable-warnings -lcuda  

REM 检查编译是否成功
if %errorlevel% equ 0 (
    echo Build succeeded.
) else (
    echo Build failed.
)

echo Start running
flash.exe

pause
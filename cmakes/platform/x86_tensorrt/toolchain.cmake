# 使用编译器
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# 设置cuda编译器路径，并启用cuda编程语言支持
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
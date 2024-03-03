cmake_minimum_required(VERSION 3.26)

# 设置C++标准和调试模式
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)

# 不使用静态cuda_runtime库，即可执行文件中不包括cuda_runtime库
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# 初始化相关路径
set(CUDA_ROOT_PATH /usr/local/cuda-11.1)
set(TRT_INCLUDE_PATH /usr/include/x86_64-linux-gnu/)
set(TRT_LIB_PATH /usr/lib/x86_64-linux-gnu/)

# tensorrt
#if(SYSTEM_TYPE STREQUAL "ARM")
#    set(TRT_INCLUDE_PATH /usr/include/aarch64-linux-gnu/)
#    set(TRT_LIB_PATH /usr/lib/aarch64-linux-gnu/)
#elseif (SYSTEM_TYPE STREQUAL "x86")
#    #    set(TRT_INCLUDE_PATH ${TRT_ROOT_PATH}/include)
#    #    set(TRT_LIB_PATH ${TRT_ROOT_PATH}/lib)
#    #    message(STATUS "use custom tensorrt in ${TRT_ROOT_PATH}")
#    set(TRT_INCLUDE_PATH /usr/include/x86_64-linux-gnu/)
#    set(TRT_LIB_PATH /usr/lib/x86_64-linux-gnu/)
#else ()
#endif()
message(STATUS "TensorRT in ${TRT_ROOT_PATH}")
include_directories(${TRT_INCLUDE_PATH})
link_directories(${TRT_LIB_PATH})

# cuda
find_package(CUDA REQUIRED)
set(CUDA_INCLUDE_PATH  ${CUDA_ROOT_PATH}/include)
set(CUDA_LIB_PATH  ${CUDA_ROOT_PATH}/lib64)
include_directories(${CUDA_INCLUDE_PATH})
link_directories(${CUDA_LIB_PATH})

# spdlog
find_package(spdlog REQUIRED)
include_directories(${spdlog_INCLUDE_DIRS})

# 添加当前目录下的所有源文件
file(GLOB SOURCES "*.cpp")

# 生成动态链接库
add_library(tensorrt_engine SHARED ${SOURCES})
target_link_libraries(lpr_sdk nvinfer nvonnxparser cudart nvinfer_plugin ${spdlog_LIBS})
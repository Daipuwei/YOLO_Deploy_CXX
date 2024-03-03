cmake_minimum_required(VERSION 3.26)
project(YOLO_Deploy_CXX)

set(CMAKE_CXX_STANDARD 14)

# 不使用静态cuda_runtime库，即可执行文件中不包括cuda_runtime库
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)


# cuda
find_package(CUDA REQUIRED)
set(CUDA_INCLUDE_PATH  ${CUDA_ROOT_PATH}/include)
set(CUDA_LIB_PATH  ${CUDA_ROOT_PATH}/lib64)
include_directories(${CUDA_INCLUDE_PATH})
link_directories(${CUDA_LIB_PATH})


# spdlog
find_package(spdlog REQUIRED)
include_directories(${spdlog_INCLUDE_DIRS})

add_executable(YOLO_Deploy_CXX main.cpp
        model/base_detection_model.h detection_common.h)
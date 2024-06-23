//
// Created by dpw on 24-6-9.
//

#ifndef YOLO_DEPLOY_TENSORRT_UTILS_H
#define YOLO_DEPLOY_TENSORRT_UTILS_H

#include "vector"

#include <numeric>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "device_launch_parameters.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferVersion.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) { \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

/**
 * @brief 这是获取张量节点数据类型大小的函数
 * @param data_type Tensor数据类型
 * @return 数据类型大小
 */
int get_tensor_data_type_size(DataType data_type);

/**
 * @brief 这是计算张量大小的函数
 * @param dimensions 维度数组
 * @return 张量大小
 */
int get_tensor_size(Dims dimensions);

/**
 * @brief 这是获取张量形状的函数
 * @param dimensions 维度数组
 * @return 张量形状数组
 */
std::vector<int> get_tensor_shape(Dims dimensions);

#endif //YOLO_DEPLOY_TENSORRT_UTILS_H

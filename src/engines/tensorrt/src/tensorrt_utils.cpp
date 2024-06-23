//
// Created by dpw on 24-6-9.
//

#include <functional>
#include <vector>
#include "numeric"
#include "tensorrt_utils.h"

/**
 * @brief 这是获取张量节点数据类型大小的函数
 * @param data_type Tensor数据类型
 * @return 数据类型大小
 */
int get_tensor_data_type_size(DataType data_type) {
    int elem_byte;
    switch (data_type) {
        case nvinfer1::DataType::kHALF:
            elem_byte = sizeof(float) / 2;
            break;
        default:
            elem_byte = sizeof(float);
            break;
    }
    return elem_byte;
}

/**
 * @brief 这是计算张量大小的函数
 * @param dimensions 维度数组
 * @return 张量大小
 */
int get_tensor_size(Dims dimensions) {
    int size = std::accumulate(dimensions.d,dimensions.d+dimensions.nbDims,
                               1,std::multiplies<int64_t>());
    return size;
}

/**
 * @brief 这是获取张量形状的函数
 * @param dimensions 维度数组
 * @return 张量形状数组
 */
std::vector<int> get_tensor_shape(Dims dimensions) {
    std::vector<int> shape;
    for(int i = 0 ; i < dimensions.nbDims ; i++){
        shape.emplace_back((int)dimensions.d[i]);
    }
    return shape;
}
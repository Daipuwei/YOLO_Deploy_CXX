//
// Created by dpw on 24-3-1.
//

#ifndef YOLO_DEPLOY_CXX_TENSORRT_ENGINE_H
#define YOLO_DEPLOY_CXX_TENSORRT_ENGINE_H

#include "mutex"
#include "vector"
#include "string"

#include "./logger.h"
#include "./cuda_utils.h"
#include "../base_engine.h"

#include <numeric>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "device_launch_parameters.h"
#include "NvUtils.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferVersion.h"


class TensorRT_Engine:public BaseEngine{
public:
    /**
     * @brief 这是TensorRT推理引擎类的构造函数
     * @param data 模型权重数组
     * @param size 模型权重数组长度
     * @param gpu_id gpu设备号
     */
    TensorRT_Engine(const void* data, int64_t size, int gpu_id);

    /**
     * @brief 这是TensorRT推理引擎类的构造函数
     * @param tensorrt_model_path tensorrt模型路径
     * @param gpu_id gpu设备号
     */
    TensorRT_Engine(std::string tensorrt_model_path,int gpu_id);

    /**
     * @brief 这是TensorRT推理引擎类的析构函数
     */
    ~TensorRT_Engine();

    /**
     * @brief 这是TensorRT推理引擎类的前向推理函数
     * @param input_tensor 输入张量二维数组
     * @param output_tensor 输出张量二维数组
     */
    void inference(std::vector<float*>& input_tensor,std::vector<float*>& output_tensor);

    /**
     * @brief 这是获取TensorRT模型输入维度的函数,不包括batchsize
     * @return 不包括batchsize的TensorRT模型输入维度数组
     */
    std::vector<std::vector<int>> get_input_shapes();

    /**
     * @brief 这是获取TensorRT模型输出维度的函数,不包括batchsize
     * @return 不包括batchsize的TensorRT模型输出维度数组
     */
    std::vector<std::vector<int>> get_output_shapes();

    /**
     * @brief 这是获取TensorRT模型输入大小的函数,不包括batchsize
     * @return 不包括batchsize的TensorRT模型输入大小数组
     */
    std::vector<int> get_input_sizes();

    /**
     * @brief 这是获取TensorRT模型输出大小的函数,不包括batchsize
     * @return 不包括batchsize的TensorRT模型输出大小数组
     */
    std::vector<int> get_output_sizes();

    /**
     * @brief 这是获取TensorRT模型输入节点名称的函数
     * @return TensorRT模型输入节点名称数组
     */
    std::vector<std::string> get_input_names();

    /**
     * @brief 这是获取TensorRT模型输出节点名称的函数
     * @return TensorRT模型输出节点名称数组
     */
    std::vector<std::string> get_output_names();

    /**
     * @brief 这是获取TensorRT模型batchsize大小的函数
     * @return TensorRT模型batchsize
     */
    int get_batch_size();

protected:
    std::string tensorrt_model_path;                // tensorrt模型文件路径
    std::vector<DataType> input_types;              // 输入节点数据类型数组
    std::vector<DataType> output_types;             // 输出节点数据类型数组
    int gpu_id;

private:
    // tensorrt参数
    Logger logger;                                  // 日志类实例
    IRuntime *runtime;                              // tensorrt运行实例指针
    ICudaEngine *engine;                            // tensorrt模型引擎指针
    IExecutionContext *context;                     // tensorrt模型上下文指针
    cudaStream_t stream;                            // cuda流

    /**
     * @brief 这是加载TensorRT模型的函数
     * @param tensorrt_model_path tensorrt模型路径
     * @param gpu_id gpu设备号
     * @return 是否成功加载TensorRT模型布尔量
     */
    bool load_engine(std::string tensorrt_model_path,int gpu_id);

    /**
     * @brief 这是申请TensorRT推理引擎所需预留空间的函数
     */
    void prepare_buffers();
};

#endif //YOLO_DEPLOY_CXX_TENSORRT_ENGINE_H

//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_TENSORRT_ENGINE_H
#define YOLO_DEPLOY_CXX_TENSORRT_ENGINE_H

#include "mutex"
#include "vector"
#include "string"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "device_launch_parameters.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferVersion.h"

#include "logger.h"
#include "base_engine.h"
#include "tensorrt_utils.h"

using namespace nvinfer1;

class TensorRTEngine:public BaseEngine{
public:
    /**
     * @brief 这是TensorRT推理引擎类的构造函数
     * @param tensorrt_model_path tensorrt模型路径
     * @param gpu_id gpu设备号
     */
    TensorRTEngine(std::string tensorrt_model_path,int gpu_id);

    /**
     * @brief 这是TensorRT推理引擎类的析构函数
     */
    ~TensorRTEngine();

    /**
     * @brief 这是TensorRT推理引擎类的前向推理函数
     * @param input_tensor 输入张量二维数组
     * @param output_tensor 输出张量二维数组
     */
    void inference(std::vector<float*>& input_tensor,std::vector<float*>& output_tensor);

    /**
     * @brief 这是抽象推理引擎类的前向推理函数
     * @param input_tensor 输入张量数组
     * @param output_tensor 输出张量二维数组
     */
    void inference(float* input_tensor,std::vector<float*>& output_tensor);

    /**
     * @brief 这是抽象推理引擎类的前向推理函数
     * @param input_tensor 输入张量数组
     * @param output_tensor 输出张量数组
     */
    void inference(float* input_tensor,float* output_tensor);

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

    /**
     * @brief 这是判断模型输入是否为nchw格式的函数
     * @return 模型输入是否为nchw格式标志位
     */
    int get_is_nchw();



private:
    // tensorrt参数
    std::string tensorrt_model_path;                // tensorrt模型文件路径
    std::vector<std::string> input_names;           // 输入节点名称数组
    std::vector<std::string> output_names;          // 输出节点名称数组
    std::vector<std::vector<int>> input_shapes;     // 输出节点形状数组
    std::vector<std::vector<int>> output_shapes;    // 输出节点形状数组
    std::vector<int> input_sizes;                   // 输出节点形状数组
    std::vector<int> output_sizes;                  // 输出节点形状数组
    int batch_size;                                 // 小批量数据规模
    int gpu_id;                                     // gpu设备号
    int is_nchw;
    Logger logger;                                  // 日志类实例
    IRuntime *runtime;                              // tensorrt运行实例指针
    ICudaEngine *engine;                            // tensorrt模型引擎指针
    IExecutionContext *context;                     // tensorrt模型上下文指针
    cudaStream_t stream;                            // cuda流
    std::vector<void*> buffers;                     // tensorRT输出输入缓存数组
    std::mutex g_mutex;

    /**
     * @brief 这是加载TensorRT模型的函数
     * @param tensorrt_model_path tensorrt模型路径
     * @param gpu_id gpu设备号
     * @return 是否成功加载TensorRT模型布尔量
     */
    bool load_engine(std::string tensorrt_model_path,int gpu_id);

    /**
     * @brief 这是初始化TensorRT相关缓存和节点信息的函数
     */
    void init();
};

#endif //YOLO_DEPLOY_CXX_TENSORRT_ENGINE_H

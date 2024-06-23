//
// Created by dpw on 24-2-29.
//

#include "iostream"
#include <ostream>
#include <fstream>
#include "numeric"
#include "vector"

#include "spdlog/spdlog.h"
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "device_launch_parameters.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferVersion.h"

#include "common_utils.h"
#include "tensorrt_utils.h"
#include "tensorrt_engine.h"

/**
 * @brief 这是TensorRT推理引擎类的构造函数
 * @param tensorrt_model_path tensorrt模型路径
 * @param gpu_id gpu设备号
 */
TensorRTEngine::TensorRTEngine(std::string tensorrt_model_path,int gpu_id) {
    // 判断tensorrt模型是否存在
    assert(is_file_exists(this->tensorrt_model_path) == false);
    // 初始化相关参数
    this->tensorrt_model_path = tensorrt_model_path;
    this->gpu_id = gpu_id;

    // 加载tensorrt模型
    bool flag = this->load_engine(this->tensorrt_model_path,this->gpu_id);
    if(flag){               // 加载tensorRT模型成功,分配缓存
        this->init();
    }
}

/**
 * @brief 这是TensorRT推理引擎类的析构函数
 */
TensorRTEngine::~TensorRTEngine() {
    // 释放相关数组
    std::vector<std::string>().swap(this->input_names);
    std::vector<std::string>().swap(this->output_names);
    std::vector<std::vector<int>>().swap(this->input_shapes);
    std::vector<std::vector<int>>().swap(this->output_shapes);
    std::vector<int>().swap(input_sizes);
    std::vector<int>().swap(output_sizes);

    // 释放cuda流及其缓存
    cudaStreamDestroy(this->stream);
    for(int i = 0 ; i < this->buffers.size() ; i++){
        CUDA_CHECK(cudaFree(this->buffers[i]));
    }
    std::vector<void*>().swap(this->buffers);
    // 释放tensort引擎
#if NV_TENSORRT_MAJOR >= 8
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#else
    delete this->context;
    delete this->engine;
    delete this->runtime;
#endif
}

/**
 * @brief 这是加载TensorRT模型的函数
 * @param tensorrt_model_path tensorrt模型路径
 * @param gpu_id gpu设备号
 * @return 是否成功加载TensorRT模型布尔量
 */
bool TensorRTEngine::load_engine(std::string tensorrt_model_path, int gpu_id) {
    // 分配GPU设备
    cudaSetDevice(gpu_id);

    // 初始化tensorRT默认插件
    initLibNvInferPlugins(&this->logger, "");

    // 从文件中加载tensorrrt模型
    char *trtModelStream{nullptr};
    size_t size{0};
    bool flag = true;           // 模型加载是否成功状态变量
    if (is_file_exists(tensorrt_model_path)) {       // tensorrt模型存在，则进行加载模型生成tensorrt的引擎实例
        std::ifstream file(tensorrt_model_path, std::ios::binary);
        if (file.good()) {               // 二进制读取tensorRT模型文件内容
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            file.read(trtModelStream, size);
            file.close();
        } else {
            spdlog::error("{}无法读取，请检查tensorRT文件是否正常",tensorrt_model_path);
            flag = false;
        }
        // 创建TensorRT引擎
        this->runtime = createInferRuntime(this->logger);
        assert(this->runtime != nullptr);
        this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(this->engine != nullptr);
        this->context = this->engine->createExecutionContext();
        assert(this->context != nullptr);
        delete[] trtModelStream;                        // 释放资源
        spdlog::debug("从{}加载tensorRT模型文件成功",tensorrt_model_path);
    } else {            // 模型文件不存在则提示
        this->context = nullptr;
        this->engine = nullptr;
        spdlog::debug("{}文件不存在，请重新生成tensorRT模型",tensorrt_model_path);
        flag = false;
    }
    return flag;
}

/**
 * @brief 这是初始化TensorRT相关缓存和节点信息的函数
 */
void TensorRTEngine::init() {
    // 创建cuda流
    CUDA_CHECK(cudaStreamCreate(&this->stream));

    // 遍历所有输入输出节点
    for (int i = 0; i < this->engine->getNbBindings(); ++i) {
        // 获取输入输出节点信息
        std::string name = std::string(this->engine->getBindingName(i));        // 节点名称
        const Dims dimensions = this->engine->getBindingDimensions(i);          // 节点数据维度
        const DataType data_type = this->engine->getBindingDataType(i);         // 节点数据类型
        int data_type_size = get_tensor_data_type_size(data_type);              // 节点数据类型大小
        int size = get_tensor_size(dimensions);                                 // 节点张量大小
        std::vector<int> shape = get_tensor_shape(dimensions);                  // 节点张量形状

        // 初始化TensorRT推理引擎输入输出缓存数组
        this->buffers.emplace_back(nullptr);
        CUDA_CHECK(cudaMalloc(&this->buffers[i],size*data_type_size));

        // 保存输入输出节点信息
        if(this->engine->bindingIsInput(i)){                        // 输入节点
            this->input_shapes.emplace_back(shape);
            this->input_sizes.emplace_back(size);
            this->input_names.emplace_back(name);
        } else{                                                     // 输出节点
            this->output_shapes.emplace_back(shape);
            this->output_sizes.emplace_back(size);
            this->output_names.emplace_back(name);
        }
    }

    // 初始化is_nchw
    if(this->input_shapes[0][1] <= 3){
        this->is_nchw = 1;
    } else{
        this->is_nchw = 0;
    }

    // 初始化batch_size
    this->batch_size = this->input_shapes[0][0];
}

/**
 * @brief 这是获取TensorRT模型输入维度的函数,不包括batchsize
 * @return 不包括batchsize的模型输入维度二维数组
 */
std::vector<std::vector<int>> TensorRTEngine::get_input_shapes() {
    return this->input_shapes;
}

/**
 * @brief 这是获取TensorRT模型输入大小的函数,不包括batchsize
 * @return 不包括batchsize的TensorRT模型输入大小数组
 */
std::vector<int> TensorRTEngine::get_input_sizes() {
    return this->input_sizes;
}

/**
 * @brief 这是获取TensorRT模型输出维度的函数,不包括batchsize
 * @return 不包括batchsize的模型输出维度二维数组
 */
std::vector<std::vector<int>> TensorRTEngine::get_output_shapes() {
    return this->output_shapes;
}

/**
 * @brief 这是获取TensorRT模型输出大小的函数,不包括batchsize
 * @return 不包括batchsize的TensorRT模型输出大小数组
 */
std::vector<int> TensorRTEngine::get_output_sizes() {
    return this->output_sizes;
}

/**
 * @brief 这是获取TensorRT模型输入节点名称的函数
 * @return TensorRT模型输入节点名称数组
 */
std::vector<std::string> TensorRTEngine::get_input_names() {
    return this->input_names;
}

/**
 * @brief 这是获取TensorRT模型输出节点名称的函数
 * @return TensorRT模型输出节点名称数组
 */
std::vector<std::string> TensorRTEngine::get_output_names() {
    return this->output_names;
}

/**
 * @brief 这是获取TensorRT模型batchsize大小的函数
 * @return TensorRT模型batchsize
 */
int TensorRTEngine::get_batch_size() {
    return this->batch_size;
}

/**
 * @brief 这是判断模型输入是否为nchw格式的函数
 * @return 模型输入是否为nchw格式标志位
 */
int TensorRTEngine::get_is_nchw() {
    return this->is_nchw;
}

/**
 * @brief 这是TensorRT推理引擎类的前向推理函数
 * @param input_tensor 输入张量二维数组
 * @param output_tensor 输出张量二维数组
 */
void TensorRTEngine::inference(std::vector<float*>& input_tensor,std::vector<float*>& output_tensor) {
    // 加锁，方便多线程运行
    std::lock_guard<std::mutex> lock(this->g_mutex);
    // 将输入数据input传到cuda缓存中
    int input_node_num = input_tensor.size();
    for(int i = 0 ; i < input_tensor.size(); i++){
        CUDA_CHECK(cudaMemcpyAsync(this->buffers[i], input_tensor[i],
                                   this->batch_size * this->input_sizes[i] * sizeof(float),
                                   cudaMemcpyHostToDevice, this->stream));
    }
    spdlog::debug("输入张量已传入TensorRT推理引擎缓冲区");

    // tensort前向推理
    this->context->enqueueV2(this->buffers.data(), this->stream, nullptr);
    spdlog::debug("TensorRT推理引擎完成模型前向推理");

    // 将cuda缓存中输出结果传到output数组中
    for(int i = 0 ; i < output_tensor.size() ; i++){
        CUDA_CHECK(cudaMemcpyAsync(output_tensor[i], this->buffers[i+input_tensor.size()],
                                   this->batch_size * this->output_sizes[i] * sizeof(float),
                                   cudaMemcpyDeviceToHost, this->stream));
    }
    spdlog::debug("输出张量已从TensorRT推理引擎缓冲区复制出来");
    cudaStreamSynchronize(stream);
}


/**
 * @brief 这是抽象推理引擎类的前向推理函数
 * @param input_tensor 输入张量数组
 * @param output_tensor 输出张量二维数组
 */
void TensorRTEngine::inference(float *input_tensor, std::vector<float *> &output_tensor) {
    std::vector<float*> _input_tensor = {input_tensor};
    this->inference(_input_tensor,output_tensor);
}

/**
 * @brief 这是抽象推理引擎类的前向推理函数
 * @param input_tensor 输入张量数组
 * @param output_tensor 输出张量数组
 */
void TensorRTEngine::inference(float *input_tensor, float *output_tensor) {
    std::vector<float*> _input_tensor = {input_tensor};
    std::vector<float*> _output_tensor = {output_tensor};
    this->inference(_input_tensor,_output_tensor);
}
//
// Created by dpw on 24-3-1.
//

#include "iostream"
#include <ostream>
#include <fstream>

#include "spdlog/spdlog.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"

#include "tensorrt_engine.h"
#include "../../utils/common_utils.h"


/**
 * @brief 这是TensorRT推理引擎类的构造函数
 * @param tensorrt_model_path tensorrt模型路径
 * @param gpu_id gpu设备号
 */
TensorRT_Engine::TensorRT_Engine(std::string tensorrt_model_path,int gpu_id)
{
    // 判断tensorrt模型是否存在
    assert(is_file_exists(this->tensorrt_model_path) == false);
    // 初始化相关参数
    this->tensorrt_model_path = tensorrt_model_path;
    this->gpu_id = gpu_id;

    // 加载tensorrt模型
    bool flag = this->load_engine(this->tensorrt_model_path,this->gpu_id);
    if(flag){               // 加载tensorRT模型成功,分配缓存
        this->prepare_buffers();
    }
}

/**
 * @brief 这是TensorRT推理引擎类的构造函数
 * @param data 模型权重数组
 * @param size 模型权重数组长度
 * @param gpu_id gpu设备号
 */
TensorRT_Engine::TensorRT_Engine(const void *data, int64_t size,int gpu_id)
{
    this->gpu_id = gpu_id;
    cudaSetDevice(this->gpu_id);                   // 分配GPU设备
    initLibNvInferPlugins(&this->logger, "");       // 初始化tensorRT默认插件
    // 创建TensorRT引擎
    this->runtime = createInferRuntime(this->logger);
    assert(this->runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(data, size);
    assert(this->engine != nullptr);
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);
    spdlog::debug("从头文件中加载tensorRT模型文件成功",tensorrt_model_path);
    this->prepare_buffers();    // 分配模型推理缓存空间
}

/**
 * @brief 这是TensorRT推理引擎类的析构函数
 */
TensorRT_Engine::~TensorRT_Engine()
{
    // 释放cuda流及其缓存
    cudaStreamDestroy(this->stream);
    for(int i = 0 ; i < this->engine->getNbBindings() ; i++){
        CUDA_CHECK(cudaFree(this->buffers[i]));
    }
    // 释放tensort引擎
#if NV_TENSORRT_MAJOR < 8
    this->context->destroy();
    this->engine->destroy();
#else
    delete this->context;
    delete this->engine;
#endif
}

/**
 * @brief 这是加载TensorRT模型的函数
 * @param tensorrt_model_path tensorrt模型路径
 * @param gpu_id gpu设备号
 * @return 是否成功加载TensorRT模型布尔量
 */
bool TensorRT_Engine::load_engine(std::string tensorrt_model_path, int gpu_id)
{
    cudaSetDevice(gpu_id);                           // 分配GPU设备
    initLibNvInferPlugins(&this->logger, "");       // 初始化tensorRT默认插件
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
 * @brief 这是申请TensorRT推理引擎所需预留空间的函数
 */
void TensorRT_Engine::prepare_buffers()
{
    CUDA_CHECK(cudaStreamCreate(&this->stream));
    const int numBindings = this->engine->getNbBindings();
    assert(numBindings == 2);
    //std::cout<<numBindings<<std::endl;
    for (int i = 0; i < numBindings; ++i) {
        std::string name = std::string(this->engine->getBindingName(i));
        const Dims dimensions = this->engine->getBindingDimensions(i);
        const DataType dataType = this->engine->getBindingDataType(i);
        int elem_byte = sizeof(float);
        switch (engine->getBindingDataType(i)) {
            case nvinfer1::DataType::kHALF:
                elem_byte = sizeof(float) / 2;
                break;
            default:
                elem_byte = sizeof(float);
                break;
        }
        std::vector<int> shape;
        int size = 1;
        for(int i = 0 ; i < dimensions.nbDims ; i++){
            if(i == 0){
                this->batch_size = dimensions.d[i];
            } else{
                shape.push_back(dimensions.d[i]);
                size *= dimensions.d[i];
            }
            //std::cout<<dimensions.d[i]<<" ";
        }
        //std::cout<<std::endl;
        // 初始化模型缓存空间
//        CUDA_CHECK(cudaMalloc((void **)&this->buffers[i],
//                              this->batch_size*size*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&this->buffers[i],
                              this->batch_size*size*elem_byte));
        if(this->engine->bindingIsInput(i)){            // 输入节点
            //std::cout<<i<<"节点为输入节点，节点名称为："<<name<<std::endl;
            this->input_shape = shape;
            this->input_size = size;
            this->input_type = dataType;
            this->input_name = name;
        } else{                                                   // 输出节点
            //std::cout<<i<<"节点为输出节点，节点名称为："<<name<<std::endl;
            this->output_shape = shape;
            this->output_size = size;
            this->output_type = dataType;
            this->output_name = name;
        }
    }
}

/**
 * @brief 这是获取TensorRT模型输入维度的函数,不包括batchsize
 * @return 不包括batchsize的模型输入维度
 */
std::vector<int> TensorRT_Engine::get_input_shape()
{
    return this->input_shape;
}

/**
 * @brief 这是获取TensorRT模型输入大小的函数,不包括batchsize
 * @return 不包括batchsize的TensorRT模型输入大小
 */
int TensorRT_Engine::get_input_size()
{
    return this->input_size;
}

/**
 * @brief 这是获取TensorRT模型输出维度的函数,不包括batchsize
 * @return 不包括batchsize的模型输出维度
 */
std::vector<int> TensorRT_Engine::get_output_shape()
{
    return this->output_shape;
}

/**
 * @brief 这是获取TensorRT模型输出大小的函数,不包括batchsize
 * @return 不包括batchsize的TensorRT模型输出大小
 */
int TensorRT_Engine::get_output_size() {
    return this->output_size;
}

/**
 * @brief 这是获取TensorRT模型输入节点名称的函数
 * @return TensorRT模型输入节点名称
 */
std::string TensorRT_Engine::get_input_name()
{
    return this->input_name;
}

/**
 * @brief 这是获取TensorRT模型输出节点名称的函数
 * @return TensorRT模型输出节点名称
 */
std::string TensorRT_Engine::get_output_name()
{
    return this->output_name;
}

/**
 * @brief 这是获取TensorRT模型batchsize大小的函数
 * @return TensorRT模型batchsize
 */
int TensorRT_Engine::get_batch_size()
{
    return this->batch_size;
}

/**
 * @brief 这是TensorRT推理引擎类的前向推理函数
 * @param input_tensor 输入张量数组
 * @param output_tensor 输出张量数组
 */
void TensorRT_Engine::inference(float *input_tensor, float *output_tensor)
{
    std::lock_guard<std::mutex> lock(this->g_mutex);
//    const int input_index = this->engine->getBindingIndex(this->input_name.c_str());
//    const int output_index = this->engine->getBindingIndex(this->output_name.c_str());
    // 将输入数据input传到cuda缓存中
    CUDA_CHECK(cudaMemcpyAsync(this->buffers[0], input_tensor,
                               this->batch_size * this->input_size * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    //this->context->enqueue(this->batch_size, (void **)this->buffers, this->stream, nullptr);
    this->context->enqueueV2((void **)this->buffers, this->stream, nullptr);
    // 将cuda缓存中输出结果传到output数组中
    CUDA_CHECK(cudaMemcpyAsync(output_tensor, this->buffers[1],
                               this->batch_size * this->output_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


//
// Created by dpw on 24-3-1.
//

#include "iostream"
#include <ostream>
#include <fstream>
#include "numeric"

#include "spdlog/spdlog.h"
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "device_launch_parameters.h"
#include "NvUtils.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferVersion.h"

#include "tensorrt_engine.h"
#include "cuda_utils.h"
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
    //assert(numBindings == 2);
    //std::cout<<numBindings<<std::endl;
    // 遍历所有输入输出节点
    for (int i = 0; i < numBindings; ++i) {
        std::string name = std::string(this->engine->getBindingName(i));        // 节点名称
        const Dims dimensions = this->engine->getBindingDimensions(i);          // 节点数据维度
        const DataType dataType = this->engine->getBindingDataType(i);          // 节点数据类型
        int elem_byte = sizeof(float);                                          // 节点数据类型大小
        switch (engine->getBindingDataType(i)) {
            case nvinfer1::DataType::kHALF:
                elem_byte = sizeof(float) / 2;
                break;
            default:
                elem_byte = sizeof(float);
                break;
        }
        // 初始化节点shape(不包括batchsize)和大小
        std::vector<int> shape;
        for(int j = 0 ; j < dimensions.nbDims ; j++){
            if(j == 0){
                this->batch_size = dimensions.d[j];
            } else{
                shape.emplace_back(dimensions.d[j]);
            }
        }
        int size = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int64_t>());
        // 初始化模型缓存空间
        CUDA_CHECK(cudaMalloc(&this->buffers[i],this->batch_size*size*elem_byte));
        // 初始化输入输出节点信息
        if(this->engine->bindingIsInput(i)){            // 输入节点
            //std::cout<<i<<"节点为输入节点，节点名称为："<<name<<std::endl;
            this->input_shapes.emplace_back(shape);
            this->input_sizes.emplace_back(size);
            this->input_types.emplace_back(dataType);
            this->input_names.emplace_back(name);
        } else{                                                   // 输出节点
            //std::cout<<i<<"节点为输出节点，节点名称为："<<name<<std::endl;
            this->output_shapes.emplace_back(shape);
            this->output_sizes.emplace_back(size);
            this->output_types.emplace_back(dataType);
            this->output_names.emplace_back(name);
        }
    }
}

/**
 * @brief 这是获取TensorRT模型输入维度的函数,不包括batchsize
 * @return 不包括batchsize的模型输入维度二维数组
 */
std::vector<std::vector<int>> TensorRT_Engine::get_input_shapes()
{
    return this->input_shapes;
}

/**
 * @brief 这是获取TensorRT模型输入大小的函数,不包括batchsize
 * @return 不包括batchsize的TensorRT模型输入大小数组
 */
std::vector<int> TensorRT_Engine::get_input_sizes()
{
    return this->input_sizes;
}

/**
 * @brief 这是获取TensorRT模型输出维度的函数,不包括batchsize
 * @return 不包括batchsize的模型输出维度二维数组
 */
std::vector<std::vector<int>> TensorRT_Engine::get_output_shapes()
{
    return this->output_shapes;
}

/**
 * @brief 这是获取TensorRT模型输出大小的函数,不包括batchsize
 * @return 不包括batchsize的TensorRT模型输出大小数组
 */
std::vector<int> TensorRT_Engine::get_output_sizes() {
    return this->output_sizes;
}

/**
 * @brief 这是获取TensorRT模型输入节点名称的函数
 * @return TensorRT模型输入节点名称数组
 */
std::vector<std::string> TensorRT_Engine::get_input_names()
{
    return this->input_names;
}

/**
 * @brief 这是获取TensorRT模型输出节点名称的函数
 * @return TensorRT模型输出节点名称数组
 */
std::vector<std::string> TensorRT_Engine::get_output_names()
{
    return this->output_names;
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
 * @param input_tensor 输入张量二维数组
 * @param output_tensor 输出张量二维数组
 */
void TensorRT_Engine::inference(std::vector<float*>& input_tensor,std::vector<float*>& output_tensor)
{
    std::lock_guard<std::mutex> lock(this->g_mutex);
//    const int input_index = this->engine->getBindingIndex(this->input_name.c_str());
//    const int output_index = this->engine->getBindingIndex(this->output_name.c_str());
    // 将输入数据input传到cuda缓存中
    int input_node_num = input_tensor.size();
    for(int i = 0 ; i < input_tensor.size(); i++){
        CUDA_CHECK(cudaMemcpyAsync(this->buffers[i], input_tensor[i],
                                   this->batch_size * this->input_sizes[i] * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    spdlog::debug("输入数据张量已传入TensorRT推理引擎缓冲区");
    //this->context->enqueue(this->batch_size, (void **)this->buffers, this->stream, nullptr);
    this->context->enqueueV2(this->buffers.data(), this->stream, nullptr);
    spdlog::debug("TensorRT推理引擎完成模型前向推理");
    // 将cuda缓存中输出结果传到output数组中
    for(int i = 0 ; i < output_tensor.size() ; i++){
        CUDA_CHECK(cudaMemcpyAsync(output_tensor[i], this->buffers[input_node_num+i],
                                   this->batch_size * this->output_sizes[i] * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    spdlog::debug("输出数据张量已传入TensorRT推理引擎缓冲区");
    cudaStreamSynchronize(stream);
}


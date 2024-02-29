//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_BASE_ENGINE_H
#define YOLO_DEPLOY_CXX_BASE_ENGINE_H

#include "mutex"
#include "string"
#include "vector"

class BaseEngine{
public:
    /**
     * @brief 这是抽象推理引擎类的构造函数
     */
    BaseEngine(){};

    /**
     * @brief 这是抽象推理引擎类的析构函数
     */
    virtual ~BaseEngine(){};

    /**
     * @brief 这是抽象推理引擎类的前向推理函数
     * @param input_tensor 输入张量二维数组
     * @param output_tensor 输出张量二维数组
     */
    virtual void inference(std::vector<float*>& input_tensor,std::vector<float*>& output_tensor);

    /**
     * @brief 这是获取模型输入维度的函数,不包括batchsize
     * @return 不包括batchsize的模型输入维度数组
     */
    virtual std::vector<std::vector<int>> get_input_shape();

    /**
     * @brief 这是获取模型输出维度的函数,不包括batchsize
     * @return 不包括batchsize的模型输出维度数组
     */
    virtual std::vector<std::vector<int>> get_output_shape();

    /**
     * @brief 这是获取模型输入大小的函数,不包括batchsize
     * @return 不包括batchsize的模型输入大小数组
     */
    virtual std::vector<int> get_input_size();

    /**
     * @brief 这是获取模型输出大小的函数,不包括batchsize
     * @return 不包括batchsize的模型输出大小数组
     */
    virtual std::vector<int> get_output_size();

    /**
     * @brief 这是获取模型输入节点名称的函数
     * @return 模型输入节点名称数组
     */
    virtual std::vector<std::string> get_input_name();

    /**
     * @brief 这是获取模型输出节点名称的函数
     * @return 模型输出节点名称数组
     */
    virtual std::vector<std::string> get_output_name();

    /**
     * @brief 这是获取模型batchsize大小的函数
     * @return 模型batchsize
     */
    virtual int get_batch_size();

private:
    std::vector<std::string> input_names;                     // 模型输入节点名称数组
    std::vector<std::string> output_names;                    // 模型输出节点名称数组
    std::vector<std::vector<int>> input_shapes;               // 模型输入节点形状数组，不包括bachsize
    std::vector<std::vector<int>> output_shapes;              // 模型输出节点形状数组，不包括bachsize
    std::vector<int> input_sizes;                             // 模型输入节点大小数组，不包括bachsize
    std::vector<int> output_sizes;                            // 模型输出节点大小数组，不包括bachsize
    int batch_size;                                           // 模型batchsize
    std::vector<void*> input_buffers;                         // 模型输入缓存数组
    std::vector<void*> output_buffers;                        // 模型输出缓存数组
    std::mutex g_mutex;                                       // 互斥锁
};

#endif //YOLO_DEPLOY_CXX_BASE_ENGINE_H

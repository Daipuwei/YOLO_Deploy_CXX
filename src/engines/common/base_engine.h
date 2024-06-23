//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_BASE_ENGINE_H
#define YOLO_DEPLOY_CXX_BASE_ENGINE_H

#include "string"
#include "vector"

class BaseEngine{
public:
    /**
     * @brief 这是抽象推理引擎类的构造函数
     */
    BaseEngine(){};

    /**
     * @brief 这是抽象推理引擎类的前向推理函数
     * @param input_tensor 输入张量二维数组
     * @param output_tensor 输出张量二维数组
     */
    virtual void inference(std::vector<float*>& input_tensor,std::vector<float*>& output_tensor)=0;

    /**
     * @brief 这是抽象推理引擎类的前向推理函数
     * @param input_tensor 输入张量数组
     * @param output_tensor 输出张量二维数组
     */
    virtual void inference(float* input_tensor,std::vector<float*>& output_tensor)=0;

    /**
     * @brief 这是抽象推理引擎类的前向推理函数
     * @param input_tensor 输入张量数组
     * @param output_tensor 输出张量数组
     */
    virtual void inference(float* input_tensor,float* output_tensor)=0;

    /**
     * @brief 这是获取模型输入维度的函数,不包括batchsize
     * @return 不包括batchsize的模型输入维度二维数组
     */
    virtual std::vector<std::vector<int>> get_input_shapes()=0;

    /**
     * @brief 这是获取模型输出维度的函数,不包括batchsize
     * @return 不包括batchsize的模型输出维度二维数组
     */
    virtual std::vector<std::vector<int>> get_output_shapes()=0;

    /**
     * @brief 这是获取模型输入大小的函数,不包括batchsize
     * @return 不包括batchsize的模型输入大小数组
     */
    virtual std::vector<int> get_input_sizes()=0;

    /**
     * @brief 这是获取模型输出大小的函数,不包括batchsize
     * @return 不包括batchsize的模型输出大小数组
     */
    virtual std::vector<int> get_output_sizes()=0;

    /**
     * @brief 这是获取模型batchsize大小的函数
     * @return 模型batchsize
     */
    virtual int get_batch_size()=0;

    /**
     * @brief 这是判断模型输入是否为nchw格式的函数
     * @return 模型输入是否为nchw格式标志位
     */
    virtual int get_is_nchw()=0;
};

#endif //YOLO_DEPLOY_CXX_BASE_ENGINE_H

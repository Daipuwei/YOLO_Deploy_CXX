//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_BASE_DETECTION_MODEL_H
#define YOLO_DEPLOY_CXX_BASE_DETECTION_MODEL_H

#include "string"
#include "vector"
#include "detection_common.h"

class BaseDetectionModel{
public:
    /**
     * @brief 这是抽象检测模型类的构造函数
     * @param model_path 模型文件路径
     * @param class_names_path 目标名称txt文件路径
     * @param iou_threshold iou阈值
     * @param confidence_threshold 置信度阈值
     * @param gpu_id gpu设备号
     * @param export_time 是否输出时间标志位
     */
    BaseDetectionModel(std::string model_path,std::string class_names_path,
                       float iou_threshold,float confidence_threshold,int gpu_id,int export_time);

    /**
     * @brief 这是基础检测模型类的析构函数
     */
    virtual ~BaseDetectionModel(){};


    /**
     * @brief 这是检测单张图像的函数
     * @param image 输入图像
     * @return 检测结果结构体数组
     */
    virtual std::vector<Detection_Result> detect(cv::Mat image){};

    /**
     * @brief 这是检测单张图像的函数
     * @param images 输入图像数组
     * @return 检测结果结构体二维数组
     */
    virtual std::vector<Detection_Result> detect(std::vector<cv::Mat> images){};


    /**
     * @brief 这是检测视频的函数
     * @param video_path 输入视频路径
     * @param result_path 检测结果视频路径
     * @param interval 抽帧间隔
     */
    virtual void detect(std::string video_path,std::string result_path,float interval){};

private:
    // 模型初始化参数
    TensorRT_Engine* engine;
    std::vector<std::string> label_names;
    std::vector<cv::Scalar> colors;                     // 目标rgb颜色数组
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int label_names_size;
    float iou_threshold;                                // iou阈值
    float confidence_threshold;                         // 置信度阈值
    int batch_size;
    int input_size;
    int output_size;
    float* input;
    float* output;
    int export_time;
    std::vector<double> preprocess_time_array;
    std::vector<double> inference_time_array;
    std::vector<double> postprocess_time_array;
    std::vector<double> recognition_time_array;
};

#endif //YOLO_DEPLOY_CXX_BASE_DETECTION_MODEL_H

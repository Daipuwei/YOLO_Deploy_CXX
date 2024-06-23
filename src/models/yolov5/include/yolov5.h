//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_YOLOV5_H
#define YOLO_DEPLOY_CXX_YOLOV5_H

#include "memory"
#include "iostream"
#include "string"
#include "vector"

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "yolov5_postprocess.h"
#include "yolov5_preprocess.h"
#include "base_detector.h"
#include "detection_common.h"

#define PRINT_RESULTS 1      // 0代表不打印检测结果，1代表打印检测结果

class YOLOv5:public BaseDetector {
public:
    /**
     * @brief 这是YOLOv5的构造函数
     */
    YOLOv5(){};

    /**
     * @brief 这是YOLOv5的构造函数
     * @param model_path 模型路径
     * @param class_names_txt_path 目标名称txt文件路径
     * @param confidence_threshold 置信度阈值
     * @param iou_threshold iou阈值
     * @param gpu_id gpu设备号
     * @param export_time 是否输出时间
     */
    YOLOv5(std::string model_path,std::string class_names_txt_path,
           float confidence_threshold, float iou_threshold,int gpu_id,int export_time);

    /**
     * @brief 这是YOLOv5的析构函数
     */
    ~YOLOv5();

    /**
     * @brief 这是YOLOv5检测单张图像的函数
     * @param image 图像
     * @return 检测结果数组
     */
    std::vector<DetectionResult> detect(cv::Mat image);

    /**
     * @brief 这是YOLOv5检测批量图像的函数
     * @param image 图像数组
     * @return 检测结果二维数组
     */
    std::vector<std::vector<DetectionResult>> detect(std::vector<cv::Mat> images);

    /**
     * @brief 这是YOLOv5检测视频的函数
     * @param video_path 视频地址
     * @param result_video_path 检测结果视频地址
     * @param interval 抽帧间隔频率，默认为-1,代表逐帧检测，1代表隔秒检测
     */
    void detect(std::string video_path,std::string result_video_path,float interval);

    /**
     * @brief 这是YOLOv5获取目标类别RGB颜色数组的函数
     * @return RGB颜色数组
     */
    std::vector<cv::Scalar> get_colors();

    /**
     * @brief 这是YOLOv5获取模型各个阶段推理速度的函数
     * @return 各个阶段推理时间数组
     */
    std::vector<double> get_model_speed();

private:
    // 模型相关参数
    int label_names_size;                                   // 目标个数
    std::vector<std::string> label_names;                   // 目标名称数组
    std::vector<cv::Scalar> colors;                         // 目标RGB颜色数组
    std::shared_ptr<YOLOv5PreProcessor> pre_processor;      // 预处理类指针
    std::shared_ptr<YOLOv5PostProcessor> post_processor;    // 后处理类指针
    void* engine= nullptr;                                  // 推理引擎指针

    // 输入输出相关变量
    int batch_size;
    int input_channel;
    int input_height;
    int input_width;
    int single_input_size;
    int single_output_size;
    float* input;
    float* output;
    bool is_nchw;

    // 统计模型各个阶段时间的全局变量
    int export_time;
    std::vector<double> DETECTION_MODEL_PREPROCESS_TIME_ARRAY;
    std::vector<double> DETECTION_MODEL_INFERENCE_TIME_ARRAY;
    std::vector<double> DETECTION_MODEL_POSTPROCESS_TIME_ARRAY;
    std::vector<double> DETECTION_MODEL_RECOGNITION_TIME_ARRAY;

private:
    /**
     * @brief 这是YOLOv5图像预处理的函数
     * @param images 图像数组
     * @param image_shapes 图像尺度数组
     */
    void preprocess(std::vector<cv::Mat> images, std::vector<std::vector<int>>& image_shapes);

    /**
     * @brief 这是YOLOv5模型的前向推理函数
     */
    void inference();

    /**
     * @brief 这是检测结果后处理的函数
     * @param image_shapes 图像尺度数组
     * @param detection_results 检测结果二维数组
     */
    void postprocess(std::vector<std::vector<int>>& image_shapes,
                     std::vector<std::vector<DetectionResult>>& detection_results);
};


#endif //YOLO_DEPLOY_CXX_YOLOV5_H

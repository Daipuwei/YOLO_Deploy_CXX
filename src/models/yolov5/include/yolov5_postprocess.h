//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_YOLOV5_POSTPROCESS_H
#define YOLO_DEPLOY_CXX_YOLOV5_POSTPROCESS_H

#include "vector"
#include "string"
#include "detection_common.h"

class YOLOv5PostProcessor{
public:
    /**
     * @brief 这是YOLOv5后处理类的构造函数
     * @param input_height 模型输入高度
     * @param input_width 模型输入宽度
     * @param confidence_threshold 置信度阈值
     * @param iou_threshold iou阈值
     * @param label_names 目标名称数组
     */
    YOLOv5PostProcessor(int input_height,int input_width,float output_size,float confidence_threshold,
                        float iou_threshold,std::vector<std::string> label_names);

    /**
     * @brief 这是YOLOv5后处理类的析构函数
     */
    ~YOLOv5PostProcessor();

    /**
     * @brief 这是YOLOv5对输出结果进行后处理的函数
     * @param outputs 模型输出结果数组
     * @param image_height 图像高度
     * @param image_width 图像宽度
     * @param detection_results 检测结果数组
     * @return 检测结果结构体数组
     */
    void postprocess(float* outputs,int image_height,int image_width,
                     std::vector<DetectionResult>& detection_results);

private:
    int input_height;                                   // 模型输入高度
    int input_width;                                    // 模型输入宽度
    int output_dim;                                     // 输出维度
    int output_size;                                    // 输出大小
    float confidence_threshold;                         // 置信度阈值
    float iou_threshold;                                // iou阈值
    std::vector<std::string> label_names;               // 目标名称数组

    /**
     * @brief 这是YOLOv5对输出结果进行解码的函数
     * @param outputs 模型输出结果数组
     * @param image_height 图像高度
     * @param image_width 图像宽度
     * @param detection_results 检测结果数组
     * @return 检测结果结构体数组
     */
    void decode(float* outputs,int image_height,int image_width,
                std::vector<DetectionResult>& detection_results);
};

#endif //YOLO_DEPLOY_CXX_YOLOV5_POSTPROCESS_H

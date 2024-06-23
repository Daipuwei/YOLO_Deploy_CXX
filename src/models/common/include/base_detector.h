//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_BASE_DETECTOR_H
#define YOLO_DEPLOY_CXX_BASE_DETECTOR_H

#include "string"
#include "vector"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "detection_common.h"

class BaseDetector{
public:
    /**
     * @brief 这是抽象检测类的析构函数
     */
    virtual ~BaseDetector(){};

    /**
     * @brief 这是抽象检测类的检测单张图像的函数
     * @param image 图像
     * @return
     */
    virtual std::vector<DetectionResult> detect(cv::Mat image)=0;

    /**
     * @brief 这是抽象检测类的检测批量图像的函数
     * @param images 图像数组
     * @return
     */
    virtual std::vector<std::vector<DetectionResult>> detect(std::vector<cv::Mat> image)=0;

    /**
     * @brief 这是抽象检测类检测视频的函数
     * @param video_path 视频地址
     * @param result_video_path 检测结果视频地址
     * @param interval 抽帧间隔频率，默认为-1,代表逐帧检测，1代表隔秒检测
     */
    virtual void detect(std::string video_path,std::string result_video_path,float interval)=0;

    /**
     * @brief 这是抽象检测类获取目标类别RGB颜色数组的函数
     * @return RGB颜色数组
     */
    virtual std::vector<cv::Scalar> get_colors()=0;

    /**
     * @brief 这是抽象检测类获取模型各个阶段推理速度的函数
     * @return 各个阶段推理时间数组
     */
    virtual std::vector<double> get_model_speed()=0;
};

#endif //YOLO_DEPLOY_CXX_BASE_DETECTOR_H

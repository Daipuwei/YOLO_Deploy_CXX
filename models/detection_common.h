//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_DETECTION_COMMON_H
#define YOLO_DEPLOY_CXX_DETECTION_COMMON_H

#ifndef EXPORT_API
#ifdef _WIN64
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__ ((visibility("default")))
#endif
#endif

#include "iostream"
#include "string"
#include "vector"
#include "opencv2/opencv.hpp"


// 自定义检测信息结构体
typedef struct EXPORT_API Detection_Result{
    float score;                            // 检测框得分
    std::vector<cv::Point2f> bbox;          // 检测框坐标，[(xmin,ymin),(xmax,ymax)]
    int cls_id;                             // 分类id
    std::string cls_name;                   // 分类名称
    Detection_Result(float score,std::vector<cv::Point2f> bbox,int cls_id,std::string cls_name){
        this->score = score;
        this->bbox = bbox;
        this->cls_id = cls_id;
        this->cls_name = cls_name;
    }
}Detection_Result;

#endif //YOLO_DEPLOY_CXX_DETECTION_COMMON_H

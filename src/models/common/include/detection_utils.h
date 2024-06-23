//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_DETECTION_UTILS_H
#define YOLO_DEPLOY_CXX_DETECTION_UTILS_H

#include "vector"
#include "algorithm"
#include "iostream"
#include "fstream"
#include "detection_common.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**
 * 这是截断数值的函数
 * @param val
 * @param min_value
 * @param max_value
 * @return
 */
float clip(float val,float min_value,float max_value);

/**
 * @brief 这是计算两个检测框之间iou的函数
 * @param detection_result1 检测框1
 * @param detection_result2 检测框2
 * @return 两个检测框之间的iou
 */
float iou(DetectionResult detection_result1,DetectionResult detection_result2);

/**
 * @brief 这是定义检测结果结构体之间的排序规则的函数
 * @param a 检测结果结构体a
 * @param b 检测结果结构体b
 * @return
 */
bool cmp(const DetectionResult & a, const DetectionResult & b);

/**
 * @brief 这是nms过滤冗余框(两阶段nms)的函数
 * @param detection_results 检测结果数组
 * @param iou_threshold iou阈值
 * @return 过滤后检测框数组
 */
std::vector<DetectionResult> nms(std::vector<DetectionResult> detection_results,float iou_threshold);

/**
 * @brief 这是nms过滤冗余框的函数
 * @param detection_results 检测结果数组
 * @param filter_detection_results 过滤后的检测数组
 * @param iou_threshold iou阈值
 * @return 过滤后检测框数组
 */
void non_maximum_suppression(std::vector<DetectionResult> detection_results,
                             std::vector<DetectionResult>& filter_detection_results,float iou_threshold);


/**
 * @brief 这是绘制检测结果的函数
 * @param image 图像
 * @param detection_results 检测结果数组
 * @param colors RGB颜色数组
 */
void draw_detection_results(cv::Mat &image,
                            std::vector<DetectionResult> detection_results,std::vector<cv::Scalar> colors);

#endif //YOLO_DEPLOY_CXX_DETECTION_UTILS_H

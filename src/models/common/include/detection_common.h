//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_DETECTION_COMMON_H
#define YOLO_DEPLOY_CXX_DETECTION_COMMON_H

#include "string"

#ifndef EXPORT_API
#ifdef _WIN64
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__ ((visibility("default")))
#endif

#endif


// 自定义检测结果结构体
typedef struct DetectionResult{
    int x1,y1,x2,y2;                // 检测框左上右下坐标
    float score;                    // 检测框得分
    int cls_id;                     // 分类id号
    std::string cls_name;           // 分类名称

    /**
     * @brief 这是检测结果结构体构造函数
     * @param x1 检测框左上点x坐标
     * @param y1 检测框左上点y坐标
     * @param x2 检测框右下点x坐标
     * @param y2 检测框右下点y坐标
     * @param score 检测框得分
     * @param cls_id 检测框分类id号
     * @param cls_name 检测框分类名称
     */
    DetectionResult(int x1, int y1, int x2, int y2, float score, int cls_id, std::string cls_name) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->score=score;
        this->cls_id = cls_id;
        this->cls_name = cls_name;
    }
}DetectionResult;


#endif //YOLO_DEPLOY_CXX_DETECTION_COMMON_H

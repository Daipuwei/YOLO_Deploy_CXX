//
// Created by dpw on 24-2-29.
//

#ifndef YOLO_DEPLOY_CXX_YOLOV5_PREPROCESS_H
#define YOLO_DEPLOY_CXX_YOLOV5_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "vector"

class YOLOv5PreProcessor{
public:
    /**
     * @brief 这是YOLOv5预处理类的初始化函数
     * @param input_height 模型输入高度
     * @param input_width 模型输出高度
     * @param input_channel 模型输入通道数
     * @param is_nchw 模型输入是否为nchw格式标志位
     */
    YOLOv5PreProcessor(int input_height,int input_width,int input_channel,int is_nchw);

    /**
     * @brief 这是YOLOv5预处理类的析构函数
     */
    ~YOLOv5PreProcessor();

    /**
     * @brief 这是YOLOv5的预处理函数
     * @param image 图像
     * @param image_shape 图像尺寸
     * @param input_buf 输入缓存
     */
    void preprocess(cv::Mat image,std::vector<int>& image_shape,uint8_t* input_buf);

    /**
     * @brief 这是YOLOv5的预处理函数
     * @param image 图像
     * @param image_shape 图像尺寸
     * @param input_buf 输入缓存
     */
    void preprocess(cv::Mat image,std::vector<int>& image_shape,int8_t* input_buf);

    /**
     * @brief 这是YOLOv5的预处理函数
     * @param image 图像
     * @param image_shape 图像尺寸
     * @param input_buf 输入缓存
     */
    void preprocess(cv::Mat image,std::vector<int>& image_shape,float* input_buf);

private:
    int input_height;
    int input_width;
    int input_channel;
    int is_nchw;
    int input_size;
//    std::vector<float> means;                               // 均值数组
//    std::vector<float> stds;                                // 标准差数值

};

#endif //YOLO_DEPLOY_CXX_YOLOV5_PREPROCESS_H

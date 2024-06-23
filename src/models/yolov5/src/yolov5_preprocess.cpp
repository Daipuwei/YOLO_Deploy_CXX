//
// Created by dpw on 24-2-29.
//

#include "spdlog/spdlog.h"
#include "yolov5_preprocess.h"
#include "opencv_image_utils.h"

/**
 * @brief 这是YOLOv5预处理类的初始化函数
 * @param input_height 模型输入高度
 * @param input_width 模型输出高度
 * @param input_channel 模型输入通道数
 * @param is_nchw 模型输入是否为nchw格式标志位
 */
YOLOv5PreProcessor::YOLOv5PreProcessor(int input_height, int input_width, int input_channel,int is_nchw) {
    this->input_height = input_height;
    this->input_width = input_width;
    this->input_channel = input_channel;
    this->is_nchw = is_nchw;
    this->input_size = this->input_height*this->input_width*this->input_channel;
}

/**
 * @brief 这是YOLOv5预处理类的析构函数
 */
YOLOv5PreProcessor::~YOLOv5PreProcessor() {
    this->input_height = 0;
    this->input_width = 0;
    this->input_channel = 0;
    this->input_size = 0;
    this->is_nchw = 0;
}


/**
 * @brief 这是YOLOv5的预处理函数
 * @param image 图像
 * @param image_shape 图像尺寸
 * @param input_buf 输入缓存
 */
void YOLOv5PreProcessor::preprocess(cv::Mat image, std::vector<int> &image_shape, float *input_buf) {
    // 初始化原始图像宽高
    int image_height = image.rows;
    int image_width = image.cols;
    int image_channel = image.channels();
    image_shape.emplace_back(image_height);
    image_shape.emplace_back(image_width);
    image_shape.emplace_back(image_channel);

    // 图像resize
    cv::Mat input_image;
    spdlog::debug("resize前图像高度为:{},图像宽度为{}",image_height,image_width);
    auto start_time = std::chrono::high_resolution_clock::now();
    input_image = resize_image(image,this->input_width,this->input_height);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count()*1.0/1000;
    spdlog::debug("resize后图像高度为:{},图像宽度为{},resize耗时:{}ms",this->input_height,this->input_width,time);

    // bgr2rgb
    start_time = std::chrono::high_resolution_clock::now();
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count()*1.0/1000;
    spdlog::debug("完成BGR转RGB操作,耗时:{}ms",time);

    // 图像归一化
    start_time = std::chrono::high_resolution_clock::now();
    input_image.convertTo(input_image, CV_32FC3,1./255);
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count()*1.0/1000;
    spdlog::debug("完成图像归一化操作,耗时:{}ms",time);

    // nhwc转nchw
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = hwc2chw(input_image);
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count()*1.0/1000;
    spdlog::debug("完成NHWC转NCHW操作,耗时:{}ms",time);

    // 复制到数组中
    memcpy(input_buf,&image_data[0],image_data.size()* sizeof(float));
    // 释放资源
    std::vector<float>().swap(image_data);
}


//
// Created by dpw on 24-2-29.
//

#include "math.h"
#include "spdlog/spdlog.h"

#include "common_utils.h"
#include "detection_utils.h"
#include "yolov5_postprocess.h"

/**
 * @brief 这是YOLOv5后处理类的构造函数
 * @param input_height 模型输入高度
 * @param input_width 模型输入宽度
 * @param confidence_threshold 置信度阈值
 * @param iou_threshold iou阈值
 * @param label_names 目标名称数组
 */
YOLOv5PostProcessor::YOLOv5PostProcessor(int input_height, int input_width, float output_size,float confidence_threshold,
                                         float iou_threshold,std::vector<std::string> label_names) {
    this->input_height = input_height;
    this->input_width = input_width;
    this->confidence_threshold = confidence_threshold;
    this->iou_threshold = iou_threshold;
    this->label_names = label_names;
    this->output_size = output_size;
    this->output_dim = 5+label_names.size();
}

/**
 * @brief 这是YOLOv5后处理类的析构函数
 */
YOLOv5PostProcessor::~YOLOv5PostProcessor() {
    this->input_height = 0;
    this->input_width = 0;
    this->confidence_threshold = 0;
    this->iou_threshold = 0;
    this->output_dim = 0;
    std::vector<std::string>().swap(this->label_names);
}


/**
 * @brief 这是YOLOv5对输出结果进行后处理的函数
 * @param outputs 模型输出结果二维数组
 * @param image_height 图像高度
 * @param image_width 图像宽度
 * @param detection_results 检测结果数组
 * @return 检测结果结构体数组
 */
void YOLOv5PostProcessor::postprocess(float *outputs, int image_height, int image_width,
                                      std::vector<DetectionResult> &detection_results) {
    // 首先对模型输出结果进行解码
    std::vector<DetectionResult> decode_detection_results;
    this->decode(outputs,image_height,image_width,decode_detection_results);
    // 进行nms操作
    detection_results = nms(decode_detection_results,this->iou_threshold);
}

/**
 * @brief 这是YOLOv5对输出结果进行解码的函数
 * @param outputs 模型输出结果数组
 * @param image_height 图像高度
 * @param image_width 图像宽度
 * @param detection_results 检测结果数组
 * @return 检测结果结构体数组
 */
void YOLOv5PostProcessor::decode(float *outputs, int image_height,
                                 int image_width,std::vector<DetectionResult>& detection_results) {
    int bbox_num = this->output_size / this->output_dim;
    float scale_h = this->input_height * 1.0 / image_height;
    float scale_w = this->input_width * 1.0 / image_width;
    float scale = std::min(scale_w,scale_h);
    if (scale >= 1){
        scale = 1;
    }
    // 遍历所有检测框，根据阈值进行过滤，并完成解码
    spdlog::debug("开始对模型结果进行解码操作");
    for(int i = 0; i < bbox_num ; i++){
        float box_confidence = outputs[i*this->output_dim+4];
        if(box_confidence > this->confidence_threshold){
            // 获取目标框坐标及其目标分类概率
            float x = outputs[i*this->output_dim];
            float y = outputs[i*this->output_dim+1];
            float w = outputs[i*this->output_dim+2];
            float h = outputs[i*this->output_dim+3];
            int x1 = (int)round(clip((x - w / 2) / scale,0,image_width));
            int y1 = (int)round(clip((y - h / 2) / scale,0,image_height));
            int x2 = (int)round(clip((x + w / 2) / scale,0,image_width));
            int y2 = (int)round(clip((y + h / 2) / scale,0,image_height));
            int max_id = argmax(&outputs[i*this->output_dim+5],&outputs[i*this->output_dim+5+this->label_names.size()]);
            float cls_prob = outputs[i*this->output_dim+5+max_id];
            float score = cls_prob*box_confidence;
            DetectionResult result(x1,y1,x2,y2,score,max_id,this->label_names[max_id]);
            detection_results.emplace_back(result);
        }
    }
    spdlog::debug("结束对模型结果进行解码操作");
}
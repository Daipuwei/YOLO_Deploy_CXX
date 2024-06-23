//
// Created by dpw on 24-2-29.
//

#include "algorithm"
#include "map"
#include "vector"
#include "detection_utils.h"

/**
 * 这是截断数值的函数
 * @param val
 * @param min_value
 * @param max_value
 * @return
 */
float clip(float val,float min_value,float max_value) {
    return val > min_value ? (val < max_value ? val : max_value) : min_value;
}

/**
 * @brief 这是计算两个检测框之间iou的函数
 * @param detection_result1 检测框1
 * @param detection_result2 检测框2
 * @return 两个检测框之间的iou
 */
float iou(DetectionResult detection_result1,DetectionResult detection_result2) {
    // 计算交集面积
    int inter_w = std::max(0,
                             std::min(detection_result1.x2, detection_result2.x2)
                             - std::max(detection_result2.x1, detection_result2.x1));
    int inter_h = std::max(0,
                             std::min(detection_result1.y2, detection_result2.y2)
                             - std::max(detection_result2.y1, detection_result2.y1));
    int inter_area = inter_w*inter_h;

    // 计算并集面积
    int bbox_area1 = (detection_result1.x2-detection_result1.x1)*(detection_result1.y2-detection_result1.y1);
    int bbox_area2 = (detection_result2.x2-detection_result2.x1)*(detection_result2.y2-detection_result2.y1);
    int union_area = bbox_area1 + bbox_area2 - inter_area;

    // 计算交并比
    float iou = inter_area*1.0 / union_area;
    return iou;
}

/**
 * @brief 这是定义检测结果结构体之间的排序规则的函数
 * @param a 检测结果结构体a
 * @param b 检测结果结构体b
 * @return
 */
bool cmp(const DetectionResult & a, const DetectionResult & b) {
    return a.score > b.score;
}

/**
 * @brief 这是nms过滤冗余框的函数
 * @param detection_results 检查结果数组
 * @param iou_threshold iou阈值
 * @return 过滤后检测框数组
 */
std::vector<DetectionResult> nms(std::vector<DetectionResult> detection_results,float iou_threshold) {
    if(detection_results.size() == 0){
        return detection_results;
    }
    // 按照类别划分检测结果
    std::map<int,std::vector<DetectionResult>> detection_result_map;
    for(int i = 0 ; i < detection_results.size() ; i++){
        if (detection_result_map.count(detection_results[i].cls_id) == 0) {
            detection_result_map.emplace(detection_results[i].cls_id, std::vector<DetectionResult>());
        }
        detection_result_map[detection_results[i].cls_id].push_back(detection_results[i]);
    }

    // 根据目标id分别检测框进行NMS
    std::vector<DetectionResult> filter_detection_results_stage1;
    for (auto it = detection_result_map.begin(); it != detection_result_map.end(); it++) {
        auto& _detection_results = it->second;
        non_maximum_suppression(_detection_results,filter_detection_results_stage1,iou_threshold);
    }
    return filter_detection_results_stage1;

//    // 第二次nms
//    std::vector<DetectionResult> filter_detection_results_stage2;
//    non_maximum_suppression(filter_detection_results_stage1,filter_detection_results_stage2,iou_threshold);
//    std::vector<DetectionResult>().swap(filter_detection_results_stage1);
//
//    return filter_detection_results_stage2;
}

/**
 * @brief 这是nms过滤冗余框的函数
 * @param detection_results 检测结果数组
 * @param iou_threshold iou阈值
 * @return 过滤后检测框数组
 */
void non_maximum_suppression(std::vector<DetectionResult> detection_results,std::vector<DetectionResult>& filter_detection_results,float iou_threshold){
    // 按照置信度对检测框进行从大到小排序
    std::sort(detection_results.begin(), detection_results.end(), cmp);
    for (size_t m = 0 ; m < detection_results.size(); ++m) {
        auto item = detection_results[m];
        filter_detection_results.emplace_back(item);
        for (size_t n = m + 1; n < detection_results.size(); ++n) {
            // 计算iou
            float _iou = iou(item, detection_results[n]);
            // iou大于阈值则将概况过滤掉
            if (_iou > iou_threshold) {
                detection_results.erase(detection_results.begin() + n);
                --n;
            }
        }
    }
}

/**
 * @brief 这是绘制检测结果的函数
 * @param image 图像
 * @param detection_results 检测结果数组
 * @param colors RGB颜色数组
 */
void draw_detection_results(cv::Mat &image,
                            std::vector<DetectionResult> detection_results,std::vector<cv::Scalar> colors) {
    // 遍历所有检测框，依次绘制结果
    for(int i = 0 ; i < detection_results.size() ; i++){
        // 初始化检测框信息
        std::string cls_name = detection_results[i].cls_name;
        int cls_id = detection_results[i].cls_id;
        float score = detection_results[i].score;
        int x1 = detection_results[i].x1;
        int y1 = detection_results[i].y1;
        int x2 = detection_results[i].x2;
        int y2 = detection_results[i].y2;
        int width = x2-x1;
        int height = y2-y1;
        cv::Scalar color = colors[cls_id];                          // 目标框的RGB颜色
        // 绘制检测框
        cv::rectangle(image, cv::Rect(x1,y1,width,height),color, 1);
        // 初始化文字及其坐标
        char tmp[256];
        sprintf(tmp, "%s %.2f%%", cls_name.c_str(),score*100);
        std::string text(tmp);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        // 绘制检测结果
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 128){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }
        int x_text = int(x1);
        int y_text = 0;
        if(y1-label_size.height < 0){
            y_text = int(y1)+1;
        }else{
            y_text = int(y1)-label_size.height-1;
            if (y_text > image.rows){
                y_text = image.rows;
            }
        }
        cv::rectangle(image,
                      cv::Rect(cv::Point(x_text, y_text),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);
        cv::putText(image, text, cv::Point(x_text, y_text+ label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
}

/**
 * @brief 这是加载下采样率数组的函数
 * @param stride_txt_path 下采样率文件路径
 * @return 下采样率数组
 */
std::vector<int> load_strides(std::string stride_txt_path) {
    // 读取下采样率文件中的数据
    std::ifstream in(stride_txt_path);
    std::string line;
    std::vector<int> strides;
    if (in) {
        while (getline(in, line)) {
            strides.emplace_back(std::stoi(line));
        }
    } else {
        std::cout << "no such stride txt file: " << stride_txt_path << ", exit the program..."
                  << std::endl;
        exit(1);
    }
    return strides;
}

/**
 * @brief 这是加载anchor尺度数组的函数
 * @param anchor_txt_path anchor尺度数组txt文件路径
 * @param output_num 模型预测分支数
 */
std::vector<std::vector<float>> load_anchors(std::string anchor_txt_path,int output_num) {
    // 读取anchor文件中的数据
    std::ifstream in(anchor_txt_path);
    std::string line;
    std::vector<float> tmp_anchors;
    if (in) {
        while (getline(in, line)) {
            tmp_anchors.emplace_back(std::stof(line));
        }
    } else {
        std::cout << "no such anchor txt file: " << anchor_txt_path << ", exit the program..."
                  << std::endl;
        exit(1);
    }

    // 转换格式
    std::vector<std::vector<float>> anchors;
    int len = tmp_anchors.size();
    int anchor_num = len / (output_num*2);
    int index = 0;
    for(int i = 0 ; i < output_num ; i++){
        std::vector<float> array(anchor_num*2);
        array.assign(tmp_anchors.begin()+index,tmp_anchors.begin()+index+anchor_num*2);
        anchors.emplace_back(array);
        index += anchor_num*2;
    }

    return anchors;
}

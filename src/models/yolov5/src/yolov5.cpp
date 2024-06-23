//
// Created by dpw on 24-2-29.
//

#include "spdlog/spdlog.h"

#include "yolov5.h"
#include "common_utils.h"
#include "opencv_image_utils.h"
#include "detection_utils.h"
#include "tensorrt_engine.h"

/**
 * @brief 这是YOLOv5的构造函数
 * @param model_path 模型路径
 * @param class_names_txt_path 目标名称txt文件路径
 * @param confidence_threshold 置信度阈值
 * @param iou_threshold iou阈值
 * @param gpu_id gpu设备号
 * @param export_time 是否输出时间
 */
YOLOv5::YOLOv5(std::string model_path,std::string class_names_txt_path,
               float confidence_threshold, float iou_threshold, int gpu_id,int export_time) {
    // 初始化推理引擎和目标名称数组
    this->engine = new TensorRTEngine(model_path,gpu_id);
    // 初始化目标名称及其对应RGB颜色数组
    this->label_names = read_label_dict(class_names_txt_path);
    this->label_names_size = this->label_names.size();
    this->colors = random_generate_colors(this->label_names_size);

    // 初始化模型参数
    this->export_time = export_time;

    // 初始化输入输出缓存
    std::vector<std::vector<int>> input_shapes = reinterpret_cast<TensorRTEngine*>(this->engine)->get_input_shapes();
    std::vector<int> output_sizes = reinterpret_cast<TensorRTEngine*>(this->engine)->get_output_sizes();
    this->is_nchw = reinterpret_cast<TensorRTEngine*>(this->engine)->get_is_nchw();
    if(this->is_nchw){
        this->batch_size = input_shapes[0][0];
        this->input_channel = input_shapes[0][1];
        this->input_height = input_shapes[0][2];
        this->input_width = input_shapes[0][3];
    } else{
        this->batch_size = input_shapes[0][0];
        this->input_height = input_shapes[0][1];
        this->input_width = input_shapes[0][2];
        this->input_channel = input_shapes[0][3];
    }
    this->single_input_size = this->input_width*this->input_height*this->input_channel;
    this->single_output_size = output_sizes[0] / this->batch_size;
    this->input = new float[this->single_input_size*this->batch_size];
    this->output = new float[this->single_output_size*this->batch_size];
    memset(this->input,0,this->single_input_size*this->batch_size*sizeof(float));
    memset(this->output,0,this->single_output_size*this->batch_size*sizeof(float));
    spdlog::debug("完成单张图像模型输出结果数组初始化");

    // 初始化输入输出
    this->pre_processor = std::make_shared<YOLOv5PreProcessor>
            (this->input_height,this->input_width,this->input_channel,is_nchw);
    this->post_processor = std::make_shared<YOLOv5PostProcessor>
            (this->input_height,this->input_width,this->single_output_size,
             confidence_threshold,iou_threshold,this->label_names);

    // 释放资源
    std::vector<std::vector<int>>().swap(input_shapes);
    std::vector<int>().swap(output_sizes);
}

/**
 * @brief 这是YOLOv5获取目标类别RGB颜色数组的函数
 * @return RGB颜色数组
 */
std::vector<cv::Scalar> YOLOv5::get_colors() {
    return this->colors;
}

/**
 * @brief 这是YOLOv5的析构函数
 */
YOLOv5::~YOLOv5(){
    std::vector<std::string>().swap(this->label_names);
    std::vector<cv::Scalar>().swap(this->colors);
    delete this->input;
    delete this->output;
    delete this->engine;
    std::vector<double>().swap(this->DETECTION_MODEL_PREPROCESS_TIME_ARRAY);
    std::vector<double>().swap(this->DETECTION_MODEL_INFERENCE_TIME_ARRAY);
    std::vector<double>().swap(this->DETECTION_MODEL_POSTPROCESS_TIME_ARRAY);
    std::vector<double>().swap(this->DETECTION_MODEL_RECOGNITION_TIME_ARRAY);
}

/**
 * @brief 这是YOLOv5检测单张图像的函数
 * @param image 图像
 * @return 检测结果数组
 */
std::vector<DetectionResult> YOLOv5::detect(cv::Mat image) {
    std::vector<cv::Mat> images = {image};
    std::vector<std::vector<DetectionResult>> detection_results = this->detect(images);
    return detection_results[0];
}

/**
 * @brief 这是YOLOv5检测批量图像的函数
 * @param image 图像数组
 * @return 检测结果二维数组
 */
std::vector<std::vector<DetectionResult>> YOLOv5::detect(std::vector<cv::Mat> images) {
    assert(images.size() <= this->batch_size);
    // 图像预处理
    std::vector<std::vector<int>> image_shapes;
    auto preprocess_start_time = std::chrono::high_resolution_clock::now();
    this->preprocess(images,image_shapes);
    auto preprocess_end_time = std::chrono::high_resolution_clock::now();
    auto preprocess_time =  std::chrono::duration_cast<std::chrono::microseconds>(
            preprocess_end_time-preprocess_start_time).count()*1.0/1000;
    spdlog::debug("YOLOv5的图像预处理时间为:{}ms",preprocess_time);

    // 模型前向推理
    auto inference_start_time = std::chrono::high_resolution_clock::now();
    this->inference();
    auto inference_end_time = std::chrono::high_resolution_clock::now();
    auto inference_time =  std::chrono::duration_cast<std::chrono::microseconds>(
            inference_end_time-inference_start_time).count()*1.0/1000;
//    printf("YOLOv5的RKNN前向推理时间为:%fms\n",inference_time);
    spdlog::debug("YOLOv5的TensorRT前向推理时间为:{}ms",inference_time);

    // 检测结果后处理
    std::vector<std::vector<DetectionResult>> detection_results;
    auto postprocess_start_time = std::chrono::high_resolution_clock::now();
    this->postprocess(image_shapes,detection_results);
    auto postprocess_end_time = std::chrono::high_resolution_clock::now();
    auto postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
            postprocess_end_time-postprocess_start_time).count()*1.0/1000;
    auto detection_time = preprocess_time+inference_time+postprocess_time;
    if(this->export_time){
        this->DETECTION_MODEL_PREPROCESS_TIME_ARRAY.emplace_back(preprocess_time);
        this->DETECTION_MODEL_INFERENCE_TIME_ARRAY.emplace_back(inference_time);
        this->DETECTION_MODEL_POSTPROCESS_TIME_ARRAY.emplace_back(postprocess_time);
        this->DETECTION_MODEL_RECOGNITION_TIME_ARRAY.emplace_back(detection_time);
    }
    spdlog::debug("YOLOv5的后处理时间为:{}ms",postprocess_time);
    spdlog::debug("YOLOv5的检测识别时间为:{}ms",detection_time);

    return detection_results;
}

/**
 * @brief 这是YOLOv5图像预处理的函数
 * @param images 图像数组
 * @param image_shapes 图像尺度数组
 */
void YOLOv5::preprocess(std::vector<cv::Mat> images, std::vector<std::vector<int>> &image_shapes) {
    // 遍历所有图像进行图像预处理
    for(int i = 0 ; i < images.size() ; i++){
        spdlog::debug("开始对第{}张图像进行预处理",i+1);
        std::vector<int> image_shape;
        this->pre_processor->preprocess(images[i],image_shape,this->input+i*this->single_input_size);
        image_shapes.emplace_back(image_shape);
    }
}

/**
 * @brief 这是YOLOv5模型的前向推理函数
 */
void YOLOv5::inference() {
    reinterpret_cast<TensorRTEngine*>(this->engine)->inference(this->input,this->output);
}

/**
 * @brief 这是检测结果后处理的函数
 * @param image_shapes 图像尺度数组
 * @param detection_results 检测结果二维数组
 */
void YOLOv5::postprocess(std::vector<std::vector<int>>& image_shapes,
                         std::vector<std::vector<DetectionResult>>& detection_results) {
    // 遍历所有图像输出结果，分别进行后处理
    for(int i = 0 ; i < image_shapes.size() ; i++){
        // 复制每张图像推理结果
        spdlog::debug("开始复制第{}张图像的模型推理结果",i+1);
        float* single_output = this->output+i*this->single_output_size;
        std::vector<DetectionResult> single_detection_results;
        this->post_processor->postprocess(single_output,image_shapes[i][0],image_shapes[i][1],single_detection_results);
        detection_results.emplace_back(single_detection_results);
    }

#if(PRINT_RESULTS)
    // 打印检测结果
    for(int i = 0 ; i < detection_results.size() ; i++){
        spdlog::debug("第{}张图像检测到{}个目标，检测结果如下:",i+1,detection_results[i].size());
        for(int j = 0 ; j < detection_results[i].size() ; j++){
            spdlog::debug("第{}个目标为:{},bbox:({},{}),({},{}),score={}",
                          j+1,detection_results[i][j].cls_name,detection_results[i][j].x1,detection_results[i][j].y1,
                          detection_results[i][j].x2,detection_results[i][j].y2,detection_results[i][j].score);
        }
    }
#endif
}

/**
 * @brief 这是YOLOv5获取模型各个阶段推理速度的函数
 * @return 各个阶段推理时间数组
 */
std::vector<double> YOLOv5::get_model_speed() {
    // 计算模型各个阶段的平均时延
    double preprocess_time = 0.0;
    double inference_time = 0.0;
    double postprocess_time = 0.0;
    double recognition_time = 0.0;
    std::vector<double> result_time_array;
    double cnt = 0.0;
    // 遍历向量组计算平均推理时间
    for(int i = 0 ; i < this->DETECTION_MODEL_PREPROCESS_TIME_ARRAY.size() ; i++) {
        preprocess_time += (this->DETECTION_MODEL_PREPROCESS_TIME_ARRAY[i] * 10000000000);
        inference_time += (this->DETECTION_MODEL_INFERENCE_TIME_ARRAY[i] * 10000000000);
        postprocess_time += (this->DETECTION_MODEL_POSTPROCESS_TIME_ARRAY[i] * 10000000000);
        recognition_time += (this->DETECTION_MODEL_RECOGNITION_TIME_ARRAY[i] * 10000000000);
        cnt += 1.0;
    }
    preprocess_time /= cnt;
    inference_time /= cnt;
    postprocess_time /= cnt;
    recognition_time /= cnt;
    preprocess_time /= 10000000000;
    inference_time /= 10000000000;
    postprocess_time /= 10000000000;
    recognition_time /= 10000000000;
    result_time_array.emplace_back(preprocess_time);
    result_time_array.emplace_back(inference_time);
    result_time_array.emplace_back(postprocess_time);
    result_time_array.emplace_back(recognition_time);
    // 清空向量组
    this->DETECTION_MODEL_PREPROCESS_TIME_ARRAY.clear();
    this->DETECTION_MODEL_INFERENCE_TIME_ARRAY.clear();
    this->DETECTION_MODEL_POSTPROCESS_TIME_ARRAY.clear();
    this->DETECTION_MODEL_RECOGNITION_TIME_ARRAY.clear();
    return result_time_array;
}

/**
 * @brief 这是YOLOv5检测视频的函数
 * @param video_path 视频地址
 * @param result_video_path 检测结果视频地址
 * @param interval 抽帧间隔频率，默认为-1,代表逐帧检测，1代表隔秒检测
 */
void YOLOv5::detect(std::string video_path, std::string result_video_path, float interval) {
    // 打开输入视频文件
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        spdlog::debug("Error: Could not open input video file.");
        return;
    }

    // 获取原始视频相关系数
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int bin;
    if(int(interval) == -1){
        bin = 1;                        // 逐帧检测
    } else{
        bin = int(fps*interval);        // 间隔帧数
    }

    // 初始化检测结果视频
    cv::VideoWriter writer(result_video_path,
                           cv::VideoWriter::fourcc('M','P','4','V'), fps, cv::Size(width, height));

    // 遍历原始视频进行检测将检测结果写入检测结果视频中
    cv::Mat frame;
    int cnt = 0;
    while(!cap.read(frame)){
        if(cnt % bin == 0){
            std::vector<DetectionResult> detection_results = this->detect(frame);
            cv::Mat detection_image = frame.clone();
            draw_detection_results(detection_image,detection_results,this->colors);
            writer.write(detection_image);
        }
    }
}


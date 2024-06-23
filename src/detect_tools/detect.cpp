//
// Created by dpw on 24-6-2.
//

#include "string"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "spdlog/spdlog.h"

#include "cmdline.h"
#include "common_utils.h"
#include "detection_utils.h"
#include "yolov5.h"

const std::vector<std::string> IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"};
const std::vector<std::string> VIDEO_EXTS = {".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv",".dav"};

void detect(cmdline::parser parser)
{
    // 获取参数
    std::string model_path = parser.get<std::string>("model_path");
    std::string label_txt_path = parser.get<std::string>("label_txt_path");
    std::string source = parser.get<std::string>("source");
    std::string result_dir = parser.get<std::string>("result_dir");
    float confidence_threshold = parser.get<float>("confidence_threshold");
    float iou_threshold = parser.get<float>("iou_threshold");
    int gpu_id = parser.get<int>("gpu_id");
    float interval = parser.get<float>("interval");
    int export_time = parser.get<int>("export_time");
    bool show_debug_msg = parser.get<bool>("show_debug_msg");

    // 设置debug级别日志
    if(show_debug_msg){
        enable_debug_msg();
    }

    // 初始化模型
    void* detector = new YOLOv5(model_path,label_txt_path,
                                confidence_threshold,iou_threshold,gpu_id,export_time);

    // 获取图片路径
    std::vector<std::string> image_paths;
    std::vector<std::string> video_paths;
    if(is_file(source)){
        // 图像文件
        if(is_contain_ext(source,IMAGE_EXTS)){
            image_paths.emplace_back(source);
        } else if(is_contain_ext(source,VIDEO_EXTS)){
            video_paths.emplace_back(source);
        }
    } else{
        // 获取图像集
        spdlog::debug("开始获取图像文件路径");
        get_file_paths(source,image_paths,IMAGE_EXTS);
        spdlog::debug("图像文件个数为:｛｝",image_paths.size());
        // 获取视频集
        spdlog::debug("开始获取视频文件路径");
        get_file_paths(source,video_paths,VIDEO_EXTS);
        spdlog::debug("视频文件个数为:｛｝",video_paths.size());
    }

    // 遍历所有图像进行检测
    int image_size = image_paths.size();
    std::string result_image_dir = join_address(result_dir,"image");
    if(!is_file_exists(result_image_dir)){
        makedirs(result_image_dir);
    }
    if(image_size > 0){
        spdlog::debug("共有{}张图像需要检测",image_size);
        std::vector<cv::Scalar> colors = reinterpret_cast<YOLOv5*>(detector)->get_colors();
        for(int i = 0; i < image_size; i++) {
            // 读取图像,并生成检测结果图像路径
            spdlog::debug("开始检测第{}/{}张图像",i+1,image_size);
            cv::Mat image = cv::imread(image_paths[i]);
            std::vector<std::string> dir_filname = split_filename(image_paths[i]);
            std::vector<std::string> fname_ext = split_ext(dir_filname[1]);
            std::string result_image_path = join_address(result_image_dir,fname_ext[0]+"_result.jpg");
            // 检测图像
            std::vector<DetectionResult> detection_results = reinterpret_cast<YOLOv5*>(detector)->detect(image);
            // 绘制结果
            cv::Mat result_image = image.clone();
            draw_detection_results(result_image,detection_results,colors);
            // 保存检测结果图像
            cv::imwrite(result_image_path,result_image);
        }
    }

    // 遍历所有视频，检测视频结果
    int video_size = video_paths.size();
    std::string result_video_dir = join_address(result_dir,"video");
    if(!is_file_exists(result_video_dir)){
        makedirs(result_video_dir);
    }
    if(video_size > 0){
        spdlog::debug("共有{}段视频需要检测",video_size);
        for(int i = 0; i < video_size; i++) {
            // 读取图像,并生成检测结果图像路径
            spdlog::debug("开始检测第{}/{}段视频",i+1,video_size);
            std::vector<std::string> dir_filname = split_filename(video_paths[i]);
            std::vector<std::string> fname_ext = split_ext(dir_filname[1]);
            std::string result_video_path = join_address(result_video_dir,fname_ext[0]+"_result.mp4");
            // 检测视频
            reinterpret_cast<YOLOv5*>(detector)->detect(video_paths[i],result_video_path,interval);
        }
    }

    // 输出时间
    if(export_time){
        std::vector<double> result_time_array = reinterpret_cast<YOLOv5*>(detector)->get_model_speed();
        spdlog::debug("模型图像预处理时间为:{}ms",result_time_array[0]);
        spdlog::debug("模型前向推理时间为:{}ms",result_time_array[1]);
        spdlog::debug("模型后处理时间为:{}ms",result_time_array[2]);
        spdlog::debug("模型检测识别时间为:{}ms",result_time_array[3]);
    }
}

int main(int argc, char *argv[])
{
    // 初始化命令行解析器
    cmdline::parser a;
    a.add<std::string>("model_path", '\0', "model_path", false, "");
    a.add<std::string>("label_txt_path", '\0', "label_txt_path", false, "");
    a.add<std::string>("source", '\0', "source", false, "");
    a.add<std::string>("result_dir", '\0', "result_dir", false, "./result");
    a.add<float>("confidence_threshold", '\0', "confidence_threshold", false, 0.0);
    a.add<float>("iou_threshold", '\0', "iou_threshold", false, 0.0);
    a.add<int>("gpu_id", '\0', "is export time", false, 0);
    a.add<float>("interval", '\0', "interval", false, -1.0);
    a.add<int>("export_time", '\0', "is export time", false, 0);
    a.add<bool>("show_debug_msg", '\0', "show debug msg", false, false);
    a.parse_check(argc, argv);

    // 检测图片(集)
    detect(a);

    return 0;
}

cmake_minimum_required(VERSION 3.26)
project(YOLO_Deploy_CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(YOLO_Deploy_CXX main.cpp
        models/base_detection_model.h
        models/yolov5/yolov5.cpp
        models/yolov5/yolov5.h
        models/detection_common.h
        utils/common_utils.cpp
        utils/common_utils.h
        engines/base_engine.h
        engines/tensorrt/tensorrt_engine.cpp
        engines/tensorrt/tensorrt_engine.h)

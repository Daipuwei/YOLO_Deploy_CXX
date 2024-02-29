cmake_minimum_required(VERSION 3.26)
project(YOLO_Deploy_CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(YOLO_Deploy_CXX main.cpp
        model/base_detection_model.h detection_common.h)

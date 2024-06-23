cmake_minimum_required(VERSION 3.10)
project(YOLO_Deploy)

# 设置C++标准和调试模式
set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)

# 设置调试和发布模式的编译选项
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

#设置cmakelist输入参数
set(SDK_PATH ${SDK_PATH})                          # SDK路径
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)    # 源代码路径

# 加载第三方依赖库
include(${CMAKE_CURRENT_SOURCE_DIR}/cmakes/platform/${PLATFORM_TYPE}/depend.cmake)
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmakes/platform/x86_tensorrt/depend.cmake)

# 加载项目相关公共头文件和源文件
include(cmakes/detection/common.cmake)

# 设置二级子项目CMakeLists.txt
add_subdirectory(${SOURCE_DIR}/detect_tools Detection_Tools)
add_subdirectory(${SOURCE_DIR}/tensorrt_tools TensorRT_Tools)

# 安装模型相关文件
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/model_data DESTINATION ${SDK_PATH}/)

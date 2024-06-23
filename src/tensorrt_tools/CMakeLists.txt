cmake_minimum_required(VERSION 3.10)
project(TensorRT_Tools)

# 设置头文件
include_directories(${COMMON_INCLUDE_PATHS}
        ${SOURCE_DIR}/tensorrt_tools/include/
)

# 设置检测可执行文件相关
add_executable(onnx2tensorrt src/onnx2tensorrt.cpp src/yolov5_trt_calibrator.cpp ${COMMON_CPP_PATHS})
target_link_libraries(onnx2tensorrt ${THIRD_PARTY_LIBS})

# 安装文件到指定目录
install(TARGETS onnx2tensorrt DESTINATION ${SDK_PATH}/bin)
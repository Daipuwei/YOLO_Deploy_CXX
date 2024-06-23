# 初始化公共头文件和源文件
set(COMMON_INCLUDE_PATHS
        ${SOURCE_DIR}/engines/common
        ${SOURCE_DIR}/models/common/include
        ${SOURCE_DIR}/models/yolov5/include
        ${SOURCE_DIR}/utils/include
)
FILE(GLOB UTILS_CPP_PATHS ${SOURCE_DIR}/utils/src/*.cpp)
FILE(GLOB YOLOV5_MODEL_CPP_PATHS ${SOURCE_DIR}/models/yolov5/src/*.cpp)
FILE(GLOB MODEL_COMMON_CPP_PATHS ${SOURCE_DIR}/models/common/src/*.cpp)
list(APPEND COMMON_CPP_PATHS ${UTILS_CPP_PATHS})
list(APPEND COMMON_CPP_PATHS ${YOLOV5_MODEL_CPP_PATHS})
list(APPEND COMMON_CPP_PATHS ${MODEL_COMMON_CPP_PATHS})
if(PLATFORM_TYPE STREQUAL "x86_tensorrt")
    list(APPEND COMMON_INCLUDE_PATHS  ${SOURCE_DIR}/engines/tensorrt/include)
    FILE(GLOB ENGINE_CPP_PATHS ${SOURCE_DIR}/engines/tensorrt/src/*.cpp)
else ()
    list(APPEND COMMON_INCLUDE_PATHS  ${SOURCE_DIR}/engines/tensorrt/include)
    FILE(GLOB ENGINE_CPP_PATHS ${SOURCE_DIR}/engines/tensorrt/src/*.cpp)
endif ()
list(APPEND COMMON_CPP_PATHS ${ENGINE_CPP_PATHS})
foreach(cpp_file ${COMMON_CPP_PATHS})
    message(STATUS "Found common cpp file: ${cpp_file}")
endforeach()

# 加载头文件
message(${COMMON_INCLUDE_PATHS})
include_directories(${COMMON_INCLUDE_PATHS})
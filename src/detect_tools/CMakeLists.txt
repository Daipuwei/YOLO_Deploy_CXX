cmake_minimum_required(VERSION 3.10)
project(DetectionTools)

# 设置头文件
include_directories(${COMMON_INCLUDE_PATHS}
        ${SOURCE_DIR}/detect_api/include
)

# detect
add_executable(detect detect.cpp ${COMMON_CPP_PATHS})
target_link_libraries(detect ${THIRD_PARTY_LIBS})

# 安装源码和可执行文件
install(TARGETS detect DESTINATION ${SDK_PATH}/bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/detect.cpp DESTINATION ${SDK_PATH}/)
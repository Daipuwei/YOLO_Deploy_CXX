#!/bin/bash
# 编译平台
supports="x86_tensorrt x86_onnxruntime x86_openvino"

# 初始化命令行参数
if (( $# == 0 )); then
  echo ""
  echo "Please Select the Compile Platform and options: build type"
  echo ""
  echo "-platform : platform which you use to compile."
  echo "support platform: ${supports}"
  echo ""
  echo "-build : build type, Debug or Release"
  echo ""
  echo "-packaged : packaged runtime && binary update package"
  echo ""
  echo "for example: $0 -platform x86_tensorrt"
  echo "for example: $0 -platform x86_tensorrt -build Debug"
  exit
fi

# 解析
i=1
for args in "$@"; do
  index=$((i+1))
  case "${args}" in
    -platform)
      if (( ${index} <= $# )); then
        PLATFRORM_TYPE=${!index}
        echo "Compile platform ${platform}"
      else
        echo "Please Select the Compile Platform!"
        echo "support platform: ${supports}"
        echo "for example: $0 -platform x86_tensorrt"
        exit
      fi
    ;;

    -build)
      if (( ${index} <= $# )); then
        if [ "${!index}" != "Debug" ] && [ "${!index}" != "Release" ]; then
          echo "Build Type Only Support Debug or Release"
          echo "for example: $0 -build Debug"
          exit
        else
          echo "Build Type ${!index}"
          COMPILE_BUILD_TYPE=${!index}
        fi
      else
        exit
      fi
    ;;
  esac
  i=$((i + 1))
done

# 初始化相关路径
u=`whoami`
BASE=$(cd $(dirname ${0})>/dev/null;pwd)
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
SDK_PATH=${BASE}/Detection_Deploy_SDK_${COMPILE_BUILD_TYPE}_${PLATFRORM_TYPE}_${CURRENT_TIME}
SDK_ZIP_PATH=${BASE}/Detection_Deploy_SDK_${COMPILE_BUILD_TYPE}_${PLATFRORM_TYPE}_${CURRENT_TIME}.zip

# 创建SDK文件夹及其子目录
rm -rf ${SDK_PATH}
mkdir -p ${SDK_PATH}/lib/
mkdir -p ${SDK_PATH}/include/
mkdir -p ${SDK_PATH}/bin/
mkdir -p ${SDK_PATH}/result/
if [ "$COMPILE_BUILD_TYPE" == "Debug" ]; then
  mkdir -p ${SDK_PATH}/model_data/
fi
chmod 777 -R ${SDK_PATH}

# 初始化交叉编译链cmake
TOOLCHAIN_PATH=${BASE}/cmakes/platform/${PLATFRORM_TYPE}/toolchain.cmake

# 创建搭建目录

rm -rf ./build/build_${PLATFRORM_TYPE}
mkdir -p ./build/build_${PLATFRORM_TYPE}
chmod 777 -R ./build/build_${PLATFRORM_TYPE}
cd ./build/build_${PLATFRORM_TYPE}

cmake -DCMAKE_BUILD_TYPE=${COMPILE_BUILD_TYPE} \
      -DSDK_PATH=${SDK_PATH} \
      -DPLATFORM_TYPE=${PLATFRORM_TYPE} \
      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_PATH} ../..
#make
make -j$(nproc)
make install

# SDK打包
cd ../../
chmod 777 -R ${SDK_PATH}
7z a ${SDK_ZIP_PATH} ${SDK_PATH}
chmod 777 ${SDK_ZIP_PATH}

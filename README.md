# YOLO_Deploy_CXX
这是YOLO系列检测算法的C++部署项目源代码。模型方面目前仅支持yolov5算法，推理引擎方面目前支持TensorRT。

---

# 版本更新日志
- **[2024-06-23]**第一次提交完整代码，实现YOLOv5模型在TensorRT上的部署，把那个在cmake、模型与推理引擎方面实现解耦，*版本分支为`v0.1`*;

---

# TODO
## 模型
- [x] YOLOv5
- [ ] YOLOv6
- [ ] YOLOv7
- [ ] YOLOv8
- [ ] YOLOv9
- [ ] YOLOv10
- [ ] YOLOX
- [ ] ...

## 推理引擎
- [x] TensorRT
- [ ] ONNXRuntime
- [ ] OpenVINO
- [ ] NCNN
- [ ] TNN
- [ ] MNN
- [ ] RKNN
- [ ] ...

---

# 一、编译环境配置
直接拉取Docker镜像，命令如下：
```bash
docker pull daipuwei/x86_deeplearning_runtime:cuda11.6_ubuntu20.04
```
创建容器，命令如下：
```bash
docker run -itd --ipc=host --restart=always --privileged=true \
--ulimit memlock=-1 --ulimit stack=67108864 \
--shm-size=32g  -p "44873:22" --name "x86_deeplearning_deploy" --gpus all \
-e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
-e NVIDIA_VISIBLE_DEVICES=all -e LANG=zh_CN.UTF-8 \
-e TZ="Asia/Shanghai" -e GDK_SCALE \
-e GDK_DPI_SCALE -v /etc/localtime:/etc/localtime:ro \
-v /tmp/.X11-unix:/tmp/.X11-unix -v /sbin/dmidecode:/sbin/dmidecode \
-v /dev/mem:/dev/mem -v /etc/locale.conf:/etc/locale.conf  \
-v /home/dpw:/home/dpw  daipuwei/x86_deeplearning_runtime:cuda11.6_ubuntu20.04
```

---
# 二、编译代码
## 2.1 X86 TensorRT 
命令如下：
```bash
# Debug
./build.sh -platform x86_tensorrt -build Debug
# Release
./build.sh -platform x86_tensorrt -build Release
```

---
# 三、运行
## 3.1 ONNX转TensorRT
```bash
# yolov5n fp32
./bin/onnx2tensorrt --onnx_model_path ./model_data/yolov5n.onnx \
                    --mode fp32 \
                    --batch_size 1 \
                    --input_width 640 \
                    --input_height 640 \
                    --input_channel 3 \
                    --gpu_device_id 0

# yolov5n fp16
./bin/onnx2tensorrt --onnx_model_path ./model_data/yolov5n.onnx \
                    --mode fp16 \
                    --batch_size 1 \
                    --input_width 640 \
                    --input_height 640 \
                    --input_channel 3 \
                    --gpu_device_id 0

# yolov5n int8
./bin/onnx2tensorrt --onnx_model_path ./model_data/yolov5n.onnx \
                    --mode int8 \
                    --batch_size 1 \
                    --input_width 640 \
                    --input_height 640 \
                    --input_channel 3 \
                    --calibrator_image_dir /home/dpw/daipuwei/coco_calib \
                    --calibrator_table_path  ./model_data/yolov5n_coco217_calibration.cache \
                    --gpu_device_id 0

```
## 3.2 检测图像与视频(detect)
```bash
./bin/detect --model_path ./model_data/yolov5n.trt.fp16 \
             --label_txt_path ./model_data/coco_names.txt \
             --source /home/dpw/daipuwei/coco_calib \
             --result_dir ./result/coco_calib/yolov5n.trt.fp16 \
             --confidence_threshold 0.1 \
             --iou_threshold 0.5 \
             --gpu_id 0 \
             --export_time 1 \
             --show_debug_msg 1
```
## 3.3 检测图像与视频(detect_api)
```bash
./bin/detect_api --source /home/dpw/daipuwei/coco_calib \
             --result_dir ./result/coco_calib/yolov5n.trt.fp16 \
             --confidence_threshold 0.1 \
             --iou_threshold 0.5 \
             --gpu_id 0 \
             --show_debug_msg 1
```

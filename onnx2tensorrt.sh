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
                    --calibrator_table_path  ./model_data/yolov5n_coco2017_calibration.cache \
                    --gpu_device_id 0
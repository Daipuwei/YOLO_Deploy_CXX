# detect
./bin/detect --model_path ./model_data/yolov5n.trt.fp16 \
             --label_txt_path ./model_data/coco_names.txt \
             --source /home/dpw/daipuwei/ObjectDetection/dataset/coco_calib \
             --result_dir ./result/coco_calib/yolov5n.trt.fp16 \
             --confidence_threshold 0.1 \
             --iou_threshold 0.5 \
             --gpu_id 0 \
             --export_time 1 \
             --show_debug_msg 1

# model2header
python model2header.py --model_path ./model_data/yolov5n.trt.fp16 \
        --label_txt_path ./model_data/coco_names.txt \
        --header_path ./src/detect_api/include/yolov5_engine.h \
        --project_name detection_engine

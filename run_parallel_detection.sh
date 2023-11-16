#!/bin/bash

# Define YOLO model names
YOLO_OUTDOOR_MODEL="20231116m"
YOLO_CABIN_MODEL="20231107m_cabin"

# Updated list of folder names
declare -a folder_names=("20230829_114310" "20230901_101413" "20230904_135906" "20230906_101400" "20230912_145556" "20230830_220436" "20230831_210234" "20230902_231627" "20230905_193841" "20230906_190938")

# Function to run detect.py for a specific video file
run_detection() {
    local folder_name=$1
    local gpu=$2

    for video_file in /mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/*.mp4; do
        local filename=$(basename "$video_file")
        local model_name
        local exp

        # Determine the model name and experiment number based on the video file name
        if [[ "$filename" =~ ^[0-3]\.mp4$ ]]; then
            model_name="20231116m"
            exp="exp449"
        elif [[ "$filename" == "4.mp4" ]]; then
            model_name="20231107m_cabin"
            exp="exp438"
        else
            echo "Invalid filename: $filename"
            return 1
        fi

        local source_file="/mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/${filename}"
        local output_dir="/mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/${model_name}"

        CUDA_VISIBLE_DEVICES="$gpu" python detect.py \
            --source "$source_file" \
            --weights "runs/train/${exp}/weights/best.pt" \
            --save-txt \
            --save-conf \
            --iou-thres=0.4 \
            --name "$output_dir/${filename%.*}"
    done
}

# Assign each folder to a different GPU
gpu=0
for folder_name in "${folder_names[@]}"; do
    run_detection "$folder_name" "$gpu" &

    # Increment GPU index, reset if it exceeds the number of available GPUs (8 in this case)
    ((gpu++))
    if [ $gpu -ge 8 ]; then
        gpu=0
    fi
done

# Wait for all background processes to finish
wait

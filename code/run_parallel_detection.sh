#!/bin/bash

# Define YOLO model names
# YOLO_OUTDOOR_MODEL="20231214l6"
# YOLO_OUTDOOR_MODEL_EXP="exp454"
# YOLO_OUTDOOR_MODEL="2023116m"
# YOLO_OUTDOOR_MODEL_EXP="exp449"
YOLO_OUTDOOR_MODEL="20240123m"
YOLO_OUTDOOR_MODEL_EXP="exp470"

YOLO_CABIN_MODEL="20240124m_cabin"

# Updated list of folder names
# declare -a folder_names=("20230829_114310" "20230901_101413" "20230904_135906" "20230906_101400" "20230912_145556" "20230830_220436" "20230831_210234" "20230902_231627" "20230905_193841" "20230906_190938")
# declare -a folder_names=("20231011" "20231012" "20231013" "20231016" "20231017" "20231018" "20231019" "20231020" "20231101" "20231102")
# declare -a folder_names=("20231011")
# declare -a folder_names=("20231201" "20231202" "20231204" "20231206" "20231207" "20231210" "20231212" "20231213" "20231214" "20231215")
# declare -a folder_names=("20231216" "20231217" "20231220" "20231221" "20231225" "20231226" "20231227" "20231228" "20231229" "20231230" "20231231")
declare -a folder_names=("20231201" "20231202" "20231204" "20231206" "20231207" "20231210" "20231212" "20231213" "20231214" "20231215" "20231216" "20231217" "20231220" "20231221" "20231225" "20231226" "20231227" "20231228" "20231229" "20231230" "20231231")
# Function to run detect.py for a specific video file
run_detection() {
    local folder_name=$1
    local gpu=$2

    local folder_path="/mnt/data/sibo/GP45/202312/video/${folder_name}"
    echo "Processing folder: $folder_path"
    
    # for video_file in /mnt/data/sibo/GP45/overall_selected/${folder_name}/*.mp4; do
    # for video_file in /mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/*.mp4; do
    for video_file in "${folder_path}"/*.mp4; do
        if [ ! -f "$video_file" ]; then
            echo "File not found: $video_file"
            continue
        fi

        local filename=$(basename "$video_file")
        local model_name
        local exp

        # Determine the model name and experiment number based on the video file name
        if [[ "$filename" =~ ^[0-3]\.mp4$ ]]; then
            model_name=$YOLO_OUTDOOR_MODEL
            exp=$YOLO_OUTDOOR_MODEL_EXP
            # model_name="20231116m"
            # exp="exp449"
        elif [[ "$filename" == "4.mp4" ]]; then
            continue
            # model_name=YOLO_CABIN_MODEL
            # exp="exp438"
        else
            echo "Invalid filename: $filename"
            return 1
        fi

        local source_file="${folder_path}/${filename}"
        local output_dir="${folder_path}/${model_name}"
        # local source_file="/mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/${filename}"
        # local output_dir="/mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/${model_name}"

        echo "Processing file: $source_file with model $model_name"
        CUDA_VISIBLE_DEVICES="$gpu" python ../detect.py \
            --source "$source_file" \
            --weights "../runs/train/${exp}/weights/best.pt" \
            --img 640 \
            --iou-thres=0.4 \
            --save-txt \
            --save-conf \
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

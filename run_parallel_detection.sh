#!/bin/bash

# List of folder names
declare -a folder_names=("20230829_114310" "20230901_101413" "20230904_135906" "20230906_101400" "20230912_145556" "20230830_220436" "20230831_210234" "20230902_231627" "20230905_193841" "20230906_190938")

# Function to run detect.py for a specific video file
run_detection() {
    local folder_name=$1
    local exp=$2
    local gpu=$3

    for video_file in /mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/*.mp4; do
        filename=$(basename "$video_file")

        # Determine the experiment model to use based on the video file name
        if [[ "$filename" =~ ^[0-3]\.mp4$ ]]; then
            exp="exp435"
        elif [[ "$filename" == "4.mp4" ]]; then
            exp="exp438"
        fi

        CUDA_VISIBLE_DEVICES="$gpu" python detect.py \
            --source "/mnt/data/sibo/GP45/overall_selected/GP45_${folder_name}/${filename}" \
            --weights "runs/train/${exp}/weights/best.pt" \
            --save-txt \
            --save-conf \
            --iou-thres=0.4 \
            --name "GP45_${folder_name}" # Sets the output directory name to only the folder name
    done
}

# Assign each folder to a different GPU
gpu=0
for folder_name in "${folder_names[@]}"; do
    run_detection "$folder_name" "exp435" "$gpu" &

    # Increment GPU index, reset if it exceeds the number of available GPUs (8 in this case)
    ((gpu++))
    if [ $gpu -ge 8 ]; then
        gpu=0
    fi
done

# Wait for all background processes to finish
wait

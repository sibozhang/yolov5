#!/bin/bash
#SBATCH --partition=A100x8_twodays,A100x8,V100x8_sixdays,V100x8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/paralell_detection_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/paralell_detection_%j.err
#SBATCH --exclude=asimov-243

cd /mnt/scratch/sibo/yolov5
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate yolov5

srun bash run_parallel_detection.sh


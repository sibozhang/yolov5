#!/bin/bash
#SBATCH --partition=A100x8_twodays
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/job_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/job_%j.err

conda activate yolov5

srun python -m torch.distributed.run --nproc_per_node 8 train.py --data coco_chile_dino_15class.yaml --weights runs/train/exp260/weights/best.pt --batch-size 128 --freeze 10 --device 0,1,2,3,4,5,6,7 --epochs 100
